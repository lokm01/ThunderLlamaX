# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 (Stage A) — the blk.64 (EAGLE nextn) drafter in PyTorch.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Draft layer port. Every structural choice is engine truth (see README):

  engine/_draft_entries + fill_draft (mtp.py L435-507) + kernels
  dnorm2.cu / q4v.cu / k0_norm.cu / aattn_d.cu / k3m_hh.cu / dfgu.cu:

  xin  = eh_proj( RMSnorm(emb(tok)) * enorm || RMSnorm(hm) * hnorm )   # cat 10240
  xh   = RMSnorm(xin) * attn_norm
  qkv  = W_q @ xh  (24 heads x [q(256)|gate(256)]) ; W_k, W_v (4 kv heads x 256)
  qn   = per-head RMS(q)*q_norm ; kn = per-head RMS(k)*k_norm           # eps 1e-6
  rope = partial 64/256 dims, pairs (i, i+32), freqs 1e7**(-i/32), absolute pos
  attn = softmax(qn_r . kn_r^T / 16) V  (causal over the drafter's own xin history)
  ao   = attn * sigmoid(gate)
  hh   = xin + W_o @ ao ; hhx = RMSnorm(hh)*post_norm
  hd   = hh + W_down @ (silu(W_g @ hhx) * (W_u @ hhx))
  head = lm_head( RMSnorm(hd) * shared_head_norm )        # frozen trunk head

NORM LAW (validated 2026-09-30): the HF qwen3_5 checkpoint stores ALL RMSNorm
weights ZERO-CENTERED (Gemma-style): functional weight = stored + 1.0. The GGUF
and the engine use the FUNCTIONAL form directly (proven element-wise: GGUF
trunk/draft norms == HF stored + 1.0 exactly; d_eh RTN cos 0.9968). from_hf
applies +1; export_hf inverts it.

Serve chain semantics (mtp.py graphs / fill_draft):
  FILL  : for prompt q: xin_q from (tok_q, OWN hd_{q-1})   [hd_{-1}=0], kv_d append
  STEP0 : xin from (committed tok, TRUNK pre-final-norm h_seed at same pos),
          kv_d[pos] OVERWRITTEN with this xin's K/V
  STEPi : xin from (OWN argmax tok, OWN hd_{i-1}) at pos t+i, kv_d appended
  argmax at serve is over the 40960-row prompt slice (slice_w) -> chain feedback
  defaults to the sequence's own token set ("slice" mode).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 5120
NH = 24          # q heads
NKV = 4          # kv heads (6 q-heads per kv head)
HD = 256         # head dim
ROPE_DIM = 64    # partial rotary (0.25)
THETA = 1.0e7
VOCAB = 248320
EPS = 1e-6
INNER = 17408

HF_MAP = {  # HF mtp.* -> pack names (engine draft_pack)
    "mtp.fc.weight": "d_eh",
    "mtp.layers.0.self_attn.q_proj.weight": "d_q",
    "mtp.layers.0.self_attn.k_proj.weight": "d_k",
    "mtp.layers.0.self_attn.v_proj.weight": "d_v",
    "mtp.layers.0.self_attn.o_proj.weight": "d_o",
    "mtp.layers.0.mlp.gate_proj.weight": "d_fg",
    "mtp.layers.0.mlp.up_proj.weight": "d_fu",
    "mtp.layers.0.mlp.down_proj.weight": "d_fd",
    "mtp.layers.0.input_layernorm.weight": "d_nw1",
    "mtp.layers.0.post_attention_layernorm.weight": "d_nw2",
    "mtp.norm.weight": "d_shnw",
    "mtp.pre_fc_norm_embedding.weight": "d_enw",
    "mtp.pre_fc_norm_hidden.weight": "d_hnw",
    "mtp.layers.0.self_attn.q_norm.weight": "d_qnw",
    "mtp.layers.0.self_attn.k_norm.weight": "d_knw",
}


def freqs_32() -> torch.Tensor:
    return torch.tensor([THETA ** (-i / 32.0) for i in range(32)], dtype=torch.float32)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # engine: r = rsqrt(mean(x^2)+eps); out = x*r*w   (fp32 stats)
    dt = x.dtype
    x32 = x.float()
    r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * r * w.float()).to(dt)


def apply_rope(x: torch.Tensor, pos: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
    """x [..., HD]; pos shape = x's dims before the head axis, minus leading
    batch dims (e.g. x [B,L,NH,HD] with pos [L], or x [B,NH,HD] with pos [B]).
    Pairs (i, i+32) i<32; ang = pos * fr[i]."""
    out = x.clone()
    xf = x.float()
    lead = (1,) * (x.dim() - 2 - pos.dim())
    ang = (pos.float().reshape(lead + pos.shape + (1, 1)) * fr[:32].float())
    cs, sn = ang.cos(), ang.sin()
    a = xf[..., :32]
    b = xf[..., 32:64]
    out[..., :32] = (a * cs - b * sn).to(x.dtype)
    out[..., 32:64] = (a * sn + b * cs).to(x.dtype)
    return out


class DraftBlock(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        p = lambda *s: nn.Parameter(torch.zeros(*s, dtype=dtype))
        self.eh_proj = p(DIM, 10240)
        self.wq = p(NH * 2 * HD, DIM)      # per head: [q(256) | gate(256)]
        self.wk = p(NKV * HD, DIM)
        self.wv = p(NKV * HD, DIM)
        self.wo = p(DIM, NH * HD)
        self.w_gate = p(INNER, DIM)
        self.w_up = p(INNER, DIM)
        self.w_down = p(DIM, INNER)
        self.enorm_w = p(DIM)
        self.hnorm_w = p(DIM)
        self.attn_norm_w = p(DIM)
        self.post_norm_w = p(DIM)
        self.shared_head_norm_w = p(DIM)
        self.q_norm_w = p(HD)
        self.k_norm_w = p(HD)
        self.register_buffer("fr", freqs_32())

    # ---- weight load ----
    NORM_KEYS = ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
                 "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
                 "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight")

    @classmethod
    def from_hf(cls, sd: dict, dtype: torch.dtype = torch.bfloat16):
        m = cls(dtype)
        with torch.no_grad():
            g = lambda k: sd[k].to(dtype) + (1.0 if k in cls.NORM_KEYS else 0.0)
            m.eh_proj.copy_(g("mtp.fc.weight"))
            m.wq.copy_(g("mtp.layers.0.self_attn.q_proj.weight"))
            m.wk.copy_(g("mtp.layers.0.self_attn.k_proj.weight"))
            m.wv.copy_(g("mtp.layers.0.self_attn.v_proj.weight"))
            m.wo.copy_(g("mtp.layers.0.self_attn.o_proj.weight"))
            m.w_gate.copy_(g("mtp.layers.0.mlp.gate_proj.weight"))
            m.w_up.copy_(g("mtp.layers.0.mlp.up_proj.weight"))
            m.w_down.copy_(g("mtp.layers.0.mlp.down_proj.weight"))
            m.enorm_w.copy_(g("mtp.pre_fc_norm_embedding.weight"))
            m.hnorm_w.copy_(g("mtp.pre_fc_norm_hidden.weight"))
            m.attn_norm_w.copy_(g("mtp.layers.0.input_layernorm.weight"))
            m.post_norm_w.copy_(g("mtp.layers.0.post_attention_layernorm.weight"))
            m.shared_head_norm_w.copy_(g("mtp.norm.weight"))
            m.q_norm_w.copy_(g("mtp.layers.0.self_attn.q_norm.weight"))
            m.k_norm_w.copy_(g("mtp.layers.0.self_attn.k_norm.weight"))
        return m

    @classmethod
    def from_pack(cls, w: dict, dtype: torch.dtype = torch.float32):
        """w: pack-dequant fp32 dict (q4pack_lib.load_pack)."""
        m = cls(dtype)
        with torch.no_grad():
            cp = lambda pm, src: pm.copy_(torch.from_numpy(w[src]).to(dtype))
            cp(m.eh_proj, "d_eh"); cp(m.wq, "d_q"); cp(m.wk, "d_k"); cp(m.wv, "d_v")
            cp(m.wo, "d_o"); cp(m.w_gate, "d_fg"); cp(m.w_up, "d_fu"); cp(m.w_down, "d_fd")
            cp(m.enorm_w, "d_enw"); cp(m.hnorm_w, "d_hnw"); cp(m.attn_norm_w, "d_nw1")
            cp(m.post_norm_w, "d_nw2"); cp(m.shared_head_norm_w, "d_shnw")
            cp(m.q_norm_w, "d_qnw"); cp(m.k_norm_w, "d_knw")
        return m

    def export_hf(self) -> dict:
        """HF-stored (zero-centered) form; pack via pack_trained.py which adds
        the +1 back for the engine's functional-form norm tensors."""
        return {
            "mtp.fc.weight": self.eh_proj,
            "mtp.layers.0.self_attn.q_proj.weight": self.wq,
            "mtp.layers.0.self_attn.k_proj.weight": self.wk,
            "mtp.layers.0.self_attn.v_proj.weight": self.wv,
            "mtp.layers.0.self_attn.o_proj.weight": self.wo,
            "mtp.layers.0.mlp.gate_proj.weight": self.w_gate,
            "mtp.layers.0.mlp.up_proj.weight": self.w_up,
            "mtp.layers.0.mlp.down_proj.weight": self.w_down,
            "mtp.layers.0.input_layernorm.weight": self.attn_norm_w - 1.0,
            "mtp.layers.0.post_attention_layernorm.weight": self.post_norm_w - 1.0,
            "mtp.norm.weight": self.shared_head_norm_w,
            "mtp.pre_fc_norm_embedding.weight": self.enorm_w,
            "mtp.pre_fc_norm_hidden.weight": self.hnorm_w,
            "mtp.layers.0.self_attn.q_norm.weight": self.q_norm_w - 1.0,
            "mtp.layers.0.self_attn.k_norm.weight": self.k_norm_w - 1.0,
        }

    # ---- pieces ----
    def xin_from(self, tok_emb: torch.Tensor, hm: torch.Tensor) -> torch.Tensor:
        """tok_emb [.., 5120] (already embedded), hm [.., 5120] -> xin [.., 5120]."""
        a = rms_norm(tok_emb, self.enorm_w)
        b = rms_norm(hm, self.hnorm_w)
        cat = torch.cat([a, b], dim=-1)
        return F.linear(cat, self.eh_proj)

    def qkv_of(self, xh: torch.Tensor):
        q = F.linear(xh, self.wq)              # [.., 12288]
        k = F.linear(xh, self.wk)              # [.., 1024]
        v = F.linear(xh, self.wv)
        return q, k, v

    def kvecs(self, k: torch.Tensor, pos: torch.Tensor):
        """k [..., 1024] -> normed+roped [.., NKV, HD]."""
        kk = k.float().reshape(*k.shape[:-1], NKV, HD)
        kk = rms_norm_dim(kk, self.k_norm_w.float())
        kk = apply_rope(kk, pos, self.fr)
        return kk

    def qvec_gate(self, q: torch.Tensor, pos: torch.Tensor):
        qf = q.float().reshape(*q.shape[:-1], NH, 2 * HD)
        qq, g = qf[..., :HD], qf[..., HD:]
        qq = rms_norm_dim(qq, self.q_norm_w.float())
        qq = apply_rope(qq, pos, self.fr)
        return qq, g  # q scaled by 1/16 inside attn

    def attn_scores(self, qq_s: torch.Tensor, kk: torch.Tensor):
        """qq_s [B,NH,HD] (already /16), kk [B,NKV,T,HD] -> [B,NH,T]"""
        NHq = qq_s.shape[1]
        rep = NHq // NKV
        kkx = kk.repeat_interleave(rep, dim=1)           # [B,NH,T,HD]
        return (qq_s.unsqueeze(2) * kkx).sum(-1)

    def block_tail(self, xin: torch.Tensor, ao: torch.Tensor) -> torch.Tensor:
        """ao [.., NH*HD] gated attn out -> hd [.., 5120]."""
        attn_out = F.linear(ao, self.wo)
        hh = xin + attn_out
        hhx = rms_norm(hh, self.post_norm_w)
        gact = F.silu(F.linear(hhx, self.w_gate)) * F.linear(hhx, self.w_up)
        return hh + F.linear(gact, self.w_down)

    # ---- full passes ----
    def fill(self, embs: torch.Tensor, h_prev: torch.Tensor):
        """Teacher-forced prefix pass (the fill_draft analog; hm = TRUE h_{q-1}).
        embs [B,L,5120], h_prev [B,L,5120] (h_prev[:,0] should be 0 = hd_{-1}).
        Returns hd [B,L,5120], K [B,NKV,L,HD], V [B,NKV,L,HD] (normed/roped)."""
        B, L, _ = embs.shape
        pos = torch.arange(L, device=embs.device)
        xin = self.xin_from(embs, h_prev)
        xh = rms_norm(xin, self.attn_norm_w)
        q, k, v = self.qkv_of(xh)                          # [B,L,*]
        qq, g = self.qvec_gate(q, pos)                     # [B,L,NH,HD],[B,L,NH,HD]
        kk = self.kvecs(k, pos)                            # [B,L,NKV,HD]
        vv = v.float().reshape(B, L, NKV, HD)
        qq = qq / math.sqrt(HD)                            # 1/16
        # causal attention, GQA broadcast
        kkT = kk.permute(0, 2, 1, 3)                       # [B,NKV,L,HD]
        qq2 = qq.permute(0, 2, 1, 3).reshape(B, NH, L, HD)
        kkT6 = kkT.repeat_interleave(NH // NKV, dim=1)
        sc = torch.matmul(qq2, kkT6.transpose(-1, -2))     # [B,NH,L,L]
        causal = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
        sc = sc.masked_fill(~causal, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p, vv.permute(0, 2, 1, 3).repeat_interleave(NH // NKV, dim=1))
        o = o.permute(0, 2, 1, 3).reshape(B, L, NH * HD)
        gate = torch.sigmoid(g.reshape(B, L, NH * HD))
        ao = o * gate
        hd = self.block_tail(xin, ao)
        return hd, kkT, vv.permute(0, 2, 1, 3)

    def step(self, tok_emb: torch.Tensor, hm: torch.Tensor, pos: torch.Tensor,
             K: torch.Tensor, V: torch.Tensor):
        """One chain step. tok_emb/hm [B,5120]; pos [B]; K,V [B,NKV,T,HD] the
        cache INCLUDING this step's own K/V already scattered at its position
        (the caller scatters via the returned k/v — see TTTTrainer.chain).
        Returns (hd [B,5120], logits_input [B,5120], k_new [B,NKV,HD], v_new,
        q_raw for reuse)."""
        xin = self.xin_from(tok_emb, hm)                   # [B,5120]
        xh = rms_norm(xin, self.attn_norm_w)
        q, k, v = self.qkv_of(xh)
        qq, g = self.qvec_gate(q, pos)                     # [B,NH,HD]
        kk = self.kvecs(k, pos)                            # [B,NKV,HD]
        vv = v.float().reshape(B, NKV, HD)
        qq = qq / math.sqrt(HD)
        sc = self.attn_scores(qq, K)                       # [B,NH,T]
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p.unsqueeze(2), V.repeat_interleave(NH // NKV, dim=1))  # [B,NH,1,HD]
        o = o.squeeze(2).reshape(B, NH * HD)
        ao = o * torch.sigmoid(g.reshape(B, NH * HD))
        hd = self.block_tail(xin, ao)
        return hd, kk, vv, xh


def rms_norm_dim(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Per-head RMS over the last dim, multiply by w (fp32 stats)."""
    x32 = x.float()
    r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return x32 * r * w


class FrozenHeadEmb(nn.Module):
    """Frozen trunk lm_head (bf16) + embedding."""

    def __init__(self, emb: torch.Tensor, head: torch.Tensor):
        super().__init__()
        self.register_buffer("emb", emb.to(torch.bfloat16), persistent=False)    # [VOCAB, 5120]
        self.register_buffer("head", head.to(torch.bfloat16), persistent=False)  # [VOCAB, 5120]

    def embed(self, toks: torch.Tensor) -> torch.Tensor:
        return self.emb[toks]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # x [.., 5120] (already shared-head-normed) -> fp32 logits
        return F.linear(x.to(torch.bfloat16), self.head).float()
