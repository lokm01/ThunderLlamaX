# TLX DRAFTER Phase 0 (S1) — chain_sim.py: the ENGINE-CALIBRATED draft-chain simulator.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Exact replay of the serve draft chain (blk.64 EAGLE nextn) against rig-dumped
traces. Fidelity targets (kernel truth, all read from the rig sources):

  dnorm2.cu   cat = [ f16(e*r*enw) || f16(hm*r2*hnw) ]  (rms fp32 stats, eps 1e-6)
  q4v.cu      acc = sum f16(x)*f16(w) fp32 ; plain outs f16(acc) ; ADDHH out =
              f32(f16(acc)) + hh   (ehproj adds zed5k zeros; ddown adds hh)
  k0_norm.cu  xh = f16(x*r*nw)
  spk_pre1qh_100k.cu (KPRE, ROWS=1):
              qn = f32(f16(q*r))*qnw ; rope fp32 ; qw16 = f16(qe*0.0625)
              kn = f32(f16(k*r))*knw ; ko = rope(kn) ; kf = f32(f16(ko))
              int8 append per (kvh,pos,32-group): sk=max|.|/127 (0->1e-8),
              skh=f16(sk), q8=clip(rint(v/f32(skh)),-127,127)+128
  spk_g4*+spk_c1g (K1S/K2S): scores over dequantized-to-f16 K (we accumulate
              fp32 — tighter than the engine's half2 chains, Tier-3 delta),
              fp32 online softmax, PV fp32, ao = f16(out*sigmoid(gate))
  k3m_hh      hh = xin + f32(attn_out_f16) ; hhx = f16(rms(hh)*nw2)
  dfgu.cu     silu in HALF domain (Tier-3 emulated), gact = f16(silu(ag)*au)
  shead.cu    Q5_K row dequant (d*sc*q5 - dm*mn), f16(x)*f16(w) fp32 acc,
              slogits = f16(acc)
  samx.cu     argmax lowest-index, out = stab[idx]
  accept5e.cu m = longest prefix(proposals == committed); targets on a greedy
              trace ARE the committed stream (probe-verifies; the bonus row is
              the true greedy token).

G0: sim(current pack, r8_prose trace) must reproduce E[m]|k4 = 0.633 +/- 0.05,
m-dist ~ [70,30,14,6,0] (per 120 = 2 deterministic reps; sim n=60 unique),
E[m]|k2 = 0.583. No projection is believed until G0 passes (the W5 META-LAW).

Inputs (rig dumps, see engine0/trace_dump_lib.py):
  <trace>/emb_raw.npy [V,2200] u8   <trace>/head_raw.npy [V,3520] u8
  <trace>/grid512.npy [512] f32     <trace>/stab.npy [40960] i32
  <trace>/final_norm_w.npy [5120] f32 (class-B targets)
  class A: prompt_ids/delta_ids/cycles.json/cycles_h_seed/kvd_decode_start/
           scd_decode_start
  class B: ids.npy/span_pos.npy/hiddens.npy[/kvd_end/scd_end ...]

Run anywhere (numpy-only by design; BLAS does the heavy lifting).
"""
import os, sys, json, math, argparse
import numpy as np

F16 = np.float16
F32 = np.float32
DIM, NH, NKV, HD, EPS = 5120, 24, 4, 256, 1e-6
INNER = 17408
THETA = 1.0e7
FREQS = (1.0 / (THETA ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(F32)
LOG2E_H = np.float32(-1.4423828125)   # dfgu.cu hsilu constant


def h16(x):
    return x.astype(F16).astype(F32)


def rms(x, w):
    r = 1.0 / np.sqrt(np.mean(x.astype(F32) ** 2) + EPS)
    return x.astype(F32) * r * w.astype(F32)


def rms_head(x, w):
    xf = x.astype(F32)
    r = 1.0 / np.sqrt((xf ** 2).mean(-1, keepdims=True) + EPS)
    return h16(xf * r) * w.astype(F32)     # KPRE: f16(x*r) THEN *nw


def rope(x, pos):
    """x [..., HD]; pairs (i, i+32), i<32; ang = pos*freqs[i] (fp32)."""
    out = x.astype(F32).copy()
    ang = F32(pos) * FREQS
    cs, sn = np.cos(ang), np.sin(ang)
    a = out[..., :32].copy()
    b = out[..., 32:64].copy()
    out[..., :32] = a * cs - b * sn
    out[..., 32:64] = a * sn + b * cs
    return out


# ---------------- quant-plane dequant (engine truth) ----------------

def dequant_iq3s_rows(raw, grid512, toks):
    """h_embed.cu: row 2200B = 20 blocks x 110B; element e = i*256+tid.
    d = f16 at blk[0:2]; g = tid>>2, j4 = tid&3;
    q = blk[2+g] + (bit g of blk[66+(g>>3)] << 8);
    sc = 1 + 2*((blk[106+(s8>>1)] >> ((s8&1)<<2)) & 0xF), s8 = tid>>5;
    sgn = bit (tid&7) of blk[74+(tid>>3)];
    x = d*sc*grid512[(q<<2)+j4]*sgn."""
    toks = np.asarray(toks, dtype=np.int64)
    out = np.empty((toks.size, DIM), F32)
    g512 = grid512.astype(F32)
    tid = np.arange(256)
    g = tid >> 2
    j4 = tid & 3
    s8 = tid >> 5
    blk = np.arange(20)
    e = (blk[:, None] * 256 + tid[None, :]).reshape(-1)     # 5120 element order
    for r, tok in enumerate(toks):
        row = raw[tok].reshape(20, 110)
        d = row[:, :2].copy().view(F16).astype(F32).reshape(20)[:, None]  # [20,1]
        q = row[:, 2 + g].astype(np.int64) + (((row[:, 66 + (g >> 3)].astype(np.uint8) >> (g & 7)) & 1).astype(np.int64) << 8)
        sc = 1.0 + 2.0 * ((row[:, 106 + (s8 >> 1)].astype(np.uint8) >> ((s8 & 1) << 2)) & 0xF).astype(F32)
        sgn = np.where((row[:, 74 + (tid >> 3)].astype(np.uint8) >> (tid & 7)) & 1, -1.0, 1.0)
        vals = d * sc * g512[(q << 2) + j4] * sgn           # [20,256]
        out[r, e] = vals.reshape(-1)
    return out


def dequant_q5k_rows(raw, rows):
    """shead.cu truth. 5120 elems = 20 blocks x 256; block = 176B:
    [d f16 @0][dm f16 @2][6-bit sc/mn pack @4..15][qh 32B @16][qs 128B @48].
    b = blk[4:16] (b col c == blk[4+c]):
      s < 4 : sc_s = blk[4+s]&63 = b[s]&63      mn_s = blk[8+s]&63 = b[4+s]&63
      s >= 4: sc_s = (blk[8+s]&0xF)|((blk[s]>>6)<<4)
              mn_s = (blk[8+s]>>4)|((blk[s+4]>>6)<<4)
              (blk[8+s]=b[8:12], blk[s]=b[0:4], blk[s+4]=b[4:8] for s in 4..7)
    element k (block-local): l=k>>3, j=k&7, s=l>>2
      qs nibble: byte qs[(l>>3)*32+(l&3)*8+j], low if (l>>2) even else high
      qh bit: byte qh[(l&3)*8+j] bit s
      w = d*sc*(q + 16*qhbit) - dm*mn."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((rows.size, DIM), F32)
    tid = np.arange(256)
    l = tid >> 3
    j = (tid & 7).astype(np.int64)
    s = (l >> 2).astype(np.int64)
    nib_hi = ((l >> 2) & 1).astype(bool)
    qs_off = ((l >> 3) * 32 + (l & 3) * 8 + j).astype(np.int64)
    qh_off = ((l & 3) * 8 + j).astype(np.int64)
    for r, rowi in enumerate(rows):
        row = raw[rowi].reshape(20, 176)
        d = row[:, :2].copy().view(F16).astype(F32).reshape(20)
        dm = row[:, 2:4].copy().view(F16).astype(F32).reshape(20)
        b = row[:, 4:16].astype(np.uint8)                     # [20,12]
        sc = np.empty((20, 8), np.int64)
        mn = np.empty((20, 8), np.int64)
        sc[:, :4] = b[:, 0:4] & 63
        mn[:, :4] = b[:, 4:8] & 63
        sc[:, 4:] = (b[:, 8:12] & 0xF).astype(np.int64) | ((b[:, 0:4] >> 6).astype(np.int64) << 4)
        mn[:, 4:] = (b[:, 8:12] >> 4).astype(np.int64) | ((b[:, 4:8] >> 6).astype(np.int64) << 4)
        qs = row[:, 48:176].astype(np.uint8)                  # [20,128]
        qh = row[:, 16:48].astype(np.uint8)                   # [20,32]
        qs_v = np.where(nib_hi[None, :], qs[:, qs_off] >> 4, qs[:, qs_off] & 0xF).astype(np.int64)
        qh_v = ((qh[:, qh_off] >> s[None, :]) & 1).astype(np.int64)
        qv = qs_v + (qh_v << 4)
        sc_e = sc[:, s].astype(F32)                           # [20,256]
        mn_e = mn[:, s].astype(F32)
        w = d[:, None] * sc_e * qv - dm[:, None] * mn_e
        out[r] = w.reshape(-1)
    return out


# ---------------- Q4 draft-pack (q4pack_lib truth, inlined) ----------------

Q4_ROWS = {
    "d_eh": (5120, 10240), "d_q": (12288, 5120), "d_k": (1024, 5120),
    "d_v": (1024, 5120), "d_o": (5120, 6144), "d_fg": (17408, 5120),
    "d_fu": (17408, 5120), "d_fd": (5120, 17408),
}
NORM_SHAPES = {"d_enw": 5120, "d_hnw": 5120, "d_shnw": 5120, "d_nw1": 5120,
               "d_nw2": 5120, "d_qnw": 256, "d_knw": 256}


def dequant_q4(arr, nout, nin):
    """dkv.cu/q4v.cu truth: element e of a 32-block = byte (e&15) of the
    sub-block's 16 qs bytes, nibble (e>>4) (low first); w = f16d*(q-8)."""
    assert arr.shape == (nout, (nin // 256) * 144), (arr.shape, nout, nin)
    qs = arr[:, : nin // 2]
    d = arr[:, nin // 2:].copy().view("<f2")
    qs = qs.reshape(nout, nin // 32, 16)
    lo = (qs & 0xF).astype(np.int32)
    hi = (qs >> 4).astype(np.int32)
    q = np.concatenate([lo, hi], axis=2)
    w = d.astype(F32)[:, :, None] * (q - 8).astype(F32)
    return w.reshape(nout, nin)


def load_pack(pack_dir, dtype=F32):
    out = {}
    for nm, (nout, nin) in Q4_ROWS.items():
        p = f"{pack_dir}/{nm}.npy"
        if os.path.exists(p):
            out[nm] = dequant_q4(np.load(p), nout, nin).astype(dtype)
    for nm in NORM_SHAPES:
        p = f"{pack_dir}/{nm}.npy"
        if os.path.exists(p):
            out[nm] = np.load(p).astype(F32)
    return out


HF_MAP = {  # bf16 HF originals (weights_v2 = corrected offsets) -> pack names
    "mtp_fc_weight": "d_eh",
    "mtp_layers_0_self_attn_q_proj_weight": "d_q",
    "mtp_layers_0_self_attn_k_proj_weight": "d_k",
    "mtp_layers_0_self_attn_v_proj_weight": "d_v",
    "mtp_layers_0_self_attn_o_proj_weight": "d_o",
    "mtp_layers_0_mlp_gate_proj_weight": "d_fg",
    "mtp_layers_0_mlp_up_proj_weight": "d_fu",
    "mtp_layers_0_mlp_down_proj_weight": "d_fd",
}
HF_NORM_KEYS = ("mtp_norm_weight", "mtp_pre_fc_norm_embedding_weight",
                "mtp_pre_fc_norm_hidden_weight", "mtp_layers_0_input_layernorm_weight",
                "mtp_layers_0_post_attention_layernorm_weight",
                "mtp_layers_0_self_attn_q_norm_weight", "mtp_layers_0_self_attn_k_norm_weight")


def load_bf16(weights_dir, dtype=F32):
    out = {}
    for hf, nm in HF_MAP.items():
        out[nm] = np.load(f"{weights_dir}/{hf}.npy").astype(dtype)
    for hf in HF_NORM_KEYS:
        nm = {"mtp_norm_weight": "d_shnw", "mtp_pre_fc_norm_embedding_weight": "d_enw",
              "mtp_pre_fc_norm_hidden_weight": "d_hnw", "mtp_layers_0_input_layernorm_weight": "d_nw1",
              "mtp_layers_0_post_attention_layernorm_weight": "d_nw2",
              "mtp_layers_0_self_attn_q_norm_weight": "d_qnw",
              "mtp_layers_0_self_attn_k_norm_weight": "d_knw"}[hf]
        out[nm] = np.load(f"{weights_dir}/{hf}.npy").astype(F32) + 1.0   # NORM LAW
    return out


def quantize_q8_rtn_sim(w):
    """q4pack_lib.quantize_q8_rtn + dequant (the Q8_0 ladder rung, in-sim)."""
    nout, nin = w.shape
    blk = w.reshape(nout, nin // 32, 32).astype(np.float64)
    amax = np.abs(blk).max(axis=2, keepdims=True)
    d = np.maximum(amax / 127.0, 1e-12)
    q = np.clip(np.rint(blk / d), -127, 127)
    return (q.astype(F32) * d.astype(F32)).reshape(nout, nin).astype(w.dtype)


def quantize_q4_rtn_sim(w):
    """RTN Q4_0 of an fp32 weight (the 'GPTQ-style' rung baseline when needed)."""
    nout, nin = w.shape
    blk = w.reshape(nout, nin // 32, 32).astype(np.float64)
    amax = np.abs(blk).max(axis=2, keepdims=True)
    d = np.maximum(amax / 8.0, 1e-12)
    q = np.clip(np.rint(blk / d + 8.0), 0, 15)
    return (q.astype(F32) - 8.0) * d.astype(F32).reshape(nout, nin // 32, 1).astype(F32) \
        if False else ((q.astype(F32) - 8.0) * np.repeat(d.astype(F32), 32, axis=2)).reshape(nout, nin)


class Weights:
    """Draft-block weights, engine-effective (fp16-rounded) + fp16 GEMV inputs."""

    def __init__(self, w, gemv_h16=True):
        self.w = w                    # fp32 dict (dequantized)
        self.gemv_h16 = gemv_h16

    @classmethod
    def from_spec(cls, spec, gemv_h16=True):
        """spec: 'pack:<dir>' | 'bf16:<dir>' | 'q8:pack:<dir>' (Q8 ladder rung)."""
        if spec.startswith("q8:"):
            base = load_pack(spec[3:])
            for nm in Q4_ROWS:
                if nm in base:
                    base[nm] = quantize_q8_rtn_sim(base[nm])
            return cls(base, gemv_h16)
        if spec.startswith("bf16:"):
            return cls(load_bf16(spec[5:]), gemv_h16)
        if spec.startswith("pack:"):
            return cls(load_pack(spec[5:]), gemv_h16)
        raise ValueError(spec)

    def gemv(self, nm, x_f16vals):
        """q4v GEMV: acc = sum f16(x)*f16(w) fp32; returns f32(f16(acc))."""
        W = self.w[nm]                              # [nout, nin] fp32 (already dequant)
        if self.gemv_h16:
            W = h16(W) if W.dtype == F32 else W
        acc = W.astype(F32) @ x_f16vals.astype(F32)
        return h16(acc)


# ---------------- the int8 draft KV (dumped state + in-place writes) ----------------

class KV8:
    """kv_d/sc_d emulation. The engine's ATTENTION OPERAND is the int8 row
    dequantized to f16 ((q-128)*f32(sc) rounded once). We materialize K/V as
    [4, cap, 256] f16 ONCE from the dump; append() reproduces the
    quantize-on-append roundtrip (spk_preqh.cu) then writes the dequantized
    row. Class-B hypothetical anchors use txn_begin/rollback."""

    def __init__(self, K, V):
        self.K = K                                  # f16 [4, cap, 256]
        self.V = V                                  # f16 [4, cap, 256]
        self._txn = None

    @classmethod
    def from_dump(cls, kvd_path, scd_path, upto=None, pad=0):
        q = np.load(kvd_path, mmap_mode="r")
        sc = np.load(scd_path, mmap_mode="r")
        if upto is not None:
            q, sc = q[:, :, :upto], sc[:, :, :upto]
        # sc layout [2,4,CAP,8]: 8 groups of 32 per row -> broadcast over d
        # (P0 bug fixed: NEVER use `w is q[0]` -- memmap re-slicing makes new
        # objects; K got dequantized with V's scales = the cos-0.78 class)
        def dq(half):
            q8 = (q[half].astype(np.int16) - 128).astype(F32)     # [4,CAP,256]
            s = np.repeat(sc[half].astype(F32), 32, axis=-1)      # [4,CAP,256]
            return (q8 * s).astype(F16)
        K = dq(0); V = dq(1)
        if pad > 0:
            # TLX P2: dumps trimmed to decode-start rows need append headroom
            # for the replay's own chain rows (the r8 dump carried full-CTXK).
            # Zero scratch: append() overwrites a row before any causal read.
            K = np.concatenate([K, np.zeros((4, pad, 256), F16)], axis=1)
            V = np.concatenate([V, np.zeros((4, pad, 256), F16)], axis=1)
        return cls(K, V)

    @classmethod
    def zeros(cls, cap):
        return cls(np.zeros((4, cap, 256), F16), np.zeros((4, cap, 256), F16))

    @staticmethod
    def _q8(g):
        """g [4,256] fp32 -> dequantized f16 row (the int8 roundtrip)."""
        gg = g.reshape(4, 8, 32)
        mk = np.abs(gg).max(axis=-1)
        sk = np.where(mk > 0, mk * (1.0 / 127.0), 1e-8)
        skh = sk.astype(F16)
        q8 = np.clip(np.rint(gg / skh.astype(F32)[..., None]), -127, 127) + 128
        return ((q8 - 128).astype(F32) * skh.astype(F32)[..., None]).astype(F16).reshape(4, 256)

    def append(self, pos, kf, vf):
        if self._txn is not None:
            self._txn.append((pos, self.K[:, pos, :].copy(), self.V[:, pos, :].copy()))
        self.K[:, pos, :] = self._q8(kf)
        self.V[:, pos, :] = self._q8(vf)

    def txn_begin(self):
        self._txn = []

    def txn_rollback(self):
        for pos, k, v in reversed(self._txn):
            self.K[:, pos, :] = k
            self.V[:, pos, :] = v
        self._txn = None


# ---------------- the draft chain step (engine-exact replay) ----------------

class ChainSim:
    """One draft-chain step = _draft_entries L443-474 kernel-for-kernel."""

    def __init__(self, W: Weights, kv: KV8, emb_raw=None, grid512=None,
                 slice_w16=None, stab=None, full_head=None, qk_engine=False):
        self.W = W
        self.kv = kv
        self.emb_raw = emb_raw
        self.grid512 = grid512.reshape(-1).astype(F32) if grid512 is not None else None
        self._emb_cache = {}
        # full-vocab head (TLX_DHEAD_FULL=1 canonical) vs slice head (diagnostic)
        self.full_head = full_head
        self.qk_engine = qk_engine
        self.slice_w16 = slice_w16
        self.stab = stab                  # row-order -> token id (slice mode)

    def emb(self, tok):
        t = int(tok)
        if t not in self._emb_cache:
            self._emb_cache[t] = dequant_iq3s_rows(self.emb_raw, self.grid512, [t])[0]
        return self._emb_cache[t]

    def step(self, tok, hm, pos, with_head=True):
        """Returns (hd f32[5120], head_in f16-vals f32[5120], proposal)."""
        W = self.W
        e = self.emb(tok) if self.emb_raw is not None else tok  # tok may be a pre-dequant row
        cat_e = rms(e, W.w["d_enw"]).astype(F16).astype(F32)
        cat_h = rms(hm.astype(F32), W.w["d_hnw"]).astype(F16).astype(F32)
        cat = np.concatenate([cat_e, cat_h]).astype(F16).astype(F32)
        xin = W.gemv("d_eh", cat)                       # ADDHH zeros -> f32(f16(acc))
        xh = rms(xin, W.w["d_nw1"]).astype(F16).astype(F32)
        qrow = W.gemv("d_q", xh)                        # f16 vals [12288]
        krow = W.gemv("d_k", xh)
        vrow = W.gemv("d_v", xh)
        qh = qrow.reshape(NH, 2, HD)
        qn = rms_head(qh[:, 0, :], W.w["d_qnw"][:HD].astype(F32))
        qe = rope(qn, pos)                              # [24,256] fp32
        qw = (qe * 0.0625).astype(F16).astype(F32)      # qw16 [24,256]
        kh = krow.reshape(NKV, HD)
        kn = rms_head(kh, W.w["d_knw"][:HD].astype(F32))
        ko = rope(kn, pos)
        kf = h16(ko)                                    # f16 round before quantize
        vf = vrow.reshape(NKV, HD).astype(F32)
        self.kv.append(pos, kf, vf)
        # ---- attention over kv rows [0..pos] (own row included; per-GQA) ----
        T = pos + 1
        ao = np.empty((NH, HD), F32)
        for g in range(NKV):
            Kh = self.kv.K[g, :T, :]                    # f16 [T,256] (engine operand)
            V32 = self.kv.V[g, :T, :].astype(F32)
            if self.qk_engine:
                # G4_QKROW: per 64-block, 4 f16 chains over block-elems r( mod 4)
                # in c-order, block sum (c0+c1)+(c2+c3) fp32, sc sequential.
                P6 = (qw[g * 6:(g + 1) * 6, None, :].astype(F32) * Kh.astype(F32)[None, :, :])
                P6 = P6.astype(F16).astype(F32).reshape(6, T, 4, 16, 4)
                ch = P6.copy()
                for s_i in range(1, 16):
                    ch[:, :, :, s_i, :] = (ch[:, :, :, s_i, :] + ch[:, :, :, s_i - 1, :]).astype(F16).astype(F32)
                cc = ch[:, :, :, 15, :]                       # [6,T,4blocks,4chains]
                S = ((cc[..., 0] + cc[..., 1]) + (cc[..., 2] + cc[..., 3])).sum(axis=2).T  # [T,6]
            else:
                S = Kh.astype(F32) @ qw[g * 6:(g + 1) * 6].T
            S -= S.max(axis=0, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=0, keepdims=True)
            O = P.T @ V32                               # [6,256]
            for hh in range(6):
                h = g * 6 + hh
                gate = 1.0 / (1.0 + np.exp(-qh[h, 1, :].astype(F32)))
                ao[h] = O[hh] * gate
        ao = ao.astype(F16).astype(F32)                 # K2S writes half
        attn_out = W.gemv("d_o", ao.reshape(-1))        # f16 vals [5120]
        hh = xin + attn_out                             # k3m_hh (fp32 add)
        hhx = rms(hh, W.w["d_nw2"]).astype(F16).astype(F32)
        ag = W.gemv("d_fg", hhx)
        au = W.gemv("d_fu", hhx)
        # dfgu hsilu: silu in the half domain (Tier-3 emulation)
        sg = 1.0 / (1.0 + np.exp2(ag.astype(F32) * LOG2E_H))
        sil = (ag * sg).astype(F16).astype(F32)
        gact = (sil * au).astype(F16).astype(F32)
        fd = W.gemv("d_fd", gact)                       # f16(acc)
        hd = hh + fd                                    # ADDHH residual
        head_in = rms(hd, W.w["d_shnw"]).astype(F16).astype(F32)
        prop = None
        if with_head and self.full_head is not None:
            prop = self.full_head.argmax(head_in)
        elif with_head and self.slice_w16 is not None:
            logits = (self.slice_w16 @ head_in).astype(F16).astype(F32)
            # samx: max value, LOWEST unique-row order index wins; map via stab
            best = int(np.argmax(logits))                # argmax = first max ✓ lowest idx
            prop = int(self.stab[best]) if self.stab is not None else best
        return hd, head_in, prop


class FullHead:
    """The engine's full-vocab draft head (TLX_DHEAD_FULL=1 canonical:
    sheadf/samxf over W[("head",0)]). Cache = f32(f16(dequant)) rows, built
    once per globals dir (5GB memmap; universal across draft-pack swaps)."""

    def __init__(self, cache_path, V=248320):
        self.H = np.load(cache_path, mmap_mode="r")
        assert self.H.shape == (V, DIM), self.H.shape

    @staticmethod
    def build(globals_dir, out_path=None, chunk=4096):
        out_path = out_path or f"{globals_dir}/head_eff_f32.npy"
        if os.path.exists(out_path):
            return out_path
        raw = np.load(f"{globals_dir}/head_raw.npy", mmap_mode="r")
        V = raw.shape[0]
        h = np.lib.format.open_memmap(out_path, mode="w+", dtype=F32, shape=(V, DIM))
        for c0 in range(0, V, chunk):
            rows = np.arange(c0, min(c0 + chunk, V))
            h[c0:c0 + len(rows)] = h16(dequant_q5k_rows(raw, rows))
            if (c0 // chunk) % 16 == 0:
                print(f"[headcache] {c0}/{V}", flush=True)
        h.flush()
        del h
        return out_path

    def argmax(self, head_in):
        """sheadf+samxf: acc f32, ONE f16 rounding, argmax lowest-index."""
        best_v, best_i = -np.inf, 0
        cs = 16384
        x = head_in.astype(F32)
        for c0 in range(0, self.H.shape[0], cs):
            lg = (np.asarray(self.H[c0:c0 + cs]) @ x).astype(F16).astype(F32)
            i = int(np.argmax(lg))
            if lg[i] > best_v:
                best_v, best_i = float(lg[i]), c0 + i
        return best_i

    def topk(self, head_in, k=8):
        V = self.H.shape[0]
        vals = np.empty(V, F32)
        cs = 16384
        x = head_in.astype(F32)
        for c0 in range(0, V, cs):
            vals[c0:c0 + cs] = (np.asarray(self.H[c0:c0 + cs]) @ x).astype(F16).astype(F32)
        idx = np.argpartition(-vals, k)[:k]
        idx = idx[np.argsort(-vals[idx])]
        return idx, vals[idx]

    def logits_rows(self, head_in):
        return (np.asarray(self.H) @ head_in.astype(F32)).astype(F16).astype(F32)


# ---------------- traces ----------------

class TraceA:
    """Class A: the r8_prose decode trace (engine cycles + dumped decode-start kv)."""

    def __init__(self, d):
        self.dir = d
        self.prompt = np.load(f"{d}/prompt_ids.npy")
        self.delta = np.load(f"{d}/delta_ids.npy")
        self.cycles = json.load(open(f"{d}/cycles.json"))
        self.hseeds = np.load(f"{d}/cycles_h_seed.npy")
        self.kvd = f"{d}/kvd_decode_start.npy"
        self.scd = f"{d}/scd_decode_start.npy"
        self.start_pos = self.cycles[0]["pos"] if self.cycles else 0

    def stream(self):
        """committed token stream: prompt (+cur0 implied by dump kv) + delta + emitted"""
        emt = []
        for r in self.cycles:
            emt += r["tokens"]
        return np.concatenate([self.prompt, self.delta, np.array(emt, np.int32)])


class TraceB:
    """Class B: teacher-forced corpus trace (ids + per-position hiddens + kvd)."""

    def __init__(self, d, kv_tag="end"):
        self.dir = d
        self.ids = np.load(f"{d}/ids.npy")
        hid = np.load(f"{d}/hiddens.npy", mmap_mode="r")
        sp = np.load(f"{d}/span_pos.npy")
        if sp.size == hid.shape[0]:
            # row-indexed layout (prompt100k spans): one abs pos per hidden row
            self.pos = sp.astype(np.int64)
        else:
            # block layout: sp[i] = first abs pos of a full contiguous block
            bl = hid.shape[0] // max(sp.size, 1)
            assert bl * sp.size == hid.shape[0], (sp.size, hid.shape)
            self.pos = np.concatenate([np.arange(int(p), int(p) + bl) for p in sp])
        self.hid = hid
        self.kvd = f"{d}/kvd_{kv_tag}.npy"
        self.scd = f"{d}/scd_{kv_tag}.npy"
        # rebuild pos->hidden-row map (block layout: hid row j corresponds to pos[j])
        self.row_of = {int(p): j for j, p in enumerate(self.pos)}


# ---------------- scoring ----------------

def m_stats(ms, K=4):
    ms = np.asarray(ms)
    n = max(len(ms), 1)
    dist = [int((ms == v).sum()) for v in range(K + 1)]
    a_cond = []
    for i in range(K):
        num = (ms >= i + 1).sum()
        den = (ms >= i).sum()
        a_cond.append(float(num) / max(den, 1))
    Em = {}
    for k in (2, 4):
        Em[k] = float(sum((ms >= i + 1).sum() for i in range(k))) / n
    return dict(n=int(len(ms)), m_dist=dist, a_cond=[round(a, 4) for a in a_cond],
                Em_k2=round(Em[2], 4), Em_k4=round(Em[4], 4) if K >= 4 else None,
                tok_per_cyc=round(float((ms + 1).mean()), 4) if len(ms) else None)


def run_class_a(trace_dir, weights, tdir_globals, kvd=None, verbose=True, max_cycles=None,
                chain_len=4, only_k4=True, cond="engine"):
    tr = TraceA(trace_dir)
    _rows_pre = tr.cycles if max_cycles is None else tr.cycles[:max_cycles]
    _qrows = np.load(tr.kvd if kvd is None else kvd, mmap_mode="r").shape[2]
    _pmax = max((int(r["pos"]) for r in _rows_pre), default=0) + 8
    kv = KV8.from_dump(tr.kvd if kvd is None else kvd, tr.scd if kvd is None else kvd.replace("kvd_", "scd_"),
                      upto=None, pad=max(0, _pmax - _qrows))
    g = load_globals(tdir_globals)
    fh = FullHead(FullHead.build(tdir_globals))
    sim = ChainSim(weights, kv, emb_raw=g["emb_raw"], grid512=g["grid512"],
                   full_head=fh)
    ms, eng_ms, prop_match = [], [], 0
    rows = tr.cycles if max_cycles is None else tr.cycles[:max_cycles]
    # THE KV-WRITE LAW: EVERY spec cycle's draft chain writes kv_d rows (K2
    # cycles write 2, K4-EAGLE 4; deep/lookup cycles write NONE -- draft-skip).
    # The replay walks ALL cycles so the kv state the scored cycles see is the
    # engine's; only K4 cycles are SCORED (the reference class).
    for r in rows:
        deep = bool(r.get("deep_at_entry")) or bool(r.get("t1_at_entry"))
        clen = 0 if deep else (chain_len if r.get("prose") else 2)
        pos, cur, hs = r["pos"], r["cur"], tr.hseeds[r["h_idx"]]
        props = []
        hm_prev = None
        for i in range(clen):
            if i == 0:
                tok = cur
            else:
                # cond=engine: chain on the ENGINE's recorded proposals (the
                # exact tokens the engine's kv rows carry -- removes the
                # compounding-through-kv divergence; measures OUR numerics).
                # cond=own: the serve-faithful self-chain.
                tok = (int(r["dring"][i - 1]) if cond == "engine" and i - 1 < len(r["dring"])
                       else props[i - 1])
            hm = hs if i == 0 else hm_prev
            hd, _hi, prop = sim.step(tok, hm, pos + i)
            hm_prev = hd
            props.append(prop)
        if only_k4 and not (r.get("prose") and not deep):
            continue
        m = 0
        for i in range(min(chain_len, len(r["tokens"]))):
            if props[i] != int(r["tokens"][i]):
                break
            m += 1
        ms.append(m)
        eng_ms.append(int(r["m"]))
        prop_match += sum(1 for i in range(min(chain_len, len(props)))
                          if props[i] == int(r["dring"][i]))
    if verbose:
        n = len(rows)
        print(f"[classA] cycles {n} | sim {m_stats(ms)}")
        print(f"[classA] engine {m_stats(eng_ms)}")
        print(f"[classA] sim-vs-engine proposal agreement: {prop_match}/{n*chain_len} "
              f"({100.0*prop_match/max(n*chain_len,1):.1f}%) | per-cycle m equal: "
              f"{sum(1 for a,b in zip(ms,eng_ms) if a==b)}/{n}")
    return dict(sim=m_stats(ms), engine=m_stats(eng_ms),
                prop_agree=round(prop_match / max(len(rows) * chain_len, 1), 4))


def load_globals(tdir, with_slice=True):
    out = {}
    out["emb_raw"] = np.load(f"{tdir}/emb_raw.npy", mmap_mode="r")
    out["grid512"] = np.load(f"{tdir}/grid512.npy")
    if os.path.exists(f"{tdir}/final_norm_w.npy"):
        out["final_norm_w"] = np.load(f"{tdir}/final_norm_w.npy")
    stab = np.load(f"{tdir}/stab.npy")
    out["stab"] = stab
    uniq, first = np.unique(stab, return_index=True)     # first = lowest slice index
    order = np.argsort(first)
    out["uniq"] = uniq[order]                            # unique ids, lowest-slice-idx order
    if with_slice:
        out["slice_w16"], _ = build_slice_head(tdir, out)
    return out


def build_slice_head(tdir, g, head_raw=None, subset=None):
    """slice_w16 [nuniq,5120] = f16-effective Q5_K rows of the UNIQUE slice ids."""
    head_raw = head_raw if head_raw is not None else np.load(f"{tdir}/head_raw.npy", mmap_mode="r")
    ids = g["uniq"] if subset is None else np.asarray(subset, np.int64)
    W = dequant_q5k_rows(head_raw, ids)
    return h16(W), ids


def head_target(head_raw, h_row, final_norm_w, chunk=16384):
    """Full-vocab argmax of the trunk head on one hidden row (engine head8:
    xh = f16(rms(h,onw)); acc = sum f16(x)*f16(w) fp32; argmax lowest-idx)."""
    xh = rms(h_row.astype(F32), final_norm_w).astype(F16).astype(F32)
    best_v, best_i = -np.inf, 0
    V = head_raw.shape[0]
    for c0 in range(0, V, chunk):
        rows = np.arange(c0, min(c0 + chunk, V))
        W = h16(dequant_q5k_rows(head_raw, rows))
        lg = (W @ xh).astype(F16).astype(F32)
        i = int(np.argmax(lg))
        if lg[i] > best_v:
            best_v, best_i = float(lg[i]), int(rows[i])
    return best_i


def head_topk(head_raw, h_row, final_norm_w, k=8, chunk=16384):
    xh = rms(h_row.astype(F32), final_norm_w).astype(F16).astype(F32)
    V = head_raw.shape[0]
    vals = np.empty(V, F32)
    for c0 in range(0, V, chunk):
        rows = np.arange(c0, min(c0 + chunk, V))
        W = h16(dequant_q5k_rows(head_raw, rows))
        vals[c0:c0 + len(rows)] = (W @ xh).astype(F16).astype(F32)
    idx = np.argpartition(-vals, k)[:k]
    idx = idx[np.argsort(-vals[idx])]
    return idx, vals[idx]


def run_class_b(trace_dir, weights, tdir_globals, stride=16, chain_len=4,
                kv_tag="end", mode="serve", max_anchors=None, verbose=True,
                fill_rebuild=False, fill_hm_true=False, target_cache=None):
    """mode: serve (own-chain) | tf (teacher-forced true tokens+hiddens).
    fill_rebuild: rebuild kv from the sim's own fill instead of the dump.
    fill_hm_true: fill with TRUE trunk hiddens as hm (the conditioning probe)."""
    tr = TraceB(trace_dir, kv_tag=kv_tag)
    g = load_globals(tdir_globals, with_slice=True)
    if fill_rebuild:
        kv = KV8.zeros(max(int(tr.pos[-1]) + 8, 256))
        sim0 = ChainSim(weights, kv, emb_raw=g["emb_raw"], grid512=g["grid512"])
        # fill over ALL ids (own-chain semantics; hm=true hiddens variant for the probe)
        hm = np.zeros(DIM, F32)
        rows_avail = dict(zip(tr.pos.tolist(), range(tr.hid.shape[0])))
        for q, tid in enumerate(tr.ids):
            e_tok = sim0.emb(tid)
            hm_use = tr.hid[rows_avail[q]].astype(F32) if (fill_hm_true and q in rows_avail) else hm
            hd, _, _ = sim0.step(e_tok, hm_use, q, with_head=False)
            hm = hd
    else:
        kv = KV8.from_dump(tr.kvd, tr.scd)
    fh = FullHead(FullHead.build(tdir_globals))
    sim = ChainSim(weights, kv, emb_raw=g["emb_raw"], grid512=g["grid512"],
                   full_head=fh)
    fnw = g["final_norm_w"]
    # anchor positions: need hid rows t..t+chain_len and ids t..t+chain_len
    pos_have = tr.row_of
    n_ids = len(tr.ids)
    anchors = [int(p) for p in tr.pos if p + chain_len < n_ids and
               all((p + i) in pos_have for i in range(chain_len + 1))][::stride]
    if max_anchors:
        anchors = anchors[:max_anchors]
    tgt = {}
    def target(p):
        if p not in tgt:
            tgt[p] = fh.argmax(rms(np.asarray(tr.hid[pos_have[p]]), fnw)
                               .astype(F16).astype(F32))
        return tgt[p]
    ms, ms_tf = [], []
    marg = []
    for t in anchors:
        kv.txn_begin()
        props = []
        hm_prev = None
        for i in range(chain_len):
            if mode == "tf":
                tok = int(tr.ids[t + i])
                hm = tr.hid[pos_have[t + i]].astype(F32)
            else:
                tok = int(tr.ids[t]) if i == 0 else props[i - 1]
                hm = tr.hid[pos_have[t]].astype(F32) if i == 0 else hm_prev
            hd, _hi, prop = sim.step(tok, hm, t + i)
            hm_prev = hd
            props.append(prop)
        kv.txn_rollback()
        # proposal_i (step at t+i) predicts the token at t+i+1; the greedy
        # truth for it = target(t+i) = argmax(head(norm(h(t+i)))) (validated
        # 39/39 vs the engine's committed stream on the class-A trace).
        m = 0
        for i in range(chain_len):
            if props[i] != target(t + i):
                break
            m += 1
        ms.append(m)
        marg.append([int(target(t + i)) for i in range(chain_len)])
    res = dict(trace=trace_dir, mode=mode, stride=stride, **m_stats(ms))
    if verbose:
        print(f"[classB {os.path.basename(trace_dir)}] {json.dumps(res)}")
    return res


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["g0", "score", "classb", "battery"])
    ap.add_argument("--trace", default=os.path.expanduser("~/drafter/phase0/traces/r8_prose"))
    ap.add_argument("--globals", default=os.path.expanduser("~/drafter/phase0/traces/r8_prose"))
    ap.add_argument("--pack", default="pack:" + os.path.expanduser("~/drafter/ref_pack"))
    ap.add_argument("--strides", default="16")
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--cond", default="engine", choices=["engine", "own"])
    ap.add_argument("--chain-len", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    W = Weights.from_spec(a.pack)
    if a.cmd in ("g0", "score"):
        r = run_class_a(a.trace, W, a.globals, max_cycles=a.max_cycles, chain_len=a.chain_len,
                        cond=a.cond)
        if a.cmd == "g0":
            ok = abs(r["sim"]["Em_k4"] - 0.633) <= 0.05 and abs(r["sim"]["Em_k2"] - 0.583) <= 0.05
            print(f"[G0] {'PASS' if ok else 'FAIL'} (sim E[m]k4 {r['sim']['Em_k4']} vs 0.633+/-0.05; "
                  f"k2 {r['sim']['Em_k2']} vs 0.583+/-0.05; engine {r['engine']['Em_k4']})")
    elif a.cmd == "classb":
        run_class_b(a.trace, W, a.globals, stride=int(a.strides), chain_len=a.chain_len)
    elif a.cmd == "battery":
        raise SystemExit("battery: drive via run_battery() (seelever-ranking script)")


if __name__ == "__main__":
    main()
