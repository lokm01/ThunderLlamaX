# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 (Stage A) — TTT trainer for the blk.64 drafter.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""EAGLE-3 recipe, chain-shaped to the engine's serve semantics:

  fill   (parallel, teacher-forced): xin_q = eh_proj(enorm(emb(x_q)) || hnorm(h_{q-1}))
           -> causal own-KV cache (K/V of xin_q) [the fill_draft analog, true-h substitution]
  anchor (per sampled position t): xin = eh_proj(enorm(emb(x_t)) || hnorm(h_t^true))
           — the serve step-0 convention: same-position token + trunk pre-norm hidden;
           its K/V REPLACE the fill slot t.
  steps  i=1..S-1: xin from (OWN argmax token, OWN hd_{i-1}) at pos t+i, K/V appended.
  loss   = sum_i CE(head(shared_norm(hd_i)), x_{t+i+1})   token-only (no feature
           regression — the EAGLE-3 ablation), full vocab, frozen trunk head.

Chain feedback argmax is SLICE-restricted by default (serve argmaxes over the
40960-row prompt slice; we approximate with the sequence's own distinct ids).

Data: memmap shards written by dump_features.py:
  <dir>/ids.npy     int32 [N, Lmax]   (-1 pad)
  <dir>/hids.npy    fp16  [N, Lmax, 5120]  teacher PRE-final-norm hiddens
  <dir>/lens.npy    int32 [N]
  <dir>/meta.json

Modes: --overfit-synthetic (Validation B machinery test, no data needed),
--overfit-real (10k-token slice), --calib (dump GEMV-input calibration for
pack_trained GPTQ), default = full training run.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DraftBlock, FrozenHeadEmb, rms_norm, DIM, NH, NKV, HD, VOCAB


# ---------------- data ----------------
class ShardDataset:
    def __init__(self, d):
        self.ids = np.load(f"{d}/ids.npy", mmap_mode="r")
        self.hids = np.load(f"{d}/hids.npy", mmap_mode="r")
        self.lens = np.load(f"{d}/lens.npy")
        self.meta = json.load(open(f"{d}/meta.json"))
        self.N = len(self.lens)

    def sample(self, idx, Lmax):
        n = int(self.lens[idx])
        ids = np.asarray(self.ids[idx, max(0, n - Lmax):n], dtype=np.int64)  # left-crop
        h = np.asarray(self.hids[idx, max(0, n - Lmax):n], dtype=np.float32)
        return ids, h


# ---------------- the TTT chain ----------------
def chain_forward(mod, head_emb, ids, h_true, t, S, feedback="slice", compute_head=True):
    """One TTT unroll. ids/h_true [B, L]; t [B] anchor positions.
    Returns (losses [S] tensor, acc [S] bool tensor, n_valid)."""
    B, L = ids.shape
    dev = ids.device
    embs = head_emb.embed(ids)                       # [B,L,5120] bf16
    h_prev = torch.zeros_like(h_true)
    h_prev[:, 1:] = h_true[:, :-1]                   # hm at q = h_{q-1}^true
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=embs.is_cuda), \
         torch.autocast("mps", dtype=torch.float16, enabled=(not embs.is_cuda and embs.device.type == "mps")):
        hd_fill, Kfill, Vfill = mod.fill(embs, h_prev)   # [B,L,5120],[B,4,L,256]
    Lcap = L + S
    Kc = torch.zeros(B, NKV, Lcap, HD, device=dev, dtype=torch.float32)
    Vc = torch.zeros(B, NKV, Lcap, HD, device=dev, dtype=torch.float32)
    Kc[:, :, :L] = Kfill.float()
    Vc[:, :, :L] = Vfill.float()
    ar = torch.arange(Lcap, device=dev)[None, None, :]
    pos_t = t.to(dev)
    hm = h_true[torch.arange(B, device=dev), pos_t]      # anchor hm = h_t^true
    tok = ids[torch.arange(B, device=dev), pos_t]
    one_t = (ar == pos_t[:, None, None]).float().unsqueeze(-1)  # [B,1,Lcap,1]
    losses, accs = [], []
    fb_tokens = ids
    slice_mask = None
    Vh = head_emb.head.shape[0]
    if feedback == "slice":
        slice_mask = torch.zeros(B, Vh, device=dev, dtype=torch.bool)
        slice_mask.scatter_(1, ids.clamp(max=Vh - 1), True)   # sequence's own ids
    for i in range(S):
        pos = pos_t + i
        xin = mod.xin_from(head_emb.embed(tok), hm)
        xh = rms_norm(xin, mod.attn_norm_w)
        q, k, v = mod.qkv_of(xh)
        qq, g = mod.qvec_gate(q, pos)
        kk = mod.kvecs(k, pos)                           # [B,NKV,HD]
        vv = v.float().reshape(B, NKV, HD)
        one = (ar == pos[:, None, None]).float().unsqueeze(-1)   # [B,1,Lcap,1]
        Kc = Kc * (1 - one) + kk.unsqueeze(2) * one
        Vc = Vc * (1 - one) + vv.unsqueeze(2) * one
        qq = qq / math.sqrt(HD)
        kkx = Kc.repeat_interleave(NH // NKV, dim=1)     # [B,NH,Lcap,HD]
        sc = torch.einsum("bhd,bhtd->bht", qq, kkx)
        causal = ar <= pos[:, None, None]                        # [B,1,Lcap]
        sc = sc.masked_fill(~causal, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        o = torch.einsum("bht,bhtd->bhd", p, Vc.repeat_interleave(NH // NKV, dim=1))
        ao = o.reshape(B, NH * HD) * torch.sigmoid(g.reshape(B, NH * HD))
        hd = mod.block_tail(xin, ao)
        hi = rms_norm(hd, mod.shared_head_norm_w)
        logits = head_emb.logits(hi)                     # fp32 [B, VOCAB]
        lbl = ids[torch.arange(B, device=dev), torch.clamp(pos + 1, max=L - 1)]
        valid = (pos + 1) < L
        lv = F.cross_entropy(logits, lbl, reduction="none")
        lv = lv * valid.float()
        losses.append(lv.sum())
        accs.append(((logits.argmax(-1) == lbl).float() * valid.float()).sum())
        # feedback token (no grad)
        with torch.no_grad():
            if feedback == "slice":
                lm = logits.masked_fill(~slice_mask, float("-inf"))
            else:
                lm = logits
            tok = lm.argmax(-1).detach()
        hm = hd                                          # own hidden carries grad
    return torch.stack(losses), torch.stack(accs), int(valid.sum().item()) if S else 0


def load_sd_ckpt(p):
    import torch as _t
    return {k: v.float() for k, v in _t.load(p, map_location="cpu", weights_only=False)["sd"].items()}


# ---------------- synthetic overfit (Validation B machinery) ----------------
def synth_corpus(B=6, L=256, seed=0, dev="cpu", voc=4096):
    """Deterministic 'teacher': h_t = tanh(emb[x_t]@R)*4 + 0.35*emb[x_{t+1}]@R2
    — leaks x_{t+1} so the task is learnable BY CONSTRUCTION (proves gradients
    flow through the TTT unroll + cache scatter + chain feedback)."""
    g = torch.Generator().manual_seed(seed)
    emb = (torch.randn(voc, DIM, generator=g) * 0.02).to(torch.bfloat16)
    head = (torch.randn(voc, DIM, generator=g) * 0.02).to(torch.bfloat16)
    # a small structured token process so chains have signal
    base = torch.randint(0, 512, (B, L), generator=g)
    ids = base.clone()
    for i in range(1, L):
        r = torch.rand(B, generator=g)
        ids[:, i] = torch.where(r < 0.55, ids[:, i - 1] + 1, ids[:, i])
    ids = ids % voc
    R = torch.randn(DIM, DIM, generator=g) * (1 / math.sqrt(DIM))
    R2 = torch.randn(DIM, DIM, generator=g) * (1 / math.sqrt(DIM))
    e = F.embedding(ids, emb.float())
    h_next = F.embedding((ids + 1) % voc, emb.float()) @ R2
    h = torch.tanh(e @ R) * 4.0 + 0.35 * h_next
    return ids.to(dev), h.to(torch.float32).to(dev), FrozenHeadEmb(emb.to(dev), head.to(dev))


def overfit_synthetic(args):
    dev = torch.device(args.device)
    torch.manual_seed(0)
    # synthetic init: normal-ish weights scaled like trained MTP
    mod = DraftBlock(torch.float32).to(dev)
    with torch.no_grad():
        for n, p in mod.named_parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.008)
            else:
                p.fill_(1.0)
    mod = mod.to(torch.float32)
    ids, h, head_emb = synth_corpus(dev=dev)
    head_emb = head_emb.to(dev)
    opt = torch.optim.AdamW([p for p in mod.parameters()], lr=3e-4)
    t_fix = torch.full((ids.shape[0],), ids.shape[1] - args.S - 2, dtype=torch.long, device=dev)
    print("[overfit-syn] S=%d B=%d L=%d" % (args.S, ids.shape[0], ids.shape[1]))
    for step in range(args.overfit_steps):
        t = torch.randint(16, ids.shape[1] - args.S - 1, (ids.shape[0],), device=dev) if step < args.overfit_steps - 5 else t_fix
        losses, accs, _ = chain_forward(mod, head_emb, ids, h, t, args.S, feedback="full")
        loss = losses.sum() / ids.shape[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:4d} loss {loss.item():.3f} acc/step {[round(float(a)/ids.shape[0],3) for a in accs]}", flush=True)
    losses, accs, _ = chain_forward(mod, head_emb, ids, h, t_fix, args.S, feedback="full")
    print("[overfit-syn] FINAL fixed-anchor loss %.4f acc %s" %
          (losses.sum().item() / ids.shape[0], [round(float(a) / ids.shape[0], 3) for a in accs]))
    ok = losses.sum().item() / ids.shape[0] < 8.0 and float(accs[0]) / ids.shape[0] > 0.9
    print("[overfit-syn] %s" % ("PASS (machinery: gradients flow through the TTT unroll + "
          "chain feedback + KV scatter; the strict near-zero criterion is overfit-REAL's)" if ok else "FAIL"))
    return ok


# ---------------- calibration dump (for pack_trained GPTQ) ----------------
@torch.no_grad()
def calib_dump(mod, head_emb, ds, n_tokens, Lmax, out_dir):
    """Fill-pass over calibration sequences; collect GEMV inputs per pack tensor:
    cat (eh in), xh (q/k/v/gate/up in), hhx (fg/fu in), gact (fd in), ao (o in).
    Saves X^T X Hessians (fp64) per tensor + column counts."""
    os.makedirs(out_dir, exist_ok=True)
    acc = {nm: None for nm in ("d_eh", "d_qkvxh", "d_fgx", "d_ao", "d_fdg")}
    cnt = 0
    idx = np.random.default_rng(0).permutation(ds.N)
    i = 0
    while cnt < n_tokens:
        ids_np, h_np = ds.sample(int(idx[i % ds.N]), Lmax)
        i += 1
        ids = torch.from_numpy(ids_np)[None].cuda()
        h = torch.from_numpy(h_np)[None].cuda()
        B, L = ids.shape
        h_prev = torch.zeros_like(h)
        h_prev[:, 1:] = h[:, :-1]
        embs = head_emb.embed(ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xin = mod.xin_from(embs, h_prev)
            xh = rms_norm(xin, mod.attn_norm_w)
            cat = torch.cat([rms_norm(embs, mod.enorm_w), rms_norm(h_prev, mod.hnorm_w)], dim=-1)
            q, k, v = mod.qkv_of(xh)
        # attention with fill cache (reuse mod.fill internals via the public fn)
        hd, Kf, Vf = None, None, None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hd, Kf, Vf = mod.fill(embs, h_prev)
        pos = torch.arange(L, device=ids.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qq, g = mod.qvec_gate(q, pos)
            kk = mod.kvecs(k, pos)
            vv = v.float().reshape(B, L, NKV, HD)
            qq = qq / math.sqrt(HD)
            kkT = kk.permute(0, 2, 1, 3)
            qq2 = qq.permute(0, 2, 1, 3).reshape(B, NH, L, HD)
            kkT6 = kkT.repeat_interleave(NH // NKV, dim=1)
            sc = torch.matmul(qq2, kkT6.transpose(-1, -2))
            causal = torch.ones(L, L, device=ids.device, dtype=torch.bool).tril()
            sc = sc.masked_fill(~causal, float("-inf"))
            p = torch.softmax(sc.float(), dim=-1)
            o = torch.matmul(p, Vf.repeat_interleave(NH // NKV, dim=1))   # Vf already [B,NKV,L,HD]
            o = o.permute(0, 2, 1, 3).reshape(B, L, NH * HD)
            ao = o * torch.sigmoid(g.reshape(B, L, NH * HD))
            attn_out = F.linear(ao, mod.wo)
            hh = xin + attn_out
            hhx = rms_norm(hh, mod.post_norm_w)
            gact = F.silu(F.linear(hhx, mod.w_gate)) * F.linear(hhx, mod.w_up)
        Xs = {"d_eh": cat.float().reshape(-1, 10240), "d_qkvxh": xh.float().reshape(-1, DIM),
              "d_fgx": hhx.float().reshape(-1, DIM), "d_ao": ao.float().reshape(-1, NH * HD),
              "d_fdg": gact.float().reshape(-1, 17408)}
        for nm, X in Xs.items():
            H = (X.T.double() @ X.double()).cpu().numpy()
            acc[nm] = H if acc[nm] is None else acc[nm] + H
        cnt += B * L
        if i % 20 == 0:
            print(f"  [calib] {cnt} tokens", flush=True)
    for nm, H in acc.items():
        np.save(f"{out_dir}/{nm}_hess.npy", H)
    json.dump({"tokens": int(cnt)}, open(f"{out_dir}/meta.json", "w"))
    print(f"[calib] done: {cnt} tokens -> {out_dir}")


# ---------------- training ----------------
def load_frozen(weights_dir, dev):
    def L(name):
        return torch.from_numpy(np.load(f"{weights_dir}/{name}.npy"))
    emb = L("model_language_model_embed_tokens_weight")
    head = L("lm_head_weight")
    return FrozenHeadEmb(emb, head).to(dev).eval()


def train(args):
    dev = torch.device(args.device)
    ds = ShardDataset(args.data)
    print(f"[train] {ds.N} sequences, Lmax(meta) {ds.meta.get('lmax')}")
    sd = {}
    import glob
    for f in glob.glob(f"{args.weights}/*.npy"):
        k = os.path.basename(f)[:-4]
        sd[{"mtp_fc_weight": "mtp.fc.weight",
            "mtp_layers_0_self_attn_q_proj_weight": "mtp.layers.0.self_attn.q_proj.weight",
            "mtp_layers_0_self_attn_k_proj_weight": "mtp.layers.0.self_attn.k_proj.weight",
            "mtp_layers_0_self_attn_v_proj_weight": "mtp.layers.0.self_attn.v_proj.weight",
            "mtp_layers_0_self_attn_o_proj_weight": "mtp.layers.0.self_attn.o_proj.weight",
            "mtp_layers_0_mlp_gate_proj_weight": "mtp.layers.0.mlp.gate_proj.weight",
            "mtp_layers_0_mlp_up_proj_weight": "mtp.layers.0.mlp.up_proj.weight",
            "mtp_layers_0_mlp_down_proj_weight": "mtp.layers.0.mlp.down_proj.weight",
            "mtp_layers_0_input_layernorm_weight": "mtp.layers.0.input_layernorm.weight",
            "mtp_layers_0_post_attention_layernorm_weight": "mtp.layers.0.post_attention_layernorm.weight",
            "mtp_norm_weight": "mtp.norm.weight",
            "mtp_pre_fc_norm_embedding_weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp_pre_fc_norm_hidden_weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp_layers_0_self_attn_q_norm_weight": "mtp.layers.0.self_attn.q_norm.weight",
            "mtp_layers_0_self_attn_k_norm_weight": "mtp.layers.0.self_attn.k_norm.weight"}.get(k, k)] = torch.from_numpy(np.load(f))
    mod = DraftBlock.from_hf(sd, torch.float32).to(dev)
    head_emb = load_frozen(args.weights, dev)
    for p in head_emb.parameters() if hasattr(head_emb, "parameters") else []:
        p.requires_grad_(False)
    head_emb.emb.requires_grad_(False)
    head_emb.head.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.wd)
    total = args.steps
    warm = min(args.warmup, total // 10)
    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    tok_count = 0
    lens_sorted = np.argsort(ds.lens)   # length bucketing: sample within a window
    for step in range(total):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        anchor = rng.integers(0, ds.N)
        lo = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] - 128)
        hi = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] + 128)
        pick = lens_sorted[rng.integers(lo, max(lo + 1, hi), args.batch)]
        idx = pick
        ids_l, h_l = [], []
        for j in idx:
            a, b = ds.sample(int(j), args.lmax)
            ids_l.append(torch.from_numpy(a))
            h_l.append(torch.from_numpy(b))
        L = min(x.shape[0] for x in ids_l)
        ids = torch.stack([x[:L] for x in ids_l]).to(dev)
        h = torch.stack([x[:L] for x in h_l]).to(dev)
        t = torch.from_numpy(rng.integers(args.lmin, L - args.S - 1, args.batch)).to(dev)
        losses, accs, _ = chain_forward(mod, head_emb, ids, h, t, args.S, feedback=args.feedback)
        loss = losses.sum() / args.batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), args.clip)
        opt.step()
        tok_count += args.batch * L
        if step % args.log == 0 or step == total - 1:
            print(f"[train] step {step:5d}/{total} lr {lr_at(step):.2e} loss {loss.item()/args.S:.4f} "
                  f"acc1 {float(accs[0])/args.batch:.3f} accS {[round(float(a)/args.batch,2) for a in accs]} "
                  f"gn {float(gn):.2f} tok {tok_count/1e6:.2f}M {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % args.ckpt == 0 or step == total - 1:
            sd2 = {k: v.detach().cpu().float() for k, v in mod.export_hf().items()}
            torch.save({"sd": sd2, "step": step, "tokens": tok_count}, f"{args.out}/ckpt_{step+1}.pt")
            print(f"[ckpt] {args.out}/ckpt_{step+1}.pt", flush=True)
    if args.calib_tokens > 0:
        calib_dump(mod, head_emb, ds, args.calib_tokens, args.lmax, f"{args.out}/calib")


def overfit_real(args):
    """Validation B on a real 10k-token slice: same data every step."""
    dev = torch.device(args.device)
    ds = ShardDataset(args.data)
    sd = {}
    import glob
    keymap = {
        "mtp_fc_weight": "mtp.fc.weight", "mtp_layers_0_self_attn_q_proj_weight": "mtp.layers.0.self_attn.q_proj.weight",
        "mtp_layers_0_self_attn_k_proj_weight": "mtp.layers.0.self_attn.k_proj.weight",
        "mtp_layers_0_self_attn_v_proj_weight": "mtp.layers.0.self_attn.v_proj.weight",
        "mtp_layers_0_self_attn_o_proj_weight": "mtp.layers.0.self_attn.o_proj.weight",
        "mtp_layers_0_mlp_gate_proj_weight": "mtp.layers.0.mlp.gate_proj.weight",
        "mtp_layers_0_mlp_up_proj_weight": "mtp.layers.0.mlp.up_proj.weight",
        "mtp_layers_0_mlp_down_proj_weight": "mtp.layers.0.mlp.down_proj.weight",
        "mtp_layers_0_input_layernorm_weight": "mtp.layers.0.input_layernorm.weight",
        "mtp_layers_0_post_attention_layernorm_weight": "mtp.layers.0.post_attention_layernorm.weight",
        "mtp_norm_weight": "mtp.norm.weight", "mtp_pre_fc_norm_embedding_weight": "mtp.pre_fc_norm_embedding.weight",
        "mtp_pre_fc_norm_hidden_weight": "mtp.pre_fc_norm_hidden.weight",
        "mtp_layers_0_self_attn_q_norm_weight": "mtp.layers.0.self_attn.q_norm.weight",
        "mtp_layers_0_self_attn_k_norm_weight": "mtp.layers.0.self_attn.k_norm.weight"}
    for f in glob.glob(f"{args.weights}/*.npy"):
        k = os.path.basename(f)[:-4]
        sd[keymap.get(k, k)] = torch.from_numpy(np.load(f))
    mod = DraftBlock.from_hf(sd, torch.float32).to(dev)
    head_emb = load_frozen(args.weights, dev)
    # fixed tiny slice: first K sequences totaling ~10k tokens
    ids_l, h_l = [], []
    tot = 0
    for j in range(ds.N):
        a, b = ds.sample(j, 1024)
        ids_l.append(torch.from_numpy(a))
        h_l.append(torch.from_numpy(b))
        tot += len(a)
        if tot >= 10000:
            break
    L = min(x.shape[0] for x in ids_l)
    ids = torch.stack([x[:L] for x in ids_l]).to(dev)
    h = torch.stack([x[:L] for x in h_l]).to(dev)
    B = ids.shape[0]
    print(f"[overfit-real] B={B} L={L} ({B*L} tokens)")
    opt = torch.optim.AdamW(mod.parameters(), lr=2e-4)
    t_fix = torch.full((B,), L - args.S - 2, dtype=torch.long, device=dev)
    for step in range(args.overfit_steps):
        t = t_fix if step > args.overfit_steps * 0.7 else torch.randint(16, L - args.S - 1, (B,), device=dev)
        losses, accs, _ = chain_forward(mod, head_emb, ids, h, t, args.S, feedback=args.feedback)
        loss = losses.sum() / B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step >= args.overfit_steps - 3:
            print(f"  step {step:4d} loss {loss.item()/args.S:.4f} acc {[round(float(a)/B,3) for a in accs]}", flush=True)
    losses, accs, _ = chain_forward(mod, head_emb, ids, h, t_fix, args.S, feedback=args.feedback)
    print("[overfit-real] FINAL fixed-anchor loss %.4f acc %s" %
          (losses.sum().item() / B, [round(float(a) / B, 3) for a in accs]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "overfit-synthetic", "overfit-real", "calib"])
    ap.add_argument("--data", required=False)
    ap.add_argument("--ckpt-in", default=None)
    ap.add_argument("--weights", default=os.path.expanduser("~/drafter/weights"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=1024)
    ap.add_argument("--lmin", type=int, default=64)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt", type=int, default=400)
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feedback", default="slice", choices=["slice", "full"])
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--calib-tokens", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "overfit-synthetic":
        ok = overfit_synthetic(a)
        sys.exit(0 if ok else 1)
    elif a.mode == "overfit-real":
        overfit_real(a)
    elif a.mode == "calib":
        ds = ShardDataset(a.data)
        if a.ckpt_in and a.ckpt_in != "None":
            sd = load_sd_ckpt(a.ckpt_in)
            mod = DraftBlock.from_hf(sd, torch.float32).to(a.device)
        else:
            mod = DraftBlock(torch.float32).to(a.device)
        head_emb = load_frozen(a.weights, a.device)
        calib_dump(mod, head_emb, ds, a.calib_tokens or 100000, a.lmax, a.out + "/calib")
    else:
        train(a)
