# TLX DRAFTER Phase 1 (Stage A) — TTT trainer for the blk.64 drafter.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""EAGLE-3 recipe, chain-shaped to the engine's serve semantics:

  fill   (parallel, teacher-forced): xin_q = eh_proj(enorm(emb(x_q)) || hnorm(h_{q-1}))
           -> causal own-KV cache (K/V of xin_q) [the fill_draft analog; the
           engine fill replays RECORDED TRUNK hiddens — true-h is serve-faithful]
  anchor (per sampled position t): xin = eh_proj(enorm(emb(x_t)) || hnorm(h_t^true))
           — the serve step-0 convention; its K/V REPLACE fill slot t.
  steps  i=1..S-1: xin from (OWN argmax token, OWN hd_{i-1}) at pos t+i, K/V appended.
  loss   = sum_i CE(head(shared_norm(hd_i)), x_{t+i+1})   token-only (no feature
           regression — the EAGLE-3 ablation), full vocab, frozen trunk head.

v2 (the G1 long-ctx fix):
  --data takes MULTIPLE shard dirs (comma-sep) with per-dir weight
  "dir:weight" — each dir's meta.json carries lmax/anchors/tail_anchor/batch_hint.
  ABSOLUTE positions from offs.npy (RoPE at serve ranges; long docs left-cropped
  so window end == doc end). hpre.npy = trunk hidden at window start-1 (the
  fill's first hm). Long windows: SDPA flash fill + SEGMENT attention (fill rows
  [0,t) + own rows, no one-hot scatter, no cross-anchor contamination) +
  MULTI-ANCHOR per window (anchors near the window end when tail_anchor=1 —
  the serve regime). fp16 rolling ckpt + weight-only interval ckpts.

Chain feedback argmax is SLICE-restricted (serve argmaxes over the 40960-row
prompt slice; approximated by the window's own distinct ids).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as tk_ckpt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DraftBlock, FrozenHeadEmb, rms_norm, DIM, NH, NKV, HD, VOCAB


# ---------------- data ----------------
class ShardDataset:
    def __init__(self, d, weight=1.0):
        self.dir = d
        self.ids = np.load(f"{d}/ids.npy", mmap_mode="r")
        self.hids = np.load(f"{d}/hids.npy", mmap_mode="r")
        self.lens = np.load(f"{d}/lens.npy")
        self.offs = np.load(f"{d}/offs.npy") if os.path.exists(f"{d}/offs.npy") else np.zeros(len(self.lens), np.int32)
        self.hpre = np.load(f"{d}/hpre.npy", mmap_mode="r") if os.path.exists(f"{d}/hpre.npy") else None
        self.meta = json.load(open(f"{d}/meta.json"))
        self.N = len(self.lens)
        self.weight = weight
        self.anchors = int(self.meta.get("anchors", 1))
        self.tail_anchor = bool(self.meta.get("tail_anchor", 0))
        self.batch_hint = int(self.meta.get("batch_hint", 0))

    def window(self, idx, Lmax):
        """-> (ids [L], h [L,5120] f32, off int, hpre [5120] f32 or None)"""
        n = int(self.lens[idx])
        lo = max(0, n - Lmax)
        ids = np.asarray(self.ids[idx, lo:n], dtype=np.int64)
        h = np.asarray(self.hids[idx, lo:n], dtype=np.float32)
        off = int(self.offs[idx]) + lo
        hp = np.asarray(self.hpre[idx], dtype=np.float32) if (self.hpre is not None and lo == 0) else None
        return ids, h, off, hp


class MultiDir:
    """dir-weighted sampler with per-dir batch/anchor config + prefetch."""

    def __init__(self, spec, lmax_default=2048):
        self.dirs = []
        weights = []
        for part in spec.split(","):
            d, _, w = part.partition(":")
            self.dirs.append(ShardDataset(d))
            weights.append(float(w) if w else 1.0)
        tot = sum(weights)
        self.p = [w / tot for w in weights]

    def pick(self, rng):
        i = int(rng.choice(len(self.dirs), p=self.p))
        return self.dirs[i]


def make_batch(ds, idxs, lmax, dev):
    ids_l, h_l, off_l, hp_l = [], [], [], []
    for j in idxs:
        a, b, o, hp = ds.window(int(j), lmax)
        ids_l.append(torch.from_numpy(a))
        h_l.append(torch.from_numpy(b))
        off_l.append(o)
        hp_l.append(torch.from_numpy(hp) if hp is not None else torch.zeros(DIM))
    L = min(x.shape[0] for x in ids_l)
    ids = torch.stack([x[:L] for x in ids_l]).to(dev)
    h = torch.stack([x[:L] for x in h_l]).to(dev)
    offs = torch.tensor(off_l, dtype=torch.long, device=dev)
    hpre = torch.stack(hp_l).to(dev)
    return ids, h, offs, hpre


# ---------------- the TTT chain (SHORT path — pilot-validated) ----------------
def chain_forward(mod, head_emb, ids, h_true, t, S, feedback="slice", compute_head=True):
    """One TTT unroll, one anchor per window. ids/h_true [B, L]; t [B]."""
    B, L = ids.shape
    dev = ids.device
    embs = head_emb.embed(ids)
    h_prev = torch.zeros_like(h_true)
    h_prev[:, 1:] = h_true[:, :-1]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=embs.is_cuda), \
         torch.autocast("mps", dtype=torch.float16, enabled=(not embs.is_cuda and embs.device.type == "mps")):
        hd_fill, Kfill, Vfill = mod.fill(embs, h_prev)
    Lcap = L + S
    Kc = torch.zeros(B, NKV, Lcap, HD, device=dev, dtype=torch.float32)
    Vc = torch.zeros(B, NKV, Lcap, HD, device=dev, dtype=torch.float32)
    Kc[:, :, :L] = Kfill.float()
    Vc[:, :, :L] = Vfill.float()
    ar = torch.arange(Lcap, device=dev)[None, None, :]
    pos_t = t.to(dev)
    hm = h_true[torch.arange(B, device=dev), pos_t]
    tok = ids[torch.arange(B, device=dev), pos_t]
    one_t = (ar == pos_t[:, None, None]).float().unsqueeze(-1)
    losses, accs = [], []
    slice_mask = None
    Vh = head_emb.head.shape[0]
    if feedback == "slice":
        slice_mask = torch.zeros(B, Vh, device=dev, dtype=torch.bool)
        slice_mask.scatter_(1, ids.clamp(max=Vh - 1), True)
    for i in range(S):
        pos = pos_t + i
        xin = mod.xin_from(head_emb.embed(tok), hm)
        xh = rms_norm(xin, mod.attn_norm_w)
        q, k, v = mod.qkv_of(xh)
        qq, g = mod.qvec_gate(q, pos)
        kk = mod.kvecs(k, pos)
        vv = v.float().reshape(B, NKV, HD)
        one = (ar == pos[:, None, None]).float().unsqueeze(-1)
        Kc = Kc * (1 - one) + kk.unsqueeze(2) * one
        Vc = Vc * (1 - one) + vv.unsqueeze(2) * one
        qq = qq / math.sqrt(HD)
        kkx = Kc.repeat_interleave(NH // NKV, dim=1)
        sc = torch.einsum("bhd,bhtd->bht", qq, kkx)
        causal = ar <= pos[:, None, None]
        sc = sc.masked_fill(~causal, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        o = torch.einsum("bht,bhtd->bhd", p, Vc.repeat_interleave(NH // NKV, dim=1))
        ao = o.reshape(B, NH * HD) * torch.sigmoid(g.reshape(B, NH * HD))
        hd = mod.block_tail(xin, ao)
        hi = rms_norm(hd, mod.shared_head_norm_w)
        logits = head_emb.logits(hi)
        lbl = ids[torch.arange(B, device=dev), torch.clamp(pos + 1, max=L - 1)]
        valid = (pos + 1) < L
        lv = F.cross_entropy(logits, lbl, reduction="none")
        lv = lv * valid.float()
        losses.append(lv.sum())
        accs.append(((logits.argmax(-1) == lbl).float() * valid.float()).sum())
        with torch.no_grad():
            if feedback == "slice":
                lm = logits.masked_fill(~slice_mask, float("-inf"))
            else:
                lm = logits
            tok = lm.argmax(-1).detach()
        hm = hd
    return torch.stack(losses), torch.stack(accs), int(valid.sum().item()) if S else 0


# ---------------- the TTT chain (LONG path: SDPA fill + segment attention) ----
def fill_sdpa(mod, embs, h_prev, offs):
    """Flash fill. embs/h_prev [B,L,5120]; offs [B] absolute window starts.
    Returns K [B,NKV,L,HD] f32, V [B,NKV,L,HD] f32 (normed/roped)."""
    B, L, _ = embs.shape
    dev = embs.device
    pos = offs[:, None] + torch.arange(L, device=dev)[None]      # [B,L]
    from model import rms_norm_dim, apply_rope
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xin = mod.xin_from(embs, h_prev)
        xh = rms_norm(xin, mod.attn_norm_w)
        q, k, v = mod.qkv_of(xh)
        qf = q.float().reshape(B, L, NH, 2, HD)
        qq = qf[..., 0, :]                                        # [B,L,NH,HD]
        g = qf[..., 1, :]
        qq = _rope_b(mod, rms_norm_dim(qq, mod.q_norm_w.float()), pos)
        kf = k.float().reshape(B, L, NKV, HD)
        kk = _rope_b(mod, rms_norm_dim(kf, mod.k_norm_w.float()), pos)
        vv = v.float().reshape(B, L, NKV, HD)
        qq = (qq / math.sqrt(HD)).permute(0, 2, 1, 3)            # [B,NH,L,HD]
        kkT = kk.permute(0, 2, 1, 3)                              # [B,NKV,L,HD]
        vvT = vv.permute(0, 2, 1, 3)
        rep = NH // NKV
        o = F.scaled_dot_product_attention(
            qq, kkT.repeat_interleave(rep, dim=1), vvT.repeat_interleave(rep, dim=1),
            is_causal=True, scale=1.0)                            # scale: qq already /16
        o = o.permute(0, 2, 1, 3).reshape(B, L, NH * HD)
        gate = torch.sigmoid(g.reshape(B, L, NH * HD))
        ao = o * gate
        attn_out = F.linear(ao, mod.wo)
        hh = xin + attn_out
        hhx = rms_norm(hh, mod.post_norm_w)
        gact = F.silu(F.linear(hhx, mod.w_gate)) * F.linear(hhx, mod.w_up)
        hd = hh + F.linear(gact, mod.w_down)
    return hd, kkT.float(), vvT.float()


def _rope_b(mod, x, pos):
    """x [B,L,H,HD] rope with per-batch pos [B,L] (vectorized)."""
    from model import apply_rope
    out = x.clone()
    for b in range(x.shape[0]):
        out[b] = apply_rope(x[b], pos[b], mod.fr)
    return out


def chain_forward_long(mod, head_emb, ids, h_true, offs, hpre, anchors_t, S,
                       feedback="slice", fill_ckpt=False):
    """Multi-anchor long-window path. ids/h_true [B,L]; offs [B] absolute window
    starts; hpre [B,5120] trunk hidden at window start-1; anchors_t [B,A].
    Returns (loss_sum_grad, accs [S], n_sup)."""
    B, L = ids.shape
    A = anchors_t.shape[1]
    dev = ids.device
    embs = head_emb.embed(ids)
    h_prev = torch.zeros_like(h_true)
    h_prev[:, 0] = hpre                      # trunk hidden at window start-1 (0 if off=0)
    h_prev[:, 1:] = h_true[:, :-1]
    if fill_ckpt:
        hd_fill, Kf, Vf = tk_ckpt(fill_sdpa, mod, embs, h_prev, offs, use_reentrant=False)
    else:
        hd_fill, Kf, Vf = fill_sdpa(mod, embs, h_prev, offs)
    tb = anchors_t.to(dev)                                   # [B,A] window-relative
    ar_b = torch.arange(B, device=dev)[:, None]
    hm = h_true[ar_b, tb].reshape(B * A, DIM)                # anchor step-0 hm = h_t^true
    tok = ids[ar_b, tb].reshape(-1)
    offs_f = (offs.to(dev)[:, None] + tb).reshape(-1)        # [B*A] absolute positions
    t_rel = tb.reshape(-1)
    T = int(t_rel.max())
    # padded fill segment (built ONCE; autograd flows into Kf/Vf views)
    Kfp = torch.zeros(B * A, NKV, T, HD, device=dev, dtype=torch.float32)
    Vfp = torch.zeros(B * A, NKV, T, HD, device=dev, dtype=torch.float32)
    for j in range(B * A):
        tj = int(t_rel[j])
        if tj > 0:
            Kfp[j, :, :tj] = Kf[j // A, :, :tj]
            Vfp[j, :, :tj] = Vf[j // A, :, :tj]
    ar_t = torch.arange(T, device=dev)
    mask1 = (ar_t[None, None, None, :] >= t_rel[:, None, None, None])   # [A,1,1,T]
    # own K/V per step kept as a LIST (no in-place writes; stacked per step — tiny)
    Ko_l, Vo_l = [], []
    losses, accs = [], []
    Vh = head_emb.head.shape[0]
    slice_mask = None
    if feedback == "slice":
        slice_mask = torch.zeros(B, Vh, device=dev, dtype=torch.bool)
        slice_mask.scatter_(1, ids.clamp(max=Vh - 1), True)
        slice_mask = slice_mask.repeat_interleave(A, dim=0)  # [B*A, V]
    Lidx = torch.arange(B * A, device=dev) // A              # anchor j -> window j//A
    for i in range(S):
        pos = offs_f + i
        xin = mod.xin_from(head_emb.embed(tok), hm)
        xh = rms_norm(xin, mod.attn_norm_w)
        q, k, v = mod.qkv_of(xh)
        qq, g = mod.qvec_gate(q, pos)
        kk = mod.kvecs(k, pos)
        vv = v.float().reshape(-1, NKV, HD)
        Ko_l.append(kk.float())
        Vo_l.append(vv)
        qq = qq / math.sqrt(HD)
        rep = NH // NKV
        # segment attention (GQA-grouped: scores [A,NKV,rep,T]); fill rows
        # [0,t) + own rows [0..i] (slot-t replacement exact: own row 0 REPLACES
        # fill row t; rows > t+i causally masked)
        Ko_c = torch.stack(Ko_l, dim=2)                     # [B*A,NKV,i+1,HD]
        Vo_c = torch.stack(Vo_l, dim=2)
        qg = qq.reshape(B * A, NKV, rep, HD)
        S1 = torch.einsum("agrd,agtd->agrt", qg, Kfp).masked_fill(mask1, float("-inf"))
        S2 = torch.einsum("agrd,agtd->agrt", qg, Ko_c.float())
        m = torch.maximum(S1.max(dim=-1, keepdim=True).values,
                          S2.max(dim=-1, keepdim=True).values)
        P1 = torch.exp(S1 - m)
        P2 = torch.exp(S2 - m)
        Z = P1.sum(-1, keepdim=True) + P2.sum(-1, keepdim=True) + 1e-20
        O = (torch.einsum("agrt,agtd->agrd", P1 / Z, Vfp) +
             torch.einsum("agrt,agtd->agrd", P2 / Z, Vo_c.float()))
        o = O.reshape(B * A, NH * HD)
        gate = torch.sigmoid(g.reshape(B * A, NH * HD))
        ao = o * gate
        hd = mod.block_tail(xin, ao)
        hi = rms_norm(hd, mod.shared_head_norm_w)
        logits = head_emb.logits(hi)                          # [B*A, V] fp32
        pos_next = t_rel + i + 1
        lbl = ids[Lidx, torch.clamp(pos_next, max=L - 1)]
        valid = pos_next < L
        lv = F.cross_entropy(logits, lbl, reduction="none")
        lv = lv * valid.float()
        losses.append(lv.sum())
        accs.append(((logits.argmax(-1) == lbl).float() * valid.float()).sum())
        with torch.no_grad():
            lm = logits.masked_fill(~slice_mask, float("-inf")) if feedback == "slice" else logits
            tok = lm.argmax(-1).detach()
        hm = hd
    return torch.stack(losses).sum(), torch.stack(accs), B * A


# ---------------- calibration dump (for pack_trained GPTQ) ----------------
@torch.no_grad()
def calib_dump(mod, head_emb, dirs, n_tokens, lmax, out_dir):
    """Fill-pass over calibration sequences from ALL dirs (weighted); collect
    GEMV-input Hessians per pack tensor + column counts."""
    os.makedirs(out_dir, exist_ok=True)
    acc = {nm: None for nm in ("d_eh", "d_qkvxh", "d_fgx", "d_ao", "d_fdg")}
    cnt = 0
    md = MultiDir(dirs)
    rng = np.random.default_rng(0)
    per = n_tokens // max(len(md.dirs), 1)
    for ds in md.dirs:
        c_ds = 0
        idx = rng.permutation(ds.N)
        i = 0
        while c_ds < per:
            if i >= ds.N:
                break
            ids_np, h_np, off, hp = ds.window(int(idx[i % ds.N]), lmax)
            i += 1
            ids = torch.from_numpy(ids_np)[None].cuda()
            h = torch.from_numpy(h_np)[None].cuda()
            hprev = torch.zeros_like(h)
            hprev[:, 0] = torch.from_numpy(hp if hp is not None else np.zeros(DIM, np.float32)).cuda()
            hprev[:, 1:] = h[:, :-1]
            offs = torch.tensor([off], dtype=torch.long, device="cuda")
            embs = head_emb.embed(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hd, Kf, Vf = fill_sdpa(mod, embs, hprev, offs)
                xin = mod.xin_from(embs, hprev)
                xh = rms_norm(xin, mod.attn_norm_w)
                cat = torch.cat([rms_norm(embs, mod.enorm_w), rms_norm(hprev, mod.hnorm_w)], dim=-1)
                q, k, v = mod.qkv_of(xh)
                # attention output at the last positions (cheap replay of fill attn)
                pos = offs[:, None] + torch.arange(ids.shape[1], device="cuda")[None]
                from model import rms_norm_dim
                qq = rms_norm_dim(q.float().reshape(1, -1, NH, 2, HD)[..., 0, :], mod.q_norm_w.float())
                qq = _rope_b(mod, qq, pos)
                kk = rms_norm_dim(k.float().reshape(1, -1, NKV, HD), mod.k_norm_w.float())
                kk = _rope_b(mod, kk, pos)
                vv = v.float().reshape(1, -1, NKV, HD).permute(0, 2, 1, 3)
                qq2 = (qq / math.sqrt(HD)).permute(0, 2, 1, 3)
                o = F.scaled_dot_product_attention(qq2, kk.permute(0, 2, 1, 3).repeat_interleave(6, 1),
                                                   vv.repeat_interleave(6, 1), is_causal=True, scale=1.0)
                o = o.permute(0, 2, 1, 3).reshape(1, -1, NH * HD)
                g = q.float().reshape(1, -1, NH, 2, HD)[..., 1, :]
                ao = o * torch.sigmoid(g.reshape(1, -1, NH * HD))
                attn_out = F.linear(ao, mod.wo)
                hh = xin + attn_out
                hhx = rms_norm(hh, mod.post_norm_w)
                gact = F.silu(F.linear(hhx, mod.w_gate)) * F.linear(hhx, mod.w_up)
            # subsample rows to bound Hessian cost on long windows
            sel = torch.randperm(ids.shape[1])[:4096]
            Xs = {"d_eh": cat.float()[:, sel].reshape(-1, 10240),
                  "d_qkvxh": xh.float()[:, sel].reshape(-1, DIM),
                  "d_fgx": hhx.float()[:, sel].reshape(-1, DIM),
                  "d_ao": ao.float()[:, sel].reshape(-1, NH * HD),
                  "d_fdg": gact.float()[:, sel].reshape(-1, 17408)}
            for nm, X in Xs.items():
                H = (X.T.double() @ X.double()).cpu().numpy()
                acc[nm] = H if acc[nm] is None else acc[nm] + H
            c_ds += ids.shape[1]
            cnt += ids.shape[1]
            if i % 10 == 0:
                print(f"  [calib] {cnt} tokens", flush=True)
    for nm, H in acc.items():
        np.save(f"{out_dir}/{nm}_hess.npy", H)
    json.dump({"tokens": int(cnt)}, open(f"{out_dir}/meta.json", "w"))
    print(f"[calib] done: {cnt} tokens -> {out_dir}")


# ---------------- synthetic overfit (Validation B machinery) ----------------
def synth_corpus(B=6, L=256, seed=0, dev="cpu", voc=4096):
    g = torch.Generator().manual_seed(seed)
    emb = (torch.randn(voc, DIM, generator=g) * 0.02).to(torch.bfloat16)
    head = (torch.randn(voc, DIM, generator=g) * 0.02).to(torch.bfloat16)
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
    ok = losses.sum().item() / ids.shape[0] < 2.0 * args.S and float(accs[0]) / ids.shape[0] > 0.9
    print("[overfit-syn] %s" % ("PASS" if ok else "FAIL"))
    return ok


def overfit_long_synthetic(args):
    """Validation B for the LONG path (multi-anchor + segment attention + offs)."""
    dev = torch.device(args.device)
    torch.manual_seed(0)
    mod = DraftBlock(torch.float32).to(dev)
    with torch.no_grad():
        for n, p in mod.named_parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.008)
            else:
                p.fill_(1.0)
    ids, h, head_emb = synth_corpus(B=2, L=384, dev="cpu", voc=4096)
    ids, head_emb = ids.to(dev), head_emb.to(dev)
    h = h.to(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=3e-4)
    A = 2
    for step in range(args.overfit_steps):
        ta = torch.randint(200, 384 - args.S - 2, (2, A), device=dev) if step < args.overfit_steps - 5 \
            else torch.full((2, A), 384 - args.S - 2, dtype=torch.long, device=dev)
        loss, accs, n_sup = chain_forward_long(
            mod, head_emb, ids, h, torch.zeros(2, dtype=torch.long, device=dev),
            torch.zeros(2, DIM, device=dev), ta, args.S, feedback="full")
        loss = loss / (2 * A)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.overfit_steps - 1:
            print(f"  [long] step {step:4d} loss {loss.item():.3f} acc/step "
                  f"{[round(float(a)/(2*A),3) for a in accs]}", flush=True)
    ta = torch.full((2, A), 384 - args.S - 2, dtype=torch.long, device=dev)
    loss, accs, n_sup = chain_forward_long(mod, head_emb, ids, h,
                                           torch.zeros(2, dtype=torch.long, device=dev),
                                           torch.zeros(2, DIM, device=dev), ta, args.S, feedback="full")
    print("[overfit-long] FINAL loss %.4f acc %s" % (loss.item() / (2 * A),
          [round(float(a) / (2 * A), 3) for a in accs]))


def overfit_real(args):
    """Validation B on a real 10k-token slice: same data every step."""
    dev = torch.device(args.device)
    ds = ShardDataset(args.data.split(",")[0])
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
        "mtp_pre_fc_norm_hidden_weight": "mtp.pre_fc_norm.hidden_weight" if False else "mtp.pre_fc_norm_hidden.weight",
        "mtp_layers_0_self_attn_q_norm_weight": "mtp.layers.0.self_attn.q_norm.weight",
        "mtp_layers_0_self_attn_k_norm_weight": "mtp.layers.0.self_attn.k_norm.weight"}
    for f in glob.glob(f"{args.weights}/*.npy"):
        k = os.path.basename(f)[:-4]
        sd[keymap.get(k, k)] = torch.from_numpy(np.load(f))
    mod = DraftBlock.from_hf(sd, torch.float32).to(dev)
    head_emb = load_frozen(args.weights, dev)
    ids_l, h_l = [], []
    tot = 0
    for j in range(ds.N):
        a, b, o, hp = ds.window(j, 1024)
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


# ---------------- training ----------------
def load_frozen(weights_dir, dev):
    weights_dir = os.path.expanduser(weights_dir)
    def L(name):
        return torch.from_numpy(np.load(f"{weights_dir}/{name}.npy"))
    emb = L("model_language_model_embed_tokens_weight")
    head = L("lm_head_weight")
    return FrozenHeadEmb(emb, head).to(dev).eval()


def load_init_sd(weights_dir):
    weights_dir = os.path.expanduser(weights_dir)
    sd = {}
    import glob
    keymap = {
        "mtp_fc_weight": "mtp.fc.weight",
        "mtp_layers_0_self_attn_q_proj_weight": "mtp.layers.0.self_attn.q_proj.weight",
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
    for f in glob.glob(f"{weights_dir}/*.npy"):
        k = os.path.basename(f)[:-4]
        sd[keymap.get(k, k)] = torch.from_numpy(np.load(f))
    return sd


class Prefetcher:
    def __init__(self, fn, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.fn = fn
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            try:
                item = self.fn()
            except Exception as e:
                self.q.put(e)
                return
            self.q.put(item)

    def next(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item


def train(args):
    dev = torch.device(args.device)
    md = MultiDir(args.data)
    for ds in md.dirs:
        print(f"[train] dir {ds.dir}: {ds.N} windows, {ds.meta.get('tokens',0)/1e6:.2f}M tok, "
              f"lmax {ds.meta.get('lmax')}, anchors {ds.anchors}, tail {ds.tail_anchor}, w {ds.weight}")
    sd = load_init_sd(args.weights)
    mod = DraftBlock.from_hf(sd, torch.float32).to(dev)
    if args.ckpt_in:
        st = torch.load(args.ckpt_in, map_location="cpu", weights_only=False)
        cur = {k: v.float() for k, v in st["sd"].items()}
        mod = DraftBlock.from_hf(cur, torch.float32).to(dev)
        print(f"[train] init FROM {args.ckpt_in} (step {st.get('step')})")
    head_emb = load_frozen(args.weights, dev)
    head_emb.emb.requires_grad_(False)
    head_emb.head.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.wd)
    if args.ckpt_in and os.path.exists(args.ckpt_in + ".opt"):
        ost = torch.load(args.ckpt_in + ".opt", map_location="cpu", weights_only=False)
        opt.load_state_dict(ost["opt"])
        print("[train] optimizer state restored")
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
    sup_count = 0
    next_ckpt_tok = args.ckpt_tokens
    n_ckpt = 0

    def next_batch():
        ds = md.pick(rng)
        B = args.batch if ds.batch_hint == 0 else min(args.batch, ds.batch_hint)
        Lmax = min(args.lmax, int(ds.meta.get("lmax", args.lmax)))
        lens_sorted = np.argsort(ds.lens)
        anchor = rng.integers(0, ds.N)
        lo = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] - 256)
        hi = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] + 256)
        pick = lens_sorted[rng.integers(lo, max(lo + 1, hi), B)]
        ids, h, offs, hpre = make_batch(ds, pick, Lmax, "cpu")
        A = ds.anchors
        Bn, Lb = ids.shape
        ta = np.zeros((Bn, A), np.int64)
        for b in range(Bn):
            if ds.tail_anchor:
                lo_a = int(Lb * 0.6)
                hi_a = Lb - args.S - 2
                ta[b] = rng.integers(lo_a, max(lo_a + 1, hi_a), A)
            else:
                ta[b] = rng.integers(args.lmin, max(args.lmin + 1, Lb - args.S - 2), A)
        return ds, ids, h, offs, hpre, torch.from_numpy(ta), Lmax

    pf = Prefetcher(next_batch, depth=2)
    for step in range(total):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        ds, ids_cpu, h_cpu, offs_cpu, hpre_cpu, ta_cpu, Lmax = pf.next()
        long_path = Lmax > 4096 or ds.anchors > 1
        ids = ids_cpu.to(dev, non_blocking=True)
        h = h_cpu.to(dev, non_blocking=True)
        offs = offs_cpu.to(dev, non_blocking=True)
        hpre = hpre_cpu.to(dev, non_blocking=True)
        ta = ta_cpu.to(dev, non_blocking=True)
        B = ids.shape[0]
        A = ds.anchors
        if long_path:
            loss, accs, n_sup = chain_forward_long(mod, head_emb, ids, h, offs, hpre, ta,
                                                   args.S, feedback=args.feedback,
                                                   fill_ckpt=args.fill_ckpt)
            loss = loss / (B * A)
        else:
            t0a = ta[:, 0]
            losses, accs, _ = chain_forward(mod, head_emb, ids, h, t0a, args.S, feedback=args.feedback)
            loss = losses.sum() / B
            n_sup = B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), args.clip)
        opt.step()
        tok_count += B * Lmax
        sup_count += n_sup * args.S
        if step % args.log == 0 or step == total - 1:
            print(f"[train] step {step:5d}/{total} lr {lr_at(step):.2e} loss {loss.item()/args.S:.4f} "
                  f"acc1 {float(accs[0])/n_sup:.3f} accS {[round(float(a)/n_sup,2) for a in accs]} "
                  f"gn {float(gn):.2f} tok {tok_count/1e6:.2f}M {time.time()-t0:.0f}s", flush=True)
        if (tok_count >= next_ckpt_tok or step == total - 1) and step > 0:
            sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
            torch.save({"sd": sd2, "step": step, "tokens": tok_count, "sup": sup_count},
                       f"{args.out}/ckpt_{step+1}.pt")
            torch.save({"sd": sd2, "step": step, "tokens": tok_count, "sup": sup_count,
                        "opt": opt.state_dict()}, f"{args.out}/last.pt")
            n_ckpt += 1
            next_ckpt_tok += args.ckpt_tokens
            print(f"[ckpt] {args.out}/ckpt_{step+1}.pt (tok {tok_count/1e6:.1f}M sup {sup_count/1e3:.0f}k)", flush=True)
    if args.calib_tokens > 0:
        calib_dump(mod, head_emb, args.data, args.calib_tokens, min(args.lmax, 8192), f"{args.out}/calib")


def load_sd_ckpt(p):
    import torch as _t
    return {k: v.float() for k, v in _t.load(p, map_location="cpu", weights_only=False)["sd"].items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "overfit-synthetic", "overfit-long",
                                                        "overfit-real", "calib"])
    ap.add_argument("--data", required=False,
                    help="comma list of shard dirs, optional :weight per dir")
    ap.add_argument("--ckpt-in", default=None)
    ap.add_argument("--weights", default=os.path.expanduser("~/drafter/weights"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=1024, help="cap on window length (per-dir lmax also applies)")
    ap.add_argument("--lmin", type=int, default=64)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt", type=int, default=400)
    ap.add_argument("--ckpt-tokens", type=int, default=2000000,
                    help="checkpoint interval by window-token count")
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feedback", default="slice", choices=["slice", "full"])
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--calib-tokens", type=int, default=0)
    ap.add_argument("--fill-ckpt", action="store_true", help="gradient-checkpoint the long fill")
    a = ap.parse_args()
    if a.mode == "overfit-synthetic":
        ok = overfit_synthetic(a)
        sys.exit(0 if ok else 1)
    elif a.mode == "overfit-long":
        overfit_long_synthetic(a)
    elif a.mode == "overfit-real":
        overfit_real(a)
    elif a.mode == "calib":
        md = MultiDir(a.data)
        if a.ckpt_in:
            sd = load_sd_ckpt(a.ckpt_in)
            mod = DraftBlock.from_hf(sd, torch.float32).to(a.device)
        else:
            mod = DraftBlock(torch.float32).to(a.device)
        head_emb = load_frozen(a.weights, a.device)
        calib_dump(mod, head_emb, a.data, a.calib_tokens or 100000, min(a.lmax, 8192), a.out + "/calib")
    else:
        train(a)
