# TLX DRAFTER Phase 1 (Stage B) — engine-trace adaptation trainer.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Short, low-LR fine-tune of the Stage-A winner on the ENGINE-distribution
traces (protects Stage-A gains):
  - shard dirs with labels.npy (greedy targets computed by stageb_build):
    prose16k / code8k / gsm8k — full chain training (fills + multi-anchor),
    CE target at step i = labels[b, t+i] (the chain_sim truth semantics).
  - r8_anchored.pt: the r8_prose decode trace — anchors on the ENGINE kv
    state (detached int8-quantized base + own appends), REAL 40960-slice
    feedback, labels = the committed stream (probe-verified greedy).
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
from model import DraftBlock, FrozenHeadEmb, rms_norm, DIM, NH, NKV, HD
from train import (ShardDataset, make_batch, load_init_sd, load_frozen, fill_sdpa,
                   MultiDir)
import train as T


def chain_forward_lab(mod, head_emb, ids, h_true, offs, hpre, labels, anchors_t, S,
                      feedback="slice"):
    """chain_forward_long with GREEDY labels: CE target at step i =
    labels[b, t+i] (labels[p] = the greedy token following position p)."""
    B, L = ids.shape
    A = anchors_t.shape[1]
    dev = ids.device
    embs = head_emb.embed(ids)
    h_prev = torch.zeros_like(h_true)
    h_prev[:, 0] = hpre
    h_prev[:, 1:] = h_true[:, :-1]
    hd_fill, Kf, Vf = fill_sdpa(mod, embs, h_prev, offs)
    tb = anchors_t.to(dev)
    ar_b = torch.arange(B, device=dev)[:, None]
    hm = h_true[ar_b, tb].reshape(B * A, DIM)
    tok = ids[ar_b, tb].reshape(-1)
    offs_f = (offs.to(dev)[:, None] + tb).reshape(-1)
    t_rel = tb.reshape(-1)
    T = int(t_rel.max())
    Kfp = torch.zeros(B * A, NKV, T, HD, device=dev, dtype=torch.float32)
    Vfp = torch.zeros(B * A, NKV, T, HD, device=dev, dtype=torch.float32)
    for j in range(B * A):
        tj = int(t_rel[j])
        if tj > 0:
            Kfp[j, :, :tj] = Kf[j // A, :, :tj]
            Vfp[j, :, :tj] = Vf[j // A, :, :tj]
    ar_t = torch.arange(T, device=dev)
    mask1 = (ar_t[None, None, None, :] >= t_rel[:, None, None, None])
    Ko_l, Vo_l = [], []
    losses, accs = [], []
    Vh = head_emb.head.shape[0]
    slice_mask = None
    if feedback == "slice":
        slice_mask = torch.zeros(B, Vh, device=dev, dtype=torch.bool)
        slice_mask.scatter_(1, ids.clamp(max=Vh - 1), True)
        slice_mask = slice_mask.repeat_interleave(A, dim=0)
    Lidx = torch.arange(B * A, device=dev) // A
    for i in range(S):
        pos = offs_f + i
        prel = t_rel + i
        xin = mod.xin_from(head_emb.embed(tok), hm)
        xh = rms_norm(xin, mod.attn_norm_w)
        q, k, v = mod.qkv_of(xh)
        qq, g = mod.qvec_gate(q, pos)
        kk = mod.kvecs(k, pos)
        Ko_l.append(kk.float())
        Vo_l.append(v.float().reshape(-1, NKV, HD))
        qq = qq / math.sqrt(HD)
        rep = NH // NKV
        Ko_c = torch.stack(Ko_l, dim=2)
        Vo_c = torch.stack(Vo_l, dim=2)
        qg = qq.reshape(B * A, NKV, rep, HD)
        S1 = torch.einsum("agrd,agtd->agrt", qg, Kfp).masked_fill(mask1, float("-inf"))
        S2 = torch.einsum("agrd,agtd->agrt", qg, Ko_c.float())
        m = torch.maximum(S1.max(dim=-1, keepdim=True).values, S2.max(dim=-1, keepdim=True).values)
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
        logits = head_emb.logits(hi)
        lbl = labels[Lidx, torch.clamp(prel, max=L - 1)]
        valid = prel < L
        lv = F.cross_entropy(logits, lbl, reduction="none") * valid.float()
        losses.append(lv.sum())
        accs.append(((logits.argmax(-1) == lbl).float() * valid.float()).sum())
        with torch.no_grad():
            lm = logits.masked_fill(~slice_mask, float("-inf")) if feedback == "slice" else logits
            tok = lm.argmax(-1).detach()
        hm = hd
    return torch.stack(losses).sum(), torch.stack(accs), B * A


def chain_forward_r8(mod, head_emb, r8, cyc_idx, S):
    """Anchors on the ENGINE kv state (detached base rows [0..pos) + own
    appends), real 40960-slice feedback, labels = committed stream."""
    dev = next(mod.parameters()).device
    cyc = [r8["cycles"][i] for i in cyc_idx]
    n = len(cyc)
    K_base, V_base = r8["K"], r8["V"]                    # [4, CAP, 256] fp32
    stab = r8["stab"]                                    # [40960]
    pos = torch.tensor([c["pos"] for c in cyc], device=dev)
    toks0 = torch.tensor([c["cur"] for c in cyc], device=dev)
    hs = r8["h_seeds"][torch.tensor([c["h_idx"] for c in cyc], device=dev)]  # [n, 5120]
    tok = toks0
    hm = hs
    offs_f = pos
    T = int(pos.max())
    Kfp = torch.zeros(n, NKV, T, HD, device=dev, dtype=torch.float32)
    Vfp = torch.zeros(n, NKV, T, HD, device=dev, dtype=torch.float32)
    for j in range(n):
        pj = int(pos[j])
        if pj > 0:
            Kfp[j, :, :pj] = K_base[:, :pj, :]
            Vfp[j, :, :pj] = V_base[:, :pj, :]
    ar_t = torch.arange(T, device=dev)
    mask1 = (ar_t[None, None, None, :] >= pos[:, None, None, None])
    sl_w = head_emb.head[stab]                           # [40960, 5120] bf16 slice rows
    Ko_l, Vo_l = [], []
    losses, accs = [], []
    for i in range(S):
        xin = mod.xin_from(head_emb.embed(tok), hm)
        xh = rms_norm(xin, mod.attn_norm_w)
        q, k, v = mod.qkv_of(xh)
        qq, g = mod.qvec_gate(q, offs_f + i)
        kk = mod.kvecs(k, offs_f + i)
        Ko_l.append(kk.float())
        Vo_l.append(v.float().reshape(-1, NKV, HD))
        qq = qq / math.sqrt(HD)
        rep = NH // NKV
        Ko_c = torch.stack(Ko_l, dim=2)
        Vo_c = torch.stack(Vo_l, dim=2)
        qg = qq.reshape(n, NKV, rep, HD)
        S1 = torch.einsum("agrd,agtd->agrt", qg, Kfp).masked_fill(mask1, float("-inf"))
        S2 = torch.einsum("agrd,agtd->agrt", qg, Ko_c.float())
        m = torch.maximum(S1.max(dim=-1, keepdim=True).values, S2.max(dim=-1, keepdim=True).values)
        P1 = torch.exp(S1 - m)
        P2 = torch.exp(S2 - m)
        Z = P1.sum(-1, keepdim=True) + P2.sum(-1, keepdim=True) + 1e-20
        O = (torch.einsum("agrt,agtd->agrd", P1 / Z, Vfp) +
             torch.einsum("agrt,agtd->agrd", P2 / Z, Vo_c.float()))
        o = O.reshape(n, NH * HD)
        gate = torch.sigmoid(g.reshape(n, NH * HD))
        ao = o * gate
        hd = mod.block_tail(xin, ao)
        hi = rms_norm(hd, mod.shared_head_norm_w)
        # slice-restricted argmax over the REAL 40960 slice (serve semantics)
        lg_slice = F.linear(hi.to(torch.bfloat16), sl_w).float()
        prop = lg_slice.argmax(-1)
        props = stab[prop]
        lbl = torch.tensor([int(c["tokens"][i]) if i < len(c["tokens"]) else -1
                            for c in cyc], device=dev)
        valid = lbl >= 0
        logits_full = head_emb.logits(hi)
        lv = F.cross_entropy(logits_full, lbl.clamp(min=0), reduction="none") * valid.float()
        losses.append(lv.sum())
        accs.append(((props == lbl).float() * valid.float()).sum())
        tok = props                                       # feedback = own slice proposal
        hm = hd
    return torch.stack(losses).sum(), torch.stack(accs), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma dirs with labels (corpus traces)")
    ap.add_argument("--r8", default=None, help="r8_anchored.pt")
    ap.add_argument("--r8-weight", type=float, default=0.3, help="frac of steps on r8")
    ap.add_argument("--ckpt-in", required=True)
    ap.add_argument("--weights", default="/root/w")
    ap.add_argument("--out", default="runs/stageb")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=49152)
    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--ckpt-tokens", type=int, default=300000)
    a = ap.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.ckpt_in, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in st["sd"].items()}
    mod = DraftBlock.from_hf(sd, torch.float32).to(dev)
    print(f"[sb] init from {a.ckpt_in} (step {st.get('step')}, {st.get('tokens',0)/1e6:.1f}M tok)")
    head_emb = load_frozen(a.weights, dev)
    head_emb.emb.requires_grad_(False)
    head_emb.head.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=a.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    md = MultiDir(a.data)
    for ds in md.dirs:
        ds.labels = np.load(f"{ds.dir}/labels.npy", mmap_mode="r") if os.path.exists(f"{ds.dir}/labels.npy") else None
        print(f"[sb] dir {ds.dir}: {ds.N} rows {ds.meta.get('tokens',0)/1e6:.2f}M tok labels={ds.labels is not None}")
    r8 = torch.load(a.r8, map_location="cpu", weights_only=False) if a.r8 else None
    if r8:
        r8["K"] = r8["K"].to(dev)
        r8["V"] = r8["V"].to(dev)
        r8["stab"] = r8["stab"].to(dev)
        r8["h_seeds"] = r8["h_seeds"].to(dev)
        print(f"[sb] r8: {len(r8['cycles'])} cycles on device")

    total = a.steps
    warm = a.warmup

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tok_count, next_ckpt = 0, a.ckpt_tokens
    for step in range(total):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        use_r8 = r8 is not None and rng.random() < a.r8_weight
        if use_r8:
            idx = rng.integers(0, len(r8["cycles"]), min(8, len(r8["cycles"])))
            loss, accs, n_sup = chain_forward_r8(mod, head_emb, r8, idx, a.S)
            loss = loss / n_sup
            tok_count += n_sup * 8            # effective-context accounting
        else:
            ds = md.pick(rng)
            B = min(a.batch, ds.batch_hint or a.batch)
            Lmax = min(a.lmax, int(ds.meta.get("lmax", a.lmax)))
            lens_sorted = np.argsort(ds.lens)
            anchor = rng.integers(0, ds.N)
            lo = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] - 256)
            hi = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] + 256)
            pick = lens_sorted[rng.integers(lo, max(lo + 1, hi), B)]
            ids, h, offs, hpre = make_batch(ds, pick, Lmax, "cpu")
            # labels batch [B, L]
            lab_l = [np.asarray(ds.labels[int(j)], dtype=np.int64)[:h.shape[1]] for j in pick]
            Lb = min(len(x) for x in lab_l)
            labels = torch.from_numpy(np.stack([x[:Lb] for x in lab_l])).to(dev)
            ids, h = ids[:, :Lb].to(dev), h[:, :Lb].to(dev)
            offs, hpre = offs.to(dev), hpre.to(dev)
            A = a.anchors
            ta = np.stack([rng.integers(64, Lb - a.S - 2, A) for _ in range(B)])
            ta = torch.from_numpy(ta).to(dev)
            loss, accs, n_sup = chain_forward_lab(mod, head_emb, ids, h, offs, hpre,
                                                  labels, ta, a.S)
            loss = loss / n_sup
            tok_count += B * Lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step % a.log == 0 or step == total - 1:
            print(f"[sb] step {step:4d}/{total} lr {lr_at(step):.2e} loss {loss.item()/a.S:.4f} "
                  f"acc1 {float(accs[0])/n_sup:.3f} accS {[round(float(x)/n_sup,2) for x in accs]} "
                  f"gn {float(gn):.2f} tok {tok_count/1e6:.2f}M {time.time()-t0:.0f}s "
                  f"({'r8' if use_r8 else 'corpus'})", flush=True)
        if tok_count >= next_ckpt or step == total - 1:
            sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
            torch.save({"sd": sd2, "step": step, "tokens": tok_count,
                        "src_ckpt": a.ckpt_in}, f"{a.out}/ckpt_{step+1}.pt")
            next_ckpt += a.ckpt_tokens
            print(f"[sb-ckpt] {a.out}/ckpt_{step+1}.pt", flush=True)


if __name__ == "__main__":
    main()
