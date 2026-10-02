# TLX DRAFTER Phase 2 (anchor-scale Stage B v2) — the multi-session anchor trainer.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Stage-B v2: fine-tune the Stage-A winner (ckpt_6000) on the SCALED anchor
set (2-10k engine decode anchors at 64k-99.4k serve positions from 30 FRESH
novel-prose sessions — the Phase-1 unlock: 51 anchors memorized, thousands
force generalization) + a protective corpus mix:

  anchors (0.55 of steps)  multi-session engine kv bases (detached) + REAL
                            40960-slice feedback, labels = committed streams
  prose16k (0.20)           repair the v1 -0.16 regression
  code8k   (0.10)           repair the v1 -0.25 regression
  gsm8k    (0.15)           protect the v1 +0.32 battery gain

CANARY EARLY-STOP (the Phase-1 lesson, mechanized): held-out anchor cycles
from 6 disjoint sessions are chain-evaluated (no_grad) every --eval-every
steps; the checkpoint kept is the BEST-CANARY one; training stops after
--patience non-improving evals. The train-subset a1 is logged alongside —
a1_train -> 1.0 while canary lags = the memorization signature.
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
from train import make_batch, load_frozen, MultiDir
import train as T
import train_stageb as SB


def chain_anchor(mod, head_emb, D, cyc_idx, S, sess_of=None):
    """v2 of chain_forward_r8: anchors on their OWN session's engine kv base
    (detached int8-quantized base + own appends), real 40960-slice feedback,
    labels = committed streams. sess_of: optional explicit cycle list (eval)."""
    dev = next(mod.parameters()).device
    cyc = [D["cycles"][i] for i in cyc_idx]
    n = len(cyc)
    sess = D["sess"]
    pos = torch.tensor([c["pos"] for c in cyc], device=dev)
    # base availability: the dump carries rows [0, base_len) (decode-start);
    # anchors past base_len (the engine's own prior-cycle draft rows) are NOT
    # in the dump — those rows are masked out (base-only conditioning; the
    # chain_sim arbiter replays true serve semantics, this only shapes
    # training input). mask = ar_t >= min(pos_j, blen_j).
    blen = torch.tensor([sess[int(c["sess"])]["base_len"] for c in cyc],
                        device=dev, dtype=pos.dtype)
    vis = torch.minimum(pos, blen)
    hs = D["h_seeds"][[int(c["h_idx"]) for c in cyc]]           # [n, 5120]
    tok = torch.tensor([c["cur"] for c in cyc], device=dev)
    hm = hs
    offs_f = pos
    T = int(pos.max())
    Kfp = torch.zeros(n, NKV, T, HD, device=dev, dtype=torch.float32)
    Vfp = torch.zeros(n, NKV, T, HD, device=dev, dtype=torch.float32)
    for j in range(n):
        pj = int(vis[j])
        if pj > 0:
            s = sess[int(cyc[j]["sess"])]
            # 24GB rental: works whether s["K"]/s["V"] live on CPU (pinned,
            # streamed H2D) or already on device (no-op .to) — the fp32 cast
            # happens at the Kfp assignment, identical numerics either way.
            Kfp[j, :, :pj] = s["K"][:, :pj, :].to(dev, non_blocking=True)
            Vfp[j, :, :pj] = s["V"][:, :pj, :].to(dev, non_blocking=True)
    ar_t = torch.arange(T, device=dev)
    mask1 = (ar_t[None, None, None, :] >= vis[:, None, None, None])
    sl_w = head_emb.head[D["stab"]]                     # [40960, 5120]
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
        lg_slice = F.linear(hi.to(torch.bfloat16), sl_w).float()
        prop = lg_slice.argmax(-1)
        props = D["stab"][prop]
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


@torch.no_grad()
def eval_anchor(mod, head_emb, D, idx, S, chunk=8):
    """Chunked no-grad chain eval (Kfp fp32 buffers are ~1.6GB per cycle row
    at pos ~99k — 48 rows would be ~98GB; chunks of 8 fit)."""
    mod.eval()
    tot_loss = 0.0
    acc_sum = torch.zeros(S)
    n_tot = 0
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        loss, accs, n = chain_anchor(mod, head_emb, D, sub, S)
        tot_loss += float(loss)
        acc_sum += torch.tensor([float(a) for a in accs])
        n_tot += n
    mod.train()
    return tot_loss / (n_tot * S), [float(a) / n_tot for a in acc_sum]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma dirs:probs (corpus mix)")
    ap.add_argument("--anchors", required=True, help="anchors_train.pt")
    ap.add_argument("--canary", required=True, help="anchors_canary.pt")
    ap.add_argument("--anchor-weight", type=float, default=0.55)
    ap.add_argument("--ckpt-in", required=True)
    ap.add_argument("--weights", default="/root/w")
    ap.add_argument("--out", default="/root/runs/stageb2")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-6)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8, help="anchor cycles per anchor step")
    ap.add_argument("--corpus-batch", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=16384)
    ap.add_argument("--anchors-per-step", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.ckpt_in, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in st["sd"].items()}
    mod = DraftBlock.from_hf(sd, torch.float32).to(dev)
    print(f"[v2] init from {a.ckpt_in} (step {st.get('step')}, {st.get('tokens',0)/1e6:.1f}M tok)", flush=True)
    head_emb = load_frozen(a.weights, dev)
    head_emb.emb.requires_grad_(False)
    head_emb.head.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=a.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    md = MultiDir(a.data)
    for ds in md.dirs:
        ds.labels = np.load(f"{ds.dir}/labels.npy", mmap_mode="r") if os.path.exists(f"{ds.dir}/labels.npy") else None
        print(f"[v2] dir {ds.dir}: {ds.N} rows {ds.meta.get('tokens',0)/1e6:.2f}M tok", flush=True)

    D = torch.load(a.anchors, map_location="cpu", weights_only=False)
    D["stab"] = D["stab"].to(dev)
    D["h_seeds"] = D["h_seeds"].to(dev)
    for s in D["sess"]:
        s["K"] = s["K"].to(dev)
        s["V"] = s["V"].to(dev)
    print(f"[v2] anchors: {len(D['cycles'])} cycles / {len(D['sess'])} sessions on device "
          f"({sum(s['K'].numel()+s['V'].numel() for s in D['sess'])*2/1e9:.1f}GB)", flush=True)
    C = torch.load(a.canary, map_location="cpu", weights_only=False)
    C["stab"] = D["stab"]
    C["h_seeds"] = C["h_seeds"].to(dev)
    for s in C["sess"]:
        s["K"] = s["K"].to(dev)
        s["V"] = s["V"].to(dev)
    print(f"[v2] canary: {len(C['cycles'])} cycles / {len(C['sess'])} sessions", flush=True)

    rng = np.random.default_rng(a.seed)
    can_idx = rng.choice(len(C["cycles"]), min(a.eval_n, len(C["cycles"])), replace=False).tolist()
    trn_idx = rng.choice(len(D["cycles"]), min(256, len(D["cycles"])), replace=False).tolist()

    # baseline canary
    cl, ca = eval_anchor(mod, head_emb, C, can_idx, a.S)
    print(f"[v2] canary BASELINE: loss {cl:.4f} a1 {ca[0]:.3f} a {[round(x,2) for x in ca]}", flush=True)

    total = a.steps
    warm = a.warmup

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    os.makedirs(a.out, exist_ok=True)
    best = dict(step=-1, canary_loss=1e9)
    hist = []
    no_improve = 0
    t0 = time.time()
    stop = False
    for step in range(total):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        use_anchor = rng.random() < a.anchor_weight
        if use_anchor:
            idx = rng.integers(0, len(D["cycles"]), a.anchors_per_step)
            loss, accs, n_sup = chain_anchor(mod, head_emb, D, idx, a.S)
            loss = loss / n_sup
            tag = "anchor"
        else:
            ds = md.pick(rng)
            B = min(a.corpus_batch, ds.batch_hint or a.corpus_batch)
            Lmax = min(a.lmax, int(ds.meta.get("lmax", a.lmax)))
            lens_sorted = np.argsort(ds.lens)
            anchor = rng.integers(0, ds.N)
            lo = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] - 256)
            hi = np.searchsorted(ds.lens[lens_sorted], ds.lens[anchor] + 256)
            pick = lens_sorted[rng.integers(lo, max(lo + 1, hi), B)]
            ids, h, offs, hpre = make_batch(ds, pick, Lmax, "cpu")
            lab_l = [np.asarray(ds.labels[int(j)], dtype=np.int64)[:h.shape[1]] for j in pick]
            Lb = min(len(x) for x in lab_l)
            labels = torch.from_numpy(np.stack([x[:Lb] for x in lab_l])).to(dev)
            ids, h = ids[:, :Lb].to(dev), h[:, :Lb].to(dev)
            offs, hpre = offs.to(dev), hpre.to(dev)
            A = 2
            ta = np.stack([rng.integers(64, Lb - a.S - 2, A) for _ in range(B)])
            ta = torch.from_numpy(ta).to(dev)
            loss, accs, n_sup = SB.chain_forward_lab(mod, head_emb, ids, h, offs, hpre,
                                                     labels, ta, a.S)
            loss = loss / n_sup
            tag = "corpus"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step % a.log == 0 or step == total - 1:
            print(f"[v2] step {step:4d}/{total} lr {lr_at(step):.2e} loss {loss.item()/a.S:.4f} "
                  f"acc1 {float(accs[0])/n_sup:.3f} gn {float(gn):.2f} "
                  f"{time.time()-t0:.0f}s ({tag})", flush=True)
        if (step + 1) % a.eval_every == 0 or step == total - 1:
            cl, ca = eval_anchor(mod, head_emb, C, can_idx, a.S)
            tl, ta_ = eval_anchor(mod, head_emb, D, trn_idx, a.S)
            # best-ckpt selection on CANARY LOSS (smooth, falling = real
            # held-out generalization; the a-mean is too quantized at these
            # levels — a1 stuck at 2-4/48 while CE falls steadily)
            hist.append(dict(step=step + 1, canary_loss=round(cl, 5),
                             canary_a=[round(x, 4) for x in ca],
                             train_a=[round(x, 4) for x in ta_], train_loss=round(tl, 5)))
            json.dump(hist, open(f"{a.out}/canary_hist.json", "w"), indent=1)
            print(f"[v2-eval] step {step+1}: canary loss {cl:.4f} a1 {ca[0]:.3f} "
                  f"a {[round(x,2) for x in ca]} | TRAIN a1 {ta_[0]:.3f} "
                  f"(gap {ta_[0]-ca[0]:+.3f})", flush=True)
            if best["step"] < 0 or cl < best["canary_loss"]:
                best = dict(step=step + 1, canary_loss=cl)
                sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
                torch.save({"sd": sd2, "step": step, "tokens": 0,
                            "src_ckpt": a.ckpt_in}, f"{a.out}/best.pt")
                print(f"[v2-ckpt] new best canary loss {cl:.4f} -> {a.out}/best.pt", flush=True)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    print(f"[v2] EARLY STOP: canary flat {no_improve} evals "
                          f"(best step {best['step']})", flush=True)
                    stop = True
        # periodic ckpt regardless (the curve)
        if (step + 1) % 200 == 0:
            sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
            torch.save({"sd": sd2, "step": step, "tokens": 0,
                        "src_ckpt": a.ckpt_in}, f"{a.out}/ckpt_{step+1}.pt")
        if stop:
            break
    # final ckpt
    sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
    torch.save({"sd": sd2, "step": step, "tokens": 0, "src_ckpt": a.ckpt_in},
               f"{a.out}/last.pt")
    print(f"[v2] DONE best canary loss {best['canary_loss']:.4f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
