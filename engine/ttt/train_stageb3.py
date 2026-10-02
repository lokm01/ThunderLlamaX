# TLX DRAFTER Phase 2 (Stage B v3) — FIRST-PARTY bf16 init anchor trainer.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Stage-B v3: the decisive iteration. Identical to v2 (train_stageb2.py)
EXCEPT:
  - init from the PRISTINE first-party bf16 nextn weights (--init-dir, the
    norm-law-corrected HF fetch = ~/drafter/weights_v2 lineage) — NO Stage-A
    lineage (falsified 3x for prose).
  - LR 2-3e-6 (protect a good init; v2 used 6e-6 on a damaged init).
  - --curve-every (default 100): periodic ckpts for the packed-sim curve.

Everything else — anchors 0.55 / protective mix 0.45 (prose16k .20 /
code8k .10 / gsm8k .15 of steps), S=6, own-slice feedback, base-only kv
conditioning, canary-CE early-stop + best-ckpt selection — is v2 verbatim.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DraftBlock
from train import load_frozen, MultiDir, load_init_sd
import train as T
import train_stageb as SB
from train_stageb2 import chain_anchor, eval_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma dirs:probs (corpus mix)")
    ap.add_argument("--anchors", required=True, help="anchors_train.pt")
    ap.add_argument("--canary", required=True, help="anchors_canary.pt")
    ap.add_argument("--anchor-weight", type=float, default=0.55)
    ap.add_argument("--init-dir", required=True, help="first-party bf16 npy dir (weights_v2)")
    ap.add_argument("--weights", default="/root/w", help="frozen emb+head dir")
    ap.add_argument("--out", default="/root/runs/stageb3")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--corpus-batch", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=16384)
    ap.add_argument("--anchors-per-step", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--curve-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device("cuda")
    sd0 = load_init_sd(a.init_dir)                     # HF keys, zero-centered norms
    mod = DraftBlock.from_hf(sd0, torch.float32).to(dev)
    print(f"[v3] init from FIRST-PARTY bf16 {a.init_dir} ({len(sd0)} tensors)", flush=True)
    head_emb = load_frozen(a.weights, dev)
    head_emb.emb.requires_grad_(False)
    head_emb.head.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=a.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    md = MultiDir(a.data)
    for ds in md.dirs:
        ds.labels = np.load(f"{ds.dir}/labels.npy", mmap_mode="r") if os.path.exists(f"{ds.dir}/labels.npy") else None
        print(f"[v3] dir {ds.dir}: {ds.N} rows {ds.meta.get('tokens',0)/1e6:.2f}M tok", flush=True)

    D = torch.load(a.anchors, map_location="cpu", weights_only=False)
    D["stab"] = D["stab"].to(dev)
    D["h_seeds"] = D["h_seeds"].to(dev)
    for s in D["sess"]:
        # 24GB law (4090 rental): session K/V stay on PINNED CPU; chain_anchor
        # streams the [4,:pj,:] slice per cycle row (identical math — fp16
        # source, fp32 cast at the Kfp assignment).
        s["K"] = s["K"].pin_memory()
        s["V"] = s["V"].pin_memory()
    print(f"[v3] anchors: {len(D['cycles'])} cycles / {len(D['sess'])} sessions (kv on pinned CPU) "
          f"({sum(s['K'].numel()+s['V'].numel() for s in D['sess'])*2/1e9:.1f}GB)", flush=True)
    C = torch.load(a.canary, map_location="cpu", weights_only=False)
    C["stab"] = D["stab"]
    C["h_seeds"] = C["h_seeds"].to(dev)
    for s in C["sess"]:
        s["K"] = s["K"].pin_memory()
        s["V"] = s["V"].pin_memory()
    print(f"[v3] canary: {len(C['cycles'])} cycles / {len(C['sess'])} sessions", flush=True)

    rng = np.random.default_rng(a.seed)
    can_idx = rng.choice(len(C["cycles"]), min(a.eval_n, len(C["cycles"])), replace=False).tolist()
    trn_idx = rng.choice(len(D["cycles"]), min(256, len(D["cycles"])), replace=False).tolist()

    cl, ca = eval_anchor(mod, head_emb, C, can_idx, a.S)
    print(f"[v3] canary BASELINE: loss {cl:.4f} a1 {ca[0]:.3f} a {[round(x,2) for x in ca]}", flush=True)

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
            ids, h, offs, hpre = T.make_batch(ds, pick, Lmax, "cpu")
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
            print(f"[v3] step {step:4d}/{total} lr {lr_at(step):.2e} loss {loss.item()/a.S:.4f} "
                  f"acc1 {float(accs[0])/n_sup:.3f} gn {float(gn):.2f} "
                  f"{time.time()-t0:.0f}s ({tag})", flush=True)
        if (step + 1) % a.eval_every == 0 or step == total - 1:
            cl, ca = eval_anchor(mod, head_emb, C, can_idx, a.S)
            tl, ta_ = eval_anchor(mod, head_emb, D, trn_idx, a.S)
            hist.append(dict(step=step + 1, canary_loss=round(cl, 5),
                             canary_a=[round(x, 4) for x in ca],
                             train_a=[round(x, 4) for x in ta_], train_loss=round(tl, 5)))
            json.dump(hist, open(f"{a.out}/canary_hist.json", "w"), indent=1)
            print(f"[v3-eval] step {step+1}: canary loss {cl:.4f} a1 {ca[0]:.3f} "
                  f"a {[round(x,2) for x in ca]} | TRAIN a1 {ta_[0]:.3f} "
                  f"(gap {ta_[0]-ca[0]:+.3f})", flush=True)
            if best["step"] < 0 or cl < best["canary_loss"]:
                best = dict(step=step + 1, canary_loss=cl)
                sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
                torch.save({"sd": sd2, "step": step, "tokens": 0,
                            "src": f"bf16-init:{a.init_dir}"}, f"{a.out}/best.pt")
                print(f"[v3-ckpt] new best canary loss {cl:.4f} -> {a.out}/best.pt", flush=True)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    print(f"[v3] EARLY STOP: canary flat {no_improve} evals "
                          f"(best step {best['step']})", flush=True)
                    stop = True
        # THE CURVE: periodic ckpts (packed + sim-scored downstream)
        if (step + 1) % a.curve_every == 0:
            sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
            torch.save({"sd": sd2, "step": step, "tokens": 0,
                        "src": f"bf16-init:{a.init_dir}"}, f"{a.out}/ckpt_{step+1}.pt")
        if stop:
            break
    sd2 = {k: v.detach().cpu().half() for k, v in mod.export_hf().items()}
    torch.save({"sd": sd2, "step": step, "tokens": 0, "src": f"bf16-init:{a.init_dir}"},
               f"{a.out}/last.pt")
    print(f"[v3] DONE best canary loss {best['canary_loss']:.4f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
