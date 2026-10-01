# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — chained-replay acceptance eval (rental GPU or Mac).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Serve-faithful offline acceptance for a checkpoint (or the CURRENT engine
pack as a dequantized baseline) on held-out shards:
  anchors spaced through each sequence; TTT chain S steps with slice-restricted
  argmax feedback (the serve draft-slice approximation).
Reports a1, conditional a_{i+1}|i, E[m]|K2, E[m]|K4, the m-distribution, and
the teacher-forced-vs-chained gap (the exposure-bias signal, G1 test #2).
E[m] convention (verified vs the program's numbers):
  E[m]|K = E[ sum_{i=1..K} prod_{j<=i} a_j ]  (a1=0.417,a2|1=.40,a3|2=.30,a4|3=0
  -> E[m]|k2 = 0.583, E[m]|k4 = 0.634)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DraftBlock, FrozenHeadEmb, rms_norm, NH, NKV, HD
from train import ShardDataset, load_frozen
from pack_trained import load_sd
import math


@torch.no_grad()
def eval_chain(mod, head_emb, ids, h, t, S, feedback="slice"):
    from train import chain_forward
    losses, accs, n = chain_forward(mod, head_emb, ids, h, t, S, feedback=feedback)
    return accs  # per-step sums over batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help=".pt ckpt | weights dir | 'pack:<dir>' | 'none' (HF init)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=os.path.expanduser("~/drafter/weights"),
                    help="dir with emb+head (+HF mtp for init baselines)")
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--n-seq", type=int, default=48)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--feedback", default="slice")
    ap.add_argument("--tag", default="ckpt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ds = ShardDataset(a.data)
    head_emb = load_frozen(a.weights, dev)
    if a.ckpt and a.ckpt.startswith("pack:"):
        from q4pack_lib import load_pack
        w = load_pack(a.ckpt[5:])
        mod = DraftBlock.from_pack(w, torch.float32).to(dev)   # pack weights are FUNCTIONAL-form
    def _torch_sd(sd):
        import torch as _t
        return {k: (_t.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in sd.items()}
    if a.ckpt and a.ckpt.startswith("pack:"):
        from q4pack_lib import load_pack
        w = load_pack(a.ckpt[5:])
        mod = DraftBlock.from_pack(w, torch.float32).to(dev)   # pack weights are FUNCTIONAL-form
    elif a.ckpt and a.ckpt != "none":
        mod = DraftBlock.from_hf(_torch_sd(load_sd(a.ckpt)), torch.float32).to(dev)
    else:
        mod = DraftBlock.from_hf(_torch_sd(load_sd(a.weights)), torch.float32).to(dev)  # HF init
    mod.eval()
    N = min(a.n_seq, ds.N)
    step_ok = np.zeros(a.S)           # unconditioned (informational)
    chain_counts = np.zeros(a.S + 1)  # c = # consecutive correct from step 0
    n_chain = 0
    for j in range(N):
        ids_np, h_np = ds.sample(j, 1600)
        ids = torch.from_numpy(ids_np)[None].to(dev)
        h = torch.from_numpy(h_np)[None].to(dev)
        L = ids.shape[1]
        if L < 128:
            continue
        ts = np.linspace(96, L - a.S - 2, a.anchors).astype(int)
        for t in ts:
            tt = torch.tensor([t], device=dev)
            accs = eval_chain(mod, head_emb, ids, h, tt, a.S, feedback=a.feedback)
            oks = [bool(x > 0) for x in accs]
            for i, ok in enumerate(oks):
                step_ok[i] += ok
            c = 0
            for ok in oks:
                if not ok:
                    break
                c += 1
            chain_counts[c] += 1
            n_chain += 1
    a_uncond = step_ok / max(1, n_chain)
    # conditional a_i: chains that reached step i with all previous correct = chains with c >= i
    a_cond = []
    for i in range(a.S):
        reach = chain_counts[i + 1:].sum() + 0  # chains with c > i ... careful:
        # chains with c >= i+1 are those whose steps 0..i were ALL correct
        reach = sum(chain_counts[c] for c in range(i + 1, a.S + 1))
        a_cond.append((step_ok[i] if i == 0 else 0) / max(1, n_chain) if i == 0 else None)
    # conditional from counts: P(step i correct | steps < i correct) = (# chains c>=i+1)/(# chains c>=i)
    a_cond = []
    for i in range(a.S):
        num = sum(chain_counts[c] for c in range(i + 1, a.S + 1))
        den = sum(chain_counts[c] for c in range(i, a.S + 1))
        a_cond.append(num / den if den else 0.0)
    Em = {}
    for K in (2, 4):
        e = 0.0
        for i in range(min(K, a.S)):
            p = 1.0
            for j in range(i + 1):
                p *= a_cond[j]
            e += p
        Em[K] = e
    res = {
        "tag": a.tag, "ckpt": a.ckpt, "feedback": a.feedback, "n_chains": int(n_chain),
        "a_uncond": [round(float(x), 4) for x in a_uncond],
        "a_cond": [round(float(x), 4) for x in a_cond],
        "E[m]|K2": round(Em[2], 4), "E[m]|K4": round(Em[4], 4) if a.S >= 4 else None,
        "m_dist_counts": chain_counts.astype(int).tolist(),
    }
    print(json.dumps(res, indent=1))
    os.makedirs("evals", exist_ok=True)
    with open(f"evals/{a.tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
