# TLX DRAFTER Phase 2 (anchor-scale Stage B) — anchor dataset builder (rental).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Converts the anchor_scale dump sessions into the v2 anchored training sets:

  <out>/anchors_train.pt   {cycles, sess, h_seeds, stab}
  <out>/anchors_canary.pt  same shape (the held-out generalization canary —
                           sessions disjoint from train; NEVER trained on)

Per session: kvd/scd int8 -> dequant K/V fp16 [4, base+16, 256] (trimmed;
fp16 holds the int8-grid x fp16-scale values to ~2^-12 rel — far below the
int8 noise itself), keep-filter = prose & ~deep & ~t1 (build_r8 semantics).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def build_split(anchor_dirs, split_names, out_pt):
    sess_list = []
    cycles = []
    hseeds_all = []
    n_rows = 0                      # CUMULATIVE h_seed rows (the h_idx offset)
    stab = None
    for d in anchor_dirs:
        man = json.load(open(f"{d}/manifest.json"))
        f = man["f"]
        cyc = json.load(open(f"{d}/cycles.json"))
        hs = np.load(f"{d}/cycles_h_seed.npy")
        keep = [c for c in cyc if c.get("prose") and not c.get("deep_at_entry")
                and not c.get("t1_at_entry")]
        base = max(int(c["pos"]) for c in keep) if keep else 0
        q = np.load(f"{d}/kvd_decode_start.npy", mmap_mode="r")
        sc = np.load(f"{d}/scd_decode_start.npy", mmap_mode="r")
        upto = min(q.shape[2], base + 16)
        Ks, Vs = [], []
        for half in (0, 1):
            q8 = (np.asarray(q[half, :, :upto, :]).astype(np.int16) - 128).astype(np.float32)
            s = np.repeat(np.asarray(sc[half, :, :upto, :]).astype(np.float32), 32, axis=-1)
            (Ks if half == 0 else Vs).append((q8 * s))
        K = torch.from_numpy(Ks[0]).to(torch.float16)
        V = torch.from_numpy(Vs[0]).to(torch.float16)
        si = len(sess_list)
        sess_list.append(dict(f=f, split=man.get("split", "?"), K=K, V=V, base_len=int(upto),
                              n_keep=len(keep)))
        for c in keep:
            c2 = dict(c)
            c2["sess"] = si
            c2["h_idx"] = n_rows + int(c["h_idx"])
            cycles.append(c2)
        hseeds_all.append(hs)
        n_rows += len(hs)
        st = np.load(f"{d}/stab.npy")
        if stab is None:
            stab = st
        else:
            assert (stab == st).all(), f"slice drift in {d}"
        print(f"[v2] {f}: {len(keep)} kept cycles, base {upto}, pos "
              f"{min(int(c['pos']) for c in keep)}..{max(int(c['pos']) for c in keep)}", flush=True)
    h_seeds = torch.from_numpy(np.concatenate(hseeds_all, 0).astype(np.float32))
    torch.save(dict(cycles=cycles, sess=sess_list, h_seeds=h_seeds,
                    stab=torch.from_numpy(stab.astype(np.int64))), out_pt)
    gb = sum(s["K"].numel() + s["V"].numel() for s in sess_list) * 2 / 1e9
    print(f"[v2] {out_pt}: {len(cycles)} cycles / {len(sess_list)} sessions, "
          f"K+V {gb:.1f}GB fp16", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="/root/anchors")
    ap.add_argument("--out", default="/root/data/sb2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(glob.glob(f"{a.anchors}/bk_*"))
    assert dirs, f"no session dirs under {a.anchors}"
    man_by_f = {}
    for d in dirs:
        man_by_f[d] = json.load(open(f"{d}/manifest.json"))
    train = [d for d in dirs if man_by_f[d].get("split") == "train"]
    canary = [d for d in dirs if man_by_f[d].get("split") == "canary"]
    print(f"[v2] {len(train)} train / {len(canary)} canary sessions", flush=True)
    build_split(train, None, f"{a.out}/anchors_train.pt")
    build_split(canary, None, f"{a.out}/anchors_canary.pt")


if __name__ == "__main__":
    main()
