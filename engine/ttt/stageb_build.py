# TLX DRAFTER Phase 1 (Stage B) — engine-trace -> training shards (runs on rental).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Converts the Phase-0 class-B engine traces into trainer shard dirs with
GREEDY targets (labels[p] = argmax(head(norm(h_p))) — the chain_sim target
semantics, computed with the frozen bf16 lm_head) and builds the r8
kv-anchored adaptation set (cycles + h_seeds + dequantized kv base + the real
40960-slice stab for chain feedback).

Inputs (upload from the Mac: ~/drafter/phase0/traces/):
  traces/corp/{prose,code}/  dense class-B: ids/hiddens/span_pos
  traces/gsm8k/t*/           dense per-transcript
  traces/r8_prose/           cycles.json cycles_h_seed kvd/scd stab + globals

Outputs:
  sb/{prose16k,code8k,gsm8k}/   trainer shards (ids/hids/lens/offs/hpre/labels)
  sb/r8_anchored.pt             {cycles, kvK, kvV, stab, h_seeds, base_pos}
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DIM, FrozenHeadEmb


def conv_dense(trace_dir, out_dir, fnw, head):
    ids = np.load(f"{trace_dir}/ids.npy")
    hid = np.load(f"{trace_dir}/hiddens.npy")
    T, D = hid.shape
    assert D == DIM
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/ids.npy", ids.astype(np.int32)[None, :])       # [1, T]
    np.save(f"{out_dir}/hids.npy", hid.astype(np.float16)[None, :, :])  # [1, T, D]
    np.save(f"{out_dir}/lens.npy", np.array([T], np.int32))
    np.save(f"{out_dir}/offs.npy", np.array([0], np.int32))
    np.save(f"{out_dir}/hpre.npy", np.zeros((1, D), np.float16))
    # greedy targets: labels[p] = argmax(head(rms(h_p * fnw)))  (chain_sim truth)
    H = torch.from_numpy(hid.astype(np.float32)).cuda()
    W = torch.from_numpy(fnw).cuda()
    hd = torch.from_numpy(head).cuda().to(torch.bfloat16)
    labels = np.zeros(T, np.int64)
    B = 2048
    with torch.no_grad():
        for i in range(0, T, B):
            x = H[i:i + B]
            r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            xh = (x * r * W).to(torch.bfloat16)
            lg = torch.nn.functional.linear(xh, hd)
            labels[i:i + B] = lg.argmax(-1).cpu().numpy()
    np.save(f"{out_dir}/labels.npy", labels.astype(np.int32)[None, :])
    # split the single long row into lmax windows? keep whole — trainer handles
    json.dump({"lmax": int(T), "hidden": int(D), "n": 1, "tokens": int(T),
               "anchors": 2, "tail_anchor": 0, "batch_hint": 1, "labels": 1,
               "src": trace_dir}, open(f"{out_dir}/meta.json", "w"))
    acc = labels[:-1] == ids[1:]      # greedy-next vs corpus token (diagnostic)
    print(f"[sb] {trace_dir} -> {out_dir}: {T} tok, greedy==corpus {acc.mean():.3f}")


def build_r8(trace_dir, out_pt, emb_grid=None):
    """The r8 kv-anchored set: engine kv state + cycle anchors + real slice."""
    cycles = json.load(open(f"{trace_dir}/cycles.json"))
    hs = np.load(f"{trace_dir}/cycles_h_seed.npy")
    q = np.load(f"{trace_dir}/kvd_decode_start.npy", mmap_mode="r")
    sc = np.load(f"{trace_dir}/scd_decode_start.npy", mmap_mode="r")

    def dq(half):
        q8 = (q[half].astype(np.int16) - 128).astype(np.float32)
        s = np.repeat(sc[half].astype(np.float32), 32, axis=-1)
        return (q8 * s).astype(np.float16)
    K = dq(0)
    V = dq(1)
    stab = np.load(f"{trace_dir}/stab.npy")
    keep = [c for c in cycles if c.get("prose") and not c.get("deep_at_entry") and not c.get("t1_at_entry")]
    torch.save({"cycles": keep, "K": torch.from_numpy(K.astype(np.float32)),
                "V": torch.from_numpy(V.astype(np.float32)),
                "stab": torch.from_numpy(stab.astype(np.int64)),
                "h_seeds": torch.from_numpy(hs.astype(np.float32))}, out_pt)
    print(f"[sb] r8: {len(keep)} prose cycles, kv {K.shape}, stab {stab.shape} -> {out_pt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/root/traces")
    ap.add_argument("--out", default="/root/data/sb")
    ap.add_argument("--weights", default="/root/w")
    ap.add_argument("--skip-r8", action="store_true",
                    help="v3: skip the r8_anchored.pt build (stageb2 anchors replace it)")
    a = ap.parse_args()
    fnw = np.load(f"{a.traces}/r8_prose/final_norm_w.npy")
    head = np.load(f"{a.weights}/lm_head_weight.npy")          # bf16->fp16 saved
    conv_dense(f"{a.traces}/prose", f"{a.out}/prose16k", fnw, head)
    conv_dense(f"{a.traces}/code", f"{a.out}/code8k", fnw, head)
    # gsm8k transcripts: each t-dir is one sequence; merge into one shard dir
    tdirs = sorted(glob.glob(f"{a.traces}/gsm8k/t*"))
    tdirs = [d for d in tdirs if os.path.exists(f"{d}/ids.npy")]
    Lmax = max(len(np.load(f"{d}/ids.npy")) for d in tdirs)
    N = len(tdirs)
    os.makedirs(f"{a.out}/gsm8k", exist_ok=True)
    ids_mm = np.lib.format.open_memmap(f"{a.out}/gsm8k/ids.npy", mode="w+", dtype=np.int32, shape=(N, Lmax))
    h_mm = np.lib.format.open_memmap(f"{a.out}/gsm8k/hids.npy", mode="w+", dtype=np.float16, shape=(N, Lmax, DIM))
    lab_mm = np.lib.format.open_memmap(f"{a.out}/gsm8k/labels.npy", mode="w+", dtype=np.int32, shape=(N, Lmax))
    lens = np.zeros(N, np.int32)
    W = torch.from_numpy(fnw).cuda()
    hd = torch.from_numpy(head).cuda().to(torch.bfloat16)
    for i, d in enumerate(tdirs):
        ids = np.load(f"{d}/ids.npy")
        hid = np.load(f"{d}/hiddens.npy")
        T = min(len(ids), len(hid))          # some transcripts dump fewer hidden rows
        with torch.no_grad():
            x = torch.from_numpy(hid.astype(np.float32)).cuda()
            r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            lg = torch.nn.functional.linear((x * r * W).to(torch.bfloat16), hd)
            lab = lg.argmax(-1).cpu().numpy()
        ids_mm[i, :T] = ids[:T]
        h_mm[i, :T] = hid[:T].astype(np.float16)
        lab_mm[i, :T] = lab[:T]
        lens[i] = T
    ids_mm.flush(); h_mm.flush(); lab_mm.flush()
    np.save(f"{a.out}/gsm8k/lens.npy", lens)
    np.save(f"{a.out}/gsm8k/offs.npy", np.zeros(N, np.int32))
    np.save(f"{a.out}/gsm8k/hpre.npy", np.zeros((N, DIM), np.float16))
    json.dump({"lmax": int(Lmax), "hidden": DIM, "n": int(N), "tokens": int(lens.sum()),
               "anchors": 1, "tail_anchor": 0, "batch_hint": 4, "labels": 1,
               "src": "gsm8k-30"}, open(f"{a.out}/gsm8k/meta.json", "w"))
    print(f"[sb] gsm8k: {N} transcripts, {lens.sum()} tok")
    if a.skip_r8:
        print("[sb] --skip-r8: skipping build_r8 (v3 uses stageb2 anchors)")
    else:
        build_r8(f"{a.traces}/r8_prose", f"{a.out}/r8_anchored.pt")


if __name__ == "__main__":
    main()
