# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2-MTP: repack the draft block (blk.64, all Q4_0) into the aligned two-region
layout q4v.cu expects, and gather the 40960-row Q5_K head slice.
Outputs engine0/draft_pack/*.npy (run once, offline — the offline-repack law)."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import parse_gguf, read_raw

OUT = "~/tinygrad-metal/engine0/draft_pack"
os.makedirs(OUT, exist_ok=True)

def pack_q4(infos, ds, name):
    t, dims, off = infos[name]
    assert t == 2, (name, t)   # Q4_0
    nin, nout = dims[0], dims[1]
    assert nin % 256 == 0
    raw = np.frombuffer(read_raw(infos[name], ds), dtype=np.uint8).reshape(nout, nin//32, 18)
    d = raw[:, :, :2].copy().reshape(nout, nin//16)
    qs = raw[:, :, 2:].copy()
    rowb = nin//2 + nin//16          # NGRP*128 + NGRP*16, NGRP = nin/256
    assert rowb == (nin//256)*144
    out = np.zeros((nout, rowb), np.uint8)
    out[:, :nin//2] = qs.reshape(nout, nin//2)
    out[:, nin//2:] = d
    return out

def main():
    ds, infos = parse_gguf()
    jobs = {
        "d_eh": "blk.64.nextn.eh_proj.weight",
        "d_q": "blk.64.attn_q.weight",
        "d_k": "blk.64.attn_k.weight",
        "d_v": "blk.64.attn_v.weight",
        "d_o": "blk.64.attn_output.weight",
        "d_fg": "blk.64.ffn_gate.weight",
        "d_fu": "blk.64.ffn_up.weight",
        "d_fd": "blk.64.ffn_down.weight",
    }
    for nm, gg in jobs.items():
        arr = pack_q4(infos, ds, gg)
        np.save(f"{OUT}/{nm}.npy", arr)
        print(f"[q4pack] {nm} {arr.shape} {arr.nbytes/1e6:.0f}MB", flush=True)
    # f32 draft norms
    for nm, gg in [("d_enw", "blk.64.nextn.enorm.weight"), ("d_hnw", "blk.64.nextn.hnorm.weight"),
                   ("d_shnw", "blk.64.nextn.shared_head_norm.weight"), ("d_nw1", "blk.64.attn_norm.weight"),
                   ("d_nw2", "blk.64.post_attention_norm.weight"), ("d_qnw", "blk.64.attn_q_norm.weight"),
                   ("d_knw", "blk.64.attn_k_norm.weight")]:
        a = np.frombuffer(read_raw(infos[gg], ds), dtype="<f4").copy()
        np.save(f"{OUT}/{nm}.npy", a)
        print(f"[q4pack] {nm} {a.shape}", flush=True)
    print("[q4pack done]", flush=True)

if __name__ == "__main__":
    main()
