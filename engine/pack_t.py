# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2D L1: transpose-repack the W1C-aligned IQ3 packed layout to LANE-CONTIGUOUS
16B chunks (E4 winner pattern). Pure layout change — decoded values identical,
per-row k-order identical -> bit-identical GEMV outputs.

Old row layout (98B per 256-elem block, NB blocks):
  [qs: 64B/blk (32 lanes x u16)][sc: 32B/blk (8 groups x u32)][d: 2B/blk]
New T layout (per row, stride 98*NB8):
  qsT: [32 lanes][NB8] u16   (per lane, 8 blocks = 16B = one uint4)
  scT: [8 groups][NB8] u32   (per group, 4 blocks = 16B = one uint4)
  dT : [NB8] u16             (8 blocks = 16B broadcast uint4)
NB8 = ceil(NB/8)*8; padded tail blocks are zeroed and guarded in-kernel.
"""
import numpy as np, os, glob, sys
SRC = "~/tinygrad-metal/engine0/packed"
DST = "~/tinygrad-metal/engine0/packed_t"
os.makedirs(DST, exist_ok=True)

K_BY_PREFIX = {"gate": 5120, "fg": 5120, "fu": 5120, "fd": 17408, "out": 6144, "q": 5120, "k": 5120}
for f in sorted(glob.glob(f"{SRC}/*.npy")):
    name = os.path.basename(f)[:-4]
    prefix = "".join(c for c in name if not c.isdigit())
    K = K_BY_PREFIX[prefix]
    NB = K // 256
    NB8 = (NB + 7) // 8 * 8
    row = np.load(f)
    nrows = row.size // (98 * NB)
    if row.size != nrows * 98 * NB:
        if row.size % (212 * (K // 256)) == 0:
            print(f"[pack_t] {name}: Q6_K layout, skipped", flush=True); continue
        raise AssertionError((name, row.size, nrows, NB))
    row = row.reshape(nrows, NB, 98)
    qs = row[:, :, :64].reshape(nrows, NB, 32, 2)          # [r][b][lane][u16]
    sc = row[:, :, 64:96].reshape(nrows, NB, 8, 4)         # [r][b][group][u32]
    d  = row[:, :, 96:98].reshape(nrows, NB, 2)            # [r][b][u16]
    qs_t = np.zeros((nrows, 32, NB8, 2), dtype=np.uint8)
    qs_t[:, :, :NB] = qs.transpose(0, 2, 1, 3)
    sc_t = np.zeros((nrows, 8, NB8, 4), dtype=np.uint8)
    sc_t[:, :, :NB] = sc.transpose(0, 2, 1, 3)
    d_t = np.zeros((nrows, NB8, 2), dtype=np.uint8)
    d_t[:, :NB] = d
    out = np.concatenate([qs_t.reshape(nrows, -1), sc_t.reshape(nrows, -1), d_t.reshape(nrows, -1)], axis=1)
    assert out.shape[1] == 98 * NB8
    np.save(f"{DST}/{name}.npy", out)
    print(f"[pack_t] {name}: {nrows} rows NB={NB} NB8={NB8} stride={98*NB8}", flush=True)
print("[pack_t] DONE", flush=True)
