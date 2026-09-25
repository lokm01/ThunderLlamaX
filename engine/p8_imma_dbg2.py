# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Identity-X probe: xq[m][k] = 127 iff k == m (m<64), sx=1, rowsum follows.
out[m][n] should = sw[n][m//128] * (127*nib[n][m] - 8*127) * 1.0 — a direct
readout of the W path; mismatches localize the broken term."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev, Bufs
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

P = Bufs()
IMMA = "p8_imma_ffn_m64_nw8k128"
lib = open(f"~/tinygrad-metal/engine0/{IMMA}.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name=IMMA, target=dev.renderer.target, signature=tuple()))
rng = np.random.default_rng(11)
KD, ND, M, NCH = 5120, 17408, 64, 40
NB = KD >> 8
nib = rng.integers(0, 16, (ND, KD), dtype=np.uint8)
out = np.zeros((ND // 8, NCH, 32, 4), dtype=np.uint32)
for c in range(NCH):
    for qc in range(4):
        blk = nib[:, c*128 + qc*32 : c*128 + qc*32 + 32]
        v = np.zeros((ND, 4), dtype=np.uint32)
        for w in range(32):
            v[:, w >> 3] |= (blk[:, w].astype(np.uint32) << (4 * (w & 7)))
        out[:, c, qc::4, :] = v.reshape(ND // 8, 8, 4)
swd = np.full((ND, NB), 0.01, dtype=np.float16)
xq = np.zeros((M, KD), dtype=np.int8)
for m in range(64):
    xq[m, m] = 127
sx = np.ones((M, NCH), dtype=np.float32)
rowsum = xq.reshape(M, NCH, 128).sum(axis=2).astype(np.int32)   # 127 per row (one chunk)
# expected: out[m][n] = sw[n][m//128] * (127*nib[n][m] - 8*127)
ref = np.zeros((M, ND))
for m in range(64):
    ref[m, :] = 0.01 * (127.0 * nib[:, m].astype(np.float64) - 8.0 * 127.0)
P.up("imma_w4", np.frombuffer(out.tobytes(), dtype=np.uint8)); P.up("imma_swd", swd)
P.up("imma_xq", xq); P.up("imma_sx", sx); P.up("imma_rs", rowsum)
P.poison("imma_out", M*ND*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()
d = P.d
pr(d["imma_w4"], d["imma_swd"], d["imma_xq"], d["imma_sx"], d["imma_rs"], d["imma_out"],
   global_size=(ND//64,1,1), local_size=(256,1,1), wait=True)
got = P.down("imma_out", (M, ND), np.float32)
P._keep.clear()
bad = 0; shown = 0
for m in range(64):
    colbad = np.nonzero(np.abs(got[m] - ref[m]) > 1e-3)[0]
    if len(colbad):
        bad += 1
        if shown < 6:
            n = int(colbad[0])
            print(f"[id] m={m} n={n}: got {got[m,n]:.4f} want {ref[m,n]:.4f} | k-in-chunk {m%128} nib {nib[n,m]}")
            # what wrong-k nib would explain got?
            want_val = (got[m,n] / 0.01 + 8*127) / 127.0
            print(f"[id]   implied nib {want_val:.3f} vs actual {nib[n,m]}")
            shown += 1
print(f"[id] rows-with-errors {bad}/64")
print("[id] DONE")
