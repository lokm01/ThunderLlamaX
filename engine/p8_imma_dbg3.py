# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Scale-path bisect: (1) all scales 1 + random xq/nib -> isolates the int
path+ring; (2) random sx only; (3) random sw only; (4) random rowsum effect."""
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
KD, ND, M, NCH = 5120, 17408, 64, 40
NB = KD >> 8
rng = np.random.default_rng(21)
nib = rng.integers(0, 16, (ND, KD), dtype=np.uint8)
w4 = np.zeros((ND // 8, NCH, 32, 4), dtype=np.uint32)
for c in range(NCH):
    for qc in range(4):
        blk = nib[:, c*128 + qc*32 : c*128 + qc*32 + 32]
        v = np.zeros((ND, 4), dtype=np.uint32)
        for w in range(32):
            v[:, w >> 3] |= (blk[:, w].astype(np.uint32) << (4 * (w & 7)))
        w4[:, c, qc::4, :] = v.reshape(ND // 8, 8, 4)
xf = (rng.standard_normal((M, KD)) * 0.35).astype(np.float16)
xq = np.zeros((M, KD), dtype=np.int8); sxr = np.zeros((M, NCH), dtype=np.float32)
for c in range(NCH):
    seg = xf[:, c*128:(c+1)*128].astype(np.float32)
    s = np.maximum(np.abs(seg).max(axis=1), 1e-6) / 127.0
    sxr[:, c] = s
    xq[:, c*128:(c+1)*128] = np.clip(np.rint(seg / s[:, None]), -127, 127).astype(np.int8)
rowsum = xq.reshape(M, NCH, 128).sum(axis=2).astype(np.int32)
swr = (np.abs(rng.standard_normal((ND, NB))) * 0.01 + 0.002).astype(np.float16)

def run(sx, swd, tag, ref):
    P.up("imma_w4", np.frombuffer(w4.tobytes(), dtype=np.uint8)); P.up("imma_swd", swd)
    P.up("imma_xq", xq); P.up("imma_sx", sx); P.up("imma_rs", rowsum)
    P.poison("imma_out", M*ND*4, np.float32, 7.7e31)
    dev.synchronize(); P._keep.clear()
    d = P.d
    pr(d["imma_w4"], d["imma_swd"], d["imma_xq"], d["imma_sx"], d["imma_rs"], d["imma_out"],
       global_size=(ND//64,1,1), local_size=(256,1,1), wait=True)
    got = P.down("imma_out", (M, ND), np.float32); P._keep.clear()
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9)
    print(f"[bis] {tag}: relerr {rel:.3e} absmax {np.abs(got-ref).max():.4f}", flush=True)
    return got

# int path (scales all 1): ref = xq @ (nib-8).T
ref1 = xq.astype(np.float64) @ (nib.astype(np.float64) - 8.0).T
run(np.ones((M, NCH), np.float32), np.ones((ND, NB), np.float16), "scales=1 random-data", ref1)
# chunk-by-chunk reference builder
def chunk_ref(sx=None, sw=None):
    r = np.zeros((M, ND))
    for c in range(NCH):
        A = xq[:, c*128:(c+1)*128].astype(np.float64)
        B = (nib[:, c*128:(c+1)*128].astype(np.float64) - 8.0)
        part = A @ B.T
        if sx is not None: part = part * sx[:, c][:, None]
        if sw is not None: part = part * sw[:, c >> 1].astype(np.float64)[None, :]
        r += part
    return r
ref1 = chunk_ref()
run(np.ones((M, NCH), np.float32), np.ones((ND, NB), np.float16), "scales=1 random-data", ref1)
ref2 = chunk_ref(sx=sxr)
run(sxr, np.ones((ND, NB), np.float16), "sx random, sw=1", ref2)
ref3 = chunk_ref(sw=swr)
run(np.ones((M, NCH), np.float32), swr, "sw random, sx=1", ref3)
print("[bis] DONE")
