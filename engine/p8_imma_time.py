# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Bare-world IMMA timing (the engine-world standalone-launch fault class is
unrelated to the IMMA kernel — v6 isolated it). Back-to-back synced min-of-8
(the census methodology) + per-launch-sync variant. Reference = the R7b
census ffn number (1220.3us, same shape, same methodology)."""
import os, sys, time
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
rng = np.random.default_rng(7)
KD, ND, M, NCH = 5120, 17408, 64, 40
NB = KD >> 8
nib = rng.integers(0, 16, (ND, KD), dtype=np.uint8)
w4 = np.zeros((ND // 8, NCH, 32, 4), dtype=np.uint32)
for c in range(NCH):
    for qc in range(4):
        blk = nib[:, c*128 + qc*32 : c*128 + qc*32 + 32]
        v = np.zeros((ND, 4), dtype=np.uint32)
        for w in range(32):
            v[:, w >> 3] |= (blk[:, w].astype(np.uint32) << (4 * (w & 7)))
        w4[:, c, qc::4, :] = v.reshape(ND // 8, 8, 4)
swd = (np.abs(rng.standard_normal((ND, NB))) * 0.01 + 0.002).astype(np.float16)
xf = (rng.standard_normal((M, KD)) * 0.35).astype(np.float16)
xq = np.zeros((M, KD), dtype=np.int8); sx = np.zeros((M, NCH), dtype=np.float32)
rowsum = np.zeros((M, NCH), dtype=np.int32)
for c in range(NCH):
    seg = xf[:, c*128:(c+1)*128].astype(np.float32)
    s = np.maximum(np.abs(seg).max(axis=1), 1e-6) / 127.0
    sx[:, c] = s
    xq[:, c*128:(c+1)*128] = np.clip(np.rint(seg / s[:, None]), -127, 127).astype(np.int8)
    rowsum[:, c] = xq[:, c*128:(c+1)*128].sum(axis=1).astype(np.int32)
P.up("imma_w4", np.frombuffer(w4.tobytes(), dtype=np.uint8)); P.up("imma_swd", swd)
P.up("imma_xq", xq); P.up("imma_sx", sx); P.up("imma_rs", rowsum)
P.poison("imma_out", M*ND*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()
d = P.d
def L():
  pr(d["imma_w4"], d["imma_swd"], d["imma_xq"], d["imma_sx"], d["imma_rs"], d["imma_out"],
     global_size=(ND//64,1,1), local_size=(256,1,1))
# warm 3 (per-launch sync, proven-stable form)
for _ in range(3):
  L(); dev.synchronize()
# back-to-back min-of-8 (census methodology)
best = 1e9
for _ in range(8):
  t0 = time.perf_counter()
  for _k in range(6): L()
  dev.synchronize()
  best = min(best, time.perf_counter() - t0)
per_b2b = best / 6
# per-launch-sync min-of-8
ts = []
for _ in range(8):
  t0 = time.perf_counter(); L(); dev.synchronize(); ts.append(time.perf_counter() - t0)
per_sync = min(ts)
w_mb = (ND*KD*0.5 + ND*NB*2)/1e6
print(f"[time] IMMA back-to-back : {per_b2b*1e6:8.1f} us/launch | W {w_mb:.1f} MB -> {w_mb/1e3/per_b2b:6.1f} GB/s | pool48 {per_b2b*48*1e3:6.1f} ms/chunk")
print(f"[time] IMMA per-launch   : {per_sync*1e6:8.1f} us/launch (incl launch+sync floor)")
print(f"[ref ] HMMA ffn census   : 1220.3 us/launch | W 55.7 MB -> 45.6 GB/s | pool48 58.6 ms/chunk")
print(f"[verdict] speedup x{1220.3/(per_b2b*1e6):.3f} back-to-back (vs banked census)")
print("[time] DONE")
