# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import parse_gguf, read_raw

# TRUE row 0 of eh_proj: standard Q4_0 dequant of raw gguf bytes (out-major rows)
ds, infos = parse_gguf()
raw = np.frombuffer(read_raw(infos["blk.64.nextn.eh_proj.weight"], ds), dtype=np.uint8)
row_bytes = 320*18
r0 = raw[:row_bytes].reshape(320, 18)
d_true = np.frombuffer(r0[:, :2].tobytes(), dtype="<f2").astype(np.float32)
qs_true = r0[:, 2:]
true = np.zeros(10240, np.float32)
for s in range(320):
    for j in range(16):
        true[s*32 + 2*j]     = ((qs_true[s, j] & 0xF) - 8) * d_true[s]
        true[s*32 + 2*j + 1] = ((qs_true[s, j] >> 4) - 8) * d_true[s]
weff = np.load("/tmp/weff.npy")   # device: weff[e] = weight the kernel pairs with x[e]
# match by value (rows are quantized; use nearest with tolerance)
perm = np.full(10240, -1, np.int64)
used = set()
order = np.argsort(-np.abs(true))
for e in range(10240):
    cand = np.where(np.abs(true - weff[e]) < 1e-6)[0]
    for c in cand:
        if c not in used:
            perm[e] = c; used.add(c); break
print("[p] matched", int((perm >= 0).sum()), "of 10240")
print("[p] perm[:32] ", perm[:32].tolist())
print("[p] perm[32:64]", perm[32:64].tolist())
# interpret: kernel element e reads packed position p = perm[e] (true element index)
# show as (sub, byte, nibble) pairs
def sbn(x):
    return (x >> 5, (x & 31) >> 1, x & 1)
kern = [sbn(e) for e in range(16)]
got = [sbn(perm[e]) if perm[e] >= 0 else None for e in range(16)]
for e in range(16):
    print("[p] e", e, "kernel reads packed(true) pos", got[e], "value", round(float(weff[e]),4), "true", round(float(true[perm[e]]),4) if perm[e]>=0 else None)
