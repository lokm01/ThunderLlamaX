# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Dissect the decode GEMV wall: grid starvation vs dequant cost.

Variants at K=6144,N=5120 (ssm_out) and K=5120,N=34816 (gate_up):
  fp16-M1     plain fp16 GEMV (baseline ~89 GB/s)
  fp16-M16    batch-16 rows -> 16x grid -> tests grid starvation
  fp16-M64
  q40-M1      Q4_0 lazy-dequant fused GEMV (dequant chain: int8*scale, cheap)
  q40-M16
  iq3xxs-M1   IQ3_XXS lazy-dequant fused GEMV (the real model path, LUT gathers)
  iq3xxs-M16
GB/s counted on RAW weight bytes (what decode actually streams).
"""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.jit import TinyJit
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.helpers import getenv
import numpy as np

SHAPES = [("ssm_out", 6144, 5120), ("gate_up", 5120, 34816)]
ITERS = 30
DEV = "NV"

print(f"[cfg] BEAM={getenv('BEAM',0)}", flush=True)

def bench(j, x, raw_bytes, tag):
    for _ in range(3): j(x.clone())
    Device[DEV].synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): j(x.clone())
    Device[DEV].synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    g = raw_bytes / dt / 1e9
    print(f"  {tag:12s} {dt*1e3:7.3f} ms  {g:7.1f} GB/s(raw)", flush=True)
    return g

for name, K, N in SHAPES:
    print(f"== {name} K={K} N={N} ==", flush=True)
    # fp16
    W = Tensor.kaiming_uniform(N, K).half().contiguous().realize()
    raw = N * K * 2
    for M in (1, 16, 64):
        x = Tensor.kaiming_uniform(M, K).half().contiguous().realize()
        j = TinyJit(lambda xx: (xx @ W.T).realize())
        j(x.clone()); j(x.clone())
        bench(j, x, raw, f"fp16-M{M}")
    # Q4_0: 32 elems / 18 bytes
    n = N * K
    nblk = n // 32
    raw_bytes = np.random.randint(0, 255, size=nblk * 18, dtype=np.uint8)
    t8 = Tensor(raw_bytes).to(DEV)
    Wq = ggml_data_to_tensor(t8, n, 2).reshape(N, K)  # lazy dequant (N,K)? blocks fill row-major
    raw_q40 = nblk * 18
    for M in (1, 16):
        x = Tensor.kaiming_uniform(M, K).half().contiguous().realize()
        j = TinyJit(lambda xx: (xx @ Wq.cast(dtypes.float32).T).cast(dtypes.float32).realize())
        j(x.clone()); j(x.clone())
        bench(j, x, raw_q40, f"q40-M{M}")
    # IQ3_XXS: 256 elems / 98 bytes
    nblk = n // 256
    raw_bytes = np.random.randint(0, 255, size=nblk * 98, dtype=np.uint8)
    t8 = Tensor(raw_bytes).to(DEV)
    Wq = ggml_data_to_tensor(t8, n, 18).reshape(N, K)
    raw_iq3 = nblk * 98
    for M in (1, 16):
        x = Tensor.kaiming_uniform(M, K).half().contiguous().realize()
        j = TinyJit(lambda xx: (xx @ Wq.cast(dtypes.float32).T).cast(dtypes.float32).realize())
        j(x.clone()); j(x.clone())
        bench(j, x, raw_iq3, f"iq3xxs-M{M}")
print("DONE_DISSECT", flush=True)
