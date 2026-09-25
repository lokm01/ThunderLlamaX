# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import time
from tinygrad.tensor import Tensor
from tinygrad import dtypes

pc=time.perf_counter

def bench(fn, warm=3, iters=10):
    r=None
    for _ in range(warm): r=fn().realize()
    float(r.float().sum().item())
    t0=pc()
    for _ in range(iters): r=fn().realize()
    float(r.float().sum().item())
    return (pc()-t0)/iters

# ---- safe peak probe: 4096x4096 fp16 GEMM (128MB weight read per pass) ----
print("== PEAK PROBE: [4,4096]@[4096,4096] fp16 ==", flush=True)
A=Tensor.kaiming_uniform(4,4096, dtype=dtypes.float16)
W=Tensor.kaiming_uniform(4096,4096, dtype=dtypes.float16)
ms=bench(lambda: A@W)
gb=(A.nbytes()+W.nbytes()+4*4096*4096*2)/1e9
print(f"  {ms*1e3:.2f} ms -> {gb/ms:.0f} GB/s", flush=True)

# GEMV variant of same matrix
v=Tensor.zeros(1,4096, dtype=dtypes.float16)
ms=bench(lambda: v@W.T)
gb=(v.nbytes()+W.nbytes()+4096*4096*2)/1e9
print(f"  GEMV: {ms*1e3:.2f} ms -> {gb/ms:.0f} GB/s", flush=True)

# ---- layer-shape family for qwen3.8-27B (dim=5120, inter=17408, heads...) ----
shapes=[
    ("qkv_proj",  5120, 10240),
    ("o_proj",    7168//2, 5120),
    ("gate_up",   5120, 34816),
    ("down_proj", 17408, 5120),
    ("lm_head",   5120, 248320),
]
for name,K,N in shapes:
    W=Tensor.kaiming_uniform(N,K, dtype=dtypes.float16)
    v=Tensor.zeros(1,K, dtype=dtypes.float16)
    ms=bench(lambda: v@W.T)
    gb=(v.nbytes()+W.nbytes()+N*K*2)/1e9
    print(f"== {name}: [1,{K}]@[{K},{N}] fp16 -> {ms*1e3:.2f} ms, {gb/ms:.0f} GB/s", flush=True)
print("DONE", flush=True)
