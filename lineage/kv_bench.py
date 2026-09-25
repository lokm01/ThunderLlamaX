# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P0 KV GEMV microbench @ L=32768, fp32 (matches current cache_kv dtype).
(a) naive per-layer q[1,d] @ K[L,d].T   (16 sequential layer GEMVs, strided-KT read)
(b) chunk-batched along ctx: K reshaped (16,32,L/32,d), one batched op
(c) 16-layer stacked batched GEMV: [16,1,d] @ [16,d,L] contiguous
Run: DEV=NV ~/tg311/bin/python kv_bench.py
"""
import os, sys, time, json
os.environ.setdefault("JIT", "2")
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("KV_L", "32768"))
NL, DKV = 16, 1024          # 16 full-attn layers, kv dim = n_kv_heads*head_dim = 1024
N = int(os.getenv("ITERS", "30"))
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

K = Tensor.rand(NL, L, DKV, dtype="float32").contiguous().realize()     # cache-like layout
KT = K.transpose(-1, -2).contiguous().realize()                          # [16, d, L]
q = Tensor.rand(NL, 1, DKV, dtype="float32").contiguous().realize()
Device["NV"].synchronize()
bytes_read = NL * L * DKV * 4
print(f"L={L} NL={NL} d={DKV} fp32  K bytes={bytes_read/1e9:.2f}GB iters={N}", flush=True)

def timeit(fn, label):
    fn().realize(); Device["NV"].synchronize()  # warm/capture
    t0 = time.perf_counter()
    for _ in range(N): o = fn()
    o.realize(); Device["NV"].synchronize()
    dt = (time.perf_counter() - t0) / N
    print(f"{label:<38} {dt*1e3:8.3f} ms  {bytes_read/dt/1e9:7.1f} GB/s", flush=True)
    return {"ms": round(dt*1e3, 3), "gbps": round(bytes_read/dt/1e9, 1)}

res = {}

# (a) naive: sequential per-layer GEMV against strided KT view of cache layout
def naive():
    outs = [(K[l] @ q[l].transpose(0, 1)).realize() for l in range(NL)]  # (L,d)@(d,1)->(L,1)
    return outs[-1]
res["a_naive_perlayer"] = timeit(naive, "(a) naive per-layer (seq, 16 launches)")

# (b) chunk-batched along ctx: (16,32,L/32,d) x (16,1,1,d) -> partial dots per chunk
KC = K.reshape(NL, 32, L // 32, DKV)
def chunked():
    return ((KC * q.reshape(NL, 1, 1, DKV)).sum(-1)).realize()
res["b_chunk_ctx"] = timeit(chunked, "(b) ctx-chunked (32/batch, one op)")

# (c) stacked batched GEMV, contiguous [16,d,L]
def stacked():
    return (q @ KT).realize()   # (16,1,d)@(16,d,L) -> (16,1,L): all layers' scores in one op
res["c_stacked"] = timeit(stacked, "(c) 16-layer stacked [16,d,L]")
json.dump(res, open(os.path.expanduser("~/tinygrad-metal/kvbench.json"), "w"), indent=1)
print("DONE_KVBENCH")
