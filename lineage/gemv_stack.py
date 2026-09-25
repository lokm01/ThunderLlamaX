# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Does weight-stacking (gate+up -> one 358MB GEMV) beat two 178MB GEMVs?
Also: 4-way stack (qkv+gate+beta+alpha equivalent shapes)."""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv

ITERS = 30
print(f"[cfg] BEAM={getenv('BEAM',0)}", flush=True)

def run(tag, jit, x, raw):
    for _ in range(3): jit(x.clone())
    Device["NV"].synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): jit(x.clone())
    Device["NV"].synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    print(f"  {tag:22s} {dt*1e3:7.3f} ms  {raw/dt/1e9:7.1f} GB/s", flush=True)

K = 5120
x = Tensor.kaiming_uniform(1, K).half().contiguous().realize()

# --- FFN gate+up: two 178MB vs one 356MB ---
Wg = Tensor.kaiming_uniform(17408, K).half().contiguous().realize()
Wu = Tensor.kaiming_uniform(17408, K).half().contiguous().realize()
Wgu = Wg.cat(Wu, dim=0).contiguous().realize()  # (34816, K)

j2 = TinyJit(lambda xx: ((xx @ Wg.T).realize(), (xx @ Wu.T).realize()))
j2(x.clone()); j2(x.clone())
run("2x178MB (gate,up)", j2, x, 17408*K*2*2)

j1 = TinyJit(lambda xx: (xx @ Wgu.T).realize())
j1(x.clone()); j1(x.clone())
run("1x356MB (gate|up)", j1, x, 34816*K*2)
del Wg, Wu, Wgu

# --- GDN input group: qkv(10240) + gate(6144) + beta(48) + alpha(48) = 16480 ---
sizes = [10240, 6144, 48, 48]
Ws = [Tensor.kaiming_uniform(n, K).half().contiguous().realize() for n in sizes]
Wm = Ws[0].cat(*Ws[1:], dim=0).contiguous().realize()
js = TinyJit(lambda xx: tuple((xx @ w.T).realize() for w in Ws))
js(x.clone()); js(x.clone())
run("4x (qkv,gate,b,a)", js, x, sum(sizes)*K*2)
jm = TinyJit(lambda xx: (xx @ Wm.T).realize())
jm(x.clone()); jm(x.clone())
run("1x16480 merged", jm, x, 16480*K*2)
del Ws, Wm
print("DONE_STACK", flush=True)
