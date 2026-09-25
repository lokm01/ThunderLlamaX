# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
dev = Device["NV"]
D = "~/tinygrad-metal/pv3"
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(3)
bufs = []
for n in (17408, 17408, 5120, 1, 20480, 34119680, 1024, 11141120):
    dt = np.uint8 if n in (20480, 34119680, 11141120) else np.float32
    bufs.append(up((rng.standard_normal(n)*0.3).astype(dt)))
gs = mk(f"{D}/ffn_gu_stock.cubin", "r_544_8_4_4_4_20_2_2_4")
ds = mk(f"{D}/ffn_dn_stock.cubin", "r_544_8_8_4_20_4_2_4")
def bench(p, args, grid, thr, n=3, inner=32):
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=(0,))
        p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
wb = 34119680 + 11141120 + 20480
t_gu = bench(gs, tuple(bufs[0:8]), (544,1,1), (8,4,4))
t_dn = bench(ds, tuple(bufs[0:1] + bufs[2:3] + bufs[3:4] + bufs[4:5] + bufs[5:6] + bufs[6:7] + bufs[7:8]), (544,1,1), (8,4,4))
print(f"STOCK gate|up: {t_gu:.3f}ms ({wb/t_gu/1e6:.0f} GB/s)   down: {t_dn:.3f}ms ({wb/t_dn/1e6:.0f} GB/s)")
print(f"a3b was: gu 0.131 (347) dn 0.154 (294)")
