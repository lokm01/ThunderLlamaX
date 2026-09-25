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
# gate|up kernel: (float*17408 x2, float*5120, float*1, uchar*20480, uchar*34119680, float*1024, uchar*11141120)
gu = mk(f"{D}/ffn_gu.cubin", "r_544_8_4_4_4_20_2_2_4_a3b")
# down kernel: (float*17408, float*5120, float*1, uchar*20480, uchar*34119680, float*1024, uchar*11141120)
dn = mk(f"{D}/ffn_dn.cubin", "r_544_8_8_4_20_4_2_4_a3b")
bufs = []
for n in (17408, 17408, 5120, 1, 20480, 34119680, 1024, 11141120):
    dt = np.float16 if False else (np.uint8 if n in (20480, 34119680, 11141120) else np.float32)
    bufs.append(up((rng.standard_normal(n)*0.3).astype(dt)))
def bench(p, args, n=3, inner=32):
    for _ in range(2): p(*args, global_size=(4352,1,1), local_size=(128,1,1), vals=(0,), wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=(4352,1,1), local_size=(128,1,1), vals=(0,))
        p(*args, global_size=(4352,1,1), local_size=(128,1,1), vals=(0,), wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
# gate|up: args order (data0..data7): 8 bufs
t_gu = bench(gu, tuple(bufs[0:8]))
# down: 7 bufs (data0_17408, data1_5120, data2_1, data3_20480, data4_34119680, data5_1024, data6_11141120)
t_dn = bench(dn, tuple(bufs[0:1] + bufs[2:3] + bufs[3:4] + bufs[4:5] + bufs[5:6] + bufs[6:7] + bufs[7:8]))
wb = 34119680 + 11141120 + 20480
print(f"gate|up: {t_gu:.3f}ms  weights={wb/1e6:.0f}MB -> {wb/t_gu/1e6:.0f} GB/s")
print(f"down:    {t_dn:.3f}ms  weights={wb/1e6:.0f}MB -> {wb/t_dn/1e6:.0f} GB/s")
print(f"per-block pair = {t_gu+t_dn:.3f}ms; x48 GDN+16 attn? -> probe FFN pool if 64 blocks: {(t_gu+t_dn)*64:.1f}ms (T=1-batched shapes; probe renders same)")
