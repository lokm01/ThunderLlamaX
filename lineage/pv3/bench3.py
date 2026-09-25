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
L = 100352

def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]

def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))

stock = mk(f"{D}/stock100352.cubin", "r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4")
qk    = mk(f"{D}/qk100352.cubin", "r_3_25088_4_4_3_4_2_16_4")

rng = np.random.default_rng(21)
P = (rng.standard_normal(72*L)*0.5).astype(np.float32)
sm = np.zeros(72, np.float32)
for i in range(72):
    s = P[i*L:(i+1)*L]; s -= s.max()
    P[i*L:(i+1)*L] = s
    sm[i] = np.exp2(s).sum()
mx = np.zeros(72, np.float32)
V = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
Q = (rng.standard_normal(18432)*0.3).astype(np.float32)
bP, bmx, bsm, bV, bG, bQ = up(P), up(mx), up(sm), up(V), up(G), up(Q)

def bench(p, args, grid, thr, vals, inners=(8, 32, 128), n=3):
    res = {}
    for inner in inners:
        for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
        t0 = time.perf_counter()
        for _ in range(n):
            for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=vals)
            p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
        res[inner] = (time.perf_counter()-t0)/n/inner*1e3
    t8, t32, t128 = res[8], res[32], res[128]
    # t(k) = k + sync/inner  ->  solve k from 32 vs 128 (largest separation, most stable)
    sync = (t32 - t128) * 128 * 32 / (128 - 32)
    k = t128 - sync / 128
    return res, sync, k

outT = Tensor.zeros(18432).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
r1, s1, k1 = bench(stock, (ob, bP, bmx, bsm, bV, bG), (16,24,3), (16,1,1), (0,))
print(f"[PV  stock] t8={r1[8]:.2f} t32={r1[32]:.2f} t128={r1[128]:.2f} sync={s1:.1f}ms -> TRUE={k1:.3f}ms", flush=True)
r2, s2, k2 = bench(qk, (bP, bQ, bV), (25088,3,1), (4,4,3), (L-1,))
print(f"[QK stock] t8={r2[8]:.2f} t32={r2[32]:.2f} t128={r2[128]:.2f} sync={s2:.1f}ms -> TRUE={k2:.3f}ms", flush=True)
print(f"[proj x16/probe] PV={k1*16:.1f}ms QK={k2*16:.1f}ms attn_total={(k1+k2)*16:.1f}ms", flush=True)
