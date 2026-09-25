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

stock  = mk(f"{D}/stock100352.cubin", "r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4")
pstk   = mk(f"{D}/pstransform.cubin", "pstransform")
kerns = {}
for Z in (1,2,4):
    kerns[f"A{Z}"] = (mk(f"{D}/pvA_Z{Z}.cubin", "pvA"), (4, Z, 1), (256//Z,1,1))
    kerns[f"B{Z}"] = (mk(f"{D}/pvB_Z{Z}.cubin", "pvB"), (4, Z, 3), (256//Z,1,1))
kerns["D"]  = (mk(f"{D}/pvD.cubin", "pvD"), (3,24,2), (128,1,1))

rng = np.random.default_rng(21)
P0 = (rng.standard_normal(72*L)*0.5).astype(np.float32)
for i in range(72):
    s = P0[i*L:(i+1)*L]; s -= s.max(); P0[i*L:(i+1)*L] = s
mx = np.zeros(72, np.float32)
sm = np.stack([np.exp2(P0[i*L:(i+1)*L]).sum() for i in range(72)]).astype(np.float32)
V = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bmx, bsm, bV, bG = up(mx), up(sm), up(V), up(G)
bP_raw = up(P0)                       # raw scores (for stock)
bP_tr  = up(np.exp2(P0).astype(np.float32))  # pre-transformed (for A/B/D)
part = Tensor.zeros(18432).contiguous().realize()
pb = part.uop.buf_uop.buffer._bufs["NV"]

def out_buf():
    o = Tensor.zeros(18432).contiguous().realize()
    return o, o.uop.buf_uop.buffer._bufs["NV"]

# reference: stock PV on raw P
o0, ob0 = out_buf()
stock(ob0, bP_raw, bmx, bsm, bV, bG, global_size=(16,24,3), local_size=(16,1,1), vals=(0,), wait=True)
ref = o0.numpy()

# transform kernel correctness: partial sums must be bit-identical to stock partsum pipeline
pstk(pb, bP_raw, bmx, global_size=(72,256,1), local_size=(49,1,1), vals=(0,), wait=True)
tr_part = part.numpy().reshape(72, 256).sum(axis=1)
rel_sm = np.abs(tr_part - sm).max() / np.abs(sm).max()
print(f"[transform] partial-sum-vs-numpy relerr={rel_sm:.2e} (want tiny); P now exp-scores", flush=True)

# variants on transformed P
for name, (k, grid, thr) in kerns.items():
    o, ob = out_buf()
    k(ob, bP_tr, bmx, bsm, bV, bG, global_size=grid, local_size=thr, vals=(0,), wait=True)
    got = o.numpy()
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
    # timing
    args = (ob, bP_tr, bmx, bsm, bV, bG)
    for _ in range(2): k(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    t0 = time.perf_counter()
    for _ in range(3):
        for _ in range(32): k(*args, global_size=grid, local_size=thr, vals=(0,))
        k(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    t = (time.perf_counter()-t0)/3/32*1e3
    print(f"[{name}] relerr={rel:.2e} t={t:.3f}ms  proj16={t*16:.1f}ms", flush=True)

# stock timing same cadence for fair compare
args = (ob0, bP_raw, bmx, bsm, bV, bG)
for _ in range(2): stock(*args, global_size=(16,24,3), local_size=(16,1,1), vals=(0,), wait=True)
t0 = time.perf_counter()
for _ in range(3):
    for _ in range(32): stock(*args, global_size=(16,24,3), local_size=(16,1,1), vals=(0,))
    stock(*args, global_size=(16,24,3), local_size=(16,1,1), vals=(0,), wait=True)
t = (time.perf_counter()-t0)/3/32*1e3
print(f"[stock] t={t:.3f}ms proj16={t*16:.1f}ms", flush=True)
# transform timing
pargs = (pb, bP_raw, bmx)
for _ in range(2): pstk(*pargs, global_size=(72,256,1), local_size=(49,1,1), vals=(0,), wait=True)
t0 = time.perf_counter()
for _ in range(3):
    for _ in range(32): pstk(*pargs, global_size=(72,256,1), local_size=(49,1,1), vals=(0,))
    pstk(*pargs, global_size=(72,256,1), local_size=(49,1,1), vals=(0,), wait=True)
t = (time.perf_counter()-t0)/3/32*1e3
print(f"[transform] t={t:.3f}ms proj16={t*16:.1f}ms", flush=True)
