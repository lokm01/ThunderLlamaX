# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
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
L = 100352; PS = 100349; S = 24; SP = L - 500
C = (L + S - 1)//S
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(3)
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
bKV = up(KVn)
k = mk(f"{D}/k1sub2.cubin", "k1sub2")
ok = True
for p_test in (0, 3, 7, 4182, 5000):
    P = np.zeros(24 * PS, np.float32)
    P[0 * PS + p_test] = 1.0
    bP = up(P)
    wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
    k(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
    ws = wst.numpy().reshape(S, 4, 6, 256)
    s_exp = p_test // C
    got = ws[s_exp, 0, 0, 0]
    want = 1.0 if (p_test % C) < 8 else 0.0
    print(f"p={p_test}: ws[{s_exp},0,0,0]={got} want={want}", "OK" if abs(got-want)<1e-6 else "MISMATCH")
    ok = ok and abs(got-want) < 1e-6
print("MORPH1:", "PASS" if ok else "FAIL")
