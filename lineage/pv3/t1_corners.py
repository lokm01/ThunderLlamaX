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
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
bKV, bG = up(KVn), up(G)
wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
k1 = mk(f"{D}/k1t1.cubin", "k1t1")
V = KVn.reshape(2, 4, L, 256)[1].astype(np.float32)
for (h, k) in ((0, 5), (1, 5000), (0, 100000), (23, 99900)):
    P = np.zeros(24 * PS, np.float32)
    P[h * PS + k] = 1.0
    bP = up(P)
    import tinygrad
    # clear ws between runs
    wst2 = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst2)
    k1(bP, bKV, wst2.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
    ws = wst2.numpy().reshape(S, 4, 6, 256)
    tot = np.abs(ws).sum()
    if tot < 1e-6:
        print(f"one-hot (h={h}, p={k}): ws == 0 (missed)")
        continue
    s_ix, g_ix, h6_ix, d_ix = np.unravel_index(np.abs(ws).argmax(), ws.shape)
    row = ws[s_ix, g_ix, h6_ix, :]
    rownz = (np.abs(row) > 1e-9).sum()
    # find which V row (in group g_ix) matches
    best, bestk = 1e9, -1
    for kk in range(0, SP, 37):   # coarse sweep first
        d = np.abs(row - V[g_ix, kk]).max()
        if d < best: best, bestk = d, kk
    for kk in range(max(0,bestk-40), min(SP, bestk+40)):
        d = np.abs(row - V[g_ix, kk]).max()
        if d < best: best, bestk = d, kk
    exp_s = k // C
    print(f"one-hot (h={h}, p={k}): lit slot (s={s_ix}, g={g_ix}, h6={h6_ix}) expected (s={exp_s}, g={h//6}, h6={h%6}); row nz={rownz}/256; closest V row k={bestk} dist={best:.4f}")
