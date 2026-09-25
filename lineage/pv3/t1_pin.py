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
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(11)
Pn = (rng.random(24 * PS) * 0.001).astype(np.float32)
KV = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
bP, bKV, bG = up(Pn), up(KV), up(G)
wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
bws = wst.uop.buf_uop.buffer._bufs["NV"]
k1 = mk(f"{D}/k1t1.cubin", "k1t1")
k1(bP, bKV, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(S, 4, 6, 256)
# numpy partials per (s, g, h6)
C = (L + S - 1)//S
Vg_np = KV.reshape(2, 4, L, 256)   # [K/V][group][p][d]
ok_heads, bad_heads = [], []
for g in range(4):
    for h6 in range(6):
        h = g*6 + h6
        exp = np.zeros(256, np.float32)
        for s in range(S):
            a, b = s*C, min((s+1)*C, SP+1)
            exp += (Pn[h*PS + a : h*PS + b][:, None] * Vg_np[1, g, a:b].astype(np.float32)).sum(axis=0)
        got = ws[:, g, h6, :].sum(axis=0)
        rel = np.abs(got - exp).max() / max(np.abs(exp).max(), 1e-9)
        (ok_heads if rel < 1e-4 else bad_heads).append((h, g, h6, float(rel)))
print("heads OK:", len(ok_heads), "BAD:", len(bad_heads))
for x in bad_heads[:8]: print("  bad head h,g,h6,rel:", x)
# if ALL heads OK in ws => the bug is in k2t1 or the harness stock-eval; if bad, check permutation: does head h's ws match numpy of some other head h'?
if bad_heads:
    h0 = bad_heads[0][0]
    got = ws[:, h0//6, h0%6, :].sum(axis=0)
    for hp in range(24):
        exp = np.zeros(256, np.float32)
        for s in range(S):
            a, b = s*C, min((s+1)*C, SP+1)
            exp += (Pn[hp*PS + a : hp*PS + b][:, None] * Vg_np[1, hp//6, a:b].astype(np.float32)).sum(axis=0)
        if np.abs(got - exp).max() < 1e-5:
            print(f"  head {h0} ws matches numpy head {hp} -> permutation found")

# per-split detail for head 0
import numpy as _np
h = 0; g = 0; h6 = 0
for s in range(0, S, max(1, S//6)):
    a, b = s*C, min((s+1)*C, SP+1)
    exp = (Pn[h*PS + a : h*PS + b][:, None] * Vg_np[1, g, a:b].astype(_np.float32)).sum(axis=0)
    got = ws[s, g, h6, :]
    rel = _np.abs(got - exp).max() / max(_np.abs(exp).max(), 1e-9)
    print(f"split {s}: range[{a},{b}) rel={rel:.3f} got[:2]={got[:2]} exp[:2]={exp[:2]}")
