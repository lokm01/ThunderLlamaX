# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""q6dbg: a_q6 kernel vs numpy GEMV from the (validated) Q6_K decode."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
import engine0
engine0.QUANT[21] = (256, 110)
from engine0 import Bufs, parse_gguf, read_raw, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

P = Bufs(); ds, infos = parse_gguf()
rng = np.random.default_rng(7)
x = (rng.standard_normal(5120) * 0.2).astype(np.float32)
nw = np.frombuffer(read_raw(infos["blk.3.attn_norm.weight"], ds), dtype="<f4")
P.up("x", x); P.up("nw", nw)
P.poison("xh", 5120*2, np.float16, 7.7)
P.poison("qrow", 12288*2, np.float16, 7.7)
W = P.up("wq", np.frombuffer(read_raw(infos["blk.3.attn_q.weight"], ds), dtype=np.uint8))
dev.synchronize()
pr = {}
for n in ["k0_norm", "a_q6"]:
    lib = open(f"~/tinygrad-metal/engine0/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
LS = (256,1,1)
pr["k0_norm"](P.d["x"], P.d["nw"], P.d["xh"], global_size=(1,1,1), local_size=LS)
pr["a_q6"](W, P.d["xh"], P.d["qrow"], global_size=(1536,1,1), local_size=LS, wait=True)
xh = P.down("xh", (5120,), np.float16).astype(np.float32)
qrow = P.down("qrow", (12288,), np.float16).astype(np.float32)

raw = read_raw(infos["blk.3.attn_q.weight"], ds)
raw = np.frombuffer(raw, np.uint8)
# xh reference (k0_norm math)
ss = float((x*x).mean())
xn = (x * (1.0/np.sqrt(ss + 1e-6))).astype(np.float16).astype(np.float32) * nw
print("xh relerr:", np.abs(xh - xn).max()/np.abs(xn).max())
# numpy GEMV with kernel half-mul semantics, vectorized decode of all rows
Wf = np.zeros((12288, 5120), np.float32)
e = np.arange(256); h = e>>7; i = e&127; c = i>>5; b = i&31; sidx = e>>4
for row in range(0, 12288, 1):
    pass  # too slow per-row python; decode 64 rows only
sel = list(range(8)) + [100, 1000, 6143, 6144, 12287]
for row in sel:
    base = row*4200
    B = raw[base:base+210]
    outs = []
    for blk in range(20):
        b0 = base + blk*210
        d = np.frombuffer(raw[b0+208:b0+210], "<f2")[0].astype(np.float32)
        Bc = raw[b0:b0+208]
        xl = np.where(i<64, Bc[h*64+i]&0xF, Bc[h*64+i-64]>>4).astype(np.int32)
        xh_ = ((Bc[128+h*32+b].astype(np.int32) >> (2*c)) & 3) << 4
        sc = np.frombuffer(raw[b0+192:b0+208], np.int8)[sidx].astype(np.float32)
        w = d * ((xl|xh_)-32).astype(np.float32) * sc
        xv = xn[blk*256:(blk+1)*256]
        outs.append(np.sum(np.float16(xv).astype(np.float32) * np.float16(w).astype(np.float32)))
    ref = np.float16(np.sum(outs))
    r = abs(float(qrow[row]) - float(ref)) / max(abs(float(ref)), 1e-9)
    print(f"row {row}: kernel={qrow[row]:.4f} numpy={float(ref):.4f} relerr={r:.2e}")
