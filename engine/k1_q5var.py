# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad import dtypes
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
import engine0
from engine0 import GDNBlockEngine
dev = Device["NV"]
e = GDNBlockEngine(0)
rng = np.random.default_rng(3)
e.set_inputs(x=(rng.standard_normal(5120)*0.2).astype(np.float32))
dev.synchronize()
# VA alignment introspection
for nm in ("xh", "w_qkv"):
    b = e.P.d[nm]
    va = getattr(b, "va", getattr(b, "va_addr", None))
    print(f"[va] {nm}: obj={type(b).__name__} va={va}", flush=True)
e.launch_one("k0_norm", wait=True)
lib = open("~/tinygrad-metal/engine0/k1_q5var.cubin","rb").read()
def mk(n): return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
P, W = e.P.d, e.w
LS = (256,1,1)
import time
which = sys.argv[1] if len(sys.argv) > 1 else "v1"
t0 = time.perf_counter()
mk(f"k1_q5_{which}")(W["qkv"], P["xh"], P["qkv_row"], global_size=(1280,1,1), local_size=LS, wait=True)
print(f"[dbg] {which} OK {(time.perf_counter()-t0)*1e3:.1f} ms", flush=True)
qkv = e.P.down("qkv_row", (10240,), np.float16).astype(np.float32)
print(f"[dbg] qkv finite={np.isfinite(qkv).all()} |qkv|max={np.abs(qkv).max():.2f} mean={qkv.mean():.3f}", flush=True)

# ---- deep diagnostics: which rows NaN, xh sanity, numpy row-0 reference ----
xh_d = e.P.down("xh", (5120,), np.float16).astype(np.float32)
print(f"[diag] xh finite={np.isfinite(xh_d).all()} |xh|max={np.abs(xh_d).max():.3f} xh[:4]={xh_d[:4]}", flush=True)
qkv = e.P.down("qkv_row", (10240,), np.float16).astype(np.float32)
nanrows = np.where(~np.isfinite(qkv))[0]
print(f"[diag] qkv nan count={nanrows.size}/10240 first={nanrows[:8]}", flush=True)
fin = np.isfinite(qkv)
if fin.any():
    print(f"[diag] finite |qkv|max={np.abs(qkv[fin]).max():.2f} sample={qkv[:6]}", flush=True)
# numpy reference row 0
import struct
ds0, infos = engine0.parse_gguf()
raw = engine0.read_raw(infos["blk.0.attn_qkv.weight"], ds0)
row0 = raw[0:3520]
def q5_dequant_row(rb):
    out = np.zeros(256, np.float32)
    b = rb  # first block
    d = np.frombuffer(b[0:2], dtype="<f2")[0].astype(np.float32)
    dm = np.frombuffer(b[2:4], dtype="<f2")[0].astype(np.float32)
    s = np.frombuffer(b[4:16], dtype=np.uint8)
    sc = np.zeros(8, np.float32); mn = np.zeros(8, np.float32)
    for i in range(4):
        sc[i] = s[i] & 63; mn[i] = s[4+i] & 63
        sc[4+i] = (s[8+i] & 0xF) | ((s[i] >> 6) << 4)
        mn[4+i] = (s[8+i] >> 4) | ((s[4+i] >> 6) << 4)
    qh = np.frombuffer(b[16:48], dtype=np.uint8)
    qs = np.frombuffer(b[48:176], dtype=np.uint8)
    for k in range(256):
        sub = k >> 5; r = k & 31
        qv = (qs[k>>1] >> 4 if (k&1) else (qs[k>>1] & 0xF))
        qv += ((qh[r] >> sub) & 1) << 4
        out[k] = d*sc[sub]*qv - dm*mn[sub]
    return out
w0 = q5_dequant_row(row0)
xh_np = xh_d.astype(np.float32)
ref0 = float((w0 * xh_np[:256]*0 + w0 * 0).sum())
# proper: row0 uses all 5120 = 20 blocks
wall = np.concatenate([q5_dequant_row(row0[i*176:(i+1)*176]) for i in range(20)])
ref0 = float((wall * xh_np).sum())
print(f"[diag] numpy row0 ref={ref0:.3f} gpu row0={qkv[0]:.3f}", flush=True)

import engine0 as _e0
xd = e.P.down("x", (5120,))
nwd = e.P.down("nw1", (5120,))
ab = e.P.down("w_alpha", (48*5120,))
print(f"[diag2] x finite={np.isfinite(xd).all()} |x|max={np.abs(xd).max():.3f} x[:3]={xd[:3]}", flush=True)
print(f"[diag2] nw1 finite={np.isfinite(nwd).all()} |nw1|max={np.abs(nwd).max():.3f} nw1[:3]={nwd[:3]}", flush=True)
print(f"[diag2] alpha_w finite={np.isfinite(ab).all()} |ab|max={np.abs(ab).max():.3f}", flush=True)
