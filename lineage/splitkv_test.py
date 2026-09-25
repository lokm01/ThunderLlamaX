# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Split-KV K1/K2 validation vs tensor attention (random KV, sp=1000, T=3)."""
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
Hkv, Rep, T, KD, VD = 8, 2, 3, 128, 128
L = 2048; POS = 1; S = 1   # micro: warp0 sees only p=0
rng = np.random.default_rng(11)
q = rng.normal(0, .3, (Hkv*Rep*T, KD)).astype(np.float16)
kc = rng.normal(0, .3, (L, Hkv, KD)).astype(np.float16)
vc = rng.normal(0, .3, (L, Hkv, VD)).astype(np.float16)

# numpy reference (the exact batched math: fp32 softmax, causal per row)
def ref():
    outs = np.zeros((16, T, VD), np.float32)
    for h in range(16):
        hk = h // Rep
        for t in range(T):
            lim = POS + t          # attend rows 0..lim INCLUSIVE (self-attn)
            kk = kc[:lim+1, hk, :].astype(np.float32)
            vv = vc[:lim+1, hk, :].astype(np.float32)
            qq = q[h*T+t].astype(np.float32)
            sc = (kk @ qq) / (KD ** 0.5)
            sc -= sc.max()
            p = np.exp(sc); p /= p.sum()
            outs[h, t] = p @ vv
    return outs

R = ref()

# compile kernels via nvcc shim
from tinygrad.helpers import getenv
import subprocess
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://<colima-socket>")
for name in ("splitkv_k1", "splitkv_k2"):
    r = subprocess.run(["nvcc", "-arch=sm_86", "-cubin", f"--output-file=~/tinygrad-metal/splitkv/{name}.cubin",
                        f"~/tinygrad-metal/splitkv/{name}.cu"], capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
print("[compiled]", flush=True)

S = int(os.getenv("SKV_S", "4"))   # splits
slots = S * 4   # per-warp slots
def up(a):
    t = Tensor(a).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"], t   # HCQBuffer
q_b, q_t = up(q); k_b, k_t = up(kc); v_b, v_t = up(vc)
oacc = Tensor.zeros(slots, 16, T, VD, dtype=dtypes.float32).contiguous().realize()
lse = Tensor.full((slots, 16, T), float("-inf")).contiguous().realize()
out = Tensor.zeros(16, T, VD, dtype=dtypes.float16).contiguous().realize()
ob, lb, xb = oacc.uop.buf_uop.buffer._bufs["NV"], lse.uop.buf_uop.buffer._bufs["NV"], out.uop.buf_uop.buffer._bufs["NV"]

_pi = ("v", 0, dtypes.int32, ())
k1 = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/splitkv_k1.cubin","rb").read(),
      name="splitkv_k1", target=dev.renderer.target, signature=(_pi, _pi, _pi, _pi)))
k2 = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/splitkv_k2.cubin","rb").read(),
      name="splitkv_k2", target=dev.renderer.target, signature=(_pi,)))
print("[progs loaded]", flush=True)

n_chunk = (POS + T + S - 1) // S
n_chunk = (POS + T + S - 1) // S
k1(q_b, k_b, v_b, ob, lb, global_size=(Hkv*S,1,1), local_size=(128,1,1), vals=(POS, T, S, n_chunk), wait=True)
k2(ob, lb, xb, global_size=(48,1,1), local_size=(128,1,1), vals=(slots,), wait=True)
# numpy combine from the GPU partials — isolates K1 vs K2
_o = oacc.numpy(); _l = lse.numpy()
_mx = _l[:, 0, 0].max()
_tot = sum(2.0 ** (v - _mx) for v in _l[:, 0, 0])
_acc = sum((2.0 ** (v - _mx)) * _o[s, 0, 0] for s, v in enumerate(_l[:, 0, 0]))
print("[dbg2] numpy-combine row0:", (_acc / _tot)[:4], flush=True)
_o0 = oacc.numpy()[0,0,0,:8]
print("[dbg5] oacc[slot0,row0,:8] =", _o0, flush=True)
print("[dbg5] v[0,0,:8]            =", vc[0,0,:8].astype(np.float32), flush=True)
print("[dbg4] lse[0,0,0] (warp0 row0: p=0 only) =", lse.numpy()[0,0,0], flush=True)
_x0 = float(kc[0,0,:].astype(np.float32) @ q[0].astype(np.float32)) * 0.0883883461356163 * 1.4426950408889634
print("[dbg4] expected x0 =", _x0, flush=True)
got = out.numpy().astype(np.float32)
print("[lse-dbg]", lse.numpy()[0,0,0], lse.numpy().max(), flush=True)
print("[k2dbg] K2: lse0={:.4f} mx={:.4f} tot={:.4f} slots={}".format(*oacc.numpy().flat[92:96]), flush=True)
_o2 = oacc.numpy(); _l2 = lse.numpy()
lv = _l2[:, 0, 0]
print("[k2dbg] REF: mx={:.4f} tot={:.4f} | lse row0:".format(lv.max(), float((2.0**(lv-lv.max())).sum())), lv[:16], flush=True)
# per-row numpy-combine vs K2 output
_o = oacc.numpy(); _l = lse.numpy()
_nc = np.zeros_like(got)
for row in range(48):
    hh, tt = row // 3, row % 3
    lv = _l[:, hh, tt]
    mx = lv.max()
    sc = np.where(np.isfinite(lv), 2.0**(lv - mx), 0.0)
    _nc[hh, tt] = (sc[:, None] * _o[:, hh, tt]).sum(0) / sc.sum()
d = np.abs(got - _nc)
bad = np.argsort(-d.reshape(-1))[:6]
print("[dbg6] worst rows/dims:", [(int(b//128), int(b%128), round(float(d.reshape(-1)[b]),4)) for b in bad], flush=True)
print("[dbg6] K2 got flat[0:4] =", got.reshape(-1)[:4], " numpy-combine:", _nc.reshape(-1)[:4], flush=True)
# debug: single warp single position sanity + lse dump
lse_np = lse.numpy()
print("[dbg] lse[:, 0, 0] =", lse_np[:8, 0, 0], flush=True)
kk = kc[:1001, 0, :].astype(np.float32); qq = q[0].astype(np.float32)
sc = (kk @ qq) / (128 ** 0.5)
print("[dbg] ref lse(row0) =", sc.max(), " sum-exp=", np.exp(sc - sc.max()).sum(), flush=True)
rel = np.abs(got - R).max() / np.abs(R).max()
print(f"[splitkv] relerr={rel:.2e} {'OK' if rel < 2e-2 else 'FAIL'}", flush=True)
print("sample got[0,0,:4]=", got[0,0,:4], " ref=", R[0,0,:4], flush=True)
