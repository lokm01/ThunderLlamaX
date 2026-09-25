# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""GDN scan kernel debug: T=1 first, then T=3."""
import os, sys, time
import numpy as np
sys.path.insert(0, "~/tinygrad-src")
os.environ.setdefault("DEV", "NV")

from tinygrad.tensor import Tensor
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVProgram

dev = Device["NV"]
with open("~/tinygrad-metal/a4/gdn_scan.cubin", "rb") as f:
    prog = NVProgram(dev, TinyELF(lib=f.read(), name="gdn_scan", target=dev.renderer.target, signature=tuple()))
prog.cbuf_0[0], prog.cbuf_0[1], prog.cbuf_0[2] = 256, 1, 1   # NTID: nvcc SASS reads from c[0][0..8]

H, V, K = 32, 128, 128

def to_dev(arr):
    t = Tensor(arr).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]

def from_dev(buf, shape):
    nbytes = int(np.prod(shape)) * 4
    mv = memoryview(bytearray(nbytes))
    dev.allocator._copyout(mv, buf)
    return np.frombuffer(mv, dtype=np.float32).reshape(shape).copy()

def test_T(T, rng):
    state_np = rng.normal(0, 0.1, (H, V, K)).astype(np.float32)
    alpha_np = rng.uniform(0.8, 1.0, (H, T)).astype(np.float32)
    beta_np = rng.uniform(0.0, 1.0, (H, T)).astype(np.float32)
    q_np = (rng.normal(0, 0.1, (H, T, K)).astype(np.float32) / np.sqrt(K))
    k_np = rng.normal(0, 0.1, (H, T, K)).astype(np.float32)
    v_np = rng.normal(0, 0.1, (H, T, V)).astype(np.float32)

    # reference
    st = state_np.copy()
    outs = np.zeros((H, T, V), dtype=np.float32)
    for t in range(T):
        s1 = st * alpha_np[:, t][:, None, None]
        kd = (s1 * k_np[:, t, :][:, None, :]).sum(axis=2)
        d = (v_np[:, t, :] - kd) * beta_np[:, t][:, None]
        st = s1 + d[:, :, None] * k_np[:, t, :][:, None, :]
        outs[:, t, :] = (st * q_np[:, t, :][:, None, :]).sum(axis=2)

    # kernel
    sb = to_dev(state_np)
    ab = to_dev(alpha_np)
    bb = to_dev(beta_np)
    qb = to_dev(q_np)
    kb = to_dev(k_np)
    vb = to_dev(v_np)
    ob = dev.allocator.alloc(H * T * V * 4, BufferSpec())
    dev.synchronize()

    prog(sb, ob, ab, bb, qb, kb, vb, global_size=(H,1,1), local_size=(256,1,1), vals=(T,), wait=True)
    dev.synchronize()

    outs_g = from_dev(ob, (H, T, V))
    state_g = from_dev(sb, (H, V, K))

    ro = np.abs(outs_g - outs).max() / max(np.abs(outs).max(), 1e-9)
    rs = np.abs(state_g - st).max() / max(np.abs(st).max(), 1e-9)
    print(f"T={T}: outs relerr={ro:.2e} state relerr={rs:.2e} {'PASS' if ro<1e-3 and rs<1e-3 else 'FAIL'}", flush=True)
    if ro >= 1e-3:
        # debug: compare single head, single V
        h, v = 0, 0
        print(f"  h={h} v={v}: ref_out={outs[h,0,v]:.6f} gpu_out={outs_g[h,0,v]:.6f}", flush=True)
        # manual step for T=1
        s1 = state_np[h,v,:] * alpha_np[h,0]
        kd_manual = (s1 * k_np[h,0,:]).sum()
        d_manual = (v_np[h,0,v] - kd_manual) * beta_np[h,0]
        st_manual = s1 + d_manual * k_np[h,0,:]
        out_manual = (st_manual * q_np[h,0,:]).sum()
        print(f"  manual: kd={kd_manual:.6f} d={d_manual:.6f} out={out_manual:.6f}", flush=True)
        print(f"  alpha[0,0]={alpha_np[0,0]:.4f} beta[0,0]={beta_np[0,0]:.4f}", flush=True)
    return ro < 1e-3 and rs < 1e-3

rng = np.random.default_rng(42)
test_T(1, rng)
test_T(3, rng)
