# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Standalone GDN scan mega-kernel correctness test vs Python reference."""
import os, sys, time
import numpy as np
sys.path.insert(0, "~/tinygrad-src")
os.environ.setdefault("DEV", "NV")

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad import dtypes
import struct

dev = Device["NV"]

# Load the cubin via NVProgram
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
with open("~/tinygrad-metal/a4/gdn_scan.cubin", "rb") as f:
    CUBIN = f.read()
prog = NVProgram(dev, TinyELF(lib=CUBIN, name="gdn_scan", target=dev.renderer.target, signature=tuple()))
print("[kernel loaded]", flush=True)

# Test parameters (match model: H=32, V=128, K=128)
H, V, K = 32, 128, 128
T = 3
B = 1

# Create test data
rng = np.random.default_rng(42)
state_np = rng.normal(0, 0.1, (H, V, K)).astype(np.float32)
alpha_np = rng.uniform(0.8, 1.0, (H, T)).astype(np.float32)  # per-head decay
beta_np = rng.uniform(0.0, 1.0, (H, T)).astype(np.float32)
q_np = rng.normal(0, 0.1, (H, T, K)).astype(np.float32) / np.sqrt(K)
k_np = rng.normal(0, 0.1, (H, T, K)).astype(np.float32)
v_np = rng.normal(0, 0.1, (H, T, V)).astype(np.float32)

# Python reference (exactly the model.py scan loop)
def scan_ref(state, alpha, beta, q, k, v, T, H, V, K):
    outs = np.zeros((H, T, V), dtype=np.float32)
    st = state.copy()
    for t in range(T):
        s1 = st * alpha[:, t][:, None, None]  # (H,V,K) * (H,1,1)
        kd = (s1 * k[:, t, :][:, None, :]).sum(axis=2)  # (H,V)
        d = (v[:, t, :] - kd) * beta[:, t][:, None]  # (H,V)
        st = s1 + d[:, :, None] * k[:, t, :][:, None, :]  # (H,V,K)
        outs[:, t, :] = (st * q[:, t, :][:, None, :]).sum(axis=2)  # (H,V)
    return st, outs

state_final_ref, outs_ref = scan_ref(state_np, alpha_np, beta_np, q_np, k_np, v_np, T, H, V, K)
print("[ref] state max:", np.abs(state_final_ref).max(), "outs max:", np.abs(outs_ref).max(), flush=True)

# GPU test
def to_dev(arr):
    t = Tensor(arr).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]

state_buf = to_dev(state_np)
alpha_buf = to_dev(alpha_np)
beta_buf = to_dev(beta_np)
q_buf = to_dev(q_np)
k_buf = to_dev(k_np)
v_buf = to_dev(v_np)
dev.synchronize()

# allocate outputs
from tinygrad.device import BufferSpec
outs_buf = dev.allocator.alloc(H * T * V * 4, BufferSpec())
dev.allocator._copyin(outs_buf, memoryview(bytearray(H*T*V*4)).cast("B"))
dev.synchronize()

# launch
t0 = time.perf_counter()
prog(state_buf, outs_buf, alpha_buf, beta_buf, q_buf, k_buf, v_buf,
     global_size=(H, 1, 1), local_size=(256, 1, 1), vals=(T,), wait=True)
dev.synchronize()
print(f"[kernel] {time.perf_counter()-t0:.6f}s", flush=True)

# read back
mv = memoryview(bytearray(H * T * V * 4))
dev.allocator._copyout(mv, outs_buf)
outs_gpu = np.frombuffer(mv, dtype=np.float32).reshape(H, T, V)

mv2 = memoryview(bytearray(H * V * K * 4))
dev.allocator._copyout(mv2, state_buf)
state_gpu = np.frombuffer(mv2, dtype=np.float32).reshape(H, V, K)

# compare
rel_out = np.abs(outs_gpu - outs_ref).max() / max(np.abs(outs_ref).max(), 1e-9)
rel_state = np.abs(state_gpu - state_final_ref).max() / max(np.abs(state_final_ref).max(), 1e-9)
print(f"[result] outs relerr={rel_out:.2e} state relerr={rel_state:.2e}", flush=True)
print(f"[verdict] {'PASS' if rel_out < 1e-3 and rel_state < 1e-3 else 'FAIL'}", flush=True)

# timing: 100 launches
dev.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    prog(state_buf, outs_buf, alpha_buf, beta_buf, q_buf, k_buf, v_buf,
         global_size=(H, 1, 1), local_size=(256, 1, 1), vals=(T,), wait=True)
dev.synchronize()
per = (time.perf_counter() - t0) / 100
print(f"[timing] {per*1e6:.1f}us per launch (x 48 blocks = {per*48*1e3:.2f}ms/pass)", flush=True)
