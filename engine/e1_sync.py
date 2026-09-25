# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-a E1: dext sync-cost measurement. Trivial kernel; (a) launch+wait loops,
(b) N-pipelined launches + one wait, (c) back-to-back waits. E4 harness pattern."""
import os, sys, time, subprocess
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
CU, CB = f"{BASE}/e1_sync.cu", f"{BASE}/e1_sync.cubin"
dev = Device["NV"]
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
r = subprocess.run(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={CB}", CU],
                   capture_output=True, text=True, env=env)
if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
print("[compiled]", flush=True)

_pi = ("v", 0, dtypes.int32, ())
lib = open(CB, "rb").read()
k_triv = NVProgram(dev, TinyELF(lib=lib, name="k_triv", target=dev.renderer.target,
                                signature=tuple(_pi for _ in range(2))))
_KEEP = []
t = Tensor.zeros(1024, dtype=dtypes.float32).contiguous().realize(); _KEEP.append(t)
buf = t.uop.buf_uop.buffer._bufs["NV"]
L = lambda w=False: k_triv(buf, buf, global_size=(4,1,1), local_size=(256,1,1), vals=(1024, 1024), wait=w)

# warmup
for _ in range(20): L()
L(True)

# (a) launch + immediate wait, R times
R = 300
t0 = time.perf_counter()
for _ in range(R): L(True)
dt_a = (time.perf_counter() - t0) / R
print(f"[E1a] launch+wait per-op: {dt_a*1e6:.1f} us", flush=True)

# (b) pipelined N launches + one wait
for N in (8, 32, 128, 512):
    reps = max(3, 2048 // N)
    t0 = time.perf_counter()
    for _ in range(reps):
        for _ in range(N): L()
        L(True)
    tot = (time.perf_counter() - t0) / reps
    print(f"[E1b] N={N:4d} pipelined: total {tot*1e3:.3f} ms  per-launch {tot/(N+1)*1e6:.1f} us", flush=True)

# (c) back-to-back waits after one launch
L(True)
t0 = time.perf_counter()
for _ in range(R): L(True)   # wait on completed timeline = cheap path?
dt_c = (time.perf_counter() - t0) / R
print(f"[E1c] b2b launch+wait (completed): {dt_c*1e6:.1f} us", flush=True)
L()
t0 = time.perf_counter()
n_w = 0
try:
    for _ in range(R): L(True)
except Exception as e:
    print("[E1c] exc", e)
dt_c2 = (time.perf_counter() - t0) / R
print(f"[E1c2] b2b after pipelined: {dt_c2*1e6:.1f} us", flush=True)
print("[done]", flush=True)
