# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W0/E4: standalone dext data-path bandwidth microbench.
K_STREAM (pure read) / K_GEMV_FP16 (warp-per-row GEMV) / K_GEMV_IQ4_MOCK (int4-g128 dequant GEMV).
Wall-clock timing (dext signal timestamps are us, not ns): pipeline inner launches,
one wait=True at the end. Buffers kept alive via _KEEP."""
import os, sys, time, subprocess
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF

BASE = "~/tinygrad-metal"
CU, CB = f"{BASE}/w0_e4_bw.cu", f"{BASE}/w0_e4_bw.cubin"
dev = Device["NV"]

env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://<colima-socket>")
r = subprocess.run(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={CB}", CU],
                   capture_output=True, text=True, env=env)
if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
print("[compiled]", flush=True)

_pi = ("v", 0, dtypes.int32, ())
lib = open(CB, "rb").read()
def mk(n, k):
    return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target,
                                  signature=tuple(_pi for _ in range(k))))
k_stream, k_gemv16, k_gemviq = mk("k_stream", 2), mk("k_gemv16", 3), mk("k_gemviq", 4)
k_gemviq2 = mk("k_gemviq2", 4)
print("[progs loaded]", flush=True)

_KEEP = []
def up(a):
    t = Tensor(a).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"], t
def zn(n):
    t = Tensor.zeros(n, dtype=dtypes.float32).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"], t

rng = np.random.default_rng(7)
def rand_fp16(n, scale=0.5):
    a = np.empty(n, np.float16)
    for i in range(0, n, 67108864):
        m = min(67108864, n - i)
        a[i:i+m] = (rng.standard_normal(m) * scale).astype(np.float16)
    return a

results = []
def bench(p, args, vals, grid, inner, reps, tag, bytes_per):
    for _ in range(3):
        p(*args, global_size=(grid,1,1), local_size=(256,1,1), vals=vals, wait=True)
    n_tot = inner * reps
    t0 = time.perf_counter()
    for _ in range(n_tot):
        p(*args, global_size=(grid,1,1), local_size=(256,1,1), vals=vals)
    p(*args, global_size=(grid,1,1), local_size=(256,1,1), vals=vals, wait=True)
    dt = (time.perf_counter() - t0) / (n_tot + 1)
    gbs = bytes_per / dt / 1e9
    print(f"[bw] {tag:46s} {dt*1e3:8.3f} ms/launch  {gbs:7.1f} GB/s", flush=True)
    results.append((tag, gbs))
    return gbs

# ---------------- validation ----------------
print("== validation ==", flush=True)
N1, K1 = 5120, 5120
W1 = rand_fp16(N1*K1, 0.1).reshape(N1, K1)
x1 = rand_fp16(K1, 0.1)
Wb, _ = up(W1); xb, _ = up(x1); yb, yt = zn(N1)
k_gemv16(Wb, xb, yb, global_size=(164,1,1), local_size=(256,1,1), vals=(N1, K1, 164*8), wait=True)
ref = W1.astype(np.float32) @ x1.astype(np.float32)
rel = np.abs(yt.numpy() - ref).max() / np.abs(ref).max()
print(f"[val] k_gemv16  relerr={rel:.2e} {'OK' if rel < 1e-3 else 'FAIL'}", flush=True)

wq1 = rng.integers(0, 256, (N1, K1//2)).astype(np.uint8)
sc1 = (rng.standard_normal((N1, K1//128)) * 0.05).astype(np.float16)
q1b, _ = up(wq1); s1b, _ = up(sc1)
k_gemviq(q1b, s1b, xb, yb, global_size=(164,1,1), local_size=(256,1,1), vals=(N1, K1, 164*8), wait=True)
e = np.arange(K1)
lo = (wq1.astype(np.int32) & 15) - 8
hi = (wq1.astype(np.int32) >> 4) - 8
scf = sc1.astype(np.float32)
deq = np.empty((N1, K1), np.float32)
deq[:, 0::2] = lo * scf[:, e[0::2] // 128]
deq[:, 1::2] = hi * scf[:, e[1::2] // 128]
ref2 = deq @ x1.astype(np.float32)
rel2 = np.abs(yt.numpy() - ref2).max() / np.abs(ref2).max()
print(f"[val] k_gemviq  relerr={rel2:.2e} {'OK' if rel2 < 1e-3 else 'FAIL'}", flush=True)
k_gemviq2(q1b, s1b, xb, yb, global_size=(164,1,1), local_size=(256,1,1), vals=(N1, K1, 164*8), wait=True)
rel2b = np.abs(yt.numpy() - ref2).max() / np.abs(ref2).max()
print(f"[val] k_gemviq2 relerr={rel2b:.2e} {'OK' if rel2b < 1e-3 else 'FAIL'}", flush=True)

sa = rand_fp16(8*1024*1024)
sb, _ = up(sa)
ob, ot = zn(656*256)
nv_s = sa.size // 8
k_stream(sb, ob, global_size=(164,1,1), local_size=(256,1,1), vals=(nv_s, 164*256), wait=True)
gs = float(ot.numpy()[:164*256].sum(dtype=np.float64))
rs = float(sa.astype(np.float64).sum())
rel3 = abs(gs - rs) / abs(rs)
print(f"[val] k_stream  relerr={rel3:.2e} {'OK' if rel3 < 1e-3 else 'FAIL'} (gpu_sum={gs:.3e} cpu_sum={rs:.3e})", flush=True)

# launch floor: 16KB buffer, pipelined
bench(k_stream, (sb, ob), (1024, 164*256), 164, 200, 3, "launch-floor (16KB stream)", 1024*16)

# ---------------- perf: pure stream ----------------
print("== perf: K_STREAM ==", flush=True)
big = rand_fp16(268435456)                      # 512 MB
bb, _ = up(big)
nv512 = big.size // 8
for ctas in (164, 328, 656):
    bench(k_stream, (bb, ob), (nv512, ctas*256), ctas, 16, 10, f"K_STREAM 512MB CTAs={ctas}x256thr", 512*1024*1024)
try:
    big2 = rand_fp16(536870912)                 # 1 GB
    b2, _ = up(big2)
    nv1g = big2.size // 8
    for ctas in (328, 656):
        bench(k_stream, (b2, ob), (nv1g, ctas*256), ctas, 8, 8, f"K_STREAM 1GB   CTAs={ctas}x256thr", 1024**3)
except Exception as ex:
    print(f"[warn] 1GB stream stage failed/skipped: {ex!r}", flush=True)

# ---------------- perf: fp16 GEMV ----------------
print("== perf: K_GEMV_FP16 (N=5120) ==", flush=True)
for K in (5120, 17408):
    Wk = rand_fp16(N1*K, 0.1).reshape(N1, K)
    xk = rand_fp16(K, 0.1)
    Wkb, _ = up(Wk); xkb, _ = up(xk)
    byt = N1 * K * 2
    inner = 48 if K == 5120 else 24
    for ctas in (164, 328, 656):
        bench(k_gemv16, (Wkb, xkb, yb), (N1, K, ctas*8), ctas, inner, 10,
              f"K_GEMV_FP16 K={K:5d} CTAs={ctas}", byt)

# ---------------- perf: int4-g128 mock GEMV ----------------
print("== perf: K_GEMV_IQ4_MOCK (N=5120) ==", flush=True)
for K in (5120, 17408):
    wqk = rng.integers(0, 256, (N1, K//2)).astype(np.uint8)
    sck = (rng.standard_normal((N1, K//128)) * 0.05).astype(np.float16)
    xk2 = rand_fp16(K, 0.1)
    qb, _ = up(wqk); scb, _ = up(sck); x2b, _ = up(xk2)
    byt = N1*(K//2) + N1*(K//128)*2
    inner = 96 if K == 5120 else 48
    for ctas in (164, 328, 656):
        bench(k_gemviq, (qb, scb, x2b, yb), (N1, K, ctas*8), ctas, inner, 10,
              f"K_GEMV_IQ4_MOCK K={K:5d} CTAs={ctas}", byt)
        bench(k_gemviq2, (qb, scb, x2b, yb), (N1, K, ctas*8), ctas, inner, 10,
              f"K_GEMV_IQ4_OPT  K={K:5d} CTAs={ctas}", byt)

print("\n== summary ==", flush=True)
for t, g in results: print(f"{t:46s} {g:7.1f} GB/s", flush=True)
print("[done]", flush=True)
