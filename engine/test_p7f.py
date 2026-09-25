# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-F probe harness: dp4a int8 QK dot on REAL engine KV (kv_3.npy,
CTXK=100352, the engine kv8 quantization verbatim). Validates the kernel vs
a host fp64 reference over the fp16-pipeline inputs (q PRE-quant fp16 x
dequant-K fp32 -> the error measured = q-int8-quant class + fp16 out), then
synced min-of-10 bench at T=16 and T=64. Reports K-read GB/s + dp4a TOPS.
Usage: ~/tg311/bin/python -u test_p7f.py
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
CTXK = 100352
NSEG = 8
P = Bufs()
_PI = ("v", 0, dtypes.int32, ())
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target,
                                signature=tuple(_PI for _ in range(6))))

print("[p7f] loading real KV (kv_3.npy K half, engine kv8 quant verbatim)...", flush=True)
a = np.load("~/snap100k/kv_3.npy", mmap_mode="r")   # (2,4,CTXK,256) fp16
k = np.asarray(a[0], dtype=np.float32).reshape(4, CTXK, 8, 32)  # K only
am = np.abs(k).max(axis=-1)                                    # (4,CTXK,8)
scq = (np.maximum(am, 1e-8) * (1.0 / 127.0)).astype(np.float16)
qq = (np.clip(np.rint(k / scq.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
del k, am
# K' = K-128 precompute (the kpre trick) as int32 quads: (4,CTXK,64)
kq8 = (qq.astype(np.int16) - 128).astype(np.int8).reshape(4, CTXK, 256)
kp = np.ascontiguousarray(kq8.view(np.int32).reshape(-1))
scf = np.ascontiguousarray(scq.astype(np.float32).reshape(-1))
kdq = (qq.astype(np.float32) - 128.0) * scq.astype(np.float32)[..., None]  # (4,CTXK,8,32) dequant
del qq, kq8
print(f"[p7f] quant done. kp {kp.nbytes/1e6:.1f}MB sc {scf.nbytes/1e6:.1f}MB", flush=True)

pr = prog("pfk_dp4a")

def run_T(T, nval):
  """Build Q from T real K rows (realistic magnitudes), run, validate nval pos, bench."""
  qpos = np.linspace(1000, 90000, T).astype(int)
  # q per (t, 1024ch) = dequant K rows; head h = ch [h*128,(h+1)*128)
  qrows = kdq.reshape(4, CTXK, 256)[:, qpos]                     # (4,T,256)
  qfull = np.ascontiguousarray(np.transpose(qrows, (1, 0, 2)).reshape(T, 1024))
  qh = qfull.reshape(T, 8, 128)                                  # (t,head,128)
  # quantize q per-(t,head) 128ch: absmax/127 -> s8 quads
  qam = np.abs(qh).max(axis=-1)
  qs = np.maximum(qam, 1e-8) * (1.0 / 127.0)
  qi8 = np.clip(np.rint(qh / qs[..., None]), -127, 127).astype(np.int8)
  qp = np.ascontiguousarray(qi8.reshape(T * 8, 128).view(np.int32).reshape(-1))
  sqp = np.ascontiguousarray(qs.reshape(-1).astype(np.float32))
  P.up("kp", kp); P.up("scf", scf); P.up("qp", qp); P.up("sqp", sqp)
  P.up("npb", np.array([T * 8], dtype=np.int32))
  P.poison("out", T * 8 * CTXK * 2, np.float16, 7.7)
  dev.synchronize()
  npairs = T * 8
  grid = (NSEG * npairs) // 8
  args = (P.d["kp"], P.d["scf"], P.d["qp"], P.d["sqp"], P.d["out"], P.d["npb"])
  pr(*args, global_size=(grid, 1, 1), local_size=(256, 1, 1))
  dev.synchronize()
  # validate: host fp64 over q fp16-pipeline inputs (q pre-quant fp16, K dequant fp32)
  got = P.down("out", (T * 8, CTXK), np.float16)[:, :nval].astype(np.float64)
  kseg = np.ascontiguousarray(np.transpose(kdq[:, :nval].reshape(4, nval, 256), (1, 0, 2)))  # (pos,g,ch)
  rels = []
  for t in range(T):
    for h in range(8):
      g, hb = h >> 1, (h & 1) * 128
      kk = kseg[:, g, hb:hb + 128].astype(np.float64)            # (nval,128)
      ref = kk @ qh[t, h].astype(np.float64)
      gl = got[t * 8 + h]
      rels.append(np.abs(gl - ref) / np.maximum(np.abs(ref), 1e-6))
  rels = np.concatenate(rels)
  med, f95 = float(np.median(rels)), float(np.quantile(rels, 0.95))
  # bench min-of-10
  best = 1e9
  for _ in range(10):
    t0 = time.perf_counter()
    pr(*args, global_size=(grid, 1, 1), local_size=(256, 1, 1))
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  kbytes = 4 * CTXK * 256 + 4 * CTXK * 8 * 4 + T * 8 * CTXK * 2  # K + sc(f32) + out
  tops = T * 8 * CTXK * 128 * 2 / best / 1e12
  print(f"[p7f] T={T}: relerr med {med:.3e} F95 {f95:.3e} | {best*1e3:.3f} ms | "
        f"K-read+eff {kbytes/best/1e9:.1f} GB/s | dp4a {tops:.1f} TOPS | "
        f"{best*1e3*32/T:.3f} ms per 32-row QK pass equiv", flush=True)
  return med, f95, best, tops

m16 = run_T(16, 4096)
m64 = run_T(64, 1024)
print(f"[p7f] SUMMARY T=16: med {m16[0]:.2e} F95 {m16[1]:.2e} {m16[2]*1e3:.3f}ms {m16[3]:.1f}TOPS | "
      f"T=64: med {m64[0]:.2e} F95 {m64[1]:.2e} {m64[2]*1e3:.3f}ms {m64[3]:.1f}TOPS", flush=True)
print("[p7f] reference: shipped pfa16 pair = 13.5 TFLOPS / ~70 GB/s eff (2x KV reads), "
      "5.85 ms/32tok/layer QK+PV at 100k", flush=True)
