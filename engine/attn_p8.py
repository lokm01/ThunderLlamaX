# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8: attention co-residency A/B — shipped pfa16 (1024thr, threads-capped 1/SM)
vs pfa16ctl_nw16 (512thr, can co-reside 2/SM under the carveout unlock).
Synthetic real-scale buffers (pfq8_probe pattern); validation + synced min-of-10.
Process env controls carveout. Usage: ~/tg311/bin/python -u attn_p8.py"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
CTXK, S, ROWS = 100352, 32, 16
RMAX = 96
rng = np.random.default_rng(11)
INT = (None, 4, dtypes.int32, ())

KSYM = {"pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16", "pfa16ctl_nw16_s32_100k": "pfa16ctl", "pfc16ctl_s32": "pfc16ctl"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))

print(f"[env] AUTO={os.getenv('NV_SMEM_CFG_AUTO','-')} TGT={os.getenv('NV_SMEM_CFG_AUTO_TGT','-')} ANAMES={os.getenv('NV_SMEM_CFG_AUTO_NAMES','-')} CFG={os.getenv('NV_SMEM_CFG','-')} NAMES={os.getenv('NV_SMEM_CFG_NAMES','-')}", flush=True)

kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8,))).astype(np.float16)
P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
qw = (rng.standard_normal(32*24*256) * 0.5).astype(np.float16)
P.poison("qw", 32*24*256*2, np.float16, np.float16(7.7)); P.up("qw", qw)
for nm, nb in [("pm", 4*S*RMAX*4), ("ps", 4*S*RMAX*4), ("pA", 4*S*RMAX*256*4)]:
  P.poison(nm, nb, np.float32, 7.7e31)
P.poison("pos_slot", 4, np.int32, -1)
P._keep.clear(); dev.synchronize()

def down():
  return (P.down("pm", (4*S*RMAX,), np.float32).copy(),
          P.down("ps", (4*S*RMAX,), np.float32).copy(),
          P.down("pA", (4*S*RMAX*256,), np.float32).copy())

def bench(fn, n=10):
  for _ in range(2): fn()
  dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter(); fn(); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best

def relerr(a, b):
  den = np.maximum(np.abs(a), 1e-6)
  return float(np.median(np.abs(a-b)/den)), float(np.max(np.abs(a-b)/den))

pfa16 = prog("pfa16nw32_s32_100k")
pfctl = prog("pfa16ctl_nw16_s32_100k")
LS = (256,1,1)
for pos in (100336, 32768, 2048):
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
  P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
  pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
        global_size=(4*S,1,1), local_size=(1024,1,1))
  dev.synchronize()
  ref = down()
  P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
  P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
  pfctl(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
        global_size=(4*S,1,1), local_size=(512,1,1))
  dev.synchronize()
  got = down()
  bit = all(np.array_equal(a, b) for a, b in zip(ref, got))
  e = relerr(ref[2], got[2])
  ts = bench(lambda: pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
                           global_size=(4*S,1,1), local_size=(1024,1,1)))
  tc = bench(lambda: pfctl(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
                           global_size=(4*S,1,1), local_size=(512,1,1)))
  kvread = min(pos+ROWS, CTXK) * 256 * 8
  print(f"[p8] pos={pos}: pfa16 {ts*1e3:7.2f} ms ({kvread/ts/1e9:6.1f} GB/s) | ctl-nw16 {tc*1e3:7.2f} ms ({kvread/tc/1e9:6.1f} GB/s) | x{ts/tc:.2f} | {'BIT-EXACT' if bit else f'relerr pA med {e[0]:.2e} max {e[1]:.2e}'}", flush=True)

# combine pair (g=24)
pfc16 = prog("pfc16_s32"); pfcctl = prog("pfc16ctl_s32")
P.poison("qrow", 32*6144*2, np.float16, np.float16(3.3))
P.up("qrow", (rng.standard_normal((32*6144))*0.3).astype(np.float16).reshape(-1))
P.poison("ao", 32*6144*2, np.float16, np.float16(7.7))
P.win_up("pos_slot", 0, np.array([100336], dtype=np.int32)); dev.synchronize()
pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1))
dev.synchronize()
t1 = bench(lambda: pfc16(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow"], P.d["ao"], global_size=(24,1,1), local_size=LS))
t2 = bench(lambda: pfcctl(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow"], P.d["ao"], global_size=(24,1,1), local_size=LS))
print(f"[p8] combine: pfc16 {t1*1e3:.3f} ms | pfc16ctl {t2*1e3:.3f} ms", flush=True)
print("[p8] DONE", flush=True)
