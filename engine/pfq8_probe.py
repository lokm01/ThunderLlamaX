# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7F-2 probe: standalone pfa16 baseline + pfa8 (IMMA QK int8) validation/bench.
Real-scale synthetic kv8/qw buffers; synced timing (min-of-N); poison-first.
Run: cd ~/tinygrad-metal/engine0 && DEV=NV ~/tg311/bin/python -u pfq8_probe.py [base|a8|all]
"""
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

KSYM = {"pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16", "pfa8nw32_s32_100k": "pfa8", "pfa8t64nw32_s32_100k": "pfa8t64", "pfk_q8nw8": "pfk_q8"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))

def bench(fn, n=5):
  for _ in range(2): fn()
  dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter(); fn(); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best

def setup():
  # kv: 8 slabs [CTXK][256] u8 (4 K + 4 V); sc: 8 [CTXK][8] f16
  kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
  P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
  sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8))).astype(np.float16)
  P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
  qw = (rng.standard_normal(32*24*256) * 0.5).astype(np.float16)
  P.poison("qw", 32*24*256*2, np.float16, np.float16(7.7)); P.up("qw", qw)
  P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
  P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
  P.poison("pos_slot", 4, np.int32, -1)
  P._keep.clear(); dev.synchronize()

def down_state(tag):
  pm = P.down("pm", (4*S*RMAX,), np.float32).copy()
  ps = P.down("ps", (4*S*RMAX,), np.float32).copy()
  pA = P.down("pA", (4*S*RMAX*256,), np.float32).copy()
  return pm, ps, pA

def relerr(a, b):
  den = np.maximum(np.abs(a), 1e-6)
  return float(np.median(np.abs(a - b) / den)), float(np.max(np.abs(a - b) / den))

mode = sys.argv[1] if len(sys.argv) > 1 else "all"
setup()
pfa16 = prog("pfa16nw32_s32_100k")

import sys as _s
if mode in ("base", "all", "attr"):
  pass
else:
  _s.modules[__name__].__dict__.setdefault("_skip_base", True)
for pos in ([] if mode not in ("base", "all") else [100336, 32768, 2048]):
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
  P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
  t = bench(lambda: pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"],
                          P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1)))
  kvread = min(pos+ROWS, CTXK) * 256 * 8
  print(f"[base] pfa16 pos={pos}: {t*1e3:8.2f} ms/launch  kv+sc read ~{kvread/1e9:.3f} GB -> {kvread/t/1e9:7.1f} GB/s  "
        f"qk+pv {(2*RMAX*4*min(pos+ROWS,CTXK)*256*2)/t/1e12:6.2f} TOPS", flush=True)
  if pos == 100336:
    globals()["BASE_STATE"] = down_state("base")
    # determinism x2
    P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
    pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1))
    st2 = down_state("base2")
    print(f"[base] determinism: pm {np.array_equal(BASE_STATE[0], st2[0])} ps {np.array_equal(BASE_STATE[1], st2[1])} pA {np.array_equal(BASE_STATE[2], st2[2])}", flush=True)
P._keep.clear()
print("[base] DONE", flush=True)

# ============================ mode a8: pfa8 ============================
if mode in ("a8", "all"):
  pfa8 = prog("pfa8nw32_s32_100k")
  pfkq8 = prog("pfk_q8nw8")
  qw_np = P.down("qw", (32*24*256,), np.float16).copy().reshape(32*24, 256)
  # numpy reference quantizer (must match pfk_q8 bit-for-bit)
  v32 = qw_np.astype(np.float32)
  am = np.abs(v32).reshape(-1, 8, 32).max(axis=2).astype(np.float32)
  qsc_np = (am / np.float32(127.0)).astype(np.float32)
  inv = np.where(am > 0, np.float32(127.0) / np.maximum(am, np.float32(1e-30)), np.float32(0.0)).astype(np.float32)
  inv2 = np.repeat(inv, 32, axis=1)
  q_np = np.round(v32 * inv2).astype(np.int32)
  qs8_np = q_np.astype(np.int8)
  # kernel quantizer (per half A view = first 384 rows)
  P.poison("qs8", 32*24*256, np.int8, np.int8(-19)); P.poison("qsc", 32*24*8*4, np.float32, 7.7e31)
  pfkq8(P.d["qw"], P.d["qs8"], P.d["qsc"], global_size=(48,1,1), local_size=(256,1,1)); dev.synchronize()
  qs8_k = P.down("qs8", (32*24*256,), np.int8).copy()
  qsc_k = P.down("qsc", (32*24*8,), np.float32).copy()
  print(f"[a8] pfk_q8 vs numpy: qs8 bit-eq {np.array_equal(qs8_k.view(np.uint8), qs8_np.view(np.uint8))} "
        f"qsc bit-eq {np.array_equal(qsc_k, qsc_np.reshape(-1))} maxdiff {np.max(np.abs(qsc_k-qsc_np.reshape(-1)))}", flush=True)
  for pos in [100336, 32768, 2048]:
    P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
    P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
    P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
    pfa8(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"],
         P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1))
    dev.synchronize()
    pm8, ps8, pA8 = down_state("a8")
    # rerun pfa16 for the SAME pos to compare
    P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
    P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
    pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
          global_size=(4*S,1,1), local_size=(1024,1,1)); dev.synchronize()
    pmr, psr, pAr = down_state("ref")
    e_pA = relerr(pAr, pA8); e_ps = relerr(psr, ps8)
    print(f"[a8] pos={pos}: pA relerr med {e_pA[0]:.3e} max {e_pA[1]:.3e} | ps relerr med {e_ps[0]:.3e}", flush=True)
    if pos == 100336:
      t = bench(lambda: pfa8(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"],
                             P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1)))
      kvread = min(pos+ROWS, CTXK) * 256 * 8
      print(f"[a8] pfa8 pos={pos}: {t*1e3:8.2f} ms/launch  kv read {kvread/t/1e9:7.1f} GB/s  "
            f"qk+pv {(2*RMAX*4*min(pos+ROWS,CTXK)*256*2)/t/1e12:6.2f} TOPS", flush=True)
      # determinism x2 + numpy-quant bit-check
      P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
      pfa8(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
           global_size=(4*S,1,1), local_size=(1024,1,1)); dev.synchronize()
      st2 = down_state("d2")
      print(f"[a8] determinism: pm {np.array_equal(pm8, st2[0])} ps {np.array_equal(ps8, st2[1])} pA {np.array_equal(pA8, st2[2])}", flush=True)
  P._keep.clear()
  print("[a8] DONE", flush=True)

# ============================ mode attr ============================
if mode == "attr":
  P.poison("qs8", 32*24*256, np.int8, np.int8(3)); P.poison("qsc", 32*24*8*4, np.float32, 0.01)
  for A, desc in [(1, "staging+QK only"), (2, "staging+owners+PV"), (3, "staging only"), (4, "empty")]:
    pk = prog(f"attn8a{A}")
    P.win_up("pos_slot", 0, np.array([100336], dtype=np.int32)); dev.synchronize()
    t = bench(lambda: pk(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"],
                         P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1)), n=5)
    print(f"[attr] A{A} ({desc}): {t*1e3:7.2f} ms", flush=True)
  P._keep.clear()

# ============================ mode a8b: pfa8t64 ============================
if mode == "a8b":
  pfa8t = prog("pfa8t64nw32_s32_100k")
  pfkq8 = prog("pfk_q8nw8")
  # quantize BOTH halves
  P.poison("qs8", 32*24*256, np.int8, np.int8(-19)); P.poison("qsc", 32*24*8*4, np.float32, 7.7e31)
  pfkq8(P.d["qw"], P.d["qs8"], P.d["qsc"], global_size=(48,1,1), local_size=(256,1,1))
  pfkq8(P.d["qw"].offset(offset=384*512, size=384*512), P.d["qs8"].offset(offset=384*256, size=384*256),
        P.d["qsc"].offset(offset=384*8*4, size=384*8*4), global_size=(48,1,1), local_size=(256,1,1))
  dev.synchronize()
  import os as _o
  if _o.getenv("A8B_NODOWN") != "1":
    qw_np = P.down("qw", (32*24*256,), np.float16).copy().reshape(32*24, 256)
    v32 = qw_np.astype(np.float32)
    am = np.abs(v32).reshape(-1, 8, 32).max(axis=2).astype(np.float32)
    qsc_np = (am / np.float32(127.0)).astype(np.float32)
    inv = np.where(am > 0, np.float32(127.0) / np.maximum(am, np.float32(1e-30)), np.float32(0.0)).astype(np.float32)
    q_np = np.round(v32 * np.repeat(inv, 32, axis=1)).astype(np.int32)
    qs8_k = P.down("qs8", (32*24*256,), np.int8).copy(); qsc_k = P.down("qsc", (32*24*8,), np.float32).copy()
    print(f"[a8b] pfk_q8 both halves vs numpy: qs8 bit-eq {np.array_equal(qs8_k.view(np.uint8), q_np.astype(np.int8).view(np.uint8))} "
          f"qsc bit-eq {np.array_equal(qsc_k, qsc_np.reshape(-1))}", flush=True)
  for pos in [100336, 32768, 2048]:
    P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
    P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
    P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
    pfa8t(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"],
          P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1))
    dev.synchronize()
    pm8, ps8, pA8 = down_state("a8b")
    P.poison("pm", 4*S*RMAX*4, np.float32, 7.7e31); P.poison("ps", 4*S*RMAX*4, np.float32, 7.7e31)
    P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
    pfa16(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
          global_size=(4*S,1,1), local_size=(1024,1,1)); dev.synchronize()
    pmr, psr, pAr = down_state("ref")
    msk = np.abs(pAr) > np.abs(pAr).max()*1e-3
    e_pA = relerr(pAr[msk], pA8[msk])
    print(f"[a8b] pos={pos}: pA(sig) relerr med {e_pA[0]:.3e} max {e_pA[1]:.3e} (n={msk.sum()})", flush=True)
    if pos == 100336:
      t = bench(lambda: pfa8t(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"],
                              P.d["pm"], P.d["ps"], P.d["pA"], global_size=(4*S,1,1), local_size=(1024,1,1)))
      print(f"[a8b] pfa8t64 pos={pos}: {t*1e3:7.2f} ms/launch", flush=True)
      P.poison("pA", 4*S*RMAX*256*4, np.float32, 7.7e31)
      pfa8t(P.d["kv"], P.d["sc"], P.d["qs8"], P.d["qsc"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
            global_size=(4*S,1,1), local_size=(1024,1,1)); dev.synchronize()
      st2 = down_state("d")
      print(f"[a8b] determinism: {np.array_equal(pm8, st2[0])} {np.array_equal(ps8, st2[1])} {np.array_equal(pA8, st2[2])}", flush=True)
  P._keep.clear()
  print("[a8b] DONE", flush=True)
