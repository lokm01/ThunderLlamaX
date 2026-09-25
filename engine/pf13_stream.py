# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P13 kernel A harness — the per-SM steady-state stream MECHANISM probe.
82 CTAs (1/SM, one wave) x NTHR threads, RINGD-deep uint4 register ring, XOR
consume (no smem/sync/mma). Synced min-of-10. The NCTA=272 variant reproduces
the shipped 3.3-wave pure-stream reference (~795 GB/s class).
Buffer: 272 parcels x 163840 uint4 (2.62MB/CTA); 82-CTA runs use the first
82 parcels. nsteps = 163840/NTHR = 640 (nw8, %8=0) / 320 (nw16) / 160 (nw32).
Verdict rule: 82-CTA steady state >=400 GB/s -> wave/ramp artifact (persistence
pays); ~265 or less -> per-SM stream limit -> KILL.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad import dtypes
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

INT = (None, 4, dtypes.int32, ())
BASE = "~/tinygrad-metal/engine0"
NPARC_MAX, NU16 = 272, 163840
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=(INT,)))

rng = np.random.default_rng(13)
buf = rng.integers(0, 256, NPARC_MAX * NU16 * 16, dtype=np.uint8)
P.up("src", buf)
P.up("sink", np.zeros(272, dtype=np.uint32))
dev.synchronize()
print(f"[setup] buffer {buf.nbytes/1e6:.0f} MB up", flush=True)

VARIANTS = [
  ("pf13s_nw8d4",   256,  82),
  ("pf13s_nw8d8",   256,  82),
  ("pf13s_nw8d16",  256,  82),
  ("pf13s_nw16d8",  512,  82),
  ("pf13s_nw32d4",  1024, 82),
  ("pf13s272_nw8d8",256, 272),
]
only = sys.argv[1:] or None
for name, nthr, ncta in VARIANTS:
  if only and not any(o in name for o in only): continue
  pr = prog(name)
  def one():
    pr(P.d["src"], P.d["sink"], vals=(NU16,), global_size=(ncta, 1, 1), local_size=(nthr, 1, 1))
  one(); dev.synchronize()          # warm
  best = 1e9
  for _ in range(10):
    t0 = time.perf_counter()
    one(); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  byts = ncta * NU16 * 16
  print(f"[stream] {name:<15} grid {ncta:3d} x {nthr:4d}thr | {best*1e3:8.3f} ms | "
        f"{byts/best/1e9:7.1f} GB/s aggregate | {byts/best/1e9/ncta:5.2f} GB/s/SM", flush=True)

print("[stream] MECHANISM VERDICT: compare 82-CTA rows vs the 272-CTA reference row", flush=True)
