# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Staged engine0 smoke: launch each kernel with wait after each; find faults."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device
import engine0
from engine0 import GDNBlockEngine
dev = Device["NV"]
rng = np.random.default_rng(3)
print("[engine init: uploading weights]", flush=True)
e = GDNBlockEngine(0)
e.set_inputs(x=(rng.standard_normal(5120)*0.2).astype(np.float32),
             conv=(rng.standard_normal(3*10240)*0.1).astype(np.float32),
             rec=(rng.standard_normal(48*128*128)*0.1).astype(np.float32))
dev.synchronize()
print("[inputs up]", flush=True)
order = ["k0_norm","k1_q5","k1_iq3","k1_ab","k2_scan","k2b_z","k3a_oproj","k3m_hh","k3b_ffn","k3c_down"]
for name in order:
  t0 = time.perf_counter()
  e.launch_one(name, wait=True)
  dev.synchronize()
  print(f"[smoke] {name:10s} OK  {(time.perf_counter()-t0)*1e3:.1f} ms", flush=True)
# sanity values
import math
qkv = e.P.down("qkv_row", (10240,), np.float16).astype(np.float32)
h = e.P.down("y", (5120,))
print(f"[smoke] qkv finite={np.isfinite(qkv).all()} |qkv|max={np.abs(qkv).max():.2f}", flush=True)
print(f"[smoke] y finite={np.isfinite(h).all()} |y|max={np.abs(h).max():.2f}", flush=True)
rec = e.P.down("rec", (48*128*128,))
print(f"[smoke] rec finite={np.isfinite(rec).all()} |rec|max={np.abs(rec).max():.2f}", flush=True)
print("[smoke done]", flush=True)
