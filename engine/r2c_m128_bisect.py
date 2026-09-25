# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c: M128 plan fault bisect — boot the trunk engine, build ensure128's plan,
run entries ONE AT A TIME (wait=True), printing each kernel name; the last
print before death localizes the hang/fault. Diagnostic host (not a gate)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C

E = TrunkEngineW1C(theta=1e7)
import pf_prefill
pf_prefill.ensure128(E)
plan = E._pf_plan128
P = E.P
P.win_up("pos_slot", 0, np.zeros(1, np.int32))
P.win_up("ids128", 0, np.zeros(128, np.int32))
P.win_up("pos_arr128", 0, np.array([64], dtype=np.int32))   # a mid-ish pos (KV rows exist at 0? kv poisoned — fine for fault bisect)
P.win_up("pos_w128", 0, (np.arange(8, dtype=np.int32) * 16) + 64)
dev.synchronize(); P._keep.clear()
print(f"[bisect] plan={len(plan)} entries; running 1-by-1", flush=True)
t0 = time.perf_counter()
for k, (p, a, g, ls) in enumerate(plan):
  nm = getattr(p, "name", f"e{k}")
  print(f"[bisect] {k}: {nm} g={g}", flush=True)
  p(*a, global_size=(g, 1, 1), local_size=ls, wait=True)
print(f"[bisect] FULL PLAN CLEAN ({time.perf_counter()-t0:.1f}s)", flush=True)
