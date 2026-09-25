# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2 graph bisect: build draft graphs on kernel-prefixes of the draft seq,
submit+wait each (increasing prefix) until fault -> smallest faulting prefix."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from mtp import MTPEngine, SLICE
from gcycle import ParityGraph
from engine0 import dev

snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
ids = snap["ids"].reshape(-1).tolist()
sl = (ids * 4000)[:SLICE]
E.init_draft(sl)
E.restore_mtp(snap)
d, pr = E.P.d, E.pr

full = E._draft_entries(d["cur_slot"], d["pos_slot"], d["h_seed"], d["hd_d0"], d["dring0"])
full += [(pr["dposadd"], (d["pos_slot"], d["dpos1"]), 1)]
full += E._draft_entries(d["dring0"], d["dpos1"], d["hd_d0"], d["hd_d1"], d["dring1"])
names = ["embed","dnorm2","ehproj","k0n","dq","dkv","aattn","doproj","hh","dfgu","ddown","k0n2","shead","samx"]*2 + ["dpos"]
print(f"[dbg] draft seq {len(full)} kernels", flush=True)
dev.synchronize()

import time
fl = ParityGraph([(pr["dposadd"], (d["fillpos"], d["dpos1"]), 1)], tag="dfl")
for n in range(2, len(full)+1):
  try:
    g = ParityGraph(full[:n], tag=f"dbg{n}")
    prev = dev.timeline_value - 1
    v = dev.next_timeline(); g.submit(prev, v)
    vf = dev.next_timeline(); fl.submit(v, vf)
    dev.timeline_signal.wait(vf, timeout=8000)
    print(f"[dbg] prefix {n} ({names[n-1]}) OK", flush=True)
  except Exception as ex:
    print(f"[dbg] prefix {n} ({names[n-1]}) FAULT: {type(ex).__name__} {str(ex)[:120]}", flush=True)
    break
print("[dbg] done", flush=True)
