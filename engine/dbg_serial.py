# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2 serial-graph debug: run draft, probe, accept of cycle 1 one at a time
(each with a chained flusher) -> names the faulting graph in the REAL context."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from mtp import MTPEngine, SLICE
from engine0 import dev

snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
ids = snap["ids"].reshape(-1).tolist()
E.init_draft((ids * 4000)[:SLICE])
E.restore_mtp(snap)
E.fill_draft(ids)
E.build_graphs()
dev.synchronize()
draft_g, probe_g, accept_g, flush_g = E.graphs

def step(label, g, prev):
  v = dev.next_timeline(); g.submit(prev, v)
  vf = dev.next_timeline(); flush_g.submit(v, vf)
  try:
    dev.timeline_signal.wait(vf, timeout=12000)
    print(f"[ser] {label} OK", flush=True)
    return v
  except Exception as e:
    print(f"[ser] {label} FAULT: {str(e)[:70]}", flush=True)
    return None

prev = dev.timeline_value - 1
v = step("DRAFT", draft_g, prev)
if v is not None:
  v = step("PROBE", probe_g, v)
  if v is not None:
    print("[ser] eager accept...", flush=True)
    import time
    E.pr["accept"](E.P.d["amds"], E.P.d["dring0"], E.P.d["dring1"], E.P.d["xA"], E.P.d["m_slot"], E.P.d["m_hist"], E.P.d["cyc_slot"], E.P.d["pos_slot"], E.P.d["cur_slot"], E.P.d["tok_hist"], E.P.d["h_seed"], global_size=(1,1,1), local_size=(256,1,1), wait=True)
    print("[ser] eager accept OK", flush=True)
    E.pr["acceptsel"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], global_size=(48,1,1), local_size=(256,1,1), wait=True)
    print("[ser] eager acceptsel OK", flush=True)
    if True:
      amds = E.P.down("amds", (3,), np.int32)
      dr = (E.P.down("dring0", (1,), np.int32)[0], E.P.down("dring1", (1,), np.int32)[0])
      m = E.P.down("m_slot", (1,), np.int32)[0]
      pos = E.P.down("pos_slot", (1,), np.int32)[0]
      cur = E.P.down("cur_slot", (1,), np.int32)[0]
      h = E.P.down("tok_hist", (2100,), np.int32)
      print(f"[ser] amds={amds.tolist()} dring={dr} m={m} pos={pos} cur={cur}", flush=True)
      print(f"[ser] hist[1987:1995]={h[1987:1995].tolist()}", flush=True)
print("[ser] done", flush=True)
