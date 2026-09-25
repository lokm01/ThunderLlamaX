# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R3 lookup in-graph repro: the kernel direct-launched clean (lut_test), but the
first draft-graph cycle faulted. Minimal discriminator: run lookup_nw32 through
a 1-kernel ParityGraph (twice — LONE-GRAPH law), then through a 2-kernel graph
chained after a dummy, at the REAL 100k pos. Narrows: QMD/ls issue vs chain
context."""
import os, sys, json
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev
from gcycle import ParityGraph
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
P = Bufs()
pr = {}
for n in ("lookup_nw32",):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
# dposadd for the flusher/dummy (1-CTA 256-thread proven class)
for n in ("dposadd",):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
SNAP = "~/snap100k"
ids = np.load(f"{SNAP}/ids.npy").tolist()
out = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
CTX = 100352
P.up("tok_hist", np.full(CTX + 128, -1, dtype=np.int32))
P.up("l_hist", np.zeros(4096, dtype=np.int32))
P.up("pos_slot", np.zeros(1, dtype=np.int32))
P.up("cur_slot", np.zeros(1, dtype=np.int32))
P.up("cyc_slot", np.zeros(1, dtype=np.int32))
P.up("dring0", np.full(1, -1, dtype=np.int32))
P.up("dring1", np.full(1, -1, dtype=np.int32))
P.up("fillpos", np.zeros(1, dtype=np.int32))
dev.synchronize(); d = P.d

def setpos(j):
  fed = ids + out[:j]
  P.win_up("tok_hist", 0, np.array(fed, dtype=np.int32))
  P.win_up("pos_slot", 0, np.array([len(fed)], dtype=np.int32))
  P.win_up("cur_slot", 0, np.array([out[j]], dtype=np.int32))
  P.win_up("dring0", 0, np.full(1, -777, dtype=np.int32))
  P.win_up("dring1", 0, np.full(1, -777, dtype=np.int32))
  P.win_up("l_hist", 0, np.zeros(1, dtype=np.int32))
  dev.synchronize()

LK = lambda: (pr["lookup_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"], d["l_hist"], d["cyc_slot"]), 1)
FL = lambda: (pr["dposadd"], (d["fillpos"], d["fillpos"]), 1)

print("[t1] single-kernel graph x2 submits (lone-graph law)", flush=True)
g1 = ParityGraph([LK()], tag="lut1")
setpos(40)
prev = dev.timeline_value - 1
v1 = dev.next_timeline(); g1.submit(prev, v1)
v2 = dev.next_timeline(); g1.submit(v1, v2)
dev.timeline_signal.wait(v2); dev.synchronize()
print("[t1] clean; dring0", int(P.down("dring0", (1,), np.int32)[0]), "l", int(P.down("l_hist", (1,), np.int32)[0]), flush=True)

print("[t2] 3-kernel graph: dposadd, lookup, dposadd (chain context)", flush=True)
g2 = ParityGraph([FL(), LK(), FL()], tag="lut2")
setpos(55)
prev = dev.timeline_value - 1
v1 = dev.next_timeline(); g2.submit(prev, v1)
v2 = dev.next_timeline(); g2.submit(v1, v2)
dev.timeline_signal.wait(v2); dev.synchronize()
print("[t2] clean; dring0", int(P.down("dring0", (1,), np.int32)[0]), "l", int(P.down("l_hist", (1,), np.int32)[0]), flush=True)
print("[lut_graph_test] DONE CLEAN", flush=True)
