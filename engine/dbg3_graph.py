# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV"); os.environ.setdefault("SKV", "1")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine
from trunk import CTX
from engine0 import dev
from gcycle import ParityGraph
LS = (256,1,1)
snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
# full restore (T=1 seeding)
E.restore(snap)
E._build_seqs()
seq = E._seq[0]
print(f"[dbg3] seq len {len(seq)}", flush=True)
# show entries 8..18 (first attn block region)
for n, (p, a, g) in enumerate(seq[7:17], 7):
    print(f"[dbg3] {n}: {getattr(p, 'name', '?')} grid={g} nargs={len(a)}", flush=True)
# eager token 0 (mid-sync at 230)
print("[dbg3] eager token 0 ...", flush=True)
E.token(0, wait=True)
print("[dbg3] eager token OK; tok_slot/pos:", E.P.down("tok_slot", (1,), np.int32), E.P.down("pos_slot", (1,), np.int32), flush=True)
# single graph submit + flusher (lone-graph law)
print("[dbg3] graph submit ...", flush=True)
g0 = ParityGraph(seq, tag="d3a")
g1 = ParityGraph(seq, tag="d3b")
prev = dev.timeline_value - 1
v0 = dev.next_timeline(); g0.submit(prev, v0); prev = v0
v1 = dev.next_timeline(); g1.submit(v0, v1); prev = v1
dev.timeline_signal.wait(v1)
dev.synchronize()
print("[dbg3] 2 graph submits OK", flush=True)
print("[dbg3 done]", flush=True)
