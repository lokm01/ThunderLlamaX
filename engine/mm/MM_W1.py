#!/usr/bin/env python3
"""GPU WINDOW 1: engine battery20 replay (saved for the P5.1 compare once
A16 lands) + the FULL P6 spec path (G1 kernel gates, G2 P-batch core,
G3 spec==T1, G4 perf) in ONE GPU process."""
import os, sys, time, json
import numpy as np
BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
PROG = os.path.expanduser("~/mm_p5_progress.txt")
def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)
def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

from MM_P56_lib import Rig, GraphRunner, feed_token, t1_argmax

rig = Rig()
aold = np.load(os.path.expanduser("~/mm_p34_anchor.npz"), allow_pickle=True)
ids20 = [np.asarray(x, dtype=np.int32) for x in aold["ids"]]

# ---- engine battery20 (the P5.1 GPU side) ----
if not done("ENG20"):
    seq1 = rig.build_seq(1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
    gr = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq1], "w1t1")
    t0 = time.time()
    out = []
    for pi, idl in enumerate(ids20):
        rig.reset_states(1024)
        pt1 = []
        for pos, tid in enumerate(idl):
            feed_token(rig, int(tid), pos)
            gr.step()
            pt1.append(t1_argmax(rig))
        out.append(pt1)
        print(f"    [ENG20] prompt {pi} ({len(idl)} pos)", flush=True)
    json.dump({"engine": out}, open(os.path.expanduser("~/mm_p5_engine20.json"), "w"))
    record("ENG20", f"engine battery20 replayed+saved ({time.time()-t0:.0f}s, ka-pool slabs={rig.pool.slabs} carves={rig.pool.carves})")

# ---- the full P6 ----
import MM_P6_run
MM_P6_run.record = record
MM_P6_run.done = done
MM_P6_run.main(rig=rig)
print("[W1 DONE]", flush=True)
