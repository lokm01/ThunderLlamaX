# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-A debug: why does draft acceptance collapse (m=0 every cycle) in a boot
WITHOUT the T=1 reference run (gate23/serve) but not in test_w100k (2.78 tok/cyc)?
Dump per-cycle amds/dring0/dring1/m/pos in variant A (no prior trunk run) and
variant B (after 3 T=1 GCycle tokens), same snapshot reset."""
import os, sys, time, json, collections
os.environ["SKV"] = "1"; os.environ["SKV_CTXK"] = "100352"
os.environ["SKV_S"] = "256"; os.environ["GEMVV"] = "1"; os.environ["KV8"] = "1"
os.environ["QH"] = "1"; os.environ["PVH"] = "1"; os.environ["HM"] = "1"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, DecodeSession, RBLK, CBLK, SLICE, CTXK
from engine0 import dev
from gcycle import GCycleEngine

SNAP = "~/snap100k"
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()
ref = np.load(f"{SNAP}/engine_t1_ref_kv8_qh.npy").tolist()

E = MTPEngine(theta=1e7)
E.load_snapshot_kv(SNAP)
for j, i in enumerate(E.gdn_idx):
  E.P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
  E.P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
  if j % 8 == 0: E._flush()
E._flush()
G = GCycleEngine(E); G.build(); dev.synchronize()
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])
E.fill_draft(ids)
E.build_graphs()
dev.synchronize()

sess = DecodeSession(E)
def dumpcycles(tag, n=3):
  E.reset_snapshot(SNAP, CUR0, P0)
  sess.begin()
  for k in range(n):
    r = sess.step()
    am = E.P.down_at("amds", 0, 3).tolist()
    d0 = int(E.P.down_at("dring0", 0, 1)[0]); d1 = int(E.P.down_at("dring1", 0, 1)[0])
    pos = int(E.P.down_at("pos_slot", 0, 1)[0])
    print(f"[{tag}] cyc{k}: m={r['m']} amds={am} dring=({d0},{d1}) pos_after={pos} emit={r['tokens']}", flush=True)

print("=== variant A: no prior trunk graph run ===", flush=True)
dumpcycles("A", 3)
print("=== variant B: after 3 T=1 GCycle tokens ===", flush=True)
G.run_tokens(3, wait_each=True); dev.synchronize()
dumpcycles("B", 3)
print("=== variant C: after B, again (idempotence) ===", flush=True)
dumpcycles("C", 3)
print("[done]", flush=True)
