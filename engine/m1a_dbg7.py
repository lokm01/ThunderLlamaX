# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time, json, collections
os.environ["SKV"] = "1"; os.environ["SKV_CTXK"] = "100352"
os.environ["SKV_S"] = "256"; os.environ["GEMVV"] = "1"; os.environ["KV8"] = "1"
os.environ["QH"] = "1"; os.environ["PVH"] = "1"; os.environ["HM"] = "1"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, DecodeSession, RBLK, CBLK, SLICE, CTXK
from trunk import CTX as TRUNK_CTX
from engine0 import dev
from gcycle import GCycleEngine

SNAP = "~/snap100k"
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); assert CTXK == int(meta["CTXK"]) and TRUNK_CTX == CTXK
CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()
ref = np.load(f"{SNAP}/engine_t1_ref_kv8_qh.npy").tolist()
print(f"[dbg7] P={P0} cur0={CUR0}", flush=True)

# ---- VERBATIM test_w100k boot (load100k body) ----
E = MTPEngine(theta=1e7)
P = E.P
E.load_snapshot_kv(SNAP, progress=False)
for j, i in enumerate(E.gdn_idx):
  P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
  P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
  if j % 8 == 0: E._flush()
P.win_up("tok_slot", 0, np.array([CUR0], dtype=np.int32))
P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
E._flush()
dev.synchronize()
G = GCycleEngine(E)
G.build(); dev.synchronize()
G.run_tokens(60, wait_each=True)
print(f"[dbg7] after 60 T=1 tokens (dbg7: + hist copyout): pos_slot={int(P.down_at('pos_slot', 0, 1)[0])} "
      f"tok_slot={int(P.down_at('tok_slot', 0, 1)[0])} (expect pos={P0+60})", flush=True)
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
sl = sl[:SLICE]
print(f"[dbg7] slice {len(set(sl))} distinct", flush=True)
E.init_draft(sl)
E.fill_draft(ids)
print(f"[dbg7] dring0 after fill: {int(P.down_at('dring0', 0, 1)[0])}", flush=True)
E.build_graphs()
dev.synchronize()
_h = E.P.down("tok_hist", (CTXK + 256,), np.int32)   # the 400KB copyout gate1 does
print(f"[dbg7] hist copyout done, hist[97810]={_h[97810]}", flush=True)
E.reset_snapshot(SNAP, CUR0, P0)
sess = DecodeSession(E); sess.begin()
mtot = 0
for k in range(10):
  r = sess.step(); mtot += r["m"]
  if k < 3 or k == 9:
    print(f"[dbg7] cyc{k}: m={r['m']} amds={P.down_at('amds', 0, 3).tolist()} "
          f"dring=({int(P.down_at('dring0', 0, 1)[0])},{int(P.down_at('dring1', 0, 1)[0])})", flush=True)
print(f"[dbg7] sum(m)={mtot} alpha={mtot/20:.3f} -> {'WORKS' if mtot >= 12 else 'BROKEN'}", flush=True)
print("[done]", flush=True)
