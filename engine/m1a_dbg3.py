# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-A debug 3: does zeroing the shared split-KV scratch (qw1/qw16_1/pm1/ps1/pA1)
BEFORE init_draft/fill_draft restore draft acceptance without any T=1 trunk run?
(A T=1 run writes exactly these buffers — the W2H boot order masked the bug.)"""
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
P = E.P
S = int(os.environ["SKV_S"])
P.win_up("qw1", 0, np.zeros(24*256, np.float32))
P.win_up("qw16_1", 0, np.zeros(24*256, np.float16))
P.win_up("pm1", 0, np.zeros(4*S*6, np.float32))
P.win_up("ps1", 0, np.zeros(4*S*6, np.float32))
P.win_up("pA1", 0, np.zeros(4*S*6*256, np.float32))
dev.synchronize()
print("[dbg3] split-KV scratch zeroed before fill", flush=True)
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])
E.fill_draft(ids)
print(f"[dbg3] dring0 after fill (expect {ref[0]}): {int(P.down_at('dring0', 0, 1)[0])}", flush=True)
E.build_graphs(); dev.synchronize()
E.reset_snapshot(SNAP, CUR0, P0)
sess = DecodeSession(E); sess.begin()
mtot = 0
for k in range(10):
  r = sess.step(); mtot += r["m"]
  if k < 3 or k == 9:
    print(f"[dbg3] cyc{k}: m={r['m']} amds={P.down_at('amds', 0, 3).tolist()} "
          f"dring=({int(P.down_at('dring0', 0, 1)[0])},{int(P.down_at('dring1', 0, 1)[0])})", flush=True)
print(f"[dbg3] acceptance over 10 cycles: sum(m)={mtot} alpha={mtot/20:.3f} "
      f"(canonical 0.892) -> {'FIX CONFIRMED' if mtot >= 12 else 'STILL BROKEN'}", flush=True)
print("[done]", flush=True)
