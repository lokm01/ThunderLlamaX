# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-A debug 2: localize the draft-chain poison. Hypothesis: fill_draft step 0
reads POISONED shared scratch (split-KV partials qw1/qw16_1/pm1/ps1/pA1 and/or
draft GEMV residue hh_d) before writing it; a prior T=1 trunk run (the W2H
canonical order) overwrote that scratch with finite values, masking the bug.
A: fill_draft with boot poisons -> dump dring0 (post-fill) + 2 cycles.
B: zero-fill the scratch group -> re-run fill_draft -> dump + 2 cycles."""
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
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])

def dumpcycles(tag, n=2):
  E.reset_snapshot(SNAP, CUR0, P0)
  sess = DecodeSession(E); sess.begin()
  for k in range(n):
    r = sess.step()
    d0 = int(E.P.down_at("dring0", 0, 1)[0]); d1 = int(E.P.down_at("dring1", 0, 1)[0])
    print(f"[{tag}] cyc{k}: m={r['m']} amds={E.P.down_at('amds', 0, 3).tolist()} dring=({d0},{d1})", flush=True)

print("=== A: fill_draft with boot poisons ===", flush=True)
E.fill_draft(ids)
print(f"[A] dring0 right after fill_draft (expect ~{ref[0]}): {int(E.P.down_at('dring0', 0, 1)[0])}", flush=True)
dumpcycles("A", 2)

print("=== B: zero split-KV + draft GEMV scratch, re-fill ===", flush=True)
P = E.P
for nm, n in (("qw1", 24*256), ("qw16_1", 24*256), ("pm1", 4*256*6), ("ps1", 4*256*6),
              ("pA1", 4*256*6*256), ("hh_d", 5120), ("e_buf", 5120), ("xin_d", 5120)):
  if nm == "qw16_1": P.win_up(nm, 0, np.zeros(n, np.float16))
  elif nm in ("e_buf", "hh_d", "xin_d"): P.win_up(nm, 0, np.zeros(n, np.float32))
  else: P.win_up(nm, 0, np.zeros(n, np.float32))
dev.synchronize()
E.fill_draft(ids)
print(f"[B] dring0 right after fill_draft: {int(E.P.down_at('dring0', 0, 1)[0])}", flush=True)
dumpcycles("B", 2)
print("[done]", flush=True)
