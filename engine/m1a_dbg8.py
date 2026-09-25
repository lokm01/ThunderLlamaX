# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W3-100k GOAL RUN: engine split-KV @100352 ctx.
 (1) T=1 greedy reference 60 tokens; (2) spec K=2 Tier-1 gate (bit-exact x2);
 (3) 3 timing reps + phase breakdown; alpha from m_hist; stock cross-check.
env: SKV=1 SKV_CTXK=100352 set below BEFORE engine imports."""
import os, sys, time, json, collections
os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, DecodeSession, RBLK, CBLK, SLICE, CTXK
from trunk import CTX as TRUNK_CTX
from engine0 import dev
from gcycle import GCycleEngine

SNAP = os.getenv("SNAPDIR", "~/snap100k")
NTOK = int(os.getenv("NTOK", "60"))
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "1"))
DO_T1 = os.getenv("DO_T1", "1") == "1"

meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); assert CTXK == int(meta["CTXK"])
CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()
# SHIFTED seeding: the mtp_v3 snapshot is POST-prefill (all P tokens fed, cur0 =
# first predicted token). Engine contract: pos_slot = P (positions 0..P-1 done),
# tok_slot = cur0 (fed at position P on the first pass). Engine 60 outputs =
# mtp_v3 outs[1:] (= spec_base_100k.json[1:]).
print(f"[w100k] P={P0} CTXK={CTXK} cur0={CUR0} prompt ids {len(ids)}", flush=True)

def vram_report(E):
  tot = 0; n = 0
  for name, b in E.P.d.items():
    nb = getattr(b, "nbytes", None)
    if nb is None:
      nb = getattr(b, "size", 0)
      try: nb = int(nb) * 4
      except: nb = 0
    tot += int(nb); n += 1
  try:
    fa = getattr(dev.allocator, "alloced", None) or getattr(dev.allocator, "_alloced", None)
  except Exception: fa = None
  print(f"[vram] engine buffers: {n} allocs, {tot/1e9:.2f} GB (arithmetic sum){' + alloced=' + str(fa) if fa else ''}", flush=True)

def win_up(P, name, off, arr):
  P.win_up(name, off, arr)   # M1-A: Bufs method (fixed-handle windowed upload)

KV8 = os.getenv("KV8", "0") == "1"
QH = os.getenv("QH", "0") == "1"
REFSUF = ("_kv8_qh" if QH else "_kv8") if KV8 else ""

def load100k(E, kv=True):
  # M1-A fixed handles: everything win_up'd into the boot-time full-CTXK buffers.
  assert TRUNK_CTX == CTXK, f"trunk CTX {TRUNK_CTX} != CTXK {CTXK} (set SKV_CTXK before launch)"
  P = E.P
  if kv:
    E.load_snapshot_kv(SNAP, progress=True)
  for j, i in enumerate(E.gdn_idx):
    P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
    P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
    if j % 8 == 0: E._flush()
  P.win_up("tok_slot", 0, np.array([CUR0], dtype=np.int32))
  P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
  E._flush()

def seed_mtp_slots(E):
  P = E.P
  for j, i in enumerate(E.gdn_idx):
    r = np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r")
    c = np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r")
    win_up(P, "rec4", (j*5+4)*RBLK*4, r)
    win_up(P, "conv4", (j*5+4)*CBLK*4, c)
    if j % 16 == 0: dev.synchronize()
  dev.synchronize()

def reset_spec(E):
  # M1-A: fixed-handle snapshot reset (mfill poison + windowed slot-4 seeds);
  # graphs were built ONCE after fill_draft and stay valid across resets.
  E.reset_snapshot(SNAP, CUR0, P0)

def hist(E):
  return E.P.down("tok_hist", (CTXK + 256,), np.int32)

print("[w100k] loading engine (~13GB weights)...", flush=True)
t0 = time.perf_counter()
E = MTPEngine(theta=1e7)
print(f"[w100k] engine loaded {time.perf_counter()-t0:.1f}s", flush=True)
load100k(E)
vram_report(E)
dev.synchronize()

# ---------- T=1 reference ----------
if DO_T1:
  G = GCycleEngine(E)
  G.build(); dev.synchronize()
  t0 = time.perf_counter()
  G.run_tokens(NTOK, wait_each=True)
  dt1 = time.perf_counter() - t0
  h = hist(E)
  ref = h[P0:P0+NTOK].tolist()
  print(f"[ref] T=1: {dt1/NTOK*1e3:.2f} ms/tok = {NTOK/dt1:.2f} tok/s; toks {ref[:12]}...", flush=True)
  assert all(t >= 0 for t in ref), "T=1 reference incomplete"
  np.save(f"{SNAP}/engine_t1_ref{REFSUF}.npy", np.array(ref, dtype=np.int64))
else:
  ref = np.load(f"{SNAP}/engine_t1_ref{REFSUF}.npy").tolist()
  G = None

# ---------- slice + draft ----------
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
sl = sl[:SLICE]
print(f"[slice] {len(set(sl))} distinct ids; ref covered: {sum(1 for t in ref if t in set(sl))}/60", flush=True)
E.init_draft(sl)

E.fill_draft(ids)          # ONCE (~5 min at 100k)
E.build_graphs()
dev.synchronize()

def spec_run(ncyc=NTOK, phase_times=False):
  pass  # reset_spec now done inline below
  r = E.run_cycles(ncyc, sync_every=SYNC_EVERY, phase_times=phase_times)
  h = hist(E)
  out = h[P0:P0+NTOK]
  m = E.P.down("m_hist", (1024,), np.int32)
  return out, m, r

sess = DecodeSession(E)   # M1-A serving core drives every spec run below

def decode_n(n=NTOK):
  """decode n cycles via DecodeSession; returns (emitted_tokens, pos_end)."""
  sess.begin()
  emt = []
  for _ in range(n):
    r = sess.step()
    emt += r["tokens"]
  return emt, r["pos_new"]


# ---------- dbg8: acceptance dump (replaces tier1/timing/phase) ----------
E.reset_snapshot(SNAP, CUR0, P0)
mtot = 0
for k in range(10):
  r = sess.step(); mtot += r["m"]
  if k < 3 or k == 9:
    print(f"[dbg8] cyc{k}: m={r['m']} amds={E.P.down_at('amds', 0, 3).tolist()} "
          f"dring=({int(E.P.down_at('dring0', 0, 1)[0])},{int(E.P.down_at('dring1', 0, 1)[0])})", flush=True)
print(f"[dbg8] sum(m)={mtot} alpha={mtot/20:.3f} -> {'WORKS' if mtot >= 12 else 'BROKEN'}", flush=True)
print("[done]", flush=True)
