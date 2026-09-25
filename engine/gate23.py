# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-A GATES 2+3 (one boot; canonical 100k Tier-1 config).
GATE 2(a): Tier-1 exact 60/60 x2 through FRESH reset -> snapshot-load -> decode.
GATE 2(b): reset path x5 consecutive, VRAM flat (arithmetic sum + alloc count + RSS).
GATE 3: two-turn conversation reuse exactness:
  RESIDENT  : snapshot -> decode turn1 (60) -> FOLLOW-UP [cur + 200 delta] -> decode turn2 (40) => A
  FRESH-FULL: snapshot -> one-shot T=1 prefill [cur0 + turn1(60) + 200 delta] -> decode turn2 => B
  A == B validates append-only state reuse end-to-end. Delta-prefill timing reported.
env: SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1."""
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

SNAP = os.getenv("SNAPDIR", "~/snap100k")
N1 = 60; N2 = 40; NDELTA = 200
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); assert CTXK == int(meta["CTXK"]) and TRUNK_CTX == CTXK
CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()
ref = np.load(f"{SNAP}/engine_t1_ref_kv8_qh.npy").tolist()
delta = ids[1000:1000+NDELTA]          # deterministic "turn-2 user tokens"
np.save(f"{SNAP}/gate3_delta.npy", np.array(delta, dtype=np.int64))
print(f"[g23] P={P0} cur0={CUR0} ref {len(ref)} delta {len(delta)}", flush=True)

def vram_report(E, tag):
  tot = 0; n = 0
  for name, b in E.P.d.items():
    nb = getattr(b, "nbytes", None)
    if nb is None:
      nb = getattr(b, "size", 0)
      try: nb = int(nb) * 4
      except: nb = 0
    tot += int(nb); n += 1
  fa = getattr(dev.allocator, "alloced", None) or getattr(dev.allocator, "_alloced", None)
  rss = int(os.popen("ps -o rss= -p %d" % os.getpid()).read().strip() or 0)
  print(f"[vram:{tag}] bufs {n} allocs, {tot/1e9:.3f} GB sum, alloced={fa}, rss={rss/1e6:.2f} GB", flush=True)
  return n, tot, rss

print("[g23] boot: engine (~13GB weights) ...", flush=True)
t0 = time.perf_counter()
E = MTPEngine(theta=1e7)
E.load_snapshot_kv(SNAP, progress=False)
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
print(f"[g23] boot done {time.perf_counter()-t0:.0f}s", flush=True)
vram_report(E, "boot")

sess = DecodeSession(E)
def decode_n(n):
  sess.begin(); emt = []
  for _ in range(n):
    r = sess.step(); emt += r["tokens"]
  return emt, r["pos_new"]

# ---------------- GATE 2(a): Tier-1 after FRESH reset x2 ----------------
g2a = True
for rep in range(2):
  E.reset_fresh(CUR0)                    # whole-engine FRESH (GDN zeros, slots zeroed)
  E.reset_snapshot(SNAP, CUR0, P0)       # prefill-snapshot-load (fixed handles, no rebuild)
  emt, pend = decode_n(N1)
  hfull = E.P.down("tok_hist", (CTXK + 256,), np.int32)
  out = hfull[P0:P0+N1]
  agree = int((out == np.array(ref)).sum())
  emit_ok = hfull[P0:P0+len(emt)].tolist() == emt   # full emitted stream == full hist window
  ok = agree == N1 and emit_ok
  g2a &= ok
  print(f"[gate2a] rep{rep}: {agree}/{N1} exact, emit==hist {emit_ok} ({len(emt)} emitted), pos_end {pend}", flush=True)
print(f"=== GATE 2(a) {'PASS' if g2a else 'FAIL'} ===", flush=True)

# ---------------- GATE 2(b): reset x5, VRAM flat ----------------
n0, t0b, r0b = vram_report(E, "pre5")
g2b = True
for rep in range(5):
  if rep % 2 == 0: E.reset_fresh(CUR0)
  else:            E.reset_snapshot(SNAP, CUR0, P0)
  emt, _ = decode_n(2)                   # real work between resets
  n1, t1b, r1b = vram_report(E, f"reset{rep}")
  flat = (n1, t1b) == (n0, t0b) and r1b <= r0b + 50_000   # bufs identical; RSS no growth (KB slop ok)
  g2b &= flat
  print(f"[gate2b] reset{rep}: vram flat={flat} (emitted {emt})", flush=True)
print(f"=== GATE 2(b) {'PASS' if g2b else 'FAIL'} ===", flush=True)

# ---------------- GATE 3: two-turn reuse exactness ----------------
# RESIDENT path
E.reset_snapshot(SNAP, CUR0, P0)
t1a, p1end = decode_n(N1)                # turn 1 = N1 CYCLES (emits sum(m+1) >= 60 tokens)
turn1 = t1a
assert int((np.array(turn1[:N1]) == np.array(ref)).sum()) == N1, "turn1 not exact — abort gate3"
print(f"[gate3] turn1: {len(turn1)} tokens from {N1} cycles, first {N1} exact, pos_end {p1end}", flush=True)
t0 = time.perf_counter()
newcur, posn, nd = E.follow_up(G, delta) # FOLLOW-UP: [cur] + 200 delta from resident pos
dt_fu = time.perf_counter() - t0
A, pA = decode_n(N2)                     # turn 2 (40)
print(f"[gate3] RESIDENT: follow_up {dt_fu:.1f}s ({dt_fu/nd*1e3:.0f} ms/tok for {nd} fed), "
      f"newcur {newcur}, pos {posn}; turn2[0:12] {A[:12]}", flush=True)

# FRESH-FULL path: same tokens prefilled in one go (T=1), then decode
E.reset_snapshot(SNAP, CUR0, P0)
feed = [CUR0] + turn1 + delta            # identical token stream as prompt+turn1+followup
dhd0 = np.zeros(5120, dtype=np.float32)
t0 = time.perf_counter()
E.fill_draft(feed, start_pos=P0, seed_hd=dhd0)
dt_dfill = time.perf_counter() - t0
E.stload_trunk()
t0 = time.perf_counter()
dt_t1 = E.prefill_t1(G, feed, log_every=1000)
E.stseed_spec(len(feed) & 1)
nc2 = int(E.P.down_at("tok_slot", 0, 1)[0])
E.P.win_up("cur_slot", 0, np.array([nc2], dtype=np.int32))
E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
dev.synchronize()
B, pB = decode_n(N2)
print(f"[gate3] FRESH-FULL: fill_draft {dt_dfill:.1f}s, T=1 prefill {dt_t1:.1f}s ({dt_t1/len(feed)*1e3:.0f} ms/tok), "
      f"cur {nc2}; turn2[0:12] {B[:12]}", flush=True)

nmin = min(len(A), len(B))
match = A[:nmin] == B[:nmin] and newcur == nc2
fd = next((k for k in range(nmin) if A[k] != B[k]), None)
print(f"[gate3] turn-2 stream match {sum(1 for k in range(nmin) if A[k] == B[k])}/{nmin} "
      f"(lenA {len(A)} lenB {len(B)}), first divergence {fd}; cur match {newcur == nc2}; "
      f"pos_end {pA} vs {pB} (equal {pA == pB})", flush=True)
np.save(f"{SNAP}/gate3_out_resident.npy", np.array(A, dtype=np.int64))
print(f"=== GATE 3 {'PASS' if match else 'FAIL'} ===", flush=True)

# 2k-class delta-prefill rate estimate (T=1 attention scales with ctx): report
print(f"[gate3] delta-prefill timing: 200-tok follow-up at 100k = {dt_fu:.1f}s total "
      f"({dt_fu/nd*1e3:.0f} ms/tok; draft-fill+T1+xfers). 2k-class expectation ~9s (T=1 rate ~45ms).", flush=True)
print("[done]", flush=True)
