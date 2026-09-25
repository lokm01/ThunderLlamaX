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
  # R3 LOOKUP: reset_snapshot's _mfill wipes tok_hist to -1 — re-seed the fed
  # prompt ids (the n-gram drafter's history; accept.cu appends decode tokens).
  if os.getenv("LOOKUP") == "1" or int(os.getenv("LOOKUP_K", "0") or 0) > 0:
    win_up(E.P, "tok_hist", 0, np.array(ids, dtype=np.int32))
    dev.synchronize()

def hist(E):
  return E.P.down("tok_hist", (CTXK + 256,), np.int32)

print("[w100k] loading engine (~13GB weights)...", flush=True)
t0 = time.perf_counter()
E = MTPEngine(theta=1e7)
print(f"[w100k] engine loaded {time.perf_counter()-t0:.1f}s", flush=True)
load100k(E)
vram_report(E)
dev.synchronize()

# ---------- R6 PHASE 3: batch banks + probe-scratch resize (BATCH_B>=2) ----
# MUST precede EVERY graph build (the fixed-handle law): the BT>RM resize
# replaces the probe-scratch P.d entries and the stream-1 banks must exist
# before any ParityGraph bakes handles. BATCH_B<2: byte-identical boot.
R6H = None
if int(os.getenv("BATCH_B", "1")) >= 2:
  import r6_serve
  R6H = r6_serve.r6_boot(E)
  vram_report(E)

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
  G = GCycleEngine(E)   # P7E3: build G without the T1@97810 decode (machine fault class — cached ref)
  G.build(); dev.synchronize()
  print("[ref] CACHED (T1@97810 decode skipped — P7E3 machine-fault workaround)", flush=True)
  G = None

# ---------- slice + draft ----------
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
# R6: optional extra token-id lists (comma-separated .npy paths) folded into
# the boot draft slice — batch gates use it for deterministic draft coverage
# of their conversation prompts (chain proposals are slice-limited; lookup
# proposals are not).
for _p in [x for x in os.getenv("R6_SLICE_EXTRA", "").split(",") if x.strip()]:
  for t in [int(v) for v in np.load(_p)]:
    if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
sl = sl[:SLICE]
print(f"[slice] {len(set(sl))} distinct ids; ref covered: {sum(1 for t in ref if t in set(sl))}/60", flush=True)
E.init_draft(sl)

if os.getenv("PF_GATE18") == "1":
  import p18_attr
  p18_attr.run(E, ids, CTXK)
  raise SystemExit(0)

E.fill_draft(ids)          # ONCE (~5 min at 100k)
E.build_graphs()
dev.synchronize()

# R4 ROWCHK: probe (T=3) vs probe5 (T=5) on identical reset state — rows 0..2
# must be BIT-IDENTICAL (per-row op order claim). Splits the bug space in half.
# R7 DECIDER 3: deep-cycle phase attribution (timing-only; partial graphs poison
# GDN live — NO gates after). Env-gated, default off (byte-identical when unset).
if os.getenv("R7_D3", "0") == "1":
  import r7_d3_attr
  r7_d3_attr.run(E, reset_spec=reset_spec, CUR0=CUR0, P0=P0)
  raise SystemExit(0)

if os.getenv("R4_ROWCHK", "0") == "1":
  import mtp as _mtp
  def run_probe(deep):
    reset_spec(E)
    for nm in ("dring0", "dring1", "dring2", "dring3", "dring4", "dring5", "dring6", "dring7", "dring8", "dring9"):   # valid ids (draft graph not run here)
      win_up(E.P, nm, 0, np.array([int(CUR0)], dtype=np.int32))
    dev.synchronize()
    prev = dev.timeline_value - 1
    dset = (E.graphs6 if int(os.getenv("LOOKUP_K", "0") or 0) >= 5 else E.graphs5) if deep else E.graphs
    pg = dset[1]
    fg = dset[3]
    v1 = dev.next_timeline(); pg.submit(prev, v1)
    v2 = dev.next_timeline(); fg.submit(v1, v2)
    dev.timeline_signal.wait(v2)
    amds = E.P.down_at("amds", 0, 5, np.int32)
    xA = E.P.down_at("xA", 0, 3*5120, np.float32)   # rows 0..2 of the pre-head hidden (post-final-write parity may differ; compare content)
    return amds, xA
  a3, x3 = run_probe(0)
  a5, x5 = run_probe(1)
  print(f"[rowchk] amds T3 {a3.tolist()}", flush=True)
  print(f"[rowchk] amds T5 {a5.tolist()}", flush=True)
  print(f"[rowchk] rows0-2 amds equal: {bool((a3[:3] == a5[:3]).all())}", flush=True)
  d = np.abs(x3 - x5); rel = d / np.maximum(np.abs(x3), 1e-6)
  print(f"[rowchk] xA rows0-2 maxabs {d.max():.3e} relmax {rel.max():.3e} frac-neq {float((x3 != x5).mean()):.4f}", flush=True)
  raise SystemExit(0)

# P0 stale-feed repro (R7B_DECIDERS §7): successive DIFFERENT-content prefill_batch
# calls (the same-content gates cannot see a stale ids feed). Default-off.
if os.getenv("P0_REPRO", "0") == "1":
  import p0_repro
  p0_repro.run(E, G, None, None, ids, CTXK)
  raise SystemExit(0)

def spec_run(ncyc=NTOK, phase_times=False):
  pass  # reset_spec now done inline below
  r = E.run_cycles(ncyc, sync_every=SYNC_EVERY, phase_times=phase_times)
  h = hist(E)
  out = h[P0:P0+NTOK]
  m = E.P.down("m_hist", (1024,), np.int32)
  return out, m, r

sess = DecodeSession(E)   # M1-A serving core drives every spec run below
R4LOG = []                # R4: per-cycle (m, hit, deep) for the deep-K instrumentation

def decode_n(n=NTOK):
  """decode n cycles via DecodeSession; returns (emitted_tokens, pos_end)."""
  sess.begin()
  emt = []
  for _ in range(n):
    deep = getattr(sess, "deep", 0)
    r = sess.step()
    if int(os.getenv("LOOKUP_K", "0") or 0) > 0:
      R4LOG.append((r["m"], r["hit"], deep))
    emt += r["tokens"]
  return emt, r["pos_new"]

# R5 TRACE: per-cycle ROW-level exactness discriminator. After each cycle,
# download amds[0..4] + dring0..3 and compare every EMITTED row (t = 0..m, the
# bonus INCLUDED) against the T=1 ref at stream index (pos_start-P0)+t. The
# FIRST mismatching row t splits the bug space sharply (rows 0..2 of the T=5
# set are bit-identical to the Tier-1-proven T=3 rows on identical state, and
# the deep=off gate proves T=3 exact on real data):
#   t <= 2  -> the state ENTERING the cycle was corrupt (post-deep-accept
#              state/KV bug — rec/conv slot select, conv5x copy, pre5 KV write)
#   t >= 3  -> the row-3/4-only path (attention deep-row cap, k2s5 t=3/4,
#              head8v5 rows 3/4, amx3 grid) computed a wrong argmax.
if os.getenv("R4_TRACE", "0") == "1":
  import mtp as _mtp
  refa = np.array(ref)
  for _mode in os.getenv("R4_TRACE_MODES", "sel,on").split(","):
    _mtp.DEEP_MODE = _mode
    reset_spec(E)
    sess.begin()
    print(f"[trace] === mode {_mode} ===", flush=True)
    bad_found = False
    for c in range(NTOK):
      deep_at_entry = int(getattr(sess, "deep", 0))
      r = sess.step()
      _lk = int(os.getenv("LOOKUP_K", "0") or 0)
      amds = E.P.down_at("amds", 0, 11 if _lk >= 10 else (10 if _lk >= 9 else (9 if _lk >= 8 else 8)), np.int32).tolist()
      d0 = int(E.P.down_at("dring0", 0, 1, np.int32)[0])
      d1 = int(E.P.down_at("dring1", 0, 1, np.int32)[0])
      d2 = int(E.P.down_at("dring2", 0, 1, np.int32)[0])
      d3 = int(E.P.down_at("dring3", 0, 1, np.int32)[0])
      LKX = int(os.getenv("LOOKUP_K", "0") or 0)
      d4 = int(E.P.down_at("dring4", 0, 1, np.int32)[0]) if LKX >= 5 else 0
      d5 = int(E.P.down_at("dring5", 0, 1, np.int32)[0]) if LKX >= 6 else 0
      d6 = int(E.P.down_at("dring6", 0, 1, np.int32)[0]) if LKX >= 7 else 0
      d7 = int(E.P.down_at("dring7", 0, 1, np.int32)[0]) if LKX >= 8 else 0
      d8 = int(E.P.down_at("dring8", 0, 1, np.int32)[0]) if LKX >= 9 else 0
      d9 = int(E.P.down_at("dring9", 0, 1, np.int32)[0]) if LKX >= 10 else 0
      m = int(r["m"]); pos = int(r["pos_new"]) - (m + 1)
      base = pos - P0
      badt = None
      for t in range(m + 1):
        if base + t < NTOK and amds[t] != refa[base + t]:
          badt = t; break
      hitf = int(r["hit"])
      line = (f"[trace] cyc{c:3d} deep={deep_at_entry} hit={hitf:>2} m={m} pos={pos} "
              f"amds={amds} dr=({d0},{d1},{d2},{d3},{d4},{d5},{d6},{d7},{d8},{d9})")
      if badt is not None:
        exp = [int(refa[base + t]) for t in range(m + 1)]
        line += (f"  ** ROW{badt} MISMATCH: amds[{badt}]={amds[badt]} != "
                 f"ref[{base+badt}]={int(refa[base+badt])}; exp_rows={exp}")
        print(line, flush=True)
        bad_found = True
        break
      print(line, flush=True)
    if not bad_found:
      print(f"[trace] mode {_mode}: ALL CYCLES CLEAN vs ref", flush=True)
  _mtp.DEEP_MODE = "sel"
  raise SystemExit(0)

# R5 DIF: ISOLATED single-probe differential at the snapshot position. Runs the
# T=5 probe with the TRUE greedy proposals (rows 0..4 = stream[P0..P0+4]), then
# re-anchors to P0+2 (acceptsel5k m=1: GDN slot1 -> live) and runs the
# BIT-PROVEN T=3 probe with the same absolute tokens (its rows 0..2 = stream
# [P0+2..P0+4]). Cross-checks amds + the pre-head hidden xA + logits:
#   T5 row 2/3/4  <->  T3 row 0/1/2   (identical math if the M=5 path is per-row exact)
# Splits trunk-vs-head: xA row mismatch => trunk (k2s5 t=3/4 / ROWS=5 attn);
# xA row equal but logits/argmax differ => head8v5 row 3/4.
if os.getenv("R4_DIF", "0") == "1":
  import mtp as _mtp
  from trunk import VOCAB as _VOCAB
  refa = np.array(ref)
  reset_spec(E)
  LKX = int(os.getenv("LOOKUP_K", "0") or 0)
  LK5 = LKX >= 5
  for nm, v in (("dring0", refa[0]), ("dring1", refa[1]), ("dring2", refa[2]), ("dring3", refa[3]),
                ("dring4", refa[4] if LK5 else 0), ("dring5", refa[5] if LKX >= 6 else 0),
                ("dring6", refa[6] if LKX >= 7 else 0), ("dring7", refa[7] if LKX >= 8 else 0),
                ("dring8", refa[8] if LKX >= 9 else 0), ("dring9", refa[9] if LKX >= 10 else 0)):
    win_up(E.P, nm, 0, np.array([int(v)], dtype=np.int32))
  dev.synchronize()
  prev = dev.timeline_value - 1
  v1 = dev.next_timeline(); (E.graphs6 if LK5 else E.graphs5)[1].submit(prev, v1)   # deep probe (graphs6 for K=5/6/7)
  v2 = dev.next_timeline(); (E.graphs6 if LK5 else E.graphs5)[3].submit(v1, v2)     # flusher
  dev.timeline_signal.wait(v2)
  NR = {0: 5, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}[LKX]
  amds5 = E.P.down_at("amds", 0, NR, np.int32).tolist()
  xA5 = E.P.down_at("xA", 0, NR*5120, np.float32).reshape(NR, 5120)
  lg5 = E.P.down_at("logits3", 0, NR*_VOCAB, np.float16).reshape(NR, _VOCAB).astype(np.float32)
  print(f"[dif] deep amds {amds5} vs ref[0..{NR-1}] {refa[:NR].tolist()}", flush=True)
  xh3rows = E.P.down_at("xh3", 0, NR*5120, np.float16).astype(np.float32).reshape(NR, 5120)
  print(f"[dif] xA row absmax: {[float(np.abs(xA5[t]).max()) for t in range(NR)]}", flush=True)
  print(f"[dif] xh3 row absmax: {[float(np.abs(xh3rows[t]).max()) for t in range(NR)]}", flush=True)
  # --- re-anchor to P0+2: GDN slot1 -> live (acceptselNK eager, m=1) ---
  win_up(E.P, "m_slot", 0, np.array([1], dtype=np.int32))
  dev.synchronize()
  if LKX == 10:
    E.pr["acceptsel11k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["conv9x"], E.P.d["conv10x"], E.P.d["conv11x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"], E.P.d["rec9x"], E.P.d["rec10x"], E.P.d["rec11x"],
                         global_size=(48, 1, 1), local_size=(256, 1, 1))
  elif LKX == 9:
    E.pr["acceptsel10k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["conv9x"], E.P.d["conv10x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"], E.P.d["rec9x"], E.P.d["rec10x"],
                         global_size=(48, 1, 1), local_size=(256, 1, 1))
  elif LKX == 8:
    E.pr["acceptsel9k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["conv9x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"], E.P.d["rec9x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))
  elif LKX == 7:
    E.pr["acceptsel8k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))
  elif LKX == 6:
    E.pr["acceptsel7k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["rec6x"], E.P.d["rec7x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))
  elif LK5:
    E.pr["acceptsel6k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["rec6x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))
  else:
    E.pr["acceptsel5k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))
  dev.synchronize()
  for nm, v in (("cur_slot", refa[1]), ("dring0", refa[2]), ("dring1", refa[3]),
                ("dring2", 0), ("dring3", 0), ("pos_slot", P0 + 2)):
    win_up(E.P, nm, 0, np.array([int(v)], dtype=np.int32))
  dev.synchronize()
  prev = dev.timeline_value - 1
  v3 = dev.next_timeline(); E.graphs[1].submit(prev, v3)    # K2 (T=3) probe
  v4 = dev.next_timeline(); E.graphs[3].submit(v3, v4)
  dev.timeline_signal.wait(v4)
  amds3 = E.P.down_at("amds", 0, 3, np.int32).tolist()
  xA3 = E.P.down_at("xA", 0, 3*5120, np.float32).reshape(3, 5120)
  lg3 = E.P.down_at("logits3", 0, 3*_VOCAB, np.float16).reshape(3, _VOCAB).astype(np.float32)
  print(f"[dif] T3@P0+2 amds {amds3} (rows 0..2 = stream[P0+2..P0+4] = {refa[2:5].tolist()})", flush=True)
  for t5, t3 in ((2, 0), (3, 1), (4, 2)):
    dxa = np.abs(xA5[t5] - xA3[t3])
    dlg = np.abs(lg5[t5] - lg3[t3])
    top5 = np.argsort(-lg5[t5])[:5]
    print(f"[dif] T5row{t5} vs T3row{t3}: amds {amds5[t5]} vs {amds3[t3]} (ref {int(refa[t5])}); "
          f"xA maxabs {dxa.max():.3e} relmax {(dxa/np.maximum(np.abs(xA3[t3]),1e-6)).max():.3e} "
          f"frac-neq {float((xA5[t5]!=xA3[t3]).mean()):.4f}; logits maxabs {dlg.max():.3e}; "
          f"T5 top5 {[(int(i), float(lg5[t5][i])) for i in top5]}", flush=True)
  raise SystemExit(0)

# R5 BISECT: find the exact kernel corrupting row 4. Method: the T5 probe at
# P0 (rows 0..4, live=snapshot) vs the T5 probe at P0+2 (rows 0..4, live=the
# full run's GDN slot 1) — row alignment t5(base) = t5(offset)+2 covers the
# same absolute stream positions. SAME kernel set both sides (isolates the
# row-4 column from the M3-vs-M5 variable). Binary search over BLOCK
# boundaries, then linear within the culprit block, comparing per-kernel
# output buffers (base row 4 vs offset row 2 must be BIT-IDENTICAL if the
# per-row claim holds; the first differing buffer at the earliest cut = culprit).
if os.getenv("R4_BISECT", "0") == "1":
  import mtp as _mtp
  from gcycle import ParityGraph
  refa = np.array(ref)
  qt = E.qtypes
  blen = [9 if i in qt else 7 for i in range(64)]
  bcuts = [1]
  for i in range(64): bcuts.append(bcuts[-1] + blen[i])
  seq5 = E._probe5_seq()
  assert len(seq5) == bcuts[-1] + 3, (len(seq5), bcuts[-1])
  fg = E.graphs5[3]

  def live_cache(slot):
    Rs, Cs = [], []
    for j in range(48):
      Rs.append(E.P.down_at("rec4", (j*5+slot)*RBLK*4, RBLK, np.float32))
      Cs.append(E.P.down_at("conv4", (j*5+slot)*CBLK*4, CBLK, np.float32))
    return Rs, Cs
  def live_set(L):
    Rs, Cs = L
    for j in range(48):
      win_up(E.P, "rec4", (j*5+4)*RBLK*4, Rs[j])
      win_up(E.P, "conv4", (j*5+4)*CBLK*4, Cs[j])
    dev.synchronize()
  def slots(base):
    vals = (("cur_slot", CUR0 if base else int(refa[1])), ("dring0", int(refa[0]) if base else int(refa[2])),
            ("dring1", int(refa[1]) if base else int(refa[3])), ("dring2", int(refa[2]) if base else 0),
            ("dring3", int(refa[3]) if base else 0), ("pos_slot", P0 if base else P0 + 2))
    for nm, v in vals: win_up(E.P, nm, 0, np.array([int(v)], dtype=np.int32))
    dev.synchronize()

  reset_spec(E)
  SNAP = live_cache(4)
  slots(True)
  prev = dev.timeline_value - 1
  v1 = dev.next_timeline(); E.graphs5[1].submit(prev, v1)
  v2 = dev.next_timeline(); fg.submit(v1, v2)
  dev.timeline_signal.wait(v2)
  print(f"[bisect] full T5@P0 amds {E.P.down_at('amds', 0, 5, np.int32).tolist()} (ref {refa[:5].tolist()})", flush=True)
  S1 = live_cache(1)

  _GCache = {}
  def run_partial(cent, base):
    if cent not in _GCache: _GCache[cent] = ParityGraph(seq5[:cent], tag=f"r4b{cent}")
    g = _GCache[cent]
    live_set(SNAP if base else S1)
    slots(base)
    prev = dev.timeline_value - 1
    va = dev.next_timeline(); g.submit(prev, va)
    vb = dev.next_timeline(); fg.submit(va, vb)
    dev.timeline_signal.wait(vb)

  # (name, dtype, row_stride_elems) — row 4 (base) vs row 2 (offset)
  BUFS = [("xh3", np.float16, 5120), ("araw3", np.float32, 48), ("braw3", np.float32, 48),
          ("qkv3", np.float16, 10240), ("gate3", np.float16, 6144), ("z3", np.float16, 6144),
          ("attn_out3", np.float16, 5120), ("hh3b", np.float32, 5120), ("hhx3", np.float16, 5120),
          ("gact3", np.float16, 17408), ("qrow3", np.float16, 12288), ("krow3", np.float16, 1024),
          ("vrow3", np.float16, 1024), ("ao_row3", np.float16, 6144),
          ("qw3", np.float32, 24*256), ("qw16_3", np.float16, 24*256),
          ("xA", np.float32, 5120), ("xB", np.float32, 5120)]
  OUT_GDN = [["xh3","araw3","braw3"], ["qkv3","gate3"], ["z3"], ["attn_out3"], ["hh3b","hhx3"], ["gact3"], ["__XOUT"]]
  OUT_ATTN = [["xh3"], ["qrow3","krow3","vrow3"], ["qw3","qw16_3"], ["__PMPS"], ["ao_row3"], ["attn_out3"], ["hh3b","hhx3"], ["gact3"], ["__XOUT"]]

  def written(cent):
    """buffers written by kernels < cent, per dataflow map; None row = trunk parity buf."""
    nb = sum(1 for i in range(64) if bcuts[i+1] <= cent)
    b = 0; rem = cent; out = set()
    while rem > 1 and b < 64:
      take = min(blen[b], rem - 1)
      for ol in (OUT_ATTN if b in qt else OUT_GDN)[:take]:
        for o in ol: out.add(o)
      rem -= take; b += 1
    out.discard("__PMPS"); out.discard("__XOUT")
    buf = "xB" if (nb % 2 == 1) else "xA"
    if nb > 0: out.add(buf)
    return out

  def snap(cent, row):
    out = written(cent)
    A = {}
    for nm, dt, stride in BUFS:
      if nm not in out: continue
      A[nm] = E.P.down_at(nm, row*stride*np.dtype(dt).itemsize, stride, dt)
    return A

  def compare(cent):
    run_partial(cent, True)
    A = snap(cent, 4)
    run_partial(cent, False)
    B = snap(cent, 2)
    return [nm for nm in A if not np.array_equal(A[nm], B[nm])]

  lo, hi = 0, 64
  print("[bisect] block-level binary search", flush=True)
  while hi - lo > 1:
    mid = (lo + hi) // 2
    bad = compare(bcuts[mid])
    print(f"[bisect] blocks={mid} (last block {mid-1} {'attn' if (mid-1) in qt else 'gdn'}): diff {bad}", flush=True)
    if bad: hi = mid
    else: lo = mid
  b = hi - 1
  print(f"[bisect] culprit block = {b} ({'attn' if b in qt else 'gdn'}) — within-block scan", flush=True)
  start = bcuts[b]
  outs = OUT_ATTN if b in qt else OUT_GDN
  for k in range(1, len(outs) + 1):
    cent = start + k
    bad = compare(cent)
    kn = str(seq5[start+k-1][0])
    print(f"[bisect]   +kernel{k} ({kn}): diff {bad}", flush=True)
  raise SystemExit(0)

# R5 NAN: find the first block whose output NaNs rows 2..5 of the K=5 deep
# probe (partial graphs at block boundaries; read the trunk parity buffer).
if os.getenv("R5_NAN", "0") == "1":
  import mtp as _mtp
  from gcycle import ParityGraph
  refa = np.array(ref)
  qt = E.qtypes
  blen = [9 if i in qt else 7 for i in range(64)]
  cuts = [1]
  for i in range(64): cuts.append(cuts[-1] + blen[i])
  seq6 = E._probe6_seq()
  fg = E.graphs6[3]
  for nm, v in (("dring0", refa[0]), ("dring1", refa[1]), ("dring2", refa[2]),
                ("dring3", refa[3]), ("dring4", refa[4])):
    win_up(E.P, nm, 0, np.array([int(v)], dtype=np.int32))
  win_up(E.P, "cur_slot", 0, np.array([int(CUR0)], dtype=np.int32))
  win_up(E.P, "pos_slot", 0, np.array([int(P0)], dtype=np.int32))
  dev.synchronize()
  # LAW (from R4_BISECT): every partial re-anchors the GDN live state — k2s6's
  # t=4 rec write LANDS IN LIVE slot 4, so chained partials poison each other.
  def live_cache(slot):
    Rs, Cs = [], []
    for j in range(48):
      Rs.append(E.P.down_at("rec4", (j*5+slot)*RBLK*4, RBLK, np.float32))
      Cs.append(E.P.down_at("conv4", (j*5+slot)*CBLK*4, CBLK, np.float32))
    return Rs, Cs
  def live_set(L):
    for j in range(48):
      win_up(E.P, "rec4", (j*5+4)*RBLK*4, L[0][j])
      win_up(E.P, "conv4", (j*5+4)*CBLK*4, L[1][j])
    dev.synchronize()
  reset_spec(E)
  for nm, v in (("dring0", refa[0]), ("dring1", refa[1]), ("dring2", refa[2]),
                ("dring3", refa[3]), ("dring4", refa[4])):
    win_up(E.P, nm, 0, np.array([int(v)], dtype=np.int32))
  win_up(E.P, "cur_slot", 0, np.array([int(CUR0)], dtype=np.int32))
  win_up(E.P, "pos_slot", 0, np.array([int(P0)], dtype=np.int32))
  dev.synchronize()
  SNAP = live_cache(4)
  for k in range(0, 65):
    live_set(SNAP)
    g = ParityGraph(seq6[:cuts[k]] if k else seq6[:1], tag=f"r5n{k}")
    prev = dev.timeline_value - 1
    v1 = dev.next_timeline(); g.submit(prev, v1)
    v2 = dev.next_timeline(); fg.submit(v1, v2)
    dev.timeline_signal.wait(v2)
    buf = "xB" if (k % 2 == 1) else "xA"
    row = E.P.down_at(buf, 5*5120*4, 5120, np.float32)
    bad = int(np.isnan(row).sum()) + int(np.isinf(row).sum())
    mx = float(np.abs(row[~np.isnan(row)]).max()) if (~np.isnan(row)).any() else 0.0
    if bad or k in (0, 1, 2) or k == 64:
      print(f"[nan] blocks={k} ({'attn' if (k-1) in qt else 'gdn'} if k else 'embed-only') row5: bad={bad} absmax={mx}", flush=True)
    if bad:
      for t in range(6):
        r = E.P.down_at(buf, t*5120*4, 5120, np.float32)
        print(f"[nan]   row{t}: nan={int(np.isnan(r).sum())} inf={int(np.isinf(r).sum())}", flush=True)
      # within-block kernel scan (R5): find the exact kernel introducing NaN
      start = cuts[k-1]
      outs = {2: "qrow3 krow3 vrow3", 3: "qw3 qw16_3", 4: "pm3 ps3 pA3", 5: "ao_row3", 6: "attn_out3", 7: "hh3b hhx3", 8: "gact3", 9: buf}
      for kk in range(1, 10):
        live_set(SNAP)
        g = ParityGraph(seq6[:start+kk], tag=f"r5w{kk}")
        prev = dev.timeline_value - 1
        v1 = dev.next_timeline(); g.submit(prev, v1)
        v2 = dev.next_timeline(); fg.submit(v1, v2)
        dev.timeline_signal.wait(v2)
        stats = []
        for nm, dt, stride, rws in (("qrow3", np.float16, 12288, 6), ("krow3", np.float16, 1024, 6), ("vrow3", np.float16, 1024, 6),
                                    ("qw3", np.float32, 24*256, 6), ("qw16_3", np.float16, 24*256, 6), ("pm3", np.float32, 4*256*36, 6),
                                    ("ps3", np.float32, 4*256*36, 6), ("ao_row3", np.float16, 6144, 6), ("attn_out3", np.float16, 5120, 6),
                                    ("hh3b", np.float32, 5120, 6), ("gact3", np.float16, 17408, 6), ("xA", np.float32, 5120, 6), ("xB", np.float32, 5120, 6)):
          nan_by_row = []
          for t in range(rws):
            a = E.P.down_at(nm, t*stride*np.dtype(dt).itemsize if nm not in ("pm3", "ps3") else 0, stride if nm in ("pm3", "ps3") else min(stride, 4096), dt).astype(np.float32)
            nan_by_row.append(int(np.isnan(a).sum()))
          stats.append(f"{nm}:r{nan_by_row}")
        print(f"[nan-w] +k{kk} ({outs.get(kk,chr(63))}): " + " ".join(stats), flush=True)
      break
  raise SystemExit(0)

# R5 DET: determinism probe — submit the SAME deep-probe graph N times (live
# re-anchored between) and diff the outputs. Flickering NaN/garbage = a RACE in
# the M=6 set; stable-wrong = deterministic bug.
if os.getenv("R5_DET", "0") == "1":
  import mtp as _mtp
  from gcycle import ParityGraph
  refa = np.array(ref)
  for nm, v in (("dring0", refa[0]), ("dring1", refa[1]), ("dring2", refa[2]),
                ("dring3", refa[3]), ("dring4", refa[4])):
    win_up(E.P, nm, 0, np.array([int(v)], dtype=np.int32))
  win_up(E.P, "cur_slot", 0, np.array([int(CUR0)], dtype=np.int32))
  win_up(E.P, "pos_slot", 0, np.array([int(P0)], dtype=np.int32))
  dev.synchronize()
  def live_cache(slot):
    Rs, Cs = [], []
    for j in range(48):
      Rs.append(E.P.down_at("rec4", (j*5+slot)*RBLK*4, RBLK, np.float32))
      Cs.append(E.P.down_at("conv4", (j*5+slot)*CBLK*4, CBLK, np.float32))
    return Rs, Cs
  def live_set(L):
    for j in range(48):
      win_up(E.P, "rec4", (j*5+4)*RBLK*4, L[0][j])
      win_up(E.P, "conv4", (j*5+4)*CBLK*4, L[1][j])
    dev.synchronize()
  SNAP = live_cache(4)
  fg = E.graphs6[3]
  seq6 = E._probe6_seq()
  g_full = ParityGraph(seq6, tag="r5det")
  _c4 = 1 + sum(9 if i in E.qtypes else 7 for i in range(4))
  g_b4 = ParityGraph(seq6[:_c4], tag="r5det4")
  for tag, g in (("full", g_full), ("b4", g_b4)):
    print(f"[det] === graph {tag} ===", flush=True)
    sigs = []
    for it in range(4):
      live_set(SNAP)
      prev = dev.timeline_value - 1
      v1 = dev.next_timeline(); g.submit(prev, v1)
      v2 = dev.next_timeline(); fg.submit(v1, v2)
      dev.timeline_signal.wait(v2)
      amds = E.P.down_at("amds", 0, 6, np.int32).tolist()
      buf = "xB"
      rows = []
      for t in range(6):
        r = E.P.down_at("xA", t*5120*4, 5120, np.float32)
        rows.append((int(np.isnan(r).sum()), float(np.abs(r[~np.isnan(r)]).max()) if (~np.isnan(r)).any() else -1))
      sig = (tuple(amds), tuple(rows))
      print(f"[det] {tag} it{it}: amds={amds} xA-rows(nan,absmax)={rows}", flush=True)
      sigs.append(sig)
    print(f"[det] {tag} deterministic: {all(s == sigs[0] for s in sigs)}", flush=True)
  raise SystemExit(0)

# R5 QUOTE: the quote-heavy workload IN VIVO (the offline c-sim class: the doc
# prompt + a verbatim 60-token quote of a random doc span, rng(0) — the same q0
# as lut_deepk.py). follow_up feeds [cur]+quote via the PROVEN T=1 path and
# bootstraps new_cur; the decode then continues the quote (the lookup sees the
# quoted span verbatim in tok_hist). Exactness: T=1 greedy reference over the
# same fed stream, then spec decode + emit==hist + determinism x2.
if os.getenv("R5_QUOTE", "0") == "1":
  import mtp as _mtp
  rng = np.random.default_rng(0)
  q0 = int(rng.integers(1000, len(ids) - 2000))
  if os.getenv("R8_PROSE", "0") == "1":
    # R8: the PROSE-class workload — feed the re-encoded novel reply (r8_corpus.py)
    # as the delta; the decode continues novel prose (0 expected lookup hits).
    delta = [int(t) for t in np.load("~/r8_prose_ids.npy")[:120]]
    print(f"[quote] PROSE delta (novel reply) {len(delta)} toks {delta[:8]}...", flush=True)
  else:
    delta = [int(t) for t in ids[q0:q0+60]]
    print(f"[quote] span q0={q0} quote tokens {delta[:8]}...", flush=True)

  Gq = GCycleEngine(E); Gq.build(); dev.synchronize()   # P7E3 nulls G in DO_T1=0 mode
  def quote_anchor():
    reset_spec(E)
    new_cur, pos_new, n_fed = E.follow_up(Gq, delta)
    fed = [CUR0] + delta
    win_up(E.P, "tok_hist", P0*4, np.array(fed, dtype=np.int32))   # the fed stream = lookup history
    dev.synchronize()
    return new_cur, pos_new, len(fed)

  # ---- T=1 greedy reference over the same fed stream ----
  new_cur, pos_q, n_fed = quote_anchor()
  win_up(E.P, "tok_slot", 0, np.array([int(new_cur)], dtype=np.int32))
  dev.synchronize()
  pb = int(E.P.down_at("pos_slot", 0, 1, np.int32)[0])
  print(f"[quote] follow_up pos_new={pos_q} pos_slot={pb} n_fed={n_fed} new_cur={new_cur}", flush=True)
  t0 = time.perf_counter()
  Gq.run_tokens(NTOK, wait_each=True)
  print(f"[quote] T=1 ref {NTOK} toks in {time.perf_counter()-t0:.1f}s", flush=True)
  qref = E.P.down_at("tok_hist", pb*4, NTOK, np.int32)
  assert all(t >= 0 for t in qref), "quote T=1 ref incomplete"
  print(f"[quote] ref[:16] {qref[:16].tolist()}", flush=True)

  # ---- spec decode (LOOKUP_K set) x2 + timing ----
  outs = []
  for rep in range(2):
    new_cur, pos_q, n_fed = quote_anchor()
    sess.begin()
    t0 = time.perf_counter()
    emt, pend = decode_n(NTOK)
    dt = time.perf_counter() - t0
    h = hist(E)
    base = pb   # the spec cycles start at the same re-anchored pos
    outw = h[base:base+NTOK]
    # the first emitted token is the prediction after the LAST FED token = new_cur
    # -> compare against [new_cur-follow-on]: ref window starts at qref[0] iff
    # new_cur == qref_pos... simplest exact contract: outw[0] must equal new_cur
    # fed onward; use the T=1 stream anchored at the same fed prefix:
    agree = int((outw == qref).sum())
    fd = next((k for k in range(NTOK) if outw[k] != qref[k]), None)
    print(f"[quote] rep{rep}: {agree}/{NTOK} exact vs T=1 (first div {fd}); emitted {len(emt)} toks, "
          f"{dt*1e3/NTOK:.2f} ms/tok-stream", flush=True)
    if agree < NTOK:
      print(f"[quote] out: {outw[:32].tolist()}")
      print(f"[quote] ref: {qref[:32].tolist()}")
    outs.append(outw.copy())
  print(f"[quote] deterministic across reps: {bool((outs[0] == outs[1]).all())}", flush=True)
  lh = E.P.down("l_hist", (4096,), np.int32)
  mh = E.P.down("m_hist", (4096,), np.int32)
  ncy = int((lh > 0).sum())
  hitm = (lh >= 9)
  print(f"[quote] lookup cycles {ncy}, hits {int(hitm.sum())} ({100.0*hitm.sum()/max(ncy,1):.1f}%), "
        f"E[m|hit] {float(mh[hitm].mean()) if hitm.any() else 0:.3f}, "
        f"tok/cyc {float((mh[:ncy]+1).mean()):.2f}", flush=True)
  raise SystemExit(0)

# P3 prefill-assembly gates (PF_GATE=1): batched M=16 prefill vs T=1, inside the
# proven host process (HOST-PROCESS BOOT LAW — a hand-rolled host faults/collapses).
if os.getenv("PF_GATE100K") == "1":
  import pf_gate100k
  pf_gate100k.run_gates(E, G, sess, ref, ids, CTXK)
  raise SystemExit(0)

if os.getenv("PF_GATE") == "1":
  import pf_gate2k
  pf_gate2k.run_gates(E, G, sess, ref, ids, CTXK)
  raise SystemExit(0)

# R1 prompt-cache gates (PC_GATE=1): run BEFORE the serve attach (GPU
# exclusive; the daemon must be down). Full harness env applies.
if os.getenv("PC_GATE") == "1":
  import pcache_gate
  pcache_gate.run_gates(E, G, sess, ref, ids, CTXK)
  raise SystemExit(0)

# M1-A serve-boot handoff: when this file is the HOST process of the daemon
# (M1A_SERVE=1 — the only launch mode proven to keep draft acceptance; the boot
# order AND the host-script identity are LOAD-BEARING, see M1A_SERVING.md),
# attach the daemon right here, after build_graphs. Also supports an explicit
# shuttle for other hosts (kept for debugging only — NOT acceptance-proven).
import builtins as _b
if os.getenv("M1A_SERVE") == "1":
  import serve
  serve.run_daemon(E=E, G=G, sess=sess, decode_n=decode_n, ref=ref, ids=ids,
                   P0=P0, CUR0=CUR0, SNAP=SNAP, CTXK=CTXK, r6h=R6H)
  raise SystemExit(0)
if getattr(_b, "_M1A_BOOT", None) is not None:
  _b._M1A_BOOT.update(E=E, G=G, sess=sess, decode_n=decode_n, ref=ref, ids=ids,
                      P0=P0, CUR0=CUR0, SNAP=SNAP, CTXK=CTXK, r6h=R6H)
  print("[w100k] M1A serve-boot handoff (E, G, sess)", flush=True)
  raise SystemExit(0)

# ---------- Tier-1: exactness x2 (emit-record path, verified vs tok_hist) ----------
outs = []
for rep in range(2):
  reset_spec(E)
  emt, pend = decode_n(NTOK)
  h = hist(E); out = h[P0:P0+NTOK]
  m = E.P.down("m_hist", (1024,), np.int32)
  agree = int((out == np.array(ref)).sum())
  first_div = next((k for k in range(NTOK) if out[k] != ref[k]), None)
  emit_ok = emt[:NTOK] == out.tolist()   # emits are the FULL stream (sum(m+1) per cycle)
  print(f"[tier1] rep{rep}: {agree}/{NTOK} exact, first divergence at {first_div}; "
        f"emit[:NTOK]==hist {emit_ok} ({len(emt)} emitted, pos_end {pend})", flush=True)
  if agree < NTOK:
    print(f"[tier1] out: {out.tolist()}")
    print(f"[tier1] ref: {ref}")
  outs.append(out.copy())
print(f"[tier1] deterministic across reps: {bool((outs[0] == outs[1]).all())}", flush=True)

# ---------- R3/R4 LOOKUP instrumentation (last rep's cycles) ----------
if os.getenv("LOOKUP") == "1" or int(os.getenv("LOOKUP_K", "0") or 0) > 0:
  lh = E.P.down("l_hist", (4096,), np.int32)
  mh = E.P.down("m_hist", (4096,), np.int32)
  ncy = int((lh > 0).sum())
  hitm = (lh >= 9); missm = (lh > 0) & ~hitm
  em_h = float(mh[hitm].mean()) if hitm.any() else 0.0
  em_m = float(mh[missm].mean()) if missm.any() else 0.0
  print(f"[lookup] cycles {ncy}, hit(l>=8/LMIN) {int(hitm.sum())} ({100.0*hitm.sum()/max(ncy,1):.1f}%), "
        f"E[m|hit] {em_h:.3f}, E[m|miss] {em_m:.3f}, l-dist(1..9) {[int((lh==v).sum()) for v in range(1,10)]}", flush=True)
  if int(os.getenv("LOOKUP_K", "0") or 0) > 0 and R4LOG:
    import numpy as _np
    L = _np.array(R4LOG, dtype=_np.int64)          # (m, hit, deep-selected)
    dl = L[:, 2] == 1
    print(f"[deepk] cycles {len(L)}, deep-selected {int(dl.sum())} ({100.0*dl.mean():.1f}%), "
          f"E[m|deep] {float(L[dl,0].mean()) if dl.any() else 0:.3f}, E[m|k2] {float(L[~dl,0].mean()) if (~dl).any() else 0:.3f}, "
          f"m-dist(deep) {[int((L[dl,0]==v).sum()) for v in range(int(os.getenv('LOOKUP_K','0') or 0)+1)]}, hits {int((L[:,1]>=9).sum())}", flush=True)
  # R4 DIAG: force deep OFF (isolates lookup5+acceptk on the K2 path) then
  # deep ON (isolates the T=5 set) — the exactness discriminator.
  if int(os.getenv("LOOKUP_K", "0") or 0) > 0 and os.getenv("R4_DIAG", "1") == "1":
    import mtp as _mtp
    refa = np.array(ref)
    for mode in ("off", "on"):
      _mtp.DEEP_MODE = mode
      reset_spec(E)
      t0 = time.perf_counter()
      emt, pend = decode_n(NTOK)
      dt = time.perf_counter() - t0
      h = hist(E); out = h[P0:P0+NTOK]
      agree = int((out == refa).sum())
      fd = next((k for k in range(NTOK) if out[k] != refa[k]), None)
      print(f"[r4diag] deep={mode}: agree {agree}/{NTOK}, first div {fd}, {dt/NTOK*1e3:.2f} ms/cyc, "
            f"tok/s {float(((E.P.down('m_hist', (1024,), np.int32)[:NTOK]+1).sum())/NTOK)/(dt/NTOK):.2f}", flush=True)
    _mtp.DEEP_MODE = "sel"

# ---------- Tier-2 overlap vs the fp16-KV sequence ----------
if KV8:
  for tag, rf in (("fp16-KV(W2D)", "engine_t1_ref.npy"), ("int8-KV(W2E)", "engine_t1_ref_kv8.npy")):
    try:
      ref16 = np.load(f"{SNAP}/{rf}").tolist()
      ov = int((outs[0] == np.array(ref16)).sum())
      fd = next((k for k in range(NTOK) if outs[0][k] != ref16[k]), None)
      print(f"[tier2] this-run vs {tag} T=1 sequence overlap: {ov}/{NTOK}, first divergence at {fd}", flush=True)
    except Exception as e:
      print(f"[tier2] {tag} overlap unavailable: {e}", flush=True)

# ---------- speed: 3 reps ----------
times = []
for rep in range(3):
  reset_spec(E)
  t0 = time.perf_counter()
  emt, _ = decode_n(NTOK)
  dt = time.perf_counter() - t0
  h = hist(E); out = h[P0:P0+NTOK]
  times.append(dt)
  m2 = E.P.down("m_hist", (1024,), np.int32)
  acc_pos = float(m2[:NTOK].sum()) / (2.0*NTOK)
  tpc = float((m2[:NTOK] + 1).sum()) / NTOK
  agree = int((out == np.array(ref)).sum())
  print(f"[time] rep{rep}: {dt*1e3:.1f} ms/60cyc = {dt/NTOK*1e3:.2f} ms/cyc, agree {agree}/{NTOK}, alpha={acc_pos:.3f}, tok/cyc={tpc:.2f}, tok/s={tpc/(dt/NTOK):.2f}", flush=True)
best = min(times)
m3 = E.P.down("m_hist", (1024,), np.int32)
tpc = float((m3[:NTOK] + 1).sum()) / NTOK
print(f"[time] BEST {best/NTOK*1e3:.2f} ms/cyc -> {tpc/(best/NTOK):.2f} tok/s", flush=True)

# ---------- phase breakdown ----------
seed_mtp_slots(E)
win_up(E.P, "cur_slot", 0, np.array([CUR0], dtype=np.int32))
win_up(E.P, "pos_slot", 0, np.array([P0], dtype=np.int32))
win_up(E.P, "m_hist", 0, np.zeros(1024, dtype=np.int32))
win_up(E.P, "cyc_slot", 0, np.zeros(1, dtype=np.int32))
dev.synchronize()
pt = E.run_cycles(NTOK, sync_every=SYNC_EVERY, phase_times=True)
td, tp, ta = pt
print(f"[phase] draft {td/NTOK*1e3:.2f} ms/cyc, probe {tp/NTOK*1e3:.2f}, accept {ta/NTOK*1e3:.2f}", flush=True)

# ---------- stock cross-check ----------
try:
  sb = np.array(json.load(open("~/tinygrad-metal/spec_base_100k.json")), dtype=np.int64)
  agree = int((np.array(ref[:NTOK-1]) == sb[1:NTOK]).sum())
  print(f"[stock] engine T=1[0:59] vs spec_base_100k[1:60]: {agree}/{NTOK-1} (first engine tok = pred P+1)", flush=True)
except Exception as e:
  print(f"[stock] cross-check unavailable: {e}", flush=True)
print("[done]", flush=True)
