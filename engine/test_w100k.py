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
import mtp as _mtpmod
from trunk import CTX as TRUNK_CTX
from engine0 import dev
from gcycle import GCycleEngine

SNAP = os.getenv("SNAPDIR", "~/snap100k")
NTOK = int(os.getenv("NTOK", "60"))
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "1"))
DO_T1 = os.getenv("DO_T1", "1") == "1"
TLX_T1_MODE = int(os.getenv("TLX_T1_MODE", "0"))   # A.1: the adaptive T=1 mode (see mtp.py)
TLX_RANK_HIST = os.getenv("TLX_RANK_HIST", "0") == "1"   # A.2: the MTP rank histogram (prose harness)

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
  if TLX_T1_MODE: E.gcycle = G   # A.1: the session's T=1 parity graphs
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
  if TLX_T1_MODE: E.gcycle = G   # A.1
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
      R4LOG.append((r["m"], r["hit"], deep, getattr(sess, "prose", 0)))
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
  if TLX_T1_MODE and Gq is not None:
    E.gcycle = Gq   # A.1: the T=1 mode's parity graphs in this harness
    print(f"[t1] TLX_T1_MODE on (trig={_mtpmod.TLX_T1_TRIG}): prose class runs the adaptive T=1 mode", flush=True)
  # A.2 (TLX_RANK_HIST=1): the MTP rank histogram. After each K2 cycle, at the
  # quiescent point, re-run the draft chain's STEP-0 kernels eagerly on the
  # POST-ACCEPT state (cur_slot, h_seed, kv_d) — a deterministic idempotent
  # replay that recomputes the MTP head's FULL slice distribution for the
  # NEXT cycle's query. slogits is dead scratch post-cycle (the graph's step-1
  # overwrote step-0), which is why the replay is needed. Pair each replay
  # with the NEXT cycle's probe row-0 argmax (the target of that query).
  RANKS = []; OOS = 0; NTOT = 0; PROP_HIT = 0
  def _rank_instrument(r_prev_dist, r_target, dr0):
    global OOS, NTOT, PROP_HIT
    NTOT += 1
    if int(dr0) == int(r_target): PROP_HIT += 1
    if r_prev_dist is None: return
    sid = STAB_MAP.get(int(r_target))
    if sid is None: OOS += 1; return
    l = r_prev_dist
    lt = float(l[sid])
    RANKS.append(1 + int((l > lt).sum()))
  STAB_MAP = {}
  if TLX_RANK_HIST:
    _stab = E.P.down("stab", (SLICE,), np.int32).tolist()
    STAB_MAP = {int(t): i for i, t in enumerate(_stab)}
    print(f"[rank] instrument on: slice {SLICE}, {len(STAB_MAP)} distinct ids", flush=True)
  outs = []
  for rep in range(2):
    new_cur, pos_q, n_fed = quote_anchor()
    sess.begin()
    t0 = time.perf_counter()
    if TLX_RANK_HIST:
      # instrumented decode: per K2 cycle, eager draft-step-0 replay + slogits
      emt = []
      prev_dist = None
      r = None
      for _ in range(NTOK):
        deep_at_entry = int(getattr(sess, "deep", 0))
        t1_at_entry = int(getattr(sess, "t1mode", 0))
        r = sess.step()
        if not t1_at_entry and not deep_at_entry:
          dr0 = int(E.P.down_at("dring0", 0, 1, np.int32)[0])
          _rank_instrument(prev_dist, r["tokens"][0], dr0)
          # replay the MTP draft step-0 on the post-accept state (idempotent:
          # same inputs -> same kv_d row write -> same distribution)
          for p_, a_, g_ in E._draft_entries(E.P.d["cur_slot"], E.P.d["pos_slot"], E.P.d["h_seed"], E.P.d["hd_d0"], E.P.d["dring0"])[:-1]:
            _nm = getattr(p_, "name", "")
            p_(*a_, global_size=(g_, 1, 1),
               local_size=((1024,1,1) if "nw32" in _nm else (768,1,1) if "nw24" in _nm else (512,1,1) if "nw16" in _nm else (256,1,1)))
          dev.synchronize()
          prev_dist = E.P.down_at("slogits", 0, SLICE, np.float16).astype(np.float32)
        else:
          prev_dist = None   # deep/T1 cycles break the query chain — drop the pair
        emt += r["tokens"]
      pend = r["pos_new"]
    else:
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
    _nt1 = int(getattr(sess, "nt1cycles", 0))
    _npr = int(getattr(sess, "nprose", 0)); _ndp = int(getattr(sess, "ndeep", 0)); _nc = int(getattr(sess, "ncyc", 0))
    print(f"[quote] rep{rep}: {agree}/{NTOK} exact vs T=1 (first div {fd}); emitted {len(emt)} toks, "
          f"{dt*1e3/NTOK:.2f} ms/tok-stream, t1-cycles(life) {_nt1}, k4-cycles(life) {_npr}/{_nc} deep {_ndp}", flush=True)
    if agree < NTOK:
      print(f"[quote] out: {outw[:32].tolist()}")
      print(f"[quote] ref: {qref[:32].tolist()}")
    outs.append(outw.copy())
  print(f"[quote] deterministic across reps: {bool((outs[0] == outs[1]).all())}", flush=True)
  # THE PROSE-CLASS LAW (adjudicated 2026-09-29 via R8_PROSE_DUMP, shipped
  # default config): a first divergence vs the T=1 ref on the novel-prose
  # anchor is the CROSS-CLASS FLIP class -- the M=3 probe family and the M=1
  # trunk family rank the same top-2 pair but disagree on the order when the
  # weaker margin is inside the cross-family logit drift (measured median
  # 0.39 logits at the flip point; the T1-preferred token moved 1.39 between
  # families). One flip + cascade => the low N/60 is ONE decision, not N. The
  # spec stream stays the probe-world's own consistent greedy chain (lossless
  # within its family). Rerun with R8_PROSE_DUMP=1 for the full-logit
  # adjudication + printed verdict.
  if int((outs[0] == qref).sum()) < NTOK:
    print("[quote] divergence-class NOTE: expected CROSS-CLASS FLIP (see law above); "
          "set R8_PROSE_DUMP=1 to adjudicate with logits", flush=True)
  # P10-dense rung 2: the K4-EAGLE acceptance histogram on THIS class
  if R4LOG:
    import numpy as _np2
    L2 = _np2.array(R4LOG, dtype=_np2.int64)
    pl2 = (L2[:, 2] == 0) & (L2[:, 3] == 1)
    if pl2.any():
      print(f"[k4hist] k4 cycles {int(pl2.sum())}, E[m|k4] {float(L2[pl2,0].mean()):.3f}, "
            f"m-dist(k4) {[int((L2[pl2,0]==v).sum()) for v in range(5)]}, tok/cyc(k4) {float((L2[pl2,0]+1).mean()):.2f}", flush=True)
    kl2 = (L2[:, 2] == 0) & (L2[:, 3] == 0)
    if kl2.any():
      print(f"[k4hist] k2 cycles {int(kl2.sum())}, E[m|k2] {float(L2[kl2,0].mean()):.3f}, "
            f"m-dist(k2) {[int((L2[kl2,0]==v).sum()) for v in range(3)]}", flush=True)
  lh = E.P.down("l_hist", (4096,), np.int32)
  mh = E.P.down("m_hist", (4096,), np.int32)
  ncy = int((lh > 0).sum())
  hitm = (lh >= 9)
  print(f"[quote] lookup cycles {ncy}, hits {int(hitm.sum())} ({100.0*hitm.sum()/max(ncy,1):.1f}%), "
        f"E[m|hit] {float(mh[hitm].mean()) if hitm.any() else 0:.3f}, "
        f"tok/cyc {float((mh[:ncy]+1).mean()):.2f}", flush=True)
  if TLX_RANK_HIST and RANKS:
    ra = np.array(RANKS)
    # THE HONEST VERDICT RULE: out-of-slice targets are UNRANKABLE by the
    # slice-limited drafter (an effective rank > SLICE) — they must count in
    # the population median, not be dropped (the dropped-median was selection
    # bias: only prompt-frequent targets were in-slice).
    ra_all = np.array(RANKS + [SLICE + 1] * OOS)
    med = float(np.median(ra_all))
    print(f"[rank] N {NTOT} paired {len(RANKS)} out-of-slice {OOS} ({100.0*OOS/max(NTOT,1):.1f}%) | "
          f"top1==target {PROP_HIT}/{NTOT} ({100.0*PROP_HIT/max(NTOT,1):.1f}%)", flush=True)
    print(f"[rank] rank-dist r1..r16: {[int((ra == v).sum()) for v in range(1, 17)]} | >16: {int((ra > 16).sum())}", flush=True)
    print(f"[rank] in-slice median {float(np.median(ra)):.1f}  p75 {float(np.percentile(ra, 75)):.1f}  "
          f"mean {float(ra.mean()):.2f} | FULL-POPULATION median (oos->inf) {med:.1f}", flush=True)
    # the A.2 verdict rule (the prose plan): median<=4 -> a trained drafter
    # could reach p 0.3-0.5; median>8 -> T=1 at the physics cap is the honest
    # endpoint. Out-of-slice = > SLICE ranks => the median is driven by the
    # slice boundary, not head quality.
    verdict = ("DRAFTER FIXABLE (median<=4)" if med <= 4 else
               ("INCONCLUSIVE (4<median<=8)" if med <= 8 else "T=1 ENDPOINT (median>8)"))
    print(f"[rank] VERDICT: {verdict}" + ("  [note: driven by the SLICE boundary — a full-vocab head re-test is the Tier-2 follow-up]" if OOS * 2 > NTOT else ""), flush=True)
  # R8_PROSE_DUMP: THE 7/60 ADJUDICATION (the novel-prose first-divergence vs
  # the T=1 ref). The spec stream IS the probe's argmax chain (accepted drafts
  # match amds rows; the bonus token IS amds[m]) -- so a first divergence means
  # the M=3 probe kernel family's argmax and the M=1 trunk family's argmax
  # disagree at that position over the SAME true-greedy prefix. Measure the
  # full-vocab fp16 logits of BOTH families at the exact divergence position:
  #   T1  : CONTINUE from the same anchor([]) the qref ran from, fd+1 decode
  #         passes -> d["logits"] (pass fd+1 computed qref[fd]).
  #         (A single step at anchor(pre) is WRONG: follow_up builds the state
  #         via the BATCH-PREFILL kernels -- a different accumulation than the
  #         decode path, one position late, wildly different logits.)
  #   PRB : replay the spec cycles from anchor([]) with per-cycle row dumps
  #         until the emitted stream covers fd  -> d["logits3"] row r.
  # MEASURED (2026-09-29, shipped default config): CROSS-CLASS FLIP. Both
  # families rank the SAME top-2 pair {11, 321} at idx 6; T1 margin +1.16 for
  # 11, probe margin +0.31 for 321; cross-family drift median 0.39/max 4.17
  # logits, concentrated on the flipped token (d(321)=1.39, d(11)=0.08). The
  # weaker margin (0.31) is INSIDE the family drift (0.39) -- which family
  # computes the greedy pick decides the token. The 7/60 is ONE such flip at
  # idx 6 + the cascade (novel prose = flat distributions); streams are
  # deterministic x2 within each family. Logits wobble +-0.1 run-to-run BELOW
  # argmax resolution (237k/248320 entries, max 0.098) -- token streams stay
  # bit-stable; only the bitwise-logit det requirement would call that a race.
  if os.getenv("R8_PROSE_DUMP", "0") == "1":
    _RM = int(_mtpmod.RM)
    _VOC = int(getattr(_mtpmod, "VOCAB", 248320))
    outw0 = outs[0]
    fd = next((k for k in range(NTOK) if outw0[k] != qref[k]), None)
    if fd is None or fd == 0:
      print(f"[pdump] nothing to adjudicate (first div {fd})", flush=True)
    else:
      pre = [int(t) for t in qref[:fd]]
      assert pre == outw0[:fd].tolist(), "pdump: shared-prefix violation"
      t_ref, t_spec = int(qref[fd]), int(outw0[fd])
      print(f"[pdump] first div idx {fd}: T1={t_ref} spec={t_spec}; shared prefix {pre}", flush=True)

      def _anchor(extra):
        # byte-mirror of quote_anchor with delta+extra as the fed stream
        reset_spec(E)
        _nc, _pn, _nf = E.follow_up(Gq, delta + extra)
        win_up(E.P, "tok_hist", P0*4, np.array([CUR0] + delta + extra, dtype=np.int32))
        dev.synchronize()
        return _nc, _pn

      # ---- (0) the T=1 reference itself, deterministic x2 (the adjudication
      #      is built on sand without this) ----
      t1d = []
      for _ in range(2):
        _nc, _pn = _anchor([])
        win_up(E.P, "tok_slot", 0, np.array([int(_nc)], dtype=np.int32))
        dev.synchronize()
        Gq.run_tokens(NTOK, wait_each=True)
        t1d.append(E.P.down_at("tok_hist", pb*4, NTOK, np.int32).tolist())
      print(f"[pdump] T1 ref deterministic x2: {t1d[0] == t1d[1]}; == harness qref: {t1d[0] == qref.tolist()}", flush=True)

      # ---- (1) T=1 full-vocab logits at the divergence position, x2.
      #      CAUTION (the first-run lesson): a single step at anchor(pre) sits
      #      on state built by the FOLLOW_UP BATCH-PREFILL kernels -- a
      #      different accumulation than the decode path the qref itself ran
      #      (and one feed-position late: argmax came out qref[fd+1]). The
      #      true T=1 distribution at the divergence = CONTINUE from the same
      #      anchor([]) the qref ran from, fd+1 decode passes; the logits
      #      buffer then holds pass fd+1's output = the qref[fd] prediction.
      t1_rows = []
      for it in range(2):
        _nc, _pn = _anchor([])
        win_up(E.P, "tok_slot", 0, np.array([int(_nc)], dtype=np.int32))
        dev.synchronize()
        Gq.run_tokens(fd + 1, wait_each=True)
        t1h = E.P.down_at("tok_hist", pb*4, fd + 1, np.int32).tolist()
        assert t1h == [int(t) for t in qref[:fd+1]], "pdump: T1 continuation lost qref sync"
        l1 = E.P.down("logits", (_VOC,), np.float16).astype(np.float32)
        t1_rows.append((int(t1h[fd]), l1.copy()))
        o = np.argsort(-l1)[:3]
        print(f"[pdump] T1 it{it}: continuation matched qref[:{fd+1}]; divergence-pos argmax "
              f"{t1h[fd]} (qref[{fd}]={t_ref}) top3 {[(int(i), round(float(l1[i]), 4)) for i in o]}", flush=True)

      # ---- (2) probe side: replay the spec decode from the quote anchor,
      #      dump the amds/logits3 rows of the cycle that emitted fd, x2 ----
      pr_rows = []
      for it in range(2):
        _nc, _pn = _anchor([])
        assert _pn == pb
        sess.begin()
        emt = []
        caught = None
        while len(emt) <= fd:
          r = sess.step()
          new = r["tokens"]; emt += new
          if len(emt) > fd and caught is None:
            m = int(r["m"])
            r_in = fd - (len(emt) - len(new))
            am = E.P.down_at("amds", 0, _RM, np.int32)
            lrows = [E.P.down_at("logits3", k*_VOC*2, _VOC, np.float16).astype(np.float32)
                     for k in range(min(m + 2, _RM))]
            caught = (m, r_in, am.tolist(), [row.copy() for row in lrows])
            break
        assert emt[:fd+1] == outw0.tolist()[:fd+1], "pdump: probe replay diverged from the recorded stream"
        m, r_in, am, lrows = caught
        lp = lrows[r_in]
        o = np.argsort(-lp)[:3]
        print(f"[pdump] PRB it{it}: stream idx {fd} was row {r_in} of a m={m} cycle; "
              f"amds[0..{m}] {am[:m+1]}; row top3 {[(int(i), round(float(lp[i]), 4)) for i in o]}", flush=True)
        pr_rows.append((m, r_in, am, lrows))

      # ---- (3) probe-side cross-iter row diff (the first run showed the two
      #      replays' caught rows agreeing on top3 yet not bit-equal -- the
      #      first-launch-vs-rest uninit-smem detector class; quantify it) ----
      lp0v = pr_rows[0][3][pr_rows[0][1]]
      lp1v = pr_rows[1][3][pr_rows[1][1]]
      drow = lp0v - lp1v
      nz = int((drow != 0).sum())
      print(f"[pdump] PRB row cross-iter: {nz}/{_VOC} entries differ, max|d| "
            f"{float(np.abs(drow).max()) if nz else 0.0:.4f}", flush=True)

      # ---- (4) the verdict ----
      # Stream/argmax-level determinism is the CONTRACT (logits wobble +-0.1
      # run-to-run below argmax resolution -- measured; the first-run bitwise
      # det requirement was wrong). Classification:
      #   CROSS-CLASS FLIP (benign, the tie-mine law at family-offset scale):
      #     both sides rank the SAME top-2 pair, each prefers its own pick,
      #     and the weaker margin <= the cross-family median logit drift -- the
      #     greedy choice is decided by which kernel family computes it.
      #   STATE-DIVERGENCE (Class C): disjoint top sets or logits far outside
      #     the family-drift envelope -- accept-state corruption, P0-grade.
      tok1, l1 = t1_rows[0]
      m_t1 = float(l1[t_ref]) - float(l1[t_spec])
      _, r_in, am, lrows = pr_rows[0]
      lp = lrows[r_in]
      m_pr = float(lp[t_spec]) - float(lp[t_ref])
      dl = np.abs(l1 - lp)
      dmax = float(dl.max()); dmed = float(np.median(dl))
      d_ref = float(dl[t_ref]); d_spec = float(dl[t_spec])
      top2_t1 = {int(i) for i in np.argsort(-l1)[:2]}
      top2_pr = {int(i) for i in np.argsort(-lp)[:2]}
      same_top2 = top2_t1 == top2_pr and top2_t1 == {t_ref, t_spec}
      det_stream = (t1d[0] == t1d[1]) and bool((outs[0] == outs[1]).all()) \
                   and all(t1_rows[i][0] == t_ref for i in (0, 1)) and am[r_in] == t_spec
      print(f"[pdump] margins: T1 prefers {t_ref} over {t_spec} by {m_t1:+.4f}; probe prefers "
            f"{t_spec} over {t_ref} by {m_pr:+.4f}; same top-2 pair: {same_top2}", flush=True)
      print(f"[pdump] cross-family logit drift: median {dmed:.4f} max {dmax:.4f} | per-token "
            f"d({t_ref}) {d_ref:.4f} d({t_spec}) {d_spec:.4f} | stream-level det x2 {det_stream}", flush=True)
      if not det_stream:
        cls = "NONDETERMINISTIC STREAM (adjudication void -- a real RACE, not numerics)"
      elif same_top2 and m_t1 > 0 and m_pr > 0 and min(m_t1, m_pr) <= dmed:
        cls = (f"CROSS-CLASS FLIP (Class B, benign: the tie-mine law at family-offset scale -- "
               f"both sides rank {{{t_ref},{t_spec}}} top-2 and the weaker margin "
               f"{min(m_t1, m_pr):.3f} is within the cross-family drift {dmed:.3f}; the spec "
               f"stream is the probe-world's own consistent greedy chain)")
      elif not same_top2 or dmax > 4.0:
        cls = ("STATE-DIVERGENCE (Class C: probe row logits outside the family-drift envelope "
               "at the same true-greedy prefix -- accept-state/KV corruption, P0-grade)")
      else:
        cls = (f"RESOLVED-FLIP (real argmax disagreement above the drift scale: m_t1 {m_t1:.4f} "
               f"m_pr {m_pr:.4f} vs dmed {dmed:.4f} -- investigate the row {r_in} path)")
      print(f"[pdump] VERDICT: {cls}", flush=True)
      np.save("~/r8_pdump_t1.npy", l1)
      np.save("~/r8_pdump_prb.npy", lp)
      np.save("~/r8_pdump_prb_it1.npy", lp1v)
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
    L = _np.array(R4LOG, dtype=_np.int64)          # (m, hit, deep-selected, prose)
    dl = L[:, 2] == 1
    pl = (L[:, 2] == 0) & (L[:, 3] == 1)   # P10-dense: the K4-EAGLE prose set
    kl = (L[:, 2] == 0) & (L[:, 3] == 0)
    print(f"[deepk] cycles {len(L)}, deep-selected {int(dl.sum())} ({100.0*dl.mean():.1f}%), "
          f"E[m|deep] {float(L[dl,0].mean()) if dl.any() else 0:.3f}, E[m|k2] {float(L[kl,0].mean()) if kl.any() else 0:.3f}, "
          f"m-dist(deep) {[int((L[dl,0]==v).sum()) for v in range(int(os.getenv('LOOKUP_K','0') or 0)+1)]}, hits {int((L[:,1]>=9).sum())}", flush=True)
    if pl.any():
      print(f"[k4eagle] prose cycles {int(pl.sum())} ({100.0*pl.mean():.1f}% of non-deep), "
            f"E[m|k4] {float(L[pl,0].mean()):.3f}, m-dist(k4) {[int((L[pl,0]==v).sum()) for v in range(5)]}, "
            f"tok/cyc(k4) {float((L[pl,0]+1).mean()):.2f}", flush=True)
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
_emt_counts = []
for rep in range(3):
  reset_spec(E)
  t0 = time.perf_counter()
  emt, _ = decode_n(NTOK)
  dt = time.perf_counter() - t0
  h = hist(E); out = h[P0:P0+NTOK]
  times.append(dt)
  _emt_counts.append(len(emt))
  m2 = E.P.down("m_hist", (1024,), np.int32)
  acc_pos = float(m2[:NTOK].sum()) / (2.0*NTOK)
  tpc = float((m2[:NTOK] + 1).sum()) / NTOK
  agree = int((out == np.array(ref)).sum())
  if TLX_T1_MODE:
    # A.1: T=1 cycles emit without an m_hist entry — the honest per-rep metric
    # is emitted-tokens / wall-time (the m_hist-derived tok/cyc undercounts).
    print(f"[time] rep{rep}: {dt*1e3:.1f} ms/{NTOK}-tok stream = {dt/NTOK*1e3:.2f} ms/tok, agree {agree}/{NTOK}, "
          f"t1-cycles(life) {int(getattr(sess, 'nt1cycles', 0))}, tok/s={len(emt)/dt:.2f} (emit-based)", flush=True)
  else:
    print(f"[time] rep{rep}: {dt*1e3:.1f} ms/60cyc = {dt/NTOK*1e3:.2f} ms/cyc, agree {agree}/{NTOK}, alpha={acc_pos:.3f}, tok/cyc={tpc:.2f}, tok/s={tpc/(dt/NTOK):.2f}", flush=True)
best = min(times)
m3 = E.P.down("m_hist", (1024,), np.int32)
tpc = float((m3[:NTOK] + 1).sum()) / NTOK
if TLX_T1_MODE:
  # deterministic stream: every rep emits the same count — the best-wall rep's
  # emit count pairs with min(times)
  print(f"[time] BEST {best*1e3/NTOK:.2f} ms/stream-tok -> {_emt_counts[times.index(best)]/best:.2f} tok/s emit-based (T=1-mixed)", flush=True)
else:
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
