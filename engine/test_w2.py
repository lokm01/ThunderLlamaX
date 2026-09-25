# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2-MTP gate: Tier-1 exactness (spec(K=2) == greedy(T=1) engine, 60/60 bit-exact
at 2k, deterministic across reruns) + tok/s + alpha + per-phase breakdown."""
import os, sys, time, collections
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, SLICE
from trunk import CTX
from engine0 import dev
from gcycle import GCycleEngine

snap = np.load(os.getenv("SNAP", "~/w1b_state_2k.npz"))
theta = float(snap["theta"].reshape(-1)[0])
P0 = int(snap["P"].reshape(-1)[0])
ids = snap["ids"].reshape(-1).tolist()
NTOK = 60
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "1"))

print("[w2] loading engine (~13GB)...", flush=True)
t0 = time.perf_counter()
E = MTPEngine(theta)
print(f"[w2] engine loaded {time.perf_counter()-t0:.1f}s", flush=True)

def hist():
  return E.P.down("tok_hist", (CTX+128,), np.int32)

# ---------- T=1 engine reference ----------
G = GCycleEngine(E)
E.restore(snap); G.build(); dev.synchronize()
G.run_tokens(NTOK, wait_each=True)
h = hist()
ref = h[P0-1:P0-1+NTOK].tolist()
print(f"[ref] T=1 engine 60 tokens: {ref[:12]}...", flush=True)
assert all(t >= 0 for t in ref), "T=1 reference incomplete"

# ---------- deterministic slice (ref outputs first, then prompt-frequent) ----------
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]   # pad to SLICE by cycling (duplicate rows tie-break to first idx in samx)
while len(sl) < SLICE: sl += base
sl = sl[:SLICE]
print(f"[slice] {len(sl)} ids ({sum(1 for t in ref if t in seen)}/60 ref tokens covered)", flush=True)
E.init_draft(sl)

def spec_run(ncyc=NTOK, phase_times=False, sync_every=1):
  E.restore_mtp(snap)
  E.fill_draft(ids)
  E.build_graphs()
  dev.synchronize()
  r = E.run_cycles(ncyc, sync_every=sync_every, phase_times=phase_times or bool(os.getenv("PHASEDBG")))
  h = hist()
  out = h[P0-1:P0-1+NTOK]
  m = E.P.down("m_hist", (1024,), np.int32)
  pos_end = int(E.P.down("pos_slot", (1,), np.int32)[0])
  return out, m, pos_end, r

# ---------- Tier-1: exactness x2 (determinism) ----------
outs = []
for rep in range(2):
  out, m, pend, _ = spec_run()
  agree = int((out == np.array(ref)).sum())
  first_div = next((k for k in range(NTOK) if out[k] != ref[k]), None)
  print(f"[tier1] rep{rep}: {agree}/{NTOK} exact, first divergence at {first_div}, m_hist[:20]={m[:20].tolist()}", flush=True)
  if agree < NTOK:
    print(f"[tier1] out: {out.tolist()}")
    print(f"[tier1] ref: {ref}")
  outs.append(out.copy())
print(f"[tier1] deterministic across reps: {bool((outs[0] == outs[1]).all())}", flush=True)

# ---------- speed: 3 reps (setup untimed; 60 cycles timed) ----------
times = []
for rep in range(3):
  E.restore_mtp(snap)
  E.fill_draft(ids)
  E.build_graphs()
  dev.synchronize()
  t0 = time.perf_counter()
  E.run_cycles(NTOK, sync_every=SYNC_EVERY)
  dt = time.perf_counter() - t0
  times.append(dt)
  out = hist()[P0-1:P0-1+NTOK]
  ntok_out = int((out >= 0).sum())
  m2 = E.P.down("m_hist", (1024,), np.int32)
  acc_pos = float(m2[:NTOK].sum()) / (2.0*NTOK)   # accepted drafts / proposed (2/cyc)
  tpc = float((m2[:NTOK] + 1).sum()) / NTOK
  agree = int((out == np.array(ref)).sum())
  print(f"[time] rep{rep}: {dt*1e3:.1f} ms/60cyc, agree {agree}/{NTOK}, "
        f"alpha(pos)={acc_pos:.3f}, tok/cyc={tpc:.2f}, tok/s={tpc/(dt/NTOK):.2f}", flush=True)
best = min(times)
mlast = E.P.down("m_hist", (1024,), np.int32)
tpc = float((mlast[:NTOK] + 1).sum()) / NTOK
print(f"[time] BEST {best*1e3:.1f} ms/60cyc = {best/NTOK*1e3:.2f} ms/cyc -> {tpc/(best/NTOK):.2f} tok/s", flush=True)

# ---------- per-phase breakdown ----------
out, m, pend, pt = spec_run(phase_times=True)
td, tp, ta = pt
print(f"[phase] draft {td*1e3:.1f} ms/cyc, probe {tp*1e3:.1f}, accept {ta*1e3:.1f} (per 60 cyc avg)", flush=True)
print("[done]", flush=True)
