# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P10-dense rung 2: K4-cycle phase bench. Boots EXACTLY like test_w100k
(snap100k, slice, fill_draft), anchors the R8_PROSE state via the [quote]
harness path (reset_spec + follow_up delta), then phase-times the K4 set
(draft4/probe5/accept5) vs the K2 set (draft/probe3/acceptk) per graph.
env: the canonical battery env + TLX_EAGLE_K=4 (PROSE_TRIG irrelevant)."""
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
from tinygrad.runtime.ops_nv import nv_wait_timeline

SNAP = "~/snap100k"
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()

def win_up(P, name, off, arr): P.win_up(name, off, arr)

def load100k(E):
  P = E.P
  E.load_snapshot_kv(SNAP, progress=True)
  for j, i in enumerate(E.gdn_idx):
    P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
    P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
    if j % 8 == 0: E._flush()
  P.win_up("tok_slot", 0, np.array([CUR0], dtype=np.int32))
  P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
  E._flush()

def reset_spec(E):
  E.reset_snapshot(SNAP, CUR0, P0)
  win_up(E.P, "tok_hist", 0, np.array(ids, dtype=np.int32))
  dev.synchronize()

print("[phase] loading engine...", flush=True)
t0 = time.perf_counter()
E = MTPEngine(theta=1e7)
print(f"[phase] engine loaded {time.perf_counter()-t0:.1f}s", flush=True)
load100k(E)
dev.synchronize()

ref = np.load(f"{SNAP}/engine_t1_ref_kv8_qh.npy").tolist()
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
sl = sl[:SLICE]
E.init_draft(sl)
E.fill_draft(ids)
E.build_graphs()
dev.synchronize()
assert E.graphs5 is not None, "K4 graphs missing (TLX_EAGLE_K != 4?)"

Gq = GCycleEngine(E); Gq.build(); dev.synchronize()
delta = [int(t) for t in np.load("~/r8_prose_ids.npy")[:120]]
reset_spec(E)
new_cur, pos_q, n_fed = E.follow_up(Gq, delta)
win_up(E.P, "tok_hist", P0*4, np.array([CUR0] + delta, dtype=np.int32))
dev.synchronize()
print(f"[phase] anchored pos={pos_q} cur={new_cur}", flush=True)

def phase_set(tag, graphs, n=16):
  draft_g, probe_g, accept_g, flush_g = graphs
  prev = dev.timeline_value - 1
  t_d = t_p = t_a = 0.0
  for c in range(n):
    t0 = time.perf_counter()
    vd = dev.next_timeline(); draft_g.submit(prev, vd); prev = vd
    dev.timeline_signal.wait(vd)
    t1 = time.perf_counter()
    vp = dev.next_timeline(); probe_g.submit(vd, vp); prev = vp
    dev.timeline_signal.wait(vp)
    t2 = time.perf_counter()
    va = dev.next_timeline(); accept_g.submit(vp, va); prev = va
    vf = dev.next_timeline(); flush_g.submit(va, vf); prev = vf
    nv_wait_timeline(dev, vf, what="phase")
    t3 = time.perf_counter()
    t_d += t1 - t0; t_p += t2 - t1; t_a += t3 - t2
  print(f"[phase] {tag}: draft {t_d/n*1e3:.2f} ms, probe {t_p/n*1e3:.2f}, accept+flush {t_a/n*1e3:.2f}", flush=True)
  return t_d/n*1e3, t_p/n*1e3, t_a/n*1e3

phase_set("K4 draft4/probe5/accept5", E.graphs5)
phase_set("K2 draft/probe3/acceptk ", E.graphs)
phase_set("K4 rep2                  ", E.graphs5)
print("[phase] done", flush=True)
