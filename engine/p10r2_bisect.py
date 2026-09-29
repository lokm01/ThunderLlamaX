# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P10-dense rung 2: probe5 per-kernel-class bisect. Boots like test_w100k,
anchors prose, then times PREFIX graphs of the probe5 seq (blocks 0..N) plus
individual kernel-class graphs to localize the K4 probe cost. Usage: after the
phase bench names probe5 as the elephant."""
import os, sys, time, json, collections
os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, RBLK, CBLK, SLICE, CTXK
from engine0 import dev
from gcycle import GCycleEngine, ParityGraph
from tinygrad.runtime.ops_nv import nv_wait_timeline

SNAP = "~/snap100k"
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()

def win_up(P, name, off, arr): P.win_up(name, off, arr)

def load100k(E):
  P = E.P
  E.load_snapshot_kv(SNAP)
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

print("[bisect] loading engine...", flush=True)
E = MTPEngine(theta=1e7)
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
E.init_draft(sl[:SLICE])
E.fill_draft(ids)
E.build_graphs()
dev.synchronize()

Gq = GCycleEngine(E); Gq.build(); dev.synchronize()
delta = [int(t) for t in np.load("~/r8_prose_ids.npy")[:120]]
reset_spec(E)
new_cur, pos_q, _ = E.follow_up(Gq, delta)
dev.synchronize()
print(f"[bisect] anchored pos={pos_q}", flush=True)

# build class-filtered probe5 variants: all / no-attention / no-ffn / no-scan
full = E._probe5_seq()
def mkname(e):
  p, a, g = e
  return getattr(p, "name", "?")
def has(e, sub): return sub in mkname(e)

variants = {
  "full":     [e for e in full],
  "no_attn":  [e for e in full if not (has(e, "spk_") or has(e, "aq") or has(e, "ao8"))],
  "no_ffn":   [e for e in full if not (has(e, "ffn8") or has(e, "down8") or has(e, "dfgu"))],
  "no_scan":  [e for e in full if not has(e, "k2s5")],
  "no_norm":  [e for e in full if not (has(e, "k0n") or has(e, "k0ab") or has(e, "hh5"))],
  "no_head":  [e for e in full if not (has(e, "head8") or has(e, "amx3"))],
}
def tgraph(seq, tag, n=8):
  g = ParityGraph(seq, tag=f"bis{tag}")
  prev = dev.timeline_value - 1
  t0 = time.perf_counter()
  for _ in range(n):
    v = dev.next_timeline(); g.submit(prev, v); prev = v
    g2 = ParityGraph([(E.pr["dposadd"], (E.P.d["fillpos"], E.P.d["dpos1"]), 1)], tag="fl")  # flusher pair law
    v2 = dev.next_timeline(); g2.submit(v, v2); prev = v2
    nv_wait_timeline(dev, v2, what="bisect")
  dt = (time.perf_counter() - t0) / n * 1e3
  try: g2.close(); g.close()
  except Exception: pass
  print(f"[bisect] {tag:10s} {len(seq):4d}k: {dt:7.2f} ms", flush=True)
  return dt

res = {k: tgraph(v, k) for k, v in variants.items()}
print(f"[bisect] attn ~ {res['full']-res['no_attn']:.1f} | ffn ~ {res['full']-res['no_ffn']:.1f} | "
      f"scan ~ {res['full']-res['no_scan']:.1f} | norms ~ {res['full']-res['no_norm']:.1f} | head ~ {res['full']-res['no_head']:.1f}", flush=True)
print("[bisect] done", flush=True)
