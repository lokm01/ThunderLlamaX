# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX ATT_RB mission — the T=3 (K=2 EAGLE, the shipped default) probe class
bisect @100k. The banked P10-r2 bisect measured the T=5 probe (attn 22.6ms);
this run measures the SHIPPED T=3 probe on the same methodology so the
attention-family number is comparable and the row-batched win-math can be
re-derived on current numbers.

Boots exactly like p10r2_bisect.py (snap100k, slice, fill_draft, prose
anchor via reset_spec + follow_up), then times ParityGraphs built from
E.graphs[1].seq (the probe3 seq) with class filters:
  full    — the whole probe3 graph (the probe ms)
  no_attn — minus spk_*/aq*/ao8*   (the attention family; cf. 22.58 T=5)
  no_spk  — minus the spk trio     (pre3 + a3 + c3 = the split-KV machinery)
  no_k1   — minus spk_g4nw32hm3    (the row-batched KV reader alone)
  no_ffn  — minus ffn8/down8       (sanity vs the banked 28.78 T=5)
Timing-valid / output-garbage (the P9 truncated-graph law). The flusher-pair
law applies (every graph submit paired with a dposadd flusher)."""
import os, sys, time, json, collections
os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, SLICE
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

print("[rb3] loading engine...", flush=True)
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
print(f"[rb3] anchored pos={pos_q} cur={new_cur}", flush=True)

seq3 = E.graphs[1].seq   # the T=3 probe graph's seq (graphs = draft,probe,accept,flush)
names = collections.Counter(getattr(p, "name", "?") for p, a, g in seq3)
print(f"[rb3] probe3 seq: {len(seq3)} kernels; attention-family names:", flush=True)
for n, c in sorted(names.items()):
  if n.startswith("spk_") or n.startswith("aq") or n.startswith("ao8"):
    print(f"[rb3]   {c:3d} x {n}", flush=True)

def mkname(e):
  p, a, g = e
  return getattr(p, "name", "?")
def has(e, sub): return sub in mkname(e)

K1 = "g4nw32hm3"   # the T=3 split-KV reader cubin name
variants = {
  "full":    [e for e in seq3],
  "no_attn": [e for e in seq3 if not (has(e, "spk_") or has(e, "aq") or has(e, "ao8"))],
  "no_spk":  [e for e in seq3 if not has(e, "spk_")],
  "no_k1":   [e for e in seq3 if not has(e, K1)],
  "no_ffn":  [e for e in seq3 if not (has(e, "ffn8") or has(e, "down8"))],
}
def tgraph(seq, tag, n=8):
  g = ParityGraph(seq, tag=f"rb3{tag}")
  prev = dev.timeline_value - 1
  t0 = time.perf_counter()
  for _ in range(n):
    v = dev.next_timeline(); g.submit(prev, v); prev = v
    g2 = ParityGraph([(E.pr["dposadd"], (E.P.d["fillpos"], E.P.d["dpos1"]), 1)], tag="fl")
    v2 = dev.next_timeline(); g2.submit(v, v2); prev = v2
    nv_wait_timeline(dev, v2, what="rb3")
  dt = (time.perf_counter() - t0) / n * 1e3
  try: g2.close(); g.close()
  except Exception: pass
  print(f"[rb3] {tag:8s} {len(seq):4d}k: {dt:7.2f} ms", flush=True)
  return dt

res = {k: tgraph(v, k) for k, v in variants.items()}
print(f"[rb3] T=3 attention family (spk+aq+ao8) ~ {res['full']-res['no_attn']:.2f} ms | "
      f"spk trio ~ {res['full']-res['no_spk']:.2f} | K1 KV-reader ~ {res['full']-res['no_k1']:.2f} | "
      f"ffn ~ {res['full']-res['no_ffn']:.2f} (banked T=5: attn 22.58, ffn 28.78)", flush=True)
print("[rb3] done", flush=True)
