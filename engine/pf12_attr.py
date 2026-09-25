# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P12 Front-1a: 2k-chunk in-graph class attribution (where do the 20-28ms go).

Boot the full trunk (G3M on = ship class), prefill 2048 dummy ids via
prefill_batch (reference chunk distribution), rewind to the last chunk, then:
  - full-graph reference (PG_SPLIT=2 path AND single-queue) min-of-5
  - ISOLATED per-class captured graphs min-of-5 (plan classes + dfill classes)
  - host decomposition: 4x win_up + submit + wait wall vs graph GPU time
Laws: FULL env; readout from first clean run; synced timing; no PROFILE.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill
from pf_prefill import PfGraph

E = MTPEngine(theta=1e7)
P, W, d = E.P, E.W, E.P.d
rng = np.random.default_rng(0)
ids = [int(t) for t in rng.integers(1000, 60000, 2048)]

from mtp import CBLK
E.reset_fresh(ids[0])
E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
# create the draft buffers + fill once (mirrors the test_w100k boot: slice from
# ids -> init_draft -> fill_draft; DFILL windows then reuse the buffers)
from mtp import SLICE
_seen, sl = set(), []
for t in ids:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.fill_draft(ids, start_pos=0, seed_hd=None)
dev.synchronize()

CT = []
t0 = time.perf_counter()
dt = pf_prefill.prefill_batch(E, None, ids, chunk_times=CT)
dev.synchronize()
cts = sorted(ms for _, ms in CT)
print(f"[attr] prefill 2048: {dt:.2f}s chunks={len(cts)} med={cts[len(cts)//2]:.1f} "
      f"min={cts[0]:.1f} max={cts[-1]:.1f} p90={cts[int(len(cts)*0.9)]:.1f}", flush=True)

# rewind to the LAST chunk (pos 2016): replays write identical bytes
P.win_up("pos_slot", 0, np.array([2016], dtype=np.int32))
P.win_up("pos_slot_b", 0, np.array([2032], dtype=np.int32))
if getattr(pf_prefill, "N32", False):
  P.win_up("ids32", 0, np.array(ids[2016:2048], dtype=np.int32))
else:
  P.win_up("ids16a", 0, np.array(ids[2016:2032], dtype=np.int32))
  P.win_up("ids16b", 0, np.array(ids[2032:2048], dtype=np.int32))
dev.synchronize()

plan = list(E._pf_plan)
dseq = pf_prefill._pf_dfill_seq(E) if (pf_prefill.M32 and pf_prefill.DFILL) else []
full = plan + dseq

PLAN_CLS = [("pfk_emb16", "emb"), ("pfk_n16", "nrm_n"), ("pfk_ab16", "nrm_ab"), ("pfk_hh16", "nrm_hh"),
            ("pfk_pre16", "pre"), ("pfa32c", "attn"), ("pfa16", "attn"), ("pfc16", "attn_comb"),
            ("pfs16", "scan"),
            ("attnqkv", "gemm_qkv"), ("gdnqg", "gemm_qg"), ("pfg_ffn", "gemm_ffn"),
            ("iq3d", "gemm_fd"), ("iq3s", "gemm_oa"), ("iq3o", "gemm_og"), ("q8o", "gemm_og")]
DF_CLS = [("pfk_rec16", "df_rec"), ("pfk_emb16", "df_emb"), ("pfd_dnorm16", "df_dn"),
          ("pfg_ehd", "df_ehd"), ("pfk_n16", "df_n"), ("dqkv", "df_dqkv"), ("pfk_pre16", "df_pre")]

def classify(p, table):
  nm = getattr(p, "name", "?")
  for tag, c in table:
    if tag in nm:
      return c
  return "misc_" + nm[:12]

groups = {}
for t in plan:
  groups.setdefault(classify(t[0], PLAN_CLS), []).append(t)
for t in dseq:
  groups.setdefault(classify(t[0], DF_CLS), []).append(t)

def time_graph(gs, n=5):
  best = 1e9
  for _ in range(n):
    prev = dev.timeline_value - 1
    t1 = time.perf_counter()
    for g in gs[:-1]:
      v = dev.next_timeline(); g.submit(prev, v); prev = v
    v = dev.next_timeline(); gs[-1].submit(prev, v)
    dev.timeline_signal.wait(v)
    best = min(best, time.perf_counter() - t1)
  return best * 1e3

rows = []
gfull2 = [PfGraph(full[:len(full)//2], "af0"), PfGraph(full[len(full)//2:], "af1")]
gfull1 = [PfGraph(full, "af_all")]
t_split2 = time_graph(gfull2)
t_split1 = time_graph(gfull1)
print(f"[attr] FULL graph: split2={t_split2:.2f}ms split1={t_split1:.2f}ms", flush=True)

for cname in sorted(groups):
  seq = groups[cname]
  g = PfGraph(seq, f"a_{cname}")
  ms = time_graph([g])
  rows.append((cname, len(seq), ms))
  print(f"[attr] {cname:12s} n={len(seq):4d} {ms:7.2f}ms  ({ms/len(seq)*1000:6.0f}us/launch)", flush=True)

tot = sum(r[2] for r in rows)
print(f"[attr] SUM(isolated)={tot:.2f}ms vs FULL split2={t_split2:.2f} split1={t_split1:.2f}", flush=True)
print(f"[attr] chunk wall med={cts[len(cts)//2]:.1f}ms -> host/submit/wait overhead vs split2 = {cts[len(cts)//2]-t_split2:.2f}ms", flush=True)

# win_up cost (the per-chunk host DMAs)
tw = 1e9
for _ in range(10):
  t1 = time.perf_counter()
  P.win_up("pos_slot", 0, np.array([2016], dtype=np.int32))
  P.win_up("pos_slot_b", 0, np.array([2032], dtype=np.int32))
  if getattr(pf_prefill, "N32", False):
    P.win_up("ids32", 0, np.array(ids[2016:2048], dtype=np.int32))
  else:
    P.win_up("ids16a", 0, np.array(ids[2016:2032], dtype=np.int32))
    P.win_up("ids16b", 0, np.array(ids[2032:2048], dtype=np.int32))
  dev.synchronize()
  tw = min(tw, time.perf_counter() - t1)
print(f"[attr] 4x win_up+sync = {tw*1e3:.2f}ms", flush=True)
print("[attr] DONE", flush=True)
