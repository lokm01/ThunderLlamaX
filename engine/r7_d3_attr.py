# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7 DECIDER 3: deep-cycle phase attribution — the 47ms split.

The K=7 sel-mode cycle = 0.817*116.32(deep) + 0.183*69.21(K2). The deep cycle
carries a ~47.1ms increment over K2 (all-deep 116.32 - deep-off 69.21) that has
never been split. This harness times the DEEP graph set per family and the K2
set per family (same boot, same slots at pos 97810) and reports the deltas:

  families (from the probe seq, classified by pr-key):
    attn_rows : spk_pre8 / spk_a8 / spk_c8   (vs spk_pre3/a3/c3)
    k2s8      : the rec-chain t-steps         (vs k2s3)
    gemv      : ALL M-extended GEMV/norm kernels (aq*q8v8/q5g8v8/k3ao/op38/
                ao8/ffn8/down8/k0n8/k0ab8/hh8/h_embed8/head8v8/amx3)
      - sub-split: attn-GEMV (aq*/ao8), gdn-GEMV (q5g8/k3ao/op38), ffn+down,
        norms+embed, head+amx
    accept    : accept8k + acceptsel8k        (vs acceptk + acceptsel)

Method: per-family isolated ParityGraphs (the p18_attr law — classes fully
account for the graph); timing = chained submit + flusher + timeline wait
(the R4_BISECT run_partial pattern; LONE-GRAPH law respected), min-of-N
synced. Warmup pair first (the P17 first-pair law). Timing-only: partial
graphs poison GDN live — NO gates after this harness.
"""
import time
import numpy as np
from gcycle import ParityGraph
from engine0 import dev

DEEP_FAM = [
  ("attn_rows", ("spk_pre8", "spk_a8", "spk_c8")),
  ("k2s8",      ("k2s8",)),
  ("gemv_attn", ("aq3k8v8", "aq6k8v8", "ao8nw32_8")),
  ("gemv_gdn",  ("q5g8v8", "k3aonw32_8", "op38nw32_8")),
  ("ffn_down",  ("ffn8v8r7", "ffn8v8", "down8nw32v8r7", "down8nw32_8")),
  ("norms_emb", ("k0n8", "k0ab8", "hh8", "h_embed8")),
  ("head_amx",  ("head8v8", "amx3")),
]
K2_FAM = [
  ("attn_rows", ("spk_pre3", "spk_a3", "spk_c3", "aattn3")),
  ("k2s8",      ("k2s3", "k2s3v", "k2z3")),
  ("gemv_attn", ("aq3k8v_3", "aq3k8_3", "aq6k8_3", "ao8nw32_3", "ao8_3")),
  ("gemv_gdn",  ("q5g8v_3", "q5g8_3", "k3aonw32_3", "k3ao3", "op38nw32_3", "op38_3")),
  ("ffn_down",  ("ffn8v3r7", "ffn8v_3", "ffn8_3", "down8nw32v3r7", "down8nw32_3", "down8_3")),
  ("norms_emb", ("k0n3", "k0ab3", "hh3", "h_embed3")),
  ("head_amx",  ("head8v_3", "head8_3", "amx3")),
]
ACCEPT_FAM = [("acceptk", ("acceptk", "accept")), ("acceptsel", ("acceptsel",)),
              ("accept8k", ("accept8k",)), ("acceptsel8k", ("acceptsel8k",))]

def classify(seq, E, table):
  key2fam = {}
  for fam, keys in table:
    for k in keys: key2fam[k] = fam
  rev = {}
  for k, v in E.pr.items(): rev[id(v)] = k
  groups = {}
  misc = []
  for p, bufs, grid in seq:
    k = rev.get(id(p), getattr(p, "name", "?"))
    fam = None
    for kk, ff in key2fam.items():
      if k == kk or k.startswith(kk):
        fam = ff; break
    if fam is None:
      misc.append((p, bufs, grid, k)); continue
    groups.setdefault(fam, []).append((p, bufs, grid))
  return groups, misc

def tmin(g, flush_g, n=5, warm=2):
  best = 1e9
  for i in range(n + warm):
    prev = dev.timeline_value - 1
    t1 = time.perf_counter()
    v1 = dev.next_timeline(); g.submit(prev, v1)
    v2 = dev.next_timeline(); flush_g.submit(v1, v2)
    dev.timeline_signal.wait(v2)
    dt = time.perf_counter() - t1
    if i >= warm:
      best = min(best, dt)
  return best * 1e3

def run(E, reset_spec=None, CUR0=4471, P0=97810):
  P = E.P
  # THE PROVEN BARE-PROBE STATE (R4_ROWCHK law): reset_spec seeds tok_hist with
  # the real prompt ids (lookup8 in the SHARED draft graph scans it), poisons +
  # re-seeds the GDN snapshot, zeroes h_seed. Without it the probe submit faults.
  if reset_spec is not None:
    reset_spec(E)
  deep_g = E.graphs6        # (draft_g, probe8_g, accept8_g, flush_g)
  k2_g = E.graphs           # (draft_g, probe3_g, acceptk_g, flush_g)
  flush_g = deep_g[3]
  seq8 = deep_g[1].seq
  seq3 = k2_g[1].seq
  aseq8 = deep_g[2].seq
  aseq3 = k2_g[2].seq
  print(f"[d3] deep probe {len(seq8)}k, k2 probe {len(seq3)}k, accept8 {len(aseq8)}k, accept {len(aseq3)}k", flush=True)

  # seed slots at the parked 100k position (phase-breakdown pattern)
  P.win_up("cur_slot", 0, np.array([int(CUR0)], dtype=np.int32))
  P.win_up("pos_slot", 0, np.array([int(P0)], dtype=np.int32))
  P.win_up("m_hist", 0, np.zeros(1024, dtype=np.int32))
  P.win_up("cyc_slot", 0, np.zeros(1, dtype=np.int32))
  # R4 PROBE-POISON LAW: probe graphs WITHOUT the draft having run need ALL
  # drings = VALID token ids (-1 -> embedding OOB -> device fault).
  for nm in ("dring0", "dring1", "dring2", "dring3", "dring4", "dring5", "dring6"):
    P.win_up(nm, 0, np.array([int(CUR0)], dtype=np.int32))
  dev.synchronize()

  g8, misc8 = classify(seq8, E, DEEP_FAM)
  g3, misc3 = classify(seq3, E, K2_FAM)
  ga8, _ = classify(aseq8, E, ACCEPT_FAM)
  ga3, _ = classify(aseq3, E, ACCEPT_FAM)
  # NOTE: accept graphs are direct-indexed below (classify prefix law: accept8k
  # startswith accept -> wrong bucket). ga8/ga3 kept only for the misc report.
  for nm, m in (("deep", misc8), ("k2", misc3)):
    for p, bufs, grid, k in m:
      print(f"[d3] WARNING {nm} misc kernel: {k} g={grid}", flush=True)

  G8 = {f: ParityGraph(s, tag=f"r7d3_{f}") for f, s in g8.items()}
  G3 = {f: ParityGraph(s, tag=f"r7k2_{f}") for f, s in g3.items()}
  print(f"[d3] family graphs: deep {sorted(G8)} k2 {sorted(G3)}", flush=True)
  dev.synchronize()

  # warmup pair (first-pair law)
  tmin(deep_g[1], flush_g, n=1, warm=0)
  tmin(k2_g[1], flush_g, n=1, warm=0)
  dev.synchronize()

  print("\n== FULL graphs (calibration anchors) ==", flush=True)
  t_p3 = tmin(k2_g[1], flush_g)       # K2 FIRST (bisect: if this faults = state, not M8)
  t_a3 = tmin(k2_g[2], flush_g)
  t_draft = tmin(deep_g[0], flush_g)
  t_p8 = tmin(deep_g[1], flush_g)
  t_a8 = tmin(deep_g[2], flush_g)
  t_f = tmin(flush_g, deep_g[0])   # flush timed with draft as the follow-up
  print(f"[d3] draft {t_draft:6.2f}  probe8 {t_p8:6.2f}  accept8 {t_a8:5.2f}  flush {t_f:5.2f}", flush=True)
  print(f"[d3]                    probe3 {t_p3:6.2f}  accept3 {t_a3:5.2f}", flush=True)
  deep_cycle = t_draft + t_p8 + t_a8 + t_f
  k2_cycle = t_draft + t_p3 + t_a3 + t_f
  print(f"[d3] deep cycle ~= {deep_cycle:6.2f}   k2 cycle ~= {k2_cycle:6.2f}   increment {deep_cycle-k2_cycle:6.2f}", flush=True)

  print("\n== per-family (deep vs k2, ms) ==", flush=True)
  print(f"{'family':10s} {'deep':>8s} {'k2':>8s} {'delta':>8s} {'n8':>4s} {'n3':>4s}", flush=True)
  fams = ["attn_rows", "k2s8", "gemv_attn", "gemv_gdn", "ffn_down", "norms_emb", "head_amx"]
  t8 = {}; t3 = {}
  for f in fams:
    t8[f] = tmin(G8[f], flush_g) if f in G8 else float("nan")
    t3[f] = tmin(G3[f], flush_g) if f in G3 else float("nan")
    n8 = len(g8.get(f, [])); n3 = len(g3.get(f, []))
    print(f"{f:10s} {t8[f]:8.2f} {t3[f]:8.2f} {t8[f]-t3[f]:8.2f} {n8:4d} {n3:4d}", flush=True)
  s8 = sum(t8[f] for f in fams if f in G8)
  s3 = sum(t3[f] for f in fams if f in G3)
  print(f"{'SUM':10s} {s8:8.2f} {s3:8.2f} {s8-s3:8.2f}  (vs full-probe deltas {t_p8-t_p3:6.2f}; attr err probe8 {s8-t_p8:+.2f} probe3 {s3-t_p3:+.2f})", flush=True)

  print("\n== accept family (direct-indexed: aseq[0]=main, aseq[1]=sel) ==", flush=True)
  GA8 = {"accept8k": ParityGraph(aseq8[:1], tag="r7da8k"), "acceptsel8k": ParityGraph(aseq8[1:], tag="r7da8s")}
  GA3 = {"acceptk": ParityGraph(aseq3[:1], tag="r7ka3k"), "acceptsel": ParityGraph(aseq3[1:], tag="r7ka3s")}
  ta8k = tmin(GA8["accept8k"], flush_g); ta8s = tmin(GA8["acceptsel8k"], flush_g)
  tak = tmin(GA3["acceptk"], flush_g); tas = tmin(GA3["acceptsel"], flush_g)
  print(f"accept8k {ta8k:5.2f} acceptsel8k {ta8s:5.2f} | acceptk {tak:5.2f} acceptsel {tas:5.2f} | delta {ta8k+ta8s-tak-tas:+.2f}", flush=True)

  print("\n== THE 47ms SPLIT (increment by family) ==", flush=True)
  inc = {f: t8[f] - t3[f] for f in fams}
  inc["accept"] = ta8k + ta8s - tak - tas
  tot_inc = sum(inc.values())
  for f, v in sorted(inc.items(), key=lambda x: -x[1]):
    print(f"  {f:10s} {v:7.2f} ms  ({100.0*v/max(tot_inc,1e-9):5.1f}%)", flush=True)
  print(f"  TOTAL     {tot_inc:7.2f} ms  (deep-cycle increment measured {deep_cycle-k2_cycle:6.2f}; ledger delta {(deep_cycle-k2_cycle)-tot_inc:+.2f})", flush=True)
  print("[d3] DONE", flush=True)
