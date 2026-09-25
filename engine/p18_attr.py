# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P18: THE GROWTH-POOL FORENSICS (in-plan; the p15_attr law).

The question: chunks 207.9ms @pos0 -> 300.5 @97k-end (+92.6) while standalone
attention growth is only ~7.7 — attribute the ~85ms.

Method (one boot, REAL snap100k KV resident, graphs built ONCE — pos enters
ONLY through the device buffers pos_arr64 (trunk append) / pos_w64 (attention
extent + dfill window pos), so the SAME graphs run at any pos):
  1. FULL-CHUNK curve: the canonical captured chunk (trunk+dfill; w64h arm
     below ATTN_THR, w64 arm above) min-of-N synced at the rebuild's pos
     sample + a FINE sweep across the observed cliff window (48.6k..62k).
  2. TRUNK-ONLY vs DFILL-ONLY graphs at each pos (the in-chunk dfill share).
  3. PER-CLASS isolated graphs (p15_attr classify) at pos {0, 48640, 97280}.
  4. POS-ABLATION at the cliff + 100k-end: same full graph with pos_w64 LOW
     (attention extent + dfill pos low) vs pos_arr64 LOW (trunk append low).
  5. QUARTER splits of the full plan, each timed alone (localize growth to a
     contiguous launch range).
Timing-only harness (numerics NOT gated): repeated runs clobber chunk rows;
kernels are loop-count-pos-driven so timing is data-honest.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
import pf_prefill
from pf_prefill import PfGraph, m64_set

SWEEP = [0, 9728, 19456, 29184, 38912, 48640, 58368, 68096, 77824, 87552, 97280]
CLIFF = [48640, 50000, 51000, 52000, 53000, 54040, 54560, 55000, 56000, 57000,
         58000, 58368, 59000, 60000, 61760, 63000]
CLS_POS = [0, 48640, 97280]
ABL_POS = [58368, 97280]

PLAN_CLS = [("pfk_emb16", "emb"), ("pfk_n16", "nrm_n"), ("pfk_ab16", "nrm_ab"), ("pfk_hh16", "nrm_hh"),
            ("pfk_pre64", "pre64"), ("pfaw", "attn"), ("pfcw", "attn_comb"), ("pfs64", "scan"),
            ("attnqkv", "gemm_qkv"), ("gdnqg", "gemm_qg"), ("ffn", "gemm_ffn"), ("iq3d", "gemm_fd"),
            ("iq3s", "gemm_oa"), ("q8o", "gemm_og"), ("iq3o", "gemm_og")]
DF_CLS = [("pfk_rec16", "df_rec"), ("pfk_emb16", "df_emb"), ("pfd_dnorm16", "df_dn"),
          ("pfg_ehd", "df_ehd"), ("pfk_n16", "df_n"), ("dqkv", "df_dqkv"), ("pfk_pre16", "df_pre")]


def _cls(p, table):
  nm = getattr(p, "name", "?")
  for tag, c in table:
    if tag in nm:
      return c
  return "misc_" + nm[:14]


def run(E, ids, CTXK):
  P = E.P
  rng = np.random.default_rng(0)
  IDS64 = np.array([int(t) for t in rng.integers(1000, 60000, 64)], dtype=np.int32)
  print(f"[p18] growth-pool forensics: sweep={SWEEP}", flush=True)

  m64_set(True)
  pf_prefill.ensure64(E)
  dseq = pf_prefill._pf_dfill_seq64(E)
  plan_hi = list(E._pf_plan26_64)   # the w64 arm (pos >= ATTN_THR)
  plan_lo = list(E._pf_plan64)      # the w64h arm (pos < ATTN_THR)
  full_hi, full_lo = plan_hi + dseq, plan_lo + dseq
  print(f"[p18] plans: hi={len(plan_hi)} dfill={len(dseq)} full={len(full_hi)}", flush=True)

  def split(seq, n, tag):
    k = (len(seq) + n - 1) // n
    return [PfGraph(seq[i:i + k], f"{tag}{j}") for j, i in enumerate(range(0, len(seq), k))]

  G_FULL_lo = split(full_lo, 2, "flo")
  G_FULL_hi = split(full_hi, 2, "fhi")
  G_TRUNK = split(plan_hi, 2, "trk")
  G_DF = [PfGraph(dseq, "dfl")]
  G_Q = split(full_hi, 4, "q")

  groups, groups_lo = {}, {}
  for t in plan_hi:
    groups.setdefault(_cls(t[0], PLAN_CLS), []).append(t)
  for t in dseq:
    groups.setdefault(_cls(t[0], DF_CLS), []).append(t)
  for t in plan_lo:
    groups_lo.setdefault(_cls(t[0], PLAN_CLS), []).append(t)
  GCLS = {c: PfGraph(seq, f"c_{c}") for c, seq in sorted(groups.items())}
  GCLS_LO = {c: PfGraph(seq, f"cl_{c}") for c, seq in sorted(groups_lo.items()) if c in ("attn", "attn_comb")}
  print(f"[p18] graph inventory: FULL_lo/hi {len(G_FULL_lo)}/{len(G_FULL_hi)} TRUNK {len(G_TRUNK)} "
        f"DF 1 Q {len(G_Q)} classes {len(GCLS)} lo-classes {len(GCLS_LO)}", flush=True)

  def set_pos(p, arr=None, w64=None):
    P.win_up("ids64", 0, IDS64)
    P.win_up("pos_arr64", 0, np.array([p if arr is None else arr], dtype=np.int32))
    pw = [p, p + 16, p + 32, p + 48] if w64 is None else list(w64)
    P.win_up("pos_w64", 0, np.array(pw, dtype=np.int32))
    dev.synchronize()

  def tmin(gs, n=4, warm=True):
    best = 1e9
    for i in range(n + (1 if warm else 0)):
      prev = dev.timeline_value - 1
      t1 = time.perf_counter()
      for g in gs[:-1]:
        v = dev.next_timeline(); g.submit(prev, v); prev = v
      v = dev.next_timeline(); gs[-1].submit(prev, v)
      dev.timeline_signal.wait(v)
      dt = time.perf_counter() - t1
      if i >= (1 if warm else 0):
        best = min(best, dt)
    return best * 1e3

  # warm-up pair FIRST (the P17 bare-world first-pair law)
  print("[p18] warm-up pair", flush=True)
  set_pos(9728)
  tmin(G_FULL_hi, n=2, warm=False)
  tmin(G_FULL_hi, n=2, warm=False)
  dev.synchronize()

  print("== ARM 1: FULL / TRUNK / DFILL vs pos (the curve) ==", flush=True)
  print(f"{'pos':>7} {'FULL':>8} {'TRUNK':>8} {'DFILL':>7} {'arm':>5}", flush=True)
  base = {}
  for p in SWEEP:
    set_pos(p)
    gs = G_FULL_lo if p < 8192 else G_FULL_hi
    f = tmin(gs, 4)
    t = tmin(G_TRUNK, 4) if p >= 8192 else float("nan")
    dd = tmin(G_DF, 4)
    base[p] = f
    print(f"{p:7d} {f:8.2f} {t:8.2f} {dd:7.2f} {'lo' if p < 8192 else 'hi':>5}", flush=True)

  print("== ARM 2: the cliff, FINE sweep (FULL only) ==", flush=True)
  for p in CLIFF:
    set_pos(p)
    f = tmin(G_FULL_hi, 3)
    print(f"{p:7d} {f:8.2f}  (vs 48640: {f - base[48640]:+7.2f})", flush=True)

  print("== ARM 3: per-class isolated graphs ==", flush=True)
  for p in CLS_POS:
    set_pos(p)
    print(f"-- pos {p} --", flush=True)
    rows = []
    for c, g in GCLS.items():
      ms = tmin([g], 3)
      rows.append((c, ms))
    if p < 8192:
      for c, g in GCLS_LO.items():
        rows.append((c + "_w64h", tmin([g], 3)))
    tot = 0.0
    for c, ms in sorted(rows, key=lambda r: -r[1]):
      n = len(groups.get(c, groups_lo.get(c[:-5] if c.endswith('_w64h') else c, [])))
      tot += ms
      print(f"[p18] {c:12s} n={n:4d} {ms:7.2f}ms ({ms / max(n, 1) * 1000:6.0f}us/launch)", flush=True)
    print(f"[p18] SUM(isolated) @pos {p}: {tot:.2f}ms", flush=True)

  print("== ARM 4: pos-ablation (same graph, pos via device buffers) ==", flush=True)
  for p in ABL_POS:
    set_pos(p)
    b = tmin(G_FULL_hi, 4)
    set_pos(p, arr=p, w64=[16, 32, 48, 64])          # attention extent + dfill pos LOW
    a1 = tmin(G_FULL_hi, 4)
    set_pos(p, arr=0, w64=[p, p + 16, p + 32, p + 48])  # trunk append LOW
    a2 = tmin(G_FULL_hi, 4)
    set_pos(p, arr=0, w64=[16, 32, 48, 64])          # everything LOW
    a3 = tmin(G_FULL_hi, 4)
    print(f"pos {p:6d}: base {b:7.2f} | attn+dfill-pos LOW {a1:7.2f} ({a1 - b:+6.2f}) | "
          f"trunk-append LOW {a2:7.2f} ({a2 - b:+6.2f}) | both LOW {a3:7.2f} ({a3 - b:+6.2f})", flush=True)

  print("== ARM 5: quarter localization (each quarter timed alone) ==", flush=True)
  for p in [9728, 48640, 97280]:
    set_pos(p)
    parts = [tmin([g], 3) for g in G_Q]
    print(f"pos {p:6d}: " + " | ".join(f"q{i} {m:6.2f}" for i, m in enumerate(parts)) +
          f" | sum {sum(parts):7.2f}", flush=True)

  print("[p18] DONE", flush=True)
