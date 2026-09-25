# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 RUNG B: the graded-LOOKUP offline sim. LMIN in {6,7,8} x K in {4,6,8}
on three corpora (gate-class repeat region / prose-at-100k follow-up / quote
ceiling), engine-exact scan semantics (r7_lut_hist3 laws + the lookup9/10
conservative reach iend = pos-8-K-1; suffix = last-7-fed + cur; best = max l
then max i). Trigger-chain cycle model (DEEP_TRIG=1): deep_j = prev cycle's
hit flag (l >= LMIN); costs c2=66.0ms, cd(K)=104.65+7*(K-8)ms (in-vivo
anchors); stale-dring deep-miss m=0.7; k2 baseline 2.78 tok/cyc; a k2 cycle
whose lookup fires consumes 1+min(m,2) tokens (dring0/1 only).

NOTE (approximation): match selection uses the widest reach (K=8); at K=4/6
the selected newest match may be out of range -> arm under-counts fires (the
K=8 arm is exact)."""
import json, collections
import numpy as np

SNAP = "~/snap100k"
LMINS = (6, 7, 8)
KS = (4, 6, 8)
C2 = 66.0
CD = {K: 104.65 + 7.0 * (K - 8) for K in KS}
M_STALE = 0.7
TOK_K2_MISS = 2.78

def sim(ids, out, tag, nmax=None):
  out = list(out); N0 = min(len(out), nmax) if nmax else len(out)
  print(f"\n=== {tag}: hist {len(ids)} tok, continuation {N0} tok ===", flush=True)
  fed = list(ids)
  idxp = {L: collections.defaultdict(list) for L in LMINS}
  for i in range(len(fed) - 7):
    w = tuple(fed[i:i+8])
    for L in LMINS:
      idxp[L][w[:L]].append(i)
  rows = []   # (l, i) per position; l=0 none
  for j in range(N0):
    cur = out[j]
    pos = len(fed) + 1
    S = tuple(fed[-7:]) + (cur,)
    full = fed + [cur]
    if pos < 19:
      rows.append((0, -1)); fed.append(cur); continue
    iend8 = pos - 17   # widest reach (K=8 arm, the engine's conservative law)
    best = (0, -1)
    for L in (8, 7, 6):
      cands = idxp[L].get(S[:L])
      if cands:
        for i in reversed(cands):
          if i <= iend8:
            best = (L, i); break
        if best[0]:
          break
    rows.append(best)
    # per-K acceptance measured lazily below (needs full)
    acc = {}
    if best[0]:
      cont = out[j+1:]
      for K in KS:
        if best[1] > pos - 8 - K - 1:
          acc[K] = None; continue
        m = 0
        while m < K and m < len(cont):
          ft = best[1] + 8 + m
          if ft >= pos: break
          if full[ft] == cont[m]: m += 1
          else: break
        acc[K] = m
    rows[-1] = (best[0], best[1], acc)
    fed.append(cur)
    w8 = tuple(fed[-8:])
    for L in LMINS:
      idxp[L][w8[:L]].append(len(fed) - 8)
  ncyc = len(rows)
  for LMIN in LMINS:
    for K in KS:
      fire = 0; md = collections.Counter()
      toks = 0.0; ms = 0.0
      prev_hit = 0
      for r in rows:
        l, i, acc = r
        m = acc.get(K)
        hit = 1 if (m is not None and l >= LMIN) else 0
        if prev_hit:
          ms += CD[K]
          if hit:
            fire += 1; md[m] += 1; toks += m + 1
          else:
            toks += 1 + M_STALE; md[0] += 1
        else:
          ms += C2
          toks += (1 + min(m, 2)) if hit else TOK_K2_MISS
        prev_hit = hit
      e_m = sum(k * v for k, v in md.items()) / max(sum(md.values()), 1)
      print(f"LMIN={LMIN} K={K}: deep-fires {fire}/{ncyc} ({100.0*fire/max(ncyc,1):5.1f}%) "
            f"E[m|deep-fire] {e_m:.3f} md {dict(sorted(md.items()))} "
            f"-> proj {toks/ms*1e3:6.2f} tok/s", flush=True)
  ldist = collections.Counter(r[0] for r in rows)
  print(f"[l-dist] {dict(sorted(ldist.items()))}", flush=True)

def main():
  ids = np.load(f"{SNAP}/ids.npy").tolist()
  base = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
  sim(ids, base, "(gate) 100k repeat-region continuation")
  prose = np.load("~/r8_prose_ids.npy").tolist()
  sim(ids, prose, "(prose) 100k-ctx follow-up novel reply", nmax=1000)
  rng = np.random.default_rng(0)
  q0 = int(rng.integers(1000, len(ids) - 2000))
  sim(ids, ids[q0:q0+120], "(quote-ceiling) verbatim doc span", nmax=118)

if __name__ == "__main__":
  main()
