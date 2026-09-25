# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R3 LOOKUP offline instrumentation: n-gram suffix-match drafter statistics
on recorded target-greedy continuations. Exact engine semantics:
- at cycle j: fed = prompt + out[0..j-1]; cur = out[j] (target argmax, Tier-1).
- suffix S = last 8 tokens of fed+[cur]; window W(i) = hist[i..i+7].
- scan i in [0, pos-11] (proposal tokens hist[i+l], hist[i+l+1] both fed).
- l(i) = leading match length of W(i) vs S; best = max (l, i) lexicographic
  (max l, tie -> max i = most recent). HIT = l >= THR.
- m = accepted count: p1==out[j+1] (+ p2==out[j+2] if p1 matched).
Outputs: hit-rate, alpha_lookup, m-distribution, blended tok/cyc vs thresholds.
"""
import json, sys
import numpy as np

def analyze(ids, out, tag, mtp_tpc=2.78, cyc_ms=69.0):
  out = list(out)
  N = len(out) - 2   # need out[j+1], out[j+2]
  print(f"\n=== {tag}: prompt {len(ids)} tok, continuation {len(out)} tok ===")
  for THR in (4, 5, 6, 7, 8):
    hits = 0; mcnt = [0, 0, 0]; alphas = []
    for j in range(N):
      fed = np.array(ids + out[:j], dtype=np.int64)   # pos = len(fed)
      pos = len(fed)
      S = np.concatenate([fed[max(0, pos-7):], [out[j]]]).astype(np.int64)
      W = len(S)
      if W < THR: continue
      imax = pos - W - 3   # proposals at i+l, i+l+1 must be fed: i+W+1 <= pos-2
      if imax < 0: continue
      # vectorized leading-match length over i in [0, imax]
      ok = np.ones(imax+1, dtype=bool)
      l = np.zeros(imax+1, dtype=np.int64)
      for u in range(W):
        ok &= (fed[u:u+imax+1] == S[u])
        l += ok
      cand = np.nonzero(l >= THR)[0]
      if len(cand) == 0: continue
      ll = l[cand]
      best = cand[np.lexsort((cand, ll))[-1]]   # max l, tie max i
      bl = int(l[best])
      hits += 1
      p1 = fed[best + bl]; p2 = fed[best + bl + 1]
      m = 1 if p1 == out[j+1] else 0
      if m == 1 and p2 == out[j+2]: m = 2
      mcnt[m] += 1
      alphas.append(m)
    nc = N
    hit_rate = hits / nc
    if hits:
      e_m = sum(mcnt) / hits - 0 + (mcnt[1] + 2*mcnt[2]) / hits - (mcnt[0]*0)  # E[m] among hits
      e_m = (mcnt[1] + 2*mcnt[2]) / hits
      alpha1 = (mcnt[1] + mcnt[2]) / hits   # p1 accepted
    else:
      e_m = alpha1 = 0.0
    # blended: hit cycles use lookup m; miss cycles use MTP m (E=1.78)
    mtp_em = mtp_tpc - 1.0
    e_tpc = (hits/nc) * (1 + e_m) + (1 - hits/nc) * (1 + mtp_em)
    print(f"THR>={THR}: hit {hits}/{nc} ({hit_rate*100:5.1f}%)  alpha1 {alpha1:.3f}  E[m|hit] {e_m:.3f}  "
          f"m-dist {mcnt}  blended tok/cyc {e_tpc:.3f} -> {e_tpc/(cyc_ms/1000):.1f} tok/s (cycle {cyc_ms:.0f}ms)")

if __name__ == "__main__":
  SNAP = "~/snap100k"
  ids = np.load(f"{SNAP}/ids.npy").tolist()
  out = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
  if isinstance(out, dict): out = out.get("tokens", out.get("outs"))
  analyze(ids, out, "(a) 100k base prompt continuation")
  # (b) real follow-up generation at 100k: history = prompt + 200-tok delta,
  # continuation = 42 real model tokens (M1-C gate3 resident output).
  dl = np.load(f"{SNAP}/gate3_delta.npy").tolist()
  orr = np.load(f"{SNAP}/gate3_out_resident.npy").tolist()
  analyze(ids + dl, orr, "(b-real) 100k follow-up reply (gate3 resident)")
  # (c) quote-heavy simulated ceiling: continuation = verbatim 60-tok quote of a
  # LATE doc span (the doc-QA "model quotes perfectly" upper bound).
  fed = ids + dl
  span = ids[-3000:-2940]   # 60 tokens from the doc tail (before the delta)
  analyze(fed, list(span), "(c-sim) quote-heavy doc-QA ceiling (verbatim quote)")
