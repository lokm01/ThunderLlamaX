# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7 DECIDER 4 v3: per-K reach law done RIGHT.

The engine at depth K scans matches i <= pos-8-K (all K proposals must be real
verified tokens). The newest IN-RANGE match differs per K (in a period-p loop
the closest self-match at pos-8-p is only reachable when K <= p... it is
reachable whenever pos-8-p <= pos-8-K i.e. K <= p — and then it offers only p
real proposals before running off the fed end). v2's single max-reach best was
wrong; v3 re-selects the newest in-range match per K and measures depth there.
"""
import json, collections
import numpy as np

SNAP = "~/snap100k"
KS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 32]
KCURVE = 32   # the depth-census reach

def main():
  ids = np.load(f"{SNAP}/ids.npy").tolist()
  base = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
  print(f"[d4] prompt {len(ids)} tok; gate continuation {len(base)} tok: {base[:16]}", flush=True)

  # index over the GROWING tok_hist (prompt + emitted) — the engine's scan space
  idx = collections.defaultdict(list)
  a = ids
  for i in range(len(a) - 7):
    idx[tuple(a[i:i+8])].append(i)
  fed = list(ids)
  n = len(base)
  perK = {K: [0, collections.Counter()] for K in KS}   # K -> [hits, md-hist]
  npos = 0
  dhis = collections.Counter()   # depth census at KCURVE reach
  nhitc = 0
  for j in range(n):
    cur = base[j]
    cont = base[j+1:]
    pos = len(fed) + 1
    S = tuple(fed[-7:]) + (cur,)
    full = fed + [cur]
    cands = idx.get(S)
    if cands and pos - 8 >= KS[0]:
      npos += 1
      for K in KS:
        lim = pos - 8 - K
        if lim < 0: continue
        best = -1
        for i in reversed(cands):
          if i <= lim: best = i; break
        if best < 0: continue
        d = 0
        while d < K and d < len(cont):
          ft = best + 8 + d
          if ft >= pos: break
          if full[ft] == cont[d]: d += 1
          else: break
        perK[K][0] += 1
        perK[K][1][d] += 1
        if K == KCURVE:
          # full depth (uncapped by K, capped by cont length + reach)
          d2 = d
          while d2 < 64 and d2 < len(cont):
            ft = best + 8 + d2
            if ft >= pos: break
            if full[ft] == cont[d2]: d2 += 1
            else: break
          dhis[d2] += 1; nhitc += 1
    fed.append(cur)
    idx[tuple(fed[-8:])].append(pos - 8)

  print(f"\n== gate-class per-K table (in-range hits, E[m|hit], all-K) ==", flush=True)
  for K in KS:
    hh, h = perK[K]
    if not npos: break
    if hh == 0:
      print(f"K={K:2d}: hits 0", flush=True); continue
    em = sum(k*v for k, v in h.items()) / hh
    print(f"K={K:2d}: hits {hh}/{npos} ({100.0*hh/npos:5.1f}%)  E[m|hit] {em:6.3f}  all-{K} {100.0*h.get(K,0)/hh:5.1f}%  md {dict(sorted(h.items()))}", flush=True)

  print(f"\n== depth census at reach K={KCURVE} (the decay curve) ==", flush=True)
  print(f"depth hist: {dict(sorted(dhis.items()))}", flush=True)
  nn = nhitc or 1
  cum = 0; curve = {}
  for d in range(64, -1, -1):
    cum += dhis.get(d, 0); curve[d] = cum
  print("P(depth>=d | hit): " + " ".join(f"d{d}={100.0*curve[d]/nn:.0f}%" for d in (1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,32,48,64)), flush=True)

  print(f"\n== ARM 3 v3: projections (gate-class law; in-vivo anchors) ==", flush=True)
  C_K2 = 69.21; F = 0.817
  hh7, h7 = perK[7]
  E7 = sum(k*v for k, v in h7.items()) / hh7
  print(f"offline E[m|hit,7] {E7:.3f} vs in-vivo 7.000 — calibration factor {7.0/max(E7,1e-9):.3f}", flush=True)
  for sname, dK in (("measured +6.9ms/rung", 6.92), ("trend +12.3ms/rung", 12.3), ("fudged +4.0ms/rung", 4.0)):
    C_deep = {7: 116.32}
    for K in (8, 9, 10, 12, 14, 16, 20):
      C_deep[K] = C_deep.get(K-1, C_deep[max(k for k in C_deep if k < K)]) + dK
    print(f"-- {sname} --", flush=True)
    prev = None
    for K in (7, 8, 9, 10, 12, 14, 16, 20):
      hh, h = perK[K]
      if not hh: print(f"  K={K:2d}: no hits", flush=True); continue
      em_off = sum(k*v for k, v in h.items()) / hh
      em_inv = min(K, em_off * (7.0 / max(E7, 1e-9)))
      cyc = F * C_deep[K] + (1 - F) * C_K2
      tpc = F * (em_inv + 1) + (1 - F) * 2.80
      tps = tpc / (cyc / 1e3)
      marg = f"  (+{tps - prev:+.2f})" if prev is not None else ""
      print(f"  K={K:2d}: C_deep {C_deep[K]:6.1f} cyc {cyc:6.1f}ms E[m|deep] {em_inv:5.2f} tok/cyc {tpc:5.2f} -> {tps:6.2f} tok/s{marg}", flush=True)
      prev = tps
  print("\n[d4] DONE", flush=True)

if __name__ == "__main__":
  main()
