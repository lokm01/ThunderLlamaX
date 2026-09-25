# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R4 deep-K offline analysis v2: E[m|hit] at K=2..6 + best-i distance
distribution. Proposal safety contract: proposal index i+8+t must be <= pos-1
(hist[pos-1]=cur is a REAL verified token id; hist>=pos is unwritten garbage
-> OOB embedding row). Two reach policies measured:
  reachA: all K proposals <= pos-1 (cur usable as last prop)
  reachB: all K proposals <= pos-2 (R3-strict, cur excluded)
"""
import json, os, collections
import numpy as np

def analyze(ids, out, tag, nmax=None):
    out = list(out); N0 = len(out)
    print(f"\n=== {tag}: prompt {len(ids)} tok, continuation {N0} tok ===")
    dist = collections.Counter()
    for K in (2, 3, 4, 5, 6):
        N = N0 - K
        if nmax: N = min(N, nmax)
        res = {}
        for pol, lim in (("A", -1), ("B", -2)):   # proposal max index = pos+lim
            hits = 0; md = [0]*(K+1)
            for j in range(N):
                fed = np.array(ids + out[:j], dtype=np.int64)
                pos = len(fed)
                S = np.concatenate([fed[max(0, pos-7):], [out[j]]]).astype(np.int64)
                W = len(S)
                imax = pos - 8 - K   # DEEP-K scan range: i+8+K-1 <= pos-1 (K props written; cur usable)
                if imax < 0: continue
                ok = np.ones(imax+1, dtype=bool)
                l = np.zeros(imax+1, dtype=np.int64)
                for u in range(W):
                    ok &= (fed[u:u+imax+1] == S[u])
                    l += ok
                cand = np.nonzero(l >= W)[0]
                if len(cand) == 0: continue
                ll = l[cand]
                best = int(cand[np.lexsort((cand, ll))[-1]])
                if pol == "A": dist[pos-1-best] += 1
                if best + 8 + K - 1 > pos + lim: continue
                props = [int(fed[best+8+t]) if best+8+t < pos else None for t in range(K)]
                if any(p is None for p in props): continue
                tgt = out[j+1:j+1+K]
                m = 0
                for t in range(K):
                    if props[t] == tgt[t]: m += 1
                    else: break
                hits += 1; md[m] += 1
            hr = hits/N if N else 0
            e_mhit = sum(i*md[i] for i in range(K+1))/hits if hits else 0.0
            res[pol] = (hr, e_mhit, md, hits, N)
        hrA, eA, mdA, hA, NA = res["A"]; hrB, eB, mdB, hB, NB = res["B"]
        print(f"K={K}: reachA(cur-ok) hit {hA}/{NA} ({hrA*100:5.1f}%) E[m|hit] {eA:.3f} md {mdA} | "
              f"reachB(strict) hit {hB}/{NB} ({hrB*100:5.1f}%) E[m|hit] {eB:.3f}", flush=True)
    print(f"[best-i dist] pos-1-best: {dict(sorted(dist.items())[:12])}", flush=True)

if __name__ == "__main__":
    SNAP = "~/snap100k"
    ids = np.load(f"{SNAP}/ids.npy").tolist()
    base = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
    analyze(ids, base, "(a) 100k base continuation")
    rng = np.random.default_rng(0)
    q0 = int(rng.integers(1000, len(ids)-2000))
    stream = ids + ids[q0:q0+60]
    analyze(ids, stream, "(c-sim) quote-heavy ceiling", nmax=58)
