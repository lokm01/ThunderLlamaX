# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7 DECIDER 4 v4: projection with f_deep(K) varying by the in-range hit law."""
import json, collections
import numpy as np
SNAP = "~/snap100k"
KS = [2,3,4,5,6,7,8,9,10,11,12,14,16,20,24,32]

ids = np.load(f"{SNAP}/ids.npy").tolist()
base = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
idx = collections.defaultdict(list)
for i in range(len(ids)-7): idx[tuple(ids[i:i+8])].append(i)
fed = list(ids); npos = 0
perK = {K: [0, collections.Counter()] for K in KS}
for j in range(len(base)):
    cur = base[j]; cont = base[j+1:]; pos = len(fed)+1
    S = tuple(fed[-7:]) + (cur,); full = fed + [cur]
    cands = idx.get(S)
    if cands and pos-8 >= 2:
        npos += 1
        for K in KS:
            lim = pos-8-K
            if lim < 0: continue
            best = -1
            for i in reversed(cands):
                if i <= lim: best = i; break
            if best < 0: continue
            d = 0
            while d < K and d < len(cont):
                ft = best+8+d
                if ft >= pos: break
                if full[ft] == cont[d]: d += 1
                else: break
            perK[K][0] += 1; perK[K][1][d] += 1
    fed.append(cur)
    idx[tuple(fed[-8:])].append(pos-8)

print(f"[d4v4] npos {npos}")
ir = {}
for K in KS:
    hh, h = perK[K]
    ir[K] = hh/npos if npos else 0
    em = sum(k*v for k,v in h.items())/hh if hh else 0
    print(f"K={K:2d}: in-range {100*ir[K]:5.1f}% E[m|hit] {em:6.3f}")
F7 = 0.817; E7 = sum(k*v for k,v in perK[7][1].items())/perK[7][0]
CAL = 7.0/E7
print(f"E7 {E7:.3f} calib {CAL:.3f}; f_deep(K) = 0.817 * ir(K)/ir(7)")
print()
C_K2 = 69.21
for sname, dK in (("measured +6.9ms/rung", 6.92), ("trend +12.3ms/rung", 12.3), ("fudged +4.0ms/rung", 4.0)):
    C_deep = {7: 116.32}
    for K in (8,9,10,12,14,16,20):
        C_deep[K] = C_deep[max(k for k in C_deep if k < K)] + dK
    print(f"-- {sname} --")
    prev = None
    for K in (7,8,9,10,12,14,16,20):
        hh, h = perK[K]
        if not hh: print(f"  K={K:2d}: no hits"); continue
        em = sum(k*v for k,v in h.items())/hh
        em_inv = min(K, em*CAL)
        f = F7 * ir[K]/ir[7]
        cyc = f*C_deep[K] + (1-f)*C_K2
        tpc = f*(em_inv+1) + (1-f)*2.80
        tps = tpc/(cyc/1e3)
        marg = f"  ({tps-prev:+6.2f})" if prev is not None else ""
        print(f"  K={K:2d}: f_deep {100*f:5.1f}% C_deep {C_deep[K]:6.1f} cyc {cyc:6.1f} E[m|deep] {em_inv:5.2f} tok/cyc {tpc:5.2f} -> {tps:6.2f} tok/s{marg}")
        prev = tps
