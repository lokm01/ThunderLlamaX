#!/usr/bin/env python3
"""MM P0 D6 — OCCUPANCY MONTE CARLO: E[distinct experts | M tokens] for the
256-expert top-8 router, and the decode envelope at a parameterized gather BW.

Models (the envelope):
  - uniform-256 popularity (the MM_PLAN lower-bound assumption)
  - Dirichlet(alpha) popularity sweep over the 256-simplex (alpha small =
    concentrated; alpha -> inf = uniform)
  - Zipf(s) popularity sweep (heavy-tail stress)

Each token = Gumbel-top-8 draw (the exact softmax+top-k sampling). We report
E[|union of M draws|] for M in {1,2,3,8,10,11}, the per-cycle byte model at the
D2-measured slab sizes, and tok/s envelopes vs gather GB/s.

Usage: ~/tg311/bin/python MM_P0_d6_occupancy.py
"""
import numpy as np

rng = np.random.default_rng(42)
E, TOPK, LAYERS = 256, 8, 40

def gumbel_topk_draws(logits, M):
    """M tokens x top-8 experts via Gumbel perturbation (exact softmax-topk)."""
    g = rng.gumbel(size=(M, logits.size))
    return np.argpartition(g + logits, -TOPK, axis=1)[:, -TOPK:]

def e_distinct(pop, M, trials=2000):
    logits = np.log(pop)
    out = np.empty(trials)
    for t in range(trials):
        d = gumbel_topk_draws(logits, M)
        out[t] = np.unique(d).size
    return out.mean(), out.std()

def main():
    print("== D6 occupancy Monte Carlo: E[distinct experts | M] (top-8 of 256, 2000 trials) ==")
    models = {"uniform": np.full(E, 1.0/E)}
    for a in (0.05, 0.2, 0.5, 1.0, 3.0, 10.0):
        models[f"Dir({a})"] = rng.dirichlet(np.full(E, a))
    for s in (0.3, 0.7, 1.1, 1.5):
        p = 1.0/np.arange(1, E+1)**s
        models[f"Zipf({s})"] = p/p.sum()
    Ms = [1, 2, 3, 8, 10, 11]
    print(f"  {'model':10s} " + " ".join(f"M={m:<6d}" for m in Ms) + "  eff-experts(M=1)")
    tab = {}
    for name, pop in models.items():
        vals = []
        for M in Ms:
            mu, sd = e_distinct(pop, M)
            vals.append(mu)
        # effective number of experts at M=1 (participation ratio of top-8 mass)
        order = np.argsort(pop)[::-1][:TOPK]
        eff1 = 1.0/np.sum((pop[order]/pop[order].sum())**2)
        tab[name] = vals
        print(f"  {name:10s} " + " ".join(f"{v:7.2f}" for v in vals) + f"   {eff1:6.1f}")

    # ---- decode envelope at the D2-measured slab sizes (exact byte model, GiB) ----
    print("\n== decode envelope (per-cycle bytes & tok/s vs gather GB/s) ==")
    GiB = 1 << 30
    SLAB = {"IQ4_XS-tier": 1458176/GiB, "IQ3_S-tier": 1171840/GiB}   # modal per-expert slab
    # dense per cycle (weights read once per probe cycle; Q8_0 1.0625 B/param):
    gdn_tr   = 30 * (2048*8192 + 2048*4096 + 4096*2048 + 4*8192 + 2*2048*32) * 1.0625 / GiB
    attn_tr  = 10 * (2048*8192 + 2*2048*512 + 4096*2048) * 1.0625 / GiB
    shared   = 40 * (2*2048*512 + 512*2048) * 1.0625 / GiB
    router   = 40 * 2048*256*4 / GiB
    head_sl  = 40960*2048*0.8203125 / GiB        # Q6_K vocab-slice head, batched (one read/cycle)
    head_fl  = 248320*2048*0.8203125 / GiB       # full-vocab head, batched
    kv96     = 96*1024 * 5120 / GiB              # int8-KV read @96k ctx (batched Q)
    gdn_state = 30 * 2*(2<<20) / GiB             # state read+write once per layer (smem-resident design, D5)
    dense = gdn_tr + attn_tr + shared + router + head_sl + kv96 + gdn_state
    print(f"  dense/cycle: GDN {gdn_tr:.3f} + attn {attn_tr:.3f} + shexp {shared:.3f} + router {router:.3f}"
          f" + head(slice) {head_sl:.3f} + KV@96k {kv96:.3f} + state {gdn_state:.3f} = {dense:.3f} GiB"
          f"  (full-vocab head would add {head_fl-head_sl:.2f})")
    # probe positions M = K+1 (n-gram draft = no expert reads; probe verifies K+1)
    KPOS = {2: 3, 8: 9, 10: 11}
    EM   = {"prose": {2: 1.30, 8: 1.60, 10: 1.70},     # n-gram acceptance classes (our engine's law)
            "quote": {2: 1.95, 8: 8.00, 10: 10.00}}    # deep-hit ALL-K accepts (R5/R8 E[m|deep]=K law)
    for tier, slab in SLAB.items():
        print(f"\n  [{tier}] slab/expert = {slab*GiB:.0f} B")
        for gb in (120, 180, 240, 300):
            row = []
            for K, M in KPOS.items():
                du = 256*(1-(1-8/256)**M)          # uniform formula (exact)
                dz = {3: 22.06, 9: 50.48*9/8, 11: 64.68}[M]   # Zipf(0.7) from the MC table
                by = 40*du*slab + dense
                bz = 40*dz*slab + dense
                tp = EM["prose"][K] / (by/gb)
                tq = EM["quote"][K] / (bz/gb)
                row.append(f"K={K:2d}: {by:.2f}G prose {tp:5.0f} | zipf {bz:.2f}G quote {tq:5.0f}")
            print(f"    @{gb:3d} GB/s  " + " ; ".join(row))

if __name__ == "__main__":
    main()
