#!/usr/bin/env python3
"""TLX P0: chain_sim vs engine per-cycle hd differential (RUN B2 traces)."""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_sim as cs

TA = os.path.expanduser("~/drafter/phase0/traces/r8_prose")
TB = os.path.expanduser("~/drafter/phase0/traces/r8_prose_hd")
W = cs.Weights.from_spec("pack:" + os.path.expanduser("~/drafter/ref_pack"))
g = cs.load_globals(TA)
tr = cs.TraceA(TA)
kv = cs.KV8.from_dump(tr.kvd, tr.scd)
sim = cs.ChainSim(W, kv, emb_raw=g["emb_raw"], grid512=g["grid512"])
eng_hd = np.load(f"{TB}/cycles_hd.npy")          # [ncyc, 4, 5120]
print("eng hd shape", eng_hd.shape)
rel0 = []
for ci, r in enumerate(tr.cycles):
    deep = bool(r.get("deep_at_entry"))
    clen = 0 if deep else (4 if r.get("prose") else 2)
    hs = tr.hseeds[r["h_idx"]]
    hm_prev = None
    for i in range(clen):
        tok = r["cur"] if i == 0 else int(r["dring"][i - 1])
        hm = hs if i == 0 else hm_prev
        hd, _hi, _p = sim.step(tok, hm, r["pos"] + i, with_head=False)
        hm_prev = hd
        e = eng_hd[ci, i]
        d = np.abs(hd - e)
        rel = d.max() / max(np.abs(e).max(), 1e-9)
        if i == 0:
            rel0.append((ci, float(d.max()), float(rel)))
        if ci < 16:
            print(f"cyc{ci} st{i}: maxabs {d.max():.4e} relmax {rel:.3e} "
                  f"cos {float(np.dot(hd,e)/(np.linalg.norm(hd)*np.linalg.norm(e)+1e-30)):.6f}")
print("--- step0 hd error quantiles (all cycles) ---")
r_ = np.array([x[2] for x in rel0])
a_ = np.array([x[1] for x in rel0])
print("relmax med/p90/max:", np.median(r_), np.percentile(r_, 90), r_.max())
print("absmax med/p90/max:", np.median(a_), np.percentile(a_, 90), a_.max())
json.dump({"rel0": rel0}, open("/tmp/hd_diff.json", "w"))
