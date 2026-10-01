# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_sim as cs
HOME = os.path.expanduser("~")
G = f"{HOME}/drafter/phase0/traces/r8_prose"
res = {}
W = cs.Weights.from_spec(f"pack:{HOME}/drafter/ref_pack")
jobs = [
    ("gsm8k", f"{HOME}/drafter/phase0/traces/gsm8k", "gsm8k"),
    ("code8k", f"{HOME}/drafter/phase0/traces/code", "corpus"),
    ("prose16k", f"{HOME}/drafter/phase0/traces/prose", "corpus"),
    ("p100k", f"{HOME}/drafter/phase0/traces/prompt100k", "corpus"),
]
for name, tdir, kind in jobs:
    if kind == "gsm8k":
        subs = sorted(x for x in os.listdir(tdir) if x.startswith("t"))[:10]
        agg = []
        for s in subs:
            r = cs.run_class_b(f"{tdir}/{s}", W, G, stride=96, verbose=False)
            agg.append((r["n"], r["Em_k2"], r["Em_k4"], r["a_cond"][0]))
        n = sum(a[0] for a in agg)
        res[name] = dict(n=n, mode="serve",
                         Em_k2=round(sum(a[0]*a[1] for a in agg)/n, 4),
                         Em_k4=round(sum(a[0]*a[2] for a in agg)/n, 4),
                         a1=round(sum(a[0]*a[3] for a in agg)/n, 4))
    else:
        st = 64 if name != "p100k" else 96
        r = cs.run_class_b(tdir, W, G, stride=st, verbose=False)
        r.pop("trace", None)
        res[name] = r
        r2 = cs.run_class_b(tdir, W, G, stride=st, mode="tf", verbose=False)
        r2.pop("trace", None)
        res[name + "_tf"] = r2
    print(f"[fast] {name}: {json.dumps(res[name])}", flush=True)
    json.dump(res, open(f"{HOME}/drafter/phase0/g1_fast.json", "w"), indent=1, default=float)
print("[fast] DONE", flush=True)
