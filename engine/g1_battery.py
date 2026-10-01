#!/usr/bin/env python3
"""TLX P0: the G1 battery driver — precision ladder, exposure bias,
conditioning, ctx sweep, coverage. Produces the lever-ranking table."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_sim as cs

HOME = os.path.expanduser("~")
G = f"{HOME}/drafter/phase0/traces/r8_prose"      # globals + class-A anchor
PACKS = {
    "q4_current": f"pack:{HOME}/drafter/ref_pack",
    "pilot_gptq": f"pack:{HOME}/drafter/ttt_results/packs/pilot_gptq",
    "pilot_rtn": f"pack:{HOME}/drafter/ttt_results/packs/pilot_rtn",
    "bf16_hf": f"bf16:{HOME}/drafter/weights_v2",
}
TRACES_B = {
    "gsm8k": f"{HOME}/drafter/phase0/traces/gsm8k",
    "code8k": f"{HOME}/drafter/phase0/traces/code",
    "prose16k": f"{HOME}/drafter/phase0/traces/prose",
    "p100k": f"{HOME}/drafter/phase0/traces/prompt100k",
}


def em(stats):
    return stats["Em_k2"], stats["Em_k4"], stats.get("a_cond", [None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HOME}/drafter/phase0/g1_battery.json")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    res = {"classA": {}, "classB": {}, "ladder": {}}

    # ---- G0/class-A scores for every pack ----
    for tag, spec in PACKS.items():
        if a.quick and tag not in ("q4_current", "pilot_gptq"):
            continue
        W = cs.Weights.from_spec(spec)
        r = cs.run_class_a(G, W, G, verbose=True)
        res["classA"][tag] = r
        print(f"[A] {tag}: sim Em {r['sim']['Em_k4']} engine {r['engine']['Em_k4']}", flush=True)

    # ---- class-B: ctx sweep + exposure bias + conditioning ----
    fh_path = cs.FullHead.build(G)
    for tname, tdir in TRACES_B.items():
        if not os.path.isdir(tdir):
            continue
        W = cs.Weights.from_spec(PACKS["q4_current"])
        if tname == "gsm8k":
            # per-transcript aggregate (weighted by n)
            subs = sorted(x for x in os.listdir(tdir) if x.startswith("t") and os.path.isdir(f"{tdir}/{x}"))
            agg = []
            for s in subs[: (6 if a.quick else 30)]:
                r = cs.run_class_b(f"{tdir}/{s}", W, G, stride=a.stride * 4, verbose=False)
                agg.append(r)
            if agg:
                n_tot = sum(x["n"] for x in agg)
                res["classB"][tname] = dict(
                    n=n_tot,
                    Em_k2=float(sum(x["Em_k2"] * x["n"] for x in agg) / n_tot),
                    Em_k4=float(sum(x["Em_k4"] * x["n"] for x in agg) / n_tot),
                    a1=float(sum(x["a_cond"][0] * x["n"] for x in agg) / n_tot))
        else:
            st = a.stride * (4 if tname == "p100k" else 1)
            r = cs.run_class_b(tdir, W, G, stride=st, verbose=True)
            res["classB"][tname] = r
            # exposure bias: teacher-forced variant
            r2 = cs.run_class_b(tdir, W, G, stride=st, mode="tf", verbose=True)
            res["classB"][tname + "_tf"] = r2

    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[battery] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
