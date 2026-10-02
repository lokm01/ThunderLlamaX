#!/usr/bin/env python3

# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX P2 v3 — the Stage-B v3 scoring driver (Mac, zero GPU).

Modes:
  full   = r8 + canary + cross-class (the ship battery; bars in the header)
  curve  = canary k2/k4 only (the LR/steps curve — fast, no cross)
Control = the current engine pack (ref_pack): canary k2 0.979 / r8 0.549 /
gsm 0.764 / prose16k 0.862 / code8k 0.861.
Bars: canary k2 >= 1.05 (primary), r8 k2 >= 0.6, cross no-regression.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_sim as cs

HOME = os.path.expanduser("~")
G = f"{HOME}/drafter/phase0/traces/r8_prose"
CANARY = f"{HOME}/drafter/anchor_scale_mirror"
TRACES_B = {
    "gsm8k": f"{HOME}/drafter/phase0/traces/gsm8k",
    "code8k": f"{HOME}/drafter/phase0/traces/code",
    "prose16k": f"{HOME}/drafter/phase0/traces/prose",
}


def score_r8(spec):
    W = cs.Weights.from_spec(spec)
    return cs.run_class_a(G, W, G, max_cycles=None, chain_len=4, cond="engine", verbose=False)


def score_canary(spec, max_per=60):
    W = cs.Weights.from_spec(spec)
    dirs = []
    for d in sorted(glob.glob(f"{CANARY}/bk_*")):
        try:
            if json.load(open(f"{d}/manifest.json")).get("split") == "canary":
                dirs.append(d)
        except Exception:
            pass
    per = []
    for d in dirs:
        try:
            r = cs.run_class_a(d, W, G, max_cycles=max_per, chain_len=4, cond="engine", verbose=False)
        except Exception as e:
            print(f"[canary] {os.path.basename(d)} SKIP ({e})")
            continue
        per.append(dict(sess=os.path.basename(d), sim=r["sim"], engine=r["engine"]))
        print(f"[canary] {os.path.basename(d)}: Em_k2 {r['sim']['Em_k2']:.3f} "
              f"k4 {r['sim']['Em_k4']:.3f} a1 {r['sim']['a_cond'][0]:.3f} "
              f"m_dist {r['sim']['m_dist']}", flush=True)
    if not per:
        return None
    return dict(
        Em_k2=float(np.mean([p["sim"]["Em_k2"] for p in per])),
        Em_k4=float(np.mean([p["sim"]["Em_k4"] for p in per])),
        a1=float(np.mean([p["sim"]["a_cond"][0] for p in per])),
        per=per,
    )


def score_cross(spec):
    W = cs.Weights.from_spec(spec)
    out = {}
    for tname, tdir in TRACES_B.items():
        if not os.path.isdir(tdir):
            continue
        if tname == "gsm8k":
            subs = sorted(x for x in os.listdir(tdir) if x.startswith("t") and os.path.isdir(f"{tdir}/{x}"))
            ems = []
            for s in subs[:30]:
                r = cs.run_class_b(f"{tdir}/{s}", W, G, stride=64, verbose=False)
                ems.append(r["Em_k2"])
            out[tname] = dict(Em_k2=float(np.mean(ems)), n=len(ems))
        else:
            r = cs.run_class_b(tdir, W, G, stride=16, verbose=False)
            out[tname] = dict(Em_k2=r["Em_k2"], Em_k4=r["Em_k4"])
        print(f"[cross] {tname}: {out[tname]}", flush=True)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith(("pack:", "bf16:")) else "full"
    packs = [p for p in sys.argv[1:] if p.startswith(("pack:", "bf16:"))]
    res = {}
    outf = f"{HOME}/drafter/phase0/score_v3.json"
    if os.path.exists(outf):
        res = json.load(open(outf))
    for spec in packs:
        tag = spec.rstrip("/").split("/")[-1]
        print(f"=== {tag} ({spec}) [{mode}] ===", flush=True)
        entry = dict(spec=spec, mode=mode)
        if mode in ("full", "curve"):
            entry["canary"] = score_canary(spec)
        if mode in ("full", "cross"):
            entry["r8"] = score_r8(spec) if mode == "full" else None
            entry["cross"] = score_cross(spec)
        res[tag] = entry
        json.dump(res, open(outf, "w"), indent=1, default=float)
        if entry.get("canary"):
            c = entry["canary"]
            line = f"[v3score] {tag}: canary k2 {c['Em_k2']:.4f} k4 {c['Em_k4']:.4f} a1 {c['a1']:.4f}"
            if entry.get("r8"):
                line += f" | r8 k2 {entry['r8']['sim']['Em_k2']:.4f}"
            if entry.get("cross"):
                line += f" | gsm {entry['cross']['gsm8k']['Em_k2']:.3f} prose {entry['cross']['prose16k']['Em_k2']:.3f} code {entry['cross']['code8k']['Em_k2']:.3f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
