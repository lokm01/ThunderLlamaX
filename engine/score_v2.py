#!/usr/bin/env python3

# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX P2 — the Stage-B v2 scoring driver (Mac, zero GPU).

Scores a candidate pack through the CALIBRATED chain_sim:
  1. r8_prose held-out (PRIMARY — untouched eval; G2 bar Em_k2 >= 0.8-1.0)
  2. anchor canary sessions (the generalization canary — must move WITH r8;
     pooled m-dist + per-session Em)
  3. cross-class: gsm8k (>= +0.32 over control 0.816), prose16k + code8k
     (>= control 0.953 / 0.992 — no regressions)
Control numbers = the current engine pack on the same traces.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_sim as cs

HOME = os.path.expanduser("~")
G = f"{HOME}/drafter/phase0/traces/r8_prose"
CANARY = f"{HOME}/drafter/anchor_scale_mirror"
CONTROL_PACK = f"pack:{HOME}/drafter/ref_pack"
TRACES_B = {
    "gsm8k": f"{HOME}/drafter/phase0/traces/gsm8k",
    "code8k": f"{HOME}/drafter/phase0/traces/code",
    "prose16k": f"{HOME}/drafter/phase0/traces/prose",
}


def score_r8(spec):
    W = cs.Weights.from_spec(spec)
    r = cs.run_class_a(G, W, G, max_cycles=None, chain_len=4, cond="engine", verbose=False)
    return r


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
    ms_all, eng_all = [], []
    for d in dirs:
        try:
            r = cs.run_class_a(d, W, G, max_cycles=max_per, chain_len=4, cond="engine", verbose=False)
        except Exception as e:
            print(f"[canary] {os.path.basename(d)} SKIP ({e})")
            continue
        per.append(dict(sess=os.path.basename(d), sim=r["sim"], engine=r["engine"]))
        ms_all.extend(r["_ms"] if "_ms" in r else [])
        print(f"[canary] {os.path.basename(d)}: Em_k2 {r['sim']['Em_k2']:.3f} "
              f"k4 {r['sim']['Em_k4']:.3f} a1 {r['sim']['a_cond'][0]:.3f} "
              f"m_dist {r['sim']['m_dist']}", flush=True)
    if not per:
        return None
    k2 = float(np.mean([p["sim"]["Em_k2"] for p in per]))
    k4 = float(np.mean([p["sim"]["Em_k4"] for p in per]))
    a1 = float(np.mean([p["sim"]["a_cond"][0] for p in per]))
    md = [np.array(p["sim"]["m_dist"], dtype=float) for p in per]
    md = np.mean([m / max(m.sum(), 1) for m in md], axis=0).tolist()
    return dict(per=per, Em_k2=k2, Em_k4=k4, a1=a1, m_dist_mean=md)


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
    packs = sys.argv[1:] or [CONTROL_PACK]
    res = {}
    outf = f"{HOME}/drafter/phase0/score_v2.json"
    if os.path.exists(outf):
        res = json.load(open(outf))
    for spec in packs:
        tag = spec.rstrip("/").split("/")[-1]
        print(f"=== {tag} ({spec}) ===", flush=True)
        r8 = score_r8(spec)
        canary = score_canary(spec)
        cross = score_cross(spec)
        res[tag] = dict(spec=spec, r8=r8, canary=canary, cross=cross)
        json.dump(res, open(outf, "w"), indent=1, default=float)
        print(f"[v2score] {tag}: r8 Em_k2 {r8['sim']['Em_k2']:.4f} k4 {r8['sim']['Em_k4']:.4f} "
              f"a_cond {r8['sim']['a_cond']} | canary k2 {canary['Em_k2']:.4f} a1 {canary['a1']:.4f}",
              flush=True)
    print(json.dumps({k: dict(r8=v["r8"]["sim"]["Em_k2"], canary=v["canary"]["Em_k2"] if v["canary"] else None,
                              gsm=v["cross"]["gsm8k"]["Em_k2"], prose=v["cross"]["prose16k"]["Em_k2"],
                              code=v["cross"]["code8k"]["Em_k2"]) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
