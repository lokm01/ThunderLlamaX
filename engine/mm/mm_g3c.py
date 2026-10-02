#!/usr/bin/env python3
"""MM SESSION C -- G3-after-G2 adjudication, round 2: order-swap + ctx
discrimination on the live daemon. The Session-B finding: FRESH t1 vs
FRESH spec diverges at N=4096 (restore paths exact). Questions:
  R1 repro: does pA(t1)-then-pB(spec) still diverge on this daemon?
  R2 order: spec-FIRST-then-t1 -- does the SECOND run inherit the first?
  R3 ctx:   N=2048 fresh t1 vs spec (the restore-exact class).
  R4 locus: first-mismatch position + stream shapes.
Output: /tmp/mm_g3c.json"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_g3probe import Eng, DOC
import numpy as np

def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return None if len(a) == len(b) else n

def main():
    doc = np.load(DOC).astype(np.int32)
    e = Eng()
    out = {}
    # R1: repro (t1 first, then spec -- the Session-B order)
    e.prefill(doc[:4096], cid="c1a"); t1a = e.gen("t1", "c1a")
    e.prefill(doc[:4096], cid="c1b"); spa = e.gen("spec", "c1b")
    fd = first_diff(t1a, spa)
    out["r1_t1_then_spec"] = {"match": fd is None, "first_diff": fd,
                               "n_t1": len(t1a), "n_spec": len(spa),
                               "t1": t1a[:12], "spec": spa[:12]}
    print("[g3c] R1:", out["r1_t1_then_spec"], flush=True)
    # R2: order swap (spec first, then t1)
    e.prefill(doc[:4096], cid="c2a"); spb = e.gen("spec", "c2a")
    e.prefill(doc[:4096], cid="c2b"); t1b = e.gen("t1", "c2b")
    fd2 = first_diff(t1b, spb)
    out["r2_spec_then_t1"] = {"match": fd2 is None, "first_diff": fd2,
                               "spec_first": spb[:12], "t1_second": t1b[:12],
                               "spec_repro_of_spa": spb == spa[:len(spb)],
                               "t1_repro_of_t1a": t1b == t1a[:len(t1b)]}
    print("[g3c] R2:", out["r2_spec_then_t1"], flush=True)
    # R3: N=2048 fresh class
    e.prefill(doc[:2048], cid="c3a"); t1c = e.gen("t1", "c3a")
    e.prefill(doc[:2048], cid="c3b"); spc = e.gen("spec", "c3b")
    fd3 = first_diff(t1c, spc)
    out["r3_2048"] = {"match": fd3 is None, "first_diff": fd3}
    print("[g3c] R3:", out["r3_2048"], flush=True)
    # R4: 8192 class (two more chunk sizes -- does the divergence grow?)
    e.prefill(doc[:8192], cid="c4a"); t1d = e.gen("t1", "c4a")
    e.prefill(doc[:8192], cid="c4b"); spd = e.gen("spec", "c4b")
    fd4 = first_diff(t1d, spd)
    out["r4_8192"] = {"match": fd4 is None, "first_diff": fd4}
    print("[g3c] R4:", out["r4_8192"], flush=True)
    json.dump(out, open("/tmp/mm_g3c.json", "w"), indent=1)
    print("[g3c] done", flush=True)

if __name__ == "__main__":
    main()
