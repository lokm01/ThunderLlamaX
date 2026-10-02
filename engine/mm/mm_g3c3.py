#!/usr/bin/env python3
"""MM SESSION C -- G3 round 4: the 3072..4096 bisect + content control.
Rounds so far: divergence is ctx-keyed (4k yes / 1k,2k,3k no within 1200
output tokens), order-independent, both modes deterministic.
  B: feeds 3328 / 3584 / 3840 (find the threshold at 256 granularity)
  C: content control -- a different 4k slice (content-keyed vs ctx-keyed)
Output: /tmp/mm_g3c3.json"""
import json
import numpy as np
import sys

sys.path.insert(0, "~/tinygrad-metal/engine0/mm")
from mm_g3c2 import Eng, first_diff  # noqa: E402


def arm(e, ids, tag, out, cyc=600):
    e.prefill(ids, cid=tag + "a")
    t1 = e.gen("t1", tag + "a", cyc)
    e.prefill(ids, cid=tag + "b")
    sp = e.gen("spec", tag + "b", cyc)
    fd = first_diff(t1, sp)
    out[tag] = {"match": fd is None, "first_diff": fd, "n_t1": len(t1), "n_spec": len(sp)}
    print(f"[g3c3] {tag}: {out[tag]}", flush=True)
    json.dump(out, open("/tmp/mm_g3c3.json", "w"), indent=1)


def main():
    doc = np.load("~/mm_p5_doc100k_ids.npy").astype(np.int32)
    e = Eng()
    out = {}
    for N in (3328, 3584, 3840):
        arm(e, doc[:N], f"b_{N}", out)
    alt = doc[1000:1000 + 4096]
    arm(e, alt, "cc_4k_slice1000", out)
    print("[g3c3] done", flush=True)


if __name__ == "__main__":
    main()
