#!/usr/bin/env python3
"""MM SESSION C -- G3 round 5 (final adjudication). The probe-gen bug:
cycle events AND the done event both append tokens -> streams doubled.
De-doubled state: doc[:N] feeds 1k/2k/3k/3328/3584/3840/4096 all MATCH
within the compared window; the ALT slice (doc[1000:5096]) diverges at
REAL token 0. This arm: the smoking-gun evidence.
  - fresh t1 vs fresh spec on the alt slice, DE-DOUBLED
  - token0 of each + the true doc continuation doc[5096..5101]
  - the second spec token (bnd class: m=0 -> [bnd] only)
Output: /tmp/mm_g3c4.json"""
import json
import numpy as np
import sys

sys.path.insert(0, "~/tinygrad-metal/engine0/mm")
from mm_g3c2 import Eng  # noqa: E402


def dedup(toks, real_n):
    # doubled = cycle-concat + done-repeat; real stream = first half IF the
    # second half repeats the first (verify, else report)
    h1 = toks[:real_n]
    h2 = toks[real_n:2 * real_n]
    return h1, h2 == h1


def main():
    doc = np.load("~/mm_p5_doc100k_ids.npy").astype(np.int32)
    alt = doc[1000:1000 + 4096]
    e = Eng()
    out = {}
    e.prefill(alt, cid="z_a")
    t1d = e.gen("t1", "z_a", 60)
    e.prefill(alt, cid="z_b")
    spd = e.gen("spec", "z_b", 60)
    t1, t1_ok = dedup(t1d, len(t1d) // 2)
    sp, sp_ok = dedup(spd, len(spd) // 2)
    out["doubling_confirmed"] = {"t1": t1_ok, "spec": sp_ok,
                                 "n_t1_doubled": len(t1d), "n_spec_doubled": len(spd)}
    fd = next((i for i in range(min(len(t1), len(sp))) if t1[i] != sp[i]), None)
    out["real_first_diff"] = fd
    out["t1_first12"] = t1[:12]
    out["spec_first12"] = sp[:12]
    out["true_doc_next12"] = [int(x) for x in doc[5096:5108]]
    print("[g3c4]", json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/tmp/mm_g3c4.json", "w"), indent=1)
    print("[g3c4] done", flush=True)


if __name__ == "__main__":
    main()
