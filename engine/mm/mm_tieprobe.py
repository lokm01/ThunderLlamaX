#!/usr/bin/env python3
"""MM SESSION C -- bank60 mismatch adjudication: is the r0 t1-vs-spec
divergence the DETERMINISTIC tie-flip class (am-head argmax vs the verify
kernel's argmax on fp16 near-ties) or a race? x2 same-content repro +
position + frequency across 4 doc offsets. Also MTP(default)==t1 check.
Output: /tmp/mm_tieprobe.json"""
import json, time, sys
import numpy as np

sys.path.insert(0, "~/tinygrad-metal/engine0/mm")
from mm_g3c2 import Eng
from mm_daemon_c import dedup, gen_timed

DOC = "~/mm_p5_doc100k_ids.npy"


def arm(e, ids, tag, out):
    e.prefill(ids, cid=f"{tag}a")
    ta, _ = gen_timed(e, "t1", f"{tag}a", 70)
    e.prefill(ids, cid=f"{tag}b")
    tb, _ = gen_timed(e, "spec", f"{tag}b", 400)
    n = min(len(ta), len(tb))
    fd = next((i for i in range(n) if ta[i] != tb[i]), None)
    # the mtp (default) twin
    e.prefill(ids, cid=f"{tag}c")
    tc, _ = gen_timed(e, None, f"{tag}c", 200)
    fdm = next((i for i in range(min(len(ta), len(tc))) if ta[i] != tc[i]), None)
    out[tag] = {"first_diff_t1_spec": fd, "n_t1": len(ta), "n_spec": len(tb),
                "first_diff_t1_mtp": fdm,
                "ctx": (ta[fd - 2:fd + 3] if fd is not None and fd >= 2 else None),
                "spec_at": (tb[fd - 2:fd + 3] if fd is not None and fd >= 2 else None)}
    print(f"[tie] {tag}: fd(t1,spec)={fd} fd(t1,mtp)={fdm} ctx={out[tag]['ctx']} spec={out[tag]['spec_at']}", flush=True)
    json.dump(out, open("/tmp/mm_tieprobe.json", "w"), indent=1)


def main():
    doc = np.load(DOC).astype(np.int32)
    e = Eng()
    out = {}
    arm(e, doc[:4096], "r0_repro_1", out)          # exact r0 content, x2
    arm(e, doc[:4096], "r0_repro_2", out)
    arm(e, doc[10000:14096], "slice10k", out)
    arm(e, doc[30000:34096], "slice30k", out)
    arm(e, doc[60000:64096], "slice60k", out)
    print("[tie] done", flush=True)


if __name__ == "__main__":
    main()
