#!/usr/bin/env python3
"""MM SESSION C -- MTP-mode decode classes + the 15-min soak (mixed
prefill/generate/restore traffic). Output: /tmp/mm_soak_c.json"""
import json, time, sys
import numpy as np

sys.path.insert(0, "~/tinygrad-metal/engine0/mm")
from mm_g3c2 import Eng
from mm_daemon_c import dedup, gen_timed

DOC = "~/mm_p5_doc100k_ids.npy"
SOAK_S = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0


def main():
    doc = np.load(DOC).astype(np.int32)
    e = Eng()
    out = {"classes": {}}
    # MTP (default) decode classes
    for tag, N in (("quote_4k", 4096), ("prose_2k", 2048), ("long_16k", 16384)):
        e.prefill(doc[:N], cid=f"mc_{tag}")
        t, s = gen_timed(e, None, f"mc_{tag}", 200)
        out["classes"][tag] = {"tok_s": round(len(t) / s, 1), "ntok": len(t)}
        print(f"[soak] mtp class {tag}: {len(t)} toks {len(t)/s:.1f} tok/s", flush=True)
    json.dump(out, open("/tmp/mm_soak_c.json", "w"), indent=1)
    # the soak: mixed traffic, faults counted
    t0 = time.perf_counter()
    rounds = 0
    try:
        while time.perf_counter() - t0 < SOAK_S:
            i = rounds % 6
            if i in (0, 3):    # fresh 4k + default gen
                e.prefill(doc[5000 + rounds * 7:5000 + rounds * 7 + 4096], cid=f"s{rounds}")
                gen_timed(e, None, f"s{rounds}", 60)
            elif i in (1, 4):  # cache restore + t1
                e.prefill(doc[:4096], cid=f"s{rounds}", mode="AUTO_CACHE")
                gen_timed(e, "t1", f"s{rounds}", 60)
            elif i == 2:       # spec on 2k
                e.prefill(doc[:2048], cid=f"s{rounds}")
                gen_timed(e, "spec", f"s{rounds}", 80)
            else:              # 16k feed + default
                e.prefill(doc[:16384], cid=f"s{rounds}")
                gen_timed(e, None, f"s{rounds}", 60)
            rounds += 1
            if rounds % 5 == 0:
                h = e.rpc("status")
                print(f"[soak] r{rounds} t={time.perf_counter()-t0:.0f}s ok", flush=True)
    except Exception as ex:
        out["soak_fault"] = repr(ex)[:300]
        print(f"[soak] FAULT after {rounds} rounds: {ex!r}", flush=True)
    out["soak"] = {"rounds": rounds, "secs": round(time.perf_counter() - t0, 1),
                   "fault": out.get("soak_fault") is None}
    json.dump(out, open("/tmp/mm_soak_c.json", "w"), indent=1)
    print(f"[soak] done: {out['soak']}", flush=True)


if __name__ == "__main__":
    main()
