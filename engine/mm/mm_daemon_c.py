#!/usr/bin/env python3
"""MM SESSION C -- the DAEMON battery (MM_PFT=1 live): bank60 spec==t1 x2,
pcache CACHE_HIT under the new fp, decode-class tok/s spot (quote/prose),
TTFT spot. Uses the FIXED line-draining + de-doupling harness (the g3c
lesson). Output: /tmp/mm_daemon_c.json"""
import json, time, sys
import numpy as np

sys.path.insert(0, "~/tinygrad-metal/engine0/mm")
from mm_g3c2 import Eng

DOC = "~/mm_p5_doc100k_ids.npy"


def dedup(toks):
    n = len(toks) // 2
    assert toks[:n] == toks[n:], "stream not doubled as expected"
    return toks[:n]


def gen_timed(e, force, cid, cyc):
    t0 = time.perf_counter()
    d = e.gen(force, cid, cyc)
    return dedup(d), time.perf_counter() - t0


def main():
    doc = np.load(DOC).astype(np.int32)
    e = Eng()
    out = {}
    # ---- bank60: spec == t1 through the new PF path, x2 (order swapped) ----
    for rnd, order in enumerate((("t1", "spec"), ("spec", "t1"))):
        a, b = order
        e.prefill(doc[:4096], cid=f"bk{rnd}a")
        ta, sa = gen_timed(e, a, f"bk{rnd}a", 70)
        e.prefill(doc[:4096], cid=f"bk{rnd}b")
        tb, sb = gen_timed(e, b, f"bk{rnd}b", 400)
        n = min(len(ta), len(tb))
        m = all(ta[i] == tb[i] for i in range(n))
        out[f"bank60_r{rnd}"] = {
            "match": bool(m), "compared": n, "n_t1": len(ta), "n_spec": len(tb),
            "t1_tok_s": round(len(ta) / sa, 1), "spec_tok_s": round(len(tb) / sb, 1)}
        print(f"[dc] bank60 r{rnd} ({a}->{b}): match={m} n={n} "
              f"t1={out[f'bank60_r{rnd}']['t1_tok_s']} spec={out[f'bank60_r{rnd}']['spec_tok_s']} tok/s", flush=True)
    # ---- pcache CACHE_HIT under the new config_fp ----
    r1 = e.prefill(doc[:8192], cid="pc1", mode="AUTO_CACHE")
    t0 = time.perf_counter()
    r2 = e.prefill(doc[:8192], cid="pc2", mode="AUTO_CACHE")
    dt = time.perf_counter() - t0
    out["pcache"] = {"first_mode": r1.get("mode"), "second_mode": r2.get("mode"),
                     "cached_tokens": r2.get("cached_tokens"), "second_s": round(dt, 2)}
    print(f"[dc] pcache: first={r1.get('mode')} second={r2.get('mode')} "
          f"cached={r2.get('cached_tokens')} in {dt:.2f}s", flush=True)
    # ---- decode classes: quote (doc continuation = lookup-heavy) / prose ----
    for tag, N in (("quote_4k", 4096), ("prose_2k", 2048)):
        e.prefill(doc[:N], cid=f"cl_{tag}")
        toks, s = gen_timed(e, "spec", f"cl_{tag}", 200)
        out[f"class_{tag}"] = {"tok_s": round(len(toks) / s, 1), "ntok": len(toks)}
        print(f"[dc] class {tag}: {len(toks)} toks in {s:.1f}s = {len(toks)/s:.1f} tok/s", flush=True)
    # ---- TTFT: GSM8K-class fresh prefill (~1.2k prompt) ----
    for i in range(2):
        t0 = time.perf_counter()
        e.prefill(doc[20000 + i:20000 + i + 1216], cid=f"tt{i}")
        out[f"ttft_{1216}tok_fresh_{i}"] = round(time.perf_counter() - t0, 2)
        print(f"[dc] ttft fresh 1216 #{i}: {out[f'ttft_{1216}tok_fresh_{i}']}s", flush=True)
    json.dump(out, open("/tmp/mm_daemon_c.json", "w"), indent=1)
    print("[dc] done", flush=True)


if __name__ == "__main__":
    main()
