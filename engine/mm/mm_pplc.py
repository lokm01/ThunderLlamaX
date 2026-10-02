#!/usr/bin/env python3
"""MM SESSION C -- the ppl_proxy arm: per-seat next-token cross-entropy +
top-1 agreement through the STOCK vs SESSION-C PF paths on the same
256-token prose chunk (the honest quality metric for the Tier-2 move:
7% hA drift with 6.7% routing flips must not move the predictive CE).
Also 4 more chunks at deeper positions (drift is position-independent per
chunk; the KV from earlier chunks is shared) -> one chunk is the unit.
Output: engine0/mm/mm_pplc.json"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn,pgmq8")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256
OUT = os.path.join(BASE, "engine0", "mm", "mm_pplc.json")
S = 8

from mm_l1_poc import fsync_json
from mm_f1b_c import build_pf, eager_top1
from mm_f1b_b import load_prose_ids


def main():
    import numpy as np
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    from MM_P7_lib import Rig7
    print("[pplc] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    targets = ids[1:]  # next token per seat

    out = res.setdefault("ce", {})
    for tag, pfc in (("stock", False), ("new", True)):
        if tag in out:
            continue
        gr, _ = build_pf(rig, pfc)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids[:PF].data)
        rig.pos_view[0] = 0
        gr.step()
        # per-seat eager head: top1 + logprob of the TRUE next token
        ces, tops, nflip = [], [], 0
        t0 = time.perf_counter()
        for seat in range(PF):
            _, lg = eager_top1(rig, seat)
            lg64 = lg.astype(np.float64)
            mx = lg64.max()
            lse = mx + np.log(np.exp(lg64 - mx).sum())
            ces.append(lse - lg64[int(targets[seat])])   # CE of the true token
            tops.append(int(np.argmax(lg64)))
        out[tag] = {"mean_ce": float(np.mean(ces)), "ce_sem": float(np.std(ces) / np.sqrt(len(ces))),
                    "scan_s": round(time.perf_counter() - t0, 1)}
        out[tag + "_tops"] = tops
        print(f"[pplc] {tag}: mean_ce={out[tag]['mean_ce']:.4f} (+/-{out[tag]['ce_sem']:.4f})", flush=True)
        fsync_json(OUT, res)
    if "dce_rel" not in out:
        out["dce_rel"] = (out["new"]["mean_ce"] - out["stock"]["mean_ce"]) / out["stock"]["mean_ce"]
        out["top1_agree"] = int(sum(1 for a, b in zip(out["stock_tops"], out["new_tops"]) if a == b))
        out["top1_agree_excl_last"] = int(sum(1 for a, b in zip(out["stock_tops"][:255], out["new_tops"][:255]) if a == b))
        del out["stock_tops"]; del out["new_tops"]
        print(f"[pplc] dCE_rel={out['dce_rel']:+.4%} top1_agree={out['top1_agree']}/256", flush=True)
        fsync_json(OUT, res)
    print("[pplc] done", flush=True)


if __name__ == "__main__":
    main()
