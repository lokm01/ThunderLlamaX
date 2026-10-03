#!/usr/bin/env python3
"""MM SESSION E -- CE bisect of the combined stack: which knob breaks it.
Base = D-config + MM_PFU; each arm adds ONE of PFD/PFR/PFS; + all-on.
Per arm: mean CE + top1 agreement vs the stock D-config."""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn,pgmq8")

CTXK = 98304
PF = 256
OUT = os.path.join(BASE, "engine0", "mm", "mm_ce_bisect.json")
from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_f1b_c import eager_top1
from mm_a_graph import mkgraph_unc as mkgraph


def build_pf(rig, extra):
    for k in ("MM_PFG", "MM_PFM", "MM_PFT", "MM_PFW", "MM_PFK"):
        os.environ[k] = "1"
    for k in ("MM_PFU", "MM_PFD", "MM_PFR", "MM_PFS"):
        os.environ[k] = "1" if k in extra else "0"
    from MM_P7_lib import build_seq7
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=f"s{CTXK}", S=8, pf=True)
    return mkgraph(rig, seq, f"cx_{'_'.join(sorted(extra)) or 'base'}")


def main():
    import numpy as np
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    from MM_P7_lib import Rig7
    print("[cx] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    targets = ids[1:]

    ARMS = [("base", []), ("pfd", ["MM_PFD"]), ("pfr", ["MM_PFR"]),
            ("pfs", ["MM_PFS"]), ("all", ["MM_PFU", "MM_PFD", "MM_PFR", "MM_PFS"])]
    for tag, extra in ARMS:
        if tag in res:
            continue
        gr = build_pf(rig, extra)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids[:PF].data)
        rig.pos_view[0] = 0
        gr.step()
        ces, tops = [], []
        for seat in range(PF):
            _, lg = eager_top1(rig, seat)
            lg64 = lg.astype(np.float64)
            mx = lg64.max()
            lse = mx + np.log(np.exp(lg64 - mx).sum())
            ces.append(lse - lg64[int(targets[seat])])
            tops.append(int(np.argmax(lg64)))
        res[tag] = {"mean_ce": float(np.mean(ces)), "sem": float(np.std(ces) / np.sqrt(PF))}
        if "base_tops" not in res and tag == "base":
            res["base_tops"] = tops
        elif "base_tops" in res:
            res[tag]["top1"] = int(sum(1 for a, b in zip(res["base_tops"], tops) if a == b))
        print(f"[cx] {tag}: CE {res[tag]['mean_ce']:.4f} "
              f"top1 {res[tag].get('top1', '-')}/256", flush=True)
        gr.fence()
        fsync_json(OUT, res)
    res.pop("base_tops", None)
    fsync_json(OUT, res)
    print("[cx] done", flush=True)


if __name__ == "__main__":
    main()
