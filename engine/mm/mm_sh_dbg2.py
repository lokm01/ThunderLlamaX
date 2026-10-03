#!/usr/bin/env python3
"""MM SESSION E -- PFR first-diverging-layer bisect: run partial seqs
(layers 0..k) stock vs PFR on the same input; find the first layer where
hA diverges beyond the Tier-2 class. Config = the exact CE-bisect pfr arm
(D-config, PFU/PFD/PFS=0, PFR=1)."""
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
OUT = os.path.join(BASE, "engine0", "mm", "mm_sh_layer.json")
from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_a_graph import mkgraph_unc as mkgraph


def build(rig, pfr, upto_layers):
    for k in ("MM_PFG", "MM_PFM", "MM_PFT", "MM_PFW", "MM_PFK"):
        os.environ[k] = "1"
    for k in ("MM_PFU", "MM_PFD", "MM_PFR", "MM_PFS"):
        os.environ[k] = "0"
    if pfr:
        os.environ["MM_PFR"] = "1"
    from MM_P7_lib import build_seq7
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=f"s{CTXK}", S=8, pf=True)
    # truncate at layer boundary: count entries until the (upto)th cmbz2048
    out, seen = [], 0
    for ent in seq:
        out.append(ent)
        if ent[0] == "cmbz2048":
            seen += 1
            if seen >= upto_layers:
                break
    return out


def main():
    import numpy as np
    from MM_P7_lib import Rig7
    res = {}
    print("[sl] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))

    def relerr(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    # CONTROL MATRIX at k=2: s1, s2 (stock twice), p1, p2 (pfr twice)
    BUFS = ["hA", "hnb", "actsh", "shb", "gatesb"]
    caps = {}
    for tag, pfr2 in (("s1", False), ("s2", False), ("p1", True), ("p2", True)):
        g = mkgraph(rig, build(rig, pfr2, 2), f"sl_{tag}_2")
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids[:PF].data)
        rig.pos_view[0] = 0
        g.step(); g.fence()
        caps[tag] = {}
        for b in BUFS:
            buf = rig.ACTSHB if b == "actsh" else rig.PFB[b]
            n = buf.size // 4
            caps[tag][b] = rig.dn(buf, (n,)).copy().astype(np.float64)
        g = None
    res["matrix"] = {}
    for b in BUFS:
        row = {}
        for x, y in (("s2", "s1"), ("p1", "s1"), ("p2", "p1")):
            row[f"{y}_vs_{x}"] = round(relerr(caps[x][b], caps[y][b]), 6)
        res["matrix"][b] = row
        print(f"[sl] {b}: {row}", flush=True)
        fsync_json(OUT, res)
    print("[sl] done", flush=True)


if __name__ == "__main__":
    main()
