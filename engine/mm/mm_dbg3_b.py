#!/usr/bin/env python3
"""MM SESSION B -- PFM per-class replacement bisect. ONE heavy process.
Build the STOCK PF seq, then produce variants where exactly ONE gv8k2048p
call site (qkv / z / q / k / v) is swapped to gvs32k2048 (same buffers,
vals (rows, P)); graph + run + compare hA vs the stock reference.
Output: engine0/mm/mm_dbg3.json"""
import os, sys, json
BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0"); sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn")
import numpy as np

OUT = os.path.join(BASE, "engine0", "mm", "mm_dbg3.json")
CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256

def fsync_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())

def load_prose_ids(n=PF):
    d = json.load(open(os.path.join(BASE, "eval", "data", "ppl_prose_ids.json")))
    out = []
    while len(out) < n:
        out += [int(t) for t in d]
    return out[:n]

# call-site tags by (weight key, rows)
SITES = {"qkv": ("qkv", 8192), "z": ("z", 4096), "q": ("q", 8192),
         "k": ("k", 512), "v": ("v", 512)}

def main():
    from MM_P7_lib import Rig7, build_seq7
    from mm_a_graph import mkgraph_unc as mkgraph
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    res = {}
    ids = np.ascontiguousarray(np.array(load_prose_ids(), dtype=np.int32))
    os.environ["MM_PFG"] = "0"; os.environ["MM_PFM"] = "0"
    stock = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                       spk=f"s{CTXK}", S=8, pf=True)

    def run(seq, tag):
        gr = mkgraph(rig, seq, tag)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data)
        rig.pos_view[0] = 0
        gr.step()
        return rig.dn(rig.PFB["hA"], (PF * 2048,), np.uint32).copy()

    ref = run(stock, "d3_ref")
    res["ref_selfcheck"] = None
    ref2 = run(stock, "d3_ref2")
    res["ref_selfcheck"] = bool((ref == ref2).all())
    print("[dbg3] stock self-check:", res["ref_selfcheck"], flush=True)
    fsync_json(OUT, res)

    for site, (wkey, rows) in SITES.items():
        seq = []; hit = 0
        for n, b, g, v in stock:
            if n == "gv8k2048p" and v == (rows,) and len(b) == 3 and \
               any(b[0] is rig.W[L][wkey] for L in range(40) if wkey in rig.W[L]):
                seq.append(("gvs32k2048", (b[0], b[1], b[2]), rows // 8, (rows, PF)))
                hit += 1
            else:
                seq.append((n, b, g, v))
        h = run(seq, f"d3_{site}")
        nz = int((ref != h).sum())
        res[site] = {"swapped_sites": hit, "mismatched": nz}
        print(f"[dbg3] {site}: swapped={hit} mismatched={nz}", flush=True)
        fsync_json(OUT, res)
    print("[dbg3] done", flush=True)

if __name__ == "__main__":
    main()
