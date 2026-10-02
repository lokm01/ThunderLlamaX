#!/usr/bin/env python3
"""MM SESSION B -- PFM in-graph failure forensics. ONE heavy process.
1. eager stock qkv (gv8k2048p) vs eager gvs32k2048 (sanity, known exact).
2. gvs32k2048 through a SINGLE-KERNEL MGUnc graph -> exact?
3. kernargs dump: the bytes the graph's fill_kernargs wrote for gvs32k2048
   (3 buf addrs + the two vals) vs the eager call's.
Output: engine0/mm/mm_dbg2.json"""
import os, sys, json
BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0"); sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn")
import numpy as np

OUT = os.path.join(BASE, "engine0", "mm", "mm_dbg2.json")

def fsync_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())

def main():
    from MM_P7_lib import Rig7
    from mm_a_graph import MGUnc
    rig = Rig7(ctx_alloc=int(os.getenv("MM_CTXS", "98304")), load_p6=False)
    res = {}
    rng = np.random.default_rng(99)
    rows = 8192
    PF = 256
    w = rig.W[0]["qkv"]
    x = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.3)
    y1 = rig.alloc(PF * rows * 4); y2 = rig.alloc(PF * rows * 4); y3 = rig.alloc(PF * rows * 4)

    # 1. eager stock vs eager gvs
    rig.K["gv8k2048p"](w, x, y1, global_size=(rows // 32, PF, 1), local_size=(1024, 1, 1),
                       vals=(rows,), wait=True)
    rig.K["gvs32k2048"](w, x, y2, global_size=(rows // 8, 1, 1), local_size=(256, 1, 1),
                        vals=(rows, PF), wait=True)
    a = rig.dn(y1, (PF * rows,), np.uint32); b = rig.dn(y2, (PF * rows,), np.uint32)
    res["eager_bit_exact"] = bool((a == b).all())
    print("[dbg2] eager stock vs gvs:", res["eager_bit_exact"], flush=True)

    # 2. single-kernel graph
    gr = MGUnc(rig, [(rig.K["gvs32k2048"], (w, x, y3), (rows // 8, 1, 1), (rows, PF))], "dbg2g")
    v = rig.dev.next_timeline(); gr.submit(v - 1, v); rig.nv_wait_timeline(rig.dev, v, what="d2", timeout_s=60.0)
    c = rig.dn(y3, (PF * rows,), np.uint32)
    res["graph_bit_exact"] = bool((a == c).all())
    res["graph_mismatch"] = int((a != c).sum())
    print(f"[dbg2] graph gvs vs eager stock: {res['graph_bit_exact']} mismatch={res['graph_mismatch']}", flush=True)
    fsync_json(OUT, res)

    # 3. kernargs dump: what did the graph write?
    ka = gr.ka if hasattr(gr, "ka") else None
    # MGUnc: the seq's single entry used ka[0:per]; rebind via a fresh fill
    p = rig.K["gvs32k2048"]
    st = p.fill_kernargs((w, x, y3), (rows, PF), kernargs=ka.offset(offset=0, size=p.kernargs_alloc_size) if ka else None)
    n = p.kernargs_alloc_size
    raw = rig.dn(ka if ka is None else ka.offset(offset=0, size=n), (n // 4,), np.uint32) if ka is not None else None
    if raw is not None:
        res["ka_dump_words"] = [int(t) for t in raw[: (n // 4)]][: 24]
        print("[dbg2] ka words:", res["ka_dump_words"][:16], flush=True)
        print("[dbg2] kernargs_alloc_size:", n, "cbuf_0 len:", len(p.cbuf_0) if p.cbuf_0 else 0, flush=True)
    fsync_json(OUT, res)
    print("[dbg2] done", flush=True)

if __name__ == "__main__":
    main()
