#!/usr/bin/env python3
"""MM SESSION D / port-2 POC: the k=2048 trunk M-GEMM mma (pgmq8k2) vs the
stock gvs32k2048 seat-loops, per class (qkv/z/q/k/v).

Per class: isolated in-graph min-of-8 (fresh fenced graphs), stock vs mma
+ relerr (Tier-2 signature, the pgmq8m32 F class ~1.5e-4) + det x2 +
sentinel. Real weights; x = live hnb after one stock chunk (garbage-tolerant
for timing; numerics compared kernel-vs-kernel on the same inputs).
Output: engine0/mm/mm_p2_poc_d.json"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn,pgmq8")
os.environ["MM_PFG"] = "0"; os.environ["MM_PFM"] = "0"
os.environ["MM_PFT"] = "0"; os.environ["MM_PFW"] = "0"

CTXK = 98304
PF = 256
S = 8
OUT = os.path.join(BASE, "engine0", "mm", "mm_p2_poc_d.json")
from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_a_graph import mkgraph_unc as mkgraph

CLASSES = [  # (name, layer, wkey, ROWS, ybuf)
    ("qkv", 0, "qkv", 8192, "qkvb"),
    ("z",   0, "z",   4096, "zb"),
    ("q",   3, "q",   8192, "qgb"),
    ("k",   3, "k",    512, "kqb"),
    ("v",   3, "v",    512, "vqb"),
]


def graph_time(rig, seq, tag, reps=8):
    m = mkgraph(rig, seq, tag, fence_every=48)
    ts = []
    for i in range(reps + 1):
        t0 = time.perf_counter()
        m.step()
        if i: ts.append((time.perf_counter() - t0) * 1e3)
    m.fence()
    return min(ts)


def main():
    import numpy as np
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    from MM_P56_lib import LSZ
    from MM_P7_lib import Rig7, build_seq7
    from MM_P34_ports import GDN_LAYERS, ATTN_LAYERS
    res = {}
    print("[p2] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev

    # load the mma variants
    for rows in (8192, 4096, 512):
        nm = f"pgmq8k2_r{rows}"
        lib = open(f"{BASE}/MM_D_{nm}.cubin", "rb").read()
        prg = NVProgram(dev, TinyELF(lib=lib, name=nm,
                                     target=dev.renderer.target, signature=(rig.INT_SIG,)))
        LSZ[nm] = (256, 1, 1)
        rig.K[nm] = prg
        print(f"[p2] {nm} loaded regs={prg.regs_usage} stack={prg.stack_usage} "
              f"smem={prg.shmem_usage}", flush=True)
    fsync_json(OUT, res)

    # one stock chunk -> live hnb + real weights in cache
    ids = np.ascontiguousarray(np.array(load_prose_ids(512), dtype=np.int32))
    seqpf = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                       spk=f"s{CTXK}", S=S, pf=True)
    grpf = mkgraph(rig, seqpf, "p2_pf")
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids[:PF].data)
    rig.pos_view[0] = 0
    grpf.step()
    grpf.fence()

    hnb = rig.PFB["hnb"]
    out = {}
    for cname, L, wkey, rows, ykey in CLASSES:
        w = rig.W[L][wkey]
        yb = rig.PFB[ykey]
        ybytes = PF * rows * 4
        nm = f"pgmq8k2_r{rows}"
        # stock run (eager) -> reference y
        rig.dev.allocator._copyin(yb, memoryview(np.full(ybytes // 4, 0x7fbfffbf, dtype=np.uint32).tobytes()))
        rig.K["gvs32k2048"](w, hnb, yb, global_size=(rows // 8, 1, 1),
                            local_size=(256, 1, 1), vals=(rows, PF), wait=True)
        yA = rig.dn(yb, (ybytes // 4,), np.uint32).copy()
        # mma run x2 (det)
        ys = []
        for _ in range(2):
            rig.dev.allocator._copyin(yb, memoryview(np.full(ybytes // 4, 0x7fbfffbf, dtype=np.uint32).tobytes()))
            rig.K[nm](w, hnb, yb, global_size=((PF // 32) * (rows // 64), 1, 1),
                      local_size=(256, 1, 1), vals=(PF,), wait=True)
            ys.append(rig.dn(yb, (ybytes // 4,), np.uint32).copy())
        det = bool(np.array_equal(ys[0], ys[1]))
        sents = int((ys[0] == 0x7fbfffbf).sum())
        re_ = float(np.linalg.norm((ys[0].view(np.float32).astype(np.float64) -
                                    yA.view(np.float32).astype(np.float64))) /
                    max(np.linalg.norm(yA.view(np.float32).astype(np.float64)), 1e-30))
        # in-graph timing pair
        seqS = [("gvs32k2048", (w, hnb, yb), rows // 8, (rows, PF))]
        seqM = [(nm, (w, hnb, yb), (PF // 32) * (rows // 64), (PF,))]
        tS = graph_time(rig, seqS, f"p2_s_{cname}")
        tM = graph_time(rig, seqM, f"p2_m_{cname}")
        out[cname] = {"rows": rows, "stock_ms": round(tS, 3), "mma_ms": round(tM, 3),
                      "x": round(tS / tM, 2), "relerr": re_, "det_x2": det,
                      "sentinel": sents}
        print(f"[p2] {cname} r{rows}: stock {tS:.2f}ms mma {tM:.2f}ms x{tS/tM:.2f} "
              f"relerr {re_:.2e} det {det} sentinel {sents}", flush=True)
        res["classes"] = out
        fsync_json(OUT, res)
    print("[p2] done", flush=True)


if __name__ == "__main__":
    main()
