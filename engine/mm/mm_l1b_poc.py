#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION C POCs: L1b trunk mma M-GEMM (the
65% residual item) + the routed-dn act-restaging fold (bit-exact route).

ARM 1 (L1b): pgmq8m32 (M=32 mma M-GEMM, dense pf_gemm3m design on Q8_0)
  vs the stock gv8k4096r at the exact in-plan shape (rows=2048, k=4096,
  P=256; grids (64,256) vs (256,)). REAL weights (GDN out + attn o).
  Numerics-class move -> metrics: relerr vs fp64 ref (stock + mma), the
  F-metric (mma vs stock), det x2. GATE >= 1.5x (the stock is at the
  in-graph ~212 GB/s latency floor; the mma tile amortization is the point).
ARM 2 (dn fold): gxm_dnf (act staged ONCE per item into 32KB smem) vs
  gxm_dn (16KB restaged 128x/item/CTA) on REAL dn slabs + REAL mmsort8
  tables from synthetic routing (uniform + skewed). BIT-EXACT gate + det x2.
  GATE >= 1.3x.

ONE heavy process. Output: engine0/mm/mm_l1b_poc_c.json"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
# L9 carveout: bake at program load (pgmq8 -> 64KB=2 CTAs/SM; gxm_dnf -> 100KB)
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn,pgmq8")

PF = 256
REPS = 12
OUT = os.path.join(BASE, "engine0", "mm", "mm_l1b_poc_c.json")

from MM_P56_lib import LSZ
from mm_a_graph import MGUnc
from mm_l1_poc import load_prog, graph_time, run_checked, fsync_json


def relerr(a, b):
    import numpy as np
    d = np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel())
    n = np.linalg.norm(b.astype(np.float64).ravel())
    return float(d / max(n, 1e-30))


def q8_dequant_f64(wbytes):
    import numpy as np
    b = np.frombuffer(wbytes, dtype=np.uint8).reshape(2048, 128, 34)
    d = b[..., :2].copy().view(np.float16).astype(np.float64)[..., 0]
    q = b[..., 2:].copy().view(np.int8).astype(np.float64)
    return (d[..., None] * q).reshape(2048, 4096)


def run_l1b(rig, res):
    import numpy as np
    rng = np.random.default_rng(4242)
    K = rig.K
    from MM_P34_ports import GDN_LAYERS, ATTN_LAYERS
    mma = load_prog(rig, "../..//MM_C_pgmq8m32.cubin", "pgmq8m32", 256)
    x4 = rig.up(rng.standard_normal((PF, 4096)).astype(np.float32) * 0.3)
    hr = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.5)
    x4_np = rig.dn(x4, (PF, 4096), np.float32)
    hr_np = rig.dn(hr, (PF, 2048), np.float32)
    a1 = res.setdefault("l1b", {})
    for tag, L, wkey in (("out", GDN_LAYERS[0], "out"), ("o", ATTN_LAYERS[0], "o")):
        if tag in a1:
            continue
        w = rig.W[L][wkey]
        w_np = rig.dn(w, (2048 * 4352,), np.uint8)
        ref = (q8_dequant_f64(w_np.tobytes()) @ x4_np.T.astype(np.float64)
               + hr_np.T.astype(np.float64)).T  # [PF][2048] fp64
        y1 = rig.alloc(PF * 2048 * 4)
        y2a = rig.alloc(PF * 2048 * 4)
        y2b = rig.alloc(PF * 2048 * 4)
        bufs_s = (w, x4, hr, y1); g_s = (64, PF, 1)
        bufs_m = (w, x4, hr, y2a); g_m = ((PF // 32) * 32, 1, 1)
        run_checked(rig, K["gv8k4096r"], bufs_s, g_s, (2048,), y1, y1.size)
        run_checked(rig, mma, bufs_m, g_m, (PF,), y2a, y2a.size)
        run_checked(rig, mma, (w, x4, hr, y2b), g_m, (PF,), y2b, y2b.size)
        s1 = rig.dn(y1, (PF, 2048), np.float32)
        s2a = rig.dn(y2a, (PF, 2048), np.float32)
        s2b = rig.dn(y2b, (PF, 2048), np.float32)
        det = bool((s2a.view(np.uint32) == s2b.view(np.uint32)).all())
        be = bool((s1.view(np.uint32) == s2a.view(np.uint32)).all())
        row = {
            "det_x2": det, "bit_exact_vs_stock": be,
            "relerr_stock_vs_fp64": relerr(s1, ref),
            "relerr_mma_vs_fp64": relerr(s2a, ref),
            "F_mma_vs_stock": relerr(s2a, s1),
            "maxabs_mma_vs_stock": float(np.abs(s2a - s1).max()),
        }
        t_s = graph_time(rig, K["gv8k4096r"], bufs_s, g_s, (2048,))
        t_m = graph_time(rig, mma, bufs_m, g_m, (PF,))
        row.update({"stock_ms": round(t_s, 4), "mma_ms": round(t_m, 4),
                    "speedup": round(t_s / t_m, 2),
                    "stock_gbs_eff": round(2048 * 4352 * PF / t_s / 1e6, 1),
                    "mma_w_gbs": round(2048 * 4352 * (PF / 32) / t_m / 1e6, 1)})
        a1[tag] = row
        print(f"[l1b] {tag}: {row}", flush=True)
        fsync_json(OUT, res)
    gate = min(v["speedup"] for v in a1.values())
    res["l1b_gate"] = {"min_speedup": gate, "verdict": "GO" if gate >= 1.5 else "KILL"}
    print(f"[l1b] GATE {gate:.2f}x -> {res['l1b_gate']['verdict']}", flush=True)
    fsync_json(OUT, res)


def run_dnf(rig, res):
    import numpy as np
    rng = np.random.default_rng(9192)
    K = rig.K
    dnf = load_prog(rig, "../..//MM_C_gxm_dnf.cubin", "gxm_dnf", 256)
    # a routed layer on the IQ4_XS down lane
    L = 0
    assert rig.man["routed"][L]["types"]["down"] == "IQ4_XS"
    NE = 256  # rig P56 PTB_DN[L] = 256 expert pointers (range(256) at build)
    act = rng.standard_normal((2048, 512)).astype(np.float32) * 0.4
    actb = rig.up(act)
    pa = rig.alloc(2048 * 2048 * 2); pb = rig.alloc(2048 * 2048 * 2)
    a2 = res.setdefault("dnf", {"n_experts": NE})
    for case in ("uniform", "skewed"):
        if case in a2:
            continue
        if case == "uniform":
            eids = rng.integers(0, NE, 2048)
        else:
            u = rng.random(2048)
            eids = np.minimum((NE * u ** 4).astype(np.int64), NE - 1)
        eidsb = rig.up(eids.astype(np.uint16))
        run_checked(rig, K["mmsort8"], (eidsb, rig.EOFFB, rig.PLISTB, rig.ITEMSB, rig.NITB),
                    (1, 1, 1), (2048,), rig.NITB, 4)
        nit = int(rig.dn(rig.NITB, (1,), np.uint32)[0])
        bufs_a = (rig.PTB_DN[L], rig.ITEMSB, rig.NITB, rig.EOFFB, rig.PLISTB, actb, rig.iq4nl, pa)
        bufs_b = (rig.PTB_DN[L], rig.ITEMSB, rig.NITB, rig.EOFFB, rig.PLISTB, actb, rig.iq4nl, pb)
        for _ in range(2):
            run_checked(rig, K["gxm_dn"], bufs_a, (1024, 1, 1), (), pa, pa.size)
            run_checked(rig, dnf, bufs_b, (1024, 1, 1), (), pb, pb.size)
        ha = rig.dn(pa, (2048 * 2048,), np.uint16)
        hb = rig.dn(pb, (2048 * 2048,), np.uint16)
        be = bool((ha == hb).all())
        row = {"nit": nit, "bit_exact": be}
        if be:
            t_a = graph_time(rig, K["gxm_dn"], bufs_a, (1024, 1, 1), ())
            t_b = graph_time(rig, dnf, bufs_b, (1024, 1, 1), ())
            row.update({"gxm_dn_ms": round(t_a, 4), "gxm_dnf_ms": round(t_b, 4),
                        "speedup": round(t_a / t_b, 2)})
        a2[case] = row
        print(f"[dnf] {case}: {row}", flush=True)
        fsync_json(OUT, res)
    sp = [v.get("speedup") for v in a2.values() if isinstance(v, dict) and "speedup" in v]
    if sp:
        res["dnf_gate"] = {"min_speedup": min(sp), "verdict": "GO" if min(sp) >= 1.3 else "KILL"}
        print(f"[dnf] GATE {min(sp):.2f}x -> {res['dnf_gate']['verdict']}", flush=True)
    fsync_json(OUT, res)


if __name__ == "__main__":
    from MM_P7_lib import Rig7
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    print("[pocC] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=int(os.getenv("MM_CTXS", "98304")), load_p6=False)
    run_l1b(rig, res)
    run_dnf(rig, res)
    print(f"[pocC] done -> {OUT}", flush=True)
