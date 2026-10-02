#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION A / L1 POCs: raster discriminator +
the two trunk M-batching variants (bench both, pick the winner on numbers).

1. RASTER DISCRIMINATOR (the L1-swap pre-gate): stock gv8k2048p vs the
   raster-swapped gv8k2048ps at the exact in-plan qkv shape (rows=8192,
   P=256, grid (256,256), the REAL 17.8MB Q8_0 weight). x-fastest CTA walk
   + L2 row-block sharing -> swapped >= 2x faster. Flat -> the fork's
   cta_raster QMD fields (ops_nv.py) must be read before proceeding.
2. L1-swap: gv8k2048ps/gv8k4096rs across ALL trunk classes (qkv 8192 /
   z 4096 / k 512 / v 512 / out|o 2048x4096), bit-exact vs stock on all
   256 seats (real weight bytes, distinct output buffers, sentinel
   pre-fill = the silent-no-launch detector), det x2, graph-mode min-of-12.
3. L1-seat-loop: gvsl_SW_BC sweep (seat-width 8/16/32/64), same gates.

KILL < 3x vs the per-seat family (expect 8-20x). Timing = one-kernel MG
graph replays (the eager path pays a sync per call; graphs are the in-plan
execution class). Output: engine0/mm/mm_l1_poc.json
"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")

PF = 256
REPS = 12
OUT = os.path.join(BASE, "engine0", "mm", "mm_l1_poc.json")
CUB = os.path.join(BASE, "engine0", "mm")

from MM_P56_lib import LSZ, MG   # LSZ: program name -> local size
from mm_a_graph import MGUnc     # THE UNCACHED-KA LAW (late-built graphs)

_GSEQ = [0]

def load_prog(rig, cub_stem, sym, lsz, scalar=True):
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    lib = open(os.path.join(CUB, cub_stem), "rb").read()
    sig = (rig.INT_SIG,) if scalar else tuple()
    prg = NVProgram(rig.dev, TinyELF(lib=lib, name=sym, target=rig.dev.renderer.target,
                                     signature=sig))
    LSZ[sym] = (lsz, 1, 1)
    rig.K[sym] = prg          # mkgraph/build_seq lookups find it
    return prg

def graph_time(rig, prg, bufs, grid, vals, reps=REPS):
    _GSEQ[0] += 1
    m = MGUnc(rig, [(prg, tuple(bufs), grid, tuple(vals))], f"l1_{prg.name}{_GSEQ[0]}")
    nv = rig.nv_wait_timeline; dev = rig.dev
    ts = []
    for i in range(reps + 1):
        t0 = time.perf_counter()
        v = dev.next_timeline(); m.submit(v - 1, v)
        nv(dev, v, what="l1", timeout_s=120.0)
        if i: ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

def run_checked(rig, prg, bufs, grid, vals, ybuf, ybytes, wait=True):
    """Eager run + sentinel check: every y byte must be overwritten."""
    import numpy as np
    rig.dev.allocator._copyin(ybuf, memoryview(np.full(ybytes // 4, 0x7fbfffbf, dtype=np.uint32).tobytes()))
    prg(*bufs, global_size=grid, local_size=LSZ[prg.name], vals=tuple(vals), wait=wait)
    y = rig.dn(ybuf, (ybytes // 4,), np.uint32)
    assert not (y == 0x7fbfffbf).any(), f"{prg.name}: sentinel survived (silent no-launch?)"

def bit_exact(rig, ya, yb):
    import numpy as np
    a = rig.dn(ya, (ya.size // 4,), np.uint32)
    b = rig.dn(yb, (yb.size // 4,), np.uint32)
    return bool((a == b).all())

def fsync_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    dfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try: os.fsync(dfd)
    finally: os.close(dfd)

def run_l1(rig, res):
    import numpy as np
    dev = rig.dev
    rng = np.random.default_rng(4242)
    K = rig.K

    # real trunk weights: layer 0 is GDN (qkv/z/out); find an attn layer for q/k/v/o
    from MM_P34_ports import GDN_LAYERS, ATTN_LAYERS
    aL = ATTN_LAYERS[0]
    swp = load_prog(rig, "MM_A_gv8k2048ps.cubin", "gv8k2048ps", 1024)
    swr = load_prog(rig, "MM_A_gv8k4096rs.cubin", "gv8k4096rs", 1024)

    x2 = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.3)
    x4 = rig.up(rng.standard_normal((PF, 4096)).astype(np.float32) * 0.3)
    hr = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.5)

    # ---- classes: (tag, kernel_kind, weight, rows, k) ----
    classes = [
        ("qkv", "p", rig.W[0]["qkv"], 8192),
        ("z",   "p", rig.W[0]["z"],   4096),
        ("k",   "p", rig.W[aL]["k"],   512),
        ("v",   "p", rig.W[aL]["v"],   512),
        ("q",   "p", rig.W[aL]["q"],  8192),
        ("out", "r", rig.W[0]["out"], 2048),   # GDN ssm_out (k=4096, +residual)
        ("o",   "r", rig.W[aL]["o"],  2048),
    ]
    l1 = res.setdefault("l1_swap", {})
    for tag, kind, w, rows in classes:
        if tag in l1:
            continue
        y1 = rig.alloc(PF * rows * 4); y2 = rig.alloc(PF * rows * 4)
        if kind == "p":
            bufs_s = (w, x2, y1); bufs_w = (w, x2, y2)
            g_s = (rows // 32, PF, 1); g_w = (PF, rows // 32, 1)
            ks, kw = K["gv8k2048p"], swp
        else:
            bufs_s = (w, x4, hr, y1); bufs_w = (w, x4, hr, y2)
            g_s = (rows // 32, PF, 1); g_w = (PF, rows // 32, 1)
            ks, kw = K["gv8k4096r"], swr
        run_checked(rig, ks, bufs_s, g_s, (rows,), y1, y1.size)
        run_checked(rig, kw, bufs_w, g_w, (rows,), y2, y2.size)
        be1 = bit_exact(rig, y1, y2)
        run_checked(rig, ks, bufs_s, g_s, (rows,), y1, y1.size)
        run_checked(rig, kw, bufs_w, g_w, (rows,), y2, y2.size)
        be2 = bit_exact(rig, y1, y2)
        t_s = graph_time(rig, ks, bufs_s, g_s, (rows,))
        t_w = graph_time(rig, kw, bufs_w, g_w, (rows,))
        wbytes = rows * (4352 if kind == "r" else 2176)
        row = {"rows": rows, "bit_exact": be1 and be2, "stock_ms": round(t_s, 4),
               "swap_ms": round(t_w, 4), "speedup": round(t_s / t_w, 2),
               "w_mb": round(wbytes / 1e6, 1),
               "stock_gbs": round(wbytes * PF / t_s / 1e6, 1),
               "swap_gbs": round(wbytes * PF / t_w / 1e6, 1)}
        l1[tag] = row
        print(f"[l1] {tag:4s} rows={rows:5d} stock {t_s:7.3f} ms swap {t_w:7.3f} ms "
              f"x{row['speedup']:5.2f} bit_exact={row['bit_exact']} "
              f"({row['stock_gbs']:.0f}->{row['swap_gbs']:.0f} GB/s eff)", flush=True)
        fsync_json(OUT, res)

    q = l1.get("qkv", {})
    res["raster_verdict"] = {
        "ratio_qkv": q.get("speedup"),
        "verdict": ("GO: x-fastest CTA walk + L2 row-block sharing confirmed"
                    if (q.get("speedup") or 0) >= 2.0 else
                    "FLAT: read cta_raster QMD fields (ops_nv.py) before proceeding"),
    }
    print(f"[l1] RASTER VERDICT: {res['raster_verdict']['verdict']}", flush=True)
    fsync_json(OUT, res)

def run_seatloop(rig, res):
    import numpy as np
    rng = np.random.default_rng(4242)
    rows = 8192
    w = rig.W[0]["qkv"]
    x2 = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.3)
    y0 = rig.alloc(PF * rows * 4)

    # stock reference ONCE (already validated in run_l1 for qkv, but this
    # module can run standalone -> recompute)
    run_checked(rig, rig.K["gv8k2048p"], (w, x2, y0), (rows // 32, PF, 1), (rows,), y0, y0.size)
    t_s = graph_time(rig, rig.K["gv8k2048p"], (w, x2, y0), (rows // 32, PF, 1), (rows,))

    sl = res.setdefault("l1_seatloop", {"stock_ms": round(t_s, 4)})
    for sw, bc, tpb in ((8, 8, 256), (16, 8, 256), (32, 8, 256), (64, 4, 256), (32, 8, 512)):
        sym = f"gvsl_{sw}_{bc}_t{tpb}"
        if sym in sl:
            continue
        rpc = tpb // 32
        yv = rig.alloc(PF * rows * 4)
        prg = load_prog(rig, f"MM_A_gvsl_{sw}_{bc}_t{tpb}.cubin", sym, tpb)
        try:
            run_checked(rig, prg, (w, x2, yv), (rows // rpc, 1, 1), (rows,), yv, yv.size)
            be1 = bit_exact(rig, y0, yv)
            run_checked(rig, prg, (w, x2, yv), (rows // rpc, 1, 1), (rows,), yv, yv.size)
            be2 = bit_exact(rig, y0, yv)
            t = graph_time(rig, prg, (w, x2, yv), (rows // rpc, 1, 1), (rows,))
            row = {"bit_exact": be1 and be2, "ms": round(t, 4),
                   "speedup_vs_stock": round(t_s / t, 2),
                   "smem_kb": sw * bc * 128 // 1024, "tpb": tpb}
        except Exception as e:
            row = {"error": str(e)[:200]}
        sl[sym] = row
        print(f"[l1] gvsl SW={sw} BC={bc} t{tpb}: {row}", flush=True)
        fsync_json(OUT, res)

    # the winner: fastest bit-exact variant across both families
    cand = [(v["ms"], k) for k, v in sl.items()
            if isinstance(v, dict) and v.get("bit_exact") and "ms" in v]
    sw_best = min(cand) if cand else None
    swap_qkv = res.get("l1_swap", {}).get("qkv", {})
    if sw_best and swap_qkv:
        res["l1_winner"] = {
            "seatloop_best": {"variant": sw_best[1], "ms": sw_best[0]},
            "swap_qkv_ms": swap_qkv.get("swap_ms"),
            "winner": "seatloop" if sw_best[0] < swap_qkv.get("swap_ms", 1e9) else "swap",
            "note": "qkv-class comparison; Session B ports the winner across all trunk classes",
        }
        print(f"[l1] WINNER: {res['l1_winner']}", flush=True)
    fsync_json(OUT, res)

if __name__ == "__main__":
    from MM_P7_lib import Rig7
    res = {}
    if os.path.exists(OUT):
        try: res = json.load(open(OUT))
        except Exception: res = {}
    print("[l1] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=int(os.getenv("MM_CTXS", "98304")), load_p6=False)
    run_l1(rig, res)
    run_seatloop(rig, res)
    print(f"[l1] done -> {OUT}", flush=True)
