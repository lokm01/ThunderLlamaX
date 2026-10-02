#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION B / G2 (+G1-ports): the grouped-expert
gates + bench, the L1 seat-loop port gates, and the shared-expert M-batch
gates. ONE heavy process (the GPU-EXIT law).

ARMS
  1. sort      mmsort8 det x2 + EXACT vs the numpy rank-count sim (eoff,
               plist, items, nit) on REAL per-layer routing (captured with
               the Session-A cp4k pattern from the production PF graph).
  2. grouped   gxm_up/up4/dn/dn6 per-pair BIT-EXACT vs the pair-walk
               (gx8e256up/up4/dn/dn6) on real routing; sentinel pre-fill.
  3. bench     pair-walk vs grouped at the chunk shape (one-kernel MGUnc
               graph replays, min of 12) per quant class + mmsort8 cost.
               KILL < 2x on the up family (expect ~10x).
  4. gvs       the L1 ports: gvs32k2048 vs gv8k2048p (qkv/z/q/k/v classes),
               gvs32k4096r vs gv8k4096r (out/o), gvsab vs gvf32ab -- all
               bit-exact x2 + bench.
  5. shexp     shgu32+shdn32 vs shexp8 bit-exact + bench.

Output: engine0/mm/mm_g2_l2.json
"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256
TS, RS, GN = 16, 4, 1024
REPS = 12
OUT = os.path.join(BASE, "engine0", "mm", "mm_g2_l2.json")

from MM_P56_lib import LSZ
from mm_l1_poc import load_prog, graph_time, run_checked, bit_exact, fsync_json
from mm_a_graph import MGUnc

B_SIGS = {}   # sym -> (lsz, nvals)


def load_b(rig):
    """Load the Session-B programs with their local sizes + scalar sigs."""
    from mm_l1_poc import CUB  # engine0/mm -- but B cubins live at BASE root
    for stem, sym, lsz, nvals in (
        ("MM_B_mmsort8.cubin", "mmsort8", 1024, 1),
        ("MM_B_gxm_up.cubin",  "gxm_up",  256, 0),
        ("MM_B_gxm_up4.cubin", "gxm_up4", 256, 0),
        ("MM_B_gxm_dn.cubin",  "gxm_dn",  256, 0),
        ("MM_B_gxm_dn6.cubin", "gxm_dn6", 256, 0),
        ("MM_B_gvs32.cubin",   "gvs32k2048", 256, 2),
        ("MM_B_gvs32r.cubin",  "gvs32k4096r", 256, 2),
        ("MM_B_gvsab.cubin",   "gvsab", 128, 1),
        ("MM_B_shgu32.cubin",  "shgu32", 256, 1),
        ("MM_B_shdn32.cubin",  "shdn32", 256, 1),
        ("MM_P34_gconv36_64.cubin", "gconv36_64", 256, 0),
        ("MM_P34_k2s36_64.cubin",   "k2s36_64", 256, 0),
    ):
        path = os.path.join(BASE, stem)
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_nv import NVProgram
        lib = open(path, "rb").read()
        sig = tuple(rig.INT_SIG for _ in range(nvals))
        rig.K[sym] = NVProgram(rig.dev, TinyELF(lib=lib, name=sym,
                                                target=rig.dev.renderer.target,
                                                signature=sig))
        LSZ[sym] = (lsz, 1, 1)
        B_SIGS[sym] = nvals
    rig.dev.synchronize()
    print("[b] session-B programs loaded", flush=True)


def capture_real_routing(rig):
    """The Session-A cp4k pattern: production PF seq + an eids capture after
    every rt8e256; feed a real prose 256-chunk; one graph run."""
    import numpy as np
    from MM_P7_lib import build_seq7
    from mm_a_graph import mkgraph_unc as mkgraph
    if "cp4k" not in rig.K:
        load_prog(rig, "MM_A_cp4k.cubin", "cp4k", 256)
    spk = f"s{CTXK}"
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=spk, S=8, pf=True)
    cap = rig.alloc(40 * PF * 8 * 2)
    seq2 = []; li = 0
    for ent in seq:
        seq2.append(ent)
        if ent[0] == "rt8e256":
            dst = cap.offset(offset=li * PF * 8 * 2, size=PF * 8 * 2)
            seq2.append(("cp4k", (rig.PFB["eidsb"], dst), 8, (PF * 8,)))
            li += 1
    assert li == 40
    gr = mkgraph(rig, seq2, "g2cap")
    rig.reset_states(1024)
    d = json.load(open(os.path.join(BASE, "eval", "data", "ppl_prose_ids.json")))
    ids = []
    while len(ids) < PF:
        ids += [int(t) for t in d]
    ids = np.ascontiguousarray(np.array(ids[:PF], dtype=np.int32))
    rig.pf_ids_view[:] = memoryview(ids.data)
    rig.pos_view[0] = 0
    gr.step()
    return rig.dn(cap, (40, PF * 8), np.uint16)


def sort_ref(eids_np):
    """numpy rank-count sim (the exact mmsort8 semantics)."""
    import numpy as np
    cnt = np.bincount(eids_np, minlength=256).astype(np.int64)
    eoff = np.zeros(257, dtype=np.int32)
    eoff[1:] = np.cumsum(cnt)
    plist = np.argsort(eids_np, kind="stable").astype(np.uint16)
    items = []
    for e in range(256):
        tc = int((cnt[e] + TS - 1) // TS)
        for it in range(tc):
            for rs in range(RS):
                items.append((e << 20) | (rs << 16) | (it * TS))
    return eoff, plist, np.array(items, dtype=np.uint32), len(items)


def run_sort_arm(rig, eids_all, res):
    import numpy as np
    arm = res.setdefault("sort", {})
    eoffb = rig.alloc(257 * 4); plistb = rig.alloc(2048 * 2)
    itemsb = rig.alloc(4096 * 4); nitb = rig.alloc(4)
    ok_det, ok_ref, worst = True, True, None
    layers = sorted(set([0, 5, 20, 33, 34, 38, 39] + list(range(0, 40, 7))))
    for L in layers:
        eids = np.ascontiguousarray(eids_all[L])
        rig.dev.allocator._copyin(rig.PFB["eidsb"], memoryview(eids.data))
        outs = []
        for _ in range(2):
            rig.K["mmsort8"](rig.PFB["eidsb"], eoffb, plistb, itemsb, nitb,
                             global_size=(1, 1, 1), local_size=(1024, 1, 1),
                             vals=(PF * 8,), wait=True)
            outs.append((rig.dn(eoffb, (257,), np.int32).copy(),
                         rig.dn(plistb, (PF * 8,), np.uint16).copy(),
                         rig.dn(itemsb, (4096,), np.uint32).copy(),
                         int(rig.dn(nitb, (1,), np.int32)[0])))
        det = all((outs[0][i] == outs[1][i]).all() for i in range(3)) and outs[0][3] == outs[1][3]
        ok_det &= det
        re_, rp, ri, rn = sort_ref(eids)
        # nit may index into items; compare the live prefix only
        ref_match = (outs[0][0] == re_).all() and (outs[0][1] == rp).all() \
            and (outs[0][2][:rn] == ri).all() and outs[0][3] == rn
        ok_ref &= ref_match
        worst = L if not (det and ref_match) else worst
        if not (det and ref_match):
            print(f"[sort] L{L}: det={det} ref={ref_match} nit={outs[0][3]} vs {rn}", flush=True)
    t = graph_time(rig, rig.K["mmsort8"],
                   (rig.PFB["eidsb"], eoffb, plistb, itemsb, nitb),
                   (1, 1, 1), (PF * 8,))
    arm.update({"det_x2": ok_det, "exact_vs_numpy": ok_ref, "worst_layer": worst,
                "layers": len(layers), "ms": round(t, 4)})
    print(f"[sort] det_x2={ok_det} exact_vs_numpy={ok_ref} ({len(layers)} layers) "
          f"mmsort8={t*1000:.1f}us", flush=True)
    fsync_json(OUT, res)
    return eoffb, plistb, itemsb, nitb


def run_grouped_arm(rig, eids_all, bufs, res):
    import numpy as np
    eoffb, plistb, itemsb, nitb = bufs
    arm = res.setdefault("grouped", {})
    rng = np.random.default_rng(777)
    hnb = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.3)
    act = rig.up(rng.standard_normal((PF * 8, 512)).astype(np.float32) * 0.3)
    y1 = rig.alloc(PF * 8 * 512 * 4)     # pair-walk out (fp32)
    y2 = rig.alloc(PF * 8 * 512 * 4)
    p1 = rig.alloc(PF * 8 * 2048 * 2)    # dn out (fp16)
    p2 = rig.alloc(PF * 8 * 2048 * 2)
    actsh1 = rig.alloc(PF * 512 * 4)
    actsh2 = rig.alloc(PF * 512 * 4)

    # layer classes: L0 modal (up IQ3_S + dn IQ4_XS), L34 (dn6), L39 (up4+dn6)
    for L in (0, 34, 39):
        eids = np.ascontiguousarray(eids_all[L])
        rig.dev.allocator._copyin(rig.PFB["eidsb"], memoryview(eids.data))
        rig.K["mmsort8"](rig.PFB["eidsb"], eoffb, plistb, itemsb, nitb,
                         global_size=(1, 1, 1), local_size=(1024, 1, 1),
                         vals=(PF * 8,), wait=True)
        row = {}
        # ---- UP family ----
        if L == 39:
            ks, kn = "gx8e256up4", "gxm_up4"
            buf_extra_s = (rig.iq4nl,); buf_extra_n = (rig.iq4nl,)
        else:
            ks, kn = "gx8e256up", "gxm_up"
            buf_extra_s = (rig.gridf,); buf_extra_n = (rig.gridf,)
        bs = (rig.PTB_UP[L], rig.PFB["eidsb"], hnb) + buf_extra_s + (y1,)
        bn = (rig.PTB_UP[L], itemsb, nitb, eoffb, plistb, hnb) + buf_extra_n + (y2,)
        run_checked(rig, rig.K[ks], bs, (PF * 8, 1, 1), (), y1, y1.size)
        run_checked(rig, rig.K[kn], bn, (GN, 1, 1), (), y2, y2.size)
        be1 = bit_exact(rig, y1, y2)
        run_checked(rig, rig.K[kn], bn, (GN, 1, 1), (), y2, y2.size)
        be2 = bit_exact(rig, y1, y2)
        t_s = graph_time(rig, rig.K[ks], bs, (PF * 8, 1, 1), ())
        t_n = graph_time(rig, rig.K[kn], bn, (GN, 1, 1), ())
        row["up"] = {"bit_exact": be1 and be2, "stock_ms": round(t_s, 4),
                     "grouped_ms": round(t_n, 4), "speedup": round(t_s / t_n, 2)}
        up_name, up_bn = kn, bn
        # ---- DN family ----
        if L in (34, 39):
            ks, kn = "gx8e256dn6", "gxm_dn6"
            bs = (rig.PTB_DN[L], rig.PFB["eidsb"], act, p1)
            bn = (rig.PTB_DN[L], itemsb, nitb, eoffb, plistb, act, p2)
        else:
            ks, kn = "gx8e256dn", "gxm_dn"
            bs = (rig.PTB_DN[L], rig.PFB["eidsb"], act, rig.iq4nl, p1)
            bn = (rig.PTB_DN[L], itemsb, nitb, eoffb, plistb, act, rig.iq4nl, p2)
        dn_name, dn_bn = kn, bn
        run_checked(rig, rig.K[ks], bs, (PF * 8, 1, 1), (), p1, p1.size)
        run_checked(rig, rig.K[kn], bn, (GN, 1, 1), (), p2, p2.size)
        be1 = bit_exact(rig, p1, p2)
        run_checked(rig, rig.K[kn], bn, (GN, 1, 1), (), p2, p2.size)
        be2 = bit_exact(rig, p1, p2)
        t_s = graph_time(rig, rig.K[ks], bs, (PF * 8, 1, 1), ())
        t_n = graph_time(rig, rig.K[kn], bn, (GN, 1, 1), ())
        row["dn"] = {"bit_exact": be1 and be2, "stock_ms": round(t_s, 4),
                     "grouped_ms": round(t_n, 4), "speedup": round(t_s / t_n, 2)}
        # ---- L9 carveout A/B at the exact in-plan grid: default min-fitting
        # (16.6KB usage -> 32KB cfg -> 1 CTA/SM) vs 64KB (3 CTAs/SM).
        if "co" not in row.get("up", {}):
            for pn in ("gxm_up", "gxm_up4", "gxm_dn", "gxm_dn6"):
                if pn in rig.K:
                    rig.K[pn].qmd.write(min_sm_config_shared_mem_size=17,
                                        target_sm_config_shared_mem_size=17)
            t_up_co = graph_time(rig, rig.K[up_name], up_bn, (GN, 1, 1), ())
            t_dn_co = graph_time(rig, rig.K[dn_name], dn_bn, (GN, 1, 1), ())
            run_checked(rig, rig.K[up_name], up_bn, (GN, 1, 1), (), y2, y2.size)
            be_co = bit_exact(rig, y1, y2)
            row["up"]["co64_ms"] = round(t_up_co, 4)
            row["up"]["co64_speedup"] = round(t_s / t_up_co, 2)
            row["up"]["co64_bit_exact"] = be_co
            row["dn"]["co64_ms"] = round(t_dn_co, 4)
            row["dn"]["co64_speedup"] = round(row["dn"]["stock_ms"] / t_dn_co, 2)
            for pn in ("gxm_up", "gxm_up4", "gxm_dn", "gxm_dn6"):
                if pn in rig.K:
                    rig.K[pn].qmd.write(min_sm_config_shared_mem_size=9,
                                        target_sm_config_shared_mem_size=9)
        arm[f"L{L}"] = row
        print(f"[grp] L{L}: up {row['up']} | dn {row['dn']}", flush=True)
        fsync_json(OUT, res)

    # ---- shared expert ----
    sh = res.setdefault("shexp", {})
    w = rig.W[0]
    ysh1 = rig.alloc(PF * 2048 * 4); ysh2 = rig.alloc(PF * 2048 * 4)
    bs = (w["sg"], w["su"], w["sd"], hnb, ysh1)
    run_checked(rig, rig.K["shexp8"], bs, (PF, 1, 1), (), ysh1, ysh1.size)
    rig.K["shgu32"](w["sg"], w["su"], hnb, actsh1, global_size=(64, 1, 1),
                    local_size=(256, 1, 1), vals=(PF,), wait=True)
    rig.K["shdn32"](w["sd"], actsh1, ysh2, global_size=(256, 1, 1),
                    local_size=(256, 1, 1), vals=(PF,), wait=True)
    be1 = bit_exact(rig, ysh1, ysh2)
    t_s = graph_time(rig, rig.K["shexp8"], bs, (PF, 1, 1), ())
    gr1 = MGUnc(rig, [(rig.K["shgu32"], (w["sg"], w["su"], hnb, actsh1), (64, 1, 1), (PF,))], "shgu1")
    gr2 = MGUnc(rig, [(rig.K["shdn32"], (w["sd"], actsh1, ysh2), (256, 1, 1), (PF,))], "shdn1")
    nv = rig.nv_wait_timeline; dev = rig.dev
    ts = []
    for i in range(REPS + 1):
        t0 = time.perf_counter()
        v1 = dev.next_timeline(); gr1.submit(v1 - 1, v1); nv(dev, v1, what="sh1", timeout_s=60.0)
        v2 = dev.next_timeline(); gr2.submit(v2 - 1, v2); nv(dev, v2, what="sh2", timeout_s=60.0)
        if i: ts.append((time.perf_counter() - t0) * 1e3)
    sh.update({"bit_exact": be1, "stock_ms": round(t_s, 4),
               "new_ms": round(min(ts), 4), "speedup": round(t_s / min(ts), 2)})
    print(f"[shexp] {sh}", flush=True)
    fsync_json(OUT, res)


def run_gvs_arm(rig, res):
    import numpy as np
    from MM_P34_ports import GDN_LAYERS, ATTN_LAYERS
    aL = ATTN_LAYERS[0]
    rng = np.random.default_rng(4242)
    x2 = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.3)
    x4 = rig.up(rng.standard_normal((PF, 4096)).astype(np.float32) * 0.3)
    hr = rig.up(rng.standard_normal((PF, 2048)).astype(np.float32) * 0.5)
    arm = res.setdefault("gvs", {})
    classes = [
        ("qkv", "p", rig.W[0]["qkv"], 8192),
        ("z",   "p", rig.W[0]["z"],   4096),
        ("k",   "p", rig.W[aL]["k"],   512),
        ("v",   "p", rig.W[aL]["v"],   512),
        ("q",   "p", rig.W[aL]["q"],  8192),
        ("out", "r", rig.W[0]["out"], 2048),
        ("o",   "r", rig.W[aL]["o"],  2048),
    ]
    for tag, kind, w, rows in classes:
        y1 = rig.alloc(PF * rows * 4); y2 = rig.alloc(PF * rows * 4)
        if kind == "p":
            bs = (w, x2, y1); bn = (w, x2, y2)
            gs, gn = (rows // 32, PF, 1), (rows // 8, 1, 1)
            ks, kn = "gv8k2048p", "gvs32k2048"
            vs, vn = (rows,), (rows, PF)
        else:
            bs = (w, x4, hr, y1); bn = (w, x4, hr, y2)
            gs, gn = (rows // 32, PF, 1), (rows // 8, 1, 1)
            ks, kn = "gv8k4096r", "gvs32k4096r"
            vs, vn = (rows,), (rows, PF)
        run_checked(rig, rig.K[ks], bs, gs, vs, y1, y1.size)
        run_checked(rig, rig.K[kn], bn, gn, vn, y2, y2.size)
        be1 = bit_exact(rig, y1, y2)
        run_checked(rig, rig.K[kn], bn, gn, vn, y2, y2.size)
        be2 = bit_exact(rig, y1, y2)
        t_s = graph_time(rig, rig.K[ks], bs, gs, vs)
        t_n = graph_time(rig, rig.K[kn], bn, gn, vn)
        arm[tag] = {"bit_exact": be1 and be2, "stock_ms": round(t_s, 4),
                    "new_ms": round(t_n, 4), "speedup": round(t_s / t_n, 2)}
        print(f"[gvs] {tag:4s} rows={rows:5d} stock {t_s:7.3f} new {t_n:7.3f} "
              f"x{t_s/t_n:5.2f} bit_exact={be1 and be2}", flush=True)
        fsync_json(OUT, res)
    # ab class
    ab1 = rig.alloc(PF * 64 * 4); ab2 = rig.alloc(PF * 64 * 4)
    bs = (rig.W[0]["wa"], rig.W[0]["wb"], x2, ab1)
    bn = (rig.W[0]["wa"], rig.W[0]["wb"], x2, ab2)
    run_checked(rig, rig.K["gvf32ab"], bs, (PF, 1, 1), (), ab1, ab1.size)
    run_checked(rig, rig.K["gvsab"], bn, (16, PF // 32, 1), (PF,), ab2, ab2.size)
    be1 = bit_exact(rig, ab1, ab2)
    run_checked(rig, rig.K["gvsab"], bn, (16, PF // 32, 1), (PF,), ab2, ab2.size)
    be2 = bit_exact(rig, ab1, ab2)
    t_s = graph_time(rig, rig.K["gvf32ab"], bs, (PF, 1, 1), ())
    t_n = graph_time(rig, rig.K["gvsab"], bn, (16, PF // 32, 1), (PF,))
    arm["ab"] = {"bit_exact": be1 and be2, "stock_ms": round(t_s, 4),
                 "new_ms": round(t_n, 4), "speedup": round(t_s / t_n, 2)}
    print(f"[gvs] ab    stock {t_s:7.3f} new {t_n:7.3f} x{t_s/t_n:5.2f} "
          f"bit_exact={be1 and be2}", flush=True)
    fsync_json(OUT, res)


if __name__ == "__main__":
    import numpy as np
    res = {}
    if os.path.exists(OUT):
        try: res = json.load(open(OUT))
        except Exception: res = {}
    print("[g2] booting Rig7...", flush=True)
    from MM_P7_lib import Rig7
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    load_b(rig)
    print("[g2] capturing real routing (prose 256-chunk)...", flush=True)
    eids_all = capture_real_routing(rig)
    print(f"[g2] eids captured: L0 distinct={len(np.unique(eids_all[0]))} "
          f"maxbin={np.bincount(eids_all[0], minlength=256).max()}", flush=True)
    bufs = run_sort_arm(rig, eids_all, res)
    run_grouped_arm(rig, eids_all, bufs, res)
    run_gvs_arm(rig, res)
    ok = (res.get("sort", {}).get("det_x2") and res.get("sort", {}).get("exact_vs_numpy")
          and all(v.get("bit_exact") for r in res.get("grouped", {}).values() for v in r.values())
          and res.get("shexp", {}).get("bit_exact")
          and all(v.get("bit_exact") for v in res.get("gvs", {}).values()))
    spd = [v["speedup"] for r in res.get("grouped", {}).values() for v in r.values()]
    res["g2_verdict"] = {"all_bit_exact": bool(ok), "grouped_speedups": spd,
                         "kill_lt_2x": any(s < 2.0 for s in spd)}
    fsync_json(OUT, res)
    print(f"[g2] VERDICT: {res['g2_verdict']}", flush=True)
    print("[g2] done ->", OUT, flush=True)
