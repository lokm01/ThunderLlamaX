#!/usr/bin/env python3
"""MM SESSION E / item-1 POC: the gathered-row-list mma routed gate+up
(gxu_gm) vs gxm_up (the Session-B grouped kernel, the 54%-of-residual miss).

ARMS (resumable, fsync after every arm)
 1. eager correctness on REAL routed data: one full D-config chunk ->
    live hnb; eager rt8e256+mmsort8 -> eoff/plist; gxm_up -> yref;
    gxu_gm x2 (det) -> relerr/maxdev/sign-flips/sentinel vs yref.
 2. isolated in-graph (fresh fenced graphs, min-of-8): single-launch
    gxm_up vs gxu_gm (the ~4x-liar number -- context only).
 3. HONEST in-graph: full PF chunk graph at L=2048, D-config (gxm_up)
    vs D-config-gxu (every gxm_up entry substituted) -> chunk ms delta
    + projected tok/s. GO GATE: in-graph per-kernel >= 1.3x
    (x = tA_iso_total / (tA_iso_total - Dchunk)).
Output: engine0/mm/mm_e1_poc.json
"""
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
S = 8
OUT = os.path.join(BASE, "engine0", "mm", "mm_e1_poc.json")

from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_a_graph import mkgraph_unc as mkgraph


def graph_time(rig, seq, tag, reps=8):
    m = mkgraph(rig, seq, tag, fence_every=48)
    ts = []
    for i in range(reps + 1):
        t0 = time.perf_counter()
        m.step()
        if i:
            ts.append((time.perf_counter() - t0) * 1e3)
    m.fence()
    return min(ts)


def main():
    import numpy as np
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    from MM_P56_lib import LSZ
    from MM_P7_lib import Rig7, build_seq7
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    print("[e1] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev

    # load gxu_gm
    lib = open(f"{BASE}/MM_E_gxu_gm.cubin", "rb").read()
    prg = NVProgram(dev, TinyELF(lib=lib, name="gxu_gm",
                                 target=dev.renderer.target, signature=()))
    LSZ["gxu_gm"] = (256, 1, 1)
    rig.K["gxu_gm"] = prg
    print(f"[e1] gxu_gm loaded regs={prg.regs_usage} stack={prg.stack_usage} "
          f"smem={prg.shmem_usage}", flush=True)

    # one full D-config chunk -> live hnb + weights warm
    os.environ["MM_PFG"] = "1"; os.environ["MM_PFM"] = "1"; os.environ["MM_PFT"] = "1"
    os.environ["MM_PFW"] = "1"; os.environ["MM_PFK"] = "1"
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    seqd = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                      spk=f"s{CTXK}", S=S, pf=True)
    grpf = mkgraph(rig, seqd, "e1_pf")
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids[:PF].data)
    rig.pos_view[0] = 0
    grpf.step(); grpf.fence()

    hnb = rig.PFB["hnb"]
    w0 = rig.W[0]
    ptb = rig.PTB_UP[0]

    # ---- arm 1: eager correctness ----
    if "eager" not in res:
        PFB = rig.PFB
        rig.K["rt8e256"](w0["rt"], w0["wsh"], hnb, PFB["eidsb"],
                         PFB["gatesb"], PFB["sgb"], global_size=(PF, 1, 1),
                         local_size=(1024, 1, 1), vals=(), wait=True)
        rig.K["mmsort8"](PFB["eidsb"], rig.EOFFB, rig.PLISTB, rig.ITEMSB,
                         rig.NITB, global_size=(1, 1, 1), local_size=(1024, 1, 1),
                         vals=(PF * 8,), wait=True)
        eoff = rig.dn(rig.EOFFB, (257,), np.int32).copy()
        plist = rig.dn(rig.PLISTB, (2048,), np.uint16).copy()
        npair = int(eoff[256])
        nbins = int((np.diff(eoff) > 0).sum())
        bsz = np.diff(eoff)
        print(f"[e1] npair={npair} live_bins={nbins} bin mean={bsz.mean():.1f} "
              f"max={bsz.max()}", flush=True)

        SENT = np.uint32(0x7fbfffbf)
        yref_b = rig.alloc(npair * 512 * 4); rig.keep.append(yref_b)
        ynew_b = rig.alloc(npair * 512 * 4); rig.keep.append(ynew_b)
        rig.dev.allocator._copyin(yref_b, memoryview(np.full(npair * 512, SENT, np.uint32).tobytes()))
        rig.K["gxm_up"](ptb, rig.ITEMSB, rig.NITB, rig.EOFFB, rig.PLISTB,
                        hnb, rig.gridf, yref_b, global_size=(1024, 1, 1),
                        local_size=(256, 1, 1), vals=(), wait=True)
        yref = rig.dn(yref_b, (npair * 512,), np.uint32).copy()

        ys_det = []
        for _ in range(2):
            rig.dev.allocator._copyin(ynew_b, memoryview(np.full(npair * 512, SENT, np.uint32).tobytes()))
            rig.K["gxu_gm"](ptb, rig.EOFFB, rig.PLISTB, hnb, rig.gridf,
                            ynew_b, global_size=(2048, 1, 1),
                            local_size=(256, 1, 1), vals=(), wait=True)
            ys_det.append(rig.dn(ynew_b, (npair * 512,), np.uint32).copy())
        det = bool(np.array_equal(ys_det[0], ys_det[1]))
        sents = int((ys_det[0] == SENT).sum())
        a = ys_det[0].view(np.float32).astype(np.float64)
        b = yref.view(np.float32).astype(np.float64)
        re_ = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        mad = float(np.abs(a - b).max())
        sfl = int((np.signbit(a) != np.signbit(b)).sum())
        res["eager"] = {"npair": npair, "live_bins": nbins,
                        "bin_mean": float(bsz.mean()), "bin_max": int(bsz.max()),
                        "relerr": re_, "maxabs": mad, "sign_flips": sfl,
                        "det_x2": det, "sentinel": sents}
        print(f"[e1] eager: relerr {re_:.3e} maxabs {mad:.3e} signflips {sfl} "
              f"det {det} sentinel {sents}", flush=True)
        fsync_json(OUT, res)

    # ---- arm 2: isolated in-graph ----
    if "isolated" not in res:
        yb = rig.alloc(PF * 8 * 512 * 4); rig.keep.append(yb)
        seqS = [("gxm_up", (ptb, rig.ITEMSB, rig.NITB, rig.EOFFB, rig.PLISTB,
                            hnb, rig.gridf, yb), 1024, ())]
        seqM = [("gxu_gm", (ptb, rig.EOFFB, rig.PLISTB, hnb, rig.gridf, yb), 2048, ())]
        tS = graph_time(rig, seqS, "e1_iso_s")
        tM = graph_time(rig, seqM, "e1_iso_m")
        res["isolated"] = {"gxm_up_ms": round(tS, 3), "gxu_gm_ms": round(tM, 3),
                           "x": round(tS / tM, 2)}
        print(f"[e1] isolated: gxm_up {tS:.3f}ms gxu_gm {tM:.3f}ms x{tS/tM:.2f}",
              flush=True)
        fsync_json(OUT, res)

    # ---- arm 3: HONEST in-graph (full PF chunk, D-config vs D-config-gxu) ----
    if "inchunk" not in res:
        def swap_gxm(seq):
            out = []
            for ent in seq:
                name, bufs = ent[0], ent[1]
                if name == "gxm_up":
                    # bufs = (PTB, ITEMSB, NITB, EOFFB, PLISTB, hnb, gridf, actb)
                    out.append(("gxu_gm", (bufs[0], bufs[3], bufs[4], bufs[5],
                                           bufs[6], bufs[7]), 2048, ()))
                else:
                    out.append(ent)
            return out

        gA = mkgraph(rig, seqd, "e1_chunk_stock")
        seqX = swap_gxm(seqd)
        gB = mkgraph(rig, seqX, "e1_chunk_gxu")
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids[:PF].data)
        rig.pos_view[0] = 0
        gA.step(); gA.fence()

        def chunk_time(g, tag):
            ts = []
            for i in range(9):
                t0 = time.perf_counter()
                g.step()
                if i:
                    ts.append((time.perf_counter() - t0) * 1e3)
            g.fence()
            return min(ts)

        tA = chunk_time(gA, "stock")
        tB = chunk_time(gB, "gxu")
        d = tA - tB
        iso = res.get("isolated", {}).get("gxm_up_ms", 0.0)
        tot = iso * 39.0
        x = tot / (tot - d) if tot > d > 0 else 0.0
        res["inchunk"] = {"stock_ms": round(tA, 3), "gxu_ms": round(tB, 3),
                          "delta_ms": round(d, 3),
                          "stock_tps": round(PF / (tA / 1e3), 1),
                          "gxu_tps": round(PF / (tB / 1e3), 1),
                          "implied_gxmup_total_ms": round(tot, 1),
                          "implied_perkernel_x": round(x, 2)}
        print(f"[e1] inchunk: stock {tA:.1f}ms gxu {tB:.1f}ms d={d:.1f}ms "
              f"tps {PF/(tA/1e3):.1f}->{PF/(tB/1e3):.1f} implied_x {x:.2f}",
              flush=True)
        fsync_json(OUT, res)

    print("[e1] done", flush=True)


if __name__ == "__main__":
    main()
