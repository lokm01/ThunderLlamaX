#!/usr/bin/env python3
"""MM SESSION E / items 1b+2+3 POC: the routed-dn gathered mma (gxd_gm),
the shared-expert mma pair (shgm512/sdm2048), the split-row scan
(k2s36h_256 + k2nz36) -- correctness + isolated + HONEST in-chunk.

ARMS (resumable, fsync)
 1. eager correctness on REAL data (post warm chunk):
    dn:   rt8e256+mmsort8 -> gxm_up(actb) -> gxm_dnf partsb_ref vs
          gxd_gm partsb_new (fp16 relerr/maxdev/signflips, det x2)
    shgu: shgu32 actsh_ref vs shgm512 actsh_new (relerr class)
    shdn: shdn32 shb_ref vs sdm2048 shb_new
    scan: stock k2s36_256 (S0 snapshot, y_ref) vs k2s36h_256+k2nz36
          (S restored, y_new) -- S must be BIT-EQUAL, y Tier-2 relerr
 2. isolated in-graph (fresh graphs, min-of-8) per pair.
 3. HONEST in-graph: full PF chunk (D+PFU) with each swap individually
    + ALL combined -> chunk ms + tps.
Output: engine0/mm/mm_e3_poc.json"""
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
OUT = os.path.join(BASE, "engine0", "mm", "mm_e3_poc.json")

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
    print("[e3] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev

    FAM = os.getenv("MM_E3_FAM", "all")
    ALLNEW = {"gxd_gm": ("MM_E_gxd_gm.cubin", (), (256, 1, 1)),
              "shgm512": ("MM_E_shgm512.cubin", (rig.INT_SIG,), (256, 1, 1)),
              "sdm2048": ("MM_E_sdm2048.cubin", (rig.INT_SIG,), (256, 1, 1)),
              "k2s36h_256": ("MM_E_k2s36h_256.cubin", (), (256, 1, 1)),
              "k2nz36": ("MM_E_k2nz36.cubin", (rig.INT_SIG,), (128, 1, 1))}
    LOAD = {"dn": ["gxd_gm"], "sh": ["shgm512", "sdm2048"],
            "scan": ["k2s36h_256", "k2nz36"], "all": list(ALLNEW)}[FAM]
    NEW = {k: ALLNEW[k] for k in LOAD}
    for nm, (cb, sig, lsz) in NEW.items():
        lib = open(f"{BASE}/{cb}", "rb").read()
        prg = NVProgram(dev, TinyELF(lib=lib, name=nm,
                                     target=dev.renderer.target,
                                     signature=sig))
        LSZ[nm] = lsz
        rig.K[nm] = prg
        print(f"[e3] {nm} loaded regs={prg.regs_usage} smem={prg.shmem_usage}",
              flush=True)
    # scan scratch (always -- the pairs dict references them eagerly)
    rig.YQB = rig.alloc(256 * 4096 * 4); rig.keep.append(rig.YQB)
    rig.YSSB = rig.alloc(256 * 64 * 4); rig.keep.append(rig.YSSB)

    for k in ("MM_PFG", "MM_PFM", "MM_PFT", "MM_PFW", "MM_PFK", "MM_PFU"):
        os.environ[k] = "1"
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    seqd = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                      spk=f"s{CTXK}", S=S, pf=True)
    grpf = mkgraph(rig, seqd, "e3_pf")
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids[:PF].data)
    rig.pos_view[0] = 0
    grpf.step(); grpf.fence()

    hnb = rig.PFB["hnb"]
    w0 = rig.W[0]
    PFB = rig.PFB
    SENT = np.uint32(0x7fbfffbf)

    # ---- arm 1: eager correctness ----
    if os.getenv("MM_E3_EAGER", "0") == "1" and "eager_detailed" not in res:
        out = {}
        # live sort state
        rig.K["rt8e256"](w0["rt"], w0["wsh"], hnb, PFB["eidsb"],
                         PFB["gatesb"], PFB["sgb"], global_size=(PF, 1, 1),
                         local_size=(1024, 1, 1), vals=(), wait=True)
        rig.K["mmsort8"](PFB["eidsb"], rig.EOFFB, rig.PLISTB, rig.ITEMSB,
                         rig.NITB, global_size=(1, 1, 1), local_size=(1024, 1, 1),
                         vals=(PF * 8,), wait=True)
        eoff = rig.dn(rig.EOFFB, (257,), np.int32).copy()
        npair = int(eoff[256])

        # ---- dn: actb via gxm_up, then gxm_dnf vs gxd_gm ----
        rig.K["gxm_up"](rig.PTB_UP[0], rig.ITEMSB, rig.NITB, rig.EOFFB,
                        rig.PLISTB, hnb, rig.gridf, PFB["actb"],
                        global_size=(1024, 1, 1), local_size=(256, 1, 1),
                        vals=(), wait=True)
        SENTU = np.uint16(0x7bff)
        # partsb is fp16: use a u16 sentinel
        pb = PFB["partsb"]
        npb = 2048 * 2048
        ref_b = rig.alloc(npb * 2); rig.keep.append(ref_b)
        new_b = rig.alloc(npb * 2); rig.keep.append(new_b)
        rig.dev.allocator._copyin(ref_b, memoryview(np.full(npb, SENTU, np.uint16).tobytes()))
        rig.K["gxm_dnf"](rig.PTB_DN[0], rig.ITEMSB, rig.NITB, rig.EOFFB,
                         rig.PLISTB, PFB["actb"], rig.gridf, ref_b,
                         global_size=(1024, 1, 1), local_size=(256, 1, 1),
                         vals=(), wait=True)
        yref = rig.dn(ref_b, (npb,), np.uint16).copy()
        det = []
        for _ in range(2):
            rig.dev.allocator._copyin(new_b, memoryview(np.full(npb, SENTU, np.uint16).tobytes()))
            rig.K["gxd_gm"](rig.PTB_DN[0], rig.EOFFB, rig.PLISTB, PFB["actb"],
                            rig.gridf, new_b, global_size=(4096, 1, 1),
                            local_size=(256, 1, 1), vals=(), wait=True)
            det.append(rig.dn(new_b, (npb,), np.uint16).copy())
        detok = bool(np.array_equal(det[0], det[1]))
        sents = int((det[0] == SENTU).sum())
        a = det[0].view(np.float16).astype(np.float64)
        b = yref.view(np.float16).astype(np.float64)
        live = b != 0  # only written region matters; sentinel rows excluded
        re_ = float(np.linalg.norm(a[live] - b[live]) / max(np.linalg.norm(b[live]), 1e-30))
        mad = float(np.abs(a[live] - b[live]).max())
        sfl = int((np.signbit(a[live]) != np.signbit(b[live])).sum())
        out["dn"] = {"npair": npair, "relerr": re_, "maxabs": mad,
                     "sign_flips": sfl, "det_x2": detok, "sentinel": sents}
        print(f"[e3] dn: relerr {re_:.3e} maxabs {mad:.3e} sfl {sfl} det {detok} sent {sents}",
              flush=True)

        # ---- shgu / shdn ----
        asz, ssz = 256 * 512, 256 * 2048
        aref = rig.alloc(asz * 4); rig.keep.append(aref)
        anew = rig.alloc(asz * 4); rig.keep.append(anew)
        rig.K["shgu32"](w0["sg"], w0["su"], hnb, aref, global_size=(512 // 8, 1, 1),
                        local_size=(256, 1, 1), vals=(PF,), wait=True)
        ya = rig.dn(aref, (asz,)).copy()
        rig.K["shgm512"](w0["sg"], w0["su"], hnb, anew,
                         global_size=((PF // 32) * (512 // 64), 1, 1),
                         local_size=(256, 1, 1), vals=(PF,), wait=True)
        yb = rig.dn(anew, (asz,)).copy()
        rea = float(np.linalg.norm(yb - ya) / max(np.linalg.norm(ya), 1e-30))
        out["shgu"] = {"relerr": rea, "maxabs": float(np.abs(yb - ya).max())}
        print(f"[e3] shgu: relerr {rea:.3e}", flush=True)

        sref = rig.alloc(ssz * 4); rig.keep.append(sref)
        snew = rig.alloc(ssz * 4); rig.keep.append(snew)
        rig.K["shdn32"](w0["sd"], aref, sref, global_size=(2048 // 8, 1, 1),
                        local_size=(256, 1, 1), vals=(PF,), wait=True)
        ys = rig.dn(sref, (ssz,)).copy()
        rig.K["sdm2048"](w0["sd"], aref, snew,
                         global_size=((PF // 32) * (2048 // 64), 1, 1),
                         local_size=(256, 1, 1), vals=(PF,), wait=True)
        ys2 = rig.dn(snew, (ssz,)).copy()
        reb = float(np.linalg.norm(ys2 - ys) / max(np.linalg.norm(ys), 1e-30))
        out["shdn"] = {"relerr": reb, "maxabs": float(np.abs(ys2 - ys).max())}
        print(f"[e3] shdn: relerr {reb:.3e}", flush=True)

        # ---- scan: S0 snapshot, stock vs split ----
        gi = 0
        nS = 32 * 128 * 128
        S0 = rig.dn(rig.SV[gi], (nS,)).copy()
        ysc = rig.alloc(PF * 4096 * 4); rig.keep.append(ysc)
        rig.K["k2s36_256"](PFB["qkvsb"], PFB["abb"], w0["al"], w0["dt"], w0["sn"],
                           PFB["zb"], rig.SV[gi], PFB["gyb"],
                           global_size=(32, 1, 1), local_size=(256, 1, 1),
                           vals=(), wait=True)
        yref_s = rig.dn(PFB["gyb"], (PF * 4096,)).copy()
        Sref = rig.dn(rig.SV[gi], (nS,)).copy()
        # restore S0; run split pair into ysc
        rig.dev.allocator._copyin(rig.SV[gi], memoryview(S0.tobytes()))
        rig.K["k2s36h_256"](PFB["qkvsb"], PFB["abb"], w0["al"], w0["dt"],
                            rig.SV[gi], rig.YQB, rig.YSSB,
                            global_size=(64, 1, 1), local_size=(256, 1, 1),
                            vals=(), wait=True)
        rig.K["k2nz36"](rig.YQB, rig.YSSB, w0["sn"], PFB["zb"], ysc, PF,
                        global_size=(PF * 32, 1, 1), local_size=(128, 1, 1),
                        vals=(PF,), wait=True)
        ynew_s = rig.dn(ysc, (PF * 4096,)).copy()
        Snew = rig.dn(rig.SV[gi], (nS,)).copy()
        S_eq = bool(np.array_equal(Sref, Snew))
        res_ = float(np.linalg.norm(ynew_s - yref_s) / max(np.linalg.norm(yref_s), 1e-30))
        out["scan"] = {"S_bitexact": S_eq, "y_relerr": res_,
                       "y_maxabs": float(np.abs(ynew_s - yref_s).max())}
        print(f"[e3] scan: S bit-exact {S_eq} y relerr {res_:.3e}", flush=True)
        res["eager"] = out
        fsync_json(OUT, res)

    # ---- arm 2: isolated ----
    if "isolated" not in res:
        iso = {}
        pairs = {
            "dn": ([("gxm_dnf", (rig.PTB_DN[0], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                 rig.PLISTB, PFB["actb"], rig.gridf, PFB["partsb"]), 1024, ())],
                   [("gxd_gm", (rig.PTB_DN[0], rig.EOFFB, rig.PLISTB, PFB["actb"],
                                rig.gridf, PFB["partsb"]), 4096, ())]),
            "shgu": ([("shgu32", (w0["sg"], w0["su"], hnb, rig.ACTSHB), 512 // 8, (PF,))],
                     [("shgm512", (w0["sg"], w0["su"], hnb, rig.ACTSHB),
                       (PF // 32) * (512 // 64), (PF,))]),
            "shdn": ([("shdn32", (w0["sd"], rig.ACTSHB, PFB["shb"]), 2048 // 8, (PF,))],
                     [("sdm2048", (w0["sd"], rig.ACTSHB, PFB["shb"]),
                       (PF // 32) * (2048 // 64), (PF,))]),
            "scan": ([("k2s36_256", (PFB["qkvsb"], PFB["abb"], w0["al"], w0["dt"],
                                     w0["sn"], PFB["zb"], rig.SV[0], PFB["gyb"]), 32, ())],
                     [("k2s36h_256", (PFB["qkvsb"], PFB["abb"], w0["al"], w0["dt"],
                                      rig.SV[0], rig.YQB, rig.YSSB), 64, ()),
                      ("k2nz36", (rig.YQB, rig.YSSB, w0["sn"], PFB["zb"], PFB["gyb"]),
                       PF * 32, (PF,))]),
        }
        fmap = {"dn": ["dn"], "sh": ["shgu", "shdn"], "scan": ["scan"]}
        fams = fmap.get(FAM, list(pairs)) if FAM != "all" else list(pairs)
        for nm, (sS, sN) in pairs.items():
            if nm not in fams:
                continue
            tS = graph_time(rig, sS, f"e3_i_{nm}_s")
            tN = graph_time(rig, sN, f"e3_i_{nm}_n")
            iso[nm] = {"stock_ms": round(tS, 4), "new_ms": round(tN, 4),
                       "x": round(tS / tN, 2)}
            print(f"[e3] iso {nm}: {tS:.3f} -> {tN:.3f} ms x{tS/tN:.2f}", flush=True)
        res["isolated"] = iso
        fsync_json(OUT, res)

    # ---- arm 3: HONEST in-chunk ----
    if os.getenv("MM_E3_NOCHUNK", "0") != "1" and "inchunk" not in res:
        def swap(seq, which):
            out = []
            for name, bufs, g, v in seq:
                if which == "dn" and name == "gxm_dnf":
                    out.append(("gxd_gm", (bufs[0], bufs[3], bufs[4], bufs[5],
                                           bufs[6], bufs[7]), 4096, ()))
                    continue
                if which == "sh" and name == "shgu32":
                    out.append(("shgm512", bufs, (PF // 32) * (512 // 64), (PF,)))
                    continue
                if which == "sh" and name == "shdn32":
                    out.append(("sdm2048", bufs, (PF // 32) * (2048 // 64), (PF,)))
                    continue
                if which == "scan" and name == "k2s36_256":
                    # bufs = (qkvs, ab, al, dt, sn, z, SV, y)
                    out.append(("k2s36h_256", (bufs[0], bufs[1], bufs[2], bufs[3],
                                               bufs[6], rig.YQB, rig.YSSB), 64, ()))
                    out.append(("k2nz36", (rig.YQB, rig.YSSB, bufs[4], bufs[5],
                                           bufs[7]), PF * 32, (PF,)))
                    continue
                out.append((name, bufs, g, v))
            return out

        def swap_all(seq):
            s = seq
            for w in ("dn", "sh", "scan"):
                s = swap(s, w)
            return s

        def chunk_time(sq, tag):
            g = mkgraph(rig, sq, tag)
            rig.reset_states(1024)
            rig.pf_ids_view[:] = memoryview(ids[:PF].data)
            rig.pos_view[0] = 0
            g.step(); g.fence()
            ts = []
            for i in range(9):
                t0 = time.perf_counter()
                g.step()
                if i:
                    ts.append((time.perf_counter() - t0) * 1e3)
            g.fence()
            return round(min(ts), 2)

        wmap = {"dn": ["dn"], "sh": ["sh"], "scan": ["scan"]}
        wsel = wmap.get(FAM, ["dn", "sh", "scan"]) if FAM != "all" else ["dn", "sh", "scan"]
        row = {"base": chunk_time(seqd, "e3_c_base")}
        for w in wsel:
            row[w] = chunk_time(swap(seqd, w), f"e3_c_{w}")
            row[f"{w}_d"] = round(row["base"] - row[w], 2)
        row["all"] = chunk_time(swap_all(seqd), "e3_c_all")
        row["all_tps"] = round(PF / (row["all"] / 1e3), 1)
        row["base_tps"] = round(PF / (row["base"] / 1e3), 1)
        res["inchunk"] = row
        print(f"[e3] inchunk: {row}", flush=True)
        fsync_json(OUT, res)
    print("[e3] done", flush=True)


if __name__ == "__main__":
    main()
