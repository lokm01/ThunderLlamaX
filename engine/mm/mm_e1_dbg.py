#!/usr/bin/env python3
"""MM SESSION E / item-1 DEBUG v2: localize the gxu_gm-vs-gxm_up divergence.

Rounds (each fault-isolated + fsynced):
  A. real-data anchor: rt8e256+mmsort8 on live hnb -> gxm_up vs gxu_gm
  B. synthetic STRICT production-like eids (max bin <= 33): indicator-x
     k-sweep on expert E0's 16-pair bin -> per-k relerr + first-diff cells
Output: engine0/mm/mm_e1_dbg.json"""
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
OUT = os.path.join(BASE, "engine0", "mm", "mm_e1_dbg.json")
from mm_l1_poc import fsync_json

E0 = 5
KJS = [0, 1, 63, 64, 127, 128, 129, 191, 192, 255, 256, 320, 511, 512,
       1023, 1024, 2047, -1]


def main():
    import numpy as np
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    from MM_P56_lib import LSZ
    from MM_P7_lib import Rig7
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    print("[dbg] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev
    lib = open(f"{BASE}/MM_E_gxu_gm.cubin", "rb").read()
    prg = NVProgram(dev, TinyELF(lib=lib, name="gxu_gm",
                                 target=dev.renderer.target, signature=()))
    LSZ["gxu_gm"] = (256, 1, 1)
    rig.K["gxu_gm"] = prg

    # warm chunk (the e1 POC precedent)
    from MM_P7_lib import build_seq7
    from mm_f1b_b import load_prose_ids
    from mm_a_graph import mkgraph_unc as mkgraph
    for k in ("MM_PFG", "MM_PFM", "MM_PFT", "MM_PFW", "MM_PFK"):
        os.environ[k] = "1"
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    seqd = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                      spk=f"s{CTXK}", S=8, pf=True)
    grpf = mkgraph(rig, seqd, "dbg_pf")
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids[:PF].data)
    rig.pos_view[0] = 0
    grpf.step(); grpf.fence()
    print("[dbg] warm chunk done", flush=True)

    hnb = rig.PFB["hnb"]
    ptb = rig.PTB_UP[0]
    SENT = np.uint32(0x7fbfffbf)

    def cmp_run(npair, yref_b, ynew_b):
        yref = rig.dn(yref_b, (npair * 512,), np.uint32).copy().view(np.float32)
        ynew = rig.dn(ynew_b, (npair * 512,), np.uint32).copy().view(np.float32)
        a, b = ynew.astype(np.float64), yref.astype(np.float64)
        re_ = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        return re_, yref, ynew

    # ---- round A: real-data anchor ----
    if "A_real" not in res:
        try:
            PFB = rig.PFB
            rig.K["rt8e256"](rig.W[0]["rt"], rig.W[0]["wsh"], hnb, PFB["eidsb"],
                             PFB["gatesb"], PFB["sgb"], global_size=(PF, 1, 1),
                             local_size=(1024, 1, 1), vals=(), wait=True)
            rig.K["mmsort8"](PFB["eidsb"], rig.EOFFB, rig.PLISTB, rig.ITEMSB,
                             rig.NITB, global_size=(1, 1, 1), local_size=(1024, 1, 1),
                             vals=(PF * 8,), wait=True)
            eoff = rig.dn(rig.EOFFB, (257,), np.int32).copy()
            npair = int(eoff[256])
            yref_b = rig.alloc(npair * 512 * 4); rig.keep.append(yref_b)
            ynew_b = rig.alloc(npair * 512 * 4); rig.keep.append(ynew_b)
            rig.dev.allocator._copyin(yref_b, memoryview(np.full(npair * 512, SENT, np.uint32).tobytes()))
            rig.K["gxm_up"](ptb, rig.ITEMSB, rig.NITB, rig.EOFFB, rig.PLISTB,
                            hnb, rig.gridf, yref_b, global_size=(1024, 1, 1),
                            local_size=(256, 1, 1), vals=(), wait=True)
            rig.dev.allocator._copyin(ynew_b, memoryview(np.full(npair * 512, SENT, np.uint32).tobytes()))
            rig.K["gxu_gm"](ptb, rig.EOFFB, rig.PLISTB, hnb, rig.gridf,
                            ynew_b, global_size=(2048, 1, 1),
                            local_size=(256, 1, 1), vals=(), wait=True)
            re_, yref, ynew = cmp_run(npair, yref_b, ynew_b)
            res["A_real"] = {"npair": npair, "relerr": re_}
            print(f"[dbg] A real: relerr {re_:.4f}", flush=True)
        except Exception as ex:
            res["A_real"] = {"error": str(ex)[:200]}
            print(f"[dbg] A FAILED: {ex}", flush=True)
        fsync_json(OUT, res)

    # ---- round B: k-sweep on the REAL post-A sort state (real bins; a
    # synthetic-eid sort table FAULTS gxm_up itself -- latent edge, not
    # our target) with indicator x staged in an OWN buffer ----
    if "B_meta" not in res:
        eoff = rig.dn(rig.EOFFB, (257,), np.int32).copy()
        plist = rig.dn(rig.PLISTB, (2048,), np.uint16).copy()
        bsz = np.diff(eoff)
        res["B_meta"] = {"npair": int(eoff[256]), "bin_max": int(bsz.max()),
                         "bin_mean": float(bsz[bsz > 0].mean())}
        print(f"[dbg] B real bins: max {bsz.max()} mean {bsz[bsz>0].mean():.1f}",
              flush=True)
        fsync_json(OUT, res)

    npair = int(res["B_meta"]["npair"])
    yref_b = rig.alloc(npair * 512 * 4); rig.keep.append(yref_b)
    ynew_b = rig.alloc(npair * 512 * 4); rig.keep.append(ynew_b)
    xb = rig.alloc(PF * 2048 * 4); rig.keep.append(xb)   # OWN x buffer

    for kj in KJS:
        key = f"B_kj{kj}"
        if key in res:
            continue
        try:
            x = np.zeros((PF, 2048), dtype=np.float32)
            if kj < 0:
                x[:] = 1.0
            else:
                x[:, kj] = 1.0
            rig.dev.allocator._copyin(xb, memoryview(x.tobytes()))
            rig.dev.allocator._copyin(yref_b, memoryview(np.full(npair * 512, SENT, np.uint32).tobytes()))
            rig.K["gxm_up"](ptb, rig.ITEMSB, rig.NITB, rig.EOFFB, rig.PLISTB,
                            xb, rig.gridf, yref_b, global_size=(1024, 1, 1),
                            local_size=(256, 1, 1), vals=(), wait=True)
            rig.dev.allocator._copyin(ynew_b, memoryview(np.full(npair * 512, SENT, np.uint32).tobytes()))
            rig.K["gxu_gm"](ptb, rig.EOFFB, rig.PLISTB, xb, rig.gridf,
                            ynew_b, global_size=(2048, 1, 1),
                            local_size=(256, 1, 1), vals=(), wait=True)
            re_, yref, ynew = cmp_run(npair, yref_b, ynew_b)
            b = yref.astype(np.float64); a = ynew.astype(np.float64)
            diff = np.nonzero(np.abs(a - b) > 1e-3 * np.maximum(np.abs(b), 1e-3))[0]
            info = {"kj": kj, "relerr": round(re_, 5), "nbad": int(len(diff))}
            if len(diff):
                c = int(diff[0]); pair, row = c // 512, c % 512
                rows_bad = np.unique(diff % 512)
                info["first"] = {"row": row, "yref": round(float(b[c]), 5),
                                 "ynew": round(float(a[c]), 5)}
                info["bad_rows"] = [int(r) for r in rows_bad[:8]]
                info["nbad_rows"] = int(len(rows_bad))
                info["bad_row_span"] = [int(rows_bad.min()), int(rows_bad.max())]
            res[key] = info
            print(f"[dbg] kj={kj}: relerr {re_:.4f} nbad {len(diff)} {info.get('first','')}",
                  flush=True)
        except Exception as ex:
            res[key] = {"error": str(ex)[:150]}
            print(f"[dbg] kj={kj} FAILED: {str(ex)[:100]}", flush=True)
            fsync_json(OUT, res)
            break
        fsync_json(OUT, res)
    print("[dbg] done", flush=True)


if __name__ == "__main__":
    main()
