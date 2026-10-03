#!/usr/bin/env python3
"""MM SESSION E -- sh-pair numerics debug (the CE-bisect culprit):
shgu32 vs shgm512 (actsh) + shdn32 vs sdm2048 (shb) on live hnb, with
per-row/per-seat error structure (rowperm vs scale signatures)."""
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
OUT = os.path.join(BASE, "engine0", "mm", "mm_sh_dbg.json")
from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_a_graph import mkgraph_unc as mkgraph


def main():
    import numpy as np
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    from MM_P56_lib import LSZ
    from MM_P7_lib import Rig7, build_seq7
    res = {}
    print("[sd] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev
    for nm, cb, sig, lsz in (("shgm512", "MM_E_shgm512.cubin", (rig.INT_SIG,), (256, 1, 1)),
                             ("sdm2048", "MM_E_sdm2048.cubin", (rig.INT_SIG,), (256, 1, 1))):
        lib = open(f"{BASE}/{cb}", "rb").read()
        prg = NVProgram(dev, TinyELF(lib=lib, name=nm,
                                     target=dev.renderer.target, signature=sig))
        LSZ[nm] = lsz
        rig.K[nm] = prg
        print(f"[sd] {nm} regs={prg.regs_usage} smem={prg.shmem_usage}", flush=True)

    for k in ("MM_PFG", "MM_PFM", "MM_PFT", "MM_PFW", "MM_PFK"):
        os.environ[k] = "1"
    os.environ["MM_PFU"] = "1"; os.environ["MM_PFD"] = "1"
    os.environ["MM_PFR"] = "0"; os.environ["MM_PFS"] = "0"
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    seqd = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                      spk=f"s{CTXK}", S=8, pf=True)
    grpf = mkgraph(rig, seqd, "sd_pf")
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids[:PF].data)
    rig.pos_view[0] = 0
    grpf.step(); grpf.fence()
    hnb = rig.PFB["hnb"]
    w0 = rig.W[0]

    asz, ssz = 256 * 512, 256 * 2048
    aref = rig.alloc(asz * 4); rig.keep.append(aref)
    anew = rig.alloc(asz * 4); rig.keep.append(anew)
    sref = rig.alloc(ssz * 4); rig.keep.append(sref)
    snew = rig.alloc(ssz * 4); rig.keep.append(snew)

    def gstep(seq, tag):
        g = mkgraph(rig, seq, tag)
        g.step(); g.fence()

    # ---- per-layer sweep: find which layers break ----
    bad = []
    rels = []
    for L in range(40):
        w = rig.W[L]
        rig.K["shgu32"](w["sg"], w["su"], hnb, aref, global_size=(512 // 8, 1, 1),
                        local_size=(256, 1, 1), vals=(PF,), wait=True)
        ya = rig.dn(aref, (asz,)).copy()
        rig.K["shgm512"](w["sg"], w["su"], hnb, anew,
                         global_size=((PF // 32) * (512 // 64), 1, 1),
                         local_size=(256, 1, 1), vals=(PF,), wait=True)
        yb = rig.dn(anew, (asz,)).copy()
        re_ = float(np.linalg.norm(yb - ya) / max(np.linalg.norm(ya), 1e-30))
        rels.append(re_)
        if re_ > 1e-3:
            bad.append(L)
    res["shgu_perlayer"] = {"bad_layers": bad, "max_relerr": max(rels),
                            "relerr_l0": rels[0]}
    print(f"[sd] shgu per-layer: bad={bad} max={max(rels):.4f}", flush=True)
    fsync_json(OUT, res)

    bad2 = []
    rels2 = []
    if len(bad) == 0:
        for L in range(40):
            w = rig.W[L]
            rig.K["shdn32"](w["sd"], aref, sref, global_size=(2048 // 8, 1, 1),
                            local_size=(256, 1, 1), vals=(PF,), wait=True)
            ys_ = rig.dn(sref, (ssz,)).copy()
            rig.K["sdm2048"](w["sd"], aref, snew,
                             global_size=((PF // 32) * (2048 // 64), 1, 1),
                             local_size=(256, 1, 1), vals=(PF,), wait=True)
            ys2 = rig.dn(snew, (ssz,)).copy()
            re2 = float(np.linalg.norm(ys2 - ys_) / max(np.linalg.norm(ys_), 1e-30))
            rels2.append(re2)
            if re2 > 1e-3:
                bad2.append(L)
        res["shdn_perlayer"] = {"bad_layers": bad2, "max_relerr": max(rels2)}
        print(f"[sd] shdn per-layer: bad={bad2} max={max(rels2):.4f}", flush=True)
        fsync_json(OUT, res)
    fsync_json(OUT, res)
    print("[sd] done", flush=True)


if __name__ == "__main__":
    main()
