#!/usr/bin/env python3
"""MM SESSION E / item-2 POC: the shared-router family.

ARMS (resumable, fsync)
 1. family table: per-kernel in-graph isolated (fresh fenced graphs,
    min-of-8, production launch shapes) at 2k: rt8e256, mmsort8, shgu32,
    shdn32, cmbz2048, gxm_dnf (the dn-side assessment data) -- ms/launch
    and ms/chunk (x40/x80 launches).
 2. router M-batch: eager rt8e256 (ref) vs rt8e_m2/rt8e_m4 on live hnb ->
    eids/gates/sg BIT-EQUAL checks (the gold-router contract).
 3. HONEST in-graph: full PF chunk (D-config + MM_PFU) stock-router vs
    rt8e_m4 vs rt8e_m2 -> chunk ms + tps.
Output: engine0/mm/mm_e2_poc.json"""
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
OUT = os.path.join(BASE, "engine0", "mm", "mm_e2_poc.json")

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
    print("[e2] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev
    for seats in (2, 4):
        nm = f"rt8e_m{seats}"
        lib = open(f"{BASE}/MM_E_{nm}.cubin", "rb").read()
        prg = NVProgram(dev, TinyELF(lib=lib, name=nm,
                                     target=dev.renderer.target,
                                     signature=(rig.INT_SIG,)))
        LSZ[nm] = (1024, 1, 1)
        rig.K[nm] = prg
        print(f"[e2] {nm} loaded regs={prg.regs_usage} smem={prg.shmem_usage}",
              flush=True)

    for k in ("MM_PFG", "MM_PFM", "MM_PFT", "MM_PFW", "MM_PFK", "MM_PFU"):
        os.environ[k] = "1"
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    seqd = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                      spk=f"s{CTXK}", S=S, pf=True)
    grpf = mkgraph(rig, seqd, "e2_pf")
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids[:PF].data)
    rig.pos_view[0] = 0
    grpf.step(); grpf.fence()

    hnb = rig.PFB["hnb"]
    w0 = rig.W[0]

    # ---- arm 1: the family table ----
    if "family" not in res:
        launches = {
            "rt8e256": (("rt8e256", (w0["rt"], w0["wsh"], hnb, rig.PFB["eidsb"],
                                      rig.PFB["gatesb"], rig.PFB["sgb"]), (PF,), ()),
                        (1024, 1, 1), 40),
            "mmsort8": (("mmsort8", (rig.PFB["eidsb"], rig.EOFFB, rig.PLISTB,
                                     rig.ITEMSB, rig.NITB), 1, (PF * 8,)),
                        (1024, 1, 1), 40),
            "shgu32": (("shgu32", (w0["sg"], w0["su"], hnb, rig.ACTSHB),
                        512 // 8, (PF,)), (256, 1, 1), 40),
            "shdn32": (("shdn32", (w0["sd"], rig.ACTSHB, rig.PFB["shb"]),
                        2048 // 8, (PF,)), (256, 1, 1), 40),
            "cmbz2048": (("cmbz2048", (rig.PFB["partsb"], rig.PFB["gatesb"],
                                       rig.PFB["sgb"], rig.PFB["shb"],
                                       rig.PFB["hB"], rig.PFB["hA"]), (PF,), ()),
                         (256, 1, 1), 40),
            "gxm_dnf": (("gxm_dnf", (rig.PTB_DN[0], rig.ITEMSB, rig.NITB,
                                     rig.EOFFB, rig.PLISTB, rig.PFB["actb"],
                                     rig.gridf, rig.PFB["partsb"]), 1024, ()),
                        (256, 1, 1), 37),
            "gxu_gm": (("gxu_gm", (rig.PTB_UP[0], rig.EOFFB, rig.PLISTB,
                                   hnb, rig.gridf, rig.PFB["actb"]), 2048, ()),
                       (256, 1, 1), 39),
        }
        fam = {}
        for nm, (ent, lsz, count) in launches.items():
            t = graph_time(rig, [ent], f"e2_f_{nm}")
            fam[nm] = {"ms": round(t, 4), "launches": count,
                       "chunk_ms": round(t * count, 2)}
            print(f"[e2] family {nm}: {t*1000:.1f}us x{count} = {t*count:.2f}ms/chunk",
                  flush=True)
        res["family"] = fam
        fsync_json(OUT, res)

    # ---- arm 2: router bit-exactness ----
    if "router_bitexact" not in res:
        out = {}
        rig.K["rt8e256"](w0["rt"], w0["wsh"], hnb, rig.PFB["eidsb"],
                         rig.PFB["gatesb"], rig.PFB["sgb"], global_size=(PF, 1, 1),
                         local_size=(1024, 1, 1), vals=(), wait=True)
        ref = {"eids": rig.dn(rig.PFB["eidsb"], (PF * 8,), np.uint16).copy(),
               "gates": rig.dn(rig.PFB["gatesb"], (PF * 8,)).copy(),
               "sg": rig.dn(rig.PFB["sgb"], (PF,)).copy()}
        for seats in (2, 4):
            nm = f"rt8e_m{seats}"
            rig.dev.allocator._copyin(rig.PFB["eidsb"], memoryview(np.zeros(PF * 8, np.uint16).tobytes()))
            rig.K[nm](w0["rt"], w0["wsh"], hnb, rig.PFB["eidsb"],
                      rig.PFB["gatesb"], rig.PFB["sgb"],
                      global_size=(PF // seats, 1, 1), local_size=(1024, 1, 1),
                      vals=(PF,), wait=True)
            got = {"eids": rig.dn(rig.PFB["eidsb"], (PF * 8,), np.uint16).copy(),
                   "gates": rig.dn(rig.PFB["gatesb"], (PF * 8,)).copy(),
                   "sg": rig.dn(rig.PFB["sgb"], (PF,)).copy()}
            out[nm] = {k: bool(np.array_equal(ref[k], got[k])) for k in ref}
            print(f"[e2] {nm} bit-exact: {out[nm]}", flush=True)
            # det x2
            rig.K[nm](w0["rt"], w0["wsh"], hnb, rig.PFB["eidsb"],
                      rig.PFB["gatesb"], rig.PFB["sgb"],
                      global_size=(PF // seats, 1, 1), local_size=(1024, 1, 1),
                      vals=(PF,), wait=True)
            got2 = {"eids": rig.dn(rig.PFB["eidsb"], (PF * 8,), np.uint16).copy(),
                    "gates": rig.dn(rig.PFB["gatesb"], (PF * 8,)).copy(),
                    "sg": rig.dn(rig.PFB["sgb"], (PF,)).copy()}
            out[nm]["det_x2"] = bool(all(np.array_equal(got[k], got2[k]) for k in got))
        res["router_bitexact"] = out
        fsync_json(OUT, res)

    # ---- arm 3: HONEST in-graph chunk ----
    if "inchunk" not in res:
        def swap_rt(seq, seats):
            out = []
            for name, bufs, g, v in seq:
                if name == "rt8e256":
                    nm = f"rt8e_m{seats}"
                    out.append((nm, bufs, PF // seats, (PF,)))
                else:
                    out.append((name, bufs, g, v))
            return out

        def chunk_time(seq, tag):
            g = mkgraph(rig, seq, tag)
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
            return min(ts)

        row = {"stock": round(chunk_time(seqd, "e2_c_s"), 2)}
        for seats in (4, 2):
            row[f"m{seats}"] = round(chunk_time(swap_rt(seqd, seats), f"e2_c_m{seats}"), 2)
        row["stock_tps"] = round(PF / (row["stock"] / 1e3), 1)
        for seats in (4, 2):
            row[f"m{seats}_tps"] = round(PF / (row[f"m{seats}"] / 1e3), 1)
        res["inchunk"] = row
        print(f"[e2] inchunk: {row}", flush=True)
        fsync_json(OUT, res)
    print("[e2] done", flush=True)


if __name__ == "__main__":
    main()
