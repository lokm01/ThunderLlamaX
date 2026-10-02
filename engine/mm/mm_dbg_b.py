#!/usr/bin/env python3
"""MM SESSION B -- the F1b failure bisect. ONE heavy process.
Arms: stock-vs-stock (harness sanity) / PFM-only / PFG-only vs stock,
each with a cp4k eids capture; hA compare per arm. Plus T1@S=8 vs the stock
chunk (the TRUE F1b -- the earlier harness used S=32 for T1 vs S=8 for PF,
which cannot be bit-exact across split boundaries).
Output: engine0/mm/mm_dbg_b.json"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256
S = 8
OUT = os.path.join(BASE, "engine0", "mm", "mm_dbg_b.json")

from mm_l1_poc import load_prog, fsync_json
from mm_a_graph import mkgraph_unc as mkgraph


def load_prose_ids(n=PF):
    d = json.load(open(os.path.join(BASE, "eval", "data", "ppl_prose_ids.json")))
    out = []
    while len(out) < n:
        out += [int(t) for t in d]
    return out[:n]


def build(rig, pfg, pfm, cap=False):
    os.environ["MM_PFG"] = "1" if pfg else "0"
    os.environ["MM_PFM"] = "1" if pfm else "0"
    from MM_P7_lib import build_seq7
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=f"s{CTXK}", S=S, pf=True)
    capbuf = None
    if cap:
        if "cp4k" not in rig.K:
            load_prog(rig, "MM_A_cp4k.cubin", "cp4k", 256)
        seq2 = []; li = 0
        capbuf = rig.alloc(40 * PF * 8 * 2)
        for ent in seq:
            seq2.append(ent)
            if ent[0] == "rt8e256":
                dst = capbuf.offset(offset=li * PF * 8 * 2, size=PF * 8 * 2)
                seq2.append(("cp4k", (rig.PFB["eidsb"], dst), 8, (PF * 8,)))
                li += 1
        assert li == 40
        seq = seq2
    return mkgraph(rig, seq, f"dbg_{int(pfg)}{int(pfm)}{'c' if cap else ''}"), capbuf


def run_chunk(rig, gr, ids):
    rig.reset_states(1024)
    rig.pf_ids_view[:] = memoryview(ids.data)
    rig.pos_view[0] = 0
    gr.step()


def main():
    import numpy as np
    res = {}
    from MM_P7_lib import Rig7, build_seq7
    print("[dbg] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    ids = np.ascontiguousarray(np.array(load_prose_ids(), dtype=np.int32))

    def hA():
        return rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()

    # sanity: stock twice
    gr0, _ = build(rig, False, False)
    run_chunk(rig, gr0, ids)
    ref = hA()
    gr0b, _ = build(rig, False, False)
    run_chunk(rig, gr0b, ids)
    be = bool((ref.view(np.uint32) == hA().view(np.uint32)).all())
    res["stock_vs_stock"] = {"bit_exact": be}
    print(f"[dbg] stock-vs-stock: {be}", flush=True)
    fsync_json(OUT, res)

    for tag, pfg, pfm in (("pfm_only", False, True), ("pfg_only", True, False), ("both", True, True)):
        if tag in res:
            continue
        gr, _ = build(rig, pfg, pfm)
        run_chunk(rig, gr, ids)
        hh = hA()
        nz = int((ref.view(np.uint32) != hh.view(np.uint32)).sum())
        first = int(np.argmax(ref.view(np.uint32) != hh.view(np.uint32))) if nz else -1
        res[tag] = {"mismatched": nz, "first_word": first,
                    "first_seat": first // 2048 if first >= 0 else -1}
        print(f"[dbg] {tag}: mismatched={nz} first_seat={res[tag]['first_seat']}", flush=True)
        fsync_json(OUT, res)

    # where do eids first diverge (PFG-only)?
    grc_s, cap_s = build(rig, False, False, cap=True)
    run_chunk(rig, grc_s, ids)
    e_s = rig.dn(cap_s, (40 * PF * 8,), np.uint16).copy()
    grc_n, cap_n = build(rig, True, False, cap=True)
    run_chunk(rig, grc_n, ids)
    e_n = rig.dn(cap_n, (40 * PF * 8,), np.uint16).copy()
    bad_layers = [L for L in range(40) if not (e_s[L] == e_n[L]).all()]
    res["pfg_eids_divergent_layers"] = bad_layers[:8]
    res["pfg_eids_n_bad_layers"] = len(bad_layers)
    print(f"[dbg] PFG eids divergent layers: {bad_layers[:8]} ({len(bad_layers)}/40)", flush=True)
    fsync_json(OUT, res)

    # the TRUE F1b: T1 chain at S=8 vs the stock chunk
    seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=False,
                      spk=f"s{CTXK}", S=S)
    gr1 = mkgraph(rig, seq1, "dbg_t1s8")
    run_chunk(rig, gr0, ids)          # stock chunk again (state)
    hA_pf = hA().copy()
    rig.reset_states(1024)
    t1_h = np.zeros((PF, 2048), dtype=np.float32)
    for p in range(PF):
        rig.feed(int(ids[p]), p)
        gr1.step()
        t1_h[p] = rig.dn(rig.hA, (2048,), np.float32)
    nz = int((hA_pf.view(np.uint32) != t1_h.reshape(-1).view(np.uint32)).sum())
    res["f1b_t1s8_vs_stockchunk"] = {"mismatched": nz}
    print(f"[dbg] T1@S=8 vs stock chunk: mismatched={nz}", flush=True)
    fsync_json(OUT, res)
    print("[dbg] done", flush=True)


if __name__ == "__main__":
    main()
