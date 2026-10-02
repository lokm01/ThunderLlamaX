#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION B / F1b + the ladder + PF64 tail
gates. ONE heavy process.

ARMS
  1. f1b_stock_new  the stock PF-256 graph vs the Session-B PF graph
                    (MM_PFG=1 MM_PFM=1, set in-process BEFORE each
                    build_seq7): same 256-token chunk -> PFB hA bit-exact
                    PLUS per-layer router eids bit-exact (cp4k captures on
                    both seqs -- routing equality is the gold-router
                    contract through the new path).
  2. f1b_chunk_t1   the Session-B PF chunk vs the per-token T1 chain
                    (gr1, 256 steps): per-seat hA bit-exact (the original
                    F1b contract).
  3. pf64           the PF-64 tail graph: 64-chunk at pos 0 vs T1 chain
                    (seats 0..63) + the MIXED feed (256-chunk then 64-chunk
                    at pos 256 vs T1 continuing) -- the in-plan shape.
  4. ladder         the stock PF graph vs the Session-B graph at
                    L in {2k, 8k, 16k, 96k}: min-of-8 chunk replays ->
                    tok/s per chunk (the honest before/after).
Output: engine0/mm/mm_f1b_b.json
NOTE: NV_SMEM_CFG_AUTO=1 + AUTO_NAMES (gxm,gvs32,shgu,shdn) MUST be set in
the process env BEFORE boot (the QMD carveout bakes at program load; the
stock kernels match no name and keep the default -- a clean in-process A/B).
"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
# THE L9 CARVEOUT (co64 measured 2.0-3.6x on up): bake at program load.
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256
OUT = os.path.join(BASE, "engine0", "mm", "mm_f1b_b.json")
S = 8

from mm_l1_poc import load_prog, fsync_json
from mm_a_graph import mkgraph_unc as mkgraph


def load_prose_ids(n=PF):
    d = json.load(open(os.path.join(BASE, "eval", "data", "ppl_prose_ids.json")))
    out = []
    while len(out) < n:
        out += [int(t) for t in d]
    return out[:n]


def build_pf(rig, pfg, pfm, with_cap=False):
    os.environ["MM_PFG"] = "1" if pfg else "0"
    os.environ["MM_PFM"] = "1" if pfm else "0"
    from MM_P7_lib import build_seq7
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=f"s{CTXK}", S=S, pf=True)
    if with_cap:
        if "cp4k" not in rig.K:
            load_prog(rig, "MM_A_cp4k.cubin", "cp4k", 256)
        seq2 = []; li = 0
        cap = rig.alloc(40 * PF * 8 * 2)
        for ent in seq:
            seq2.append(ent)
            if ent[0] == "rt8e256":
                dst = cap.offset(offset=li * PF * 8 * 2, size=PF * 8 * 2)
                seq2.append(("cp4k", (rig.PFB["eidsb"], dst), 8, (PF * 8,)))
                li += 1
        assert li == 40
        seq = seq2
    return mkgraph(rig, seq, f"f1b_{'new' if pfg else 'stock'}{'_c' if with_cap else ''}"), seq


def main():
    import numpy as np
    res = {}
    if os.path.exists(OUT):
        try: res = json.load(open(OUT))
        except Exception: res = {}
    from MM_P7_lib import Rig7, build_seq7
    print("[f1b] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    ids = np.ascontiguousarray(np.array(load_prose_ids(), dtype=np.int32))
    dev = rig.dev

    # ---------- arm 1: stock vs new (hA + eids bit-exact) ----------
    a1 = res.setdefault("f1b_stock_new", {})
    if "hA_bit_exact" not in a1:
        gr_s, _ = build_pf(rig, False, False)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data)
        rig.pos_view[0] = 0
        gr_s.step()
        hA_s = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        gr_n, _ = build_pf(rig, True, True)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data)
        rig.pos_view[0] = 0
        gr_n.step()
        hA_n = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        be = bool((hA_s.view(np.uint32) == hA_n.view(np.uint32)).all())
        nz = int((hA_s.view(np.uint32) != hA_n.view(np.uint32)).sum())
        a1.update({"hA_bit_exact": be, "mismatched_words": nz})
        print(f"[f1b] stock-vs-new hA bit_exact={be} mismatched={nz}", flush=True)
        fsync_json(OUT, res)

    # eids equality through the new path (captures on both)
    if "eids_bit_exact" not in a1:
        gr_sc, _ = build_pf(rig, False, False, with_cap=True)
        cap_s = rig.alloc(40 * PF * 8 * 2)
        # rebuild with cap pointing at cap_s: simpler -- rerun the cap build
        gr_sc, seq_sc = build_pf(rig, False, False, with_cap=True)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_sc.step()
        # the cap buffer lives inside build_pf's local -- rebind via seq walk
        cap_s = None
        for n, b, g, v in seq_sc:
            if n == "cp4k":
                cap_s = b[1]; break
        e_s = rig.dn(cap_s, (40 * PF * 8,), np.uint16).copy()
        gr_nc, seq_nc = build_pf(rig, True, True, with_cap=True)
        cap_n = None
        for n, b, g, v in seq_nc:
            if n == "cp4k":
                cap_n = b[1]; break
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_nc.step()
        e_n = rig.dn(cap_n, (40 * PF * 8,), np.uint16).copy()
        bee = bool((e_s == e_n).all())
        a1.update({"eids_bit_exact": bee})
        print(f"[f1b] router eids through the new path bit_exact={bee}", flush=True)
        fsync_json(OUT, res)

    # ---------- arm 2: new chunk vs per-token T1 ----------
    a2 = res.setdefault("f1b_chunk_t1", {})
    if "seats_bit_exact" not in a2:
        seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=False,
                          spk=f"s{CTXK}", S=32)
        gr1 = mkgraph(rig, seq1, "f1b_t1")
        rig.reset_states(1024)
        gr_n, _ = build_pf(rig, True, True)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_n.step()
        hA_pf = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        rig.reset_states(1024)
        t1_h = np.zeros((PF, 2048), dtype=np.float32)
        for p in range(PF):
            rig.feed(int(ids[p]), p)
            gr1.step()
            t1_h[p] = rig.dn(rig.hA, (2048,), np.float32)
        be = bool((hA_pf.view(np.uint32) == t1_h.reshape(-1).view(np.uint32)).all())
        nz = int((hA_pf.view(np.uint32) != t1_h.reshape(-1).view(np.uint32)).sum())
        a2.update({"seats_bit_exact": be, "mismatched_words": nz})
        print(f"[f1b] new-chunk vs T1-chain bit_exact={be} mismatched={nz}", flush=True)
        fsync_json(OUT, res)

    # ---------- arm 3: PF64 tail ----------
    a3 = res.setdefault("pf64", {})
    os.environ["MM_PF64"] = "1"
    if "f64_vs_t1" not in a3 and "gconv36_64" in rig.K:
        os.environ["MM_PFG"] = "1"; os.environ["MM_PFM"] = "1"
        seq64 = build_seq7(rig, 64, "gconv36_64", "k2s36_64", with_head=False,
                           spk=f"s{CTXK}", S=S, pf=True)
        gr64 = mkgraph(rig, seq64, "f1b_pf64")
        ids64 = ids[:64]
        a = rig.alloc(256 * 4)
        rig.reset_states(1024)
        full = np.zeros(256, dtype=np.int32); full[:64] = ids64
        rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(full).data)
        rig.pos_view[0] = 0
        gr64.step()
        h64 = rig.dn(rig.PFB["hA"], (64 * 2048,), np.float32).copy()
        rig.reset_states(1024)
        seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=False,
                          spk=f"s{CTXK}", S=32)
        gr1 = mkgraph(rig, seq1, "f1b_t1b")
        t1_h = np.zeros((64, 2048), dtype=np.float32)
        for p in range(64):
            rig.feed(int(ids64[p]), p)
            gr1.step()
            t1_h[p] = rig.dn(rig.hA, (2048,), np.float32)
        be = bool((h64.view(np.uint32) == t1_h.reshape(-1).view(np.uint32)).all())
        nz = int((h64.view(np.uint32) != t1_h.reshape(-1).view(np.uint32)).sum())
        a3["f64_vs_t1"] = {"bit_exact": be, "mismatched_words": nz}
        print(f"[f1b] PF-64 vs T1(64) bit_exact={be} mismatched={nz}", flush=True)
        fsync_json(OUT, res)
        # mixed: 256-chunk then 64-chunk at pos 256, vs T1 continuing
        gr_n, _ = build_pf(rig, True, True)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_n.step()
        tail64 = np.zeros(256, dtype=np.int32); tail64[:64] = ids[256:320] if len(ids) >= 320 else ids[:64]
        rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(tail64).data)
        rig.pos_view[0] = 256
        gr64.step()
        hmix = rig.dn(rig.PFB["hA"], (64 * 2048,), np.float32).copy()
        rig.reset_states(1024)
        t1_h = np.zeros((64, 2048), dtype=np.float32)
        # replay 320 tokens through T1 (256 + 64)
        ids320 = np.concatenate([ids, ids[:64]]) if len(ids) < 320 else ids[:320]
        for p in range(320):
            rig.feed(int(ids320[p]), p)
            gr1.step()
            if p >= 256:
                t1_h[p - 256] = rig.dn(rig.hA, (2048,), np.float32)
        be = bool((hmix.view(np.uint32) == t1_h.reshape(-1).view(np.uint32)).all())
        nz = int((hmix.view(np.uint32) != t1_h.reshape(-1).view(np.uint32)).sum())
        a3["mixed_256_64_vs_t1"] = {"bit_exact": be, "mismatched_words": nz}
        print(f"[f1b] mixed 256+64 vs T1 bit_exact={be} mismatched={nz}", flush=True)
        fsync_json(OUT, res)

    # ---------- arm 4: the ladder ----------
    a4 = res.setdefault("ladder", {})
    if not a4:
        for tag, pfg, pfm in (("stock", "0", "0"), ("new", "1", "1")):
            os.environ["MM_PFG"] = pfg; os.environ["MM_PFM"] = pfm
            gr, _ = build_pf(rig, pfg == "1", pfm == "1")
            row = a4.setdefault(tag, {})
            for L in (2048, 8192, 16384, 98304):
                ts = []
                for i in range(9):
                    rig.pos_view[0] = L - PF
                    dev.synchronize()
                    t0 = time.perf_counter()
                    gr.step()
                    ts.append((time.perf_counter() - t0) * 1e3)
                mn = min(ts[1:])
                row[str(L)] = {"min_ms": round(mn, 2),
                               "tok_s": round(PF / (mn / 1e3), 1)}
                print(f"[lad] {tag} L={L:6d} {mn:8.2f} ms  {PF/(mn/1e3):7.1f} tok/s", flush=True)
                fsync_json(OUT, res)
    print("[f1b] done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
