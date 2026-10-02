#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION C battery (Tier-2, the dense-M32
precedent): MM_PFT (out/o mma M-GEMM) + gxm_dnf (bit-exact fold).

ARMS
  1. f_bank   stock PF-256 vs Session-C PF-256 (PFG+PFM+PFT): hA relerr F
              + maxabs + router eids first-diverging layer + flip count
              (EXPECTED non-zero -- Tier-2 numerics) + per-seat top1 token
              flips through the head + logits relerr. det x2 (whole graph).
  2. t1_chain new-PF chunk vs per-token T1 chain: relerr (informational --
              the decode T1 keeps stock numerics by design).
  3. pf64     the 64-seat tail on the new path vs T1: relerr class check.
  4. ladder   stock vs new at L in {2k, 8k, 16k, 96k}: min-of-8 chunk
              replays (the honest before/after; the 96k leg = the feed
              class with attention growth).
Output: engine0/mm/mm_f1b_c.json
NOTE: NV_SMEM_CFG_AUTO=1 + AUTO_NAMES (...,pgmq8) in env BEFORE boot."""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_SMEM_CFG_AUTO", "1")
os.environ.setdefault("NV_SMEM_CFG_AUTO_NAMES", "gxm,gvs32,shgu,shdn,pgmq8")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256
OUT = os.path.join(BASE, "engine0", "mm", "mm_f1b_c.json")
S = 8

from mm_l1_poc import load_prog, fsync_json
from mm_a_graph import mkgraph_unc as mkgraph
from mm_f1b_b import load_prose_ids


def relerr(a, b):
    import numpy as np
    d = np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel())
    n = np.linalg.norm(b.astype(np.float64).ravel())
    return float(d / max(n, 1e-30))


def eager_top1(rig, seat):
    """The eager per-seat head (the eager_head_cur pattern: a PFB seat of
    the last PF chunk). Returns (top1, logits_copy)."""
    import numpy as np
    hin = rig.PFB["hA"].offset(offset=int(seat) * 2048 * 4, size=2048 * 4)
    rig.K["rmsz2048g"](hin, rig.ONORM, rig.normhb,
                       global_size=(1, 1, 1), local_size=(256, 1, 1), wait=True)
    rig.K["h6k2048"](rig.HEAD, rig.normhb, rig.logitsb,
                     global_size=(7760, 1, 1), local_size=(1024, 1, 1),
                     vals=(248320,), wait=True)
    lg = rig.dn(rig.logitsb, (248320,), np.float32).copy()
    return int(np.argmax(lg)), lg


def build_pf(rig, pfc, with_cap=False, with_head=False):
    os.environ["MM_PFG"] = "1" if pfc else "0"
    os.environ["MM_PFM"] = "1" if pfc else "0"
    os.environ["MM_PFT"] = "1" if pfc else "0"
    from MM_P7_lib import build_seq7
    # with_head at P=256 is ILLEGAL (logitsb is a single-seat buffer -- the
    # full-chunk head OOBs and device-faults; the daemon computes heads
    # per-seat eagerly). Kept False always.
    assert not with_head
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
    return mkgraph(rig, seq, f"fc_{'new' if pfc else 'stock'}"), seq


def main():
    import numpy as np
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    from MM_P7_lib import Rig7, build_seq7
    print("[fc] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    ids = np.ascontiguousarray(np.array(load_prose_ids(), dtype=np.int32))
    dev = rig.dev

    # ---------- arm 1: the F bank ----------
    a1 = res.setdefault("f_bank", {})
    SEATS = list(range(31, PF, 32)) + [PF - 1]
    if "F_hA" not in a1:
        gr_s, _ = build_pf(rig, False)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_s.step()
        hA_s = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        top_s = [eager_top1(rig, s) for s in SEATS]
        gr_n, _ = build_pf(rig, True)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_n.step()
        hA_n = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        top_n = [eager_top1(rig, s) for s in SEATS]
        # det x2 of the new graph
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_n.step()
        hA_n2 = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        det = bool((hA_n.view(np.uint32) == hA_n2.view(np.uint32)).all())
        i255 = SEATS.index(PF - 1)
        a1.update({
            "F_hA": relerr(hA_n, hA_s), "maxabs_hA": float(np.abs(hA_n - hA_s).max()),
            "F_logits_seat255": relerr(top_n[i255][1], top_s[i255][1]),
            "det_x2_new_graph": det,
            "top1_flips_sampled": int(sum(1 for i in range(len(SEATS))
                                          if top_s[i][0] != top_n[i][0])),
            "n_seats_sampled": len(SEATS),
        })
        print(f"[fc] F bank: {a1}", flush=True)
        fsync_json(OUT, res)

    # router eids first-diverging layer + flips
    if "eids_first_div_layer" not in a1:
        gr_sc, seq_sc = build_pf(rig, False, with_cap=True)
        cap_s = None
        for n, b, g, v in seq_sc:
            if n == "cp4k":
                cap_s = b[1]; break
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_sc.step()
        e_s = rig.dn(cap_s, (40 * PF * 8,), np.uint16).copy().reshape(40, PF, 8)
        gr_nc, seq_nc = build_pf(rig, True, with_cap=True)
        cap_n = None
        for n, b, g, v in seq_nc:
            if n == "cp4k":
                cap_n = b[1]; break
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_nc.step()
        e_n = rig.dn(cap_n, (40 * PF * 8,), np.uint16).copy().reshape(40, PF, 8)
        per_layer = [(e_s[L] != e_n[L]).sum() for L in range(40)]
        first = next((L for L in range(40) if per_layer[L]), None)
        a1.update({"eids_first_div_layer": first,
                   "eids_total_flips": int(sum(per_layer)),
                   "eids_per_layer": [int(x) for x in per_layer]})
        print(f"[fc] eids: first_div_layer={first} total_flips={sum(per_layer)}", flush=True)
        fsync_json(OUT, res)

    # ---------- arm 2: T1-chain relerr (informational) ----------
    a2 = res.setdefault("t1_chain", {})
    if "relerr" not in a2:
        seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=False,
                          spk=f"s{CTXK}", S=32)
        gr1 = mkgraph(rig, seq1, "fc_t1")
        rig.reset_states(1024)
        gr_n, _ = build_pf(rig, True)
        rig.pf_ids_view[:] = memoryview(ids.data); rig.pos_view[0] = 0
        gr_n.step()
        hA_pf = rig.dn(rig.PFB["hA"], (PF * 2048,), np.float32).copy()
        rig.reset_states(1024)
        t1_h = np.zeros((PF, 2048), dtype=np.float32)
        for p in range(PF):
            rig.feed(int(ids[p]), p)
            gr1.step()
            t1_h[p] = rig.dn(rig.hA, (2048,), np.float32)
        a2.update({"relerr": relerr(hA_pf.reshape(PF, 2048), t1_h),
                   "maxabs": float(np.abs(hA_pf.reshape(PF, 2048) - t1_h).max())})
        print(f"[fc] T1-chain relerr: {a2}", flush=True)
        fsync_json(OUT, res)

    # ---------- arm 3: PF64 tail on the new path ----------
    a3 = res.setdefault("pf64", {})
    if "relerr" not in a3:
        from MM_P7_lib import build_seq7 as bs7
        os.environ["MM_PFT"] = "1"
        seq64 = bs7(rig, 64, "gconv36_64", "k2s36_64", with_head=False,
                    spk=f"s{CTXK}", S=S, pf=True)
        gr64 = mkgraph(rig, seq64, "fc_pf64")
        ids64 = np.zeros(256, dtype=np.int32); ids64[:64] = ids[:64]
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(ids64).data)
        rig.pos_view[0] = 0
        gr64.step()
        h64 = rig.dn(rig.PFB["hA"], (64 * 2048,), np.float32).copy()
        rig.reset_states(1024)
        t1_h = np.zeros((64, 2048), dtype=np.float32)
        for p in range(64):
            rig.feed(int(ids64[p]), p)
            gr1.step()
            t1_h[p] = rig.dn(rig.hA, (2048,), np.float32)
        a3.update({"relerr": relerr(h64.reshape(64, 2048), t1_h),
                   "maxabs": float(np.abs(h64.reshape(64, 2048) - t1_h).max())})
        print(f"[fc] PF64 relerr: {a3}", flush=True)
        fsync_json(OUT, res)

    # ---------- arm 4: the ladder ----------
    a4 = res.setdefault("ladder", {})
    if not a4:
        for tag, pfc in (("stock", False), ("new", True)):
            os.environ["MM_PFG"] = "1" if pfc else "0"
            os.environ["MM_PFM"] = "1" if pfc else "0"
            os.environ["MM_PFT"] = "1" if pfc else "0"
            gr, _ = build_pf(rig, pfc)
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
                row[str(L)] = {"min_ms": round(mn, 2), "tok_s": round(PF / (mn / 1e3), 1)}
                print(f"[lad] {tag} L={L:6d} {mn:8.2f} ms  {PF/(mn/1e3):7.1f} tok/s", flush=True)
                fsync_json(OUT, res)
    print("[fc] done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
