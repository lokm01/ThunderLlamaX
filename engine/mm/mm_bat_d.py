#!/usr/bin/env python3
"""MM SESSION D battery -- the L4 ship gates (MM_PFW, the wide PF attn).

ARMS (resumable; fsync after every arm)
  1. ce      per-seat next-token CE + top-1, stock (PFG+PFM+PFT) vs new
             (+PFW) on the same 256-tok prose chunk (the Session-C
             discipline: dCE_rel within SEM, top1 ~252/256 class).
  2. fbank   hA relerr stock-vs-new after 1 chunk + first-diverging layer
             (EXPECTED non-zero -- Tier-2 reassociation class).
  3. ladder  full-PF-graph chunk replays min-of-8 at L in {2k, 8k, 16k,
             49k, 96k}: stock vs new + projected feed tok/s.
Output: engine0/mm/mm_bat_d.json
"""
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
S = 8
OUT = os.path.join(BASE, "engine0", "mm",
                   os.getenv("MM_BAT_OUT", "mm_bat_d.json"))

from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_f1b_c import eager_top1
from mm_a_graph import mkgraph_unc as mkgraph


def relerr(a, b):
    import numpy as np
    d = np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel())
    n = np.linalg.norm(b.astype(np.float64).ravel())
    return float(d / max(n, 1e-30))


def build_pf(rig, pfw):
    """The session-D combined arm: PFW (wide attn) + PFK (k=2048 mma ports)
    on top of the session-C baseline (PFG+PFM+PFT)."""
    os.environ["MM_PFG"] = "1"; os.environ["MM_PFM"] = "1"; os.environ["MM_PFT"] = "1"
    os.environ["MM_PFW"] = "1" if pfw else "0"
    os.environ["MM_PFK"] = ("1" if pfw else "0") if os.getenv("MM_BAT_PFK", "1") == "1" else "0"
    from MM_P7_lib import build_seq7
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=f"s{CTXK}", S=S, pf=True)
    return mkgraph(rig, seq, f"bd_{'new' if pfw else 'stock'}")


def main():
    import numpy as np
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    from MM_P7_lib import Rig7
    print("[bd] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    assert "spkqw4_98304" in rig.K, "MM_PFW wiring missing (run mm_wire_d.py)"
    ids = np.ascontiguousarray(np.array(load_prose_ids(1024), dtype=np.int32))
    targets = ids[1:]

    # ---- arm 1: the CE gate ----
    ce = res.setdefault("ce", {})
    for tag, pfw in (("stock", False), ("new", True)):
        if tag in ce and "mean_ce" in ce[tag]:
            continue
        gr = build_pf(rig, pfw)
        rig.reset_states(1024)
        rig.pf_ids_view[:] = memoryview(ids[:PF].data)
        rig.pos_view[0] = 0
        gr.step()
        ces, tops = [], []
        for seat in range(PF):
            _, lg = eager_top1(rig, seat)
            lg64 = lg.astype(np.float64)
            mx = lg64.max()
            lse = mx + np.log(np.exp(lg64 - mx).sum())
            ces.append(lse - lg64[int(targets[seat])])
            tops.append(int(np.argmax(lg64)))
        ce[tag] = {"mean_ce": float(np.mean(ces)), "ce_sem": float(np.std(ces) / np.sqrt(len(ces)))}
        ce[tag + "_tops"] = tops
        print(f"[bd] ce/{tag}: mean_ce={ce[tag]['mean_ce']:.4f} (+/-{ce[tag]['ce_sem']:.4f})", flush=True)
        gr.fence()
        res["ce"] = ce; fsync_json(OUT, res)
    if "dce_rel" not in ce:
        ce["dce_rel"] = (ce["new"]["mean_ce"] - ce["stock"]["mean_ce"]) / ce["stock"]["mean_ce"]
        ce["top1_agree"] = int(sum(1 for a, b in zip(ce["stock_tops"], ce["new_tops"]) if a == b))
        ce["top1_agree_excl_last"] = int(sum(1 for a, b in zip(ce["stock_tops"][:255], ce["new_tops"][:255]) if a == b))
        del ce["stock_tops"]; del ce["new_tops"]
        print(f"[bd] dCE_rel={ce['dce_rel']:+.4%} top1_agree={ce['top1_agree']}/256", flush=True)
        res["ce"] = ce; fsync_json(OUT, res)

    # ---- arm 2: the F bank (hA drift after 1 chunk) ----
    if "fbank" not in res:
        hs = {}
        for tag, pfw in (("stock", False), ("new", True)):
            gr = build_pf(rig, pfw)
            rig.reset_states(1024)
            rig.pf_ids_view[:] = memoryview(ids[:PF].data)
            rig.pos_view[0] = 0
            gr.step(); gr.fence()
            hs[tag] = rig.dn(rig.PFB["hA"], (PF * 2048,)).copy()
        res["fbank"] = {"hA_relerr": relerr(hs["new"], hs["stock"]),
                        "hA_maxabs": float(np.abs(hs["new"] - hs["stock"]).max())}
        print(f"[bd] fbank: hA relerr {res['fbank']['hA_relerr']:.3e} "
              f"maxabs {res['fbank']['hA_maxabs']:.3e}", flush=True)
        fsync_json(OUT, res)

    # ---- arm 3: the ladder (full PF graph, min-of-8 per L) ----
    lad = res.setdefault("ladder", {})
    graphs = {"stock": build_pf(rig, False), "new": build_pf(rig, True)}
    for L in (2048, 8192, 16384, 49152, 98304):
        leg = lad.get(str(L), {})
        if "stock_ms" in leg and "new_ms" in leg:
            continue
        try:
            rig.pos_view[0] = L - PF
            for tag in ("stock", "new"):
                ts = []
                for i in range(9):
                    t0 = time.perf_counter()
                    graphs[tag].step()
                    if i: ts.append((time.perf_counter() - t0) * 1e3)
                graphs[tag].fence()
                leg[f"{tag}_ms"] = round(min(ts), 3)
            leg["speedup"] = round(leg["stock_ms"] / leg["new_ms"], 3)
            leg["feed_tps"] = round(256 / (leg["new_ms"] / 1e3), 1)
            leg["feed_tps_stock"] = round(256 / (leg["stock_ms"] / 1e3), 1)
            lad[str(L)] = leg
            print(f"[bd] L={L}: stock {leg['stock_ms']}ms new {leg['new_ms']}ms "
                  f"x{leg['speedup']} -> feed {leg['feed_tps_stock']} -> {leg['feed_tps']} tok/s", flush=True)
            res["ladder"] = lad; fsync_json(OUT, res)
        except Exception as e:
            leg["error"] = str(e)[:200]
            lad[str(L)] = leg
            res["ladder"] = lad; fsync_json(OUT, res)
            print(f"[bd] L={L} FAILED: {str(e)[:200]}", flush=True)
            break
    print("[bd] done", flush=True)


if __name__ == "__main__":
    main()
