#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION D / L4 POC: the row-grouped wide PF
attention (spkqw) vs the stock spkq256s.

ARMS (resumable -- results fsynced after every arm; arms skip when present)
  1. build+audit spkqw_98304 (zero-spill law; QMD dyn-smem patch).
  2. anchor agreement @ L=2048 REAL data: 7 stock chunks + 1 truncated
     chunk (seq up to the FIRST spka256m of layer 3) -> qgb holds layer-3
     q, KV[0] rows 0..2047 real. Eager stock pair (spkq256s+spkc NP=64)
     vs eager wide pair (spkqw+spkc NP=8): wide-vs-anchor maxdev (the P7
     S1-gate precedent: the anchor is a NEAR-model gated by maxdev, not
     bits), stock-vs-stock-anchor maxdev (harness sanity), det x2,
     sentinel, wide-vs-stock relerr (the Tier-2 reassociation signature).
  3. timing @ L in {2k, 8k, 16k, 48k, 96k}: isolated MGUnc pairs, min-of-8
     + warm (the first-clean-run law); fresh graphs + fences per L (the
     ~950-cycle dext budget law). GO >= 1.5x at 16k.
Output: engine0/mm/mm_l4_poc.json  (fsync after every arm -- /tmp dies)
"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
os.environ["MM_PFG"] = "0"; os.environ["MM_PFM"] = "0"; os.environ["MM_PFT"] = "0"

OUT = os.path.join(BASE, "engine0", "mm", "mm_l4_poc.json")
S = 8
CTXK = 98304
PF = 256
from mm_l1_poc import fsync_json
from mm_f1b_b import load_prose_ids
from mm_a_graph import mkgraph_unc


def spkqw_ref(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos, h, S):
    """Kernel-order mirror of spkqw+spkc256(NP=S): ONE online chain per
    split (positions asc), combine i=s asc (strict max, w=expf(m-m*)),
    y=(acc/Z)*sigmoid(gate). Dot/preamble VERBATIM spkq_h_split_ref."""
    import numpy as np
    import MM_P34_ports as P34
    _f, F32, SCA = P34._f, P34.F32, P34.SCA
    q = _f(qg[h * 512:h * 512 + 256]); gate = _f(qg[h * 512 + 256:h * 512 + 512])
    qn = P34.rmszc_ref(q, qw)
    qr = P34.rope_apply(qn[None, :], cos[None, :], sin[None, :])[0]
    L = pos + 1
    kd = _f(Kq[:L].astype(np.float32) * np.repeat(Ks[:L], 128, axis=1))
    lanes = np.arange(32)
    lp = np.zeros((L, 32), dtype=np.float32)
    for j in range(8):
        lp = _f(lp + _f(qr[lanes * 8 + j][None, :] * kd[:, lanes * 8 + j]))
    for o in (16, 8, 4, 2, 1):
        lp = _f(lp + lp[:, np.arange(32) ^ o])
    scores = _f(lp[:, 0] * SCA)
    vd = _f(Vq[:L].astype(np.float32) * np.repeat(Vs[:L], 128, axis=1))
    ms = np.full(S, -3.402823466e38, dtype=np.float32)
    zs = np.zeros(S, dtype=np.float32)
    outs = np.zeros((S, 256), dtype=np.float32)
    for s in range(S):
        begin = s * L // S; end = (s + 1) * L // S
        m = F32(-3.402823466e38); z = F32(0.0); acc = np.zeros(256, dtype=np.float32)
        for p in range(begin, end):
            mn = F32(max(m, scores[p]))
            r = np.exp(_f(m - mn), dtype=np.float32)
            e = np.exp(_f(scores[p] - mn), dtype=np.float32)
            z = _f(_f(z * r) + e)
            acc = _f(acc * r + _f(e * vd[p]))
            m = mn
        ms[s] = m; zs[s] = z; outs[s] = acc
    mstar = ms[0]
    for i in range(1, S):
        if ms[i] > mstar: mstar = ms[i]
    Z = F32(0.0); acc = np.zeros(256, dtype=np.float32)
    for i in range(S):
        wv = np.exp(_f(ms[i] - mstar), dtype=np.float32)
        Z = _f(Z + _f(wv * zs[i]))
        acc = _f(acc + _f(wv * outs[i]))
    sg = _f(_f(1.0) / (_f(1.0) + np.exp(-gate, dtype=np.float32)))
    return _f(_f(acc / Z) * sg)


def graph_time_pair(rig, seq, tag, reps=8, fence_every=48):
    m = mkgraph_unc(rig, seq, tag, fence_every=fence_every)
    ts = []
    for i in range(reps + 1):
        t0 = time.perf_counter()
        m.step()                             # proper submit + timeline wait
        if i: ts.append((time.perf_counter() - t0) * 1e3)
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
    DYN_B = 64 * (256 + 8 + 256 + 8)   # TILE*528 -- must match MM_D_spkqw.cu

    print("[l4] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=False)
    dev = rig.dev

    # ---- arm 1: load the RW variants + QMD dyn-smem patches ----
    VARIANTS = {}   # rw -> (prg, dyn_b)
    for rw in (4, 8):
        cb = f"{BASE}/MM_D_spkqw{rw}_98304.cubin"
        lib = open(cb, "rb").read()
        nm = f"spkqw{rw}_98304"
        prg = NVProgram(dev, TinyELF(lib=lib, name=nm,
                                     target=dev.renderer.target, signature=(rig.INT_SIG,)))
        qq_b = rw * 8 * 256 * 4
        kv_b = 64 * 528
        dyn = max(qq_b, kv_b)
        smem_total = (prg.shmem_usage + 127) // 128 * 128 + dyn
        prg.qmd.write(shared_memory_size=smem_total,
                      min_sm_config_shared_mem_size=100 * 1024 // 4096 + 1,
                      target_sm_config_shared_mem_size=100 * 1024 // 4096 + 1)
        LSZ[nm] = (256, 1, 1)
        rig.K[nm] = prg
        VARIANTS[rw] = (prg, dyn)
        res.setdefault("load", {})[f"rw{rw}"] = {
            "shmem_usage_static": prg.shmem_usage, "dyn_b": dyn,
            "qmd_smem_total": smem_total, "regs": prg.regs_usage,
            "stack": prg.stack_usage}
        print(f"[l4] {nm} loaded: static={prg.shmem_usage}B dyn={dyn}B "
              f"regs={prg.regs_usage} stack={prg.stack_usage}", flush=True)
    fsync_json(OUT, res)

    # ---- arm 2: anchor agreement on real data @ L=2048 ----
    from MM_P34_ports import ATTN_LAYERS
    from MM_P7_lib import spkq_h_split_ref    # the STOCK anchor (harness sanity)
    ai0 = 0                                   # the first attn layer (L=3)
    if "rw4" not in res or "rw8" not in res:
        ids = np.ascontiguousarray(np.array(load_prose_ids(4096), dtype=np.int32))
        seq_full = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                              spk=f"s{CTXK}", S=S, pf=True)
        gr_full = mkgraph_unc(rig, seq_full, "l4_full")
        # truncated: everything up to and including the FIRST spka256m (layer 3)
        idx_spka = [i for i, e in enumerate(seq_full) if e[0].startswith("spka256m")][0]
        seq_trunc = seq_full[:idx_spka + 1]
        gr_trunc = mkgraph_unc(rig, seq_trunc, "l4_trunc")
        print(f"[l4] truncated seq: {len(seq_trunc)} entries (ends at {seq_trunc[-1][0]})", flush=True)

        rig.reset_states(2048)
        for c in range(7):                   # chunks 0..6 -> KV rows 0..1791
            rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(ids[c * PF:(c + 1) * PF], dtype=np.int32).data)
            rig.pos_view[0] = c * PF
            gr_full.step()
        rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(ids[7 * PF:8 * PF], dtype=np.int32).data)
        rig.pos_view[0] = 7 * PF
        gr_trunc.step()                      # qgb = layer-3 q; KV[0] rows 0..2047
        gr_full.fence(); gr_trunc.fence()    # the ~950-cycle dext budget reset

        L0 = 2048
        qgb = rig.dn(rig.PFB["qgb"], (PF * 8192,)).copy()
        KVQ0 = rig.dn(rig.KVQ[ai0], (2 * L0 * 256,), np.int8).reshape(2, L0, 256).copy()
        KVS0 = rig.dn(rig.KVS[ai0], (2 * L0 * 2,)).reshape(2, L0, 2).copy()
        VVQ0 = rig.dn(rig.VVQ[ai0], (2 * L0 * 256,), np.int8).reshape(2, L0, 256).copy()
        VVS0 = rig.dn(rig.VVS[ai0], (2 * L0 * 2,)).reshape(2, L0, 2).copy()
        qw = rig.dn(rig.W[ATTN_LAYERS[0]]["qw"], (256,)).copy()
        CS = rig.dn(rig.COSB, (CTXK * 32,)).reshape(CTXK, 32).copy()
        SN = rig.dn(rig.SINB, (CTXK * 32,)).reshape(CTXK, 32).copy()
        ptbl = rig.SPTB[ai0]
        scr = rig.SCR_PF
        SENT = np.full(16 * PF * 8 * 8 * 258, 0x7fbfffbf, dtype=np.uint32)
        used_words = 16 * PF * 8 * 258       # NP=S layout: 1/8 of the buffer

        def run_pair(rw):                 # rw=None -> stock
            rig.dev.allocator._copyin(scr, memoryview(SENT[:used_words].tobytes()))
            ay = rig.PFB["ayb"]
            rig.dev.allocator._copyin(ay, memoryview(np.full(PF * 4096, 0x7fbfffbf, dtype=np.uint32).tobytes()))
            if rw:
                VARIANTS[rw][0](rig.PFB["qgb"], scr, ptbl, global_size=(PF // rw, 2 * S, 1),
                                local_size=(256, 1, 1), vals=(S,), wait=True)
                rig.K["spkc256"](rig.PFB["qgb"], ay, scr, global_size=(16 * PF, 1, 1),
                                 local_size=(256, 1, 1), vals=(S,), wait=True)
            else:
                rig.K[f"spkq256s_{CTXK}"](rig.PFB["qgb"], scr, ptbl,
                                          global_size=(16 * PF, S, 1), local_size=(256, 1, 1),
                                          vals=(S,), wait=True)
                rig.K["spkc256"](rig.PFB["qgb"], ay, scr, global_size=(16 * PF, 1, 1),
                                 local_size=(256, 1, 1), vals=(8 * S,), wait=True)
            ayv = rig.dn(ay, (PF * 4096,), np.uint32).copy()
            scv = rig.dn(scr, (used_words,), np.uint32).copy()
            return ayv, scv

        pos_in = 1792
        ayA, _ = run_pair(None)             # stock
        # harness sanity: STOCK kernel vs STOCK anchor -- maxdev class (the
        # P7 S1-gate precedent: the anchor is a near-model, maxdev-gated)
        ayAf = ayA.view(np.float32).reshape(PF, 4096)
        devs0 = []
        for t in (0, 1, 255):
            pos = pos_in + t
            for h in range(0, 16, 4):
                j_ = h >> 3
                ref0 = spkq_h_split_ref(qgb[t * 8192:t * 8192 + 8192], qw,
                                        KVQ0[j_, :pos + 1], KVS0[j_, :pos + 1],
                                        VVQ0[j_, :pos + 1], VVS0[j_, :pos + 1],
                                        CS[pos], SN[pos], pos, h, S)
                devs0.append(float(np.abs(ref0 - ayAf[t, h * 256:(h + 1) * 256]).max()))
        res["stock_vs_stock_anchor"] = {"checked": len(devs0), "maxdev": max(devs0)}
        print(f"[l4] HARNESS SANITY stock-vs-stock-anchor maxdev {max(devs0):.2e} "
              f"({len(devs0)} rows)", flush=True)
        fsync_json(OUT, res)
        for rw in (4, 8):
            ayB, scB1 = run_pair(rw)        # wide run 1
            ayB2, scB2 = run_pair(rw)       # wide run 2 (det x2)
            tag = f"rw{rw}"
            res.setdefault(tag, {})["det_x2"] = bool(np.array_equal(ayB, ayB2) and
                                                     np.array_equal(scB1, scB2))
            res[tag]["sentinel_survived"] = int((scB1 == 0x7fbfffbf).sum())
            res[tag]["vs_stock"] = {
                "exact_words": int((ayA == ayB).sum()), "total": int(ayA.size),
                "relerr": float(np.linalg.norm((ayB.view(np.float32).astype(np.float64) -
                                                ayA.view(np.float32).astype(np.float64))) /
                                max(np.linalg.norm(ayA.view(np.float32).astype(np.float64)), 1e-30))}
            # anchor on sampled seats x all 16 heads (maxdev class)
            ayBf = ayB.view(np.float32).reshape(PF, 4096)
            devs = []
            for t in (0, 1, 63, 128, 200, 255):
                pos = pos_in + t
                for h in range(16):
                    j_ = h >> 3
                    ref = spkqw_ref(qgb[t * 8192:t * 8192 + 8192], qw,
                                    KVQ0[j_, :pos + 1], KVS0[j_, :pos + 1],
                                    VVQ0[j_, :pos + 1], VVS0[j_, :pos + 1],
                                    CS[pos], SN[pos], pos, h, S)
                    devs.append(float(np.abs(ref - ayBf[t, h * 256:(h + 1) * 256]).max()))
            res[tag]["anchor_maxdev"] = max(devs)
            print(f"[l4] rw{rw}: anchor maxdev {max(devs):.2e} ({len(devs)} rows); "
                  f"det_x2={res[tag]['det_x2']} sentinel={res[tag]['sentinel_survived']}; "
                  f"vs_stock exact {res[tag]['vs_stock']['exact_words']}/{res[tag]['vs_stock']['total']} "
                  f"relerr {res[tag]['vs_stock']['relerr']:.3e}", flush=True)
            fsync_json(OUT, res)

    # ---- arm 3: timing at the ladder lengths (fresh fenced graphs per L) ----
    qgbB = rig.PFB["qgb"]; aybB = rig.PFB["ayb"]
    scr = rig.SCR_PF; ptbl = rig.SPTB[0]
    tim = res.get("timing", {})
    for L in (2048, 8192, 16384, 49152, 98304):
        leg = tim.get(str(L), {})
        if "stock_ms" in leg and "rw4_ms" in leg and "rw8_ms" in leg:
            continue                          # resume: only completed legs
        try:
            rig.pos_view[0] = L - PF
            seqS = [(f"spkq256s_{CTXK}", (qgbB, scr, ptbl), (16 * PF, S), (S,)),
                    ("spkc256", (qgbB, aybB, scr), (16 * PF,), (8 * S,))]
            leg["stock_ms"] = round(graph_time_pair(rig, seqS, f"l4_ts{L}"), 3)
            for rw in (4, 8):
                seqW = [(f"spkqw{rw}_98304", (qgbB, scr, ptbl), (PF // rw, 2 * S), (S,)),
                        ("spkc256", (qgbB, aybB, scr), (16 * PF,), (S,))]
                leg[f"rw{rw}_ms"] = round(graph_time_pair(rig, seqW, f"l4_tw{rw}_{L}"), 3)
                leg[f"rw{rw}_x"] = round(leg["stock_ms"] / leg[f"rw{rw}_ms"], 2)
            gb_stock = 16 * PF * S * (L // S) * 512 / 1e9      # load traffic/launch
            leg["stock_gbs"] = round(gb_stock / (leg["stock_ms"] / 1e3))
            tim[str(L)] = leg
            print(f"[l4] L={L}: stock {leg['stock_ms']}ms rw4 {leg['rw4_ms']}ms "
                  f"x{leg['rw4_x']} | rw8 {leg['rw8_ms']}ms x{leg['rw8_x']} "
                  f"(stock {leg['stock_gbs']} GB/s eff)", flush=True)
            res["timing"] = tim
            fsync_json(OUT, res)
        except Exception as e:
            leg["error"] = str(e)[:200]
            tim[str(L)] = leg
            res["timing"] = tim
            fsync_json(OUT, res)
            print(f"[l4] L={L} FAILED: {str(e)[:200]}", flush=True)
            break                             # a wedge kills the GPU; stop
    print("[l4] done", flush=True)


if __name__ == "__main__":
    main()
