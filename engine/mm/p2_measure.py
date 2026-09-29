#!/usr/bin/env python3
"""P2 measurement: the kernel-count vs cycle-time slope (the launch-serialization
discriminator). Builds T1-variant graphs with progressively fewer layers
(truncated models -- WRONG outputs, VALID timing: same kernels, same buffers)
plus a no-MoE variant (the quartet stripped). If cycle ~= a + b*kernel_count
with b ~= 0.09-0.1 ms, the T1 cycle is launch-floor-bound -> fusion is THE fix.
"""
import os, sys, time
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
os.environ.setdefault("DEV", "NV")

from MM_P7_lib import Rig7, build_seq7
from MM_P9_mtp import GraphRunnerUnc
import MM_P56_lib as L56
from MM_P34_ports import GDN_LAYERS, ATTN_LAYERS

CTXK = int(os.getenv("MM_CTXK", "98304"))
rig = Rig7(ctx_alloc=CTXK, load_p6=True)
spk = f"s{CTXK}"
print("[p2m] rig up", flush=True)

def truncated_seq(nlayers, drop_moe=False, drop_attn=False):
    """build_seq7's structure but only the first nlayers layers; optionally
    strip the MoE quartet (rt/shexp/up/dn/cmb) or the attention block."""
    K = rig.K
    P = 1
    idsb = rig.idsb
    B = {n: getattr(rig, n) for n in ["hA","hB","hnb","qkvb","zb","abb","qkvsb","gyb","qgb","kqb","vqb","ayb","eidsb","gatesb","sgb","actb","partsb","shb","normhb"]}
    rig._P2D = P
    seq = [("embg248", (rig.EMB, idsb, B["hA"]), 1, (P,))]
    for L in range(nlayers):
        w = rig.W[L]
        hin = B["hA"]; hmid = B["hB"]
        seq.append(("rmsz2048g", (hin, w["an"], B["hnb"]), (P,), ()))
        if L in GDN_LAYERS:
            gi = GDN_LAYERS.index(L)
            seq.append(("gv8k2048p", (w["qkv"], B["hnb"], B["qkvb"]), (256,), (8192,)))
            seq.append(("gv8k2048p", (w["z"], B["hnb"], B["zb"]), (128,), (4096,)))
            seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))
            seq.append(("gconv36_1", (w["cw"], B["qkvb"], rig.CSV[gi], B["qkvsb"]), (32,), ()))
            seq.append(("k2s36_1", (B["qkvsb"], B["abb"], w["al"], w["dt"], w["sn"], B["zb"], rig.SV[gi], B["gyb"]), (32,), ()))
            seq.append(("gv8k4096r", (w["out"], B["gyb"], hin, hmid), (64,), (2048,)))
        else:
            ai = ATTN_LAYERS.index(L)
            ptbl = rig.SPTB[ai]
            seq.append(("gv8k2048p", (w["q"], B["hnb"], B["qgb"]), (256,), (8192,)))
            seq.append(("gv8k2048p", (w["k"], B["hnb"], B["kqb"]), (16,), (512,)))
            seq.append(("gv8k2048p", (w["v"], B["hnb"], B["vqb"]), (16,), (512,)))
            if not drop_attn:
                seq.append((f"spka256m_{CTXK}", (B["kqb"], B["vqb"], rig.SPTB[(ai, "m")]), (2*P,), ()))
                seq.append((f"spkq256s_{CTXK}", (B["qgb"], rig.SCR_DEC, ptbl), (16*P, 32), (32,)))
                seq.append(("spkc256", (B["qgb"], B["ayb"], rig.SCR_DEC), (16*P,), (8*32,)))
            seq.append(("gv8k4096r", (w["o"], B["ayb"], hin, hmid), (64,), (2048,)))
        seq.append(("rmsz2048g", (hmid, w["pn"], B["hnb"]), (P,), ()))
        if not drop_moe:
            seq.append(("rt8e256", (w["rt"], w["wsh"], B["hnb"], B["eidsb"], B["gatesb"], B["sgb"]), (P,), ()))
            seq.append(("shexp8", (w["sg"], w["su"], w["sd"], B["hnb"], B["shb"]), (P,), ()))
            upk = "gx8e256up4" if rig.man["routed"][L]["types"]["gate"] == "IQ4_XS" else "gx8e256up"
            upbufs = (rig.PTB_UP[L], B["eidsb"], B["hnb"], rig.iq4nl, B["actb"]) if upk == "gx8e256up4" else (rig.PTB_UP[L], B["eidsb"], B["hnb"], rig.gridf, B["actb"])
            seq.append((upk, upbufs, (P*8,), ()))
            dnk = "gx8e256dn6" if rig.man["routed"][L]["types"]["down"] == "Q6_K" else "gx8e256dn"
            dnbufs = (rig.PTB_DN[L], B["eidsb"], B["actb"], B["partsb"]) if dnk == "gx8e256dn6" else (rig.PTB_DN[L], B["eidsb"], B["actb"], rig.iq4nl, B["partsb"])
            seq.append((dnk, dnbufs, (P*8,), ()))
            seq.append(("cmbz2048", (B["partsb"], B["gatesb"], B["sgb"], B["shb"], hmid, hin), (P,), ()))
        else:
            seq.append(("gvf32ab", (w["wa"], w["wb"], hin, hmid), (1,), ()))  # shape-standin 1 kernel
    def _grid(n, g):
        gx = g[0] if isinstance(g, tuple) else g
        if n in ("gv8k2048p", "gv8k4096r") and P > 1: return (gx, P)
        return gx
    return [(rig.K[n], b, _grid(n, g), v) for n, b, g, v in seq]

def timeit(seq, tag, reps=30):
    gr = GraphRunnerUnc(rig, seq, tag)
    rig.feed(970, 100)
    gr.step()  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        gr.step()
    dt = (time.perf_counter() - t0) / reps * 1e3
    print(f"[p2m] {tag}: {len(seq)} kernels, {dt:.2f} ms/cycle", flush=True)
    return len(seq), dt

rows = []
rows.append(timeit(truncated_seq(40), "full40"))
rows.append(timeit(truncated_seq(20), "half20"))
rows.append(timeit(truncated_seq(10), "q10"))
rows.append(timeit(truncated_seq(40, drop_moe=True), "no-moe (quartet->1 standin)"))
rows.append(timeit(truncated_seq(40, drop_moe=True, drop_attn=True), "no-moe no-attn"))
(fk, ft), (hk, ht) = rows[0], rows[1]
slope = (ft - ht) / (fk - hk)
print(f"[p2m] SLOPE (full-vs-half): {slope*1000:.1f} us/kernel | intercept {ht - hk*slope:.2f} ms", flush=True)
json_out = dict(rows=[dict(k=k, ms=round(m, 3)) for k, m in rows], slope_us_per_kernel=round(slope*1000, 1))
import json
json.dump(json_out, open(os.path.expanduser("~/mm_p2_slope.json"), "w"))
print("[p2m] saved ~/mm_p2_slope.json", flush=True)
