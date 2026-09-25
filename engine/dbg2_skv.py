# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV"); os.environ.setdefault("SKV", "1")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine
from engine0 import dev
LS = (256,1,1)
snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
P, W, d, pr = E.P, E.W, E.P.d, E.pr
P.up("pos_slot", np.array([int(snap["P"].reshape(-1)[0]) - 1], dtype=np.int32))
i0 = E.attn_idx[0]
kvs = snap[f"kv{E.attn_idx.index(i0)}"].astype(np.float16).reshape(2, 4, 2048, 256)
kvbig = np.full((2, 4, 2304, 256), 7.7, dtype=np.float16); kvbig[:, :, :2048] = kvs
rng = np.random.default_rng(1)
P.up("x0", (0.1*rng.standard_normal(5120)).astype(np.float32))
P.up(f"kv{i0}", kvbig.reshape(-1))
dev.synchronize()
pr["k0_norm"](d["x0"], W[("nw1",i0)], d["xh"], global_size=(1,1,1), local_size=LS)
qkname = "aq6k8" if E.qtypes[i0] == 14 else "aq3k8"
pr[qkname](W[("q",i0)], W[("k",i0)], W[("v",i0)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=True)
# ---- reference old path (on the CURRENT kv state; appends row pos) ----
pr["a_attn"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i0)], W[("knw",i0)], d["freqs"], d[f"kv{i0}"], d["pos_slot"], d["ao_row"], global_size=(24,1,1), local_size=LS, wait=True)
old = P.down("ao_row", (6144,), np.float16).copy()
kva = P.down(f"kv{i0}", (2,4,2304,256), np.float16).copy()
# ---- spk_pre1 ONLY (on the SAME kv state: row pos already appended by a_attn;
#      re-appending identical values) ----
pr["spk_pre1"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i0)], W[("knw",i0)], d["freqs"], d[f"kv{i0}"], d["pos_slot"], d["qw1"], global_size=(24,1,1), local_size=LS, wait=True)
qw = P.down("qw1", (24, 256), np.float32).copy()
kvb = P.down(f"kv{i0}", (2,4,2304,256), np.float16).copy()
same_append = bool((kva[:, :, 1987:1988, :] == kvb[:, :, 1987:1988, :]).all())
print(f"[dbg2] append-same={same_append} qw absmax={np.abs(qw).max():.3f} (0.0625-scaled q)", flush=True)
# numpy check of qw for head 0: q-norm+rope from qrow, qnw, freqs, pos
pos = int(snap["P"].reshape(-1)[0]) - 1
qrow = P.down("qrow", (12288,), np.float16)
qnw = P.down(W and "w_qnw_%d" % i0, (256,), np.float32)
fr = P.down("freqs", (32,), np.float32)
h = 0; qv = qrow[h*512:h*512+256].astype(np.float64)
r = 1.0/np.sqrt((qv*qv).mean() + 1e-6)
qn = np.float64(np.float16(qv*r)) * qnw.astype(np.float64)
qe = np.zeros(256)
for dd in range(256):
    c, s_ = (np.cos(pos*fr[dd&31]), np.sin(pos*fr[dd&31])) if dd < 64 else (1.0, 0.0)
    if dd < 32: qe[dd] = qn[dd]*c - qn[dd+32]*s_
    elif dd < 64: qe[dd] = qn[dd]*c + qn[dd-32]*s_
    else: qe[dd] = qn[dd]
print(f"[dbg2] qw head0 relerr vs numpy: {np.abs(qw[0]-qe*0.0625).max()/np.abs(qe*0.0625).max():.3e}", flush=True)
# ---- spk_a1 alone ----
pr["spk_a1"](d[f"kv{i0}"], d["qw1"], d["pos_slot"], d["pm1"], d["ps1"], d["pA1"], global_size=(128,1,1), local_size=LS, wait=True)
pm = P.down("pm1", (4,32,6), np.float32); ps = P.down("ps1", (4,32,6), np.float32)
print(f"[dbg2] pm range [{pm.min():.2f},{pm.max():.2f}] ps range [{ps.min():.3f},{ps.max():.3f}]", flush=True)
# numpy softmax check for head0 (kvh 0): scores over l<=pos
K = kvb[0,0,:pos+1,:].astype(np.float64)
sc = K @ qw[0].astype(np.float64)
p_ = np.exp(sc - sc.max()); p_ /= p_.sum()
V = kvb[1,0,:pos+1,:].astype(np.float64)
ref_o = p_ @ V
gf = qrow[0*512+256:0*512+512].astype(np.float64)
ref_o = ref_o * (1.0/(1.0+np.exp(-gf)))
pr["spk_c1"](d["pm1"], d["ps1"], d["pA1"], d["qrow"], d["ao_row"], global_size=(24,1,1), local_size=LS, wait=True)
new = P.down("ao_row", (6144,), np.float16)
rel_new = np.abs(new[:256].astype(np.float64)-ref_o).max()/max(np.abs(ref_o).max(),1e-30)
rel_old = np.abs(old[:256].astype(np.float64)-ref_o).max()/max(np.abs(ref_o).max(),1e-30)
print(f"[dbg2] head0 relerr: trio={rel_new:.3e} old={rel_old:.3e}", flush=True)
print("[dbg2 done]", flush=True)
