# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""numpy end-to-end draft step on the same device state -> localizes semantics vs kernel."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from mtp import MTPEngine, SLICE
from engine0 import dev
snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
ids = snap["ids"].reshape(-1).tolist()
seen, sl = set(), []
for t in ([3204,40224]*30) + ids:
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])
E.restore_mtp(snap)
E.fill_draft(ids)
E.build_graphs()
dev.synchronize()
P, d = E.P, E.P.d
draft_g, probe_g, accept_g, flush_g = E.graphs
prev = dev.timeline_value - 1
for c in range(2):
  vd = dev.next_timeline(); draft_g.submit(prev, vd)
  vp = dev.next_timeline(); probe_g.submit(vd, vp)
  va = dev.next_timeline(); accept_g.submit(vp, va)
  vf = dev.next_timeline(); flush_g.submit(va, vf)
  dev.timeline_signal.wait(vf); prev = vf
# state for an eager-equivalent step0 of cycle 2
cur = int(P.down("cur_slot", (1,), np.int32)[0])
pos = int(P.down("pos_slot", (1,), np.int32)[0])
h_seed = P.down("h_seed", (5120,)).copy()
kv_d = P.down("kv_d", (2,4,2304,256), np.float16).astype(np.float32)
print(f"[npy] cur={cur} pos={pos} h_seed absmax={np.abs(h_seed).max():.3g}", flush=True)

# ---- numpy draft step ----
import struct
def deq(npy):
  w = npy  # (nout, rowb)
  nout, rowb = w.shape
  ngrp = rowb // 144
  qs = w[:, :ngrp*128].reshape(nout, ngrp*8, 16)
  dd = np.frombuffer(w[:, ngrp*128:].tobytes(), dtype="<f2").reshape(nout, ngrp*8).astype(np.float32)
  nib = np.zeros((nout, ngrp*8, 32), np.float32)
  for s in range(16):
    nib[:, :, 2*s] = (qs[:, :, s] & 0xF) - 8
    nib[:, :, 2*s+1] = (qs[:, :, s] >> 4) - 8
  # to NOUT x NIN: element e of sub s
  out = np.zeros((nout, ngrp*256), np.float32)
  for s in range(ngrp*8):
    out[:, s*32:(s+1)*32] = nib[:, s, :] * dd[:, s:s+1]
  return out

W_eh = deq(np.load("draft_pack/d_eh.npy"))        # (5120, 10240)
# embedding row (IQ3_S dequant) — reuse GPU: upload via h_embed? simpler: run h_embed once
E.pr["h_embed"](E.W[("emb",0)], d["grid512"], d["cur_slot"], d["e_buf"], global_size=(1,1,1), local_size=(256,1,1), wait=True)
e = P.down("e_buf", (5120,)).copy()
def rmsn(x, w):
  xs = x.astype(np.float32)
  r = 1.0/np.sqrt((xs*xs).sum()/5120 + 1e-6)
  return (xs*r*w).astype(np.float16).astype(np.float32)
cat = np.concatenate([rmsn(e, np.load("draft_pack/d_enw.npy")), rmsn(h_seed, np.load("draft_pack/d_hnw.npy"))])
xin = (W_eh @ cat).astype(np.float16).astype(np.float32)
xh = rmsn(xin, np.load("draft_pack/d_nw1.npy"))
W_q = deq(np.load("draft_pack/d_q.npy"))
W_k = deq(np.load("draft_pack/d_k.npy"))
W_v = deq(np.load("draft_pack/d_v.npy"))
qrow = (W_q @ xh).astype(np.float16).astype(np.float32)
krow = (W_k @ xh).astype(np.float16).astype(np.float32)
vrow = (W_v @ xh).astype(np.float16).astype(np.float32)
qnw = np.load("draft_pack/d_qnw.npy"); knw = np.load("draft_pack/d_knw.npy")
freqs = (1.0/(1e7**(np.arange(0,64,2,dtype=np.float64)/64.0))).astype(np.float32)
ao = np.zeros(6144, np.float32)
for h in range(24):
  kvh = h // 6
  q = qrow[h*512:h*512+256].copy(); g = qrow[h*512+256:h*512+512]
  k = krow[kvh*256:(kvh+1)*256].copy(); v = vrow[kvh*256:(kvh+1)*256].copy()
  q = (np.float16(q/np.sqrt((q*q).mean()+1e-6)).astype(np.float32))*qnw
  k = (np.float16(k/np.sqrt((k*k).mean()+1e-6)).astype(np.float32))*knw
  # rope partial 64: pairs (i, i+32)
  ang = pos*freqs[np.arange(64)//2]
  qr = q.copy()
  for i in range(32):
    c, s = np.cos(ang[i]), np.sin(ang[i])
    qr[i] = q[i]*c - q[i+32]*s
    qr[i+32] = q[i+32]*c + q[i]*s
  kr = k.copy()
  for i in range(32):
    c, s = np.cos(ang[i]), np.sin(ang[i])
    kr[i] = k[i]*c - k[i+32]*s
    kr[i+32] = k[i+32]*c + k[i]*s
  K = kv_d[0, kvh].copy(); V = kv_d[1, kvh].copy()
  K[pos] = np.float16(kr); V[pos] = np.float16(v)
  qq = qr * 0.0625
  sc = (K[:pos+1] @ qq).astype(np.float32)
  p = np.exp(sc - sc.max()); p /= p.sum()
  o = p @ V[:pos+1]
  ao[h*256:(h+1)*256] = o * (1.0/(1.0+np.exp(-g)))
W_o = deq(np.load("draft_pack/d_o.npy"))
attn_out = (W_o @ ao).astype(np.float16).astype(np.float32)
hh = xin + attn_out
hhx = rmsn(hh, np.load("draft_pack/d_nw2.npy"))
W_fg = deq(np.load("draft_pack/d_fg.npy")); W_fu = deq(np.load("draft_pack/d_fu.npy")); W_fd = deq(np.load("draft_pack/d_fd.npy"))
g = (W_fg @ hhx).astype(np.float16).astype(np.float32)
u = (W_fu @ hhx).astype(np.float16).astype(np.float32)
gact = (g/(1+np.exp(-g))*u).astype(np.float16).astype(np.float32)
hd = hh + (W_fd @ gact).astype(np.float16).astype(np.float32)
hdn = rmsn(hd, np.load("draft_pack/d_shnw.npy"))
# head slice via GPU shead on the NUMPY hdn
P.up("xh_d", hdn.astype(np.float16))
P.up("dr1w", np.array([-9], np.int32))
dev.synchronize()
E.pr["shead"](d["slice_w"], d["xh_d"], d["slogits"], global_size=(SLICE//8,1,1), local_size=(256,1,1))
E.pr["samx"](d["slogits"], d["stab"], d["dr1w"], global_size=(1,1,1), local_size=(256,1,1), wait=True)
prop = int(P.down("dr1w",(1,),np.int32)[0])
slg = P.down("slogits",(SLICE,),np.float16).astype(np.float32)
stab_d = P.down("stab",(SLICE,),np.int32)
i3204 = int(np.argmax(stab_d == 3204)); i40224 = int(np.argmax(stab_d == 40224))
print(f"[npy] NUMPY-block + GPU-head proposal: {prop}; logit(3204)={slg[i3204]:.3f} logit(40224)={slg[i40224]:.3f} top={slg.max():.3f}", flush=True)
# and the full-GPU eager step for comparison (same cur/pos/h_seed state)
LS=(256,1,1); pr = E.pr
pr["h_embed"](E.W[("emb",0)], d["grid512"], d["cur_slot"], d["e_buf"], global_size=(1,1,1), local_size=LS)
pr["dnorm2"](d["e_buf"], d["h_seed"], d["d_enw"], d["d_hnw"], d["cat"], global_size=(1,1,1), local_size=LS)
pr["ehproj"](d["d_eh"], d["cat"], d["zed5k"], d["xin_d"], global_size=(640,1,1), local_size=LS)
pr["k0_norm"](d["xin_d"], d["d_nw1"], d["xh_d"], global_size=(1,1,1), local_size=LS)
pr["dq"](d["d_q"], d["xh_d"], d["zed5k"], d["qrow_d"], global_size=(1536,1,1), local_size=LS)
pr["dkv"](d["d_k"], d["d_v"], d["xh_d"], d["krow_d"], d["vrow_d"], global_size=(256,1,1), local_size=LS)
pr["aattn_d"](d["qrow_d"], d["krow_d"], d["vrow_d"], d["d_qnw"], d["d_knw"], d["freqs"], d["kv_d"], d["pos_slot"], d["ao_row_d"], global_size=(24,1,1), local_size=LS)
pr["doproj"](d["d_o"], d["ao_row_d"], d["hh_d"], d["attn_out_d"], global_size=(640,1,1), local_size=LS)
pr["k3m_hh"](d["xin_d"], d["attn_out_d"], d["d_nw2"], d["hh_d"], d["hhx_d"], global_size=(1,1,1), local_size=LS)
pr["dfgu"](d["d_fg"], d["d_fu"], d["hhx_d"], d["gact_d"], global_size=(2176,1,1), local_size=LS)
pr["ddown"](d["d_fd"], d["gact_d"], d["hh_d"], d["hd_d0"], global_size=(640,1,1), local_size=LS)
pr["k0_norm"](d["hd_d0"], d["d_shnw"], d["xh_d2"], global_size=(1,1,1), local_size=LS) if "xh_d2" in d else pr["k0_norm"](d["hd_d0"], d["d_shnw"], d["xh_d"], global_size=(1,1,1), local_size=LS)
pr["shead"](d["slice_w"], d["xh_d"], d["slogits"], global_size=(SLICE//8,1,1), local_size=LS)
pr["samx"](d["slogits"], d["stab"], d["dr1w"], global_size=(1,1,1), local_size=LS, wait=True)
print(f"[npy] FULL-GPU proposal: {int(P.down('dr1w',(1,),np.int32)[0])}", flush=True)
