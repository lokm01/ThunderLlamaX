# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
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
P, d, pr = E.P, E.P.d, E.pr
LS=(256,1,1)
draft_g, probe_g, accept_g, flush_g = E.graphs
prev = dev.timeline_value - 1
for c in range(2):
  vd = dev.next_timeline(); draft_g.submit(prev, vd)
  vp = dev.next_timeline(); probe_g.submit(vd, vp)
  va = dev.next_timeline(); accept_g.submit(vp, va)
  vf = dev.next_timeline(); flush_g.submit(va, vf)
  dev.timeline_signal.wait(vf); prev = vf
cur = int(P.down("cur_slot", (1,), np.int32)[0])
pos = int(P.down("pos_slot", (1,), np.int32)[0])
h_seed = P.down("h_seed", (5120,)).copy()
kv_d = P.down("kv_d", (2,4,2304,256), np.float16).astype(np.float32)
# eager GPU chain
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
pr["ddown"](d["d_fd"], d["gact_d"], d["hh_d"], d["hd_d0"], global_size=(640,1,1), local_size=LS, wait=True)
dev.synchronize()
gpu = {}
for nm, shape, dt in [("e_buf",(5120,),np.float32),("cat",(10240,),np.float16),("xin_d",(5120,),np.float32),
                      ("xh_d",(5120,),np.float16),("qrow_d",(12288,),np.float16),("krow_d",(1024,),np.float16),
                      ("vrow_d",(1024,),np.float16),("ao_row_d",(6144,),np.float16),("attn_out_d",(5120,),np.float16),
                      ("hhx_d",(5120,),np.float16),("gact_d",(17408,),np.float16),("hd_d0",(5120,),np.float32)]:
  gpu[nm] = P.down(nm, shape, dt).astype(np.float32)

def deq(fp):
  w = np.load(fp); nout, rowb = w.shape; ngrp = rowb // 144
  qs = w[:, :ngrp*128].reshape(nout, ngrp*8, 16)
  dd = np.frombuffer(w[:, ngrp*128:].tobytes(), dtype="<f2").reshape(nout, ngrp*8).astype(np.float32)
  out = np.zeros((nout, ngrp*256), np.float32)
  for s in range(ngrp*8):
    b = qs[:, s, :]
    nib = np.zeros((nout, 32), np.float32)
    for j in range(16):
      nib[:, 2*j] = (b[:, j] & 0xF) - 8
      nib[:, 2*j+1] = (b[:, j] >> 4) - 8
    out[:, s*32:(s+1)*32] = nib * dd[:, s:s+1]
  return out
def rmsn(x, w):
  xs = x.astype(np.float32); r = 1.0/np.sqrt((xs*xs).sum()/5120 + 1e-6)
  return (xs*r*w).astype(np.float16).astype(np.float32)
def rel(a, b): return float(np.abs(a-b).max()/max(1e-9, np.abs(b).max()))

enw = np.load("draft_pack/d_enw.npy"); hnw = np.load("draft_pack/d_hnw.npy")
e = gpu["e_buf"]
cat = np.concatenate([rmsn(e, enw), rmsn(h_seed, hnw)])
print("[c] cat", rel(gpu["cat"], cat), flush=True)
xin = (deq("draft_pack/d_eh.npy") @ cat).astype(np.float16).astype(np.float32)
print("[c] xin", rel(gpu["xin_d"], xin), flush=True)
print("[c] xinG[:6]", gpu["xin_d"][:6], flush=True)
print("[c] xinR[:6]", xin[:6], flush=True)
xh = rmsn(xin, np.load("draft_pack/d_nw1.npy"))
print("[c] xh", rel(gpu["xh_d"], xh), flush=True)
qrow = (deq("draft_pack/d_q.npy") @ xh).astype(np.float16).astype(np.float32)
krow = (deq("draft_pack/d_k.npy") @ xh).astype(np.float16).astype(np.float32)
vrow = (deq("draft_pack/d_v.npy") @ xh).astype(np.float16).astype(np.float32)
print("[c] qrow", rel(gpu["qrow_d"], qrow), "krow", rel(gpu["krow_d"], krow), "vrow", rel(gpu["vrow_d"], vrow), flush=True)
qnw = np.load("draft_pack/d_qnw.npy"); knw = np.load("draft_pack/d_knw.npy")
freqs = (1.0/(1e7**(np.arange(0,64,2,dtype=np.float64)/64.0))).astype(np.float32)
ao = np.zeros(6144, np.float32)
for h in range(24):
  kvh = h // 6
  q = qrow[h*512:h*512+256].copy(); g = qrow[h*512+256:h*512+512]
  k = krow[kvh*256:(kvh+1)*256].copy(); v = vrow[kvh*256:(kvh+1)*256].copy()
  q = np.float16(q/np.sqrt((q*q).mean()+1e-6)).astype(np.float32)*qnw
  k = np.float16(k/np.sqrt((k*k).mean()+1e-6)).astype(np.float32)*knw
  ang = pos*freqs[np.arange(64)//2]
  qr = q.copy(); kr = k.copy()
  for i in range(32):
    c_, s_ = np.cos(ang[i]), np.sin(ang[i])
    qr[i], qr[i+32] = q[i]*c_-q[i+32]*s_, q[i+32]*c_+q[i]*s_
    kr[i], kr[i+32] = k[i]*c_-k[i+32]*s_, k[i+32]*c_+k[i]*s_
  K = kv_d[0, kvh].copy(); V = kv_d[1, kvh].copy()
  K[pos] = np.float16(kr); V[pos] = np.float16(v)
  sc = (K[:pos+1] @ (qr*0.0625)).astype(np.float32)
  p = np.exp(sc - sc.max()); p /= p.sum()
  ao[h*256:(h+1)*256] = (p @ V[:pos+1]) * (1.0/(1.0+np.exp(-g)))
print("[c] ao_row", rel(gpu["ao_row_d"], ao), flush=True)
attn_out = (deq("draft_pack/d_o.npy") @ ao).astype(np.float16).astype(np.float32)
print("[c] attn_out", rel(gpu["attn_out_d"], attn_out), flush=True)
hh = xin + attn_out
hhx = rmsn(hh, np.load("draft_pack/d_nw2.npy"))
print("[c] hhx", rel(gpu["hhx_d"], hhx), flush=True)
g_ = (deq("draft_pack/d_fg.npy") @ hhx).astype(np.float16).astype(np.float32)
u_ = (deq("draft_pack/d_fu.npy") @ hhx).astype(np.float16).astype(np.float32)
gact = (g_/(1+np.exp(-g_))*u_).astype(np.float16).astype(np.float32)
print("[c] gact", rel(gpu["gact_d"], gact), flush=True)
hd = hh + (deq("draft_pack/d_fd.npy") @ gact).astype(np.float16).astype(np.float32)
print("[c] hd", rel(gpu["hd_d0"], hd), flush=True)
