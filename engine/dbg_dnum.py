# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2 draft numeric validation: run ONE draft step eagerly on real state, dump
every intermediate, verify each stage vs numpy (pack decode + norms + gemvs)."""
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
for t in ([3204, 40224]*30) + ids:
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])
E.restore_mtp(snap)
E.fill_draft(ids)
dev.synchronize()
d, pr, P = E.P.d, E.pr, E.P

LS = (256,1,1)
# one eager draft step (step-0 semantics): embed cur, hm = h_seed (zeros), pos_slot
pr["h_embed"](E.W[("emb",0)], d["grid512"], d["cur_slot"], d["e_buf"], global_size=(1,1,1), local_size=LS)
pr["dnorm2"](d["e_buf"], d["h_seed"], d["d_enw"], d["d_hnw"], d["cat"], global_size=(1,1,1), local_size=LS)
pr["ehproj"](d["d_eh"], d["cat"], d["zed5k"], d["xin_d"], global_size=(640,1,1), local_size=LS)
pr["k0_norm"](d["xin_d"], d["d_nw1"], d["xh_d"], global_size=(1,1,1), local_size=LS)
pr["dq"](d["d_q"], d["xh_d"], d["zed5k"], d["qrow_d"], global_size=(1536,1,1), local_size=LS)
pr["dkv"](d["d_k"], d["d_v"], d["xh_d"], d["krow_d"], d["vrow_d"], global_size=(256,1,1), local_size=LS)
pr["aattn_d"](d["qrow_d"], d["krow_d"], d["vrow_d"], d["d_qnw"], d["d_knw"], d["freqs"], d["kv_d"], d["pos_slot"], d["ao_row_d"], global_size=(24,1,1), local_size=LS, wait=True)
dev.synchronize()

e = P.down("e_buf", (5120,))
cat = P.down("cat", (10240,), np.float16)
xin = P.down("xin_d", (5120,))
xh = P.down("xh_d", (5120,), np.float16)
qr = P.down("qrow_d", (12288,), np.float16)
kr = P.down("krow_d", (1024,), np.float16)
vr = P.down("vrow_d", (1024,), np.float16)
ao = P.down("ao_row_d", (6144,), np.float16)

# numpy checks
enw = np.load("draft_pack/d_enw.npy"); hnw = np.load("draft_pack/d_hnw.npy")
def rms(x, w):
  r = 1.0/np.sqrt((x.astype(np.float64)**2).mean() + 1e-6)
  return (x*r*w).astype(np.float16).astype(np.float32)
# NOTE: kernel computes fp32 r then half(x*r*nw[i]) per element
def rms32(x, w):
  ss = np.float32(0)
  xs = x.astype(np.float32)
  ss = (xs*xs).sum(dtype=np.float32)
  r = 1.0/np.sqrt(ss/5120 + 1e-6)
  return (xs*r*w).astype(np.float16).astype(np.float32)
cat_ref = np.concatenate([rms32(e, enw), rms32(np.zeros(5120, np.float32), hnw)])
cat_g = cat[:5120].astype(np.float32)
print(f"[n] cat enorm relerr: {np.abs(cat_g-cat_ref[:5120]).max()/max(1e-9,np.abs(cat_ref[:5120]).max()):.3g}", flush=True)

def deq(npy, xin_rows):
  # npy: (NOUT, rowb) packed; returns dequant float matrix NOUT x NIN lazily per check
  return npy
w_eh = np.load("draft_pack/d_eh.npy")
def q4_matvec(w, x):
  nout = w.shape[0]; ngrp = w.shape[1]//144
  out = np.zeros(nout, np.float32)
  for r in range(nout):
    row = w[r]
    acc = 0.0
    for grp in range(ngrp):
      qs = row[:ngrp*128].reshape(-1, 16)
      dd = np.frombuffer(row[ngrp*128:].tobytes(), dtype="<f2")
      for lane in range(32):
        subi = grp*8 + (lane>>2)
        q8 = qs[subi]
        dv = float(dd[subi])
        for j in range(8):
          e = grp*256 + lane*8 + j
          byte = q8[(lane&1)*4 + (j>>1)] if ((lane>>1)&1)==0 else q8[8 + (lane&1)*4 + (j>>1)]
          qv = (byte >> ((j&1)*4)) & 0xF
          acc += np.float32(np.float16(x[e]) * np.float16(np.float32(dv)*(np.float32(qv)-8.0)))
    out[r] = acc
  return out
x = cat.astype(np.float32)
xin_ref = q4_matvec(w_eh[:8], x)  # first 8 rows only (slow python)
xin_g = xin[:8]
print(f"[n] xin_d[:8] relerr: {np.abs(xin_g-xin_ref).max()/max(1e-9,np.abs(xin_ref).max()):.3g}", flush=True)
print(f"[n] xin[:6] gpu={np.round(xin_g[:6],4)} ref={np.round(xin_ref[:6],4)}", flush=True)
nw1 = np.load("draft_pack/d_nw1.npy")
xh_ref = rms32(xin, nw1)
print(f"[n] xh relerr: {np.abs(xh.astype(np.float32)-xh_ref).max()/max(1e-9,np.abs(xh_ref).max()):.3g}", flush=True)
w_q = np.load("draft_pack/d_q.npy")
qr_ref = q4_matvec(w_q[:8], xh.astype(np.float32))
print(f"[n] qrow[:8] relerr: {np.abs(qr[:8].astype(np.float32)-qr_ref).max()/max(1e-9,np.abs(qr_ref).max()):.3g}", flush=True)
print(f"[n] qrow[:6] gpu={np.round(qr[:6].astype(np.float32),4)} ref={np.round(qr_ref[:6],4)}", flush=True)
print(f"[n] krow absmax={np.abs(kr.astype(np.float32)).max():.3g} vrow absmax={np.abs(vr.astype(np.float32)).max():.3g} ao absmax={np.abs(ao.astype(np.float32)).max():.3g}", flush=True)
print("[n] done", flush=True)
