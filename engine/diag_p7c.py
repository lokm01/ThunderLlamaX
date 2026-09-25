# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-C diag: run pfca alone, compare scratch factors vs numpy for h=0..3;
then pfcb; report nan patterns + per-factor relerr."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from test_p7c import make_inputs, up_weights, up_run_bufs, relerr, prog, P, LS, HC_BYTES
from engine0 import dev

C, NC, T = 64, 1, 64
inp = make_inputs(T, 11)
up_weights(P, inp)
up_run_bufs(P, "d", T, inp)
P.poison("d_scr", 48 * NC * HC_BYTES[C], np.uint8, 0xAB)
P.poison("d_o", T * 6144 * 4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()

ca = prog("pfca_c64_nc1_nw16")
ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d["d_scr"],
   global_size=(48 * NC, 1, 1), local_size=(512, 1, 1))
dev.synchronize()
scr = P.down("d_scr", (48 * NC * HC_BYTES[C] // 2,), np.float16)
LDK, LDTC = 136, 72
KH_C, KHT_C, QE_C, U_C, VR_C, M_C = C*LDK, 128*LDTC, C*LDK, C*LDK, C*LDK, C*LDTC
META_OFF = (2*(KH_C + KHT_C + QE_C + U_C + VR_C + 2*M_C) + 15)//16*16
for h in [0, 5, 17, 47]:
  hc = scr[h*NC*(HC_BYTES[C]//2):][:HC_BYTES[C]//2]
  kh = hc[:KH_C].reshape(C, LDK)[:, :128].astype(np.float64)
  kht = hc[KH_C:KH_C+KHT_C].reshape(128, LDTC)[:, :C].astype(np.float64)
  qe = hc[KH_C+KHT_C:KH_C+KHT_C+QE_C].reshape(C, LDK)[:, :128].astype(np.float64)
  u = hc[KH_C+KHT_C+QE_C:KH_C+KHT_C+QE_C+U_C].reshape(C, LDK)[:, :128].astype(np.float64)
  m = hc[KH_C+KHT_C+QE_C+U_C+VR_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C].reshape(C, LDTC)[:, :C].astype(np.float64)
  met = np.frombuffer(hc[META_OFF//2:META_OFF//2+(2*C+4)*2].tobytes(), dtype=np.float32)
  print(f"[h{h}] nan: kh {np.isnan(kh).sum()} kht {np.isnan(kht).sum()} qe {np.isnan(qe).sum()} "
        f"u {np.isnan(u).sum()} m {np.isnan(m).sum()} | gend {met[2*C]:.3f} bg[:3] {met[:3]} sg[:3] {met[C:C+3]}", flush=True)
  if h == 0:
    print(f"[h0] kht^T vs kh maxdiff {np.abs(kht.T - kh).max():.3e}", flush=True)

# numpy factor check for h=0
qkv, araw, braw, convw, dtb, ssma = inp["qkv"], inp["araw"], inp["braw"], inp["convw"].reshape(10240,4), inp["dtb"], inp["ssma"]
conv_live = inp["conv0"].astype(np.float64)
h = 0; khq = h % 16
qc0, kc0, vc0 = khq*128, 2048+khq*128, 4096+h*128
lam = np.zeros(C); bet = np.zeros(C); KH_ = np.zeros((C,128)); QH_ = np.zeros((C,128)); V_ = np.zeros((C,128))
for t in range(C):
  rt = qkv[t].astype(np.float64)
  r3 = qkv[t-3].astype(np.float64) if t>=3 else conv_live[t+0]
  r2 = qkv[t-2].astype(np.float64) if t>=2 else conv_live[t+1]
  r1 = qkv[t-1].astype(np.float64) if t>=1 else conv_live[t+2]
  sq = r3[qc0:qc0+128]*convw[qc0:qc0+128,0]+r2[qc0:qc0+128]*convw[qc0:qc0+128,1]+r1[qc0:qc0+128]*convw[qc0:qc0+128,2]+rt[qc0:qc0+128]*convw[qc0:qc0+128,3]
  sk = r3[kc0:kc0+128]*convw[kc0:kc0+128,0]+r2[kc0:kc0+128]*convw[kc0:kc0+128,1]+r1[kc0:kc0+128]*convw[kc0:kc0+128,2]+rt[kc0:kc0+128]*convw[kc0:kc0+128,3]
  sv = r3[vc0:vc0+128]*convw[vc0:vc0+128,0]+r2[vc0:vc0+128]*convw[vc0:vc0+128,1]+r1[vc0:vc0+128]*convw[vc0:vc0+128,2]+rt[vc0:vc0+128]*convw[vc0:vc0+128,3]
  sq = sq/(1+np.exp(-sq)); sk = sk/(1+np.exp(-sk)); sv = sv/(1+np.exp(-sv))
  qn = (1/max(np.sqrt((sq*sq).sum()), 1e-6))*0.08838834764831845
  kn = 1/max(np.sqrt((sk*sk).sum()), 1e-6)
  KH_[t], QH_[t], V_[t] = sk*kn, sq*qn, sv
  x = float(araw[t,h]+dtb[h]); lam[t] = (max(x,0)+np.log1p(np.exp(-abs(x))))*ssma[h]
  bet[t] = 1/(1+np.exp(-float(braw[t,h])))
g2 = np.cumsum(lam*np.log2(np.e))
hc = scr[:HC_BYTES[C]//2]
kh0 = hc[:KH_C].reshape(C, LDK)[:, :128].astype(np.float64)
qe0 = hc[KH_C+KHT_C:KH_C+KHT_C+QE_C].reshape(C, LDK)[:, :128].astype(np.float64)
print(f"[factors h0] khat relerr {relerr(kh0, KH_)[0]:.3e} | qe relerr {relerr(qe0, QH_*2**g2[:,None])[0]:.3e}", flush=True)
# T, U checks
Bm = np.zeros((C,C))
for i in range(C):
  for j in range(i):
    Bm[i,j] = bet[i]*(KH_[i]@KH_[j])*2**(g2[i]-g2[j])
T_ = np.linalg.inv(np.eye(C)+Bm)
U_ = T_@(bet[:,None]*V_)
t0 = hc[KH_C+KHT_C+QE_C+U_C+VR_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C].reshape(C, LDTC)[:, :C].astype(np.float64)  # t region after m
# careful: order is m then t
m0 = hc[KH_C+KHT_C+QE_C+U_C+VR_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C].reshape(C, LDTC)[:, :C].astype(np.float64)
t0 = hc[KH_C+KHT_C+QE_C+U_C+VR_C+M_C:KH_C+KHT_C+QE_C+U_C+VR_C+2*M_C].reshape(C, LDTC)[:, :C].astype(np.float64)
u0 = hc[KH_C+KHT_C+QE_C:KH_C+KHT_C+QE_C+U_C].reshape(C, LDK)[:, :128].astype(np.float64)
M_ = np.zeros((C,C))
for i in range(C):
  for j in range(i+1):
    M_[i,j] = (QH_[i]@KH_[j])*2**(g2[i]-g2[j])
print(f"[factors h0] T relerr {relerr(t0, T_)[0]:.3e} | U relerr {relerr(u0, U_)[0]:.3e} | M relerr {relerr(m0, M_)[0]:.3e}", flush=True)

cb = prog("pfcb_c64_nc1_nw8")
cb(P.d["d_scr"], P.d["d_rec"], P.d["d_o"], global_size=(192, 1, 1), local_size=LS)
dev.synchronize()
o = P.down("d_o", (T, 6144), np.float32)
nn = np.isnan(o).sum()
print(f"[pfcb] oout nan {nn}/{o.size}", flush=True)
if nn:
  rows = np.where(np.isnan(o).any(axis=1))[0]
  cols = np.where(np.isnan(o).any(axis=0))[0]
  print(f"[pfcb] nan rows {rows[:10]}... cols(heads*128+v) {cols[:10]}", flush=True)
  print(f"[pfcb] col//128 hist {np.unique(cols//128, return_counts=True)}", flush=True)
