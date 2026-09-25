# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2b diag: C=32 NC=1 pfca/pfcb/pfcz factor-level check vs numpy fp64 oracle.
Layout constants mirror the CURRENT pf_scanchunk.cu macros (hi-lo era).
Usage: ~/tg311/bin/python -u diag_c32.py [C] [NC]
"""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from test_p7c import make_inputs, up_weights, relerr, prog, P, LS, scan_ref
from engine0 import dev

C = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T = C * NC
LDK, LDTC = 136, C + 8
KH_C, KHT_C, QE_C, U_C, VR_C = C*LDK, 128*LDTC, C*LDK, C*LDK, C*LDK
M_C, T_C = C*LDTC, C*LDTC
ULO_C, VRLO_C, MLO_C, TLO_C = U_C, VR_C, M_C, T_C
DZ_C = 256 * C
HC_HALF = 4*C*LDK + 128*LDTC + 4*C*LDTC + 2*C*LDK + DZ_C
META_OFF = (HC_HALF*2 + 15)//16*16
LO_OFF = (META_OFF + (2*C+4)*4 + 15)//16*16
HC_BYTES = (LO_OFF + (C*LDK + C*LDK + C*LDK + 128*LDTC)*2 + 15)//16*16
KHLO_C, QHLO_C, QELO_C, KHTLO_C = C*LDK, C*LDK, C*LDK, 128*LDTC
print(f"[diag] C={C} NC={NC} LDK={LDK} LDTC={LDTC} HC_HALF={HC_HALF} META_OFF={META_OFF} LO_OFF={LO_OFF} HC_BYTES={HC_BYTES}", flush=True)
assert HC_BYTES == {64: 263696, 32: 125712}[C], "layout mismatch vs bank"

inp = make_inputs(T, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_gate", inp["gate"].reshape(-1))
P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
P.poison("d_scr", 48 * NC * HC_BYTES, np.uint8, 0xAB)
P.poison("d_o", T * 6144 * 4, np.float32, 7.7e31)
P.poison("d_z", T * 6144 * 2, np.float16, 7.7)
dev.synchronize(); P._keep.clear()

O_ref, Z_ref, S_ref, convf_ref = scan_ref(inp["qkv"], inp["gate"], inp["araw"], inp["braw"],
                                          inp["convw"], inp["dtb"], inp["ssma"], inp["snw"],
                                          inp["conv0"], inp["rec0"])

ca = prog(f"pfca_c{C}_nc{NC}_nw16")
ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d["d_scr"],
   global_size=(48 * NC, 1, 1), local_size=(512, 1, 1))
dev.synchronize()
scr_bytes = P.down("d_scr", (48 * NC * HC_BYTES,), np.uint8)
scr = scr_bytes.view(np.float16).astype(np.float32)
HC_H = HC_BYTES // 2

def nan_of(a): return int(np.isnan(a).sum())
for h in [0, 1, 5, 47]:
  hc = scr[h*NC*HC_H:][:HC_H]
  kh = hc[:KH_C].reshape(C, LDK)[:, :128]
  kht = hc[KH_C:KH_C+KHT_C].reshape(128, LDTC)[:, :C]
  qe = hc[KH_C+KHT_C:KH_C+KHT_C+QE_C].reshape(C, LDK)[:, :128]
  u = hc[KH_C+KHT_C+QE_C:KH_C+KHT_C+QE_C+U_C].reshape(C, LDK)[:, :128]
  vr = hc[KH_C+KHT_C+QE_C+U_C:KH_C+KHT_C+QE_C+U_C+VR_C].reshape(C, LDK)[:, :128]
  m = hc[KH_C+KHT_C+QE_C+U_C+VR_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C].reshape(C, LDTC)[:, :C]
  t = hc[KH_C+KHT_C+QE_C+U_C+VR_C+M_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C].reshape(C, LDTC)[:, :C]
  u_lo = hc[KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C].reshape(C, LDK)[:, :128]
  m_lo = hc[KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C+MLO_C].reshape(C, LDTC)[:, :C]
  t_lo = hc[KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C+MLO_C:KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C+MLO_C+TLO_C].reshape(C, LDTC)[:, :C]
  lo = scr[h*NC*HC_H + LO_OFF//2:][: (KHLO_C+QHLO_C+QELO_C+KHTLO_C)]
  kh_lo = lo[:KHLO_C].reshape(C, LDK)[:, :128]
  qh_lo = lo[KHLO_C:KHLO_C+QHLO_C].reshape(C, LDK)[:, :128]
  qe_lo = lo[KHLO_C+QHLO_C:KHLO_C+QHLO_C+QELO_C].reshape(C, LDK)[:, :128]
  kht_lo = lo[KHLO_C+QHLO_C+QELO_C:].reshape(128, LDTC)[:, :C]
  met = np.frombuffer(scr_bytes[h*NC*HC_BYTES + META_OFF: h*NC*HC_BYTES + META_OFF + (2*C+4)*4].tobytes(), dtype=np.float32)
  print(f"[h{h}] nan: kh {nan_of(kh)} kht {nan_of(kht)} qe {nan_of(qe)} u {nan_of(u)} vr {nan_of(vr)} m {nan_of(m)} t {nan_of(t)} "
        f"u_lo {nan_of(u_lo)} m_lo {nan_of(m_lo)} t_lo {nan_of(t_lo)} kh_lo {nan_of(kh_lo)} qe_lo {nan_of(qe_lo)} kht_lo {nan_of(kht_lo)} | "
        f"absmax: u {np.nanmax(np.abs(u)):.2f} t {np.nanmax(np.abs(t)):.2f} m {np.nanmax(np.abs(m)):.2f} | gend {met[2*C]:.4f}", flush=True)

# ---- numpy factor oracle for h=0 (fp64) ----
qkv, araw, braw = inp["qkv"], inp["araw"], inp["braw"]
convw = inp["convw"].reshape(10240, 4); dtb, ssma = inp["dtb"], inp["ssma"]
conv_live = inp["conv0"].astype(np.float64)
h = 0; khq = h % 16
qc0, kc0, vc0 = khq*128, 2048+khq*128, 4096+h*128
lam = np.zeros(C); bet = np.zeros(C); KH_ = np.zeros((C,128)); QH_ = np.zeros((C,128)); V_ = np.zeros((C,128))
for tt in range(C):
  rt = qkv[tt].astype(np.float64)
  r3 = qkv[tt-3].astype(np.float64) if tt>=3 else conv_live[tt+0]
  r2 = qkv[tt-2].astype(np.float64) if tt>=2 else conv_live[tt+1]
  r1 = qkv[tt-1].astype(np.float64) if tt>=1 else conv_live[tt+2]
  sq = r3[qc0:qc0+128]*convw[qc0:qc0+128,0]+r2[qc0:qc0+128]*convw[qc0:qc0+128,1]+r1[qc0:qc0+128]*convw[qc0:qc0+128,2]+rt[qc0:qc0+128]*convw[qc0:qc0+128,3]
  sk = r3[kc0:kc0+128]*convw[kc0:kc0+128,0]+r2[kc0:kc0+128]*convw[kc0:kc0+128,1]+r1[kc0:kc0+128]*convw[kc0:kc0+128,2]+rt[kc0:kc0+128]*convw[kc0:kc0+128,3]
  sv = r3[vc0:vc0+128]*convw[vc0:vc0+128,0]+r2[vc0:vc0+128]*convw[vc0:vc0+128,1]+r1[vc0:vc0+128]*convw[vc0:vc0+128,2]+rt[vc0:vc0+128]*convw[vc0:vc0+128,3]
  sq = sq/(1+np.exp(-sq)); sk = sk/(1+np.exp(-sk)); sv = sv/(1+np.exp(-sv))
  qn = (1/max(np.sqrt((sq*sq).sum()), 1e-6))*0.08838834764831845
  kn = 1/max(np.sqrt((sk*sk).sum()), 1e-6)
  KH_[tt], QH_[tt], V_[tt] = sk*kn, sq*qn, sv
  x = float(araw[tt,h]+dtb[h]); lam[tt] = (max(x,0)+np.log1p(np.exp(-abs(x))))*ssma[h]
  bet[tt] = 1/(1+np.exp(-float(braw[tt,h])))
g2 = np.cumsum(lam*np.log2(np.e))
Bm = np.zeros((C,C))
for i in range(C):
  for j in range(i):
    Bm[i,j] = bet[i]*(KH_[i]@KH_[j])*2**(g2[i]-g2[j])
T_ = np.linalg.inv(np.eye(C)+Bm)
U_ = T_@(bet[:,None]*V_)
M_ = np.zeros((C,C))
for i in range(C):
  for j in range(i+1):
    M_[i,j] = (QH_[i]@KH_[j])*2**(g2[i]-g2[j])

hc = scr[:HC_H].astype(np.float64)
allnan = [int(np.isnan(scr[hh*HC_H:(hh+1)*HC_H]).sum()) for hh in range(48)]
print(f"[scan] per-head scratch NaN counts: nonzero heads {[(hh,n) for hh,n in enumerate(allnan) if n]}", flush=True)
def region(off, n, sh): return hc[off:off+n].reshape(sh)
kh0 = region(0, KH_C, (C, LDK))[:, :128]
kht0 = region(KH_C, KHT_C, (128, LDTC))[:, :C]
qe0 = region(KH_C+KHT_C, QE_C, (C, LDK))[:, :128]
u0 = region(KH_C+KHT_C+QE_C, U_C, (C, LDK))[:, :128]
vr0 = region(KH_C+KHT_C+QE_C+U_C, VR_C, (C, LDK))[:, :128]
m0 = region(KH_C+KHT_C+QE_C+U_C+VR_C, M_C, (C, LDTC))[:, :C]
t0 = region(KH_C+KHT_C+QE_C+U_C+VR_C+M_C, T_C, (C, LDTC))[:, :C]
u_lo0 = region(KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C, ULO_C, (C, LDK))[:, :128]
m_lo0 = region(KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C, MLO_C, (C, LDTC))[:, :C]
t_lo0 = region(KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C+MLO_C, TLO_C, (C, LDTC))[:, :C]
lo0 = hc[LO_OFF//2:LO_OFF//2+KHLO_C+QHLO_C+QELO_C+KHTLO_C]
kh_lo0 = lo0[:KHLO_C].reshape(C, LDK)[:, :128]
qh_lo0 = lo0[KHLO_C:KHLO_C+QHLO_C].reshape(C, LDK)[:, :128]
qe_lo0 = lo0[KHLO_C+QHLO_C:KHLO_C+QHLO_C+QELO_C].reshape(C, LDK)[:, :128]
kht_lo0 = lo0[KHLO_C+QHLO_C+QELO_C:].reshape(128, LDTC)[:, :C]

print(f"[f0] khat   relerr {relerr(kh0+kh_lo0, KH_)[0]:.3e} (hi-only {relerr(kh0, KH_)[0]:.3e})", flush=True)
print(f"[f0] kht^T  relerr {relerr(kht0, KH_.T)[0]:.3e} | kht_lo-consistency {relerr(kht0+kht_lo0, KH_.T)[0]:.3e}", flush=True)
print(f"[f0] qe     relerr {relerr(qe0+qe_lo0, QH_*2**g2[:,None])[0]:.3e} (hi-only {relerr(qe0, QH_*2**g2[:,None])[0]:.3e})", flush=True)
print(f"[f0] vr     relerr {relerr(vr0, V_)[0]:.3e}", flush=True)
print(f"[f0] T      relerr {relerr(t0+t_lo0, T_)[0]:.3e} (hi-only {relerr(t0, T_)[0]:.3e})", flush=True)
print(f"[f0] U      relerr {relerr(u0+u_lo0, U_)[0]:.3e} (hi-only {relerr(u0, U_)[0]:.3e})", flush=True)
print(f"[f0] M      relerr {relerr(m0+m_lo0, M_)[0]:.3e} (hi-only {relerr(m0, M_)[0]:.3e})", flush=True)
print(f"[f0] per-row T diag: " + " ".join(f"r{i}:{np.abs((t0+t_lo0)[i,:i+1]-T_[i,:i+1]).max()/max(np.abs(T_[i,:i+1]).max(),1e-9):.1e}" for i in range(0, C, max(1,C//8))), flush=True)
print(f"[f0] per-row U diag: " + " ".join(f"r{i}:{np.abs((u0+u_lo0)[i]-U_[i]).max()/max(np.abs(U_[i]).max(),1e-9):.1e}" for i in range(0, C, max(1,C//8))), flush=True)

cb = prog(f"pfcb_c{C}_nc{NC}_nw8")
cb(P.d["d_scr"], P.d["d_rec"], P.d["d_o"], global_size=(192, 1, 1), local_size=LS)
dev.synchronize()
o = P.down("d_o", (T, 6144), np.float32)
S0 = S_ref.reshape(48, 128, 128)
print(f"[pfcb] oout nan {int(np.isnan(o).sum())}/{o.size}; O relerr h0 {relerr(o[:, :128], O_ref[:, :128])[0]:.3e} F {relerr(o[:, :128], O_ref[:, :128])[1]:.3e}", flush=True)
for hh in range(4):
  print(f"[pfcb] O h{hh} med {relerr(o[:, hh*128:(hh+1)*128], O_ref[:, hh*128:(hh+1)*128])[0]:.3e}", flush=True)
rec = P.down("d_rec", (48*128*128,), np.float32)
print(f"[pfcb] rec relerr h0 {relerr(rec[:16384], S0[0].reshape(-1))[0]:.3e} F {relerr(rec[:16384], S0[0].reshape(-1))[1]:.3e}", flush=True)
cz = prog(f"pfcz_c{C}_nc{NC}_nw8")
cz(P.d["d_o"], P.d["d_gate"], P.d["wp0"].offset(offset=41056*4, size=6144*4),
   P.d["d_z"], P.d["d_qkv"], P.d["d_conv"], global_size=(48*NC*(C//8), 1, 1), local_size=LS)
dev.synchronize()
z = P.down("d_z", (T, 6144), np.float16)
print(f"[pfcz] z nan {int(np.isnan(z).sum())}/{z.size}; z med relerr (floor 0.05) {relerr(z.astype(np.float64), Z_ref.astype(np.float64), floor=0.05)[0]:.3e}", flush=True)
