# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-C diag2: pfcb in isolation with synthetic scratch (h=0 only)."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from test_p7c import relerr, prog, P, HC_BYTES
from engine0 import dev

C, NC = 64, 1
LDK, LDTC = 136, 72
KH_C, KHT_C, QE_C, U_C, VR_C, M_C = C*LDK, 128*LDTC, C*LDK, C*LDK, C*LDK, C*LDTC
HC_HALF = 4*C*LDK + 128*LDTC + 2*C*LDTC
META_OFF = (HC_HALF*2 + 15)//16*16
rng = np.random.default_rng(5)
scr = np.full((48 * HC_BYTES[C] // 2,), 0, dtype=np.float16)
def putp(off, arr, ld):  # padded-stride put: arr [rows][cols] -> stride ld
  z = np.zeros((arr.shape[0], ld), np.float16); z[:, :arr.shape[1]] = arr
  scr[off:off+z.size] = z.reshape(-1)
hc0 = 0
kh = (rng.standard_normal((C,128))*0.09).astype(np.float16)
kht = kh.T.copy()
qe = (rng.standard_normal((C,128))*0.09).astype(np.float16)
u = (rng.standard_normal((C,128))*0.2).astype(np.float16)
m = np.tril(rng.standard_normal((C,C))*0.15).astype(np.float16)
t = np.eye(C, dtype=np.float16)
putp(hc0, kh, LDK); putp(hc0+KH_C, kht, LDTC); putp(hc0+KH_C+KHT_C, qe, LDK); putp(hc0+KH_C+KHT_C+QE_C, u, LDK)
putp(hc0+KH_C+KHT_C+QE_C+U_C, np.zeros((C,128), np.float16), LDK)
putp(hc0+KH_C+KHT_C+QE_C+U_C+VR_C, m, LDTC); putp(hc0+KH_C+KHT_C+QE_C+U_C+VR_C+M_C, t, LDTC)
meta = np.zeros((2*C+4)*2, dtype=np.float16).view(np.float32)
meta[:C] = 1.0; meta[C:2*C] = 1.0; meta[2*C] = 0.0
scr[hc0 + META_OFF//2:hc0 + META_OFF//2 + (2*C+4)*2] = meta.view(np.float16)
rec0 = (rng.standard_normal((48,128,128))*0.3).astype(np.float32)
P.up("s_scr", scr); P.up("s_rec", rec0.reshape(-1)); P.poison("s_o", C*6144*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()
prog("pfcb_c64_nc1_nw8")(P.d["s_scr"], P.d["s_rec"], P.d["s_o"], global_size=(192,1,1), local_size=(256,1,1))
dev.synchronize()
o = P.down("s_o", (C, 6144), np.float32).reshape(C,48,128)
rec = P.down("s_rec", (48,128,128), np.float32)
S = rec0[0].astype(np.float64).T  # [k][v]
khf, qef, uf, mf = kh.astype(np.float64), qe.astype(np.float64), u.astype(np.float64), m.astype(np.float64)
Y = khf @ S
d = uf - Y
Opred = mf @ d + qef @ S
Spred = (S + khf.T @ d).T
print("O relerr:", relerr(o[:,0,:], Opred))
print("rec relerr:", relerr(rec[0], Spred))
om = o[:,0,:]
print("om[0,:3]", om[0,:3], "Opred[0,:3]", Opred[0,:3])
print("om[1,:3]", om[1,:3], "Opred[1,:3]", Opred[1,:3])
# error structure
E = om - Opred
print("maxerr row", np.abs(E).max(axis=1)[:8], "maxerr col", np.abs(E).max(axis=0)[:8])
# hypothesis: O = M@d + Qe@S vs O = M@d + Qe@S with qe as [v][k] read wrong?
print("E vs M@d-only:", relerr(om, mf@d)[0], "E vs Qe@S-only:", relerr(om, qef@S)[0])
print("E vs d+QeS:", relerr(om, d + qef@S)[0])

# permutation forensics
On = (om - om.mean(axis=1, keepdims=True)) / (np.linalg.norm(om, axis=1, keepdims=True) + 1e-12)
Pn = (Opred - Opred.mean(axis=1, keepdims=True)) / (np.linalg.norm(Opred, axis=1, keepdims=True) + 1e-12)
Mm = np.abs(On @ Pn.T)
match = Mm.argmax(axis=1)
print("row-match om[i] -> Opred[match[i]]:", match[:16])
print("row-match quality:", Mm.max(axis=1)[:8])
En = rec[0].astype(np.float64) - Spred
print("rec err by vq-slice (v%32//32):", [float(np.linalg.norm(En[:, (vq*32):(vq*32+32)])) for vq in range(4)])
print("rec err by k-halves:", [float(np.linalg.norm(En[(kk*64):(kk*64+64), :])) for kk in range(2)])
print("S sum sanity: rec0[0].sum() %.3f kernel %.3f pred %.3f" % (rec0[0].sum(), rec[0].sum(), Spred.sum()))
# does om row1 match Opred with d/u swapped...?
print("om[1,:3] vs d[1,:3]:", om[1,:3], d[1,:3])
print("om[1,:3] vs Y[1,:3]:", om[1,:3], Y[1,:3])
