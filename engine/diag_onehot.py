# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-C diag3: pfcb one-hot probe. kh[i] = 0.5*e_(i%128); qe=0; u=0; m=I; t=I;
bg=sg=1; gend=0. Expect: O[i] = -0.5*S[i%128][vq-slice], S' = 0.75*S."""
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
rng = np.random.default_rng(9)
scr = np.zeros((48 * HC_BYTES[C] // 2,), dtype=np.float16)
def put(off, arr): scr[off:off+arr.size] = arr.reshape(-1)
kh = np.zeros((C,128), np.float16)
for i in range(C): kh[i, i] = 0.5
kht = kh.T.copy()
meta = np.zeros((2*C+4)*2, dtype=np.float16).view(np.float32)
meta[:C] = 1.0; meta[C:2*C] = 1.0; meta[2*C] = 0.0
put(0, kh); put(KH_C, kht); put(KH_C+KHT_C, np.zeros((C,128), np.float16))
put(KH_C+KHT_C+QE_C, np.zeros((C,128), np.float16))
put(KH_C+KHT_C+QE_C+U_C, np.zeros((C,128), np.float16))
put(KH_C+KHT_C+QE_C+U_C+VR_C, np.eye(C, dtype=np.float16))
put(KH_C+KHT_C+QE_C+U_C+VR_C+M_C, np.eye(C, dtype=np.float16))
put(META_OFF//2, meta.view(np.float16))
rec0 = (rng.standard_normal((48,128,128))*0.3).astype(np.float32)
P.up("s_scr", scr); P.up("s_rec", rec0.reshape(-1)); P.poison("s_o", C*6144*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()
prog("pfcb_c64_nc1_nw8")(P.d["s_scr"], P.d["s_rec"], P.d["s_o"], global_size=(192,1,1), local_size=(256,1,1))
dev.synchronize()
o = P.down("s_o", (C, 6144), np.float32).reshape(C,48,128)
rec = P.down("s_rec", (48,128,128), np.float32)
S = rec0[0].astype(np.float64)   # [v][k]
Opred = -0.5 * S[:, :C].T        # [i][v] = -0.5 S[k=i][v]
om = o[:,0,:]
print("O relerr:", relerr(om, Opred))
# which S row does each om row equal?
Sc = S.T  # [k][v]
for i in range(6):
  c = np.abs(np.corrcoef(np.vstack([om[i], Sc[i]]))[0,1])
  print(f"om[{i}] corr with S[{i}]: {c:.3f} | om[{i}][:3] {om[i][:3]} vs -0.5S[{i}][:3] {-0.5*Sc[i][:3]}")
# find actual match
On = om / (np.linalg.norm(om, axis=1, keepdims=True)+1e-12)
Sn = Sc / (np.linalg.norm(Sc, axis=1, keepdims=True)+1e-12)
mm = np.abs(On @ Sn.T).argmax(axis=1)
print("om[i] -> S-row match:", mm[:24])
print("rec: kernel 0.75*sum %.3f vs actual %.3f (rec0 %.3f)" % (0.75*rec0[0].sum(), rec[0].sum(), rec0[0].sum()))
E = rec[0].astype(np.float64) - 0.75*rec0[0].astype(np.float64)
print("rec relerr:", relerr(rec[0], 0.75*rec0[0]))
print("rec err by k-quarters:", [float(np.linalg.norm(E[(kk*32):(kk*32+32),:])) for kk in range(4)])
