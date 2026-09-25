# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from test_p7c import make_inputs, up_weights, prog, P, LS
from engine0 import dev

C, NC = 32, 1
LDK, LDTC = 136, C + 8
KH_C, KHT_C, QE_C, U_C, VR_C = C*LDK, 128*LDTC, C*LDK, C*LDK, C*LDK
M_C, T_C = C*LDTC, C*LDTC
ULO_C, VRLO_C, MLO_C, TLO_C = U_C, VR_C, M_C, T_C
DZ_C = 256*C
HC_HALF = 4*C*LDK + 128*LDTC + 4*C*LDTC + 2*C*LDK + DZ_C
META_OFF = (HC_HALF*2+15)//16*16
LO_OFF = (META_OFF + (2*C+4)*4 + 15)//16*16
HC_BYTES = (LO_OFF + (C*LDK*3 + 128*LDTC)*2 + 15)//16*16
KHLO_C, QHLO_C, QELO_C, KHTLO_C = C*LDK, C*LDK, C*LDK, 128*LDTC
HC_H = HC_BYTES//2
REG = [("kh",0,KH_C),("kht",KH_C,KHT_C),("qe",KH_C+KHT_C,QE_C),("u",KH_C+KHT_C+QE_C,U_C),
       ("vr",KH_C+KHT_C+QE_C+U_C,VR_C),("m",KH_C+KHT_C+QE_C+U_C+VR_C,M_C),
       ("t",KH_C+KHT_C+QE_C+U_C+VR_C+M_C,T_C),("u_lo",KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C,ULO_C),
       ("vr_lo",KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C,VRLO_C),
       ("m_lo",KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C,MLO_C),
       ("t_lo",KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C+MLO_C,TLO_C),
       ("dz",KH_C+KHT_C+QE_C+U_C+VR_C+M_C+T_C+ULO_C+VRLO_C+MLO_C+TLO_C,DZ_C),
       ("meta",META_OFF//2,(2*C+4)),("kh_lo",LO_OFF//2,KHLO_C),("qh_lo",LO_OFF//2+KHLO_C,QHLO_C),
       ("qe_lo",LO_OFF//2+KHLO_C+QHLO_C,QELO_C),("kht_lo",LO_OFF//2+KHLO_C+QHLO_C+QELO_C,KHTLO_C)]

inp = make_inputs(32, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_gate", inp["gate"].reshape(-1))
P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
for run in (1, 2):
  P.poison(f"d_scr{run}", 48*NC*HC_BYTES, np.uint8, 0xAB)
dev.synchronize(); P._keep.clear()
ca = prog("pfca_c32_nc1_nw16")
outs = []
for run in (1, 2):
  ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d[f"d_scr{run}"],
     global_size=(48*NC,1,1), local_size=(512,1,1))
  dev.synchronize()
  outs.append(P.down(f"d_scr{run}", (48*NC*HC_BYTES,), np.uint8))
d = (outs[0] != outs[1])
print(f"[det] pfca x2 byte-diff: {int(d.sum())} bytes differ", flush=True)
if d.sum():
  idx = np.where(d.reshape(48*NC, HC_BYTES).any(axis=1))[0]
  for h in idx[:12]:
    bo = np.where(d.reshape(48*NC, HC_BYTES)[h])[0]
    regs = [(nm, int(((bo>=o)&(bo<o+n)).sum())) for nm,o,n in [(nm, (o2 if isinstance(o2,int) and o2>META_OFF else o2*2)*1, n2) for nm,o2,n2 in REG]]
    print(f"[det] h{h}: {len(bo)} bytes at [{bo.min()}..{bo.max()}] regs {[(nm,c) for nm,c in regs if c]}", flush=True)
a = outs[0].view(np.float16)
tot_n = np.isnan(a).sum()
print(f"[nan] total {int(tot_n)}/{a.size}", flush=True)
nz = np.where(np.isnan(a.reshape(48*NC, HC_H)).any(axis=1))[0]
for h in nz[:14]:
  hh = np.where(np.isnan(a.reshape(48*NC, HC_H)[h]))[0]
  for e in hh[:6]:
    for nm, o, n in REG:
      if o <= e < o+n and (nm != "meta" or e < o+(2*C+4)):
        row = (e-o)//LDK if "LDK" not in nm else 0
        rw = (e-o)//LDK if nm in ("kh","qe","u","vr","u_lo","vr_lo","kh_lo","qh_lo","qe_lo") else ((e-o)//LDTC if nm in ("kht","m","t","m_lo","t_lo","kht_lo") else "-")
        cl = (e-o)%LDK if nm in ("kh","qe","u","vr","u_lo","vr_lo","kh_lo","qh_lo","qe_lo") else ((e-o)%LDTC if nm in ("kht","m","t","m_lo","t_lo","kht_lo") else "-")
        print(f"[nan] h{h} hlocal {e} region {nm} local {e-o} row {rw} col {cl}", flush=True)
        break
