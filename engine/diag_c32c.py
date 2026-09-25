# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from test_p7c import make_inputs, up_weights, prog, P
from engine0 import dev

C, NC = (int(sys.argv[1]) if len(sys.argv) > 1 else 32), 1
LDK, LDTC = 136, C + 8
# byte-offset region map (hi-lo layout)
R = []
def add(nm, off, sz, roww=None): R.append((nm, off, sz, roww))
add("kh", 0, C*LDK*2, LDK); add("kht", C*LDK*2, 128*LDTC*2, LDTC)
add("qe", C*LDK*2+128*LDTC*2, C*LDK*2, LDK)
u0 = C*LDK*2+128*LDTC*2+C*LDK*2
add("u", u0, C*LDK*2, LDK); add("vr", u0+C*LDK*2, C*LDK*2, LDK)
add("m", u0+2*C*LDK*2, C*LDTC*2, LDTC); add("t", u0+2*C*LDK*2+C*LDTC*2, C*LDTC*2, LDTC)
add("u_lo", u0+2*C*LDK*2+2*C*LDTC*2, C*LDK*2, LDK)
add("vr_lo", u0+3*C*LDK*2+2*C*LDTC*2, C*LDK*2, LDK)
add("m_lo", u0+4*C*LDK*2+2*C*LDTC*2, C*LDTC*2, LDTC)
add("t_lo", u0+4*C*LDK*2+3*C*LDTC*2, C*LDTC*2, LDTC)
HC_HALF = 4*C*LDK + 128*LDTC + 4*C*LDTC + 2*C*LDK + 256*C
META_OFF = (HC_HALF*2+15)//16*16
LO_OFF = (META_OFF + (2*C+4)*4 + 15)//16*16
add("dz", u0+4*C*LDK*2+4*C*LDTC*2, META_OFF-(u0+4*C*LDK*2+4*C*LDTC*2), 2*C)
add("meta", META_OFF, (2*C+4)*4, 1)
add("kh_lo", LO_OFF, C*LDK*2, LDK); add("qh_lo", LO_OFF+C*LDK*2, C*LDK*2, LDK)
add("qe_lo", LO_OFF+2*C*LDK*2, C*LDK*2, LDK)
add("kht_lo", LO_OFF+3*C*LDK*2, 128*LDTC*2, LDTC)
HC_BYTES = (LO_OFF + (3*C*LDK+128*LDTC)*2 + 15)//16*16
assert HC_BYTES == {64: 263696, 32: 125712}[C]

inp = make_inputs(C*NC, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_gate", inp["gate"].reshape(-1))
P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
for run in (1, 2, 3):
  P.poison(f"s{run}", 48*NC*HC_BYTES, np.uint8, 0xAB)
dev.synchronize(); P._keep.clear()
ca = prog(f"pfca_c{C}_nc1_nw16")
outs = []
for run in (1, 2, 3):
  ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d[f"s{run}"],
     global_size=(48*NC,1,1), local_size=(512,1,1))
  dev.synchronize()
  outs.append(P.down(f"s{run}", (48*NC*HC_BYTES,), np.uint8))

def classify(h, byte_off):
  for nm, o, sz, rw in R:
    if o <= byte_off < o+sz:
      loc = byte_off - o
      if rw: return f"{nm}[row {loc//(rw*2)} col {(loc//2)%rw}]"
      return f"{nm}[+{loc}]"
  return f"?[{byte_off}]"

for pair in ((0,1),(0,2)):
  d = outs[pair[0]] != outs[pair[1]]
  print(f"[det {pair}] pfca byte-diff total {int(d.sum())}", flush=True)
  D = d.reshape(48*NC, HC_BYTES)
  for h in np.where(D.any(axis=1))[0][:6]:
    bo = np.where(D[h])[0]
    cls = {}
    for b in bo: cls[classify(h, int(b))] = cls.get(classify(h, int(b)), 0) + 1
    print(f"[det {pair}] h{h}: {len(bo)}B " + " ".join(f"{k}x{v}" for k,v in sorted(cls.items())[:14]), flush=True)
  if d.sum() == 0: break
# NaN positions
a = outs[0].view(np.float16)
A = a.reshape(48*NC, HC_BYTES//2)
for h in np.where(np.isnan(A).any(axis=1))[0][:8]:
  e = np.where(np.isnan(A[h]))[0]
  for x in e[:8]: print(f"[nan] h{h} half {x} byte {x*2} -> {classify(h, x*2)}", flush=True)
