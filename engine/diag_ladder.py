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

C, NC, HC = 32, 1, 125712
LDK, LDTC = 136, 40
DZ_BYTE = (C*LDK + 128*LDTC + C*LDK + C*LDK + C*LDK + C*LDTC + C*LDTC + C*LDK + C*LDK + C*LDTC + C*LDTC)*2
assert DZ_BYTE == 72704, DZ_BYTE
inp = make_inputs(32, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
for run in (1, 2):
  P.poison(f"s{run}", 48*NC*HC, np.uint8, 0xAB)
dev.synchronize(); P._keep.clear()

d = int(sys.argv[1])
ca = prog(f"pfca_c32d{d}")
outs = []
for run in (1, 2):
  ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d[f"s{run}"],
     global_size=(48*NC,1,1), local_size=(512,1,1))
  dev.synchronize()
  outs.append(P.down(f"s{run}", (48*NC*HC,), np.uint8))
diff = outs[0] != outs[1]
print(f"[d{d}] det x2: {int(diff.sum())} bytes differ", flush=True)

def gz_of(h, run):
  b = outs[run].reshape(48, HC)
  g = np.frombuffer(b[h, DZ_BYTE:DZ_BYTE+4*C*4].tobytes(), dtype=np.float32).reshape(4, C)
  return g  # lam, g2, bet, bge
# numpy oracle for lam/g2/bet per head
convw = inp["convw"].reshape(10240,4); dtb = inp["dtb"]; ssma = inp["ssma"]
for h in (0, 4, 9):
  lam = (np.maximum(inp["araw"][:,h]+dtb[h],0)+np.log1p(np.exp(-np.abs(inp["araw"][:,h]+dtb[h]))))*ssma[h]
  g2 = np.cumsum(lam*np.log2(np.e)); bet = 1/(1+np.exp(-inp["braw"][:,h].astype(np.float64)))
  for run in (0,1):
    g = gz_of(h, run)
    nnan = int(np.isnan(g).sum())
    el = np.abs(g[0]-lam).max()/max(np.abs(lam).max(),1e-9)
    eg = np.abs(g[1]-g2).max()/max(np.abs(g2).max(),1e-9)
    eb = np.abs(g[2]-bet).max()
    ebge = np.abs(g[3]-bet*np.exp2(g2)).max()/max(np.abs(bet*np.exp2(g2)).max(),1e-9)
    print(f"[d{d}] h{h} run{run}: gz nan {nnan} | lam maxrel {el:.2e} g2 maxrel {eg:.2e} bet maxdiff {eb:.2e} bge maxrel {ebge:.2e}", flush=True)
    if nnan:
      w = np.where(np.isnan(g))
      print(f"[d{d}] h{h} run{run}: nan at rows {g_names if False else w[0][:6]} cols {w[1][:6]}", flush=True)
if d >= 2:
  for h in (0, 4, 9):
    for run in (0,1):
      b = outs[run].reshape(48, HC)
      tf = np.frombuffer(b[h, DZ_BYTE+1024:DZ_BYTE+1024+C*36*4].tobytes(), dtype=np.float32).reshape(C,36)
      print(f"[d{d}] h{h} run{run}: Tf nan {int(np.isnan(tf).sum())}/{C*C} absmax {np.nanmax(np.abs(tf)):.3e}", flush=True)
if d >= 3:
  for h in (0, 4, 9):
    for run in (0,1):
      b = outs[run].reshape(48, HC)
      xf = np.frombuffer(b[h, DZ_BYTE+8192:DZ_BYTE+8192+C*36*4].tobytes(), dtype=np.float32).reshape(C,36)
      print(f"[d{d}] h{h} run{run}: Xf nan {int(np.isnan(xf).sum())}/{C*C} absmax {np.nanmax(np.abs(xf)):.3e}", flush=True)
