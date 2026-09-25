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
C, HC = 32, 125712
LDK, LDTC = 136, 40
T_BASE, TLO_BASE = 47616, 70144
inp = make_inputs(32, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
for r in (1,2,3,4,5): P.poison(f"s{r}", 48*HC, np.uint8, 0xAB)
dev.synchronize(); P._keep.clear()
ca = prog("pfca_c32_nc1_nw16")
res = []
for r in (1,2,3,4,5):
  ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d[f"s{r}"],
     global_size=(48,1,1), local_size=(512,1,1))
  dev.synchronize()
  res.append(P.down(f"s{r}", (48*HC,), np.uint8))
for a in range(5):
  for b in range(a+1,5):
    d = res[a] != res[b]
    if d.sum():
      D = d.reshape(48, HC)
      hs = np.where(D.any(axis=1))[0]
      print(f"[pair {a}{b}] {int(d.sum())}B heads {hs[:6].tolist()}", flush=True)
      h = hs[0]
      bo = np.where(D[h])[0][:8]
      for x in bo:
        reg = "t" if T_BASE <= x < T_BASE+C*LDTC*2 else ("t_lo" if TLO_BASE <= x < TLO_BASE+C*LDTC*2 else str(x))
        va = res[a].reshape(48,HC)[h, x:x+2].view(np.float16)[0]
        vb = res[b].reshape(48,HC)[h, x:x+2].view(np.float16)[0]
        print(f"  h{h} {reg} byte {x}: {float(va):.6e} vs {float(vb):.6e}", flush=True)
    else:
      print(f"[pair {a}{b}] identical", flush=True)
