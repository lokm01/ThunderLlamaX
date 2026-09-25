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
T_BASE, TLO_BASE = 47616, 70144
inp = make_inputs(32, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
for r in (1,2,3): P.poison(f"s{r}", 48*HC, np.uint8, 0xAB)
dev.synchronize(); P._keep.clear()
ca = prog("pfca_c32_nc1_nw16")
ts = []
for r in (1,2,3):
  ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d[f"s{r}"],
     global_size=(48,1,1), local_size=(512,1,1))
  dev.synchronize()
  b = P.down(f"s{r}", (48*HC,), np.uint8).reshape(48, HC)
  ts.append(b[0, T_BASE:T_BASE+C*LDTC*2].view(np.float16).reshape(C, LDTC)[:, :C].astype(np.float32))
d01 = ts[0] != ts[1]
print(f"[t] h0 differ {int(d01.sum())} elements", flush=True)
w = np.argwhere(d01)
for r in range(3):
  print(f"[t] run{r}: " + " ".join(f"({a},{b})={ts[r][a,b]:.6e}" for a,b in w[:8]), flush=True)
# also: are the differing values CLOSE to each other (ulp) or wildly different?
rel = [abs(ts[0][a,b]-ts[1][a,b])/max(abs(ts[1][a,b]),1e-9) for a,b in w[:20]]
print("[t] rel diffs:", [format(x, ".1e") for x in rel], flush=True)
