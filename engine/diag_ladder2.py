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
T_BASE, TLO_BASE, META = 47616, 70144, 89088
DZ = 72704
inp = make_inputs(32, 11)
up_weights(P, inp)
P.up("d_qkv", inp["qkv"].reshape(-1)); P.up("d_ar", inp["araw"].reshape(-1)); P.up("d_br", inp["braw"].reshape(-1))
P.up("d_conv", inp["conv0"].reshape(-1)); P.up("d_rec", inp["rec0"])
P.poison("s0", 48*NC*HC, np.uint8, 0xAB)
P.poison("s1", 48*NC*HC, np.uint8, 0xAB)
dev.synchronize(); P._keep.clear()

# oracle Xf from d3 (deterministic)
ca3 = prog("pfca_c32d3")
ca3(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d["s0"],
    global_size=(48,1,1), local_size=(512,1,1))
dev.synchronize()
b0 = P.down("s0", (48*NC*HC,), np.uint8).reshape(48, HC)
d = int(sys.argv[1])
ca = prog(f"pfca_c32d{d}")
ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d["s1"],
   global_size=(48,1,1), local_size=(512,1,1))
dev.synchronize()
b1 = P.down("s1", (48*NC*HC,), np.uint8).reshape(48, HC)
ca(P.d["wp0"], P.d["d_conv"], P.d["d_qkv"], P.d["d_ar"], P.d["d_br"], P.d["s1"],
   global_size=(48,1,1), local_size=(512,1,1))
dev.synchronize()
b2 = P.down("s1", (48*NC*HC,), np.uint8).reshape(48, HC)

print(f"[d{d}] det x2: {int((b1 != b2).sum())} bytes differ (vs d3-run too: {int((b1 != b0).sum())})", flush=True)
diff = (b1 != b2)
for h in np.where(diff.any(axis=1))[0][:8]:
  cols = {}
  for bo in np.where(diff[h])[0]:
    if T_BASE <= bo < T_BASE+C*LDTC*2: nm = f"t[r{(bo-T_BASE)//(LDTC*2)}c{((bo-T_BASE)//2)%LDTC}]"
    elif TLO_BASE <= bo < TLO_BASE+C*LDTC*2: nm = f"t_lo[r{(bo-TLO_BASE)//(LDTC*2)}c{((bo-TLO_BASE)//2)%LDTC}]"
    elif META <= bo < META+(2*C+4)*4: nm = f"meta_f{(bo-META)//4}"
    else: nm = f"?{bo}"
    cols[nm] = cols.get(nm, 0)+1
  print(f"[d{d}] h{h}: " + " ".join(f"{k}x{v}" for k,v in sorted(cols.items())[:16]), flush=True)

# t_g correctness vs the Xf oracle
bad = 0; tot = 0
for h in range(48):
  xf = np.frombuffer(b0[h, DZ+8192:DZ+8192+C*36*4].tobytes(), dtype=np.float32).reshape(C,36)[:, :C]
  t_or = xf.astype(np.float16)
  t_m = b1[h, T_BASE:T_BASE+C*LDTC*2].view(np.float16).reshape(C, LDTC)[:, :C]
  bad += int((t_or != t_m).sum()); tot += C*C
  if h < 4 and (t_or != t_m).any():
    w = np.argwhere(t_or != t_m)
    print(f"[d{d}] h{h}: t_g vs Xf-oracle mismatch {len(w)} e.g. {w[:6].tolist()} mine {t_m[w[:3,0],w[:3,1]].tolist()} ref {t_or[w[:3,0],w[:3,1]].tolist()}", flush=True)
print(f"[d{d}] t_g vs Xf-oracle: {bad}/{tot} mismatched elements", flush=True)
met = b1[:, META:META+(2*C+4)*4].copy().view(np.float32).reshape(48, 2*C+4)
print(f"[d{d}] meta NaN total {int(np.isnan(met).sum())} | gend NaN heads {np.where(np.isnan(met[:,2*C]))[0].tolist()}", flush=True)
