# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E5 conv-halo discrimination: which conv-state seeding drives SC-vs-M32 divergence?
Variants (rec=0 zeros everywhere; conv per variant): C0 full snapshot | C1 row0 only
| C2 row1 only | C3 row2 only | C4 all rows = row0 content (kills row-order bugs)
| C5 snapshot*0.01 (magnitude test). 512 tok, SC vs M32, G3SC=0, DFILL=0."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, d = E.P, E.P.d
N = 512
import json as _j; ids = np.array(_j.load(open("~/ids8k.json"))[:N], dtype=np.int32)
GB = [E.gdn_idx[0]] + list(E.gdn_idx[-2:])
SNAPC = {i: np.load(f"~/snap100k/conv_{i}.npy").reshape(3, 10240) for i in E.gdn_idx}

def conv_of(variant, i):
    cv = SNAPC[i]
    z = np.zeros_like(cv)
    if variant == "C0": return cv.copy()
    if variant == "C1": z[0] = cv[0]; return z
    if variant == "C2": z[1] = cv[1]; return z
    if variant == "C3": z[2] = cv[2]; return z
    if variant == "C4": return np.broadcast_to(cv[0], (3, 10240)).copy()
    if variant == "C5": return cv * 0.01

def reset(variant):
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
  for i in E.gdn_idx:
    P.win_up(f"conv{i}_0", 0, conv_of(variant, i).reshape(-1))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

def snap():
  s = {}
  for i in GB:
    s[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32).copy(); P._keep.clear()
  s["logits"] = P.down("logits", (248320,), np.float16).copy(); P._keep.clear()
  return s

class G: pass
def rel(a, b):
  a = a.astype(np.float64); b = b.astype(np.float64)
  dd = np.abs(a - b); sc = np.maximum(np.abs(a), 1e-9)
  return dd.max(), float(np.median(dd / sc)), int(np.isnan(b).sum())

for variant in ["C0", "C1", "C2", "C3", "C4", "C5"]:
  os.environ["PF_SUPER"] = "1"
  reset(variant); ct = []
  pf_prefill.prefill_batch(E, G, ids, chunk_times=ct); dev.synchronize()
  sc = snap(); P._keep.clear()
  os.environ["PF_SUPER"] = "0"
  reset(variant)
  pf_prefill.prefill_batch(E, G, ids[:256]); dev.synchronize()
  pf_prefill.prefill_batch(E, G, ids[256:]); dev.synchronize()
  m32 = snap(); P._keep.clear()
  print(f"===== {variant} =====", flush=True)
  for k in [f"rec{i}" for i in GB] + ["logits"]:
    mx, md, nn = rel(m32[k], sc[k])
    print(f"{variant} {k:9s} maxabs {mx:.3e} medrel {md:.3e} nan {nn}", flush=True)
