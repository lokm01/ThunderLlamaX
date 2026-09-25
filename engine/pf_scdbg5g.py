# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E5 discrimination: is the nonzero-seed divergence the fp16 s16 quantization?
Variants (each: reset(seed) -> SC 512 -> reset(seed) -> M32 512 -> compare late states+logits):
  P0 snapshot seed (repro sanity)   P1 rec fp16-grid-rounded + snapshot conv
  P2 rec-only nonzero (conv zeros)  P3 conv-only nonzero (rec zeros)
If P1 clean => pure s16 fp16 quantization of the incoming state. Env as scdbg5n."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C, LS
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, d = E.P, E.P.d
N = 512
rng = np.random.default_rng(99)
import json as _j; ids = np.array(_j.load(open("~/ids8k.json"))[:N], dtype=np.int32)
GB = list(E.gdn_idx[:3]) + list(E.gdn_idx[-2:])
SNAP = {i: (np.load(f"~/snap100k/rc_{i}.npy"), np.load(f"~/snap100k/conv_{i}.npy")) for i in E.gdn_idx}

def reset(variant):
  def rec_of(i):
    rc = SNAP[i][0]
    if variant in ("P0", "P2", "P3"): return rc
    if variant == "P1": return rc.astype(np.float16).astype(np.float32)   # fp16-grid: s16 reproduces exactly
  def conv_of(i):
    cv = SNAP[i][1]
    if variant in ("P0", "P1"): return cv
    if variant in ("P2", "P3"): return np.zeros_like(cv)
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
  for i in E.gdn_idx:
    P.win_up(f"conv{i}_0", 0, conv_of(i))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, rec_of(i))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

def snap():
  s = {}
  for i in GB:
    s[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32).copy(); P._keep.clear()
    s[f"conv{i}"] = P.down(f"conv{i}_0", (3*10240,), np.float32).copy(); P._keep.clear()
  s["logits"] = P.down("logits", (248320,), np.float16).copy(); P._keep.clear()
  return s

class G: pass
def rel(a, b):
  a = a.astype(np.float64); b = b.astype(np.float64)
  dd = np.abs(a - b); sc = np.maximum(np.abs(a), 1e-9)
  return dd.max(), float(np.median(dd / sc)), int(np.isnan(b).sum())

for variant in ["P0", "P1", "P2", "P3"]:
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
  for k in [f"rec{i}" for i in GB] + [f"conv{i}" for i in GB] + ["logits"]:
    mx, md, nn = rel(m32[k], sc[k])
    print(f"{variant} {k:9s} maxabs {mx:.3e} medrel {md:.3e} nan {nn}", flush=True)
