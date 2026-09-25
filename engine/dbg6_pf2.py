# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Isolate: does the canonical T=1 KV8+QH trunk NaN at pos 0..15 from zero state?"""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
import trunk_w1c
from trunk_w1c import TrunkEngineW1C, LS
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
M = 16
E = TrunkEngineW1C(theta=1e7)
P, W, pr = E.P, E.W, E.pr
_lib = open(f"{BASE}/spk_c1g_100k.cubin", "rb").read()
pr["spk_c1"] = NVProgram(dev, TinyELF(lib=_lib, name="spk_c1g_100k", target=dev.renderer.target, signature=tuple()))
for i in E.gdn_idx:
  P.up(f"conv{i}_0", np.zeros(3*10240, dtype=np.float32))
  P.up(f"conv{i}_1", np.zeros(3*10240, dtype=np.float32))
  P.up(f"rec{i}", np.zeros(48*128*128, dtype=np.float32))
dev.synchronize(); P._keep.clear()
ids = np.arange(100, 100 + M).astype(np.int32)
for t in range(M):
  P.win_up("tok_slot", 0, np.array([int(ids[t])], dtype=np.int32))
  E.token(t, wait=True)
  lg = P.down("logits", (248320,), np.float16).astype(np.float32)
  x0 = P.down("x0", (5120,), np.float32)
  print(f"[tok {t}] logits finite={np.isfinite(lg).all()} max={lg.max():.2e} | x0 finite={np.isfinite(x0).all()} max={np.abs(x0).max():.2e}", flush=True)
for i in E.gdn_idx[:6]:
  r = P.down(f"rec{i}", (48*128*128,), np.float32)
  print(f"[rec blk {i}] finite={np.isfinite(r).all()}", flush=True)
print("[dbg6 done]", flush=True)
