# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Bisect the ref NaN: run _seq[0] entry-by-entry, check residual x after each block."""
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
E = TrunkEngineW1C(theta=1e7)
P, W, pr = E.P, E.W, E.pr
_lib = open(f"{BASE}/spk_c1g_100k.cubin", "rb").read()
pr["spk_c1"] = NVProgram(dev, TinyELF(lib=_lib, name="spk_c1g_100k", target=dev.renderer.target, signature=tuple()))
for i in E.gdn_idx:
  P.up(f"conv{i}_0", np.zeros(3*10240, dtype=np.float32))
  P.up(f"conv{i}_1", np.zeros(3*10240, dtype=np.float32))
  P.up(f"rec{i}", np.zeros(48*128*128, dtype=np.float32))
dev.synchronize(); P._keep.clear()
P.win_up("tok_slot", 0, np.array([100], dtype=np.int32))
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()
if not hasattr(E, "_seq"): E._build_seqs()
seq = E._seq[0]
blk = -1
for n, (p, a, g) in enumerate(seq):
  p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1), local_size=(1024,1,1) if "nw32" in getattr(p, "name", "") else LS)
  nm = getattr(p, "name", "?")
  if nm in ("down8", "k3c_down"):
    blk += 1
    dev.synchronize()
    xo = P.down("x1" if blk % 2 == 0 else "x0", (5120,), np.float32)
    fin = np.isfinite(xo).all()
    if not fin or blk < 6:
      print(f"[after blk {blk}] finite={fin} absmax={np.abs(xo).max():.3e} name={nm}", flush=True)
    if not fin:
      print(f"  FIRST NONFINITE at seq index {n} ({nm})", flush=True)
      break
dev.synchronize()
lg = P.down("logits", (248320,), np.float16)
print(f"[final] logits finite={np.isfinite(lg.astype(np.float32)).all()}", flush=True)
