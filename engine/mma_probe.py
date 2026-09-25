# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
P = Bufs()
uid = np.arange(256, dtype=np.float32).reshape(16,16).astype(np.float16)   # A uid: value = m*16+k
I = np.eye(16, 8, dtype=np.float32).astype(np.float16)                     # k==n
I8 = np.eye(16, 8, k=-8, dtype=np.float32).astype(np.float16)              # k==n+8
buid = np.arange(128, dtype=np.float32).reshape(16,8).astype(np.float16)   # B uid: value = k*8+n
ai = (np.eye(16,16,dtype=np.float32)).astype(np.float16)                   # A k==m
ai8 = (np.eye(16,16,k=-8,dtype=np.float32)).astype(np.float16)             # A k==m+8
lib = open(f"{BASE}/mma_probe.cubin","rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="mma_probe", target=dev.renderer.target, signature=tuple()))
for mode, (A, B) in enumerate([(uid, I), (uid, I8), (ai, buid), (ai8, buid)]):
  P.up("A", A.reshape(-1)); P.up("B", B.reshape(-1)); P.poison("D", 128*4, np.float32, 7.7e31); dev.synchronize()
  pr(P.d["A"], P.d["B"], P.d["D"], P.d["mm"], global_size=(1,1,1), local_size=(32,1,1)) if False else None
  import struct
  P.up("mm", np.array([mode], dtype=np.int32)); dev.synchronize()
  pr(P.d["A"], P.d["B"], P.d["D"], P.d["mm"], global_size=(1,1,1), local_size=(32,1,1)); dev.synchronize()
  D = P.down("D", (16,8), np.float32)
  print(f"--- mode {mode} ---")
  for r in range(16): print(" ".join(f"{int(v):4d}" for v in D[r]))
