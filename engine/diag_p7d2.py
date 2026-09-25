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
BASE = "~/tinygrad-metal/engine0"; LS = (256,1,1); CTXK = 100352
V = os.getenv("V", "a")
P = Bufs()
rng = np.random.default_rng(7)
if V == "b":  # dummy 2GB alloc first
  P.up("pad", np.zeros(2*1024*1024*256, dtype=np.uint8)); dev.synchronize(); P._keep.clear()
  print("[d] pad 2GB ok", flush=True)
P.up("qrow16", (rng.standard_normal((16, 12288)) * 0.6).astype(np.float16))
P.up("krow16", (rng.standard_normal((16, 1024)) * 0.6).astype(np.float16))
P.up("vrow16", (rng.standard_normal((16, 1024)) * 0.6).astype(np.float16))
P.up("qnw", (1.0 + rng.standard_normal(256) * 0.1).astype(np.float32))
P.up("knw", (1.0 + rng.standard_normal(256) * 0.1).astype(np.float32))
P.up("freqs", (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32))
if V == "c":  # 64-row inputs like the original harness
  P.d["qrow16"] = P.up("qrow64b", (rng.standard_normal((64, 12288)) * 0.6).astype(np.float16))
  P.d["krow16"] = P.up("krow64b", (rng.standard_normal((64, 1024)) * 0.6).astype(np.float16))
  P.d["vrow16"] = P.up("vrow64b", (rng.standard_normal((64, 1024)) * 0.6).astype(np.float16))
  P.d.pop("qrow16", None)  # keep names simple: rebind
if V in ("d", "e"):  # poisoned kv (0xAB f32-pattern) instead of zeros
  val = 0xAB if V == "d" else 0.0
  P.poison("kv", 2*4*CTXK*256, np.uint8, val); P.poison("sc", 2*4*CTXK*8, np.float16, 0.0)
else:
  P.up("kv", np.zeros(2*4*CTXK*256, dtype=np.uint8)); P.up("sc", np.zeros(2*4*CTXK*8, dtype=np.float16))
P.up("qw16", np.zeros(16*12288, dtype=np.float16))
P.up("pos0", np.zeros(1, dtype=np.int32))
dev.synchronize(); P._keep.clear()
print(f"[d] V={V} inputs synced", flush=True)
lib = open(f"{BASE}/pfk_pre16_100k.cubin", "rb").read()
kpre = NVProgram(dev, TinyELF(lib=lib, name="pfk_pre16", target=dev.renderer.target, signature=tuple()))
q = P.d["qrow64b"] if V == "c" else P.d["qrow16"]
k = P.d["krow64b"] if V == "c" else P.d["krow16"]
v = P.d["vrow64b"] if V == "c" else P.d["vrow16"]
qw = P.d["qw16"] if V != "c" else P.d["qw16"]
kpre(q, k, v, P.d["qnw"], P.d["knw"], P.d["freqs"], P.d["kv"], P.d["sc"], P.d["pos0"], qw,
     global_size=(24,1,1), local_size=LS)
dev.synchronize(); print(f"[d] V={V} kpre16 CLEAN", flush=True)
