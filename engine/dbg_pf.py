# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Minimal pGEMM single-launch debugger: one class, one kernel, one synced
launch + poison readback. Usage: dbg_pf.py <class> <kernelname>"""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from trunk import iq3s_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
cls, kname = sys.argv[1], sys.argv[2]
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
G0 = gdn_idx[0]
P.up("gridf", iq3_grid_f32())
P.up("grid512", iq3s_grid_f32())

CFG = {  # class -> (K, N, weightloader)
  "iq3g": (5120, 6144, lambda: np.load(f"{PACKED}/gate{G0}.npy")),
  "iq3d": (17408, 5120, lambda: np.load(f"{PACKED}/fd{G0}.npy")),
  "iq3n": (5120, 17408, lambda: np.load(f"{PACKED}/fg{G0}.npy")),
  "q5kv": (5120, 10240, lambda: np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8)),
}
K, N, wl = CFG[cls]
W = wl()
print(f"[dbg] {cls}: K={K} N={N} wbytes={W.nbytes/1e6:.1f}MB", flush=True)
P.up("w", np.ascontiguousarray(W)); dev.synchronize(); del W
rng = np.random.default_rng(3)
x = (rng.standard_normal((16, K)) * 0.8).astype(np.float16)
P.up("x16buf", x.reshape(-1))
P.poison("out16", 16*N*2, np.float16, 7.7)
dev.synchronize()
args = (P.d["w"], P.d["gridf"], P.d["x16buf"], P.d["out16"])
grid = N // 64
print(f"[dbg] launching {kname} grid={grid} ls=256", flush=True)
pr = prog(kname)
pr(*args, global_size=(grid, 1, 1), local_size=(256, 1, 1))
dev.synchronize()
out = P.down("out16", (16, N), np.float16).astype(np.float32)
nz = int((np.abs(out) < 1e30).sum())
act = int((np.abs(out) > 1e-3).sum())
print(f"[dbg] OK: finite={nz}/{out.size} active={act} sample={out[0,:4]}", flush=True)
