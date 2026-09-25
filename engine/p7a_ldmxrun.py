# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-A ldmatrix probe, isolated. mode arg: 1 = LDS.32x4 only, 0 = ldmatrix."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

dev = Device["NV"]; P = Bufs(); rng = np.random.default_rng(11)
INT = (None, 4, dtypes.int32, ())
NTHR = 256
MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
src = rng.integers(0, 2**32, 128, dtype=np.uint64).astype(np.uint32)
P.up("src", src); P.up("out4", np.zeros(NTHR*4, np.uint32)); P.up("outacc", np.zeros(NTHR, np.uint32))
dev.synchronize()
lib = open("~/tinygrad-metal/engine0/p7a_ldmx_nw8.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="p7a_ldmx_nw8", target=dev.renderer.target, signature=(INT, INT)))
tag = "ldmatrix_x4" if MODE == 0 else "lds32_x4"
print(f"lmx: launching mode {MODE} ({tag})", flush=True)
pr(P.d["src"], P.d["out4"], P.d["outacc"], global_size=(82,1,1), local_size=(NTHR,1,1), vals=(1, MODE))
dev.synchronize()
print(f"lmx: mode {MODE} LAUNCH CLEAN", flush=True)
o4 = P.down("out4", (NTHR, 4), np.uint32); oa = P.down("outacc", (NTHR,), np.uint32)
def exp_jL(j, L): return int(src[(j*128 + (L >> 2)*16 + (L & 3)*4)//4])
ok = all(int(o4[w*32+L, j]) == exp_jL(j, L) for L in range(32) for j in range(4) for w in range(8))
oka = all(int(oa[w*32+L]) == (sum(exp_jL(j, L) for j in range(4)) & 0xFFFFFFFF)
          for L in range(32) for w in range(8))
print(f"lmx: {tag} roundtrip frags={'PASS' if ok else 'FAIL'} acc={'PASS' if oka else 'FAIL'}", flush=True)
t0 = time.perf_counter()
pr(P.d["src"], P.d["out4"], P.d["outacc"], global_size=(82,1,1), local_size=(NTHR,1,1), vals=(100000, MODE))
dev.synchronize()
t = time.perf_counter() - t0
print(f"lmx: {tag} bench {t*1e3:.3f} ms -> {t*1e9/100000:.3f} ns/warp-iter, "
      f"{82*8*100000*512/t/1e9:.1f} GB/s smem aggregate", flush=True)
