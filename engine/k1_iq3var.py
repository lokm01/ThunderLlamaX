# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad import dtypes
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
import engine0
from engine0 import GDNBlockEngine
dev = Device["NV"]
e = GDNBlockEngine(0)
rng = np.random.default_rng(3)
e.set_inputs(x=(rng.standard_normal(5120)*0.2).astype(np.float32))
dev.synchronize()
e.launch_one("k0_norm", wait=True)
lib = open("~/tinygrad-metal/engine0/k1_iq3var.cubin","rb").read()
def mk(n): return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
P, W = e.P.d, e.w
LS = (256,1,1)
for v in (sys.argv[1:] or ["v1","v2"]):
    t0 = time.perf_counter()
    mk(f"k1_iq3_{v}")(W["gate"], P["gridf"], P["xh"], P["gate_row"], global_size=(768,1,1), local_size=LS, wait=True)
    g = e.P.down("gate_row", (6144,), np.float16).astype(np.float32)
    print(f"[dbg] iq3_{v} OK {(time.perf_counter()-t0)*1e3:.1f} ms finite={np.isfinite(g).all()} |g|max={np.abs(g).max():.2f}", flush=True)
print("[dbg done]", flush=True)
