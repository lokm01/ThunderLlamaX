# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad import dtypes
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import GDNBlockEngine
dev = Device["NV"]
e = GDNBlockEngine(0)
rng = np.random.default_rng(3)
e.set_inputs(x=(rng.standard_normal(5120)*0.2).astype(np.float32))
dev.synchronize()
e.launch_one("k0_norm", wait=True)
lib = open("~/tinygrad-metal/engine0/k1_split.cubin","rb").read()
def mk(n): return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
P, W = e.P.d, e.w
LS = (256,1,1)
order = sys.argv[1:]
fams = {"q5": (lambda: mk("k1_q5")(W["qkv"], P["xh"], P["qkv_row"], global_size=(1280,1,1), local_size=LS, wait=True)),
        "iq3": (lambda: mk("k1_iq3")(W["gate"], P["gridf"], P["xh"], P["gate_row"], global_size=(768,1,1), local_size=LS, wait=True)),
        "ab":  (lambda: mk("k1_ab")(P["w_alpha"], P["w_beta"], P["xh"], P["alpharaw"], P["betaraw"], global_size=(12,1,1), local_size=LS, wait=True))}
for f in (order or ["ab","q5","iq3"]):
  print("[dbg] running", f, flush=True)
  fams[f]()
  print("[dbg]", f, "OK", flush=True)
print("[dbg done]", flush=True)
