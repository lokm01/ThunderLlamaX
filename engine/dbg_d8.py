# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""down8 fault bisect: d8a (orig byte-q) vs d8b (u16-q, unroll 1) vs down8."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, parse_gguf, read_raw, iq3_grid_f32, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
which = sys.argv[1] if len(sys.argv) > 1 else "d8a"
P = Bufs(); ds, infos = parse_gguf()
P.up("gridf", iq3_grid_f32())
W = P.up("w_fd", np.frombuffer(read_raw(infos["blk.0.ffn_down.weight"], ds), dtype=np.uint8))
P.poison("gact", 17408*2, np.float16, 7.7)
P.up("gact", (np.random.default_rng(11).standard_normal(17408)*0.2).astype(np.float16))
P.poison("y", 5120*4, np.float32, 7.7e31)
P.up("hh", (np.random.default_rng(5).standard_normal(5120)*0.4).astype(np.float32))
P.poison("hh", 5120*4, np.float32, 7.7e31)
P.up("hh", (np.random.default_rng(5).standard_normal(5120)*0.4).astype(np.float32))
dev.synchronize()
lib = open(f"{BASE}/{which}.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name=which, target=dev.renderer.target, signature=tuple()))
print(f"[run] {which} launching...", flush=True)
pr(W, P.d["gridf"], P.d["gact"], P.d["hh"], P.d["y"], global_size=(640,1,1), local_size=LS, wait=True)
got = P.down("y", (5120,))
print(f"[run] {which} DONE |y|max={np.abs(got).max():.3f} finite={np.isfinite(got).all()}", flush=True)
