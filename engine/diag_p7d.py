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
P = Bufs()
rng = np.random.default_rng(7)
P.up("qrow64", (rng.standard_normal((64, 12288)) * 0.6).astype(np.float16))
P.up("krow64", (rng.standard_normal((64, 1024)) * 0.6).astype(np.float16))
P.up("vrow64", (rng.standard_normal((64, 1024)) * 0.6).astype(np.float16))
P.up("qnw", (1.0 + rng.standard_normal(256) * 0.1).astype(np.float32))
P.up("knw", (1.0 + rng.standard_normal(256) * 0.1).astype(np.float32))
P.up("freqs", np.exp(-np.arange(32, dtype=np.float64) / 32.0 * np.log(1e7)).astype(np.float32))
P.poison("kva", 2*4*CTXK*256, np.uint8, 0xAB)
P.poison("sca", 2*4*CTXK*8, np.float16, 0.0)
P.poison("qw64a", 64*12288*2, np.float16, 7.7)
P.up("pos0", np.zeros(1, dtype=np.int32))
dev.synchronize(); P._keep.clear()
print("[d] inputs synced clean", flush=True)
lib = open(f"{BASE}/pfk_pre16_100k.cubin", "rb").read()
kpre = NVProgram(dev, TinyELF(lib=lib, name="pfk_pre16", target=dev.renderer.target, signature=tuple()))
dev.synchronize(); print("[d] kpre16 loaded", flush=True)
kpre(P.d["qrow64"], P.d["krow64"], P.d["vrow64"], P.d["qnw"], P.d["knw"], P.d["freqs"],
     P.d["kva"], P.d["sca"], P.d["pos0"], P.d["qw64a"], global_size=(24,1,1), local_size=LS)
dev.synchronize(); print("[d] kpre16 pos0 CLEAN", flush=True)
kpre(P.d["qrow64"].offset(offset=16*12288*2, size=16*12288*2),
     P.d["krow64"].offset(offset=16*1024*2, size=16*1024*2),
     P.d["vrow64"].offset(offset=16*1024*2, size=16*1024*2),
     P.d["qnw"], P.d["knw"], P.d["freqs"], P.d["kva"], P.d["sca"], P.d["pos0"],
     P.d["qw64a"].offset(offset=16*12288*2, size=16*12288*2), global_size=(24,1,1), local_size=LS)
dev.synchronize(); print("[d] kpre16 pos0 OFFSET-VIEW CLEAN", flush=True)
