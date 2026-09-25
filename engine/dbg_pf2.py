# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P2 fault bisect: run each new attention kernel alone, smallest first."""
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
LS = (256, 1, 1); LS1K = (1024, 1, 1)
M = 16; CTXK = 100352; S = 32
P = Bufs()
KSYM = {"pfk_pre16_2k": "pfk_pre16", "pfk_pre16_100k": "pfk_pre16",
        "pfa16nw32_s32_100k": "pfa16", "pfa16nw32_s8_2k": "pfa16",
        "pfc16_s32": "pfc16", "pfc16_s8": "pfc16", "pfs16": "pfs16"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))
pr = {n: prog(n) for n in ["pfk_pre16_100k", "pfa16nw32_s32_100k", "pfc16_s32"]}

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
preA = f"blk.{attn_idx[0]}."
P.up("qnw", np.frombuffer(read_raw(infos[preA+"attn_q_norm.weight"], ds), dtype="<f4"))
P.up("knw", np.frombuffer(read_raw(infos[preA+"attn_k_norm.weight"], ds), dtype="<f4"))
freqs = (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
P.up("freqs", freqs)
rng = np.random.default_rng(7)
P.up("qrow16", (rng.standard_normal((M, 12288)) * 0.35).astype(np.float16))
P.up("krow16", (rng.standard_normal((M, 1024)) * 0.35).astype(np.float16))
P.up("vrow16", (rng.standard_normal((M, 1024)) * 0.5).astype(np.float16))
P.poison("qw16", 16*24*256*2, np.float16, 7.7)
P.poison("pm16", 4*S*96*4, np.float32, 7.7e31)
P.poison("ps16", 4*S*96*4, np.float32, 7.7e31)
P.poison("pA16", 4*S*96*256*4, np.float32, 7.7e31)
P.poison("ao16", 16*6144*2, np.float16, 7.7)
P.up("pos_slot", np.array([0], dtype=np.int32))
P.up("kv", np.zeros(2*4*CTXK*256, dtype=np.uint8))
P.up("sc", np.zeros(2*4*CTXK*8, dtype=np.float16))
dev.synchronize()
d = P.d
step = sys.argv[1] if len(sys.argv) > 1 else "all"

if step in ("all", "kpre"):
  pr["pfk_pre16_100k"](d["qrow16"], d["krow16"], d["vrow16"], d["qnw"], d["knw"], d["freqs"],
                       d["kv"], d["sc"], d["pos_slot"], d["qw16"], global_size=(24,1,1), local_size=LS, wait=True)
  print("[kpre] OK", flush=True)
if step in ("all", "k1"):
  pr["pfa16nw32_s32_100k"](d["kv"], d["sc"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"],
                           global_size=(4*S,1,1), local_size=LS1K, wait=True)
  pm = P.down("pm16", (4*S*96,), np.float32)
  print(f"[k1] OK pm finite={np.isfinite(pm).all()} pm[0:4]={pm[:4]}", flush=True)
if step in ("all", "k2"):
  pr["pfc16_s32"](d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], global_size=(24,1,1), local_size=LS, wait=True)
  ao = P.down("ao16", (M, 6144), np.float16)
  print(f"[k2] OK ao finite={np.isfinite(ao.astype(np.float32)).all()}", flush=True)
print("[dbg done]", flush=True)
