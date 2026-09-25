# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Numpy ground truth for the M=16 attention at pos=0; compare mine + T=1 ref."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1); LS1K = (1024, 1, 1)
M = 16; CTXK = 100352; S = 32; SR = 256
P = Bufs()
KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))
pr = {n: prog(n) for n in ["pfk_pre16_100k", "pfa16nw32_s32_100k", "pfc16_s32",
                            "spk_pre1qh_100k", "spk_g4nw32qh1_100k", "spk_c1g_100k"]}
ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
preA = f"blk.{attn_idx[0]}."
qnw = np.frombuffer(read_raw(infos[preA+"attn_q_norm.weight"], ds), dtype="<f4")
knw = np.frombuffer(read_raw(infos[preA+"attn_k_norm.weight"], ds), dtype="<f4")
P.up("qnw", qnw); P.up("knw", knw)
P.up("freqs", (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32))
rng = np.random.default_rng(21)
qrow16 = (rng.standard_normal((M, 12288)) * 0.35).astype(np.float16)
krow16 = (rng.standard_normal((M, 1024)) * 0.35).astype(np.float16)
vrow16 = (rng.standard_normal((M, 1024)) * 0.5).astype(np.float16)
P.up("qrow16", qrow16); P.up("krow16", krow16); P.up("vrow16", vrow16)
P.poison("qw16", 16*24*256*2, np.float16, 7.7)
P.poison("pm16", 4*S*96*4, np.float32, 7.7e31); P.poison("ps16", 4*S*96*4, np.float32, 7.7e31)
P.poison("pA16", 4*S*96*256*4, np.float32, 7.7e31)
P.poison("ao16", 16*6144*2, np.float16, 7.7)
P.poison("qw_t1", 24*256*4, np.float32, 7.7e31); P.poison("qw16_t1", 24*256*2, np.float16, 7.7)
P.poison("pm_t1", 4*SR*6*4, np.float32, 7.7e31); P.poison("ps_t1", 4*SR*6*4, np.float32, 7.7e31)
P.poison("pA_t1", 4*SR*6*256*4, np.float32, 7.7e31)
P.poison("ao_t1", 6144*2, np.float16, 7.7)
P.up("pos_slot", np.array([0], dtype=np.int32))
P.up("kv", np.zeros(2*4*CTXK*256, dtype=np.uint8))
P.up("sc", np.zeros(2*4*CTXK*8, dtype=np.float16))
dev.synchronize()
d = P.d

# ---- mine ----
pr["pfk_pre16_100k"](d["qrow16"], d["krow16"], d["vrow16"], d["qnw"], d["knw"], d["freqs"],
                     d["kv"], d["sc"], d["pos_slot"], d["qw16"], global_size=(24,1,1), local_size=LS)
pr["pfa16nw32_s32_100k"](d["kv"], d["sc"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"],
                         global_size=(4*S,1,1), local_size=LS1K)
pr["pfc16_s32"](d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], global_size=(24,1,1), local_size=LS, wait=True)
ao_mine = P.down("ao16", (M, 6144), np.float16).astype(np.float32)
kv_bytes = P.down("kv", (2*4*CTXK*256,), np.uint8).copy()
sc_h = P.down("sc", (2*4*CTXK*8,), np.float16).astype(np.float32)
qw16 = P.down("qw16", (M, 24*256), np.float16).astype(np.float32).reshape(-1)

# ---- numpy ground truth from the BIT-EXACT kv + qw16 ----
Kq = np.zeros((4, 16, 256), dtype=np.float32); Vq = np.zeros((4, 16, 256), dtype=np.float32)
for g in range(4):
  for l in range(16):
    for c in range(8):
      sidx = slice(l*256 + c*32, l*256 + c*32 + 32)
      Kq[g, l, c*32:(c+1)*32] = ((kv_bytes[g*CTXK*256 + l*256 + c*32: g*CTXK*256 + l*256 + c*32 + 32].astype(np.int32) - 128).astype(np.float32) * sc_h[g*CTXK*8 + l*8 + c])
      Vq[g, l, c*32:(c+1)*32] = ((kv_bytes[(4+g)*CTXK*256 + l*256 + c*32: (4+g)*CTXK*256 + l*256 + c*32 + 32].astype(np.int32) - 128).astype(np.float32) * sc_h[(4+g)*CTXK*8 + l*8 + c])
ao_np = np.zeros((M, 6144), dtype=np.float32)
for h in range(24):
  g, hl = h // 6, h % 6
  for t in range(M):
    q = qw16[t*6144 + h*256: t*6144 + h*256 + 256]
    sc_ = np.array([np.dot(q, Kq[g, l]) for l in range(t + 1)], dtype=np.float32)
    p = np.exp(sc_ - sc_.max()); p /= p.sum()
    out = p @ Vq[g, :t+1]
    gf = qrow16[t, h*512 + 256:h*512 + 512].astype(np.float32)
    ao_np[t, h*256:(h+1)*256] = out / (1.0 + np.exp(-gf))

def rel(m, r):
  m = np.asarray(m, np.float32); r = np.asarray(r, np.float32)
  act = np.abs(r) > 1e-6
  e = np.abs(m[act] - r[act]) / np.abs(r[act])
  return f"max {e.max():.3e} med {np.median(e):.3e} F {np.linalg.norm(m-r)/np.linalg.norm(r):.3e}"
print(f"[mine vs numpy]  {rel(ao_mine, ao_np)}", flush=True)

# ---- T=1 reference (kv restored to pre-chunk state = all zeros rows>=16; rows 0..15 identical) ----
out_t1 = np.zeros((M, 6144), dtype=np.float32)
for t in range(M):
  P.win_up("pos_slot", 0, np.array([t], dtype=np.int32))
  qr = d["qrow16"].offset(offset=t*12288*2, size=12288*2)
  kr = d["krow16"].offset(offset=t*1024*2, size=1024*2)
  vr = d["vrow16"].offset(offset=t*1024*2, size=1024*2)
  pr["spk_pre1qh_100k"](qr, kr, vr, d["qnw"], d["knw"], d["freqs"], d["kv"], d["sc"], d["pos_slot"],
                        d["qw_t1"], d["qw16_t1"], global_size=(24,1,1), local_size=LS)
  pr["spk_g4nw32qh1_100k"](d["kv"], d["sc"], d["qw16_t1"], d["pos_slot"], d["pm_t1"], d["ps_t1"], d["pA_t1"],
                           global_size=(4*SR,1,1), local_size=LS1K)
  pr["spk_c1g_100k"](d["pm_t1"], d["ps_t1"], d["pA_t1"], qr, d["ao_t1"], global_size=(24,1,1), local_size=LS, wait=True)
  out_t1[t] = P.down("ao_t1", (6144,), np.float16).astype(np.float32)
print(f"[t1 vs numpy]   {rel(out_t1, ao_np)}", flush=True)
print(f"[t1 vs mine]    {rel(out_t1, ao_mine)}", flush=True)
# per-row diagnostics
for t in [0, 1, 5, 15]:
  print(f"  row {t}: t1 {rel(out_t1[t], ao_np[t])} | mine {rel(ao_mine[t], ao_np[t])}", flush=True)
