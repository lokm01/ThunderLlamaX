# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2D L1: synced bench of current M=3 probe GEMV kernels on real weights.
Synced-per-launch only (pipelined lies on this dext)."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256,1,1)
DIM, VOCAB = 5120, 248320
REPS = 30
P = Bufs()
pr = {}
def load(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
for n in ["q5g8_3","op38_3","k3ao3","ffn8_3","down8_3","ao8_3","head8_3","aq3k8_3","k2s3","k0n3","k0ab3","hh3"]:
  load(n)
def iq3s_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3s_grid
  return np.array([(w >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.float32)
ds, infos = parse_gguf()
P.up("gridf", iq3_grid_f32()); P.up("grid512", iq3s_grid_f32())
W = {}
SZ = {}
def upk(key, nm):
  arr = np.ascontiguousarray(np.load(f"{BASE}/packed/{key}.npy"))
  W[nm] = P.up(nm.replace(".","_"), arr); SZ[nm] = arr.nbytes
def upraw(gg):
  arr = np.frombuffer(read_raw(infos[gg], ds), dtype=np.uint8)
  W[gg] = P.up(gg.replace(".","_"), arr); SZ[gg] = arr.nbytes
def upf(gg):
  arr = np.frombuffer(read_raw(infos[gg], ds), dtype="<f4").copy()
  W[gg] = P.up(gg.replace(".","_"), arr); SZ[gg] = arr.nbytes
for bi in (0, 8):
  upk(f"gate{bi}", f"g{bi}"); upk(f"fg{bi}", f"fg{bi}"); upk(f"fu{bi}", f"fu{bi}"); upk(f"fd{bi}", f"fd{bi}")
  upraw(f"blk.{bi}.attn_qkv.weight")
  if bi == 0: upraw(f"blk.{bi}.ssm_out.weight")
  else: upk(f"out{bi}", f"o{bi}")
  for t in ("ssm_alpha.weight","ssm_beta.weight","ssm_conv1d.weight","ssm_dt.bias","ssm_a",
            "attn_norm.weight","post_attention_norm.weight","ssm_norm.weight"):
    upf(f"blk.{bi}.{t}")
for bi in (11,):
  upk(f"q{bi}", f"q{bi}"); upk(f"k{bi}", f"k{bi}")
  upraw(f"blk.{bi}.attn_v.weight"); upraw(f"blk.{bi}.attn_output.weight")
  for t in ("attn_norm.weight","post_attention_norm.weight"):
    upf(f"blk.{bi}.{t}")
upraw("output.weight"); upf("output_norm.weight")
# scratch (poison)
for nm, nb, dt, pv in [("xh3",3*DIM*2,np.float16,0.5),("qkv3",3*10240*2,np.float16,0.5),
  ("gate3",3*6144*2,np.float16,0.5),("araw3",3*48*4,np.float32,0.5),("braw3",3*48*4,np.float32,0.5),
  ("z3",3*6144*2,np.float16,0.5),("attn_out3",3*DIM*2,np.float16,0.5),("gact3",3*17408*2,np.float16,0.5),
  ("qrow3",3*12288*2,np.float16,0.5),("krow3",3*1024*2,np.float16,0.5),("vrow3",3*1024*2,np.float16,0.5),
  ("ao_row3",3*6144*2,np.float16,0.5),("hh3b",3*DIM*4,np.float32,0.5),("hhx3",3*DIM*2,np.float16,0.5),
  ("xA",3*DIM*4,np.float32,0.5),("xB",3*DIM*4,np.float32,0.5)]:
  P.poison(nm, nb, dt, pv)
P.poison("conv4", 5*3*10240*4, np.float32, 0.5)
P.poison("rec4", 5*48*128*128*4, np.float32, 0.5)
P.poison("q",48*128*4,np.float32,0.5); P.poison("k",48*128*4,np.float32,0.5); P.poison("v",48*128*4,np.float32,0.5)
P.poison("core",6144*4,np.float32,0.5)
dev.synchronize(); d = P.d
print("[bench] buffers up", flush=True)

def bench(name, nbytes, launch, reps=REPS):
  launch(False); dev.synchronize()  # warm
  t0 = time.perf_counter()
  for _ in range(reps):
    launch(True)
  dt = (time.perf_counter()-t0)/reps
  t0 = time.perf_counter()
  for _ in range(reps):
    launch(False)
  dev.synchronize()
  dp = (time.perf_counter()-t0)/reps
  print(f"[bench] {name:14s} sync {dt*1e6:8.1f}us pipe {dp*1e6:8.1f}us  GB/s sync {nbytes/dt/1e9:6.1f} pipe {nbytes/dp/1e9:6.1f}  ({nbytes/1e6:.1f} MB)", flush=True)
  return dt

# --- per-kernel, byte counts from actual buffers ---
bq = SZ["blk.8.attn_qkv.weight"]
bg = SZ["g8"]
bf = SZ["fg8"] + SZ["fu8"]
bd = SZ["fd8"]
bo8 = SZ["blk.0.ssm_out.weight"]
bo3 = SZ["o8"]
bh = SZ["blk.11.attn_output.weight"]
bq3 = SZ["q11"] + SZ["k11"] + SZ["blk.11.attn_v.weight"]
bhv = SZ["output.weight"]
bench("q5g8_3", bq+bg, lambda w: pr["q5g8_3"](d["blk_8_attn_qkv_weight"], d["g8"], d["gridf"], d["xh3"], d["qkv3"], d["gate3"], global_size=(2048,1,1), local_size=LS, wait=w))
bench("ffn8_3", bf, lambda w: pr["ffn8_3"](d["fg8"], d["fu8"], d["gridf"], d["hhx3"], d["gact3"], global_size=(2176,1,1), local_size=LS, wait=w))
bench("down8_3", bd, lambda w: pr["down8_3"](d["fd8"], d["gridf"], d["gact3"], d["hh3b"], d["xB"], global_size=(640,1,1), local_size=LS, wait=w))
bench("k3ao3(Q8)", bo8, lambda w: pr["k3ao3"](d["blk_0_ssm_out_weight"], d["z3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=w))
bench("op38_3(IQ3)", bo3, lambda w: pr["op38_3"](d["o8"], d["gridf"], d["z3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=w))
bench("aq3k8_3", bq3, lambda w: pr["aq3k8_3"](d["q11"], d["k11"], d["blk_11_attn_v_weight"], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"], global_size=(1792,1,1), local_size=LS, wait=w))
bench("ao8_3", bh, lambda w: pr["ao8_3"](d["blk_11_attn_output_weight"], d["grid512"], d["ao_row3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=w))
P.poison("logits3", 3*VOCAB*2, np.float16, 0.5)
bench("head8_3", bhv, lambda w: pr["head8_3"](d["output_weight"], d["xh3"], d["logits3"], global_size=(VOCAB//8,1,1), local_size=LS, wait=w))
# scan + norms for the non-GEMV probe share
bench("k2s3", 6*1024*1024, lambda w: pr["k2s3"](d["conv4"], d["rec4"], d["qkv3"], d["gate3"], d["blk_8_ssm_conv1d_weight"], d["blk_8_ssm_dt_bias"], d["blk_8_ssm_a_weight"], d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], d["blk_8_ssm_norm_weight"], d["z3"], global_size=(48,1,1), local_size=LS, wait=w))
print("[bench] DONE", flush=True)
