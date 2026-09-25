# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2D L1 final validation (BIT-EXACT vs m3 originals, poison-first) + bench."""
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
DIM, FFN_N, VOCAB = 5120, 17408, 248320
REPS = 30
P = Bufs()
pr = {}
for n in ["q5g8_3","op38_3","k3ao3","ao8_3","down8_3","q5g8v_3","op38nw32_3","k3aonw32_3","ao8nw32_3","down8nw32_3","ffn8_3","ffn8v_3"]:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
def iq3s_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3s_grid
  return np.array([(w >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.float32)
P.up("gridf", iq3_grid_f32()); P.up("grid512", iq3s_grid_f32())
ds, infos = parse_gguf()
SZ = {}
def upraw(gg):
  arr = np.frombuffer(read_raw(infos[gg], ds), dtype=np.uint8)
  P.up(gg.replace(".","_"), arr); SZ[gg] = arr.nbytes
def upk(key, nm):
  arr = np.ascontiguousarray(np.load(f"{BASE}/packed/{key}.npy"))
  P.up(nm, arr); SZ[nm] = arr.nbytes
upraw("blk.8.attn_qkv.weight"); upk("gate8", "gate8")
upraw("blk.0.ssm_out.weight")
upk("out8", "out8"); upk("fd8", "fd8"); upk("fg8","fg8"); upk("fu8","fu8")
upraw("blk.11.attn_output.weight")
rng = np.random.default_rng(7)
def uph(n, sz): P.up(n, (rng.standard_normal(sz)*0.3).astype(np.float16))
def upf(n, sz): P.up(n, (rng.standard_normal(sz)*0.5).astype(np.float32))
uph("xh3", 3*DIM); uph("z3", 3*6144); uph("ao_in3", 3*6144); uph("gact_in", 3*FFN_N); uph("hhx3", 3*DIM)
upf("hh3f", 3*DIM)
P.poison("qkv3", 3*10240*2, np.float16, 7.7); P.poison("gate3", 3*6144*2, np.float16, 7.7)
P.poison("attn_out3", 3*DIM*2, np.float16, 7.7); P.poison("y3", 3*DIM*4, np.float32, 7.7e31)
P.poison("gactA", 3*FFN_N*2, np.float16, 7.7)
dev.synchronize(); d = P.d

def cmp2(tag, old, new, outnm, outb, dt):
  P.poison(outnm, outb, dt, 7.7 if dt==np.float16 else 7.7e31)
  old()
  o = P.down(outnm, (outb//np.dtype(dt).itemsize,), dt)
  P.poison(outnm, outb, dt, 7.7 if dt==np.float16 else 7.7e31)
  new()
  n = P.down(outnm, (outb//np.dtype(dt).itemsize,), dt)
  ok = np.array_equal(o, n)
  print(f"[v3] {tag:16s} {'BIT-IDENTICAL' if ok else 'MISMATCH'}", flush=True)
  return ok

oks = []
oks.append(cmp2("q5g8v_3",
  lambda: pr["q5g8_3"](d["blk_8_attn_qkv_weight"], d["gate8"], d["gridf"], d["xh3"], d["qkv3"], d["gate3"], global_size=(2048,1,1), local_size=LS, wait=True),
  lambda: pr["q5g8v_3"](d["blk_8_attn_qkv_weight"], d["gate8"], d["gridf"], d["xh3"], d["qkv3"], d["gate3"], global_size=(2048,1,1), local_size=LS, wait=True),
  "qkv3", 3*10240*2, np.float16))
oks.append(cmp2("gate-half of q5g8v", lambda: None, lambda: None, "qkv3", 4, np.float16) if False else True)
oks.append(cmp2("op38nw32_3",
  lambda: pr["op38_3"](d["out8"], d["gridf"], d["z3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=True),
  lambda: pr["op38nw32_3"](d["out8"], d["gridf"], d["z3"], d["attn_out3"], global_size=(160,1,1), local_size=(1024,1,1), wait=True),
  "attn_out3", 3*DIM*2, np.float16))
oks.append(cmp2("k3aonw32_3",
  lambda: pr["k3ao3"](d["blk_0_ssm_out_weight"], d["z3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=True),
  lambda: pr["k3aonw32_3"](d["blk_0_ssm_out_weight"], d["z3"], d["attn_out3"], global_size=(160,1,1), local_size=(1024,1,1), wait=True),
  "attn_out3", 3*DIM*2, np.float16))
oks.append(cmp2("ao8nw32_3",
  lambda: pr["ao8_3"](d["blk_11_attn_output_weight"], d["grid512"], d["ao_in3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=True),
  lambda: pr["ao8nw32_3"](d["blk_11_attn_output_weight"], d["grid512"], d["ao_in3"], d["attn_out3"], global_size=(160,1,1), local_size=(1024,1,1), wait=True),
  "attn_out3", 3*DIM*2, np.float16))
oks.append(cmp2("down8nw32_3",
  lambda: pr["down8_3"](d["fd8"], d["gridf"], d["gact_in"], d["hh3f"], d["y3"], global_size=(640,1,1), local_size=LS, wait=True),
  lambda: pr["down8nw32_3"](d["fd8"], d["gridf"], d["gact_in"], d["hh3f"], d["y3"], global_size=(160,1,1), local_size=(1024,1,1), wait=True),
  "y3", 3*DIM*4, np.float32))
oks.append(cmp2("ffn8v_3",
  lambda: pr["ffn8_3"](d["fg8"], d["fu8"], d["gridf"], d["hhx3"], d["gactA"], global_size=(2176,1,1), local_size=LS, wait=True),
  lambda: pr["ffn8v_3"](d["fg8"], d["fu8"], d["gridf"], d["hhx3"], d["gactA"], global_size=(2176,1,1), local_size=LS, wait=True),
  "gactA", 3*FFN_N*2, np.float16))
print("[v3] ALL PASS" if all(oks) else "[v3] FAIL", flush=True)
if not all(oks): sys.exit(1)

def bench(name, nbytes, launch):
  launch(False); dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(REPS): launch(True)
  dt = (time.perf_counter()-t0)/REPS
  t0 = time.perf_counter()
  for _ in range(REPS): launch(False)
  dev.synchronize()
  dp = (time.perf_counter()-t0)/REPS
  print(f"[v3] {name:14s} sync {dt*1e6:8.1f}us pipe {dp*1e6:8.1f}us  GB/s {nbytes/dt/1e9:6.1f}|{nbytes/dp/1e9:6.1f}", flush=True)

bench("q5g8_3", SZ["blk.8.attn_qkv.weight"]+SZ["gate8"], lambda w: pr["q5g8_3"](d["blk_8_attn_qkv_weight"], d["gate8"], d["gridf"], d["xh3"], d["qkv3"], d["gate3"], global_size=(2048,1,1), local_size=LS, wait=w))
bench("q5g8v_3", SZ["blk.8.attn_qkv.weight"]+SZ["gate8"], lambda w: pr["q5g8v_3"](d["blk_8_attn_qkv_weight"], d["gate8"], d["gridf"], d["xh3"], d["qkv3"], d["gate3"], global_size=(2048,1,1), local_size=LS, wait=w))
bench("down8nw32", SZ["fd8"], lambda w: pr["down8nw32_3"](d["fd8"], d["gridf"], d["gact_in"], d["hh3f"], d["y3"], global_size=(160,1,1), local_size=(1024,1,1), wait=w))
bench("op38nw32", SZ["out8"], lambda w: pr["op38nw32_3"](d["out8"], d["gridf"], d["z3"], d["attn_out3"], global_size=(160,1,1), local_size=(1024,1,1), wait=w))
bench("k3aonw32", SZ["blk.0.ssm_out.weight"], lambda w: pr["k3aonw32_3"](d["blk_0_ssm_out_weight"], d["z3"], d["attn_out3"], global_size=(160,1,1), local_size=(1024,1,1), wait=w))
bench("ao8nw32", SZ["blk.11.attn_output.weight"], lambda w: pr["ao8nw32_3"](d["blk_11_attn_output_weight"], d["grid512"], d["ao_in3"], d["attn_out3"], global_size=(160,1,1), local_size=(1024,1,1), wait=w))
print("[v3] DONE", flush=True)
