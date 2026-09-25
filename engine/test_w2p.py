# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2 probe-kernel smoke: M=3 trunk kernels on REAL single-block weights (no 13GB
load): GDN blk0 (Q8 out) + blk8 (IQ3 out) + attn blk3 (Q6 q) + blk11 (IQ3 q) +
head. Waits after every kernel -> names the faulting one."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32

BASE = "~/tinygrad-metal/engine0"
LS = (256,1,1)
DIM, VOCAB, CTXK = 5120, 248320, 2304
P = Bufs()
pr = {}
for n in ["h_embed3","k0n3","k0ab3","q5g8_3","k2s3","op38_3","k3ao3","hh3","ffn8_3","down8_3",
          "aq6k8_3","aq3k8_3","aattn3","ao8_3","head8_3","amx3"]:
  from tinygrad.device import TinyELF
  from tinygrad.runtime.ops_nv import NVProgram
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
print("[smokep] cubins loaded", flush=True)

def iq3s_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3s_grid
  return np.array([(w >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.float32)

ds, infos = parse_gguf()
P.up("gridf", iq3_grid_f32()); P.up("grid512", iq3s_grid_f32())
P.up("freqs", (1.0/(10000000.0**(np.arange(0,64,2,dtype=np.float64)/64.0))).astype(np.float32))
W = {}
def upk(key, gg):
  arr = np.ascontiguousarray(np.load(f"{BASE}/packed/{key}.npy"))
  W[gg] = P.up(gg.replace(".","_"), arr)
def upraw(gg):
  W[gg] = P.up(gg.replace(".","_"), np.frombuffer(read_raw(infos[gg], ds), dtype=np.uint8))
def upf(gg):
  W[gg] = P.up(gg.replace(".","_"), np.frombuffer(read_raw(infos[gg], ds), dtype="<f4").copy())

for bi in (0, 8):
  upk(f"gate{bi}", f"blk.{bi}.attn_gate.weight"); upk(f"fg{bi}", f"blk.{bi}.ffn_gate.weight")
  upk(f"fu{bi}", f"blk.{bi}.ffn_up.weight"); upk(f"fd{bi}", f"blk.{bi}.ffn_down.weight")
  upraw(f"blk.{bi}.attn_qkv.weight")
  if bi == 0: upraw(f"blk.{bi}.ssm_out.weight")          # Q8_0
  else: upk(f"out{bi}", f"blk.{bi}.ssm_out.weight")      # IQ3 packed
  for t in ("ssm_alpha.weight","ssm_beta.weight","ssm_conv1d.weight","ssm_dt.bias","ssm_a",
            "attn_norm.weight","post_attention_norm.weight","ssm_norm.weight"):
    upf(f"blk.{bi}.{t}")
for bi in (3, 11):
  if bi == 3: upk(f"q{bi}", f"blk.{bi}.attn_q.weight")
  else: upk(f"q{bi}", f"blk.{bi}.attn_q.weight")
  upk(f"k{bi}", f"blk.{bi}.attn_k.weight")
  upraw(f"blk.{bi}.attn_v.weight"); upraw(f"blk.{bi}.attn_output.weight")
  upk(f"fg{bi}", f"blk.{bi}.ffn_gate.weight"); upk(f"fu{bi}", f"blk.{bi}.ffn_up.weight"); upk(f"fd{bi}", f"blk.{bi}.ffn_down.weight")
  for t in ("attn_norm.weight","post_attention_norm.weight","attn_q_norm.weight","attn_k_norm.weight"):
    upf(f"blk.{bi}.{t}")
upraw("output.weight"); upf("output_norm.weight")
upraw("token_embd.weight")
P.poison("kv", 2*4*CTXK*256*2, np.float16, 0.5)
P.poison("conv4", 5*3*10240*4, np.float32, 0.5)
P.poison("rec4", 5*48*128*128*4, np.float32, 0.5)
for nm, nb, dt, pv in [("xA",3*DIM*4,np.float32,0.5),("xB",3*DIM*4,np.float32,0.5),("xh3",3*DIM*2,np.float16,0.5),
  ("hh3b",3*DIM*4,np.float32,0.5),("hhx3",3*DIM*2,np.float16,0.5),("qkv3",3*10240*2,np.float16,0.5),
  ("gate3",3*6144*2,np.float16,0.5),("araw3",3*48*4,np.float32,0.5),("braw3",3*48*4,np.float32,0.5),
  ("z3",3*6144*2,np.float16,0.5),("attn_out3",3*DIM*2,np.float16,0.5),("gact3",3*17408*2,np.float16,0.5),
  ("qrow3",3*12288*2,np.float16,0.5),("krow3",3*1024*2,np.float16,0.5),("vrow3",3*1024*2,np.float16,0.5),
  ("ao_row3",3*6144*2,np.float16,0.5),("logits3",3*VOCAB*2,np.float16,0.5)]:
  P.poison(nm, nb, dt, pv)
P.poison("q",48*128*4,np.float32,0.5); P.poison("k",48*128*4,np.float32,0.5); P.poison("v",48*128*4,np.float32,0.5)
P.poison("core",6144*4,np.float32,0.5)
P.up("s0", np.array([11],np.int32)); P.up("s1", np.array([12],np.int32)); P.up("s2", np.array([13],np.int32))
P.up("pos", np.array([200],np.int32)); P.up("amds", np.full(3,-1,np.int32))
dev.synchronize(); d = P.d
print("[smokep] buffers up", flush=True)

def go(name, fn):
  print(f"[smokep] {name}", flush=True); fn(True)

go("h_embed3", lambda wait: pr["h_embed3"](d["token_embd_weight"], d["grid512"], d["s0"], d["s1"], d["s2"], d["xA"], global_size=(1,1,1), local_size=LS, wait=wait))
for bi, outname in ((0,"k3ao3"),(8,"op38_3")):
  pre = f"blk_{bi}_"
  go(f"k0ab3_{bi}", lambda w,pre=pre: pr["k0ab3"](d["xA"], d[pre+"attn_norm_weight"], d[pre+"ssm_alpha_weight"], d[pre+"ssm_beta_weight"], d["xh3"], d["araw3"], d["braw3"], global_size=(13,1,1), local_size=LS, wait=w))
  go(f"q5g8_3_{bi}", lambda w,pre=pre: pr["q5g8_3"](d["blk_0_attn_qkv_weight" if bi==0 else "blk_8_attn_qkv_weight"], d[f"blk_{bi}_attn_gate_weight"], d["gridf"], d["xh3"], d["qkv3"], d["gate3"], global_size=(2048,1,1), local_size=LS, wait=w))
  go(f"k2s3_{bi}", lambda w,pre=pre,bi=bi: pr["k2s3"](d["conv4"], d["rec4"], d["qkv3"], d["gate3"], d[pre+"ssm_conv1d_weight"], d[pre+"ssm_dt_bias"], d[pre+"ssm_a"], d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], d[pre+"ssm_norm_weight"], d["z3"], global_size=(48,1,1), local_size=LS, wait=w))
  ow = "blk_0_ssm_out_weight" if bi == 0 else "blk_8_ssm_out_weight"
  go(f"{outname}_{bi}", lambda w,ow=ow: pr["k3ao3" if ow=="blk_0_ssm_out_weight" else "op38_3"](d[ow], d["z3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=w) if ow=="blk_0_ssm_out_weight" else pr["op38_3"](d[ow], d["gridf"], d["z3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=w))
  go(f"hh3_{bi}", lambda w,pre=pre: pr["hh3"](d["xA"], d["attn_out3"], d[pre+"post_attention_norm_weight"], d["hh3b"], d["hhx3"], global_size=(1,1,1), local_size=LS, wait=w))
  go(f"ffn8_3_{bi}", lambda w: pr["ffn8_3"](d[f"blk_{bi}_ffn_gate_weight"], d[f"blk_{bi}_ffn_up_weight"], d["gridf"], d["hhx3"], d["gact3"], global_size=(2176,1,1), local_size=LS, wait=w))
  go(f"down8_3_{bi}", lambda w: pr["down8_3"](d[f"blk_{bi}_ffn_down_weight"], d["gridf"], d["gact3"], d["hh3b"], d["xB"], global_size=(640,1,1), local_size=LS, wait=w))
for bi, qk in ((3,"aq6k8_3"),(11,"aq3k8_3")):
  pre = f"blk_{bi}_"
  go(f"k0n3_{bi}", lambda w,pre=pre: pr["k0n3"](d["xA"], d[pre+"attn_norm_weight"], d["xh3"], global_size=(1,1,1), local_size=LS, wait=w))
  go(f"{qk}_{bi}", lambda w,bi=bi: pr["aq6k8_3" if bi==3 else "aq3k8_3"](d[f"blk_{bi}_attn_q_weight"], d[f"blk_{bi}_attn_k_weight"], d[f"blk_{bi}_attn_v_weight"], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"], global_size=(1792,1,1), local_size=LS, wait=w))
  go(f"aattn3_{bi}", lambda w,bi=bi,pre=pre: pr["aattn3"](d["qrow3"], d["krow3"], d["vrow3"], d[pre+"attn_q_norm_weight"], d[pre+"attn_k_norm_weight"], d["freqs"], d["kv"], d["pos"], d["ao_row3"], global_size=(24,1,1), local_size=LS, wait=w))
  go(f"ao8_3_{bi}", lambda w,bi=bi: pr["ao8_3"](d[f"blk_{bi}_attn_output_weight"], d["grid512"], d["ao_row3"], d["attn_out3"], global_size=(640,1,1), local_size=LS, wait=w))
go("head8_3", lambda w: pr["k0n3"](d["xA"], d["output_norm_weight"], d["xh3"], global_size=(1,1,1), local_size=LS, wait=w))
go("head8_3", lambda w: pr["head8_3"](d["output_weight"], d["xh3"], d["logits3"], global_size=(VOCAB//8,1,1), local_size=LS, wait=w))
go("amx3", lambda w: pr["amx3"](d["logits3"], d["amds"], global_size=(3,1,1), local_size=LS, wait=w))
print("[smokep] ALL PROBE KERNELS PASSED", flush=True)
