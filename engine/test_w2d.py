# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2 draft-kernel smoke: load ONLY draft cubins + buffers, run one fill-style
chain with a wait+print after EVERY kernel -> names the faulting kernel."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev
def iq3s_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3s_grid
  vals = np.array([(w >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.float32)
  assert vals.size == 2048
  return vals
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
DPACK = f"{BASE}/draft_pack"
LS = (256, 1, 1)
SLICE = 40960
CTXK = 2304

P = Bufs()
cubins = ["dnorm2","dfgu","dkv","aattn_d","shead","samx","dposadd","ehproj","dq","doproj","ddown"]
pr = {}
for n in cubins:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
pr["h_embed"] = NVProgram(dev, TinyELF(lib=open(f"{BASE}/h_embed.cubin","rb").read(), name="h_embed", target=dev.renderer.target, signature=tuple()))
pr["k0_norm"] = NVProgram(dev, TinyELF(lib=open(f"{BASE}/k0_norm.cubin","rb").read(), name="k0_norm", target=dev.renderer.target, signature=tuple()))
pr["k3m_hh"] = NVProgram(dev, TinyELF(lib=open(f"{BASE}/k3m_hh.cubin","rb").read(), name="k3m_hh", target=dev.renderer.target, signature=tuple()))
print("[smoke] cubins loaded", flush=True)

# weights + tables
for nm in ("d_eh","d_q","d_k","d_v","d_o","d_fg","d_fu","d_fd","d_enw","d_hnw","d_shnw","d_nw1","d_nw2","d_qnw","d_knw"):
  P.up(nm, np.load(f"{DPACK}/{nm}.npy"))
P.up("emb", np.frombuffer(open(f"{DPACK}/emb_sample.bin","rb").read(), dtype=np.uint8) if os.path.exists(f"{DPACK}/emb_sample.bin") else np.zeros(248320*2200, np.uint8))
P.up("grid512", iq3s_grid_f32())
freqs = (1.0 / (10000000.0 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
P.up("freqs", freqs)
# slice: 40960 fake rows of Q5 data (zeros = valid layout) + table
P.up("slice_w", np.zeros(SLICE*3520, np.uint8))
P.up("stab", np.arange(SLICE, dtype=np.int32))
# scratch
for nm, nb, dt, pv in [("e_buf", 5120*4, np.float32, 0.5), ("cat", 10240*2, np.float16, 0.5),
                       ("xin_d", 5120*4, np.float32, 0.5), ("xh_d", 5120*2, np.float16, 0.5),
                       ("qrow_d", 12288*2, np.float16, 0.5), ("krow_d", 1024*2, np.float16, 0.5),
                       ("vrow_d", 1024*2, np.float16, 0.5), ("ao_row_d", 6144*2, np.float16, 0.5),
                       ("attn_out_d", 5120*2, np.float16, 0.5), ("hh_d", 5120*4, np.float32, 0.5),
                       ("hhx_d", 5120*2, np.float16, 0.5), ("gact_d", 17408*2, np.float16, 0.5),
                       ("hd_d", 5120*4, np.float32, 0.5), ("slogits", SLICE*2, np.float16, 0.5)]:
  P.poison(nm, nb, dt, pv)
P.poison("kv_d", 2*4*CTXK*256*2, np.float16, 0.5)
P.up("tok", np.array([7], dtype=np.int32))
P.up("pos", np.array([5], dtype=np.int32))
P.up("dpos", np.array([6], dtype=np.int32))
P.up("dring", np.array([-1], dtype=np.int32))
dev.synchronize()
print("[smoke] buffers up", flush=True)
d = P.d

def step(name):
  print(f"[smoke] launch {name}", flush=True)

# one-by-one with wait
step("h_embed");   pr["h_embed"](d["emb"], d["grid512"], d["tok"], d["e_buf"], global_size=(1,1,1), local_size=LS, wait=True)
step("dnorm2");    pr["dnorm2"](d["e_buf"], d["hd_d"], d["d_enw"], d["d_hnw"], d["cat"], global_size=(1,1,1), local_size=LS, wait=True)
step("ehproj");    pr["ehproj"](d["d_eh"], d["cat"], d["hh_d"], d["xin_d"], global_size=(640,1,1), local_size=LS, wait=True)
step("k0_norm");   pr["k0_norm"](d["xin_d"], d["d_nw1"], d["xh_d"], global_size=(1,1,1), local_size=LS, wait=True)
step("dq");        pr["dq"](d["d_q"], d["xh_d"], d["hh_d"], d["qrow_d"], global_size=(1536,1,1), local_size=LS, wait=True)
step("dkv");       pr["dkv"](d["d_k"], d["d_v"], d["xh_d"], d["krow_d"], d["vrow_d"], global_size=(256,1,1), local_size=LS, wait=True)
step("aattn_d");   pr["aattn_d"](d["qrow_d"], d["krow_d"], d["vrow_d"], d["d_qnw"], d["d_knw"], d["freqs"], d["kv_d"], d["pos"], d["ao_row_d"], global_size=(24,1,1), local_size=LS, wait=True)
step("do");        pr["doproj"](d["d_o"], d["ao_row_d"], d["hh_d"], d["attn_out_d"], global_size=(640,1,1), local_size=LS, wait=True)
step("k3m_hh");    pr["k3m_hh"](d["xin_d"], d["attn_out_d"], d["d_nw2"], d["hh_d"], d["hhx_d"], global_size=(1,1,1), local_size=LS, wait=True)
step("dfgu");      pr["dfgu"](d["d_fg"], d["d_fu"], d["hhx_d"], d["gact_d"], global_size=(2176,1,1), local_size=LS, wait=True)
step("ddown");     pr["ddown"](d["d_fd"], d["gact_d"], d["hh_d"], d["hd_d"], global_size=(640,1,1), local_size=LS, wait=True)
step("k0_norm2");  pr["k0_norm"](d["hd_d"], d["d_shnw"], d["xh_d"], global_size=(1,1,1), local_size=LS, wait=True)
step("shead");     pr["shead"](d["slice_w"], d["xh_d"], d["slogits"], global_size=(SLICE//8,1,1), local_size=LS, wait=True)
step("samx");      pr["samx"](d["slogits"], d["stab"], d["dring"], global_size=(1,1,1), local_size=LS, wait=True)
step("dposadd");   pr["dposadd"](d["pos"], d["dpos"], global_size=(1,1,1), local_size=LS, wait=True)
print("[smoke] ALL DRAFT KERNELS PASSED", flush=True)
