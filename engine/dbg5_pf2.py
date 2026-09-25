# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Perturbation test: does a ~2e-4 input diff amplify to ~0.3 rec drift in 16 steps?
Runs the T=1 k2s 16x on qkv16 vs qkv16+eps, plus dumps my harness rec magnitudes."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
LS = (256,1,1); M = 16
P = Bufs()
def prog(n):
  return NVProgram(dev, TinyELF(lib=open(f"{BASE}/{n}.cubin","rb").read(), name=n, target=dev.renderer.target, signature=tuple()))
pr = {n: prog(n) for n in ["k0ab", "k2s"]}
ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
G0 = [i for i in range(64) if i not in set(attn_idx)][0]
pre = f"blk.{G0}."
for nm, key in [("nw1","attn_norm.weight"),("snw","ssm_norm.weight")]:
  P.up(nm, np.frombuffer(read_raw(infos[pre+key], ds), dtype="<f4"))
P.up("convw", np.ascontiguousarray(np.frombuffer(read_raw(infos[pre+"ssm_conv1d.weight"], ds), dtype="<f4").reshape(10240,4)))
P.up("dtb", np.frombuffer(read_raw(infos[pre+"ssm_dt.bias"], ds), dtype="<f4"))
P.up("ssma", np.frombuffer(read_raw(infos[pre+"ssm_a"], ds), dtype="<f4"))
P.up("alpha", np.frombuffer(read_raw(infos[pre+"ssm_alpha.weight"], ds), dtype="<f4").reshape(48,5120))
P.up("beta", np.frombuffer(read_raw(infos[pre+"ssm_beta.weight"], ds), dtype="<f4").reshape(48,5120))
rng = np.random.default_rng(31)
xin = (rng.standard_normal((M, 5120)) * 0.7).astype(np.float32)
qkv16 = (rng.standard_normal((M, 10240)) * 0.45).astype(np.float16)
gate16 = (rng.standard_normal((M, 6144)) * 0.4).astype(np.float16)
# T=1-sized copies + perturbed copy (2e-4 relative noise on qkv)
P.up("qkvA", qkv16); P.up("gateA", gate16); P.up("xinA", xin)
pert = (qkv16.astype(np.float32) * (1 + rng.standard_normal(qkv16.shape) * 2e-4)).astype(np.float16)
P.up("qkvB", pert)
for nm in ("convA","convB"): P.up(nm, np.zeros(3*10240, dtype=np.float32))
for nm, nb in (("recA",48*128*128*4),("recB",48*128*128*4),("q",48*128*4),("k",48*128*4),("v",48*128*4),("core",48*128*4),("araw",48*4),("betaraw",48*4)): P.up(nm, np.zeros(nb//4, dtype=np.float32))
P.up("araw", np.zeros(48, dtype=np.float32)); P.up("betaraw", np.zeros(48, dtype=np.float32))
P.poison("z_t1", 6144*2, np.float16, 7.7)
dev.synchronize()
d = P.d
# alpha/beta per row from xin via k0ab, gather to per-row araw16 host then re-up per step (match harness semantics loosely: single alpha for all rows of the perturbed test is fine — same both sides)
def run16(qkv, conv0, conv1, rec, tag):
  par = 0
  for t in range(M):
    P.win_up("araw", 0, ar[t]); P.win_up("betaraw", 0, br[t])
    qr = d[qkv].offset(offset=t*10240*2, size=10240*2)
    gr = d["gateA"].offset(offset=t*6144*2, size=6144*2)
    pr["k2s"](d[conv0 if par==0 else conv1], d[conv1 if par==0 else conv0], qr, gr, d["convw"], d["dtb"], d["ssma"],
              d["araw"], d["betaraw"], d["q"], d["k"], d["v"], d[rec], d["core"], d["snw"], d["z_t1"],
              global_size=(48,1,1), local_size=LS, wait=(t==M-1))
    par ^= 1
  return P.down(rec, (48*128*128,), np.float32), P.down(conv1 if par==1 else conv0, (3*10240,), np.float32)
# per-row alpha/beta
ar = np.zeros((M,48), np.float32); br = np.zeros((M,48), np.float32)
for t in range(M):
  P.win_up("xinA", t*5120*4, xin[t])
  pr["k0ab"](d["xinA"].offset(offset=t*5120*4, size=5120*4), d["nw1"], d["alpha"], d["beta"], d["z_t1"],
             d["araw"], d["betaraw"], global_size=(13,1,1), local_size=LS, wait=True)
  ar[t] = P.down("araw", (48,), np.float32); br[t] = P.down("betaraw", (48,), np.float32)
recA, convA_ = run16("qkvA", "convA", "convB", "recA", "A")
P.up("recB", np.zeros(48*128*128, dtype=np.float32)); P.up("convA", np.zeros(3*10240, dtype=np.float32)); P.up("convB", np.zeros(3*10240, dtype=np.float32))
dev.synchronize()
recB, convB_ = run16("qkvB", "convA", "convB", "recB", "B")
print(f"[pert 2e-4] rec rel-drift {np.linalg.norm(recB-recA)/max(np.linalg.norm(recA),1e-9):.3e}  (|recA| {np.linalg.norm(recA):.3e} |recB| {np.linalg.norm(recB):.3e})", flush=True)
print(f"[pert 2e-4] conv rel-drift {np.linalg.norm(convB_-convA_)/max(np.linalg.norm(convA_),1e-9):.3e}", flush=True)
print(f"[finite] recA {np.isfinite(recA).all()} recB {np.isfinite(recB).all()}", flush=True)
