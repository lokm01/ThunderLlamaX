# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-a: engine0 fused GDN block — differential validation vs stock block on REAL
weights (T=1) + timing. Poison-first, fully-written-buffer checks."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad import Tensor
from tinygrad.device import Device

os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1")
os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_MAXCTX", "2048")
os.environ.setdefault("MTP_CKPT_DIR", "/tmp/ckpt_empty")

from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
print("[loading model...]", flush=True)
model = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", 2048)[0]
dev = Device["NV"]
blk = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
assert abs(blk.attn_norm.eps - 1e-6) < 1e-12, f"norm_eps={blk.attn_norm.eps} != baked 1e-6"
nv, hv, hk = blk.num_v_heads, blk.head_v_dim, blk.head_k_dim
assert (nv, hv, hk, blk.num_k_heads) == (48, 128, 128, 16), "dims drifted"
print(f"[block0 GDN dims ok] nv={nv} hv={hv} hk={hk}", flush=True)

rng = np.random.default_rng(11)
DIM = 5120
x_np = (rng.standard_normal((1, 1, DIM)) * 0.2).astype(np.float32)
conv_np = (rng.standard_normal((1, 3, 10240)) * 0.1).astype(np.float32)
rec_np = (rng.standard_normal((1, 48, 128, 128)) * 0.1).astype(np.float32)

# ---- stock reference (T=1, live-state path) ----
x_in = Tensor(x_np).contiguous().realize()
blk._init_state(x_in)
blk.conv_state.assign(Tensor(conv_np).contiguous()).realize()
blk.recurrent_state.assign(Tensor(rec_np).contiguous()).realize()
y_ref = blk(x_in, 100).float().numpy()
ref_h = y_ref.reshape(-1)
ref_conv = blk.conv_state.float().numpy().reshape(-1)
ref_rec = blk.recurrent_state.float().numpy().reshape(-1)
print(f"[ref] |h|max={np.abs(ref_h).max():.4f} |rec|max={np.abs(ref_rec).max():.4f}", flush=True)

# ---- engine0 ----
import engine0
from engine0 import GDNBlockEngine
e = GDNBlockEngine(0)
e.set_inputs(x=x_np.reshape(-1), conv=conv_np.reshape(-1), rec=rec_np.reshape(-1))
dev.synchronize()
t0 = time.perf_counter()
e.run(wait=True)
print(f"[engine first run +wait] {(time.perf_counter()-t0)*1e3:.1f} ms", flush=True)

got_h = e.P.down("y", (DIM,))
got_rec = e.P.down("rec", (48*128*128,))
got_conv = e.P.down("convB", (3*10240,))
def rel(a, b):
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))
r_h, r_rec, r_conv = rel(got_h, ref_h), rel(got_rec, ref_rec), rel(got_conv, ref_conv)
print(f"[val] h relerr={r_h:.2e}  rec relerr={r_rec:.2e}  conv relerr={r_conv:.2e}", flush=True)
ok = max(r_h, r_rec, r_conv) < 1e-3
verdict = "PASS" if ok else "FAIL"; print(f"[val] VERDICT: {verdict}", flush=True)

# stage dump on failure for debugging
if not ok:
  qkv = e.P.down("qkv_row", (10240,), np.float16).astype(np.float32)
  xh = e.P.down("xh", (DIM,), np.float16).astype(np.float32)
  core = e.P.down("core", (48,128))
  z = e.P.down("z", (6144,), np.float16).astype(np.float32)
  print(f"[dbg] xh[:4]={xh[:4]} qkv[:4]={qkv[:4]} core[0,:4]={core[0,:4]} z[:4]={z[:4]}", flush=True)
  np.save("/tmp/w1a_got_h.npy", got_h); np.save("/tmp/w1a_ref_h.npy", ref_h)

# ---- timing ----
print("== timing ==", flush=True)
NB = 48
# full-block pipeline x48, one wait
for _ in range(3):
  e.run("convA","convB"); e.run("convB","convA", wait=True)
best = 1e9
for rep in range(5):
  t0 = time.perf_counter()
  for i in range(NB):
    e.run("convA","convB") if (i&1)==0 else e.run("convB","convA")
  e.run("convA","convB", wait=True) if False else None
  # final wait via a trivial launch
  prg, args, grid = e.KP["k0_norm"](e)
  prg(*args, global_size=(1,1,1), local_size=(256,1,1), wait=True)
  dt = time.perf_counter() - t0
  best = min(best, dt)
print(f"[time] full-block x{NB} pipelined: {best*1e3:.2f} ms total -> {best/NB*1e3:.0f} us/block", flush=True)

# per-kernel attribution: 48 pipelined launches of one kernel + 1 wait
WB = {"k1_q5": 36044800, "k1_iq3": 12042240, "k1_ab": 2*983040, "OLD": 36044800+12042240+2*983040, "k3a_oproj": 33423360, "k3b_ffn": 2*34119680,
      "k3c_down": 34119680, "k2_scan": 3*10240*4*2 + 48*128*128*4*2 + 10240*4 + 3*6144*4*2}
tot_k = 0.0
for name in ["k0_norm","k1_q5","k1_iq3","k1_ab","k2_scan","k2b_z","k3a_oproj","k3m_hh","k3b_ffn","k3c_down"]:
  for _ in range(3): e.launch_one(name)
  e.launch_one(name, wait=True)
  t0 = time.perf_counter()
  for _ in range(NB): e.launch_one(name)
  e.launch_one(name, wait=True)
  dt = (time.perf_counter() - t0)/NB
  gbs = WB.get(name, 0)/dt/1e9
  tot_k += dt
  print(f"[attr] {name:10s} {dt*1e6:8.1f} us/launch  {gbs:7.1f} GB/s", flush=True)
print(f"[attr] kernel-sum = {tot_k*1e3:.2f} ms/block", flush=True)
print("[done]", flush=True)
