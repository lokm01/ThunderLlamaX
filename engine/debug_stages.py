# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad import Tensor
from tinygrad.device import Device
os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1"); os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_MAXCTX", "2048"); os.environ.setdefault("MTP_CKPT_DIR", "/tmp/ckpt_empty")
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
model = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", 2048)[0]
dev = Device["NV"]
blk = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
nv, hv, hk = 48, 128, 128
rng = np.random.default_rng(11)
x_np = (rng.standard_normal((1, 1, 5120)) * 0.2).astype(np.float32)
conv_np = (rng.standard_normal((1, 3, 10240)) * 0.1).astype(np.float32)
rec_np = (rng.standard_normal((1, nv, hv, hk)) * 0.1).astype(np.float32)

import engine0
from engine0 import GDNBlockEngine
e = GDNBlockEngine(0)
e.set_inputs(x=x_np.reshape(-1), conv=conv_np.reshape(-1), rec=rec_np.reshape(-1))
dev.synchronize()
e.launch_one("k0_norm", wait=True)

def rel(a, b): return float(np.abs(np.asarray(a,float) - np.asarray(b,float)).max() / max(np.abs(np.asarray(b,float)).max(), 1e-9))

# ---- STAGE 1: xh ----
x_in = Tensor(x_np).contiguous().realize()
xh_ref = blk.attn_norm(x_in).half().realize().numpy().reshape(-1)
xh_g = e.P.down("xh", (5120,), np.float16).astype(np.float32)
print(f"[st1] xh relerr={rel(xh_g, xh_ref):.2e}", flush=True)

# ---- STAGE 2: qkv / gate / alpha / beta from xh_ref ----
xh_t = Tensor(xh_ref.reshape(1,1,5120).astype(np.float16)).contiguous().realize()
qkv_ref = blk.attn_qkv(xh_t).float().realize().numpy().reshape(-1)
gate_ref = blk.attn_gate(xh_t).float().realize().numpy().reshape(-1)
al_ref = blk.ssm_alpha(xh_t).float().realize().numpy().reshape(-1)
be_ref = blk.ssm_beta(xh_t).float().realize().numpy().reshape(-1)
e.launch_one("k1_q5", wait=True)
e.launch_one("k1_iq3", wait=True)
e.launch_one("k1_ab", wait=True)
qkv_g = e.P.down("qkv_row", (10240,), np.float16).astype(np.float32)
gate_g = e.P.down("gate_row", (6144,), np.float16).astype(np.float32)
al_g = e.P.down("alpharaw", (48,)); be_g = e.P.down("betaraw", (48,))
print(f"[st2] qkv relerr={rel(qkv_g, qkv_ref):.2e}", flush=True)
print(f"[st2] gate relerr={rel(gate_g, gate_ref):.2e}", flush=True)
print(f"[st2] alpharaw relerr={rel(al_g, al_ref):.2e}", flush=True)
print(f"[st2] betaraw relerr={rel(be_g, be_ref):.2e}", flush=True)
# per-slice qkv errors (q=0:2048, k=2048:4096, v=4096:10240)
for nm, s in (("q", slice(0,2048)), ("k", slice(2048,4096)), ("v", slice(4096,10240))):
    print(f"[st2] qkv[{nm}] relerr={rel(qkv_g[s], qkv_ref[s]):.2e}", flush=True)
# lane-group error pattern on v slice: groups of 1024 values = 8 lanes*... per-sub pattern
verr = np.abs(qkv_g[4096:10240] - qkv_ref[4096:10240])
for g in range(4):
    seg = verr[g*16::64]  # sample pattern
    print(f"[st2] v err group{g} mean={verr[g*1536:(g+1)*1536].mean():.4f}", flush=True)

# ---- STAGE 3: k2 with STOCK qkv/gate/alpha/beta injected ----
e.P.up("qkv_row", qkv_ref.astype(np.float16))
e.P.up("gate_row", gate_ref.astype(np.float16))
e.P.up("alpharaw", al_ref); e.P.up("betaraw", be_ref)
dev.synchronize()
e.launch_one("k2_scan", wait=True)
# numpy k2 reference from the same inputs
cw = e.P.down("conv_w", (10240,4))
def sig(x): return 1.0/(1.0+np.exp2(-x*1.4426950408889634))
win = np.concatenate([conv_np.reshape(3,10240), qkv_ref.reshape(1,10240)], 0)
co = win[0]*cw[:,0] + win[1]*cw[:,1] + win[2]*cw[:,2] + win[3]*cw[:,3]
co = co * sig(co)
q_r = co[0:2048].reshape(16,128); k_r = co[2048:4096].reshape(16,128); v_r = co[4096:10240].reshape(48,128)
qn = np.zeros((48,128), np.float32); kn = np.zeros((48,128), np.float32)
for h in range(48):
    qq = q_r[h//3]; kk = k_r[h//3]
    qn[h] = qq / max(np.linalg.norm(qq), 1e-6) / np.sqrt(128.0)
    kn[h] = kk / max(np.linalg.norm(kk), 1e-6)
q_g = e.P.down("q", (48,128)); k_g = e.P.down("k", (48,128)); v_g = e.P.down("v", (48,128))
print(f"[st3] q relerr={rel(q_g, qn):.2e}  k relerr={rel(k_g, kn):.2e}  v relerr={rel(v_g, v_r):.2e}", flush=True)
# scan: numpy from same state
dtb = e.P.down("dt_b", (48,)); sa = e.P.down("ssm_a", (48,))
alv = np.exp(np.logaddexp(al_ref + dtb, 0.0) * sa)
bev = 1.0/(1.0+np.exp2(-be_ref*1.4426950408889634))
st = rec_np.reshape(48,128,128).copy()
outs = np.zeros((48,128), np.float32)
for h in range(48):
    s1 = st[h]*alv[h]
    kd = (s1*kn[h]).sum(1)
    dl = (v_r[h]-kd)*bev[h]
    st[h] = s1 + dl[:,None]*kn[h][None,:]
    outs[h] = (st[h]*qn[h]).sum(1)
rec_g = e.P.down("rec", (48,128,128)); core_g = e.P.down("core", (48,128))
convB_g = e.P.down("convB", (3,10240))
conv_ref = win[1:4]
print(f"[st3] rec relerr={rel(rec_g, st):.2e}  core relerr={rel(core_g, outs):.2e}  conv relerr={rel(convB_g, conv_ref):.2e}", flush=True)
print("[debug done]", flush=True)
