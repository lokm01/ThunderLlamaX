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
rng = np.random.default_rng(11)
x_np = (rng.standard_normal((1, 1, 5120)) * 0.2).astype(np.float32)
conv_np = (rng.standard_normal((1, 3, 10240)) * 0.1).astype(np.float32)
rec_np = (rng.standard_normal((1, 48, 128, 128)) * 0.1).astype(np.float32)
x_in = Tensor(x_np).contiguous().realize()
blk._init_state(x_in)
blk.conv_state.assign(Tensor(conv_np).contiguous()).realize()
blk.recurrent_state.assign(Tensor(rec_np).contiguous()).realize()
y_ref = blk(x_in, 100).float().numpy().reshape(-1)
ref_rec = blk.recurrent_state.float().numpy().reshape(48,128,128)
ref_conv = blk.conv_state.float().numpy().reshape(3,10240)

import engine0
from engine0 import GDNBlockEngine
e = GDNBlockEngine(0)
e.set_inputs(x=x_np.reshape(-1), conv=conv_np.reshape(-1), rec=rec_np.reshape(-1))
dev.synchronize()
e.run(wait=True)
got_rec = e.P.down("rec", (48,128,128))
got_h = e.P.down("y", (5120,))
err = np.abs(got_rec - ref_rec).max(axis=(1,2)) / max(np.abs(ref_rec).max(), 1e-9)
print("[pat] per-head rec relerr:", np.array2string(err, precision=3, max_line_width=200), flush=True)
eh = np.abs(got_h - y_ref) / max(np.abs(y_ref).max(), 1e-9)
print(f"[pat] h relerr overall={eh.max():.3f}", flush=True)
# engine k2 outputs vs numpy chain on ENGINE OWN inputs
qkv_g = e.P.down("qkv_row", (10240,), np.float16).astype(np.float32)
gate_g = e.P.down("gate_row", (6144,), np.float16).astype(np.float32)
al_g = e.P.down("alpharaw", (48,)); be_g = e.P.down("betaraw", (48,))
q_g = e.P.down("q", (48,128)); k_g = e.P.down("k", (48,128)); v_g = e.P.down("v", (48,128))
cw = e.P.down("conv_w", (10240,4)); dtb = e.P.down("dt_b", (48,)); sa = e.P.down("ssm_a", (48,))
def sig(x): return 1.0/(1.0+np.exp2(-x*1.4426950408889634))
win = np.concatenate([conv_np.reshape(3,10240), qkv_g.reshape(1,10240)], 0)
co = (win[0]*cw[:,0] + win[1]*cw[:,1] + win[2]*cw[:,2] + win[3]*cw[:,3]); co = co*sig(co)
v_r = co[4096:10240].reshape(48,128)
alv = np.exp(np.logaddexp(al_g + dtb, 0.0) * sa)
bev = 1.0/(1.0+np.exp2(-be_g*1.4426950408889634))
print(f"[chk] v(engine) vs numpy: {float(np.abs(v_g-v_r).max()/np.abs(v_r).max()):.2e}", flush=True)
st = rec_np.reshape(48,128,128).copy()
kn = k_g; qn = q_g
outs = np.zeros((48,128), np.float32)
for h in range(48):
    s1 = st[h]*alv[h]
    kd = (s1*kn[h]).sum(1)
    dl = (v_r[h]-kd)*bev[h]
    st[h] = s1 + dl[:,None]*kn[h][None,:]
    outs[h] = (st[h]*qn[h]).sum(1)
print(f"[chk] engine rec vs numpy-scan(on engine k1 outs): {float(np.abs(got_rec-st).max()/np.abs(st).max()):.2e}", flush=True)
# so if this is ~0, engine==my-numpy; then diff must be stock vs my-numpy on the SAME inputs.
# compare STOCK final rec vs numpy-scan using STOCK-path inputs: use engine k1 (close to stock)
print(f"[chk] stock rec vs numpy-scan(engine inputs): {float(np.abs(ref_rec-st).max()/np.abs(st).max()):.2e}", flush=True)
print(f"[chk] alpha range: alv[:4]={alv[:4]} beta[:4]={bev[:4]}", flush=True)
core_g = e.P.down("core", (48,128))
print(f"[chk] engine core vs numpy outs: {float(np.abs(core_g-outs).max()/np.abs(outs).max()):.2e}", flush=True)
