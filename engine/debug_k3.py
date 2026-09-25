# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad import Tensor, dtypes
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
import engine0
from engine0 import GDNBlockEngine
e = GDNBlockEngine(0)
e.set_inputs(x=x_np.reshape(-1), conv=conv_np.reshape(-1), rec=rec_np.reshape(-1))
dev.synchronize()
e.run(wait=True)
def rel(a,b): return float(np.abs(np.asarray(a,float)-np.asarray(b,float)).max()/max(np.abs(np.asarray(b,float)).max(),1e-9))
# k2b self-check
core = e.P.down("core", (48,128)); gate = e.P.down("gate_row", (6144,), np.float16).astype(np.float32)
snw = e.P.down("snw", (128,))
z_g = e.P.down("z", (6144,), np.float16).astype(np.float32)
zn = np.zeros((48,128), np.float32)
for h in range(48):
    r = 1.0/np.sqrt((core[h]**2).mean() + 1e-6)
    zn[h] = core[h]*r*snw
gs = gate.reshape(48,128) * (1.0/(1.0+np.exp2(-gate.reshape(48,128)*1.4426950408889634)))
z_np = (zn * gs).astype(np.float16).astype(np.float32)
print(f"[k3] z(engine) vs numpy: {rel(z_g, z_np.reshape(-1)):.2e}", flush=True)
# stock o_proj on MY z
z_t = Tensor(z_g.reshape(1,1,6144).astype(np.float16)).contiguous().realize()
ao_ref = blk.ssm_out(z_t).float().realize().numpy().reshape(-1)
ao_g = e.P.down("attn_out", (5120,), np.float16).astype(np.float32)
print(f"[k3] attn_out(engine Q8) vs stock: {rel(ao_g, ao_ref):.2e}", flush=True)
# hh
hh_g = e.P.down("hh", (5120,)); hhx_g = e.P.down("hhx", (5120,), np.float16).astype(np.float32)
hh_np = x_np.reshape(-1) + ao_ref if rel(ao_g, ao_ref) < 1e-3 else x_np.reshape(-1) + ao_g
print(f"[k3] hh vs numpy(x+stock_ao): {rel(hh_g, hh_np):.2e}", flush=True)
nw2 = e.P.down("nw2", (5120,))
r2 = 1.0/np.sqrt((hh_np**2).mean() + 1e-6)
hhx_np = (hh_np*r2*nw2).astype(np.float16).astype(np.float32)
print(f"[k3] hhx vs numpy: {rel(hhx_g, hhx_np):.2e}", flush=True)
# stock gate/up on MY hhx
hhx_t = Tensor(hhx_g.reshape(1,1,5120).astype(np.float16)).contiguous().realize()
g_ref = blk.ffn_gate(hhx_t).float().realize().numpy().reshape(-1)
u_ref = blk.ffn_up(hhx_t).float().realize().numpy().reshape(-1)
gact_ref = (g_ref * (1.0/(1.0+np.exp2(-g_ref*1.4426950408889634))) * u_ref)
gact_g = e.P.down("gact", (17408,), np.float16).astype(np.float32)
print(f"[k3] gact(engine) vs stock-numpy: {rel(gact_g, gact_ref):.2e}", flush=True)
# stock down on MY gact
ga_t = Tensor(gact_g.reshape(1,1,17408).astype(np.float16)).contiguous().realize()
d_ref = blk.ffn_down(ga_t).float().realize().numpy().reshape(-1)
y_np2 = hh_np + d_ref
y_g = e.P.down("y", (5120,))
print(f"[k3] y(engine) vs numpy(stock down): {rel(y_g, y_np2):.2e}", flush=True)
print("[k3 done]", flush=True)
