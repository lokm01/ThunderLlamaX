# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1"); os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_MAXCTX", "2048"); os.environ.setdefault("MTP_CKPT_DIR", "/tmp/ckpt_empty")
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
model = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", 2048)[0]
blk = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
w = blk.ssm_out.weight
truth = w[0, :64].float().realize().numpy()
import engine0
ds, infos = engine0.parse_gguf()
raw = engine0.read_raw(infos["blk.0.ssm_out.weight"], ds)
b0 = raw[:34]; b1 = raw[34:68]
d0 = np.frombuffer(b0[0:2], dtype="<f2")[0]; d1 = np.frombuffer(b1[0:2], dtype="<f2")[0]
q0 = np.frombuffer(b0[2:34], dtype=np.int8); q1 = np.frombuffer(b1[2:34], dtype=np.int8)
cand_a = np.concatenate([d0*q0, d1*q1]).astype(np.float32)   # byte i = value i
print("[q8] truth[:8]:", truth[:8], flush=True)
print("[q8] cand_a[:8]:", cand_a[:8], flush=True)
print("[q8] cand_a relerr:", float(np.abs(cand_a-truth).max()/np.abs(truth).max()), flush=True)
# transposed nibble-free layouts? try byte = k%32 d = block (k/32) -> same as a. try d float32?
d0f = np.frombuffer(b0[0:4], dtype="<f4")[0]
print("[q8] d as f32 =", d0f, "as f16 =", d0, flush=True)
# maybe blocks are 32B+2B but d at END?
d0e = np.frombuffer(b0[32:34], dtype="<f2")[0]
print("[q8] d-at-end:", d0e, " -> relerr:", float(np.abs(np.concatenate([d0e*np.frombuffer(b0[0:32],dtype=np.int8), d1*np.frombuffer(b1[0:32],dtype=np.int8)]).astype(np.float32)-truth).max()/np.abs(truth).max()), flush=True)

# ---- kernel vs numpy-formula on the REAL engine z (from debug_k3 context) ----
import sys
sys.path.insert(0, "~/tinygrad-metal/engine0")
from tinygrad.tensor import Tensor
from engine0 import GDNBlockEngine
from tinygrad.device import Device
import numpy as np
rng = np.random.default_rng(11)
x_np = (rng.standard_normal((1,1,5120))*0.2).astype(np.float32)
conv_np = (rng.standard_normal((1,3,10240))*0.1).astype(np.float32)
rec_np = (rng.standard_normal((1,48,128,128))*0.1).astype(np.float32)
e = GDNBlockEngine(0)
e.set_inputs(x=x_np.reshape(-1), conv=conv_np.reshape(-1), rec=rec_np.reshape(-1))
Device["NV"].synchronize()
e.run(wait=True)
z_g = e.P.down("z", (6144,), np.float16)
ao_g = e.P.down("attn_out", (5120,), np.float16).astype(np.float32)
# numpy: my formula + half-product chain, rows 0..7 only (fast)
zs = z_g.astype(np.float32)
ao_np = np.zeros(8, np.float32)
for r in range(8):
    rowb = raw[r*6528:(r+1)*6528]
    acc = 0.0
    for blk_i in range(192):
        blk = rowb[blk_i*34:(blk_i+1)*34]
        d = np.frombuffer(blk[0:2], dtype="<f2")[0]
        for l in range(32):
            sv = blk[2+l] - 256 if blk[2+l] > 127 else blk[2+l]
            w = np.float16(np.float16(d) * np.float16(np.float32(sv))).astype(np.float32)
            acc += np.float16(zs[blk_i*32+l] * w).astype(np.float32) if False else np.float16(np.float16(zs[blk_i*32+l]).astype(np.float32) * w).astype(np.float32)
    ao_np[r] = np.float16(acc).astype(np.float32)
print("[q8k] kernel rows[:8]:", ao_g[:8], flush=True)
print("[q8k] numpy  rows[:8]:", ao_np, flush=True)
print("[q8k] kernel-vs-numpy relerr:", float(np.abs(ao_g[:8]-ao_np).max()/np.abs(ao_np).max()), flush=True)

# ---- device-side data integrity check ----
w_dev = e.P.down("w_out", (6528,), np.uint8)
print("[q8d] w_out first 68 bytes match disk:", bytes(w_dev[:68]) == raw[:68], flush=True)
mism = np.where(w_dev != np.frombuffer(raw[:6528], dtype=np.uint8))[0]
print("[q8d] mismatches in row0 bytes:", len(mism), mism[:10], flush=True)
import hashlib
print("[q8d] dev hash:", hashlib.md5(w_dev.tobytes()).hexdigest()[:12], "disk hash:", hashlib.md5(raw[:6528]).hexdigest()[:12], flush=True)
# full buffer hash (33MB)
wfull = e.P.down("w_out", (33423360,), np.uint8)
print("[q8d] full dev hash:", hashlib.md5(wfull.tobytes()).hexdigest()[:12], "disk:", hashlib.md5(raw).hexdigest()[:12], flush=True)

# ---- numpy vs stock on same z ----
from tinygrad.tensor import Tensor as TT
z_t = TT(z_g.reshape(1,1,6144).astype(np.float16)).contiguous().realize()
ao_ref = blk.ssm_out(z_t).float().realize().numpy().reshape(-1)[:8]
print("[q8n] stock rows[:8]:", ao_ref, flush=True)
print("[q8n] numpy-vs-stock relerr:", float(np.abs(ao_np-ao_ref).max()/np.abs(ao_ref).max()), flush=True)
print("[q8n] kernel-vs-stock relerr:", float(np.abs(ao_g[:8]-ao_ref).max()/np.abs(ao_ref).max()), flush=True)
