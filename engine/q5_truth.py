# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
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
blk = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
w = blk.attn_qkv.weight     # lazy dequant [10240, 5120]
truth = w[0, :256].float().realize().numpy()
print("[truth] first 8:", truth[:8], flush=True)
import engine0
ds, infos = engine0.parse_gguf()
raw = engine0.read_raw(infos["blk.0.attn_qkv.weight"], ds)[:176]
d = np.frombuffer(raw[0:2], dtype="<f2")[0]; dm = np.frombuffer(raw[2:4], dtype="<f2")[0]
s = np.frombuffer(raw[4:16], dtype=np.uint8)
sc = np.zeros(8, np.float32); mn = np.zeros(8, np.float32)
for i in range(4):
    sc[i] = s[i] & 63; mn[i] = s[4+i] & 63
    sc[4+i] = (s[8+i] & 0xF) | ((s[i] >> 6) << 4)
    mn[4+i] = (s[8+i] >> 4) | ((s[4+i] >> 6) << 4)
qh = np.frombuffer(raw[16:48], dtype=np.uint8)
qs = np.frombuffer(raw[48:176], dtype=np.uint8)
def build(byteperm, nswap, qhmode):
    out = np.zeros(256, np.float32)
    for k in range(256):
        i = k >> 5; j = k & 31
        if byteperm == "a": byte = (i>>1)*32 + (i&1)*16 + (j>>1)
        elif byteperm == "b": byte = i*16 + (j>>1)
        else: byte = j//1*0 + (k>>1)
        nib = (j&1) ^ nswap
        qv = (qs[byte] >> (4 if nib else 0)) & 0xF
        if qhmode == 0: qv += ((qh[k & 31] >> (i & 7)) & 1) << 4
        elif qhmode == 1: qv += ((qh[byte & 31] >> (i & 7)) & 1) << 4
        else: qv += ((qh[k & 31] >> ((k>>5)&7)) & 1) << 4
        out[k] = d*sc[i]*qv - dm*mn[i]
    return out
best = None
for bp in ("a","b","c"):
    for ns in (0,1):
        for qm in (0,1,2):
            v = build(bp, ns, qm)
            err = float(np.abs(v - truth).max() / max(np.abs(truth).max(),1e-9))
            tag = f"byte={bp} nswap={ns} qh={qm}"
            if err < 1e-5: print("[FIT] MATCH", tag, flush=True)
            if best is None or err < best[0]: best = (err, tag)
print("[FIT] best:", best, flush=True)
# also: does pure sub-major with j mapping differ? print truth vs b/0/0 first mismatch
v = build("b",0,0)
mm = np.where(np.abs(v-truth) > 1e-5*np.abs(truth).max())[0]
print("[FIT] b/0/0 mismatch ks:", mm[:10], "of 256", flush=True)

print("[raw] d=", d, "dm=", dm)
print("[raw] scales s[0..11] =", list(s), flush=True)
print("[raw] qh[0..7] =", [hex(x) for x in qh[:8]], flush=True)
print("[raw] qs[0..15] =", [hex(x) for x in qs[:16]], flush=True)
t32 = truth[:32]
print("[raw] truth[0:32] =", np.array2string(t32, precision=6, max_line_width=200), flush=True)
print("[raw] truth[32:64] =", np.array2string(truth[32:64], precision=6, max_line_width=200), flush=True)
A = d  # try: if q=0 gives -dm*mn: min of sub0?
