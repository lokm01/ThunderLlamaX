# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W3-100k bootstrap: resume stock model from ckpt@97200 -> feed to P-1 -> snapshot
engine-layout state (per-layer npy) + stock greedy 60-token baseline.
Separate process from the engine (VRAM). Run:
  cd ~/tinygrad-metal && env DEV=NV BEAM=1 MTP_A3_OVERRIDE=... python bootstrap_100k.py
"""
import os, sys, time, json
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1")
os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_HEAD16_DIRECT", "1")
os.environ.setdefault("MTP_MAXCTX", "100352")
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, TransformerBlock
from tinygrad.llm.cli import SimpleTokenizer

GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
CKPT = "~/ckpt100k"
OUT = "~/snap100k"
CTXK = 100352
os.makedirs(OUT, exist_ok=True)
DONEFLAG = f"{OUT}/DONE.txt"

if os.path.exists(DONEFLAG):
    print("[boot] already done"); sys.exit(0)

print("[boot] loading stock model...", flush=True)
t0 = time.perf_counter()
model, kv = Transformer.from_gguf(GGUF, CTXK)
tok = SimpleTokenizer.from_gguf_kv(kv)
print(f"[boot] model loaded {time.perf_counter()-t0:.0f}s", flush=True)

gdn_blks = [(i, b) for i, b in enumerate(model.blk) if isinstance(b, GatedDeltaNetBlock)]
attn_blks = [(i, b) for i, b in enumerate(model.blk) if isinstance(b, TransformerBlock)]
print(f"[boot] {len(gdn_blks)} GDN + {len(attn_blks)} attn", flush=True)

# ---- lazily create state buffers (mtp_v3 _ckpt_load pattern) ----
_dim = model.blk[0].config.dim
_z = Tensor.zeros(1, 1, _dim)
for b in model.blk: b._init_state(_z)
print("[boot] state buffers created", flush=True)

# ---- ckpt resume ----
pos_done = int(open(f"{CKPT}/pos.txt").read())
print(f"[boot] ckpt pos={pos_done}", flush=True)
t0 = time.perf_counter()
for i, b in attn_blks:
    a = np.load(f"{CKPT}/kv_{i}.npy")
    assert a.dtype == np.float16 and a.size == 2*4*CTXK*256, (i, a.shape, a.dtype)
    b.cache_kv.assign(Tensor(a).cast(b.cache_kv.dtype)).realize()
    del a
for i, b in gdn_blks:
    b.conv_state.assign(Tensor(np.load(f"{CKPT}/cv_{i}.npy")).cast(b.conv_state.dtype)).realize()
    b.recurrent_state.assign(Tensor(np.load(f"{CKPT}/rc_{i}.npy")).cast(b.recurrent_state.dtype)).realize()
print(f"[boot] ckpt restored {time.perf_counter()-t0:.0f}s", flush=True)

ids = [0] + tok.encode(open("~/prompt100k.txt").read())
P = len(ids)
print(f"[boot] prompt tokens P={P} (canonical 97810)", flush=True)
np.save(f"{OUT}/ids.npy", np.array(ids, dtype=np.int64))

temp = Tensor([0.0])
def fwd(idv, sp):
    inp = Tensor([[int(idv)]], dtype="int32").contiguous().realize()
    return model.forward(inp, sp, temp).realize()

# ---- feed remaining prompt tokens ids[pos_done .. P-2] ----
t0 = time.perf_counter()
for sp in range(pos_done, P - 1):
    fwd(ids[sp], sp)
    if (sp - pos_done) % 50 == 0:
        print(f"[boot] feed {sp}/{P-2} ({time.perf_counter()-t0:.0f}s)", flush=True)
print(f"[boot] feed done in {time.perf_counter()-t0:.0f}s; pos now {P-1}", flush=True)

# ---- snapshot in ENGINE layout ----
t0 = time.perf_counter()
for i, b in attn_blks:
    a = b.cache_kv.numpy().reshape(2, 4, CTXK, 256).astype(np.float16)
    np.save(f"{OUT}/kv_{i}.npy", a); del a
    print(f"[snap] kv_{i} ({time.perf_counter()-t0:.0f}s)", flush=True)
for i, b in gdn_blks:
    np.save(f"{OUT}/conv_{i}.npy", b.conv_state.float().numpy().reshape(-1).astype(np.float32))
    np.save(f"{OUT}/rec_{i}.npy", b.recurrent_state.float().numpy().reshape(-1).astype(np.float32))
json.dump({"P": P, "CTXK": CTXK}, open(f"{OUT}/meta.json", "w"))
print(f"[snap] states done {time.perf_counter()-t0:.0f}s", flush=True)

# ---- stock greedy 60-token baseline from the same state ----
t0 = time.perf_counter()
out, sp, toks = None, P - 1, []
for k in range(60):
    idv = ids[sp] if out is None else int(out.item())
    out = fwd(idv, sp)
    toks.append(int(out.item())); sp += 1
np.save(f"{OUT}/base_out.npy", np.array(toks, dtype=np.int64))
print(f"[base] 60 stock tokens in {time.perf_counter()-t0:.0f}s: {toks[:12]}...", flush=True)
open(DONEFLAG, "w").write("ok")
print("[boot done]", flush=True)
