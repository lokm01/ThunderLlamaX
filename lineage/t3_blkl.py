# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""T=3 probe block-limited attribution: forward blk[0:NBLK] only (no head). JIT=1."""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("HIST_L", "2048"))
N = int(os.getenv("N_PASS", "3"))
NBLK = int(os.getenv("NBLK", "64"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

from tinygrad.llm.model import Transformer
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context

model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
Device["NV"].synchronize()
model.blk = model.blk[:NBLK]
print("[cfg] NBLK=%d attn=%d" % (NBLK, sum(1 for b in model.blk if hasattr(b, chr(99)+chr(97)+chr(99)+chr(104)+chr(101)+chr(95)+chr(107)+chr(118)))), flush=True)

v_sp = UOp.variable("mtp_sp", 0, L - 4)
pos = L - 3*N - 8
tt = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
toks = tt[:, pos:pos+3].contiguous().realize()

def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    return x.contiguous()

probe_j = TinyJit(_fwd)
with Context(JIT=1):
    h = probe_j(toks, v_sp.bind(pos)).realize()
Device["NV"].synchronize()
print("[warm] OK", flush=True)
for i in range(N):
    t0 = time.time()
    with Context(JIT=1):
        h = probe_j(toks, v_sp.bind(pos)).realize()
    Device["NV"].synchronize()
    print(f"[cyc {i}] {time.time()-t0:.3f}s", flush=True)
    pos += 3
print(f"BLKLIM PASS NBLK={NBLK}", flush=True)
