# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Minimal sp-boundary repro: T=8 chunk forwards at increasing sp until device fault.
Env: DEV=NV BEAM=1 MTP_T3_LAZY=1 MTP_SEQ_ATTN=1 (NO PROBE_RO — chunk semantics)"""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
L = int(os.getenv("REPRO_L", "2048"))
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device

model, kv = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", L)
cfg = model.blk[-1].config
print(f"[cfg] block max_context={cfg.max_context} L={L}", flush=True)
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()
print("[load ok]", flush=True)

dev = Device["NV"]
v_sp = UOp.variable("rsp", 0, L - 9)
t0 = time.perf_counter()
sp = 0
while sp + 8 <= L:
    toks = Tensor.zeros(1, 8, dtype=dtypes.int32).contiguous().realize()
    dev.allocator._copyin(toks.uop.buf_uop.buffer._bufs["NV"], memoryview(np.zeros((1, 8), dtype=np.int32).tobytes()).cast("B"))
    x = model.token_embd(toks).float()
    for b in model.blk:
        x = b(x, v_sp.bind(sp))
    x = x.contiguous().realize()
    dev.synchronize()
    print(f"[sp {sp}] ok ({time.perf_counter()-t0:.1f}s) h[0,0,:3]={x.numpy()[0,0,:3]}", flush=True)
    if os.getenv("REPRO_DRAIN"):
        dev.allocator.free_cache()
        import gc; gc.collect()
    sp += int(os.getenv("REPRO_STEP", "8"))
print("COMPLETED ALL", flush=True)
