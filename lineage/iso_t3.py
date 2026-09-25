# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Isolate: can stock TinyJit(forward) at T=3 replay without faulting?"""
import os, time, sys
sys.path.insert(0, "~/tinygrad-src")
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import GlobalCounters, getenv

MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
print(f"JIT={getenv('JIT',1)} BEAM={getenv('BEAM',0)}", flush=True)
t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, 256)
print(f"[load] {time.perf_counter()-t0:.1f}s mem={ {k:round(v/1e9,2) for k,v in GlobalCounters.mem_used_per_device.items()} }", flush=True)

v = UOp.variable("sp", 0, 254)
def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    return x.contiguous()
j1 = TinyJit(_fwd)
j3 = TinyJit(_fwd)

def step1(tok, pos):
    h = j1(Tensor([[tok]], dtype="int32").contiguous(), v.bind(pos)).realize()
    return int(h.shape[1])

def step3(toks, pos):
    h = j3(Tensor([toks], dtype="int32").contiguous(), v.bind(pos)).realize()
    return tuple(h.shape)

# T=1 warmup like generate
for i in range(8):
    step1(10+i, i)
    print(f"t1 {i} ok", flush=True)

# T=3 repeatedly
pos = 8
for c in range(5):
    try:
        sh = step3([100, 101, 102], pos)
        print(f"t3 cyc{c} pos={pos} {sh} ok", flush=True)
        pos += 3
    except Exception as e:
        print(f"t3 cyc{c} FAIL {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        break
print("ISO_T3 DONE", flush=True)
