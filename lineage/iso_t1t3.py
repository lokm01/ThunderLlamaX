# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""T=1 warmup then T=3 replay — the actual spec pattern. JIT=1."""
import os, time, sys
sys.path.insert(0, "~/tinygrad-src")
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv

MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
print(f"JIT={getenv('JIT',1)} BEAM={getenv('BEAM',0)}", flush=True)
t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, 256)
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

v = UOp.variable("sp", 0, 254)
def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    return x.contiguous()
j1 = TinyJit(_fwd)
j3 = TinyJit(_fwd)

for i in range(8):
    h = j1(Tensor([[10+i]], dtype="int32").contiguous(), v.bind(i)).realize()
    print(f"t1 {i} ok {tuple(h.shape)}", flush=True)

pos = 8
for c in range(4):
    try:
        h = j3(Tensor([[100,101,102]], dtype="int32").contiguous(), v.bind(pos)).realize()
        print(f"t3 cyc{c} pos={pos} {tuple(h.shape)} ok", flush=True)
        pos += 3
    except Exception as e:
        print(f"t3 cyc{c} FAIL {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        break
print("ISO_T1T3 DONE", flush=True)
