# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P0: verify MTP_SKIP_DUPHEAD effect on load + steady-state VRAM."""
import os, sys, time
os.environ.setdefault("JIT", "2")
sys.path.insert(0, "~/tinygrad-src")
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import GlobalCounters

def vram(): return GlobalCounters.mem_used_per_device.get("NV", 0)/1e9

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, L)
print(f"after_load: {vram():.2f} GB ({time.perf_counter()-t0:.0f}s)", flush=True)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()
# realistic warmup: 8 T=1 steps through rollout_jit (lazy weights realize here)
t = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
temp = Tensor([0.0]); sp = UOp.variable("start_pos", 0, L - 1)
out = None
for i in range(8):
    inp = t[:, i:i+1] if out is None else out
    out = model(inp, sp.bind(L-148+i), temp).realize()
Device["NV"].synchronize()
print(f"after_warmup_steady: {vram():.2f} GB", flush=True)
