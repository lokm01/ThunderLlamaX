# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Single-slice head cast isolation: cast one Q5_K slice (34816x5120) fp16.
If this faults -> the cast/dequant kernel at slice shapes; if clean -> the mtp context."""
import sys, os
sys.path.insert(0, "~/tinygrad-src")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("BEAM", "1")
import time
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.device import Device

L = int(os.getenv("ISO_L", "1024"))
model, kv = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", L)
print("[load ok]", flush=True)
Device["NV"].synchronize()

W = model.output.weight
rows, cols = W.shape[0], W.shape[1]
SLICE = int(os.getenv("ISO_SLICE", "34816"))
r0 = int(os.getenv("ISO_R0", "0"))
r1 = min(r0 + SLICE, rows)
t0 = time.perf_counter()
chunk = W[r0:r1].cast(dtypes.float16).contiguous().realize()
Device["NV"].synchronize()
n = chunk.numpy()
print(f"[cast] rows {r0}:{r1} OK {time.perf_counter()-t0:.1f}s shape={n.shape} dtype={n.dtype} absmax={abs(n).max():.3f}", flush=True)
print("ISO-CAST-PASS", flush=True)
