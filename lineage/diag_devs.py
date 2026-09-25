# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.tensor import Tensor
from tinygrad.device import Device

print(f"Device.DEFAULT = {Device.DEFAULT}", flush=True)

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)

from tinygrad.nn.state import get_state_dict
bad=[]
for k,v in get_state_dict(model).items():
    items = v.items() if isinstance(v, dict) else [(None,v)]
    for kk,t in items:
        if isinstance(t, Tensor):
            try: dev = t.device if isinstance(t.device,str) else str(t.device)
            except Exception: dev="?"
            if "NV" not in str(dev): bad.append((f"{k}.{kk}" if kk else k, str(dev), tuple(t.shape), str(t.dtype)))
print(f"\n== NON-NV MODEL TENSORS: {len(bad)} ==", flush=True)
for b in bad[:30]: print(" ", b, flush=True)
print("DONE", flush=True)
