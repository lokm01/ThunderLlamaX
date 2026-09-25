# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time, traceback, collections
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.tensor import Tensor
from tinygrad.device import Device

TRACES=collections.Counter()

# Tensor creation with explicit device=PYTHON happens via Tensor(uop) / Tensor.zeros(...,device) etc.
# Hook Tensor.__init__ is messy; hook the point where a Tensor gets a lazydata on PYTHON:
_orig_tensor_new = Tensor.__new__
def spy_new(cls, *args, **kwargs):
    return _orig_tensor_new(cls, *args, **kwargs)
# Instead: scan for uop.device=='PYTHON' at schedule entry: patch Tensor.realize
_orig_realize = Tensor.realize
def spy_realize(self, *a, **k):
    try:
        u = self.uop
        devs = set()
        for uu in u.toposort():
            d = uu.device if isinstance(uu.device, str) else None
            if d == "PYTHON": devs.add(d)
        if devs and len(TRACES) < 6:
            stk = "".join(traceback.format_stack(limit=14)[:-1])
            key = tuple(f.lstrip() for f in stk.splitlines() if "tinygrad" not in f and ("model.py" in f or "jit.py" in f or "llm" in f))[-4:]
            if key: TRACES[key] += 1
    except Exception: pass
    return _orig_realize(self, *a, **k)
Tensor.realize = spy_realize

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)

text="Machine learning is a field of study in artificial intelligence concerned with the development "*12
ids=[0]+tok.encode(text)
gen=model.generate(list(ids), chunk_size=32, temperature=0.0)
for _ in range(6): next(gen)

print(f"\n== PYTHON-device realize traces ({len(TRACES)}) ==", flush=True)
for key,c in list(TRACES.items())[:6]:
    print(f"count={c}", flush=True)
    for l in key: print("   ", l, flush=True)
print("DONE", flush=True)
