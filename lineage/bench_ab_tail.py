# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad import dtypes

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter

# ---- instrument realize/item ----
_orig_realize, _orig_item = Tensor.realize, Tensor.item
durs={"realize":[], "item":[]}
def t_realize(self,*a,**k):
    t0=pc(); r=_orig_realize(self,*a,**k); durs["realize"].append(pc()-t0); return r
def t_item(self,*a,**k):
    t0=pc(); r=_orig_item(self,*a,**k); durs["item"].append(pc()-t0); return r
Tensor.realize=t_realize; Tensor.item=t_item

t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s VRAM={GlobalCounters.mem_used_per_device.get('NV',0)/1e9:.2f}GB", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)

text="Machine learning is a field of study in artificial intelligence concerned with the development "*12
ids=[0]+tok.encode(text)
gen=model.generate(list(ids), chunk_size=32, temperature=0.0)
for _ in range(6): next(gen)          # warm

N=12
durs["realize"].clear(); durs["item"].clear()
