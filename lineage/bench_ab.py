# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter

durs={"call":[], "realize":[], "item":[]}
_orig_call = Transformer.__call__
def t_call(self,*a,**k):
    t0=pc(); r=_orig_call(self,*a,**k); durs["call"].append(pc()-t0); return r
Transformer.__call__=t_call
_orig_realize, _orig_item = Tensor.realize, Tensor.item
def t_realize(self,*a,**k):
    t0=pc(); r=_orig_realize(self,*a,**k); durs["realize"].append(pc()-t0); return r
def t_item(self,*a,**k):
    t0=pc(); r=_orig_item(self,*a,**k); durs["item"].append(pc()-t0); return r
Tensor.realize=t_realize; Tensor.item=t_item

t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)

text="Machine learning is a field of study in artificial intelligence concerned with the development "*12
ids=[0]+tok.encode(text)
gen=model.generate(list(ids), chunk_size=32, temperature=0.0)
for _ in range(6): next(gen)

N=12
for k_ in durs: durs[k_].clear()
w0=pc(); kc0=GlobalCounters.kernel_count
for _ in range(N): next(gen)
wall=pc()-w0; kc=GlobalCounters.kernel_count-kc0
print(f"\n== RESULT ({N} tok) ==", flush=True)
print(f"decode: {N/wall:.2f} tok/s ({wall/N*1e3:.1f} ms/tok)", flush=True)
print(f"model __call__: {sum(durs['call'][-N:])/N*1e3:.1f} ms/tok | realize: {sum(durs['realize'][-N:])/N*1e3:.2f} ms | item sync: {sum(durs['item'][-N:])/N*1e3:.2f} ms", flush=True)
print(f"kernels/token: {kc/N:.0f}", flush=True)
print("DONE", flush=True)
