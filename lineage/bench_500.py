# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
MC = 1024

t0=time.time()
model, kv = Transformer.from_gguf(f, MC)
tok = SimpleTokenizer.from_gguf_kv(kv)
jenv=jinja2.Environment(); jenv.filters['tojson']=lambda obj,**k:json.dumps(obj,**k)
jenv.globals['bos_token']=tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
jenv.globals['eos_token']=tok.decode([tok.eos_id])
print(f"[load] {time.time()-t0:.2f}s VRAM={GlobalCounters.mem_used_per_device.get('NV',0)/1e9:.2f}GB", flush=True)

text = ("Machine learning is a field of study in artificial intelligence concerned with the development " * 12)  # ~ ~200 tokens
seed = tok.encode(text)
ids = [tok.bos_id if tok.bos_id is not None else 0] + seed
print(f"seed tokens: {len(ids)}", flush=True)

gen = model.generate(list(ids), chunk_size=32, temperature=0.0)

# warm (first tokens build the rollout JIT)
t0=time.time()
for _ in range(6): next(gen)
print(f"[warm 6 tok] {time.time()-t0:.2f}s", flush=True)

# measure steady decode over 30 tokens
times=[]
for _ in range(30):
    s=time.time(); next(gen); times.append(time.time()-s)
dt=sum(times)/len(times)
print(f"WARM_DECODE@{len(ids)+6}ctx: {1/dt:.2f} tok/s ({dt*1000:.1f} ms/tok)", flush=True)
print("DONE", flush=True)