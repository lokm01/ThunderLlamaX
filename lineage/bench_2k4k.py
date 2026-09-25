# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
MC = 5000   # fits 2k/4k with headroom for growth test

t0=time.time()
model, kv = Transformer.from_gguf(f, MC)
tok = SimpleTokenizer.from_gguf_kv(kv)
jenv=jinja2.Environment(); jenv.filters['tojson']=lambda obj,**k:json.dumps(obj,**k)
jenv.globals['bos_token']=tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
jenv.globals['eos_token']=tok.decode([tok.eos_id])
print(f"[load] {time.time()-t0:.1f}s  VRAM={GlobalCounters.mem_used_per_device.get('NV',0)/1e9:.2f}GB", flush=True)

# persistent prompt for streaming (keeps model hot)
seed = tok.encode("Write a detailed technical article about transformer neural networks, attention mechanisms, and self-supervised learning.")
ids = [tok.bos_id if tok.bos_id is not None else 0] + seed
print(f"seed tokens: {len(ids)}", flush=True)

gen = model.generate(list(ids), chunk_size=32, temperature=0.0)

# warm rollout
t0=time.time()
for _ in range(8): next(gen)
print(f"[warm 8 tok] {time.time()-t0:.1f}s", flush=True)

cur = len(ids) + 8

# measure a steady window at each target context (2k, 4k)
for target, wnd in [(2048,10), (4096,10)]:
    while cur < target:
        next(gen); cur += 1
    times=[]
    for _ in range(wnd):
        s=time.time(); next(gen); times.append(time.time()-s); cur += 1
    dt=sum(times)/len(times)
    print(f"CONTEXT~{target:>6}: {1/dt:6.2f} tok/s  ({dt*1000:6.1f} ms/tok)", flush=True)

print("DONE", flush=True)