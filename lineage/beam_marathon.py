# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s VRAM={GlobalCounters.mem_used_per_device.get('NV',0)/1e9:.2f}GB", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)
jenv=jinja2.Environment(); jenv.filters['tojson']=lambda obj,**k:json.dumps(obj,**k)
jenv.globals['bos_token']=tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
jenv.globals['eos_token']=tok.decode([tok.eos_token_id] if hasattr(tok,'eos_token_id') else [tok.eos_id])

ids=[0]+tok.encode("The theory of relativity transformed our understanding of space and time.")
gen=model.generate(list(ids), chunk_size=32, temperature=0.0)

N=40
times=[]
text=[]
for i in range(N):
    t0=pc(); t=next(gen); dt=pc()-t0
    times.append(dt); text.append(t)
    if i%5==4:
        print(f"tok {i+1:3d}: {dt*1e3:7.0f} ms  (last5 avg {sum(times[-5:])/5:.2f} s)  [{time.strftime('%H:%M:%S')}]", flush=True)
wall=sum(times)
print(f"\n== BEAM MARATHON RESULT ==", flush=True)
print(f"decode: {N/wall:.2f} tok/s ({wall/N*1e3:.0f} ms/tok)", flush=True)
print("generated:", " ".join(tok.decode([t]) for t in text), flush=True)
print("DONE", flush=True)
