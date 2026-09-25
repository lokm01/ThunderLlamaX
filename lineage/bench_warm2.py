# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time, sys
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
t0=time.time()
model, kv = Transformer.from_gguf(f, MC)
tok = SimpleTokenizer.from_gguf_kv(kv)
jenv=jinja2.Environment(); jenv.filters['tojson']=lambda obj,**k:json.dumps(obj,**k)
jenv.globals['bos_token']=tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
jenv.globals['eos_token']=tok.decode([tok.eos_id])
print(f"[load mc={MC}] {time.time()-t0:.1f}s", flush=True)

# seed with a real prompt then stream long-form generation to grow context
seed = tok.encode("Write a detailed technical explanation of transformer neural network architectures.")
ids = [tok.bos_id if tok.bos_id is not None else 0] + seed
gen = model.generate(list(ids), chunk_size=32, temperature=0.0)

# warm up the rollout graph with a handful of tokens
t0=time.time()
for _ in range(5):
    next(gen)
print(f"[warmup 5 tok] {time.time()-t0:.1f}s", flush=True)

# measure decode throughput over windows at increasing context lengths
# each decode step's cost grows with context (attention over growing KV),
# so we measure avg tok/s within a window anchored at each target ctx length
targets = [1024, 4096, 16384, 32768, 65536, min(MC-512, 90000)]
cur_len = len(ids) + 5   # approximate current position (after seed + 5 warmup)
results = []
IW = 5  # tokens per measurement window
for target in targets:
    # stream until we reach the target context
    while cur_len < target:
        next(gen)
        cur_len += 1
    # measure window at this context
    times = []
    for _ in range(IW):
        s=time.time(); next(gen); times.append(time.time()-s); cur_len += 1
    dt = sum(times)/len(times)
    results.append((target, 1.0/dt))
    print(f"ctx={target:>6}  {1.0/dt:6.2f} tok/s  ({dt*1000:7.1f} ms/tok)", flush=True)

print("done", flush=True)
