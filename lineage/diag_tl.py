# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.tensor import Tensor
import tinygrad.engine.jit as J
from tinygrad.uop.ops import Ops
from tinygrad.device import Device

LAST_TIMELINE=[]
CAPTURE_COUNT=[0]

def dedup_devs(si):
    seen=[]
    for b in si.src[1:]:
        if b.is_bound_var: continue
        for x in (b.device if isinstance(b.device, tuple) else (b.device,)):
            d=Device[x]
            if d not in seen: seen.append(d)
    return seen

def kname(k):
    try:
        # PROGRAM uop: arg may be dict-like or object with name
        a=k.arg if hasattr(k,'arg') else None
        return str(a)[:70]
    except Exception: return "?"

def spy_split(linear, max_batch_size=0):
    TL=[]
    def rec(s): TL.append(s)
    new_src=[]; current=[]; cur_devs=[]
    def flush(breaker):
        if len(current)<=1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current)
        else:
            rec(f"GRAPH<{len(current)}>")
            new_src.append(J.create_graph_call(current))
        current.clear(); cur_devs.clear()
    for si in linear.src:
        k = si.src[0]
        devs = dedup_devs(si)
        graph_t = devs[0].graph.func if isinstance(devs[0].graph, __import__('functools').partial) else devs[0].graph
        can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
        can_extend = can_graph and (not cur_devs or graph_t.supports_uop(cur_devs, si)) and (max_batch_size==0 or len(current)<max_batch_size)
        if not can_extend and current:
            desc=str(k.op)+("" if k.op!=Ops.COPY else f"({k.src[0].device}->{k.device})")
            flush(desc)
            rec(f"**BREAKER:{desc}**")
        (current if can_graph else new_src).append(si)
        if not can_graph:
            rec(f"SOLO:{kname(k)}" if k.op==Ops.PROGRAM else f"SOLOOP:{k.op}")
        elif len(current)==1:
            rec(f"BATCHSTART:{kname(k)}")
        if can_graph: cur_devs = cur_devs + [d for d in devs if d not in cur_devs]
    if current: flush("end")
    CAPTURE_COUNT[0]+=1
    LAST_TIMELINE[:] = TL
    return linear.replace(src=tuple(new_src))

J.graph_split_rewrite = spy_split

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

print(f"\ncaptures total: {CAPTURE_COUNT[0]}, last timeline entries: {len(LAST_TIMELINE)}", flush=True)
with open('/tmp/timeline.txt','w') as fh:
    for e in LAST_TIMELINE: fh.write(e+"\n")
print("timeline written", flush=True)
print("DONE", flush=True)
