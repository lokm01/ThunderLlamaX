# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time, collections
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.tensor import Tensor
import tinygrad.engine.jit as J
from tinygrad.uop.ops import Ops
from tinygrad.device import Device

FLUSH=[]  # (batch_size, breaker_desc)

def spy_split(linear, max_batch_size=0):
    orig_create = J.create_graph_call
    def create(batch):
        FLUSH.append((len(batch), "GRAPH"))
        return orig_create(batch)
    _saved = J.create_graph_call
    J.create_graph_call = create
    try:
        new_src=[]; current=[]; cur_devs=[]
        def flush(breaker):
            if len(current)<=1 and not getenv("GRAPH_ONE_KERNEL"):
                new_src.extend(current)
            else:
                FLUSH.append((len(current), f"BATCH{breaker}"))
                new_src.append(J.create_graph_call(current))
            current.clear(); cur_devs.clear()
        for si in linear.src:
            devs = J.__dict__ and None or dedup_devs(si)
            graph_t = devs[0].graph.func if isinstance(devs[0].graph, __import__('functools').partial) else devs[0].graph
            can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
            can_extend = can_graph and (not cur_devs or graph_t.supports_uop(cur_devs, si)) and (max_batch_size==0 or len(current)<max_batch_size)
            if not can_extend and current:
                k = si.src[0]
                desc = str(k.op)
                if k.op == Ops.COPY: desc += f"({k.src[0].device}->{k.device})"
                elif k.op in (Ops.CUSTOM_FUNCTION,): desc += f":{k.arg}"
                flush(desc)
            (current if can_graph else new_src).append(si)
            if can_graph: cur_devs = cur_devs + [d for d in devs if d not in cur_devs]
        if current: flush("end")
        return linear.replace(src=tuple(new_src))
    finally:
        J.create_graph_call = _saved

def dedup_devs(si):
    seen=[]
    for b in si.src[1:]:
        if b.is_bound_var: continue
        for x in (b.device if isinstance(b.device, tuple) else (b.device,)):
            d=Device[x]
            if d not in seen: seen.append(d)
    return seen

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

print("\n== FLUSH ANALYSIS ==", flush=True)
batches=[f for f in FLUSH if f[1]=="GRAPH"]
breaks=collections.Counter(f[1] for f in FLUSH if f[1]!="GRAPH")
print(f"total graphs created: {len(batches)}, kernels in graphs: {sum(b[0] for b in batches)}", flush=True)
print("flush breakers:", dict(breaks), flush=True)
# decode-token-only estimate: last capture dominates; show last 80 entries
for sz,r in FLUSH[-70:]:
    print(f"  {sz:5d} {r}", flush=True)
print("DONE", flush=True)
