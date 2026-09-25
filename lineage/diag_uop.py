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

DUMPS=[]

def dedup_devs(si):
    seen=[]
    for b in si.src[1:]:
        if b.is_bound_var: continue
        for x in (b.device if isinstance(b.device, tuple) else (b.device,)):
            d=Device[x]
            if d not in seen: seen.append(d)
    return seen

def spy_split(linear, max_batch_size=0):
    new_src=[]; current=[]; cur_devs=[]
    def flush(breaker):
        if len(current)<=1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current)
        else: new_src.append(J.create_graph_call(current))
        current.clear(); cur_devs.clear()
    for si in linear.src:
        k = si.src[0]
        devs = dedup_devs(si)
        graph_t = devs[0].graph.func if isinstance(devs[0].graph, __import__('functools').partial) else devs[0].graph
        can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
        can_extend = can_graph and (not cur_devs or graph_t.supports_uop(cur_devs, si)) and (max_batch_size==0 or len(current)<max_batch_size)
        if not can_extend and current: flush(str(k.op))
        if k.op == Ops.COPY and len(DUMPS)<10:
            try:
                dst_u, src_u = si.src[1], si.src[2]
                DUMPS.append((repr(k)[:120], repr(src_u)[:400], repr(dst_u)[:400]))
            except Exception as e: DUMPS.append((f"err {e}","",""))
        (current if can_graph else new_src).append(si)
        if can_graph: cur_devs = cur_devs + [d for d in devs if d not in cur_devs]
    if current: flush("end")
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

print(f"\ncopy dumps: {len(DUMPS)}", flush=True)
for kop, s, d in DUMPS:
    print(f"\nCOPY: {kop}\n  SRC: {s}\n  DST: {d}", flush=True)
print("DONE", flush=True)
