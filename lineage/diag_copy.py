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

REPORT=[]

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
    last_kernel_name=["?"]
    def flush(breaker):
        if len(current)<=1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current)
        else:
            REPORT.append((len(current), breaker, last_kernel_name[0]))
            new_src.append(J.create_graph_call(current))
        current.clear(); cur_devs.clear()
    for si in linear.src:
        k = si.src[0]
        if k.op == Ops.PROGRAM:
            try: last_kernel_name[0] = k.src[0].arg.get("name","?") if isinstance(k.src[0].arg, dict) else str(k.src[0])[:40]
            except Exception: pass
        devs = dedup_devs(si)
        graph_t = devs[0].graph.func if isinstance(devs[0].graph, __import__('functools').partial) else devs[0].graph
        can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
        can_extend = can_graph and (not cur_devs or graph_t.supports_uop(cur_devs, si)) and (max_batch_size==0 or len(current)<max_batch_size)
        if not can_extend and current:
            desc=str(k.op)
            extra=""
            if k.op == Ops.COPY:
                desc=f"COPY({k.src[0].device}->{k.device})"
                # dump lineage of the source operand (si.src[2] is src buffer uop typically)
                try:
                    src_uop = si.src[2]
                    ops_seen=[]
                    def walk(u, d=0):
                        if d>6 or len(ops_seen)>10: return
                        nm = f"{u.op}"
                        if u.op == Ops.PROGRAM:
                            try: nm += ":"+(u.src[0].arg.get("name") if isinstance(u.src[0].arg,dict) else "?")
                            except Exception: pass
                        ops_seen.append(nm[:50])
                        for s in u.src[:3]: walk(s, d+1)
                    walk(src_uop)
                    extra=" <- ".join(ops_seen)
                except Exception as e: extra=f"(lineage err {e})"
            REPORT.append((len(current), desc, extra))
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

print("\n== COPY LINEAGE (first 8 flush events) ==", flush=True)
for sz, br, info in REPORT[:8]:
    print(f"batch={sz:4d} breaker={br}\n   lineage: {info}", flush=True)
print("DONE", flush=True)
