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

COPIES=[]   # (src_shape, dtype, nbytes)
LAST_TIMELINE=[]

def dedup_devs(si):
    seen=[]
    for b in si.src[1:]:
        if b.is_bound_var: continue
        for x in (b.device if isinstance(b.device, tuple) else (b.device,)):
            d=Device[x]
            if d not in seen: seen.append(d)
    return seen

def spy_split(linear, max_batch_size=0):
    TL=[]
    new_src=[]; current=[]; cur_devs=[]
    def flush(breaker):
        if len(current)<=1 and not getenv("GRAPH_ONE_KERNEL"): new_src.extend(current)
        else:
            TL.append(f"GRAPH<{len(current)}>")
            new_src.append(J.create_graph_call(current))
        current.clear(); cur_devs.clear()
    for si in linear.src:
        k = si.src[0]
        devs = dedup_devs(si)
        graph_t = devs[0].graph.func if isinstance(devs[0].graph, __import__('functools').partial) else devs[0].graph
        can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
        can_extend = can_graph and (not cur_devs or graph_t.supports_uop(cur_devs, si)) and (max_batch_size==0 or len(current)<max_batch_size)
        if not can_extend and current:
            flush(str(k.op)+("" if k.op!=Ops.COPY else f"({k.src[0].device}->{k.device})"))
        if k.op == Ops.COPY and len(si.src)>2:
            try:
                dst_u, src_u = si.src[1], si.src[2]
                shp = getattr(src_u, 'shape', None)
                dt  = getattr(src_u, 'dtype', None)
                COPIES.append((tuple(shp) if shp else None, str(dt), src_u.max_numel()*dt.itemsize if shp and dt else -1))
                TL.append(f"COPY n={COPIES[-1][2]} shape={shp} {dt}")
            except Exception as e:
                COPIES.append((None,None,-2)); TL.append(f"COPY err {e}")
        (current if can_graph else new_src).append(si)
        if can_graph: cur_devs = cur_devs + [d for d in devs if d not in cur_devs]
    if current: flush("end")
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

print(f"\ncopies captured: {len(COPIES)}", flush=True)
import collections
byshape=collections.Counter((s,d,n) for s,d,n in COPIES)
for (s,d,n),c in sorted(byshape.items(), key=lambda kv:-kv[1])[:15]:
    print(f"  x{c:4d}  shape={s} dtype={d} bytes={n}", flush=True)
with open('/tmp/timeline2.txt','w') as fh:
    for e in LAST_TIMELINE[:120]: fh.write(e+"\n")
print("DONE", flush=True)
