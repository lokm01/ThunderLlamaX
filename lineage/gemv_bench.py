# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.nn.state import get_parameters

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)

# biggest 2D params
ps=[p for p in get_parameters(model) if p.ndim==2]
ps.sort(key=lambda p:-p.numel())
for p in ps[:5]:
    print(f"param: shape={p.shape} dtype={p.dtype} bytes={p.nbytes()/1e9:.3f}GB", flush=True)

w=ps[0]
n_in=w.shape[1]
x=Tensor.zeros(1, n_in, dtype=dtypes.default_float).cast(dtypes.float16)
y=(x@w.T).realize(); y.float().sum().item()

ts=[]
for _ in range(20):
    t0=pc(); yy=(x@w.T).realize(); ts.append(pc()-t0)
yy.float().sum().item()
enq=sum(ts)/len(ts)
t0=pc()
for _ in range(30): yy=(x@w.T).realize()
yy.float().sum().item(); tot=pc()-t0; b2b=tot/30
gb=(w.nbytes()+x.nbytes()+yy.nbytes())/1e9
print(f"\n== QUANT GEMV {tuple(w.shape)} ==", flush=True)
print(f"enqueue/launch : {enq*1e3:.3f} ms", flush=True)
print(f"back-to-back   : {b2b*1e3:.3f} ms -> {gb/b2b:.0f} GB/s effective", flush=True)

# fp16 reference (dequant cost isolated)
try:
    w16=w.cast(dtypes.float16).realize(); w16.item()
    t0=pc()
    for _ in range(30): zz=(x@w16.T).realize()
    zz.float().sum().item(); per16=(pc()-t0)/30
    gb16=(w16.nbytes()+x.nbytes()+zz.nbytes())/1e9
    print(f"fp16 GEMV      : {per16*1e3:.3f} ms -> {gb16/per16:.0f} GB/s effective", flush=True)
except Exception as e:
    print("fp16 ref failed:", e, flush=True)
print("DONE", flush=True)
