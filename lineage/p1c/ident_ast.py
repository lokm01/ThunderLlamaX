# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1c: capture pre-linearize AST for r_544 kernels."""
import os, sys, re
os.environ.setdefault("JIT","2")
sys.path.insert(0,"~/tinygrad-src")
L=2048; N=3
PAT="r_544"
MODEL="~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
import tinygrad.engine.realize as R
from tinygrad.helpers import ansistrip
_orig=R.to_program; _seen=set()
def hook(ast, renderer):
    prg=_orig(ast,renderer)
    name=ansistrip(prg.arg.name)
    if re.match(PAT,name) and name not in _seen:
        _seen.add(name)
        with open(f"/tmp/p1c/ast_{name.replace('/','_')}.txt","w") as f:
            f.write(f"name: {name}\nglobal: {prg.arg.global_size} local: {prg.arg.local_size}\n")
            f.write(f"outs={prg.arg.outs} ins={prg.arg.ins}\n")
            try: f.write("AST:\n"+str(ast)[:100000]+"\n")
            except Exception as e: f.write(f"ast dump fail {e}\n")
        print(f"[captured] {name}", flush=True)
    return prg
R.to_program=hook

import time
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
t0=time.perf_counter()
model,_=Transformer.from_gguf(MODEL,L)
print(f"[load] {time.perf_counter()-t0:.1f}s",flush=True)
cfg=model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1,1,cfg.dim))
    ck=getattr(b,"cache_kv",None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape)).realize()
Device["NV"].synchronize()
pos0=L-16
t=Tensor.zeros(1,L,dtype=dtypes.int32).contiguous().realize()
temp=Tensor([0.0]); sp=UOp.variable("start_pos",0,L-1); dev=Device["NV"]
pos=pos0; out=None
for _ in range(8):
    inp=t[:,pos:pos+1] if out is None else out
    out=model(inp,sp.bind(pos),temp).realize(); pos+=1
dev.synchronize()
for _ in range(N):
    out=model(out,sp.bind(pos),temp).realize(); dev.synchronize(); pos+=1
print("[done]", _seen, flush=True)
