# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1d probe: kernel counts (to_program hook) + fp16 control -> element-bound test."""
import os, sys, struct, time
sys.path.insert(0, "~/tinygrad-src")
import numpy as _np
from tinygrad import Tensor, Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT
import tinygrad.engine.realize as R

MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
DEV = Device.DEFAULT
N, K = 17408, 5120

kern = {}
_orig = R.to_program
def hook(ast, renderer):
    prg = _orig(ast, renderer)
    try:
        nm = str(prg.arg.name).split('~')[0][:44]
        kern[nm] = kern.get(nm, 0)+1
    except Exception: pass
    return prg
R.to_program = hook

def rstr(r):
    n = struct.unpack("<Q", r.read(8))[0]; return r.read(n).decode()
def rdval(r, t):
    if t == 8: return rstr(r)
    if t == 9:
        et = struct.unpack("<I", r.read(4))[0]; n = struct.unpack("<Q", r.read(8))[0]
        return [rdval(r, et) for _ in range(n)]
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
    return struct.unpack("<"+{0:"c",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[t], r.read(sz))[0]
r = open(MODEL,"rb"); assert r.read(4)==b"GGUF"
struct.unpack("<I",r.read(4)); nt=struct.unpack("<Q",r.read(8))[0]; nk=struct.unpack("<Q",r.read(8))[0]
align=32
for _ in range(nk):
    k=rstr(r); t=struct.unpack("<I",r.read(4))[0]; v=rdval(r,t)
    if k=="general.alignment": align=int(v)
infos=[]
for _ in range(nt):
    nm=rstr(r); nd=struct.unpack("<I",r.read(4))[0]
    dims=tuple(struct.unpack("<Q",r.read(8))[0] for _ in range(nd))
    typ=struct.unpack("<I",r.read(4))[0]; off=struct.unpack("<Q",r.read(8))[0]
    infos.append((nm,dims,typ,off))
data_start=(r.tell()+align-1)//align*align
dims,typ,off=dict((n,(d,t,o)) for n,d,t,o in infos)["blk.0.ffn_gate.weight"]
n=prod(dims); ne,nb=_GGML_QUANT[typ]; nbytes=(n//ne)*nb
r.seek(data_start+off); rawg=_np.frombuffer(r.read(nbytes),_np.uint8).copy(); r.close()

t8 = Tensor(rawg).to(DEV).realize()
Wq  = ggml_data_to_tensor(t8, n, typ).reshape(N,K).half()   # lazy dequant over raw
Wf  = Wq.contiguous().realize()                             # fp16 materialized 178MB
Wqq = Wq.cat(Wq, dim=0)                                     # stacked lazy-dequant [34816,K]
x   = Tensor.kaiming_uniform(1,K).half().to(DEV).contiguous().realize()

def run(tag, fn, iters=30):
    j = TinyJit(fn)
    kern.clear()
    j(x.clone()); j(x.clone())      # capture
    ncapt = dict(kern); kern.clear()
    for _ in range(3): j(x.clone())
    Device[DEV].synchronize()
    t0=time.perf_counter()
    for _ in range(iters): j(x.clone())
    Device[DEV].synchronize()
    dt=(time.perf_counter()-t0)/iters
    elems = N*K*fn_elems_mult
    print(f"[time] {tag:28s} {dt*1e3:7.3f} ms | {dt*1e9/elems*1e6:.0f} ns/M-elem | kernels captured: {ncapt}", flush=True)

fn_elems_mult=1
run("2x IQ3 sep", lambda xx: ((xx@Wq.T).realize(), (xx@Wq.T).realize()))
fn_elems_mult=2
run("1x IQ3 stacked [34816]", lambda xx: (xx@Wqq.T).realize())
del Wqq
fn_elems_mult=1
run("fp16 178MB x2", lambda xx: ((xx@Wf.T).realize(), (xx@Wf.T).realize()))
fn_elems_mult=2
Wff = Wf.cat(Wf, dim=0)
run("fp16 356MB x1", lambda xx: (xx@Wff.T).realize())
print("DONE_PROBE2", flush=True)
