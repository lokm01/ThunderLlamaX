# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1d verdict bench: interleaved sep-vs-stacked IQ3 raw-block GEMV, one process."""
import os, sys, struct, time, statistics
sys.path.insert(0, "~/tinygrad-src")
import numpy as _np
from tinygrad import Tensor, Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import prod, getenv
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT

MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
DEV = Device.DEFAULT; N, K = 17408, 5120
ITERS = int(getenv("ITERS", "30")); ROUNDS = int(getenv("ROUNDS", "5"))

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
by=dict((n,(d,t,o)) for n,d,t,o in infos)
def raw(name):
    dims,typ,off = by[name]; n=prod(dims); ne,nb=_GGML_QUANT[typ]
    r.seek(data_start+off); return _np.frombuffer(r.read((n//ne)*nb),_np.uint8).copy(), n, typ
rg,ng,tg = raw("blk.0.ffn_gate.weight")
ru,nu,tu = raw("blk.0.ffn_up.weight")
t8g = Tensor(rg).to(DEV).realize(); t8u = Tensor(ru).to(DEV).realize()
Wg = ggml_data_to_tensor(t8g, ng, tg).reshape(N,K).half()
Wu = ggml_data_to_tensor(t8u, nu, tu).reshape(N,K).half()
Wgu = ggml_data_to_tensor(t8g.cat(t8u), ng+nu, tg).reshape(2*N,K).half()
x = Tensor.kaiming_uniform(1,K).half().to(DEV).contiguous().realize()

j_sep = TinyJit(lambda xx: ((xx@Wg.T).realize(), (xx@Wu.T).realize()))
j_stk = TinyJit(lambda xx: (xx@Wgu.T).realize())
for j in (j_sep, j_stk): j(x.clone()); j(x.clone())

def t_of(j):
    for _ in range(3): j(x.clone())
    Device[DEV].synchronize()
    t0=time.perf_counter()
    for _ in range(ITERS): j(x.clone())
    Device[DEV].synchronize()
    return (time.perf_counter()-t0)/ITERS

seps, stks = [], []
for i in range(ROUNDS):
    seps.append(t_of(j_sep)); stks.append(t_of(j_stk))
    print(f"[r{i}] sep={seps[-1]*1e3:.3f}ms stk={stks[-1]*1e3:.3f}ms", flush=True)
ms, mm = statistics.mean(seps), statistics.mean(stks)
print(f"[VERDICT] sep mean {ms*1e3:.3f}ms (min {min(seps)*1e3:.3f}) | stacked mean {mm*1e3:.3f}ms (min {min(stks)*1e3:.3f}) | delta {(ms-mm)/ms*100:+.1f}%", flush=True)
# bytes view: 68.24MB raw per cycle
byts = 2*(N*K//256)*98
print(f"[bw] effective raw-byte BW: sep {byts/ms/1e9:.0f} GB/s | stacked {byts/mm/1e9:.0f} GB/s (fp16 wall = 447)", flush=True)
print("DONE_FINAL", flush=True)
