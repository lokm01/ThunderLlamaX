# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import struct, sys
fn = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
f=open(fn,"rb")
def rd(fmt): s=struct.calcsize(fmt); v=struct.unpack(fmt,f.read(s)); return v if len(v)>1 else v[0]
magic,ver=rd("<4sI"); nt,nkv=rd("<QQ")
SC={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
FM={0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}
def rstr(): n=rd("<Q"); return f.read(n).decode(errors="replace")
def skipval(t):
    if t==8: rstr(); return
    if t==9:
        et=rd("<I"); n=rd("<Q")
        if et==8:
            for _ in range(n): rstr()
        else: f.seek(n*SC[et],1)
        return
    f.seek(SC[t],1)
for _ in range(nkv):
    k=rstr(); vt=rd("<I"); skipval(vt)
TENUM={0:("F32",4),1:("F16",2),2:("Q4_0",18),7:("Q8_0",34),16:("Q2_K",84),17:("Q3_K_S",110),18:("Q3_K_M",110),19:("Q3_K_L",110),20:("Q4_K_S",144),21:("Q4_K_M",144),22:("Q5_K_S",176),23:("Q5_K_M",176),24:("Q6_K",210),25:("IQ2_XXS",42),26:("IQ2_XS",50),28:("Q3_K_XS",110),29:("IQ1_S",50),30:("IQ4_NL",136),31:("IQ3_S",106),32:("IQ2_S",58),33:("IQ4_XS",136),36:("IQ3_XXS",98),38:("IQ1_M",56)}
tsize={"F32":4,"F16":2,"BF16":2,"Q8_0":34,"Q4_0":18,"Q4_K":144,"Q5_K":176,"Q6_K":210,"Q2_K":84,"Q3_K":110,"IQ3_XXS":98,"IQ1_S":50,"IQ4_XS":136,"IQ4_NL":136,"TQ1_0":26,"TQ2_0":41}
import collections
agg=collections.Counter()
for _ in range(nt):
    n=rstr(); nd=rd("<I"); dims=[rd("<Q") for _ in range(nd)]; tt=rd("<I"); off=rd("<Q")
    tn,ts=TENUM.get(tt,(str(tt),None))
    t=tn
    if any(s in n for s in ["blk.0.","blk.47.","blk.48.","output.weight","blk.63."]):
        ne=1
        for d in dims: ne*=d
        ts=tsize.get(t)
        raw = ts*ne/32 if ts in (34,18,144,176,210,84,110,98,50,136,26,41) else ts*ne if ts else -1
        agg[(dims_tuple:=tuple(dims),t)]+=1
        print(f"{n}: dims={list(dims)} type={t} raw_bytes={int(raw)}")
