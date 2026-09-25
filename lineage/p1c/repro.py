# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1c repro: which model subgraph emits r_544_*? Load once, probe pieces, drain names.
Run: cd ~/tinygrad-metal && DEV=NV BEAM=1 JIT=2 KERNEL_HIST=1 ~/tg311/bin/python /tmp/p1c/repro.py
"""
import os, sys, time
os.environ.setdefault("JIT","2")
sys.path.insert(0,"~/tinygrad-src")
L=512
MODEL="~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, TransformerBlock
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.helpers import ansistrip
dev = Device["NV"]
def drain():
    out=[]
    for st,en,name,dn,pk in getattr(dev,"sig_prof_records",[]):
        try: dt=float(en.timestamp)-float(st.timestamp)
        except Exception: continue
        if dt>0: out.append((ansistrip(name),dt/1e3))
    del dev.sig_prof_records[:]
    return out
t0=time.perf_counter()
model,_ = Transformer.from_gguf(MODEL,L)
print(f"[load] {time.perf_counter()-t0:.1f}s",flush=True)
cfg=model.blk[-1].config
for b in model.blk: b._init_state(Tensor.zeros(1,1,cfg.dim))
dev.synchronize(); drain()

gdn = next(b for b in model.blk if isinstance(b,GatedDeltaNetBlock))
att = next(b for b in model.blk if isinstance(b,TransformerBlock))
x = Tensor.randn(1,1,cfg.dim).half().realize()
h32 = Tensor.randn(1,1,cfg.dim).realize()

def probe(tag, fn, n=5):
    drain()
    try:
        for _ in range(n): ys = fn().realize()
        dev.synchronize()
    except Exception as e:
        print(f"\n== {tag}: FAILED {type(e).__name__}: {e}"); return
    agg={}
    for name,dt in drain(): agg[name]=agg.get(name,0)+dt/n
    hits={k:round(v,3) for k,v in agg.items() if "544" in k}
    print("== %s: R544 -> %s" % (tag, hits or "none"))
    top=sorted(agg.items(),key=lambda i:-i[1])[:4]
    for k,v in top: print(f"     {v:8.4f} ms  {k}")

probe("attn_norm(dim)", lambda: att.attn_norm(h32))
probe("output_norm(dim)", lambda: model.output_norm(h32))
probe("ssm_norm(head_v_dim)", lambda: gdn.ssm_norm(Tensor.randn(1,1,gdn.num_v_heads,gdn.head_v_dim).half()))
probe("q normalize", lambda: list(gdn._attention_t1(x, Tensor.ones(1), False))[0] if False else Tensor.randn(1,1,8,128).normalize(dim=-1))
print(f"[gdn cfg] dim={cfg.dim} conv_ch={gdn.conv_channels} q_dim={gdn.q_dim} nvh={gdn.num_v_heads} nkh={gdn.num_k_heads} hvd={gdn.head_v_dim} hkd={gdn.head_k_dim}")
probe("attn_qkv", lambda: gdn.attn_qkv(x))
probe("ssm_out", lambda: gdn.ssm_out(Tensor.randn(1,1,gdn.ssm_out.weight.shape[1]).half()))
probe("attn_gate", lambda: gdn.attn_gate(x))
probe("ssm_beta", lambda: gdn.ssm_beta(x))
probe("ssm_alpha", lambda: gdn.ssm_alpha(x))
probe("FFN(gdn) full", lambda: gdn._feed_forward(h32))
probe("ffn_gate only", lambda: h32.linear(gdn.ffn_gate.weight.transpose()))
probe("ffn_up only", lambda: h32.linear(gdn.ffn_up.weight.transpose()))
probe("ffn_down only", lambda: h32.linear(gdn.ffn_down.weight.transpose()))
