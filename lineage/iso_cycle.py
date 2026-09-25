# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, time
sys.path.insert(0,"~/tinygrad-src")
from tinygrad import Tensor, UOp, TinyJit, nn
from tinygrad.llm.model import GatedDeltaNetBlock, TransformerConfig
from dataclasses import replace
import functools

base = TransformerConfig(
  num_blocks=1, dim=5120, hidden_dim=17408, n_heads=24, n_kv_heads=4,
  norm_eps=1e-6, vocab_size=1000, head_dim=256, v_head_dim=256,
  rope_theta=1000000.0, rope_dim=256, max_context=256, qk_norm=256)
ssm_layers=tuple((i+1)%4!=0 for i in range(1))
from tinygrad.llm.model import SSMConfig
ssm=SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=6144)
cfg=replace(base, ssm=ssm, ssm_layers=ssm_layers, qk_norm=128)

b=GatedDeltaNetBlock(cfg, ssm)

def ssm_forward(b, x, sp, conv_in, rec_in):
    from tinygrad import function
    @function(precompile=True, allow_implicit=True)
    def _run(x, start_pos, conv_in, rec_in):
        B, T, _ = x.shape
        start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("spg", 0, cfg.max_context-1).bind(start_pos)
        T_pad = x.max_shape[1]
        xh = x.half()
        out_gate = b.attn_gate(xh).reshape(B, T, b.num_v_heads, b.head_v_dim)
        beta = b.ssm_beta(xh).sigmoid().reshape(B, T, b.num_v_heads)
        alpha = b.ssm_alpha(xh)
        log_alpha = ((alpha.float() + b.ssm_dt["bias"]).softplus().reshape(B, T, b.num_v_heads, -1) *
                     b.ssm_a.reshape(b.num_v_heads, -1))
        cs = conv_in
        rows = b.attn_qkv(xh).cast(cs.dtype)
        win = cs.cat(rows, dim=1)
        conv_next = win[:, T:T+b.ssm_conv_kernel-1].contiguous()
        conv_out = functools.reduce(lambda a,c: a+c,
          (win[:, i:i+T_pad] * b.ssm_conv1d["weight"][:, i] for i in range(b.ssm_conv_kernel))).silu()
        q, k, v = conv_out.split([b.q_dim, b.q_dim, b.conv_channels - 2*b.q_dim], dim=-1)
        q, k = (z.reshape(B, T_pad, b.num_k_heads, b.head_k_dim).normalize(dim=-1, eps=1e-6)
                .repeat(1, 1, b.num_v_heads//b.num_k_heads, 1) for z in (q, k))
        v = v.reshape(B, T_pad, b.num_v_heads, b.head_v_dim)
        q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
        q, k, v, beta = q.unsqueeze(-2) * b.head_k_dim**-0.5, k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
        alpha = log_alpha.transpose(1, 2).exp().unsqueeze(-1)
        state = rec_in.float()
        outs=[]
        for t2 in range(T_pad):
            s1 = state * alpha[:, :, t2]
            delta = (v[:, :, t2] - (s1*k[:, :, t2]).sum(-1, keepdim=True)) * beta[:, :, t2]
            state = s1 + delta * k[:, :, t2]
            outs.append((state * q[:, :, t2]).sum(-1))
        core = (outs[0].reshape(1,b.num_v_heads,b.head_v_dim) if len(outs)==1 else
                outs[0].stack(*outs[1:], dim=1)).contiguous()
        z = (b.ssm_norm(core) * out_gate.silu()).cast(x.dtype).contiguous()
        return b.ssm_out(z.reshape(B, T, -1)), conv_next, state.cast(rec_in.dtype).contiguous()
    return _run(x, sp, conv_in, rec_in)

v=UOp.variable("spx",0,254)
jit3=TinyJit(lambda x,sp,cv,rs: ssm_forward(b,x,sp,cv,rs))   # T=3
jit1=TinyJit(lambda x,sp,cv,rs: ssm_forward(b,x,sp,cv,rs))   # T=1

def mkx(T): return Tensor.kaiming_uniform(1,T,5120).float().realize()
cv=Tensor.zeros(1,3,b.conv_channels).realize()
rs=Tensor.zeros(1,b.num_v_heads,b.head_v_dim,b.head_k_dim).realize()

# cycle like spec: probe(T=K+1=3) then commit(T=1)
for cyc in range(3):
    o3,c3,r3=jit3(mkx(3),v.bind(cyc*10),cv,rs)
    o3,c3,r3=o3.realize(),c3.realize(),r3.realize()
    print(f"cyc{cyc} probe types: {type(o3).__name__},{type(c3).__name__},{type(r3).__name__}", flush=True)
    o1,c1,r1=jit1(mkx(1),v.bind(cyc*10),cv,rs)
    o1,c1,r1=o1.realize(),c1.realize(),r1.realize()
    print(f"cyc{cyc} commit types: {type(o1).__name__},{type(c1).__name__},{type(r1).__name__}", flush=True)
    cv,rsc=c3.contiguous().clone(), r3.contiguous().clone()
print("ISO CYCLE TEST OK", flush=True)
