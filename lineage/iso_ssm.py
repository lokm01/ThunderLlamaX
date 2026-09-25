# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, time
sys.path.insert(0,"~/tinygrad-src")
from tinygrad import Tensor, UOp, TinyJit, nn
from tinygrad.llm.model import GatedDeltaNetBlock, TransformerConfig
from dataclasses import replace
import functools

# minimal qwen35-ish ssm config
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
        win = Tensor.zeros(B, b.ssm_conv_kernel-1 + T_pad, b.conv_channels).uop
        win = win.after(win[:, :b.ssm_conv_kernel-1].store(cs.cast(win.dtype).uop))
        win = win.after(win[:, b.ssm_conv_kernel-1:b.ssm_conv_kernel-1+T].store(b.attn_qkv(xh).cast(win.dtype).uop))
        conv_window = Tensor(win)
        conv_next = conv_window[:, T:T+b.ssm_conv_kernel-1].cast(conv_in.dtype).contiguous()
        conv_out = functools.reduce(lambda a,c: a+c,
          (conv_window[:, i:i+T_pad] * b.ssm_conv1d["weight"][:, i] for i in range(b.ssm_conv_kernel))).silu()
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
        core = outs[0].stack(*outs[1:], dim=1).contiguous()
        z = (b.ssm_norm(core) * out_gate.silu()).cast(x.dtype).contiguous()
        return b.ssm_out(z.reshape(B, T, -1)), conv_next, state.cast(rec_in.dtype).contiguous()
    return _run(x, sp, conv_in, rec_in)

v=UOp.variable("spx",0,254)
jit=TinyJit(lambda x,sp,cv,rs: ssm_forward(b,x,sp,cv,rs))
cv=Tensor.zeros(1,3,b.conv_channels).realize()
rs=Tensor.zeros(1,b.num_v_heads,b.head_v_dim,b.head_k_dim).realize()
x=Tensor.kaiming_uniform(1,3,5120).float().realize()
for step in range(5):
    o,cn,rn=jit(x,v.bind(step*3),cv,rs)
    o,cn,rn=o.realize(),cn.realize(),rn.realize()
    print(f"step {step}: out={o.shape} cv={float(cn.float().sum().item()):.2f} rs={float(rn.float().sum().item()):.2f}", flush=True)
    x=Tensor.kaiming_uniform(1,3,5120).float().realize()  # fresh input each step
    cv,rs=cn.contiguous().clone(), rn.contiguous().clone()
print("ISO SSM OK", flush=True)
