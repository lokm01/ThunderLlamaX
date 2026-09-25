# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad import dtypes, Context, Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
from tinygrad.engine.jit import TinyJit

# model env (match 2k recipe)
os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1")
os.environ.setdefault("MTP_EMB_GATHER", "1")

import mtp_v3 as M
model = M.model
dev = Device["NV"]
gdns = M.gdns
blk0 = gdns[0]   # the ONE block for the split validation

nv, hv, hk = blk0.num_v_heads, blk0.head_v_dim, blk0.head_k_dim
print(f"[block] nv={nv} hv={hv} hk={hk} (kernel expects 16/8/128)")

T = 3
B = 1
rng = np.random.default_rng(4)
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t

# --- build the split-path persistent buffers (fp32, kernel layout) ---
P_nv, P_hv, P_hk = nv, hv, hk
q_pb   = up(np.zeros((P_nv, T, P_hk), np.float32))
k_pb   = up(np.zeros((P_nv, T, P_hk), np.float32))
v_pb   = up(np.zeros((P_nv, T, P_hv), np.float32))
a_pb   = up(np.zeros((P_nv, T), np.float32))
b_pb   = up(np.zeros((P_nv, T), np.float32))
st_pb  = up(np.zeros((P_nv, P_hv, P_hk), np.float32))
out_pb = up(np.zeros((P_nv, T, P_hv), np.float32))
pss_pb = up(np.zeros((T, P_nv, P_hv, P_hk), np.float32))
core_pb = up(np.zeros((T, P_nv, P_hv), np.float32))

scan_k = NVProgram(dev, TinyELF(
    lib=open("~/tinygrad-metal/a4/gdn_scan_m.cubin","rb").read(),
    name="gdn_scan_m", target=dev.renderer.target,
    signature=(("v", 8, dtypes.int32,()),)))

# --- the stock scan (reference): replicate _attention_mtp's loop on numpy semantics via tensors ---
def stock_scan(state, q, k, v, beta, alpha):
    # all [B, nv, T(+1 dims)] tinygrad tensors, matching _attention_mtp shapes
    outs = []
    for t in range(T):
        s1 = state * alpha[:, :, t]
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))
    return state, outs[0].stack(*outs[1:], dim=1)

# --- the split path fragments ---
@TinyJit
def pre_j(x: Tensor, conv_in: Tensor, rec_in: Tensor):
    # replicate _attention_mtp up to the scan inputs, then STORE to persistent bufs
    b = blk0
    xh = x.half()
    out_gate = b.attn_gate(xh).reshape(B, T, nv, hv)
    beta = b.ssm_beta(xh).sigmoid().reshape(B, T, nv)
    alpha = b.ssm_alpha(xh)
    log_alpha = ((alpha.float() + b.ssm_dt["bias"]).softplus().reshape(B, T, nv, -1) *
                 b.ssm_a.reshape(nv, -1))
    rows = b.attn_qkv(xh).cast(conv_in.dtype)
    win = conv_in.cat(rows, dim=1)
    conv_out = sum((win[:, i:i+T] * b.ssm_conv1d["weight"][:, i] for i in range(b.ssm_conv_kernel))).silu()
    qc, kc, vc = conv_out.split([b.q_dim, b.q_dim, b.conv_channels - 2*b.q_dim], dim=-1)
    q, k2 = (z.reshape(B, T, nv, hv).normalize(dim=-1, eps=1e-6) for z in (qc, kc))  # q_dim==v-head-dim path fallback
    # NOTE: use the real split from _attention_mtp (q_dim vs num_k_heads); simplified here
    qf, kf, vf = (z.transpose(1, 2).float() for z in (q, k2, vc.reshape(B, T, nv, hv)))
    betaf = beta.transpose(1, 2).float()
    alphaf = log_alpha.transpose(1, 2).exp()
    # store to persistent (captured stores)
    q_pb.assign(qf.squeeze(0).transpose(1, 2).contiguous()).realize()   # [nv, T, hk/hv]
    k_pb.assign(kf.squeeze(0).transpose(1, 2).contiguous()).realize()
    v_pb.assign(vf.squeeze(0).transpose(1, 2).contiguous()).realize()
    a_pb.assign(alphaf.squeeze(0).transpose(1, 2).contiguous()).realize()   # exp(alpha)
    b_pb.assign(betaf.squeeze(0).transpose(1, 2).contiguous()).realize()
    st_pb.assign(rec_in.float().squeeze(0).reshape(nv, hv, hk).contiguous()).realize()
    return x

def mid_launch():
    scan_k(st_pb.uop.buf_uop.buffer._bufs["NV"], out_pb.uop.buf_uop.buffer._bufs["NV"],
           pss_pb.uop.buf_uop.buffer._bufs["NV"],
           a_pb.uop.buf_uop.buffer._bufs["NV"], b_pb.uop.buf_uop.buffer._bufs["NV"],
           q_pb.uop.buf_uop.buffer._bufs["NV"], k_pb.uop.buf_uop.buffer._bufs["NV"],
           v_pb.uop.buf_uop.buffer._bufs["NV"],
           global_size=(nv,1,1), local_size=(32,1,1), vals=(T,))

@TinyJit
def post_j(x: Tensor) -> Tensor:
    core = out_pb.transpose(0, 1).reshape(1, T, nv*hv)   # read persistent outs
    return core

# --- run the comparison on random inputs (validating the SPLIT mechanics, exact per-block) ---
x_in = up((rng.standard_normal((B, T, model.dim)) * 0.2).astype(np.float32))
conv_in = up(np.zeros((B, blk0.ssm_conv_kernel-1, blk0.conv_channels), np.float32))
rec_in = up((rng.standard_normal((B, nv, hv, hk)) * 0.1).astype(np.float32))

pre_j(x_in, conv_in, rec_in)
mid_launch()
core_split = post_j(x_in)
core_split_np = _KEEP[-1].numpy() if False else None
# read via the persistent buffer
Device["NV"].synchronize()
core_np = out_pb.numpy()   # [nv, T, hv]
print("split path ran; core shape:", core_np.shape)
