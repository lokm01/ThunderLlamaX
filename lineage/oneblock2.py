# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
from tinygrad.engine.jit import TinyJit

os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1")
os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_MAXCTX", "2048")
os.environ.setdefault("MTP_CKPT_DIR", "/tmp/ckpt_empty")

from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
model = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", 2048)[0]
dev = Device["NV"]
_gdns = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
blk = _gdns[0]
nv, hv, hk = blk.num_v_heads, blk.head_v_dim, blk.head_k_dim
nk = blk.num_k_heads
print(f"[block0] nv={nv} hv={hv} hk={hk} nk={nk}", flush=True)
assert nv==48 and hv==128 and hk==128, f"dims changed: nv={nv} hv={hv} hk={hk}"

B, T = 1, 3
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t

# persistent buffers in kernel layout [nv, T, d] / [nv,T] / [nv,hv,hk]
q_pb  = up(np.zeros((nv, T, hk), np.float32))
k_pb  = up(np.zeros((nv, T, hk), np.float32))
v_pb  = up(np.zeros((nv, T, hv), np.float32))
a_pb  = up(np.zeros((nv, T), np.float32))
b_pb  = up(np.zeros((nv, T), np.float32))
st_pb = up(np.zeros((nv, hv, hk), np.float32))
out_pb  = up(np.zeros((nv, T, hv), np.float32))
pss_pb  = up(np.zeros((T, nv, hv, hk), np.float32))

scan_k = NVProgram(dev, TinyELF(
    lib=open("~/tinygrad-metal/a4/gdn_scan_m.cubin","rb").read(),
    name="gdn_scan_m", target=dev.renderer.target,
    signature=(("v", 8, dtypes.int32,()),)))

import functools

@TinyJit
def pre_j(x: Tensor, conv_in: Tensor, rec_in: Tensor):
    # ==== VERBATIM from _attention_mtp 527-569 (through the scan-input prep) ====
    Bv, Tv, _ = x.shape
    T_pad = x.max_shape[1]
    xh = x.half()
    out_gate = blk.attn_gate(xh)
    out_gate = out_gate.reshape(Bv, Tv, nv, hv)
    beta = blk.ssm_beta(xh).sigmoid().reshape(Bv, Tv, nv)
    alpha = blk.ssm_alpha(xh)
    log_alpha = ((alpha.float() + blk.ssm_dt["bias"]).softplus().reshape(Bv, Tv, nv, -1) *
                 blk.ssm_a.reshape(nv, -1))
    conv_state = conv_in
    rows = blk.attn_qkv(xh).cast(conv_state.dtype)
    win = conv_state.cat(rows, dim=1)
    conv_next = win[:, Tv:Tv+blk.ssm_conv_kernel-1].cast(conv_in.dtype).contiguous()
    conv_out = functools.reduce(lambda a,b: a+b,
      (win[:, i:i+T_pad] * blk.ssm_conv1d["weight"][:, i] for i in range(blk.ssm_conv_kernel))).silu()
    q, k2, v2 = conv_out.split([blk.q_dim, blk.q_dim, blk.conv_channels - 2*blk.q_dim], dim=-1)
    q, k2 = (z.reshape(Bv, T_pad, nk, hk).normalize(dim=-1, eps=1e-6)
            .repeat(1, 1, nv//nk, 1) for z in (q, k2))
    v2 = v2.reshape(Bv, T_pad, nv, hv)
    q, k2, v2, beta = (z.transpose(1, 2).float() for z in (q, k2, v2, beta))
    q = q.unsqueeze(-2) * hk**-0.5
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    alphaf = log_alpha.transpose(1, 2).exp().unsqueeze(-1)
    # ==== store to persistent (captured stores) ====
    q_pb.assign(q.squeeze(-2).squeeze(0).contiguous()).realize()   # [nv,T,hk] SCALED
    k_pb.assign(k2.squeeze(-2).squeeze(0).contiguous()).realize()
    v_pb.assign(v2.squeeze(-1).squeeze(0).contiguous()).realize()
    a_pb.assign(alphaf.squeeze(-1).squeeze(-1).squeeze(0).contiguous()).realize()
    b_pb.assign(beta.squeeze(-1).squeeze(-1).squeeze(0).contiguous()).realize()
    st_pb.assign(rec_in.float().squeeze(0).reshape(nv, hv, hk).contiguous()).realize()
    return conv_next

def mid_launch():
    scan_k(st_pb.uop.buf_uop.buffer._bufs["NV"], out_pb.uop.buf_uop.buffer._bufs["NV"],
           pss_pb.uop.buf_uop.buffer._bufs["NV"],
           a_pb.uop.buf_uop.buffer._bufs["NV"], b_pb.uop.buf_uop.buffer._bufs["NV"],
           q_pb.uop.buf_uop.buffer._bufs["NV"], k_pb.uop.buf_uop.buffer._bufs["NV"],
           v_pb.uop.buf_uop.buffer._bufs["NV"],
           global_size=(nv,1,1), local_size=(256,1,1), vals=(T,))

# ---- inputs ----
rng = np.random.default_rng(4)
x_in = up((rng.standard_normal((B, T, blk.attn_norm.weight.shape[0])) * 0.2).astype(np.float32))
conv_in = up((rng.standard_normal((B, blk.ssm_conv_kernel-1, blk.conv_channels)) * 0.1).astype(np.float32))
rec_in = up((rng.standard_normal((B, nv, hv, hk)) * 0.1).astype(np.float32))

# ---- REFERENCE: the stock scan on the same intermediates (computed eagerly, non-jit) ----
def ref_scan():
    Bv, Tv, _ = x_in.shape
    T_pad = x_in.max_shape[1]
    xh = x_in.half()
    beta = blk.ssm_beta(xh).sigmoid().reshape(Bv, Tv, nv)
    alpha = blk.ssm_alpha(xh)
    log_alpha = ((alpha.float() + blk.ssm_dt["bias"]).softplus().reshape(Bv, Tv, nv, -1) *
                 blk.ssm_a.reshape(nv, -1))
    conv_state = conv_in
    rows = blk.attn_qkv(xh).cast(conv_state.dtype)
    win = conv_state.cat(rows, dim=1)
    conv_out = functools.reduce(lambda a,b: a+b,
      (win[:, i:i+T_pad] * blk.ssm_conv1d["weight"][:, i] for i in range(blk.ssm_conv_kernel))).silu()
    q, k2, v2 = conv_out.split([blk.q_dim, blk.q_dim, blk.conv_channels - 2*blk.q_dim], dim=-1)
    q, k2 = (z.reshape(Bv, T_pad, nk, hk).normalize(dim=-1, eps=1e-6)
            .repeat(1, 1, nv//nk, 1) for z in (q, k2))
    v2 = v2.reshape(Bv, T_pad, nv, hv)
    q, k2, v2, beta = (z.transpose(1, 2).float() for z in (q, k2, v2, beta))
    q = q.unsqueeze(-2) * hk**-0.5
    k2 = k2.unsqueeze(-2); v2 = v2.unsqueeze(-1); beta = beta.unsqueeze(-1).unsqueeze(-1)
    alphaf = log_alpha.transpose(1, 2).exp().unsqueeze(-1)
    state = rec_in.float()
    outs = []
    for t in range(T):
        s1 = state * alphaf[:, :, t]
        delta = (v2[:, :, t] - (s1*k2[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
        state = s1 + delta * k2[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))
    core = outs[0].stack(*outs[1:], dim=1)   # [B, nv, T, 1] -> squeeze
    return core.squeeze(-1).squeeze(0).transpose(1, 0).numpy(), state.squeeze(0).numpy()

ref_core, ref_state = ref_scan()   # [nv,T,hv]... check orientation below

# ---- SPLIT PATH ----
pre_j(x_in, conv_in, rec_in)
mid_launch()
dev.synchronize()
got_core = out_pb.numpy()          # [nv, T, hv]
got_state = st_pb.numpy()          # [nv, hv, hk] (kernel updates in place)
got_pss = pss_pb.numpy()           # [T, nv, hv, hk]

# reference core from ref: outs stacked dim=1 -> [B, T?]... outs[t] = [B,nv,1]; stack dim=1 -> [B,T,nv,1]
# we transposed to [nv,T,hv]: ref_core above = transpose(1,2) of [T,nv,hv] -> [nv,T,hv] ✓
r_core = np.abs(got_core - ref_core).max() / max(np.abs(ref_core).max(), 1e-9)
r_state = np.abs(got_state - ref_state).max() / max(np.abs(ref_state).max(), 1e-9)
# per-step states: ref computes states at t=0,1,2; last == final
st_t = ref_state
r_pss = []
for t in range(T):
    # recompute ref per-step states
    pass
print(f"core relerr={r_core:.2e}  final-state relerr={r_state:.2e}", flush=True)
print("VERDICT:", "PASS" if max(r_core, r_state) < 1e-4 else "FAIL", flush=True)
