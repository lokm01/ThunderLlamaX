# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""fwd3_split v3 — group-merged families (the ~50-family build for the 100k
capture wall). Structure: ONE TinyJit per GDN boundary covering
[post(g) + following attn blocks + pre(g')] — all tensor dataflow, no eager
mid inside. The eager gdn_scan_m launches run between merged families.
Families: 1 leading pre + 48 merged + 1 trailing post ~= 50 (vs 112 before).
Validated pieces unchanged (verbatim chains from oneblock2/v2)."""
import os, sys, functools
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
from tinygrad.llm.model import GatedDeltaNetBlock

dev = Device["NV"]
_T = 3
_pool, _fams = {}, {}     # _fams: "lead" | ("mid", gbi) | "tail" -> TinyJit
_scan_k, _model = None, None
_gdn_idx = []             # model.blk indices of GDN blocks (capture order)

def _up(a):
    return Tensor(np.ascontiguousarray(a)).contiguous().realize()

def init(model, T=3):
    global _T, _model, _scan_k, _gdn_idx
    _T, _model = T, model
    _gdn_idx = [i for i, b in enumerate(model.blk) if isinstance(b, GatedDeltaNetBlock)]
    _scan_k = NVProgram(dev, TinyELF(
        lib=open(os.path.expanduser("~/tinygrad-metal/a4/gdn_scan_m.cubin"),"rb").read(),
        name="gdn_scan_m", target=dev.renderer.target,
        signature=(("v", 8, dtypes.int32,()),)))

_XBUF, _CVBUF, _RCBUF = {}, {}, {}

def _xbuf(bi, x):
    if bi not in _XBUF:
        _XBUF[bi] = _up(np.zeros(tuple(x.shape), np.float32))
    _XBUF[bi].assign(x).realize()
    return _XBUF[bi]

def _stbuf(d, bi, t):
    if bi not in d:
        import tinygrad as _tg
        d[bi] = _tg.Tensor.zeros(*t.shape, dtype=t.dtype).contiguous().realize()
    d[bi].assign(t).realize()
    return d[bi]

def _bufs_for(gbi):
    """gbi = index into _gdn_idx"""
    if gbi not in _pool:
        if gbi % 8 == 0:
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache()
            import gc as _gc; _gc.collect()
        b = _model.blk[_gdn_idx[gbi]]
        nv, hv, hk = b.num_v_heads, b.head_v_dim, b.head_k_dim
        z = lambda *s: _up(np.zeros(s, np.float32))
        _pool[gbi] = dict(q=z(nv,_T,hk), k=z(nv,_T,hk), v=z(nv,_T,hv),
                          a=z(nv,_T), bb=z(nv,_T), st=z(nv,hv,hk),
                          out=z(nv,_T,hv), pss=z(_T,nv,hv,hk),
                          gate=z(1,_T,nv,hv))
    return _pool[gbi]

def _raw(t): return t.uop.buf_uop.buffer._bufs["NV"]

# ---- verbatim fragment bodies (from validated v2) ----
def _pre_body(P, b, xh, conv_state):
    Bv, Tv = xh.shape[0], xh.shape[1]
    T_pad = xh.max_shape[1]
    xn = b.attn_norm(xh.float())
    xh = xn.half()
    out_gate = b.attn_gate(xh).reshape(Bv, Tv, b.num_v_heads, b.head_v_dim)
    beta = b.ssm_beta(xh).sigmoid().reshape(Bv, Tv, b.num_v_heads)
    alpha = b.ssm_alpha(xh)
    log_alpha = ((alpha.float() + b.ssm_dt["bias"]).softplus().reshape(Bv, Tv, b.num_v_heads, -1) *
                 b.ssm_a.reshape(b.num_v_heads, -1))
    rows = b.attn_qkv(xh).cast(conv_state.dtype)
    win = conv_state.cat(rows, dim=1)
    conv_next = win[:, Tv:Tv+b.ssm_conv_kernel-1].cast(conv_state.dtype).contiguous()
    conv_out = functools.reduce(lambda a2,b2: a2+b2,
      (win[:, i:i+T_pad] * b.ssm_conv1d["weight"][:, i] for i in range(b.ssm_kernel() if hasattr(b, "ssm_kernel") else b.ssm_conv_kernel))).silu()
    q, k2, v2 = conv_out.split([b.q_dim, b.q_dim, b.conv_channels - 2*b.q_dim], dim=-1)
    q, k2 = (z.reshape(Bv, T_pad, b.num_k_heads, b.head_k_dim).normalize(dim=-1, eps=1e-6)
            .repeat(1, 1, b.num_v_heads//b.num_k_heads, 1) for z in (q, k2))
    v2 = v2.reshape(Bv, T_pad, b.num_v_heads, b.head_v_dim)
    q, k2, v2, beta = (z.transpose(1, 2).float() for z in (q, k2, v2, beta))
    q = q.unsqueeze(-2) * b.head_k_dim**-0.5
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    alphaf = log_alpha.transpose(1, 2).exp().unsqueeze(-1)
    P["q"].assign(q.squeeze(-2).squeeze(0).contiguous()).realize()
    P["k"].assign(k2.squeeze(-2).squeeze(0).contiguous()).realize()
    P["v"].assign(v2.squeeze(-1).squeeze(0).contiguous()).realize()
    P["a"].assign(alphaf.squeeze(-1).squeeze(-1).squeeze(0).contiguous()).realize()
    P["bb"].assign(beta.squeeze(-1).squeeze(-1).squeeze(0).contiguous()).realize()
    P["gate"].assign(out_gate.float().contiguous()).realize()
    return conv_next

def _post_body(P, b, x):
    core = P["out"].transpose(1, 0).reshape(1, _T, b.num_v_heads, b.head_v_dim)
    z = (b.ssm_norm(core) * P["gate"].silu()).cast(x.dtype).contiguous()
    attn_out = b.ssm_out(z.reshape(1, _T, -1))
    h = x + attn_out
    return (h + b._feed_forward(b.ffn_norm(h))).contiguous()

# ---- merged family: [attn-run] + pre(g') ----
def _make_lead(attn_blocks, b, gbi):
    @TinyJit
    def lead_j(x: Tensor, conv_in: Tensor, rec_in: Tensor):
        P = _bufs_for(gbi)
        _xs = x
        for ab in attn_blocks:
            _xs = ab(_xs, _sp_global[0]).contiguous()
        _xb = _xbuf(("lead", gbi), _xs)
        _cv = _stbuf(_CVBUF, gbi, conv_in)
        _rc = _stbuf(_RCBUF, gbi, rec_in)
        _pre_body(P, b, _xb.half(), _cv)
        return _xb
    return lead_j

_sp_global = [None]

def _make_merged(gbi_prev, attn_blocks, b_next, gbi_next):
    """[post(gbi_prev) + attn run + pre(gbi_next)] in ONE TinyJit."""
    @TinyJit
    def merged_j(x: Tensor, conv_in: Tensor, rec_in: Tensor):
        P_prev = _pool[gbi_prev]
        b_prev = _model.blk[_gdn_idx[gbi_prev]]
        h = _post_body(P_prev, b_prev, x)
        _xs = h
        for ab in attn_blocks:
            _xs = ab(_xs, _sp_global[0]).contiguous()
        P_next = _bufs_for(gbi_next)
        _xb = _xbuf(("m", gbi_next), _xs)
        _cv = _stbuf(_CVBUF, gbi_next, conv_in)
        _rc = _stbuf(_RCBUF, gbi_next, rec_in)
        _pre_body(P_next, b_next, _xb.half(), _cv)
        return _xb
    return merged_j

def _make_tail(gbi):
    @TinyJit
    def tail_j(x: Tensor) -> Tensor:
        P = _pool[gbi]
        b = _model.blk[_gdn_idx[gbi]]
        # any trailing attn blocks after the last GDN post fold here via caller
        return _post_body(P, b, x)
    return tail_j

def _mid(gbi):
    P = _pool[gbi]
    _scan_k(_raw(P["st"]), _raw(P["out"]), _raw(P["pss"]),
            _raw(P["a"]), _raw(P["bb"]), _raw(P["q"]), _raw(P["k"]), _raw(P["v"]),
            global_size=(P["q"].shape[0],1,1), local_size=(256,1,1), vals=(_T,))

def fwd3_split(model, tokens, start_pos, v_sp=None):
    import os as _ose
    global _CAPTURED_ALL
    _sp_global[0] = v_sp.bind(start_pos) if v_sp is not None else start_pos
    if _ose.getenv("MTP_EMB_GATHER") and getattr(model, "emb_rows", None) is not None:
        from tinygrad.llm.gguf import ggml_data_to_tensor as _g2t
        _n = model.emb_out_dim
        x = _g2t(model.emb_rows[tokens.reshape(-1)].cast(dtypes.uint8),
                 _n * tokens.numel(), model.emb_ggml_type) \
            .reshape(tokens.shape[0], tokens.shape[1], _n).cast(dtypes.float32)
    else:
        x = model.token_embd(tokens).float()
    blocks = model.blk
    n_gdn = len(_gdn_idx)
    # build families once
    if not _fams:
        # leading: attns before first GDN + pre(0)
        first_g = _gdn_idx[0]
        lead_attns = [blocks[i] for i in range(first_g)]
        _fams["lead"] = _make_lead(lead_attns, blocks[first_g], 0)
        # merged pairs
        for k in range(n_gdn - 1):
            g_prev, g_next = _gdn_idx[k], _gdn_idx[k+1]
            attns = [blocks[i] for i in range(g_prev+1, g_next)]
            _fams[("m", k)] = _make_merged(k, attns, blocks[g_next], k+1)
        _fams["tail"] = _make_tail(n_gdn - 1)
        _CAPTURED_ALL = False
    # execute
    import os as _osx
    _dbg = _osx.getenv("FRAGDBG")
    if _dbg: print("[v3fam] lead", flush=True)
    lead = _fams["lead"]
    b0 = blocks[_gdn_idx[0]]
    conv_in = b0.conv_state.cast(dtypes.float32)
    rec_in = b0.recurrent_state
    x = lead(x, conv_in, rec_in)
    _mid(0)
    for k in range(n_gdn - 1):
        if _dbg and k % 2 == 0: print(f"[v3fam] m{k}", flush=True)
        m = _fams[("m", k)]
        bn = blocks[_gdn_idx[k+1]]
        conv_in = bn.conv_state.cast(dtypes.float32)
        rec_in = bn.recurrent_state
        x = m(x, conv_in, rec_in)
        _mid(k+1)
    x = _fams["tail"](x)
    # trailing attns after the last GDN block
    last_g = _gdn_idx[-1]
    for i in range(last_g+1, len(blocks)):
        x = blocks[i](x, _sp_global[0]).contiguous()
    _CAPTURED_ALL = True
    return x.contiguous()

_CAPTURED_ALL = False

def final_states():
    return {gbi: dict(st=P["st"], pss=P["pss"]) for gbi, P in _pool.items()}
