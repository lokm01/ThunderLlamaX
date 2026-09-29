#!/usr/bin/env python3
"""MM P3+P4 — numpy ports for the NON-MoE halves (GDN k2s scan, full attention)
+ kernel-order refs for the new kernels + the FULL-MODEL fp32 anchor.

Ground truth: transformers modeling_qwen3_5_moe.py (fetched to
~/modeling_qwen3_5_moe.py; the qwen35moe arch of the GGUF):
  GDN layer (30): rmsz(attn_norm) -> in_proj_qkv [q 2048|k 2048|v 4096] +
    in_proj_z [4096] + in_proj_a/b [32] -> causal dw-conv k=4 over 8192 ch
    (silu on conv output) -> recurrent gated delta rule
    (q/k l2norm eps 1e-6 in-sqrt, q * 1/sqrt(128), state [32 v-heads][128 v]
    [128 k] fp32, per-head decay exp(-exp(A_log)*softplus(a+dt_bias)),
    beta = sigmoid(b); k-head = v-head>>1 [repeat_interleave 2]) ->
    RMSNormGated(weight[128], plain w, silu(z) gate, eps 1e-6) ->
    out_proj [4096->2048] + residual.
  Full-attn layer (10): rmsz -> q_proj [2048->8192] PER-HEAD INTERLEAVED
    (head h: rows h*512..h*512+255 = Q, +256..+511 = sigmoid GATE -- NOT the
    0..4095/4096..8191 split the P0 note assumed) + k/v_proj [512] ->
    q/k_norm RMSNormZeroCentered per-head-256 -> partial RoPE 64 dims theta
    1e7 (cos/sin cat-duplicated; dims 64:256 passthrough) -> eager attention
    (scores*(1/16), fp32 softmax) -> out * sigmoid(gate) -> o_proj + resid.
  MoE (all 40): rmsz(post) -> router F32 -> top8 renorm -> routed experts ->
    combine + sigmoid-gated shared expert (the P2 kernels own this half).

Everything CPU-only. EPS = fp32(1e-6) = 9.999999974752427e-07 (GGUF kv).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, "~/tinygrad-metal")
from MM_P0_d2_repack import _f16
from MM_P2_ports import (dq_q8_0, dq_q6_k, dq_iq4_xs, dq_iq3_s, rmsz_ref,
    rt_prod_ref, gxup_ref, gxup4_ref, gxdn4_ref, gxdn6_ref, shexp_ref,
    cmb_ref, load_router, load_wsh, load_norm, load_shexp_raw, routed_rows,
    RB, PACK, GGUF)

EPS = np.float32(9.999999974752427e-07)
ISQ128 = np.float32(0.08838834764831845)   # 1/sqrt(128)
SCA = np.float32(0.0625)                    # 256 ** -0.5
THETA = np.float32(10000000.0)
F32 = np.float32

GDN_LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38]
ATTN_LAYERS = [3,7,11,15,19,23,27,31,35,39]

def _f(x): return np.asarray(x, dtype=np.float32)

def silu_np(v):
    v = _f(v)
    return (v * (_f(1.0) / (_f(1.0) + np.exp(-v, dtype=np.float32)))).astype(np.float32)

def softplus_np(v):
    v = _f(v)
    return (np.maximum(v, _f(0.0)) + np.log1p(np.exp(-np.abs(v), dtype=np.float32))).astype(np.float32)

# ---------------- F32 / Q8 trunk loaders ----------------
_TRCACHE = {}
def trunk_raw(L, name):
    key = (L, name)
    if key not in _TRCACHE:
        fn = f"{PACK}/trunk/{name}.bin" if L is None else f"{PACK}/trunk/blk_{L}_{name}.bin"
        _TRCACHE[key] = np.fromfile(fn, dtype=np.uint8)
    return _TRCACHE[key]

def load_f32(L, name):
    return trunk_raw(L, name).view(np.float32).copy()

def load_q8(L, name, rows, k=2048):
    rowb = 34 * (k // 32)
    return trunk_raw(L, name).reshape(rows, rowb)

# ---------------- rope ----------------
def rope_tables(ctx):
    """cos/sin [ctx][32] fp32, the modeling order (partial 64d, cat-duplicated)."""
    idx = (np.arange(0, 64, 2, dtype=np.float32) / _f(64.0)).astype(np.float32)
    inv = (_f(1.0) / np.power(_f(THETA), idx)).astype(np.float32)
    pos = np.arange(ctx, dtype=np.float32)
    fr = (pos[:, None] * inv[None, :]).astype(np.float32)
    return np.cos(fr).astype(np.float32), np.sin(fr).astype(np.float32)

def rope_apply(x, cos, sin):
    """x [.., 256] fp32; rotary section = first 64 dims; cat((f,f)) convention.
    out[i]    = x[i]*cos[i]   + (-(x[i+32]))*sin[i]   (i < 32)
    out[i+32] = x[i+32]*cos[i] + x[i]*sin[i]
    Kernel order: (x1*c) + ((-x2)*s)."""
    x1 = x[..., :32]; x2 = x[..., 32:64]
    o1 = (x1 * cos + (-x2) * sin).astype(np.float32)
    o2 = (x2 * cos + x1 * sin).astype(np.float32)
    return np.concatenate([o1, o2, x[..., 64:]], axis=-1).astype(np.float32)

def _tree_warp8(pr):
    """pr [256] (1 elem/thread): per-warp lane trees -> 8 partials -> seq sum."""
    pw = pr.reshape(8, 32).copy()
    for o in (16, 8, 4, 2, 1):
        pw = _f(pw + pw[:, np.arange(32) ^ o])
    s = _f(0.0)
    for w in range(8):
        s = _f(s + pw[w, 0])
    return s

def rmszc_ref(x, w):
    """Per-head-256 rmsnorm (q/k norms): (x*rstd)*w PLAIN -- the GGUF q/k
    norm weights carry the +1 FOLDED at conversion (converter law: all
    norm.weight except linear_attn.norm get +1). KERNEL order: per-thread
    squares, warp lane trees, sequential warp-asc sum, ms = s*(1/256)."""
    p = _f(_f(x) * _f(x))
    s = _tree_warp8(p)
    ms = _f(s * _f(1.0 / 256.0))
    rstd = _f(_f(1.0) / np.sqrt(_f(ms + EPS)))
    return ((_f(x) * rstd) * w).astype(np.float32)

# ---------------- gconv36 (causal dw conv k=4 + silu) ----------------
def gconv_ref(w, xchain, state):
    """w [8192][4] fp32 (channel-major taps), xchain [T][8192] pre-conv fp32,
    state [8192][3] fp32 (inputs at t-3,t-2,t-1). Kernel order: per channel,
    a = w0*p0; a += w1*p1; a += w2*p2; a += w3*cur (sequential); silu.
    Returns ([T][8192] silu'd, new state [8192][3])."""
    T = xchain.shape[0]
    p0 = _f(state[:, 0]); p1 = _f(state[:, 1]); p2 = _f(state[:, 2])
    out = np.empty((T, 8192), dtype=np.float32)
    for t in range(T):
        cur = _f(xchain[t])
        a = _f(w[:, 0] * p0)
        a = _f(a + _f(w[:, 1] * p1))
        a = _f(a + _f(w[:, 2] * p2))
        a = _f(a + _f(w[:, 3] * cur))
        out[t] = silu_np(a)
        p0 = p1; p1 = p2; p2 = cur
    return out, np.stack([p0, p1, p2], axis=1).astype(np.float32)

# ---------------- k2s36 (the real recurrent gated-delta port) ----------------
def _tree32(v):
    """xor-shfl tree over 32 lanes: v [32] -> total."""
    v = v.copy()
    for o in (16, 8, 4, 2, 1):
        v = _f(v + v[np.arange(32) ^ o])
    return v[0]

def _tree32_pr(pr):
    """pr [128] per-lane products -> lane partials [32] (j asc adds) -> tree."""
    r = pr.reshape(32, 4)
    p = _f(r[:, 0])
    p = _f(p + r[:, 1]); p = _f(p + r[:, 2]); p = _f(p + r[:, 3])
    return _tree32(p)

def _sq4_partial(vals):
    """per-lane sum of squares over 4 elems j asc: [128] -> [32]."""
    r = vals.reshape(32, 4)
    p = _f(r[:, 0] * r[:, 0])
    p = _f(p + _f(r[:, 1] * r[:, 1]))
    p = _f(p + _f(r[:, 2] * r[:, 2]))
    p = _f(p + _f(r[:, 3] * r[:, 3]))
    return p

def k2s_ref(qkvs, ab, alog, dtb, wn, z, S):
    """Kernel-order port of the T-chain scan.
    qkvs [T][8192] fp32 POST-conv silu'd (q [16][128] | k [16][128] | v [32][128])
    ab   [T][64]   fp32 (a logits [0:32] | b logits [32:64])
    alog [32] dtb [32] fp32; wn [128]; z [T][4096] fp32; S [32][128][128] fp32.
    Returns (y [T][4096] AFTER the gated norm, S_new)."""
    T = qkvs.shape[0]
    y = np.empty((T, 4096), dtype=np.float32)
    St = _f(S.copy())
    for t in range(T):
        for h in range(32):
            # GGUF v-heads are TILED: head t pairs k-head (t & 15)
            q = _f(qkvs[t, (h & 15) * 128:(h & 15) * 128 + 128])
            k = _f(qkvs[t, 2048 + (h & 15) * 128:2048 + (h & 15) * 128 + 128])
            v = _f(qkvs[t, 4096 + h * 128:4096 + h * 128 + 128])
            qn = _f(_f(1.0) / np.sqrt(_f(_tree32(_sq4_partial(q)) + EPS)))
            kn = _f(_f(1.0) / np.sqrt(_f(_tree32(_sq4_partial(k)) + EPS)))
            qr = _f(_f(q * qn) * ISQ128)             # two multiplies, ref order
            kr = _f(k * kn)
            be = _f(_f(1.0) / (_f(1.0) + np.exp(-_f(ab[t, 32 + h]), dtype=np.float32)))
            # ssm_a in the GGUF = -exp(A_log) FOLDED (converter law): al = exp(a*sp)
            sp = softplus_np(_f(ab[t, h]) + _f(dtb[h]))
            al = np.exp(_f(_f(alog[h]) * sp), dtype=np.float32)
            yrow = np.empty(128, dtype=np.float32)
            ysm = np.empty(8, dtype=np.float32)
            for w in range(8):                        # warp w owns v rows 16w..16w+15
                pw = _f(0.0)
                for vv in range(16):
                    vi = w * 16 + vv
                    s = _f(St[h, vi] * al)            # decay on read
                    kd = _tree32_pr(_f(s * kr))
                    dl = _f(_f(v[vi] - kd) * be)
                    s = _f(s + _f(dl * kr))
                    St[h, vi] = s
                    qd = _tree32_pr(_f(s * qr))
                    yrow[vi] = qd
                    pw = _f(pw + _f(qd * qd))
                ysm[w] = pw
            yss = _f(0.0)
            for w in range(8):
                yss = _f(yss + ysm[w])
            ms = _f(yss * _f(1.0 / 128.0))
            rstd = _f(_f(1.0) / np.sqrt(_f(ms + EPS)))
            zh = _f(z[t, h * 128:h * 128 + 128])
            y[t, h * 128:h * 128 + 128] = _f(_f(_f(yrow * rstd) * wn) * silu_np(zh))
    return y, St

# ---------------- gvf32ab (F32 2048-col GEMV, a+b fused) ----------------
def gvf32_ref(Wa, Wb, x):
    """Wa/Wb [32][2048] fp32; x [2048]. Kernel order: warp per row, 64 blocks
    of 32 (per-lane elems {b*32+l}, b asc running sums), xor tree."""
    def one(Wr):
        out = np.empty(32, dtype=np.float32)
        lanes = np.arange(32)
        for j in range(32):
            row = _f(Wr[j])
            acc = np.zeros(32, dtype=np.float32)
            for b in range(64):
                acc = _f(acc + _f(row[b * 32 + lanes] * x[b * 32 + lanes]))
            out[j] = _tree32(acc)
        return out
    return one(Wa), one(Wb)

# ---------------- int8 g128 KV quant ----------------
def kv_quant(v):
    """s = max|x|/127 (0 -> 1), q = rint(x/s), per 128-block."""
    q = np.empty(256, dtype=np.int8); s = np.empty(2, dtype=np.float32)
    for b in range(2):
        seg = _f(v[b * 128:(b + 1) * 128])
        mx = np.abs(seg).max()
        sb = _f(mx / 127.0)
        if sb == 0: sb = _f(1.0)
        s[b] = sb
        q[b * 128:(b + 1) * 128] = np.rint(seg / sb).astype(np.int8)
    return q, s

def spka_ref(kq, vq, qw, kw, cos, sin, pos, Kq, Ks, Vq, Vs):
    """Append one token's k/v to the int8 cache (one kv-head).
    kq/vq [256] fp32 raw proj outs; cos/sin [32] at pos."""
    kn = rmszc_ref(kq, kw)
    kr = rope_apply(kn[None, :], cos[None, :], sin[None, :])[0]
    kq8, ks = kv_quant(kr)
    Kq[pos] = kq8; Ks[pos] = ks
    vq8, vs = kv_quant(_f(vq))
    Vq[pos] = vq8; Vs[pos] = vs

def spkq_ref(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos):
    """One q-head (h = qg offset by caller? no -- pass h) attention vs the
    int8 cache up to pos inclusive. qg [8192] fp32; h the q-head index.
    Kernel order: rmszc, rope, per-position dot (thread-per-d, 8-warp tree,
    warp-asc sum) * 1/16, p asc; softmax (seq max; exp; seq sum; divide);
    out = running sum p asc of prob*v; * sigmoid(gate)."""
    raise NotImplementedError("use spkq_h_ref")

def spkq_h_ref(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos, h):
    q = _f(qg[h * 512:h * 512 + 256])
    gate = _f(qg[h * 512 + 256:h * 512 + 512])
    qn = rmszc_ref(q, qw)
    qr = rope_apply(qn[None, :], cos[None, :], sin[None, :])[0]
    L = pos + 1
    scores = np.empty(L, dtype=np.float32)
    for p in range(L):
        kd = _f(Kq[p].astype(np.float32) * np.repeat(Ks[p], 128))
        scores[p] = _f(_tree_warp8(_f(qr * kd)) * SCA)
    m = scores[0]
    for p in range(1, L):
        if scores[p] > m: m = scores[p]
    e = _f(np.exp(_f(scores - m), dtype=np.float32))
    ssum = _f(0.0)
    for p in range(L):
        ssum = _f(ssum + e[p])
    prob = _f(e / ssum)
    out = np.zeros(256, dtype=np.float32)
    for p in range(L):
        vd = _f(Vq[p].astype(np.float32) * np.repeat(Vs[p], 128))
        out = _f(out + _f(prob[p] * vd))
    sg = _f(_f(1.0) / (_f(1.0) + np.exp(-gate, dtype=np.float32)))
    return _f(out * sg)

# ---------------- gv8k4096r (out_proj + residual) ----------------
def q8_dot4096_ref(raw_rows, x, resid):
    """gv8k4096 kernel order (warp/row, 128 blocks) + resid add."""
    from MM_P2_ports import _dot_lane_ref
    W = dq_q8_0(np.ascontiguousarray(raw_rows), 4096)
    y = _dot_lane_ref(W, x, 128, 1)
    return _f(y + resid)

# ---------------- h6k2048 (Q6_K head GEMV, k=2048) ----------------
def h6k_rows_ref(raw_rows, x):
    """Q6_K rows [rows][1680]; kernel order: warp/row, 8 blocks; per lane,
    elems {b*256 + hh*128 + l + {0,32,64,96}}, b asc / hh asc running sums;
    mult (d*sc)*q; x fp32 [2048]."""
    rows = raw_rows.shape[0]
    out = np.empty(rows, dtype=np.float32)
    lane = np.arange(32)
    is_ = lane >> 4
    for r in range(rows):
        row = raw_rows[r]
        a = np.zeros(32, dtype=np.float32)
        for b in range(8):
            blk = row[b * 210:(b + 1) * 210]
            d = _f16(blk[208:210])
            sc = np.frombuffer(blk[192:208], dtype=np.int8)
            for hh in range(2):
                qb = 64 * hh + lane; hb = 128 + 32 * hh + lane
                qhb = blk[hb]
                ql0 = blk[qb]; ql32 = blk[qb + 32]
                x0 = x[b * 256 + hh * 128 + lane + 0]
                x1 = x[b * 256 + hh * 128 + lane + 32]
                x2 = x[b * 256 + hh * 128 + lane + 64]
                x3 = x[b * 256 + hh * 128 + lane + 96]
                t = _f(d * sc[8 * hh + is_ + 0].astype(np.float32))
                w_ = _f(t * (_f((ql0 & 0xF).astype(np.float32) + _f(((qhb & 3) << 4).astype(np.float32))) - 32.0))
                a = _f(a + _f(w_ * x0))
                t = _f(d * sc[8 * hh + is_ + 2].astype(np.float32))
                w_ = _f(t * (_f((ql32 & 0xF).astype(np.float32) + _f((((qhb >> 2) & 3) << 4).astype(np.float32))) - 32.0))
                a = _f(a + _f(w_ * x1))
                t = _f(d * sc[8 * hh + is_ + 4].astype(np.float32))
                w_ = _f(t * (_f((ql0 >> 4).astype(np.float32) + _f((((qhb >> 4) & 3) << 4).astype(np.float32))) - 32.0))
                a = _f(a + _f(w_ * x2))
                t = _f(d * sc[8 * hh + is_ + 6].astype(np.float32))
                w_ = _f(t * (_f((ql32 >> 4).astype(np.float32) + _f((((qhb >> 6) & 3) << 4).astype(np.float32))) - 32.0))
                a = _f(a + _f(w_ * x3))
        out[r] = _tree32(a)
    return out

# ---------------- cmbz (combine + residual, fp32 out) ----------------
def cmbz_ref(gates, sg, part, shared, resid):
    """mx8e256cmbz: r asc then shared; + resid; fp32 out."""
    P = gates.shape[0]
    out = np.empty((P, 2048), dtype=np.float32)
    for p in range(P):
        acc = np.zeros(2048, dtype=np.float32)
        for r in range(8):
            acc = _f(acc + _f(_f(gates[p, r]) * part[p, r].astype(np.float32)))
        acc = _f(acc + _f(_f(sg[p]) * shared[p]))
        out[p] = _f(resid[p] + acc)
    return out

# ============================================================================
# THE FULL-MODEL fp32 ANCHOR (modeling math)
# ============================================================================
class Anchor:
    """Layered lazy-loading fp32 reference. Expert dequant LRU-capped (384)."""
    def __init__(self, ctx=1024, fp16_partials=False):
        # P5.1: fp16_partials=True = the ENGINE-ORDER anchor (rounds each
        # routed-expert down-proj partial to fp16 before the gate combine,
        # exactly where gx8e256dn stores fp16 partials for cmbz2048 to read).
        self.fp16_partials = fp16_partials
        self.man = json.load(open(f"{PACK}/manifest.json"))
        self.ctx = ctx
        self.cos, self.sin = rope_tables(ctx)
        self._exp = {}
        self._exp_order = []
        self._w = {}
        self._headW = None

    def _experts(self, L, ids):
        meta = self.man["routed"][L]
        ty = meta["types"]
        need = [e for e in ids if (L, e) not in self._exp]
        for e in need:
            # single-expert call returns the UNWRAPPED [nrow][rowb] mat
            rg = routed_rows(L, "gate", [e])
            ru = routed_rows(L, "up", [e])
            rd = routed_rows(L, "down", [e])
            Wg = dq_iq3_s(rg, 2048) if ty["gate"] == "IQ3_S" else dq_iq4_xs(rg, 2048)
            Wu = dq_iq3_s(ru, 2048) if ty["up"] == "IQ3_S" else dq_iq4_xs(ru, 2048)
            Wd = dq_q6_k(rd, 512) if ty["down"] == "Q6_K" else dq_iq4_xs(rd, 512)
            self._exp[(L, e)] = (Wg, Wu, Wd); self._exp_order.append((L, e))
        while len(self._exp) > 384:
            k = self._exp_order.pop(0); del self._exp[k]
        return [self._exp[(L, e)] for e in ids]

    def _tw(self, L, kind):
        key = (L, kind)
        if key in self._w: return self._w[key]
        if kind == "attn_norm": v = load_f32(L, "attn_norm_weight")
        elif kind == "post_norm": v = load_f32(L, "post_attention_norm_weight")
        elif kind == "qkv": v = dq_q8_0(load_q8(L, "attn_qkv_weight", 8192), 2048) if L in GDN_LAYERS \
            else dq_q8_0(load_q8(L, "attn_q_weight", 8192), 2048)
        elif kind == "z": v = dq_q8_0(load_q8(L, "attn_gate_weight", 4096), 2048)
        elif kind == "k": v = dq_q8_0(load_q8(L, "attn_k_weight", 512), 2048)
        elif kind == "vv": v = dq_q8_0(load_q8(L, "attn_v_weight", 512), 2048)
        elif kind == "o": v = dq_q8_0(load_q8(L, "attn_output_weight", 2048, 4096), 4096)
        elif kind == "conv": v = load_f32(L, "ssm_conv1d_weight").reshape(8192, 4)
        elif kind == "alog": v = load_f32(L, "ssm_a")
        elif kind == "dtb": v = load_f32(L, "ssm_dt_bias")
        elif kind == "alpha": v = load_f32(L, "ssm_alpha_weight").reshape(32, 2048)
        elif kind == "beta": v = load_f32(L, "ssm_beta_weight").reshape(32, 2048)
        elif kind == "snorm": v = load_f32(L, "ssm_norm_weight")
        elif kind == "outp": v = dq_q8_0(load_q8(L, "ssm_out_weight", 2048, 4096), 4096)
        elif kind == "qnorm": v = load_f32(L, "attn_q_norm_weight")
        elif kind == "knorm": v = load_f32(L, "attn_k_norm_weight")
        elif kind == "router": v = load_router(L)
        elif kind == "wsh": v = load_wsh(L)
        elif kind == "shg": v = dq_q8_0(load_q8(L, "ffn_gate_shexp_weight", 512), 2048)
        elif kind == "shu": v = dq_q8_0(load_q8(L, "ffn_up_shexp_weight", 512), 2048)
        elif kind == "shd": v = dq_q8_0(load_q8(L, "ffn_down_shexp_weight", 2048, 512), 512)
        else: raise KeyError(kind)
        self._w[key] = v
        return v

    def norm_zc(self, x, w):
        # GGUF trunk norm weights have (1+w) FOLDED -> plain multiply
        ms = _f(np.mean(_f(_f(x) * _f(x)), axis=-1, keepdims=True))
        rstd = _f(_f(1.0) / np.sqrt(_f(ms + EPS)))
        return ((_f(x) * rstd) * w).astype(np.float32)

    def head_dot(self, h):
        if self._headW is None:
            raw = np.fromfile(f"{PACK}/trunk/output_weight.bin", dtype=np.uint8).reshape(248320, 1680)
            W = np.empty((248320, 2048), dtype=np.float32)
            CH = 16384
            for c0 in range(0, 248320, CH):
                n = min(CH, 248320 - c0)
                W[c0:c0+n] = dq_q6_k(np.ascontiguousarray(raw[c0:c0+n]), 2048)
            self._headW = W
        return (self._headW @ h).astype(np.float32)

    def embed_row(self, tid):
        raw = np.fromfile(f"{PACK}/trunk/token_embd_weight.bin", dtype=np.uint8).reshape(248320, 2176)
        row = dq_q8_0(np.ascontiguousarray(raw[tid:tid+1]), 2048)
        return _f(row[0])

    def forward_token(self, tid, pos, S, convst, KV, h=None, want_logits=False, trace=None):
        """One token through all 40 layers.
        S [30][32][128][128] fp32, convst [30][8192][3] fp32,
        KV: list per attn layer of [(Kq,Ks,Vq,Vs) x 2 kv-heads].
        trace: optional list appended with h after each layer (41 entries)."""
        h = self.embed_row(tid).copy() if h is None else h
        if trace is not None: trace.append(h.copy())
        for L in range(40):
            resid = h
            hn = self.norm_zc(h, self._tw(L, "attn_norm"))
            if L in GDN_LAYERS:
                gi = GDN_LAYERS.index(L)
                qkv = (hn @ self._tw(L, "qkv").T).astype(np.float32)
                z = (hn @ self._tw(L, "z").T).astype(np.float32)
                a = (hn @ self._tw(L, "alpha").T).astype(np.float32)
                b = (hn @ self._tw(L, "beta").T).astype(np.float32)
                qkvs, convst[gi] = gconv_ref(self._tw(L, "conv"), qkv[None, :], convst[gi])
                ab = np.concatenate([a, b])[None, :]
                y, S[gi] = k2s_ref(qkvs, ab, self._tw(L, "alog"), self._tw(L, "dtb"),
                                   self._tw(L, "snorm"), z[None, :], S[gi])
                h = _f(resid + (y[0] @ self._tw(L, "outp").T).astype(np.float32))
            else:
                ai = ATTN_LAYERS.index(L)
                qg = (hn @ self._tw(L, "qkv").T).astype(np.float32)
                kq = (hn @ self._tw(L, "k").T).astype(np.float32)
                vq = (hn @ self._tw(L, "vv").T).astype(np.float32)
                qw = self._tw(L, "qnorm"); kw = self._tw(L, "knorm")
                for j in range(2):
                    Kq, Ks, Vq, Vs = KV[ai][j]
                    spka_ref(_f(kq[j*256:(j+1)*256]), _f(vq[j*256:(j+1)*256]), qw, kw,
                             self.cos[pos], self.sin[pos], pos, Kq, Ks, Vq, Vs)
                yh = np.empty(4096, dtype=np.float32)
                for hh in range(16):
                    j = hh >> 3
                    Kq, Ks, Vq, Vs = KV[ai][j]
                    yh[hh*256:(hh+1)*256] = spkq_h_ref(qg, qw, Kq, Ks, Vq, Vs,
                                                       self.cos[pos], self.sin[pos], pos, hh)
                h = _f(resid + (yh @ self._tw(L, "o").T).astype(np.float32))
            # ---- MoE half ----
            resid = h
            hn = self.norm_zc(h, self._tw(L, "post_norm"))
            lg = (self._tw(L, "router") @ hn).astype(np.float32)
            l = lg.copy(); ids = []; ex = []
            for r in range(8):
                be = int(np.argmax(l)); bv = l[be]
                ids.append(be); ex.append(np.exp(np.float64(bv - lg.max()))); l[be] = np.float32(-3.4e38)
            gates = (_f(ex) / _f(np.sum(ex))).astype(np.float32)
            sg = _f(_f(1.0) / (_f(1.0) + np.exp(-np.float64(self._tw(L, "wsh") @ hn))))
            y = np.zeros(2048, dtype=np.float32)
            for r in range(8):
                Wg, Wu, Wd = self._experts(L, [ids[r]])[0]
                t = _f(silu_np(_f(Wg @ hn)) * _f(Wu @ hn))
                part = _f(Wd @ t)
                if self.fp16_partials:
                    part = part.astype(np.float16).astype(np.float32)
                y = _f(y + _f(gates[r] * part))
            shg = silu_np(_f(self._tw(L, "shg") @ hn)); shu = _f(self._tw(L, "shu") @ hn)
            sh = _f(self._tw(L, "shd") @ _f(shg * shu))
            h = _f(resid + _f(y + _f(sg * sh)))
            if trace is not None: trace.append(h.copy())
        h = self.norm_zc(h, load_f32(None, "output_norm_weight"))
        if trace is not None: trace.append(h.copy())
        logits = self.head_dot(h)
        top1 = int(np.argmax(logits))
        return top1, (logits if want_logits else None), h

def fresh_state(ctx=1024):
    S = np.zeros((30, 32, 128, 128), dtype=np.float32)
    convst = np.zeros((30, 8192, 3), dtype=np.float32)
    KV = []
    for _ in ATTN_LAYERS:
        lay = []
        for j in range(2):
            Kq = np.zeros((ctx, 256), dtype=np.int8); Ks = np.ones((ctx, 2), dtype=np.float32)
            Vq = np.zeros((ctx, 256), dtype=np.int8); Vs = np.ones((ctx, 2), dtype=np.float32)
            lay.append((Kq, Ks, Vq, Vs))
        KV.append(lay)
    return S, convst, KV

if __name__ == "__main__":
    print("ports import OK; quick self-tests")
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((8192, 4)) * 0.1).astype(np.float32)
    xs = (rng.standard_normal((3, 8192)) * 0.3).astype(np.float32)
    st = (rng.standard_normal((8192, 3)) * 0.3).astype(np.float32)
    y, st2 = gconv_ref(w, xs, st)
    print("gconv:", y.shape, np.isfinite(y).all())
    qkvs = (rng.standard_normal((2, 8192)) * 0.2).astype(np.float32)
    ab = (rng.standard_normal((2, 64)) * 0.5).astype(np.float32)
    alog = rng.uniform(-2, 2, 32).astype(np.float32)
    dtb = rng.uniform(-2, 2, 32).astype(np.float32)
    wn = (rng.standard_normal(128) * 0.1).astype(np.float32)
    z = (rng.standard_normal((2, 4096)) * 0.2).astype(np.float32)
    S0 = (rng.standard_normal((32, 128, 128)) * 0.05).astype(np.float32)
    y2, S2 = k2s_ref(qkvs, ab, alog, dtb, wn, z, S0)
    print("k2s:", y2.shape, np.isfinite(y2).all(), "state delta", float(np.abs(S2 - S0).max()))
    Wa = (rng.standard_normal((32, 2048)) * 0.05).astype(np.float32)
    Wb = (rng.standard_normal((32, 2048)) * 0.05).astype(np.float32)
    x = (rng.standard_normal(2048) * 0.5).astype(np.float32)
    aa, bb = gvf32_ref(Wa, Wb, x)
    print("gvf32:", aa.shape, np.allclose(aa, Wa @ x, rtol=1e-5), np.allclose(bb, Wb @ x, rtol=1e-5))
    c, s = rope_tables(16)
    print("rope:", c.shape, c[1, 0], s[1, 0])
    print("self-tests done")
