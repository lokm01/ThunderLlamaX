# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — engine-math reference replay (numpy, fp16 round-trips).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Exact replay of the engine draft-chain math from a (dequantized) draft pack:
dnorm2 -> q4v(ehproj) -> k0_norm -> dq/dkv -> aattn_d (per-head rms+rope+gated
attn, scale 1/16) -> doproj -> k3m_hh -> dfgu -> ddown -> k0_norm(shared) ->
lm_head. With emulate_fp16=True every half round-trip the kernels do is
reproduced; False = pure fp32 structural reference.
This is the Validation-A ground truth (Phase 0's chain_sim.py supersedes it for
sim scoring; here it exists to prove the torch port)."""
import numpy as np

F16 = np.float16
F32 = np.float32
DIM, NH, NKV, HD, EPS = 5120, 24, 4, 256, 1e-6
FREQS = (1e7 ** (-np.arange(32, dtype=np.float64) / 32.0)).astype(F32)


def h16(x):
    return x.astype(F16).astype(F32)


def gemv(w, x, emulate_fp16, add=None):
    """q4v.cu: acc = sum_i half(x_i)*half(w_i) fp32; out = float(half(acc))
    (+ hh for ADDHH). w fp32 [nout, nin] (already dequantized)."""
    if emulate_fp16:
        prod = h16(x[None, :].astype(F32)) * h16(w)
        acc = prod.sum(axis=1, dtype=F32)
        acc = h16(acc)
    else:
        acc = w @ x.astype(F32)
    if add is not None:
        acc = add.astype(F32) + acc
    return acc


def rms(x, w, emulate_fp16, eps=EPS):
    r = 1.0 / np.sqrt(np.mean(x.astype(F32) ** 2) + eps)
    return x.astype(F32) * r * w.astype(F32)


def rms_head(x, w, emulate_fp16, eps=EPS):
    """per-head over last dim; engine does half(x*r) then *nw."""
    xf = x.astype(F32)
    r = 1.0 / np.sqrt((xf ** 2).mean(-1, keepdims=True) + eps)
    if emulate_fp16:
        return h16(xf * r) * w.astype(F32)
    return xf * r * w.astype(F32)


def rope(x, pos, emulate_fp16):
    """x [..., HD]; pairs (i, i+32)."""
    out = x.astype(F32).copy()
    ang = np.float32(pos) * FREQS
    cs, sn = np.cos(ang), np.sin(ang)
    a = out[..., :32].copy()
    b = out[..., 32:64].copy()
    out[..., :32] = a * cs - b * sn
    out[..., 32:64] = a * sn + b * cs
    if emulate_fp16:
        out[..., :64] = h16(out[..., :64])
    return out


def draft_step(w, tok_emb, hm, pos, kv, emulate_fp16=True):
    """One draft-chain step on ONE sequence (batch dim absent).
    w: pack weight dict. tok_emb/hm: [5120] fp32. pos: int.
    kv: dict {'K': [NKV, CTX, HD] fp16, 'V': same} mutated in place at pos.
    Returns (hd [5120] fp32, head_in [5120] fp16-normed)."""
    e = tok_emb.astype(F32)
    # dnorm2
    cat_e = rms(e, w["d_enw"], emulate_fp16)
    cat_h = rms(hm.astype(F32), w["d_hnw"], emulate_fp16)
    if emulate_fp16:
        cat_e, cat_h = h16(cat_e), h16(cat_h)
    cat = np.concatenate([cat_e, cat_h])
    xin = gemv(w["d_eh"], cat, emulate_fp16)          # ADDHH + zeros
    xh = rms(xin, w["d_nw1"], emulate_fp16)
    if emulate_fp16:
        xh = h16(xh)
    qrow = gemv(w["d_q"], xh, emulate_fp16)           # [12288]
    krow = gemv(w["d_k"], xh, emulate_fp16)           # [1024]
    vrow = gemv(w["d_v"], xh, emulate_fp16)
    # aattn_d
    q_heads = qrow.reshape(NH, 2, HD)                 # [h][q|g][d]
    k_head = krow.reshape(NKV, HD)
    v_head = vrow.reshape(NKV, HD)
    qn = rms_head(q_heads[:, 0, :], w["d_qnw"][:HD], emulate_fp16)
    kn = rms_head(k_head, w["d_knw"][:HD], emulate_fp16)
    qr = rope(qn, pos, emulate_fp16)
    kr = rope(kn, pos, emulate_fp16)
    kv["K"][:, pos, :] = kr.astype(F16)
    kv["V"][:, pos, :] = v_head.astype(F16)
    ao = np.zeros((NH, HD), F32)
    for h in range(NH):
        kvh = h // 6
        sc = (qr[h].astype(F32) / 16.0) @ kv["K"][kvh, : pos + 1, :].astype(F32).T
        sc = sc - sc.max()
        p = np.exp(sc)
        p /= p.sum()
        o = p @ kv["V"][kvh, : pos + 1, :].astype(F32)
        g = q_heads[h, 1, :].astype(F32)
        ao[h] = o * (1.0 / (1.0 + np.exp(-g)))
    if emulate_fp16:
        ao = h16(ao)
    attn_out = gemv(w["d_o"], ao.reshape(-1), emulate_fp16)   # [5120]
    # k3m_hh
    hh = xin + (h16(attn_out) if emulate_fp16 else attn_out).astype(F32)
    hhx = rms(hh, w["d_nw2"], emulate_fp16)
    if emulate_fp16:
        hhx = h16(hhx)
    # dfgu
    ag = gemv(w["d_fg"], hhx, emulate_fp16)
    au = gemv(w["d_fu"], hhx, emulate_fp16)
    if emulate_fp16:
        gact = h16(h16(ag) / (1.0 + np.exp(-h16(ag)))) * h16(au)  # silu in half
    else:
        gact = ag / (1.0 + np.exp(-ag)) * au
    # ddown (ADDHH: hh + half(acc))
    fd = gemv(w["d_fd"], gact, emulate_fp16)
    hd = hh + (h16(fd) if emulate_fp16 else fd).astype(F32)
    head_in = rms(hd, w["d_shnw"], emulate_fp16)
    if emulate_fp16:
        head_in = h16(head_in)
    return hd, head_in


def chain(w, emb, h_seed, toks, pos0, emulate_fp16=True, CTX=4096):
    """Serve-chain replay: step0 = (toks[0], h_seed) at pos0; steps i>=1 =
    (toks[i], own hd_{i-1}). emb: [VOCAB,5120] or a fn(tok)->[5120].
    Returns (hds [S,5120], head_ins [S,5120])."""
    kv = {"K": np.zeros((NKV, CTX, HD), F16), "V": np.zeros((NKV, CTX, HD), F16)}
    hds, his = [], []
    hm = h_seed.astype(F32)
    for i, t in enumerate(toks):
        te = emb[t] if hasattr(emb, "shape") else emb(t)
        hd, hi = draft_step(w, te.astype(F32), hm, pos0 + i, kv, emulate_fp16)
        hds.append(hd)
        his.append(hi)
        hm = hd
    return np.stack(hds), np.stack(his)
