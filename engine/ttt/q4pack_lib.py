# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 (Stage A) — pack layout library.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Q4_0 two-region draft-pack format (engine/q4pack.py + engine/q4v.cu truth).

Packed row layout per weight row (nin % 256 == 0, NGRP = nin//256):
    [ qs: nin//2 bytes ][ d: nin//16 bytes ]   -> rowb = NGRP*144

Dequant truth (q4v.cu): element e of a 32-elem sub-block lives at
    byte (e & 15) of the sub-block's 16 qs bytes, nibble (e >> 4)
    (LOW nibble first), value w = fp16d * (q - 8), q in [0,15].
Sub-block i's fp16 scale d sits at d_region_offset + 2*i.
Sub-blocks per 256-group: 8; sub-block of element k = k // 32.

Q8_0 (writer only — no engine kernels yet): per 32-block [2B fp16 d][32 x int8],
w = d * q. Two-region analog: [ qs: nin bytes ][ d: nin//16 bytes ].
"""
import numpy as np

Q4_ROWS = {
    # pack name: (nout, nin)
    "d_eh": (5120, 10240),  # eh_proj
    "d_q":  (12288, 5120),  # fused q+gate (per head: [q(256) | gate(256)])
    "d_k":  (1024, 5120),
    "d_v":  (1024, 5120),
    "d_o":  (5120, 6144),
    "d_fg": (17408, 5120),
    "d_fu": (17408, 5120),
    "d_fd": (5120, 17408),
}
NORM_SHAPES = {  # fp32, exact .npy shapes (data bytes; npy header is 128B)
    "d_enw": 5120, "d_hnw": 5120, "d_shnw": 5120, "d_nw1": 5120, "d_nw2": 5120,
    "d_qnw": 256, "d_knw": 256,
}


def pack_row_bytes(nin: int) -> int:
    return (nin // 256) * 144


def dequant_q4(arr: np.ndarray, nout: int, nin: int) -> np.ndarray:
    """packed uint8 [nout, NGRP*144] -> fp32 [nout, nin] (engine dequant order)."""
    assert arr.shape == (nout, pack_row_bytes(nin)), (arr.shape, nout, nin)
    qs = arr[:, : nin // 2]
    d = arr[:, nin // 2:].copy().view("<f2")          # [nout, nin//32] fp16
    qs = qs.reshape(nout, nin // 32, 16)
    lo = (qs & 0xF).astype(np.int32)                  # elements 0..15
    hi = (qs >> 4).astype(np.int32)                   # elements 16..31
    q = np.concatenate([lo, hi], axis=2)              # [nout, nblk, 32]
    d = d.reshape(nout, nin // 32, 1)
    w = d.astype(np.float32) * (q - 8).astype(np.float32)
    return w.reshape(nout, nin)


def quantize_q4_rtn(w: np.ndarray) -> np.ndarray:
    """fp32 [nout, nin] -> packed uint8 RTN Q4_0 (the engine layout)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    nout, nin = w.shape
    assert nin % 256 == 0
    blk = w.reshape(nout, nin // 32, 32).astype(np.float64)
    amax = np.abs(blk).max(axis=2, keepdims=True)
    d = np.maximum(amax / 8.0, 1e-12)
    q = np.rint(blk / d + 8.0)
    q = np.clip(q, 0, 15).astype(np.uint8)
    d16 = d.astype(np.float16)
    lo = q[:, :, :16]                                  # elements 0..15 -> low nibbles
    hi = q[:, :, 16:]                                  # elements 16..31 -> high nibbles
    qs = (lo | (hi << 4)).reshape(nout, nin // 2)
    out = np.zeros((nout, pack_row_bytes(nin)), dtype=np.uint8)
    out[:, : nin // 2] = qs
    out[:, nin // 2:] = d16.view(np.uint8).reshape(nout, nin // 16)
    return out


def quantize_q8_rtn(w: np.ndarray) -> np.ndarray:
    """fp32 [nout, nin] -> packed uint8 RTN Q8_0 ([qs nin B][d nin//16 B])."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    nout, nin = w.shape
    assert nin % 256 == 0
    blk = w.reshape(nout, nin // 32, 32).astype(np.float64)
    amax = np.abs(blk).max(axis=2, keepdims=True)
    d = np.maximum(amax / 127.0, 1e-12)
    q = np.rint(blk / d)
    q = np.clip(q, -127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    out = np.zeros((nout, nin + nin // 16), dtype=np.uint8)
    out[:, :nin] = q.view(np.uint8).reshape(nout, nin)
    out[:, nin:] = d16.view(np.uint8).reshape(nout, nin // 16)
    return out


def dequant_q8(arr: np.ndarray, nout: int, nin: int) -> np.ndarray:
    qs = arr[:, :nin].view(np.int8).reshape(nout, nin // 32, 32).astype(np.float32)
    d = arr[:, nin:].copy().view("<f2").astype(np.float32)
    return (d[:, :, None] * qs).reshape(nout, nin)


def load_pack(pack_dir: str, dequant: bool = True):
    """Load a draft_pack dir -> dict of fp32 weights {pack_name: array}."""
    import os
    out = {}
    for nm, (nout, nin) in Q4_ROWS.items():
        p = f"{pack_dir}/{nm}.npy"
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        out[nm] = dequant_q4(arr, nout, nin) if dequant else arr
    for nm, n in NORM_SHAPES.items():
        p = f"{pack_dir}/{nm}.npy"
        if os.path.exists(p):
            out[nm] = np.load(p).astype(np.float32)
    return out
