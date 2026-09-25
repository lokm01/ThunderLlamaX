# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8w4: offline int4 requant of the ffn_gate/ffn_up IQ3_XXS planes (Tier-2 W4A8).

Dequants each [17408, 5120] IQ3_XXS tensor (the packed/ raw row bytes, the
dq_iq3_r math VERBATIM incl. the fp16 rounding of the dequant product — the
values the engine actually consumes) then requantizes per-(row, 128-k-chunk)
symmetric int4: nib = clip(rint(w/sw)+8, 0, 15), sw = fp16(absmax(chunk)/7).

  packed4/fg{i}.npy   u32 [N/8][NCH][32][4]  nibble units (p8_imma layout:
      unit u = r_local*4 + qc of group g=n/8; nibble w at bits 4w covers
      k = c*128 + qc*32 + w — one uint4/lane/consecutive per warp slice)
  packed4/fgs{i}.npy  f16 [N][NCH]           per-(row,chunk) scales
  (same for fu). Per-tensor 45.95 MB -> fg+fu all 64 blocks = 5.89 GB disk.

PURE REQUANT = a numerics change (Tier-2; PF_W4A8-gated, decode untouched).
Usage: ~/tg311/bin/python -u pack_w4.py [--check-only] [blk ...]
"""
import os, sys, glob
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = f"{BASE}/packed"
OUT = f"{BASE}/packed4"
KCH = 128
CLASSES = {"fg": 5120, "fu": 5120}

_GRID = None
def grid_f32():
    global _GRID
    if _GRID is None:
        from tinygrad.runtime.autogen.ggml_common import iq3xxs_grid
        v = np.array([(w >> (8 * i)) & 0xFF for w in iq3xxs_grid for i in range(4)], dtype=np.float32)
        assert v.size == 1024
        _GRID = v.reshape(256, 4)
    return _GRID

def dequant_iq3xxs(arr_u8, kdim):
    """[N, 98*(kdim/256)] u8 -> [N, kdim] f32 (fp16-rounded, dq_iq3_r verbatim)."""
    n = arr_u8.shape[0]
    nb = kdim >> 8
    assert arr_u8.shape[1] == 98 * nb, arr_u8.shape
    row16 = arr_u8.view(np.uint16)
    q = row16[:, :nb * 32]                                   # qs u16[32*nb]
    scp = np.ascontiguousarray(arr_u8[:, 64 * nb:96 * nb]).view(np.uint32)   # u32[8*nb]
    dpp = np.ascontiguousarray(row16[:, nb * 48:nb * 49]).view(np.float16)   # u16[nb]
    d = dpp.astype(np.float32)
    G = grid_f32()
    W = np.empty((n, nb, 32, 8), dtype=np.float32)
    for lc in range(32):
        cc = lc & 3
        qv = q[:, lc::32].astype(np.uint32)                  # [n, nb]
        swv = scp[:, (lc >> 2)::8]                           # [n, nb]
        db = d * ((swv >> 28).astype(np.float32) + 0.5) * 0.5
        sidx = (swv >> (7 * cc)) & 0x7F
        spar = (sidx ^ (sidx >> 1) ^ (sidx >> 2) ^ (sidx >> 3) ^ (sidx >> 4) ^
                (sidx >> 5) ^ (sidx >> 6)) & 1
        i0 = (qv & 0xFF).astype(np.int64)
        i1 = (qv >> 8).astype(np.int64)
        g0 = G[i0]                                           # [n, nb, 4]
        g1 = G[i1]
        sg = np.empty((n, sidx.shape[1], 8), dtype=np.float32)
        for b in range(7):
            sg[:, :, b] = np.where((sidx >> b) & 1, -1.0, 1.0)
        sg[:, :, 7] = np.where(spar != 0, -1.0, 1.0)
        W[:, :, lc, :] = db[..., None] * np.concatenate([g0, g1], axis=2) * sg
    return W.reshape(n, nb * 256).astype(np.float16).astype(np.float32)

def requant_int4(Wf):
    """[N, KDIM] f32 (the fp16-rounded dequant values) -> (units u32[N/8,NCH,32,4], sw f16[N,NCH])."""
    n, kd = Wf.shape
    assert kd % KCH == 0 and n % 8 == 0
    nch = kd // KCH
    Wc = Wf.reshape(n, nch, KCH)
    amax = np.abs(Wc).max(axis=2)
    sw = (np.maximum(amax, 1e-7) / 7.0).astype(np.float16)
    swf = sw.astype(np.float32)
    nib = np.rint(Wc / swf[:, :, None]).astype(np.int32) + 8
    np.clip(nib, 0, 15, out=nib)
    nib = nib.astype(np.uint8).reshape(n, nch, KCH)
    out = np.zeros((n // 8, nch, 32, 4), dtype=np.uint32)
    for c in range(nch):
        for qc in range(4):
            blk = nib[:, c, qc * 32:qc * 32 + 32]            # [n, 32]
            v = np.zeros((n, 4), dtype=np.uint32)
            for w in range(32):
                v[:, w >> 3] |= blk[:, w].astype(np.uint32) << (4 * (w & 7))
            out[:, c, qc::4, :] = v.reshape(n // 8, 8, 4)
    return out, sw

def main():
    os.makedirs(OUT, exist_ok=True)
    check_only = "--check-only" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = []
    for tag, kdim in CLASSES.items():
        for f in sorted(glob.glob(f"{SRC}/{tag}*.npy"),
                        key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()))):
            blk = "".join(ch for ch in os.path.basename(f) if ch.isdigit())
            if only and blk not in only:
                continue
            dst = f"{OUT}/{tag}{blk}.npy"
            if os.path.exists(dst) and not check_only:
                continue
            todo.append((tag, blk, f, dst, kdim))
    print(f"[pack_w4] {len(todo)} tensors to build", flush=True)
    for i, (tag, blk, f, dst, kdim) in enumerate(todo):
        arr = np.load(f)
        if arr.ndim == 1:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != 98 * (kdim >> 8):
            print(f"[pack_w4] {tag}{blk}: rowbytes {arr.shape[1]} != IQ3 {98*(kdim>>8)} — SKIP", flush=True)
            continue
        W = dequant_iq3xxs(arr, kdim)
        units, sw = requant_int4(W)
        np.save(dst, units)
        np.save(f"{OUT}/{tag}s{blk}.npy", sw)
        # roundtrip check: nibbles decode back to <= sw/2 + fp16 rounding error
        n, nch = sw.shape
        u = units.reshape(n // 8, nch, 32, 4)
        err_max = 0.0
        for qc in (0, 3):   # spot lanes
            v = u[:, :, qc::4, :]                                # [g, c, r, word]
            dec = np.zeros((n, nch, 32), dtype=np.float32)
            for w in range(32):
                nbl = (v[..., w >> 3] >> (4 * (w & 7))) & 0xF    # [g, c, r]
                dec[:, :, w] = nbl.transpose(0, 2, 1).reshape(n, nch)
            back = (dec - 8.0) * sw.astype(np.float32)[:, :, None]
            orig = W.reshape(n, nch, KCH)[:, :, qc * 32:qc * 32 + 32]
            err_max = max(err_max, float(np.abs(back - orig).max()))
        rel = err_max / max(float(np.abs(W).max()), 1e-9)
        print(f"[pack_w4] {tag}{blk}: done ({i+1}/{len(todo)}) units {units.shape} spot-deq err "
              f"{err_max:.5f} (rel {rel:.5f})", flush=True)
    print("[pack_w4] DONE", flush=True)

if __name__ == "__main__":
    main()
