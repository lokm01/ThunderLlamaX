#!/usr/bin/env python3
"""CPU-only debug: gx_ref (the harness reference) vs a from-D2-dq_iq3_s
reconstruction with the kernel's exact accumulation order. Pure numpy."""
import sys
import numpy as np
sys.path.insert(0, "~/tinygrad-metal")
from MM_P0_d2_repack import dq_iq3_s

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

def gx_ref(rows, x, gridf):
    lane = np.arange(32, dtype=np.int64); koff = lane << 3
    partial = np.zeros((512, 32), dtype=np.float32)
    for b in range(8):
        blk = rows[:, b*110:(b+1)*110]
        d = np.ascontiguousarray(blk[:, 0:2]).view(np.float16).astype(np.float32)[:, 0]
        sraw = lane >> 2
        nib = (np.take(blk, 106 + (sraw >> 1), axis=1) >> ((sraw & 1) << 2)) & 0xF
        sc = 1.0 + 2.0*nib.astype(np.float32)
        sg = np.take(blk, 74 + lane, axis=1)
        qlo = np.take(blk, 2 + 2*lane, axis=1).astype(np.uint16)
        qhi = np.take(blk, 3 + 2*lane, axis=1).astype(np.uint16)
        q16 = qlo | (qhi << 8)
        g0i, g1i = lane*2, lane*2 + 1
        bit0 = (np.take(blk, 66 + (g0i >> 3), axis=1) >> (g0i & 7)) & 1
        bit1 = (np.take(blk, 66 + (g1i >> 3), axis=1) >> (g1i & 7)) & 1
        qb0 = (q16 & 0xFF).astype(np.int64) + (bit0.astype(np.int64) << 8)
        qb1 = (q16 >> 8).astype(np.int64) + (bit1.astype(np.int64) << 8)
        t1 = d[:, None] * sc
        w = np.empty((512, 32, 8), dtype=np.float32)
        w[:, :, 0:4] = t1[:, :, None] * gridf[qb0]
        w[:, :, 4:8] = t1[:, :, None] * gridf[qb1]
        for j in range(8):
            wj = w[:, :, j].copy()
            m = (sg & (1 << j)) != 0
            wj[m] = -wj[m]
            partial = partial + (wj * x[koff + j][None, :]).astype(np.float32)
    p = partial
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return np.ascontiguousarray(p[:, 0])

def dq_ref_lanes(rows, x):
    """From the D2-VALIDATED dq_iq3_s: full dequant per row, then the kernel's
    per-lane partial + xor-tree order."""
    W = dq_iq3_s(rows, 2048)                # [512, 2048] f32 (bit-exact vs llama.cpp)
    partial = np.zeros((512, 32), dtype=np.float32)
    for l in range(32):
        k = np.arange(l*8, l*8+8)
        for j in range(8):                   # j ascending
            partial[:, l] = partial[:, l] + (W[:, k[j]] * x[k[j]]).astype(np.float32)
    p = partial
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return p[:, 0].copy()

def main():
    gridf = iq3s_grid_f32()
    rng = np.random.default_rng(7)
    _ = rng.integers(0, 256, 1458176*32, dtype=np.uint8)   # consume same as harness
    rng2 = np.random.default_rng(11)
    x = (rng2.uniform(-0.5, 0.5, 2048)).astype(np.float32)
    rows = rng.integers(0, 256, 512*880, dtype=np.uint8).reshape(512, 880).copy()
    y1 = gx_ref(rows, x, gridf)
    y2 = dq_ref_lanes(rows, x)
    fin = np.isfinite(y1) & np.isfinite(y2)
    eq = (y1 == y2)[fin]
    print(f"rows finite both: {fin.sum()}/512; bit-equal on finite: {eq.sum()}/{fin.sum()}")
    if eq.sum() < fin.sum():
        bad = np.where(fin & (y1 != y2))[0][:3]
        for r in bad:
            print(f"  row {r}: gx_ref {y1[r]:.6f} vs dq_ref {y2[r]:.6f} rel {abs(y1[r]-y2[r])/max(abs(y2[r]),1e-9):.2e}")
            # compare the dequant VALUES: gx_ref's implied W vs dq W on this row
            # rebuild gx_ref W for row r
            blk = rows[r]
            lane = np.arange(32, dtype=np.int64)
            Wg = np.zeros((32, 8), dtype=np.float32)
            for b in range(8):
                bb = blk[b*110:(b+1)*110]
                d = np.frombuffer(bb[0:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
                sraw = lane >> 2
                nib = (bb[106 + (sraw >> 1)] >> ((sraw & 1) << 2)) & 0xF
                sc = 1.0 + 2.0*nib.astype(np.float32)
                sg = bb[74 + lane]
                qlo = bb[2 + 2*lane].astype(np.uint16); qhi = bb[3 + 2*lane].astype(np.uint16)
                g0i, g1i = lane*2, lane*2 + 1
                bit0 = (bb[66 + (g0i >> 3)] >> (g0i & 7)) & 1
                bit1 = (bb[66 + (g1i >> 3)] >> (g1i & 7)) & 1
                qb0 = qlo.astype(np.int64) + (bit0.astype(np.int64) << 8)
                qb1 = qhi.astype(np.int64) + (bit1.astype(np.int64) << 8)
                for j in range(4):
                    w = d*sc*gridf[qb0, j]
                    if sg & (1 << j): w = -w
                    Wg[:, j] += w   # NOTE: accumulation differs; this is just a diff probe
                for j in range(4):
                    w = d*sc*gridf[qb1, j]
                    if sg & (1 << (4+j)): w = -w
                    Wg[:, 4+j] += w
            Wd = dq_iq3_s(rows[r:r+1], 2048)[0].reshape(32, 8)
            nz = (Wg != Wd) & np.isfinite(Wg) & np.isfinite(Wd)
            print(f"    W mismatch count (rough probe): {nz.sum()}/256")

if __name__ == "__main__":
    main()
