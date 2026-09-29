#!/usr/bin/env python3
"""MM P2 — numpy dequant ports (Q6_K, Q8_0 VERBATIM from ggml-quants.c) +
kernel-order references for the P2 kernels + C cross-validation.

Everything here is CPU-only (safe to run while the engine daemon is live).
The GPU harness (MM_P2_run.py) imports the refs from here.
"""
import os, re, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, "~/tinygrad-metal")
from MM_P0_d2_repack import dq_iq3_s, dq_iq4_xs, dq_iq2_s, get_tables, _f16, RB

RB = dict(RB); RB["Q8_0"] = 34
PACK = os.path.expanduser("~/models36/packed/qwen3.6-35b-a3b-iq4_xs")
GGUF = os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")

_KM = np.array([1,2,4,8,16,32,64,128], dtype=np.uint8)

# ---------------- Q6_K (VERBATIM dequantize_row_q6_K) ----------------
def dq_q6_k(raw, kdim):
    n = raw.shape[0]; nb = kdim >> 8
    assert raw.shape[1] == 210*nb, raw.shape
    blk = raw.reshape(n, nb, 210)
    ql = blk[:, :, 0:128]; qh = blk[:, :, 128:192]
    sc = blk[:, :, 192:208].astype(np.int8).astype(np.float32)
    d = _f16(blk[:, :, 208:210])
    y = np.empty((n, nb, 256), dtype=np.float32)
    for half in range(2):
        qb = 64*half; hb = 32*half; sb = 8*half
        for l in range(32):
            is_ = l >> 4
            t = d * sc[:, :, sb + is_ + 0]          # (d*sc) first -- C assoc order
            q1 = ((ql[:, :, qb + l] & 0xF) | ((qh[:, :, hb + l] & 3) << 4)).astype(np.int32) - 32
            y[:, :, 128*half + l + 0] = t * q1.astype(np.float32)
            t = d * sc[:, :, sb + is_ + 2]
            q2 = ((ql[:, :, qb + l + 32] & 0xF) | (((qh[:, :, hb + l] >> 2) & 3) << 4)).astype(np.int32) - 32
            y[:, :, 128*half + l + 32] = t * q2.astype(np.float32)
            t = d * sc[:, :, sb + is_ + 4]
            q3 = ((ql[:, :, qb + l] >> 4) | (((qh[:, :, hb + l] >> 4) & 3) << 4)).astype(np.int32) - 32
            y[:, :, 128*half + l + 64] = t * q3.astype(np.float32)
            t = d * sc[:, :, sb + is_ + 6]
            q4 = ((ql[:, :, qb + l + 32] >> 4) | (((qh[:, :, hb + l] >> 6) & 3) << 4)).astype(np.int32) - 32
            y[:, :, 128*half + l + 96] = t * q4.astype(np.float32)
    return y.reshape(n, kdim)

# ---------------- Q8_0 (VERBATIM: y = d * q) ----------------
def dq_q8_0(raw, kdim):
    n = raw.shape[0]; nb = kdim >> 5
    assert raw.shape[1] == 34*nb, raw.shape
    blk = raw.reshape(n, nb, 34)
    d = _f16(blk[:, :, 0:2])
    q = blk[:, :, 2:34].astype(np.int8).astype(np.float32)
    return (d[:, :, None] * q).reshape(n, kdim)

DQ = {"IQ4_XS": dq_iq4_xs, "IQ3_S": dq_iq3_s, "IQ2_S": dq_iq2_s,
      "Q6_K": dq_q6_k, "Q8_0": dq_q8_0}

# ---------------- C reference (verbatim llama.cpp) for the two new classes ----------------
from MM_P0_d2_repack import C_PROLOGUE

C_MAIN2 = r"""
int main(int argc, char **argv) {
    const char *cls = argv[1]; int kdim = atoi(argv[2]); long nrow = atol(argv[3]);
    int bpb = !strcmp(cls,"Q6_K")?210 : !strcmp(cls,"Q8_0")?34 : -1;
    if (bpb < 0) { fprintf(stderr, "bad class\n"); return 2; }
    long rowb = (long)(kdim/256)*bpb;
    if (!strcmp(cls, "Q8_0")) rowb = (long)(kdim/32)*bpb;
    uint8_t * buf = malloc(rowb*nrow);
    FILE * fi = fopen(argv[4], "rb");
    if (fread(buf, 1, rowb*nrow, fi) != (size_t)(rowb*nrow)) { fprintf(stderr,"short read\n"); return 3; }
    fclose(fi);
    float * out = malloc(sizeof(float)*kdim*nrow);
    if      (!strcmp(cls,"Q6_K")) dequantize_row_q6_K((const block_q6_K *)buf, out, (int64_t)kdim*nrow);
    else if (!strcmp(cls,"Q8_0")) dequantize_row_q8_0((const block_q8_0 *)buf, out, (int64_t)kdim*nrow);
    FILE * fo = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), (size_t)kdim*nrow, fo);
    fclose(fo);
    return 0;
}
"""

def build_c_ref2():
    hdr = "/tmp/ggml-common.h"; src = "/tmp/ggml-quants.c"
    if not (os.path.exists(hdr) and os.path.exists(src)): return None, "no llama.cpp sources in /tmp"
    body = open(src).read()
    funcs = []
    for fname in ("dequantize_row_q6_K", "dequantize_row_q8_0"):
        m = re.search(rf"(void {fname}\(.*?\n}})", body, re.S)
        if not m: return None, f"{fname} not found"
        funcs.append(m.group(1))
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MM_P2_dref.c")
    with open(cpath, "w") as f:
        f.write(C_PROLOGUE % hdr + "\n\n" + "\n\n".join(funcs) + "\n" + C_MAIN2)
    exe = "/tmp/mm_p2_dref"
    r = subprocess.run(["cc", "-O2", "-o", exe, cpath], capture_output=True, text=True)
    if r.returncode != 0: return None, r.stderr[:500]
    return exe, None

def c_ref_dq(cls, rows, kdim, exe):
    with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".bin") as tf:
        tf.write(np.ascontiguousarray(rows).tobytes()); ip = tf.name
    r = subprocess.run([exe, cls, str(kdim), str(rows.shape[0]), ip, ip+".out"],
                       capture_output=True, text=True)
    os.unlink(ip)
    if r.returncode != 0: raise RuntimeError(r.stderr[:300])
    y = np.fromfile(ip+".out", dtype=np.float32).reshape(rows.shape[0], kdim)
    os.unlink(ip+".out")
    return y

# ---------------- lookup tables as device-side float buffers ----------------
def iq4nl_f32():
    from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
    return np.array(kvalues_iq4nl, dtype=np.float32)

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

# ---------------- kernel-order references ----------------
def _xor32(p):
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return p[:, 0]

def _dot_lane_ref(W, x, nblk, jperelem=1):
    """per-lane running sums over blocks (b asc) then the lane dim (j asc),
    then the xor-shfl tree. W: [rows, k] fp32 dequanted; x: [k]."""
    rows = W.shape[0]
    partial = np.zeros((rows, 32), dtype=np.float32)
    Wb = W.reshape(rows, nblk, 32, -1)
    xb = x.reshape(nblk, 32, -1)
    prod = (Wb * xb[None, :, :, :]).astype(np.float32)
    for b in range(nblk):
        for j in range(prod.shape[3]):
            partial = (partial + prod[:, b, :, j]).astype(np.float32)
    return _xor32(partial)

def _dot_lane32_ref(W, x, nblk):
    """IQ4_XS lane (gx8e256dn/up4): per-lane elems {b*256 + ib*32 + l},
    (b asc, ib asc) running sums, then the xor tree. W: [rows, k] fp32."""
    rows = W.shape[0]
    partial = np.zeros((rows, 32), dtype=np.float32)
    Wb = W.reshape(rows, nblk, 8, 32)          # [rows, b, ib, lane]
    xb = x.reshape(nblk, 8, 32)
    prod = (Wb * xb[None, :, :, :]).astype(np.float32)
    for b in range(nblk):
        for ib in range(8):
            partial = (partial + prod[:, b, ib, :]).astype(np.float32)
    return _xor32(partial)

def gxdn4_ref(rows_dn, act):
    """gx8e256dn (IQ4_XS lane): k=512, 2 blocks; fp32 -> fp16 store."""
    W = dq_iq4_xs(np.ascontiguousarray(rows_dn), 512)     # [2048, 512]
    return _dot_lane32_ref(W, act, 2).astype(np.float16)

def gxdn6_ref(rows_dn, act):
    """gx8e256dn6 (Q6_K lane): k=512, 2 blocks; per-lane elems {b*256+h*128+l+{0,32,64,96}},
    h asc, jj asc; fp32 -> fp16 store."""
    W = dq_q6_k(np.ascontiguousarray(rows_dn), 512)       # [2048, 512]
    rows = W.shape[0]
    partial = np.zeros((rows, 32), dtype=np.float32)
    # lane elems: k = b*256 + h*128 + l + off[jj], off = [0,32,64,96] (a GATHER,
    # not a reshape -- the lane stride over jj is 32, not 1)
    OFF = np.array([0, 32, 64, 96])
    kk = (np.arange(2)[:, None, None, None]*256 + np.arange(2)[None, :, None, None]*128
          + np.arange(32)[None, None, :, None] + OFF[None, None, None, :])   # [b,h,l,jj]
    Wb = W[:, kk]           # [rows, b, h, l, jj]
    xb = act[kk]            # [b, h, l, jj]
    prod = (Wb * xb[None]).astype(np.float32)
    for b in range(2):
        for h in range(2):
            for j in range(4):
                partial = (partial + prod[:, b, h, :, j]).astype(np.float32)
    return _xor32(partial).astype(np.float16)

def gxup4_ref(rows_g, rows_u, x):
    """gx8e256up4 (IQ4_XS gate+up lane, k=2048, 8 blocks) + silu*u epilogue."""
    Wg = dq_iq4_xs(np.ascontiguousarray(rows_g), 2048)
    Wu = dq_iq4_xs(np.ascontiguousarray(rows_u), 2048)
    g = _dot_lane32_ref(Wg, x, 8)
    u = _dot_lane32_ref(Wu, x, 8)
    return (silu_f32(g) * u).astype(np.float32)

def silu_f32(g):
    return (g / (1.0 + np.exp(-g.astype(np.float64))).astype(np.float32)).astype(np.float32)

def q8_dot_ref(raw_rows, x, kdim):
    """Q8_0 lane dot: per-lane elems {b*32+l}, b asc (jperelem=1); fp32."""
    W = dq_q8_0(np.ascontiguousarray(raw_rows), kdim)
    return _dot_lane_ref(W, x, kdim >> 5, 1)

def shexp_ref(wg_raw, wu_raw, wd_raw, x):
    """shexp8: gate/up Q8_0 [512 rows, k=2048] dual dots + silu*u -> t[512]
    (fp32), then down Q8_0 [2048 rows, k=512] dot -> fp32 out [2048]."""
    g = q8_dot_ref(wg_raw, x, 2048)
    u = q8_dot_ref(wu_raw, x, 2048)
    t = (silu_f32(g) * u).astype(np.float32)
    return q8_dot_ref(wd_raw, t, 512)

def rmsz_ref(x, w, eps=np.float32(9.999999974752427e-07)):
    """rmsz2048 kernel order: per-thread strided partials {i + 256*t}, t asc,
    xor tree within warps then sequential sum of the 8 warp partials (warp asc);
    ms = s/2048; rstd = 1/sqrtf(ms+eps); y = (x*rstd)*(1+w)."""
    P = x.shape[0]
    out = np.empty((P, 2048), dtype=np.float32)
    for p in range(P):
        prod = (x[p].astype(np.float32) * x[p].astype(np.float32)).reshape(8, 32, 8)  # [t, lane, warp]
        # per-thread partial: t asc
        pt = np.zeros((8, 32), dtype=np.float32)
        for t in range(8):
            for w_ in range(32):
                pass
        # threads are (t*32 + warp)? NO: threadIdx.x = warp*32 + lane; stride 256 => elem = threadIdx + 256*t
        # elem index i = (warp*32+lane) + 256*t  =>  i = lane + 32*warp + 256*t
        pt2 = np.zeros(256, dtype=np.float32)
        for th in range(256):
            s_ = np.float32(0)
            for t in range(8):
                s_ = np.float32(s_ + np.float32(x[p, th + 256*t] * x[p, th + 256*t]))
            pt2[th] = s_
        # warp xor trees
        pw = pt2.reshape(8, 32)
        for o in (16, 8, 4, 2, 1):
            pw = pw + pw[:, np.arange(32) ^ o]
        wp = pw[:, 0]
        s_ = np.float32(0)
        for k in range(8):
            s_ = np.float32(s_ + wp[k])
        ms = np.float32(s_ / np.float32(2048.0))
        rstd = np.float32(np.float32(1.0) / np.float32(np.sqrt(np.float32(ms + eps))))
        out[p] = ((x[p] * rstd).astype(np.float32) * (np.float32(1.0) + w).astype(np.float32)).astype(np.float32)
    return out

def cmb_ref(gates, sg, part, shared):
    """mx8e256cmb kernel order: r asc then shared; fp32; fp16 store.
    gates [P,8] f32, sg [P] f32, part [P,8,2048] f16, shared [P,2048] f32."""
    P = gates.shape[0]
    out = np.empty((P, 2048), dtype=np.float16)
    for p in range(P):
        acc = np.zeros(2048, dtype=np.float32)
        for r in range(8):
            acc = (acc + np.float32(gates[p, r]) * part[p, r].astype(np.float32)).astype(np.float32)
        acc = (acc + np.float32(sg[p]) * shared[p]).astype(np.float32)
        out[p] = acc.astype(np.float16)
    return out

def h8i_quant(w, g=128):
    """int8 g128 row quantizer (the repack-side tool). w: [rows, k] fp32 -> (q int8, s fp32 [rows, k/g])."""
    rows, k = w.shape
    assert k % g == 0
    wg = w.reshape(rows, k // g, g)
    mx = np.abs(wg).max(axis=2)
    s = (mx / 127.0).astype(np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    q = np.rint(wg / s[:, :, None]).astype(np.int8)
    return q.reshape(rows, k), s.reshape(rows, k // g).copy()

def h8i_ref(q, s, x, g=128):
    """h8i lane order: per group (g asc), 4 consecutive int8 per lane, j asc."""
    rows, k = q.shape
    qg = q.reshape(rows, k // g, 32, 4).astype(np.float32)
    xg = x.reshape(k // g, 32, 4)
    partial = np.zeros((rows, 32), dtype=np.float32)
    for gi in range(k // g):
        for j in range(4):
            partial = (partial + (np.float32(s[:, gi])[:, None] * qg[:, gi, :, j] * xg[gi, :, j][None, :]).astype(np.float32)).astype(np.float32)
    return _xor32(partial)

# ---------------- P1 router refs (copied verbatim; bit-exact logit order) ----------------
def logits_ref(W, h, p):
    hr = h[p]
    lg = np.empty(256, dtype=np.float32)
    for e in range(256):
        parts = []
        for part in range(4):
            w = W[e, part*512:(part+1)*512]; hh = hr[part*512:(part+1)*512]
            s_ = np.float32(0)
            for i in range(0, 512, 4):
                s_ = np.float32(s_ + np.float32(w[i]*hh[i]))
                s_ = np.float32(s_ + np.float32(w[i+1]*hh[i+1]))
                s_ = np.float32(s_ + np.float32(w[i+2]*hh[i+2]))
                s_ = np.float32(s_ + np.float32(w[i+3]*hh[i+3]))
            parts.append(s_)
        lg[e] = np.float32(np.float32(np.float32(parts[0]+parts[1])+parts[2])+parts[3])
    return lg

def rt_prod_ref(W, wsh, h, positions):
    out = {}
    for p in positions:
        lg = logits_ref(W, h, p)
        l = lg.copy()
        ids, ex = [], []
        for r in range(8):
            be = 0; bv = l[0]
            for e2 in range(1, 256):
                if l[e2] > bv: bv = l[e2]; be = e2
            ids.append(be); ex.append(np.exp(np.float64(bv - lg.max()))); l[be] = np.float32(-3.4e38)
        s8 = np.sum(ex)
        out[p] = (np.array(ids, dtype=np.uint16), np.array(ex, dtype=np.float32)/np.float32(s8),
                  np.float32(1.0/(1.0+np.exp(-np.float64(np.dot(wsh.astype(np.float64), h[p].astype(np.float64)))))))
    return out

def gxup_ref(rows_g, rows_u, x, gridf=None):
    """Kernel-order reference for gx8e256up (IQ3_S lane; P1 verbatim)."""
    Wg = dq_iq3_s(np.ascontiguousarray(rows_g), 2048)
    Wu = dq_iq3_s(np.ascontiguousarray(rows_u), 2048)
    g = _dot_lane_ref(Wg, x, 8, 8)
    u = _dot_lane_ref(Wu, x, 8, 8)
    return (silu_f32(g) * u).astype(np.float32)

# ---------------- real-tensor accessors ----------------
def load_router(L=0):
    raw = np.fromfile(f"{PACK}/trunk/blk_{L}_ffn_gate_inp_weight.bin", dtype=np.uint8)
    return raw.view(np.float32).reshape(256, 2048).copy()

def load_wsh(L=0):
    raw = np.fromfile(f"{PACK}/trunk/blk_{L}_ffn_gate_inp_shexp_weight.bin", dtype=np.uint8)
    return raw.view(np.float32).copy()

def load_norm(L=0, which="post_attention"):
    raw = np.fromfile(f"{PACK}/trunk/blk_{L}_{which}_norm_weight.bin", dtype=np.uint8)
    return raw.view(np.float32).copy()

def load_shexp_raw(L=0):
    wg = np.fromfile(f"{PACK}/trunk/blk_{L}_ffn_gate_shexp_weight.bin", dtype=np.uint8).reshape(512, 2176)
    wu = np.fromfile(f"{PACK}/trunk/blk_{L}_ffn_up_shexp_weight.bin", dtype=np.uint8).reshape(512, 2176)
    wd = np.fromfile(f"{PACK}/trunk/blk_{L}_ffn_down_shexp_weight.bin", dtype=np.uint8).reshape(2048, 544)
    return wg, wu, wd

def bank_reader():
    m = __import__("json").load(open(f"{PACK}/manifest.json"))
    return m

def routed_rows(L, mat, experts):
    """raw quant rows [len(experts), rowb] for the given layer/mat."""
    m = bank_reader()
    meta = m["routed"][L]
    rec = None
    for r in meta["files"]:
        if mat in r["expb"]: rec = r; break
    assert rec, (L, mat)
    rowb = RB[meta["types"][mat]] * ({"gate": 8, "up": 8, "down": 2}[mat])
    nrow = {"gate": 512, "up": 512, "down": 2048}[mat]
    off = rec["offsets"][mat]; slab = rec["slab"]; expb = rec["expb"][mat]
    path = os.path.join(PACK, rec["file"])
    out = []
    with open(path, "rb") as f:
        for e in experts:
            f.seek(e*slab + off)
            out.append(np.frombuffer(f.read(expb), dtype=np.uint8).reshape(nrow, rowb))
    return np.stack(out) if len(out) > 1 else out[0]

# ---------------- CPU self-test ----------------
if __name__ == "__main__":
    import json
    print("== MM_P2_ports CPU validation")
    exe, err = build_c_ref2()
    print(f"   C ref: {'built' if exe else 'FAILED: '+str(err)}")
    # Q6_K on real L34 down rows
    m = bank_reader()
    rows6 = routed_rows(34, "down", [0, 7, 123])
    y6 = dq_q6_k(rows6, 512)
    assert np.isfinite(y6).all()
    if exe:
        for i in range(rows6.shape[0]):
            yc = c_ref_dq("Q6_K", rows6[i], 512, exe)
            nz = int((yc != y6[i]).sum())
            print(f"   Q6_K L34 down e{[0,7,123][i]}: {'BIT-EXACT' if nz==0 else f'DIFF nz={nz}'} vs llama.cpp C")
            assert nz == 0
    # Q8_0 on real trunk rows
    wg, wu, wd = load_shexp_raw(0)
    for nm, rr, kd in (("gate", wg[:16], 2048), ("down", wd[:16], 512)):
        y8 = dq_q8_0(rr, kd)
        assert np.isfinite(y8).all()
        if exe:
            yc = c_ref_dq("Q8_0", rr, kd, exe)
            nz = int((yc != y8).sum())
            print(f"   Q8_0 shexp {nm}: {'BIT-EXACT' if nz==0 else f'DIFF nz={nz}'} vs llama.cpp C")
            assert nz == 0
    # sanity: IQ4_XS down rows from the modal layer through the D2 port
    rows4 = routed_rows(0, "down", [5])
    y4 = dq_iq4_xs(rows4, 512)
    assert np.isfinite(y4).all()
    print(f"   IQ4_XS L00 down e5: finite, shape {y4.shape}")
    # router fp32 + norms
    W = load_router(0); wsh = load_wsh(0); wn = load_norm(0)
    print(f"   router {W.shape} std={W.std():.4f} | wsh {wsh.shape} | norm {wn.shape} min={wn.min():.4f} max={wn.max():.4f}")
    print("ALL PORT VALIDATIONS PASSED")
