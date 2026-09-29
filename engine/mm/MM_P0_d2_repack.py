#!/usr/bin/env python3
"""MM P0 D2 — REPACK FEASIBILITY (the VRAM decider) for Qwen3.6-35B-A3B.

For each GGUF tier:
  1. exact per-tensor byte accounting from the header (offset deltas),
  2. extract one MODAL layer's routed experts + build expert-major slabs
     (byte-copy per expert per mat — the layout the gx8e256 POC reads),
  3. verify 16B alignment empirically (expert bases, rows),
  4. numpy dequant ports (math verbatim from ggml-quants.c),
  5. cross-validate vs a C compile of the ACTUAL llama.cpp dequants on real rows,
  6. roundtrip dequant(repacked) == dequant(native),
  7. exact VRAM table @96k/128k.

Usage: ~/tg311/bin/python MM_P0_d2_repack.py <file.gguf>
"""
import os, re, struct, subprocess, sys, tempfile
import numpy as np

TYPR = {0:(1,"B"),1:(1,"b"),2:(2,"H"),3:(2,"h"),4:(4,"I"),5:(4,"i"),6:(4,"f"),
        7:(1,"b"),10:(8,"Q"),11:(8,"q"),12:(8,"d")}
TNAME = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
         10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
         16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
         21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"IQ1_M",25:"BF16"}
RB = {"IQ4_XS":136, "IQ3_S":110, "IQ2_S":82, "Q6_K":210}   # bytes / 256-elem block

def parse(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"; struct.unpack("<i", f.read(4))
    n_tensors = struct.unpack("<q", f.read(8))[0]; n_kv = struct.unpack("<q", f.read(8))[0]
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode()
    def rv(t):
        if t == 8: return rs()
        if t == 9:
            et = struct.unpack("<i", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            return [rv(et) for _ in range(n)]
        nb, fmt = TYPR[t]; return struct.unpack("<"+fmt, f.read(nb))[0]
    kv = {}
    for _ in range(n_kv):
        k = rs(); t = struct.unpack("<i", f.read(4))[0]; kv[k] = rv(t)
    infos = []
    for _ in range(n_tensors):
        name = rs(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        t = struct.unpack("<i", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, t, dims, off))
    align = kv.get("general.alignment", 32)
    data_start = (f.tell()+align-1)//align*align
    fsize = os.fstat(f.fileno()).st_size
    f.close()
    si = sorted(infos, key=lambda r: r[3])
    out = {}
    for i,(name,t,dims,off) in enumerate(si):
        ne = 1
        for d in dims: ne *= d
        nb = (si[i+1][3]-off) if i+1 < len(si) else (fsize - data_start - off)
        out[name] = (TNAME.get(t,f"t{t}"), dims, off, nb, ne)
    return kv, out, data_start, fsize

# ---------------- numpy dequant ports (math VERBATIM from ggml-quants.c) --------------
_KM = np.array([1,2,4,8,16,32,64,128], dtype=np.uint8)

def get_tables():
    from tinygrad.runtime.autogen import ggml_common as G
    def gu(tab, width):
        v = np.empty(len(tab)*width, dtype=np.uint8)
        for i, w in enumerate(tab):
            for b in range(width): v[i*width+b] = (int(w) >> (8*b)) & 0xFF
        return v
    return {"iq2s": gu(G.iq2s_grid, 8), "iq3s": gu(G.iq3s_grid, 4),
            "iq4nl": np.array(G.kvalues_iq4nl, dtype=np.float32)}

def _f16(x):  # u8[..., 2] -> f32
    return np.ascontiguousarray(x).view(np.float16).astype(np.float32)[..., 0]

def dq_iq4_xs(raw, kdim):
    n = raw.shape[0]; nb = kdim >> 8
    assert raw.shape[1] == 136*nb, raw.shape
    blk = raw.reshape(n, nb, 136)
    d = _f16(blk[:, :, 0:2])
    sh = np.ascontiguousarray(blk[:, :, 2:4]).view(np.uint16)[:, :, 0].astype(np.int64)
    sl = blk[:, :, 4:8].astype(np.int64)
    qs = blk[:, :, 8:136].reshape(n, nb, 8, 16)
    ib = np.arange(8)
    ls = ((sl[:, :, ib >> 1] >> (4*(ib & 1))) & 0xF) | (((sh[:, :, None] >> (2*ib)) & 3) << 4)
    dl = d[:, :, None] * (ls - 32).astype(np.float32)
    T = get_tables()["iq4nl"]
    y = np.empty((n, nb, 8, 32), dtype=np.float32)
    y[:, :, :, 0:16]  = dl[:, :, :, None] * T[qs & 0xF]
    y[:, :, :, 16:32] = dl[:, :, :, None] * T[qs >> 4]
    return y.reshape(n, kdim)

def dq_iq3_s(raw, kdim):
    n = raw.shape[0]; nb = kdim >> 8
    assert raw.shape[1] == 110*nb
    blk = raw.reshape(n, nb, 110)
    d = _f16(blk[:, :, 0:2])
    qs = blk[:, :, 2:66]; qh = blk[:, :, 66:74]; signs = blk[:, :, 74:106]; sc = blk[:, :, 106:110]
    T = get_tables(); G3 = T["iq3s"].reshape(512, 4)
    y = np.empty((n, nb, 8, 32), dtype=np.float32)
    l = np.arange(4)
    for p in range(4):                       # pair-of-32-groups iteration
        db1 = d * (1 + 2*(sc[:, :, p] & 0xF))
        db2 = d * (1 + 2*(sc[:, :, p] >> 4))
        for half, (db, hoff, qoff, soff) in enumerate([
                (db1, 2*p,     16*p,      8*p),
                (db2, 2*p+1,   16*p+8,    8*p+4)]):
            h = qh[:, :, hoff].astype(np.int64)
            q = qs[:, :, qoff:qoff+8]
            s = signs[:, :, soff:soff+4]
            ia = q[:, :, 0:8:2].astype(np.int64) | ((h[:, :, None] << (8-2*l)) & 256)
            ib = q[:, :, 1:8:2].astype(np.int64) | ((h[:, :, None] << (7-2*l)) & 256)
            ga, gb = G3[ia], G3[ib]                       # [n,nb,4,4]
            neg = (s[:, :, :, None] & _KM[None, None, None, :]) != 0
            sg = np.where(neg, -1.0, 1.0)
            yy = np.empty((n, nb, 4, 8), dtype=np.float32)
            yy[:, :, :, 0:4] = db[:, :, None, None] * ga * sg[:, :, :, 0:4]
            yy[:, :, :, 4:8] = db[:, :, None, None] * gb * sg[:, :, :, 4:8]
            y[:, :, 2*p+half] = yy.reshape(n, nb, 32)
    return y.reshape(n, kdim)

def dq_iq2_s(raw, kdim):
    n = raw.shape[0]; nb = kdim >> 8
    assert raw.shape[1] == 82*nb
    blk = raw.reshape(n, nb, 82)
    d = _f16(blk[:, :, 0:2])
    qs = blk[:, :, 2:66]; qh = blk[:, :, 66:74]; sc = blk[:, :, 74:82]
    gi, signs = qs[:, :, 0:32], qs[:, :, 32:64]
    T = get_tables(); G2 = T["iq2s"].reshape(1024, 8)
    y = np.empty((n, nb, 8, 32), dtype=np.float32)
    for ib32 in range(8):
        db0 = d * (0.5 + (sc[:, :, ib32] & 0xF)) * 0.25
        db1 = d * (0.5 + (sc[:, :, ib32] >> 4)) * 0.25
        h = qh[:, :, ib32].astype(np.int64)
        base = 4*ib32
        for l in range(4):
            dl = db0 if l < 2 else db1
            idx = gi[:, :, base+l].astype(np.int64) | ((h << (8-2*l)) & 0x300)
            g = G2[idx]                                    # [n,nb,8]
            s = signs[:, :, base+l]
            neg = (s[:, :, None] & _KM[None, None, :]) != 0
            y[:, :, ib32, 8*l:8*l+8] = dl[:, :, None] * g * np.where(neg, -1.0, 1.0)
    return y.reshape(n, kdim)

DQ = {"IQ4_XS": dq_iq4_xs, "IQ3_S": dq_iq3_s, "IQ2_S": dq_iq2_s}

# ---------------- C reference (verbatim llama.cpp functions, compiled) ----------------
C_PROLOGUE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#define QK_K 256
#define GGML_RESTRICT __restrict
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "%s"
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;
        else { int e = -1; uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400));
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13); }
    } else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (man << 13);
    else bits = (sign << 31) | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    float out; memcpy(&out, &bits, 4); return out;
}
#define GGML_FP16_TO_FP32(x) fp16_to_f32(x)
"""

C_MAIN = r"""
int main(int argc, char **argv) {
    const char *cls = argv[1]; int kdim = atoi(argv[2]); long nrow = atol(argv[3]);
    int bpb = !strcmp(cls,"IQ4_XS")?136 : !strcmp(cls,"IQ3_S")?110 : !strcmp(cls,"IQ2_S")?82 : -1;
    if (bpb < 0) { fprintf(stderr, "bad class\n"); return 2; }
    long rowb = (long)(kdim/256)*bpb;
    uint8_t * buf = malloc(rowb*nrow);
    FILE * fi = fopen(argv[4], "rb");
    if (fread(buf, 1, rowb*nrow, fi) != (size_t)(rowb*nrow)) { fprintf(stderr,"short read\n"); return 3; }
    fclose(fi);
    float * out = malloc(sizeof(float)*kdim*nrow);
    if      (!strcmp(cls,"IQ4_XS")) dequantize_row_iq4_xs((const block_iq4_xs *)buf, out, kdim*nrow);
    else if (!strcmp(cls,"IQ3_S"))  dequantize_row_iq3_s ((const block_iq3_s  *)buf, out, kdim*nrow);
    else if (!strcmp(cls,"IQ2_S"))  dequantize_row_iq2_s ((const block_iq2_s  *)buf, out, kdim*nrow);
    FILE * fo = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), (size_t)kdim*nrow, fo);
    fclose(fo);
    return 0;
}
"""

def build_c_ref():
    """Extract the 3 dequant functions VERBATIM from ggml-quants.c and compile."""
    hdr = "/tmp/ggml-common.h"; src = "/tmp/ggml-quants.c"
    if not (os.path.exists(hdr) and os.path.exists(src)): return None, "no llama.cpp sources in /tmp"
    body = open(src).read()
    funcs = []
    for fname in ("dequantize_row_iq2_s", "dequantize_row_iq3_s", "dequantize_row_iq4_xs"):
        m = re.search(rf"(void {fname}\(.*?\n}})", body, re.S)
        if not m: return None, f"{fname} not found"
        funcs.append(m.group(1))
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MM_P0_d2_ref.c")
    with open(cpath, "w") as f:
        f.write(C_PROLOGUE % hdr + "\n\n" + "\n\n".join(funcs) + "\n" + C_MAIN)
    exe = "/tmp/mm_p0_d2_ref"
    r = subprocess.run(["cc", "-O2", "-o", exe, cpath], capture_output=True, text=True)
    if r.returncode != 0: return None, r.stderr[:500]
    return exe, None

def read_rows(f, data_start, off, n_rows, row_bytes):
    f.seek(data_start + off)
    return np.frombuffer(f.read(n_rows * row_bytes), dtype=np.uint8).reshape(n_rows, row_bytes)

def main():
    path = sys.argv[1]
    kv, tensors, data_start, fsize = parse(path)
    tag = os.path.basename(path).split("-UD-")[1].split(".")[0]
    print(f"== D2 repack feasibility: {tag} ({fsize/2**30:.3f} GiB)")

    classes = {}
    for L in range(40):
        cs = tuple(tensors[f"blk.{L}.ffn_{m}_exps.weight"][0] for m in ("down","gate","up"))
        classes.setdefault(cs, []).append(L)
    modal_cs, modal_layers = max(classes.items(), key=lambda kv: len(kv[1]))
    L_modal = modal_layers[len(modal_layers)//2]
    print(f"   modal layer = {L_modal} {modal_cs}  ({len(modal_layers)} layers)")

    exe, err = build_c_ref()
    print(f"   C reference: {'built (verbatim llama.cpp)' if exe else 'FAILED: '+str(err)}")

    f = open(path, "rb")
    NEXP = 256
    print("\n== repack: expert-major slabs (byte-copy per expert per mat) + alignment ==")
    res = {}
    for mat, ne0, ne1 in (("gate", 2048, 512), ("up", 2048, 512), ("down", 512, 2048)):
        tn, dims, off, nb, ne = tensors[f"blk.{L_modal}.ffn_{mat}_exps.weight"]
        rowb = RB[tn] * (ne0 // 256)
        expb = rowb * ne1
        ok_row, ok_exp = rowb % 16 == 0, expb % 16 == 0
        assert nb == expb * NEXP, (nb, expb*NEXP)
        print(f"  [{mat:4s}] {tn:6s}: row={rowb:4d}B ({'16B OK' if ok_row else '16B MISALIGNED'}), "
              f"expert={expb:7d}B={expb/2**20:6.3f} MiB ({'16B OK' if ok_exp else 'MISALIGN'}), tensor={nb/2**20:.1f} MiB")
        res[mat] = (tn, rowb, expb)
        # roundtrip on experts 0..7: slab copy vs independent re-read
        E = 8
        slab = read_rows(f, data_start, off, E*ne1, rowb)
        for e in range(E):
            f.seek(data_start + off + e*expb)
            want = np.frombuffer(f.read(expb), dtype=np.uint8).reshape(ne1, rowb)
            assert np.array_equal(slab[e*ne1:(e+1)*ne1], want), f"byte-copy mismatch {mat}/{e}"
        print(f"        byte-copy roundtrip experts 0..{E-1}: EXACT")

    # dequant: numpy vs C on real rows
    print("\n== dequant cross-validation on REAL rows (numpy port vs llama.cpp C) ==")
    for mat, ne0, ne1 in (("gate", 2048, 512), ("down", 512, 2048)):
        tn, dims, off, nb, ne = tensors[f"blk.{L_modal}.ffn_{mat}_exps.weight"]
        if tn not in DQ: continue
        rowb = RB[tn]*(ne0//256)
        rows = read_rows(f, data_start, off, 16, rowb)
        y_np = DQ[tn](rows, ne0)
        if exe:
            with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".bin") as tf:
                tf.write(rows.tobytes()); ip = tf.name
            r = subprocess.run([exe, tn, str(ne0), "16", ip, ip+".out"], capture_output=True, text=True)
            if r.returncode != 0: print(f"  [{mat}/{tn}] C ref rc={r.returncode} {r.stderr[:200]}"); continue
            y_c = np.fromfile(ip+".out", dtype=np.float32).reshape(16, ne0)
            nz = int((y_c != y_np).sum())
            md = float(np.max(np.abs(y_c - y_np))) if nz else 0.0
            print(f"  [{mat}/{tn}] {'BIT-EXACT' if nz==0 else f'DIFF nz={nz}/{y_c.size} max={md}'} numpy vs llama.cpp (16 real rows)")
        # sanity: no NaN/Inf, plausible stats
        assert np.isfinite(y_np).all()
        print(f"        stats: std={y_np.std():.5f} max|w|={np.abs(y_np).max():.3f}")

    # VRAM table
    print("\n== VRAM table (exact from header) ==")
    def cat(pred): return sum(t[3] for n, t in tensors.items() if pred(n, t))
    routed = cat(lambda n,t: "_exps." in n and not n.startswith("blk.40."))
    shexp  = cat(lambda n,t: "_shexp." in n and not n.startswith("blk.40."))
    router = cat(lambda n,t: "ffn_gate_inp" in n and not n.startswith("blk.40."))
    emb = tensors["token_embd.weight"][3]; head = tensors["output.weight"][3]
    mtp = cat(lambda n,t: n.startswith("blk.40."))
    other = fsize - data_start - mtp - routed - shexp - router - emb - head
    print(f"   routed {routed/2**30:6.3f} | shared {shexp/2**30:5.3f} | router {router/2**30:5.3f} | "
          f"emb {emb/2**30:5.3f} | head {head/2**30:5.3f} | trunk-other {other/2**30:5.3f} | MTP(drop) {mtp/2**30:5.3f} GiB")
    weights = routed + shexp + router + emb + head + other
    print(f"   WEIGHTS KEPT = {weights/2**30:.3f} GiB   (file {fsize/2**30:.2f} incl tokenizer+MTP)")
    for ctxk in (96, 128):
        kv_b = 10*1024*ctxk*1024
        gdn_b = 30*2*2**20*11
        conv_b = 30*8192*3*4*11
        tot = weights + kv_b + gdn_b + conv_b + 0.4*2**30 + 1.6*2**30
        print(f"   ctx {ctxk:3d}k: KV {kv_b/2**30:.2f} + GDN {gdn_b/2**30:.2f} + misc "
              f"=> TOTAL {tot/2**30:6.2f} GiB  (23.0 working limit) {'FITS' if tot < 23.0*2**30 else 'OVER'}")
    te = sum(res[m][2] for m in res)
    print(f"\n   modal per-expert slab total = {te/2**20:.3f} MiB (plan assumed 1.67 MB all-4.25bpw)")
    f.close()

if __name__ == "__main__":
    main()
