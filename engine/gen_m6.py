# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R5 K=5: generate m6.cu (M=6/T=6 trunk kernels) from the FIXED m5.cu.
Laws baked in (the R4/R5 postmortems):
  - launch bounds PRESERVED through renames (nw32 family = 1024; name law);
  - EVERY row-write family gains the row-5 store (the aq3k8v5/k3ao R4 bug class);
  - every a4 accumulator gains a5; ACC6H2/RED6/IQ3V6 macros;
  - k2s6 hand-surgery: t<6; rec t=5 -> rec6x; conv t=4 -> conv5x, t=5 -> conv6x
    (keeps the [48][5] rec4/conv4 layout + live=slot-4: NO trunk/serve surgery).
Audits FAIL-LOUD at the end; nothing is written unless all pass."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/m5.cu").read()
BS = chr(92)

def rep(pat, repl, expect, label):
    global src
    src, cnt = re.subn(pat, repl, src)
    print(f"[gen] {label:56s} {cnt}")
    if cnt != expect:
        print(f"FAIL: {label}: {cnt} != {expect}"); sys.exit(1)

# ---- 1) macros: RED5 -> RED6 (add A5) ----
rep(r"#define RED5\(A0,A1,A2,A3,A4\) \{ \\",
    lambda m: "#define RED6(A0,A1,A2,A3,A4,A5) { " + BS, 1, "RED6 def rename")
rep(r"A4 \+= __shfl_down_sync\(FULL, A4, o\); \} \}",
    "A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); } }", 1, "RED6 A5 line")

# ---- 2) ACC5H2 -> ACC6H2 macro (params + x5 ptr + x5/A5 block) ----
rep(r"#define ACC5H2\(X0, X1, X2, X3, X4, WV, A0, A1, A2, A3, A4\) \{ \\",
    lambda m: "#define ACC6H2(X0, X1, X2, X3, X4, X5, WV, A0, A1, A2, A3, A4, A5) { " + BS, 1, "ACC6H2 def rename")
rep(r"const __half2\* x4 = \(X4\); \\",
    lambda m: "const __half2* x4 = (X4); const __half2* x5 = (X5); " + BS, 1, "ACC6H2 x5 ptr")
x4blk_lines = [
    "    { const float2 p = __half22float2(__hmul2(x4[0], w01)); A4 += p.x; A4 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x4[1], w23)); A4 += p.x; A4 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x4[2], w45)); A4 += p.x; A4 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x4[3], w67)); A4 += p.x; A4 += p.y; } }",
]
x5blk_lines = [l.replace("x4", "x5").replace("A4", "A5") for l in x4blk_lines]
x4blk = "\n".join(x4blk_lines)
x5blk = "\n".join(x5blk_lines)
assert x4blk in src, "x4/A4 macro block not found verbatim"
src = src.replace(x4blk, x4blk.replace("} }", "; } " + BS) + "\n" + x5blk, 1)
print("[gen] ACC6H2 x5/A5 block appended            1")

# ---- 3) ACC5H2 call sites: add the 5th x-arg and a-arg ----
def acc_call(m):
    xs = m.group(1)
    assert xs[:2] in ("xv", "xg"), m.group(0)
    x5 = xs[:2] + "5"
    a5 = m.group(11)[:-1] + "5"
    g = [m.group(i) for i in range(1, 12)]
    return "ACC6H2(" + ", ".join(g[:5] + [x5] + [g[5]] + g[6:] + [a5]) + ")"
pat = r"ACC5H2\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\)"
n = len(re.findall(pat, src))
src = re.sub(pat, acc_call, src)
print(f"[gen] {'ACC6H2 call sites':56s} {n}")
assert n == 13, n   # 13 calls (the 14th grep hit is the #define)

# ---- 4) RED5 call sites ----
def red_call(m):
    a5 = m.group(5)[:-1] + "5"
    return "RED6(" + ",".join(m.group(i) for i in range(1, 6)) + "," + a5 + ")"
pat = r"RED5\((\w+),(\w+),(\w+),(\w+),(\w+)\)"
n = len(re.findall(pat, src)); src = re.sub(pat, red_call, src)
print(f"[gen] {'RED6 call sites':56s} {n}"); assert n == 15, n   # 15 calls (16th hit = #define)

# ---- 5) LDH2 row-4 reads -> append row-5 ----
pat = r"LDH2\((\w+4), (\w+), (\w+), 4, (\w+)\)"
def ldh(m): return m.group(0) + f" LDH2({m.group(1)[:-1]}5, {m.group(2)}, {m.group(3)}, 5, {m.group(4)})"
n = len(re.findall(pat, src)); src = re.sub(pat, ldh, src)
print(f"[gen] {'LDH2 row-5 appends':56s} {n}"); assert n == 13, n

# ---- 6) accumulator decls ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f,a4=0\.f;",
    "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f;", 11, "compact a-decls")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f, a4 = 0\.f;",
    "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f;", 2, "spaced a-decls")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f,ag4=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f,au4=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f;", 1, "ffn ag/au decl")

# ---- 7) row-5 STORES (the R4 bug class — every write family) ----
rep(r"(\w+)\[4\*(\w+)\+(\w+)\] = \(__half\)(\w+)4;",
    r"\g<0> \1[5*\2+\3] = (__half)\g<4>5;", 11, "direct half stores")
rep(r"y4\[4\*DIM\+warp\] = hh4b\[4\*DIM\+warp\] \+ \(float\)\(\(__half\)a4\);",
    "y4[4*DIM+warp] = hh4b[4*DIM+warp] + (float)((__half)a4); y4[5*DIM+warp] = hh4b[5*DIM+warp] + (float)((__half)a5);", 1, "down residual store")
rep(r"gact4\[4\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag4\), \(__half\)au4\);",
    "gact4[4*FFN_N + warp] = __hmul(hsilu_h4((__half)ag4), (__half)au4); gact4[5*FFN_N + warp] = __hmul(hsilu_h4((__half)ag5), (__half)au5);", 1, "ffn gact store")
rep(r"\(OB\)\[4\*\(OS\)\+\(RIDX\)\] = \(__half\)a4; ",
    "(OB)[4*(OS)+(RIDX)] = (__half)a4; (OB)[5*(OS)+(RIDX)] = (__half)a5; ", 1, "aq3 macro store")
rep(r"a4 \+= __half2float\(__hmul\(z4\[4\*6144 \+ \(b<<5\)\+lane\], wh\)\);",
    "a4 += __half2float(__hmul(z4[4*6144 + (b<<5)+lane], wh));\n    a5 += __half2float(__hmul(z4[5*6144 + (b<<5)+lane], wh));", 1, "k3ao z accumulate")

# ---- 8) t-loops and h_embed/k0ab ----
rep(r"for \(int t = 0; t < 5; \+\+t\)", "for (int t = 0; t < 6; ++t)", 7, "t<6 loops")
rep(r"const int toks\[5\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\], s4\[0\] \};",
    "const int toks[6] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0] };", 1, "toks[6]")
rep(r"const int\* __restrict__ s4,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s4, const int* __restrict__ s5,\n    float* __restrict__ x4)", 1, "h_embed s5 arg")
rep(r"float rr\[5\];", "float rr[6];", 1, "rr[6]")

# ---- 9) k2s6 surgery (rec6x/conv6x scratch) ----
rep(r"void __launch_bounds__\(256\) k2s5\(", "void __launch_bounds__(256) k2s6(", 1, "k2s6 rename")
rep(r"float\* __restrict__ conv_b, float\* __restrict__ rec_b, float\* __restrict__ conv5x,",
    "float* __restrict__ conv_b, float* __restrict__ rec_b, float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ rec6x,", 1, "k2s6 sig scratch args")
rep(r"float\* rec_out = rec_b \+ \(size_t\)t\*\(NVH\*128\*128\) \+ \(size_t\)h\*128\*128;",
    "float* rec_out = (t == 5) ? (rec6x + (size_t)h*128*128)\n                                : (rec_b + (size_t)t*(NVH*128*128) + (size_t)h*128*128);", 1, "k2s6 rec_out t5->rec6x")
rep(r"float\* dst = \(t == 4\) \? conv5x : \(conv_b \+ \(size_t\)t \* \(3\*CONV_CH\)\);",
    "float* dst = (t == 4) ? conv5x : (t == 5) ? conv6x : (conv_b + (size_t)t * (3*CONV_CH));", 1, "k2s6 conv dst t4/5 scratch")

# ---- 10) kernel renames _5 -> _6 (bounds PRESERVED) ----
REN = {"h_embed5":"h_embed6","k0n5":"k0n6","k0ab5":"k0ab6","q5g8v5":"q5g8v6",
       "op38nw32_5":"op38nw32_6","k3aonw32_5":"k3aonw32_6","ao8nw32_5":"ao8nw32_6","hh5":"hh6",
       "ffn8v5":"ffn8v6","down8nw32_5":"down8nw32_6","aq3k8v5":"aq3k8v6","aq6k8v5":"aq6k8v6","head8v5":"head8v6"}
for a, b in REN.items():
    rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__(\1) {b}(", 1, f"rename {b}")
src = src.replace("#define IQ3V4(QP, SP, DP, A0, A1, A2, A3, A4) { " + BS, "#define IQ3V6(QP, SP, DP, A0, A1, A2, A3, A4, A5) { " + BS, 1)
src = src.replace("IQ3V4(qg, sg, dg, ag0, ag1, ag2, ag3, ag4)", "IQ3V6(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5)", 1)
src = src.replace("IQ3V4(qu, su, du, au0, au1, au2, au3, au4)", "IQ3V6(qu, su, du, au0, au1, au2, au3, au4, au5)", 1)
src = src.replace("IQ3V4(", "IQ3V6(")
src = src.replace("#define IQ3V4(", "#define IQ3V6(")
src = src.replace("// engine0 W2D-L2: T=5 PROBE trunk kernels (M=5;",
                  "// engine0 R5-K5: T=6 PROBE trunk kernels (M=6; k2s6 slots 0..4 + rec6x t=5; conv slots 0..3 + conv5x t=4 + conv6x t=5; [48][5] layout + live=slot-4 PRESERVED);", 1)

# ---- AUDITS (fail-loud; the R4 bug class) ----
for bad in ("ACC5H2(", "RED5(", "IQ3V4(", "toks[5]", "rr[5]"):
    assert bad not in src, f"leftover: {bad}"
assert " t < 5;" not in src, "leftover t<5 loop"
bodies = re.split(r'(?=extern "C" __global__ void)', src)
nk = 0
for b in bodies:
    m = re.match(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(', b)
    if not m: continue
    nk += 1
    nm, bounds = m.group(2), m.group(1)
    if "nw32" in nm: assert bounds == "1024", f"{nm} bounds {bounds} != 1024 (name law)"
    if re.search(r"\ba4\b", b) and "float* __restrict__ a4" not in b: assert re.search(r"\ba5\b", b), f"{nm} has a4 but no a5"
    if re.search(r"\bag4\b", b): assert re.search(r"\bag5\b", b), f"{nm} ag4 no ag5"
    for arr, s in (("qkv4","10240"),("gate4","6144"),("krow4","1024"),("vrow4","1024"),
                   ("qrow4","12288"),("logits4","VOCAB"),("attn_out4","DIM"),("gact4","FFN_N"),("y4","DIM")):
        if re.search(rf"{arr}\[4\*{s}", b):
            assert re.search(rf"{arr}\[5\*{s}", b), f"{nm} {arr}[4*{s} store without [5*"
assert nk == 14, nk
assert "rec6x" in src and "conv6x" in src
open(f"{BASE}/m6.cu", "w").write(src)
print(f"[gen] m6.cu written ({nk} kernels) — ALL AUDITS PASS")
