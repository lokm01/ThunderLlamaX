# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R5d K=7: generate m8.cu (M=8/T=8 trunk kernels) from the FIXED m7.cu.
Same laws as gen_m6/gen_m7 (bounds preserved; every write family gains the
row-7 store; a6->a7; ACC8H2/RED8/IQ3V8; k2s8: t=7 rec->rec8x, conv->conv8x,
t=7 rec_in reads rec7x = the t=6 state — the REC-CHAIN SLOT LAW at depth 8)."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/m7.cu").read()
BS = chr(92)

def rep(pat, repl, expect, label):
    global src
    src, cnt = re.subn(pat, repl, src)
    print(f"[gen] {label:56s} {cnt}")
    if cnt != expect:
        print(f"FAIL: {label}: {cnt} != {expect}"); sys.exit(1)

# ---- 1) RED7 -> RED8 ----
rep(r"#define RED7\(A0,A1,A2,A3,A4,A5,A6\) \{ \\",
    lambda m: "#define RED8(A0,A1,A2,A3,A4,A5,A6,A7) { " + BS, 1, "RED8 def rename")
rep(r"A6 \+= __shfl_down_sync\(FULL, A6, o\); \} \}",
    "A6 += __shfl_down_sync(FULL, A6, o); A7 += __shfl_down_sync(FULL, A7, o); } }", 1, "RED8 A7 line")

# ---- 2) ACC7H2 -> ACC8H2 ----
rep(r"#define ACC7H2\(X0, X1, X2, X3, X4, X5, X6, WV, A0, A1, A2, A3, A4, A5, A6\) \{ \\",
    lambda m: "#define ACC8H2(X0, X1, X2, X3, X4, X5, X6, X7, WV, A0, A1, A2, A3, A4, A5, A6, A7) { " + BS, 1, "ACC8H2 def rename")
rep(r"const __half2\* x6 = \(X6\); \\",
    lambda m: "const __half2* x6 = (X6); const __half2* x7 = (X7); " + BS, 1, "ACC8H2 x7 ptr")
x6blk = "\n".join([
    "    { const float2 p = __half22float2(__hmul2(x6[0], w01)); A6 += p.x; A6 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x6[1], w23)); A6 += p.x; A6 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x6[2], w45)); A6 += p.x; A6 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x6[3], w67)); A6 += p.x; A6 += p.y; } }",
])
x7blk = "\n".join(l.replace("x6", "x7").replace("A6", "A7") for l in x6blk.split("\n"))
assert x6blk in src, "x6/A6 macro block not found verbatim"
src = src.replace(x6blk, x6blk.replace("} }", "; } " + BS) + "\n" + x7blk, 1)
print("[gen] ACC8H2 x7/A7 block appended            1")

def acc_call(m):
    xs = m.group(1)
    assert xs[:2] in ("xv", "xg"), m.group(0)
    x7 = xs[:2] + "7"
    a7 = m.group(15)[:-1] + "7"
    g = [m.group(i) for i in range(1, 16)]
    return "ACC8H2(" + ", ".join(g[:7] + [x7] + [g[7]] + g[8:] + [a7]) + ")"
pat = r"ACC7H2\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\)"
n = len(re.findall(pat, src))
src = re.sub(pat, acc_call, src)
print(f"[gen] {'ACC8H2 call sites':56s} {n}")
assert n == 13, n

def red_call(m):
    a7 = m.group(7)[:-1] + "7"
    return "RED8(" + ",".join(m.group(i) for i in range(1, 8)) + "," + a7 + ")"
pat = r"RED7\((\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+)\)"
n = len(re.findall(pat, src)); src = re.sub(pat, red_call, src)
print(f"[gen] {'RED8 call sites':56s} {n}"); assert n == 15, n

pat = r"LDH2\((\w+6), (\w+), (\w+), 6, (\w+)\)"
def ldh(m): return m.group(0) + f" LDH2({m.group(1)[:-1]}7, {m.group(2)}, {m.group(3)}, 7, {m.group(4)})"
n = len(re.findall(pat, src)); src = re.sub(pat, ldh, src)
print(f"[gen] {'LDH2 row-7 appends':56s} {n}"); assert n == 13, n

# ---- decls ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f,a4=0\.f,a5=0\.f,a6=0\.f;",
    "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;", 11, "compact a-decls")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f, a4 = 0\.f, a5 = 0\.f, a6 = 0\.f;",
    "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;", 2, "spaced a-decls")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f,ag4=0\.f,ag5=0\.f,ag6=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f,au4=0\.f,au5=0\.f,au6=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f;", 1, "ffn ag/au decl")

# ---- row-7 stores ----
rep(r"(\w+)\[6\*(\w+)\+(\w+)\] = \(__half\)(\w+)6;",
    r"\g<0> \1[7*\2+\3] = (__half)\g<4>7;", 11, "direct half stores row7")
rep(r"y4\[6\*DIM\+warp\] = hh4b\[6\*DIM\+warp\] \+ \(float\)\(\(__half\)a6\);",
    "y4[6*DIM+warp] = hh4b[6*DIM+warp] + (float)((__half)a6); y4[7*DIM+warp] = hh4b[7*DIM+warp] + (float)((__half)a7);", 1, "down residual store row7")
rep(r"gact4\[6\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag6\), \(__half\)au6\);",
    "gact4[6*FFN_N + warp] = __hmul(hsilu_h4((__half)ag6), (__half)au6); gact4[7*FFN_N + warp] = __hmul(hsilu_h4((__half)ag7), (__half)au7);", 1, "ffn gact store row7")
rep(r"\(OB\)\[6\*\(OS\)\+\(RIDX\)\] = \(__half\)a6; ",
    "(OB)[6*(OS)+(RIDX)] = (__half)a6; (OB)[7*(OS)+(RIDX)] = (__half)a7; ", 1, "aq3 macro store row7")
rep(r"a6 \+= __half2float\(__hmul\(z4\[6\*6144 \+ \(b<<5\)\+lane\], wh\)\);",
    "a6 += __half2float(__hmul(z4[6*6144 + (b<<5)+lane], wh));\n    a7 += __half2float(__hmul(z4[7*6144 + (b<<5)+lane], wh));", 1, "k3ao z accumulate row7")

# ---- t-loops / h_embed / k0ab ----
rep(r"for \(int t = 0; t < 7; \+\+t\)", "for (int t = 0; t < 8; ++t)", 7, "t<8 loops")
rep(r"const int toks\[7\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\], s4\[0\], s5\[0\], s6\[0\] \};",
    "const int toks[8] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0], s7[0] };", 1, "toks[8]")
rep(r"const int\* __restrict__ s6,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s6, const int* __restrict__ s7,\n    float* __restrict__ x4)", 1, "h_embed s7 arg")
rep(r"float rr\[7\];", "float rr[8];", 1, "rr[8]")

# ---- k2s8 surgery: t=7 rec->rec8x, conv->conv8x; t=7 rec_in -> rec7x ----
rep(r"void __launch_bounds__\(256\) k2s7\(", "void __launch_bounds__(256) k2s8(", 1, "k2s8 rename")

def srep(a, b, label):
    global src
    n = src.count(a)
    print(f"[gen] {label:56s} {n}")
    assert n == 1, f"{label}: {n} != 1"
    src = src.replace(a, b, 1)

srep("float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ rec6x, float* __restrict__ rec7x,",
     "float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x,",
     "k2s8 sig scratch args")
# REC-CHAIN SLOT LAW at depth 8: t=7's rec_in reads rec7x (the t=6 scratch —
# slot 5 does not exist in [48][5]; the naive (t-1) index reads the NEXT
# BLOCK's slot 0 = garbage -> NaN row 7, the k2s7 bug class).
srep(": (t == 6) ? (rec6x + (size_t)h*128*128)\n",
     ": (t == 6) ? (rec6x + (size_t)h*128*128)\n" + " "*35 + ": (t == 7) ? (rec7x + (size_t)h*128*128)\n",
     "k2s8 rec_in t7->rec7x")
srep("(((t == 5) ? rec6x : rec7x) + (size_t)h*128*128)",
     "(((t == 5) ? rec6x : (t == 6) ? rec7x : rec8x) + (size_t)h*128*128)",
     "k2s8 rec_out t5/6/7->scratch")
srep("(t == 6) ? conv7x : (conv_b + (size_t)t * (3*CONV_CH));",
     "(t == 6) ? conv7x : (t == 7) ? conv8x : (conv_b + (size_t)t * (3*CONV_CH));",
     "k2s8 conv dst t4/5/6/7")

# ---- renames _7 -> _8 (bounds preserved) ----
REN = {"h_embed7":"h_embed8","k0n7":"k0n8","k0ab7":"k0ab8","q5g8v7":"q5g8v8",
       "op38nw32_7":"op38nw32_8","k3aonw32_7":"k3aonw32_8","ao8nw32_7":"ao8nw32_8","hh7":"hh8",
       "ffn8v7":"ffn8v8","down8nw32_7":"down8nw32_8","aq3k8v7":"aq3k8v8","aq6k8v7":"aq6k8v8","head8v7":"head8v8"}
for a, b in REN.items():
    rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__(\1) {b}(", 1, f"rename {b}")
src = src.replace("#define IQ3V7(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6) { " + BS,
                  "#define IQ3V8(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7) { " + BS, 1)
src = src.replace("IQ3V7(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6)",
                  "IQ3V8(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7)", 1)
src = src.replace("IQ3V7(qu, su, du, au0, au1, au2, au3, au4, au5, au6)",
                  "IQ3V8(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7)", 1)
src = src.replace("IQ3V7(", "IQ3V8(")
src = src.replace("// engine0 R5-K6: T=7 PROBE trunk kernels (M=7; k2s7 slots 0..4 + rec6x t=5 + rec7x t=6; conv slots 0..3 + conv5x/6x/7x; [48][5] layout + live=slot-4 PRESERVED);",
                  "// engine0 R5d-K7: T=8 PROBE trunk kernels (M=8; k2s8 slots 0..4 + rec6x t=5 + rec7x t=6 + rec8x t=7; conv slots 0..3 + conv5x/6x/7x/8x; [48][5] layout + live=slot-4 PRESERVED);", 1)

# ---- AUDITS ----
for bad in ("ACC7H2(", "RED7(", "IQ3V7(", "toks[7]", "rr[7]"):
    assert bad not in src, f"leftover: {bad}"
assert " t < 7;" not in src, "leftover t<7 loop"
bodies = re.split(r'(?=extern "C" __global__ void)', src)
nk = 0
for b in bodies:
    m = re.match(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(', b)
    if not m: continue
    nk += 1
    nm, bounds = m.group(2), m.group(1)
    if "nw32" in nm: assert bounds == "1024", f"{nm} bounds {bounds} != 1024 (name law)"
    if re.search(r"\ba6\b", b) and "float* __restrict__ a4" not in b:
        assert re.search(r"\ba7\b", b), f"{nm} has a6 but no a7"
    if re.search(r"\bag6\b", b): assert re.search(r"\bag7\b", b), f"{nm} ag6 no ag7"
    if re.search(r"\bau6\b", b): assert re.search(r"\bau7\b", b), f"{nm} au6 no au7"
    for arr, s in (("qkv4","10240"),("gate4","6144"),("krow4","1024"),("vrow4","1024"),
                   ("qrow4","12288"),("logits4","VOCAB"),("attn_out4","DIM"),("gact4","FFN_N"),("y4","DIM")):
        if re.search(rf"{arr}\[6\*{s}", b):
            assert re.search(rf"{arr}\[7\*{s}", b), f"{nm} {arr}[6*{s} store without [7*"
assert nk == 14, nk
assert "rec8x" in src and "conv8x" in src
# REC-CHAIN SLOT LAW at depth 8: t=7's rec_in must read rec7x (the t=6 scratch)
k2s8 = next(b for b in bodies if re.match(r'extern "C" __global__ void __launch_bounds__\(\d+\) k2s8\(', b))
assert "(t == 7) ? (rec7x" in k2s8, "k2s8 t=7 rec_in does not read rec7x"
assert "t < 8" in k2s8, "k2s8 t-loop not extended"
open(f"{BASE}/m8.cu", "w").write(src)
print(f"[gen] m8.cu written ({nk} kernels) — ALL AUDITS PASS")
