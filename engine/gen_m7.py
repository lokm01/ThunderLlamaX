# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R5 K=6: generate m7.cu (M=7/T=7 trunk kernels) from the FIXED m6.cu.
Same laws as gen_m6 (bounds preserved; every write family gains the row-6
store; a5->a6; ACC7H2/RED7/IQ3V7; k2s7: t=6 rec->rec7x, conv->conv7x)."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/m6.cu").read()
BS = chr(92)

def rep(pat, repl, expect, label):
    global src
    src, cnt = re.subn(pat, repl, src)
    print(f"[gen] {label:56s} {cnt}")
    if cnt != expect:
        print(f"FAIL: {label}: {cnt} != {expect}"); sys.exit(1)

# ---- 1) RED6 -> RED7 ----
rep(r"#define RED6\(A0,A1,A2,A3,A4,A5\) \{ \\",
    lambda m: "#define RED7(A0,A1,A2,A3,A4,A5,A6) { " + BS, 1, "RED7 def rename")
rep(r"A5 \+= __shfl_down_sync\(FULL, A5, o\); \} \}",
    "A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); } }", 1, "RED7 A6 line")

# ---- 2) ACC6H2 -> ACC7H2 ----
rep(r"#define ACC6H2\(X0, X1, X2, X3, X4, X5, WV, A0, A1, A2, A3, A4, A5\) \{ \\",
    lambda m: "#define ACC7H2(X0, X1, X2, X3, X4, X5, X6, WV, A0, A1, A2, A3, A4, A5, A6) { " + BS, 1, "ACC7H2 def rename")
rep(r"const __half2\* x5 = \(X5\); \\",
    lambda m: "const __half2* x5 = (X5); const __half2* x6 = (X6); " + BS, 1, "ACC7H2 x6 ptr")
x5blk = "\n".join([
    "    { const float2 p = __half22float2(__hmul2(x5[0], w01)); A5 += p.x; A5 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x5[1], w23)); A5 += p.x; A5 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x5[2], w45)); A5 += p.x; A5 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x5[3], w67)); A5 += p.x; A5 += p.y; } }",
])
x6blk = "\n".join(l.replace("x5", "x6").replace("A5", "A6") for l in x5blk.split("\n"))
assert x5blk in src, "x5/A5 macro block not found verbatim"
src = src.replace(x5blk, x5blk.replace("} }", "; } " + BS) + "\n" + x6blk, 1)
print("[gen] ACC7H2 x6/A6 block appended            1")

def acc_call(m):
    xs = m.group(1)
    assert xs[:2] in ("xv", "xg"), m.group(0)
    x6 = xs[:2] + "6"
    a6 = m.group(13)[:-1] + "6"
    g = [m.group(i) for i in range(1, 14)]
    return "ACC7H2(" + ", ".join(g[:6] + [x6] + [g[6]] + g[7:] + [a6]) + ")"
pat = r"ACC6H2\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\)"
n = len(re.findall(pat, src))
src = re.sub(pat, acc_call, src)
print(f"[gen] {'ACC7H2 call sites':56s} {n}")
assert n == 13, n

def red_call(m):
    a6 = m.group(6)[:-1] + "6"
    return "RED7(" + ",".join(m.group(i) for i in range(1, 7)) + "," + a6 + ")"
pat = r"RED6\((\w+),(\w+),(\w+),(\w+),(\w+),(\w+)\)"
n = len(re.findall(pat, src)); src = re.sub(pat, red_call, src)
print(f"[gen] {'RED7 call sites':56s} {n}"); assert n == 15, n

pat = r"LDH2\((\w+5), (\w+), (\w+), 5, (\w+)\)"
def ldh(m): return m.group(0) + f" LDH2({m.group(1)[:-1]}6, {m.group(2)}, {m.group(3)}, 6, {m.group(4)})"
n = len(re.findall(pat, src)); src = re.sub(pat, ldh, src)
print(f"[gen] {'LDH2 row-6 appends':56s} {n}"); assert n == 13, n

# ---- decls ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f,a4=0\.f,a5=0\.f;",
    "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f;", 11, "compact a-decls")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f, a4 = 0\.f, a5 = 0\.f;",
    "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f;", 2, "spaced a-decls")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f,ag4=0\.f,ag5=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f,au4=0\.f,au5=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f;", 1, "ffn ag/au decl")

# ---- row-6 stores ----
rep(r"(\w+)\[5\*(\w+)\+(\w+)\] = \(__half\)(\w+)5;",
    r"\g<0> \1[6*\2+\3] = (__half)\g<4>6;", 11, "direct half stores row6")
rep(r"y4\[5\*DIM\+warp\] = hh4b\[5\*DIM\+warp\] \+ \(float\)\(\(__half\)a5\);",
    "y4[5*DIM+warp] = hh4b[5*DIM+warp] + (float)((__half)a5); y4[6*DIM+warp] = hh4b[6*DIM+warp] + (float)((__half)a6);", 1, "down residual store row6")
rep(r"gact4\[5\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag5\), \(__half\)au5\);",
    "gact4[5*FFN_N + warp] = __hmul(hsilu_h4((__half)ag5), (__half)au5); gact4[6*FFN_N + warp] = __hmul(hsilu_h4((__half)ag6), (__half)au6);", 1, "ffn gact store row6")
rep(r"\(OB\)\[5\*\(OS\)\+\(RIDX\)\] = \(__half\)a5; ",
    "(OB)[5*(OS)+(RIDX)] = (__half)a5; (OB)[6*(OS)+(RIDX)] = (__half)a6; ", 1, "aq3 macro store row6")
rep(r"a5 \+= __half2float\(__hmul\(z4\[5\*6144 \+ \(b<<5\)\+lane\], wh\)\);",
    "a5 += __half2float(__hmul(z4[5*6144 + (b<<5)+lane], wh));\n    a6 += __half2float(__hmul(z4[6*6144 + (b<<5)+lane], wh));", 1, "k3ao z accumulate row6")

# ---- t-loops / h_embed / k0ab ----
rep(r"for \(int t = 0; t < 6; \+\+t\)", "for (int t = 0; t < 7; ++t)", 7, "t<7 loops")
rep(r"const int toks\[6\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\], s4\[0\], s5\[0\] \};",
    "const int toks[7] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0] };", 1, "toks[7]")
rep(r"const int\* __restrict__ s5,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s5, const int* __restrict__ s6,\n    float* __restrict__ x4)", 1, "h_embed s6 arg")
rep(r"float rr\[6\];", "float rr[7];", 1, "rr[7]")

# ---- k2s7 surgery: t=6 rec->rec7x, conv->conv7x ----
rep(r"void __launch_bounds__\(256\) k2s6\(", "void __launch_bounds__(256) k2s7(", 1, "k2s7 rename")
rep(r"float\* __restrict__ conv5x, float\* __restrict__ conv6x, float\* __restrict__ rec6x,",
    "float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ rec6x, float* __restrict__ rec7x,", 1, "k2s7 sig scratch args")
rep(r"float\* rec_out = \(t == 5\) \? \(rec6x \+ \(size_t\)h\*128\*128\)\n                                : \(rec_b \+ \(size_t\)t\*\(NVH\*128\*128\) \+ \(size_t\)h\*128\*128\);",
    "float* rec_out = (t >= 5) ? (((t == 5) ? rec6x : rec7x) + (size_t)h*128*128)\n                                : (rec_b + (size_t)t*(NVH*128*128) + (size_t)h*128*128);", 1, "k2s7 rec_out t5/6->scratch")
rep(r"float\* dst = \(t == 4\) \? conv5x : \(t == 5\) \? conv6x : \(conv_b \+ \(size_t\)t \* \(3\*CONV_CH\)\);",
    "float* dst = (t == 4) ? conv5x : (t == 5) ? conv6x : (t == 6) ? conv7x : (conv_b + (size_t)t * (3*CONV_CH));", 1, "k2s7 conv dst t4/5/6")

# ---- renames _6 -> _7 (bounds preserved) ----
REN = {"h_embed6":"h_embed7","k0n6":"k0n7","k0ab6":"k0ab7","q5g8v6":"q5g8v7",
       "op38nw32_6":"op38nw32_7","k3aonw32_6":"k3aonw32_7","ao8nw32_6":"ao8nw32_7","hh6":"hh7",
       "ffn8v6":"ffn8v7","down8nw32_6":"down8nw32_7","aq3k8v6":"aq3k8v7","aq6k8v6":"aq6k8v7","head8v6":"head8v7"}
for a, b in REN.items():
    rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__(\1) {b}(", 1, f"rename {b}")
src = src.replace("#define IQ3V6(QP, SP, DP, A0, A1, A2, A3, A4, A5) { " + BS,
                  "#define IQ3V7(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6) { " + BS, 1)
src = src.replace("IQ3V6(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5)",
                  "IQ3V7(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6)", 1)
src = src.replace("IQ3V6(qu, su, du, au0, au1, au2, au3, au4, au5)",
                  "IQ3V7(qu, su, du, au0, au1, au2, au3, au4, au5, au6)", 1)
src = src.replace("IQ3V6(", "IQ3V7(")
src = src.replace("// engine0 R5-K5: T=6 PROBE trunk kernels (M=6;",
                  "// engine0 R5-K6: T=7 PROBE trunk kernels (M=7; k2s7 slots 0..4 + rec6x t=5 + rec7x t=6; conv slots 0..3 + conv5x/6x/7x; [48][5] layout + live=slot-4 PRESERVED);", 1)

# ---- AUDITS ----
for bad in ("ACC6H2(", "RED6(", "IQ3V6(", "toks[6]", "rr[6]"):
    assert bad not in src, f"leftover: {bad}"
assert " t < 6;" not in src, "leftover t<6 loop"
bodies = re.split(r'(?=extern "C" __global__ void)', src)
nk = 0
for b in bodies:
    m = re.match(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(', b)
    if not m: continue
    nk += 1
    nm, bounds = m.group(2), m.group(1)
    if "nw32" in nm: assert bounds == "1024", f"{nm} bounds {bounds} != 1024 (name law)"
    if re.search(r"\ba5\b", b) and "float* __restrict__ a4" not in b:
        assert re.search(r"\ba6\b", b), f"{nm} has a5 but no a6"
    if re.search(r"\bag5\b", b): assert re.search(r"\bag6\b", b), f"{nm} ag5 no ag6"
    for arr, s in (("qkv4","10240"),("gate4","6144"),("krow4","1024"),("vrow4","1024"),
                   ("qrow4","12288"),("logits4","VOCAB"),("attn_out4","DIM"),("gact4","FFN_N"),("y4","DIM")):
        if re.search(rf"{arr}\[5\*{s}", b):
            assert re.search(rf"{arr}\[6\*{s}", b), f"{nm} {arr}[5*{s} store without [6*"
assert nk == 14, nk
assert "rec7x" in src and "conv7x" in src
open(f"{BASE}/m7.cu", "w").write(src)
print(f"[gen] m7.cu written ({nk} kernels) — ALL AUDITS PASS")
