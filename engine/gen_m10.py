# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 K=9: generate m10.cu (M=10/T=10 trunk kernels) from m9.cu (the R7a K=8
state). Same laws as gen_m9 (bounds preserved; every write family gains the
row-9 store; a8->a9; ACC10H2/RED10/IQ3V10; k2s10: t=9 rec->rec10x, conv->
conv10x, t=9 rec_in reads rec9x = the t=8 state — the REC-CHAIN SLOT LAW at
depth 10)."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/m9.cu").read()
BS = chr(92)

def rep(pat, repl, expect, label):
  global src
  src, cnt = re.subn(pat, repl, src)
  print(f"[gen] {label:56s} {cnt}")
  if cnt != expect:
    print(f"FAIL: {label}: {cnt} != {expect}"); sys.exit(1)

# ---- 1) RED9 -> RED10 ----
rep(r"#define RED9\(A0,A1,A2,A3,A4,A5,A6,A7,A8\) \{ \\",
    lambda m: "#define RED10(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9) { " + BS, 1, "RED10 def rename")
rep(r"A8 \+= __shfl_down_sync\(FULL, A8, o\); \} \}",
    "A8 += __shfl_down_sync(FULL, A8, o); A9 += __shfl_down_sync(FULL, A9, o); } }", 1, "RED10 A9 line")

# ---- 2) ACC9H2 -> ACC10H2 ----
rep(r"#define ACC9H2\(X0, X1, X2, X3, X4, X5, X6, X7, X8, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8\) \{ \\",
    lambda m: "#define ACC10H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { " + BS, 1, "ACC10H2 def rename")
rep(r"const __half2\* x8 = \(X8\); \\",
    lambda m: "const __half2* x8 = (X8); const __half2* x9 = (X9); " + BS, 1, "ACC10H2 x9 ptr")
x8blk = "\n".join([
    "    { const float2 p = __half22float2(__hmul2(x8[0], w01)); A8 += p.x; A8 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x8[1], w23)); A8 += p.x; A8 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x8[2], w45)); A8 += p.x; A8 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x8[3], w67)); A8 += p.x; A8 += p.y; } }",
])
x9blk = "\n".join(l.replace("x8", "x9").replace("A8", "A9") for l in x8blk.split("\n"))
if x8blk not in src:
  print("FAIL: x8/A8 macro block not found verbatim"); sys.exit(1)
src = src.replace(x8blk, x8blk.replace("} }", "; } " + BS) + "\n" + x9blk, 1)
print("[gen] ACC10H2 x9/A9 block appended           1")

def acc_call(m):
  xs = m.group(1)
  assert xs[:2] in ("xv", "xg"), m.group(0)
  x9 = xs[:2] + "9"
  a9 = m.group(19)[:-1] + "9"
  g = [m.group(i) for i in range(1, 20)]
  return "ACC10H2(" + ", ".join(g[:9] + [x9] + [g[9]] + g[10:] + [a9]) + ")"
pat = r"ACC9H2\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\)"
n = len(re.findall(pat, src))
src = re.sub(pat, acc_call, src)
print(f"[gen] {'ACC10H2 call sites':56s} {n}")
assert n == 13, n

def red_call(m):
  a9 = m.group(9)[:-1] + "9"
  return "RED10(" + ",".join(m.group(i) for i in range(1, 10)) + "," + a9 + ")"
pat = r"RED9\((\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+)\)"
n = len(re.findall(pat, src)); src = re.sub(pat, red_call, src)
print(f"[gen] {'RED10 call sites':56s} {n}"); assert n == 15, n

pat = r"LDH2\((\w+)8, (\w+), (\w+), 8, (\w+)\)"
def ldh(m): return m.group(0) + f" LDH2({m.group(1)}9, {m.group(2)}, {m.group(3)}, 9, {m.group(4)})"
n = len(re.findall(pat, src)); src = re.sub(pat, ldh, src)
print(f"[gen] {'LDH2 row-9 appends':56s} {n}"); assert n == 13, n

# ---- decls ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f,a4=0\.f,a5=0\.f,a6=0\.f,a7=0\.f,a8=0\.f;",
    "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f;", 11, "compact a-decls")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f, a4 = 0\.f, a5 = 0\.f, a6 = 0\.f, a7 = 0\.f, a8 = 0\.f;",
    "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f;", 2, "spaced a-decls")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f,ag4=0\.f,ag5=0\.f,ag6=0\.f,ag7=0\.f,ag8=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f,au4=0\.f,au5=0\.f,au6=0\.f,au7=0\.f,au8=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f;", 1, "ffn ag/au decl")

# ---- row-9 stores ----
rep(r"(\w+)\[8\*(\w+)\+(\w+)\] = \(__half\)(\w+)8;",
    r"\g<0> \1[9*\2+\3] = (__half)\g<4>9;", 11, "direct half stores row9")
rep(r"y4\[8\*DIM\+warp\] = hh4b\[8\*DIM\+warp\] \+ \(float\)\(\(__half\)a8\);",
    "y4[8*DIM+warp] = hh4b[8*DIM+warp] + (float)((__half)a8); y4[9*DIM+warp] = hh4b[9*DIM+warp] + (float)((__half)a9);", 1, "down residual store row9")
rep(r"gact4\[8\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag8\), \(__half\)au8\);",
    "gact4[8*FFN_N + warp] = __hmul(hsilu_h4((__half)ag8), (__half)au8); gact4[9*FFN_N + warp] = __hmul(hsilu_h4((__half)ag9), (__half)au9);", 1, "ffn gact store row9")
rep(r"\(OB\)\[8\*\(OS\)\+\(RIDX\)\] = \(__half\)a8;",
    "(OB)[8*(OS)+(RIDX)] = (__half)a8; (OB)[9*(OS)+(RIDX)] = (__half)a9;", 1, "aq3 macro store row9")
rep(r"a8 \+= __half2float\(__hmul\(z4\[8\*6144 \+ \(b<<5\)\+lane\], wh\)\);",
    "a8 += __half2float(__hmul(z4[8*6144 + (b<<5)+lane], wh));\n    a9 += __half2float(__hmul(z4[9*6144 + (b<<5)+lane], wh));", 1, "k3ao z accumulate row9")

# ---- h_embed10: toks[10] + s9 arg ----
rep(r"const int toks\[9\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\], s4\[0\], s5\[0\], s6\[0\], s7\[0\], s8\[0\] \};",
    "const int toks[10] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0], s7[0], s8[0], s9[0] };", 1, "toks[10]")
rep(r"const int\* __restrict__ s7, const int\* __restrict__ s8,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s7, const int* __restrict__ s8, const int* __restrict__ s9,\n    float* __restrict__ x4)", 1, "h_embed s9 arg")

# ---- k0ab10: 10-row decode (t = blockIdx.x % 10, g = blockIdx.x / 10) ----
rep(r"const int t = blockIdx\.x % 9;\n  const int g = blockIdx\.x / 9;",
    "const int t = blockIdx.x % 10;\n  const int g = blockIdx.x / 10;", 1, "k0ab10 10-row decode")

# ---- k2s10 surgery ----
rep(r"void __launch_bounds__\(256\) k2s9\(", "void __launch_bounds__(256) k2s10(", 1, "k2s10 rename")
rep(r"for \(int t = 0; t < 9; \+\+t\)", "for (int t = 0; t < 10; ++t)", 1, "k2s10 t<10 loop")

def srep(a, b, label):
  global src
  n = src.count(a)
  print(f"[gen] {label:56s} {n}")
  assert n == 1, f"{label}: {n} != 1"
  src = src.replace(a, b, 1)

srep("float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ conv9x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x, float* __restrict__ rec9x,",
     "float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ conv9x, float* __restrict__ conv10x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x, float* __restrict__ rec9x, float* __restrict__ rec10x,",
     "k2s10 sig scratch args")
# REC-CHAIN SLOT LAW at depth 10: t=9's rec_in reads rec9x (the t=8 scratch).
srep("                                   : (t == 8) ? (rec8x + (size_t)h*128*128)\n",
     "                                   : (t == 8) ? (rec8x + (size_t)h*128*128)\n" + " "*35 + ": (t == 9) ? (rec9x + (size_t)h*128*128)\n",
     "k2s10 rec_in t9->rec9x")
srep("(((t == 5) ? rec6x : (t == 6) ? rec7x : (t == 7) ? rec8x : rec9x) + (size_t)h*128*128)",
     "(((t == 5) ? rec6x : (t == 6) ? rec7x : (t == 7) ? rec8x : (t == 8) ? rec9x : rec10x) + (size_t)h*128*128)",
     "k2s10 rec_out t5..9->scratch")
srep("(t == 7) ? conv8x : (t == 8) ? conv9x : (conv_b + (size_t)t * (3*CONV_CH));",
     "(t == 7) ? conv8x : (t == 8) ? conv9x : (t == 9) ? conv10x : (conv_b + (size_t)t * (3*CONV_CH));",
     "k2s10 conv dst t4..9")

# ---- renames _9 -> _10 (bounds preserved) ----
REN = {"h_embed9":"h_embed10","k0n9":"k0n10","k0ab9":"k0ab10","q5g8v9":"q5g8v10",
       "op38nw32_9":"op38nw32_10","k3aonw32_9":"k3aonw32_10","ao8nw32_9":"ao8nw32_10","hh9":"hh10",
       "ffn8v9":"ffn8v10","down8nw32_9":"down8nw32_10","aq3k8v9":"aq3k8v10","aq6k8v9":"aq6k8v10","head8v9":"head8v10"}
for a, b in REN.items():
  rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__(\1) {b}(", 1, f"rename {b}")
src = src.replace("#define IQ3V9(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8) { " + BS,
                  "#define IQ3V10(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { " + BS, 1)
src = src.replace("IQ3V9(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8)",
                  "IQ3V10(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9)", 1)
src = src.replace("IQ3V9(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8)",
                  "IQ3V10(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9)", 1)
src = src.replace("IQ3V9(", "IQ3V10(")
# R8 zero-spill knob: ao8nw32 at M=10 needs unroll 4->2 (8B spill at 4; the
# per-row fp order is untouched — unroll depth only). Scoped to the ao8 body
# (op38/k3ao share the pragma text and hold 64 regs at unroll 4).
_i0 = src.find("void __launch_bounds__(1024) ao8nw32_10(")
assert _i0 > 0, "ao8nw32_10 sig not found"
_i1 = src.find('extern "C" __global__', _i0)
_seg = src[_i0:_i1]
assert _seg.count("  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds") == 1, _seg.count("  #pragma unroll 4")
src = src[:_i0] + _seg.replace("  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds",
                               "  #pragma unroll 2  // R8: 64-reg budget at M=10 (was 4 at M<=9)", 1) + src[_i1:]
print("[gen] ao8nw32_10 unroll 4->2 (body-scoped)     1")
# R8 zero-spill knob 2: ffn8v10 (the DR7=0 fallback path) spills 16B at
# unroll 5 at M=10 — same knob as ao8 above.
_i0 = src.find("void __launch_bounds__(256) ffn8v10(")
assert _i0 > 0, "ffn8v10 sig not found"
_i1 = src.find('extern "C" __global__', _i0)
_seg = src[_i0:_i1]
assert _seg.count("  #pragma unroll 5\n") == 1, _seg.count("  #pragma unroll 5")
src = src[:_i0] + _seg.replace("  #pragma unroll 5\n", "  #pragma unroll 2  // R8: zero-spill at M=10\n", 1) + src[_i1:]
print("[gen] ffn8v10 unroll 5->2 (body-scoped)       1")
src = src.replace("// engine0 R7a-K8: T=9 PROBE trunk kernels (M=9; k2s9 slots 0..4 + rec6x t=5 + rec7x t=6 + rec8x t=7 + rec9x t=8; conv slots 0..3 + conv5x/6x/7x/8x; [48][5] layout + live=slot-4 PRESERVED);",
                  "// engine0 R8-K9: T=10 PROBE trunk kernels (M=10; k2s10 slots 0..4 + rec6x t=5..rec10x t=9; conv slots 0..3 + conv5x..conv10x; [48][5] layout + live=slot-4 PRESERVED);", 1)

# ---- AUDITS ----
for bad in ("ACC9H2(", "RED9(", "IQ3V9(", "toks[9]", "for (int t = 0; t < 9", "k2s9"):
  assert bad not in src, f"leftover: {bad}"
bodies = re.split(r'(?=extern "C" __global__ void)', src)
bodies = [b for b in bodies if b.strip()]
nk = 0
for b in bodies:
  m = re.match(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(', b)
  if not m: continue
  nk += 1
  nm, bounds = m.group(2), m.group(1)
  if "nw32" in nm: assert bounds == "1024", f"{nm} bounds {bounds} != 1024 (name law)"
  if re.search(r"\ba8\b", b) and "float* __restrict__ a4" not in b:
    assert re.search(r"\ba9\b", b), f"{nm} has a8 but no a9"
  if re.search(r"\bag8\b", b): assert re.search(r"\bag9\b", b), f"{nm} ag8 no ag9"
  if re.search(r"\bau8\b", b): assert re.search(r"\bau9\b", b), f"{nm} au8 no au9"
  for arr, s in (("qkv4","10240"),("gate4","6144"),("krow4","1024"),("vrow4","1024"),
                 ("qrow4","12288"),("logits4","VOCAB"),("attn_out4","DIM"),("gact4","FFN_N"),("y4","DIM")):
    if re.search(rf"{arr}\[8\*{s}", b):
      assert re.search(rf"{arr}\[9\*{s}", b), f"{nm} {arr}[8*{s} store without [9*"
assert nk == 14, nk
assert "rec10x" in src and "conv10x" in src
k2s10 = next(b for b in bodies if re.match(r'extern "C" __global__ void __launch_bounds__\(\d+\) k2s10\(', b))
assert "(t == 9) ? (rec9x" in k2s10, "k2s10 t=9 rec_in does not read rec9x"
assert "t < 10" in k2s10, "k2s10 t-loop not extended"
open(f"{BASE}/m10.cu", "w").write(src)
print(f"[gen] m10.cu written ({nk} kernels) — ALL AUDITS PASS")
