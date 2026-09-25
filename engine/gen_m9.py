# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a K=8: generate m9.cu (M=9/T=9 trunk kernels) from m8.cu (rung-3 state:
norms already per-row CTAs). Same laws as gen_m8 (bounds preserved; every
write family gains the row-8 store; a7->a8; ACC9H2/RED9/IQ3V9; k2s9: t=8
rec->rec9x, conv->conv9x, t=8 rec_in reads rec8x = the t=7 state — the
REC-CHAIN SLOT LAW at depth 9)."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/m8.cu").read()
BS = chr(92)

def rep(pat, repl, expect, label):
  global src
  src, cnt = re.subn(pat, repl, src)
  print(f"[gen] {label:56s} {cnt}")
  if cnt != expect:
    print(f"FAIL: {label}: {cnt} != {expect}"); sys.exit(1)

# ---- 1) RED8 -> RED9 ----
rep(r"#define RED8\(A0,A1,A2,A3,A4,A5,A6,A7\) \{ \\",
    lambda m: "#define RED9(A0,A1,A2,A3,A4,A5,A6,A7,A8) { " + BS, 1, "RED9 def rename")
rep(r"A7 \+= __shfl_down_sync\(FULL, A7, o\); \} \}",
    "A7 += __shfl_down_sync(FULL, A7, o); A8 += __shfl_down_sync(FULL, A8, o); } }", 1, "RED9 A8 line")

# ---- 2) ACC8H2 -> ACC9H2 ----
rep(r"#define ACC8H2\(X0, X1, X2, X3, X4, X5, X6, X7, WV, A0, A1, A2, A3, A4, A5, A6, A7\) \{ \\",
    lambda m: "#define ACC9H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8) { " + BS, 1, "ACC9H2 def rename")
rep(r"const __half2\* x7 = \(X7\); \\",
    lambda m: "const __half2* x7 = (X7); const __half2* x8 = (X8); " + BS, 1, "ACC9H2 x8 ptr")
x7blk = "\n".join([
    "    { const float2 p = __half22float2(__hmul2(x7[0], w01)); A7 += p.x; A7 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x7[1], w23)); A7 += p.x; A7 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x7[2], w45)); A7 += p.x; A7 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x7[3], w67)); A7 += p.x; A7 += p.y; } }",
])
x8blk = "\n".join(l.replace("x7", "x8").replace("A7", "A8") for l in x7blk.split("\n"))
if x7blk not in src:
  # gen_m8's emitted block had the '; } ' continuation quirk — find the actual
  print("FAIL: x7/A7 macro block not found verbatim"); sys.exit(1)
src = src.replace(x7blk, x7blk.replace("} }", "; } " + BS) + "\n" + x8blk, 1)
print("[gen] ACC9H2 x8/A8 block appended            1")

def acc_call(m):
  xs = m.group(1)
  assert xs[:2] in ("xv", "xg"), m.group(0)
  x8 = xs[:2] + "8"
  a8 = m.group(17)[:-1] + "8"
  g = [m.group(i) for i in range(1, 18)]
  return "ACC9H2(" + ", ".join(g[:8] + [x8] + [g[8]] + g[9:] + [a8]) + ")"
pat = r"ACC8H2\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\)"
n = len(re.findall(pat, src))
src = re.sub(pat, acc_call, src)
print(f"[gen] {'ACC9H2 call sites':56s} {n}")
assert n == 13, n

def red_call(m):
  a8 = m.group(8)[:-1] + "8"
  return "RED9(" + ",".join(m.group(i) for i in range(1, 9)) + "," + a8 + ")"
pat = r"RED8\((\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+)\)"
n = len(re.findall(pat, src)); src = re.sub(pat, red_call, src)
print(f"[gen] {'RED9 call sites':56s} {n}"); assert n == 15, n

pat = r"LDH2\((\w+7), (\w+), (\w+), 7, (\w+)\)"
def ldh(m): return m.group(0) + f" LDH2({m.group(1)[:-1]}8, {m.group(2)}, {m.group(3)}, 8, {m.group(4)})"
n = len(re.findall(pat, src)); src = re.sub(pat, ldh, src)
print(f"[gen] {'LDH2 row-8 appends':56s} {n}"); assert n == 13, n

# ---- decls ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f,a4=0\.f,a5=0\.f,a6=0\.f,a7=0\.f;",
    "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f;", 11, "compact a-decls")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f, a4 = 0\.f, a5 = 0\.f, a6 = 0\.f, a7 = 0\.f;",
    "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f, a8 = 0.f;", 2, "spaced a-decls")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f,ag4=0\.f,ag5=0\.f,ag6=0\.f,ag7=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f,au4=0\.f,au5=0\.f,au6=0\.f,au7=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f;", 1, "ffn ag/au decl")

# ---- row-8 stores ----
rep(r"(\w+)\[7\*(\w+)\+(\w+)\] = \(__half\)(\w+)7;",
    r"\g<0> \1[8*\2+\3] = (__half)\g<4>8;", 11, "direct half stores row8")
rep(r"y4\[7\*DIM\+warp\] = hh4b\[7\*DIM\+warp\] \+ \(float\)\(\(__half\)a7\);",
    "y4[7*DIM+warp] = hh4b[7*DIM+warp] + (float)((__half)a7); y4[8*DIM+warp] = hh4b[8*DIM+warp] + (float)((__half)a8);", 1, "down residual store row8")
rep(r"gact4\[7\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag7\), \(__half\)au7\);",
    "gact4[7*FFN_N + warp] = __hmul(hsilu_h4((__half)ag7), (__half)au7); gact4[8*FFN_N + warp] = __hmul(hsilu_h4((__half)ag8), (__half)au8);", 1, "ffn gact store row8")
rep(r"\(OB\)\[7\*\(OS\)\+\(RIDX\)\] = \(__half\)a7; ",
    "(OB)[7*(OS)+(RIDX)] = (__half)a7; (OB)[8*(OS)+(RIDX)] = (__half)a8; ", 1, "aq3 macro store row8")
rep(r"a7 \+= __half2float\(__hmul\(z4\[7\*6144 \+ \(b<<5\)\+lane\], wh\)\);",
    "a7 += __half2float(__hmul(z4[7*6144 + (b<<5)+lane], wh));\n    a8 += __half2float(__hmul(z4[8*6144 + (b<<5)+lane], wh));", 1, "k3ao z accumulate row8")

# ---- h_embed9: toks[9] + s8 arg ----
rep(r"const int toks\[8\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\], s4\[0\], s5\[0\], s6\[0\], s7\[0\] \};",
    "const int toks[9] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0], s7[0], s8[0] };", 1, "toks[9]")
rep(r"const int\* __restrict__ s7,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s7, const int* __restrict__ s8,\n    float* __restrict__ x4)", 1, "h_embed s8 arg")

# ---- k0ab9: 9-row decode (t = blockIdx.x % 9, g = blockIdx.x / 9) ----
rep(r"const int t = blockIdx\.x & 7;\n  const int g = blockIdx\.x >> 3;",
    "const int t = blockIdx.x % 9;\n  const int g = blockIdx.x / 9;", 1, "k0ab9 9-row decode")

# ---- k2s9 surgery ----
rep(r"void __launch_bounds__\(256\) k2s8\(", "void __launch_bounds__(256) k2s9(", 1, "k2s9 rename")
rep(r"for \(int t = 0; t < 8; \+\+t\)", "for (int t = 0; t < 9; ++t)", 1, "k2s9 t<9 loop")

def srep(a, b, label):
  global src
  n = src.count(a)
  print(f"[gen] {label:56s} {n}")
  assert n == 1, f"{label}: {n} != 1"
  src = src.replace(a, b, 1)

srep("float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x,",
     "float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ conv9x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x, float* __restrict__ rec9x,",
     "k2s9 sig scratch args")
# REC-CHAIN SLOT LAW at depth 9: t=8's rec_in reads rec8x (the t=7 scratch).
srep("                                   : (t == 7) ? (rec7x + (size_t)h*128*128)\n",
     "                                   : (t == 7) ? (rec7x + (size_t)h*128*128)\n" + " "*35 + ": (t == 8) ? (rec8x + (size_t)h*128*128)\n",
     "k2s9 rec_in t8->rec8x")
srep("(((t == 5) ? rec6x : (t == 6) ? rec7x : rec8x) + (size_t)h*128*128)",
     "(((t == 5) ? rec6x : (t == 6) ? rec7x : (t == 7) ? rec8x : rec9x) + (size_t)h*128*128)",
     "k2s9 rec_out t5/6/7/8->scratch")
srep("(t == 6) ? conv7x : (t == 7) ? conv8x : (conv_b + (size_t)t * (3*CONV_CH));",
     "(t == 6) ? conv7x : (t == 7) ? conv8x : (t == 8) ? conv9x : (conv_b + (size_t)t * (3*CONV_CH));",
     "k2s9 conv dst t4..8")

# ---- renames _8 -> _9 (bounds preserved) ----
REN = {"h_embed8":"h_embed9","k0n8":"k0n9","k0ab8":"k0ab9","q5g8v8":"q5g8v9",
       "op38nw32_8":"op38nw32_9","k3aonw32_8":"k3aonw32_9","ao8nw32_8":"ao8nw32_9","hh8":"hh9",
       "ffn8v8":"ffn8v9","down8nw32_8":"down8nw32_9","aq3k8v8":"aq3k8v9","aq6k8v8":"aq6k8v9","head8v8":"head8v9"}
for a, b in REN.items():
  rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__(\1) {b}(", 1, f"rename {b}")
src = src.replace("#define IQ3V8(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7) { " + BS,
                  "#define IQ3V9(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8) { " + BS, 1)
src = src.replace("IQ3V8(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7)",
                  "IQ3V9(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8)", 1)
src = src.replace("IQ3V8(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7)",
                  "IQ3V9(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8)", 1)
src = src.replace("IQ3V8(", "IQ3V9(")
src = src.replace("// engine0 R5d-K7: T=8 PROBE trunk kernels (M=8; k2s8 slots 0..4 + rec6x t=5 + rec7x t=6 + rec8x t=7;",
                  "// engine0 R7a-K8: T=9 PROBE trunk kernels (M=9; k2s9 slots 0..4 + rec6x t=5 + rec7x t=6 + rec8x t=7 + rec9x t=8;", 1)

# ---- AUDITS ----
for bad in ("ACC8H2(", "RED8(", "IQ3V8(", "toks[8]", "for (int t = 0; t < 8"):
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
  if re.search(r"\ba7\b", b) and "float* __restrict__ a4" not in b:
    assert re.search(r"\ba8\b", b), f"{nm} has a7 but no a8"
  if re.search(r"\bag7\b", b): assert re.search(r"\bag8\b", b), f"{nm} ag7 no ag8"
  if re.search(r"\bau7\b", b): assert re.search(r"\bau8\b", b), f"{nm} au7 no au8"
  for arr, s in (("qkv4","10240"),("gate4","6144"),("krow4","1024"),("vrow4","1024"),
                 ("qrow4","12288"),("logits4","VOCAB"),("attn_out4","DIM"),("gact4","FFN_N"),("y4","DIM")):
    if re.search(rf"{arr}\[7\*{s}", b):
      assert re.search(rf"{arr}\[8\*{s}", b), f"{nm} {arr}[7*{s} store without [8*"
assert nk == 14, nk
assert "rec9x" in src and "conv9x" in src
k2s9 = next(b for b in bodies if re.match(r'extern "C" __global__ void __launch_bounds__\(\d+\) k2s9\(', b))
assert "(t == 8) ? (rec8x" in k2s9, "k2s9 t=8 rec_in does not read rec8x"
assert "t < 9" in k2s9, "k2s9 t-loop not extended"
open(f"{BASE}/m9.cu", "w").write(src)
print(f"[gen] m9.cu written ({nk} kernels) — ALL AUDITS PASS")
