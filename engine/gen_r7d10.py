# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 K=9: append ffn8v10r7/down8nw32v10r7 (M=10 r7-unit GEMVs) to r7d.cu by
M-extending the _9r7 bodies (same laws: R7U uint4 fetch verbatim, per-row fp
order verbatim). Adds RED10/ACC10H2 macro defs."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/r7d.cu").read()

def extract(name):
  for m in re.finditer(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(', src):
    if m.group(2) == name:
      start = m.start()
      nxt = src.find('extern "C" __global__ void', start + 10)
      return src[start:nxt if nxt != -1 else len(src)].rstrip()
  raise AssertionError(name)

def srep1(s, a, b, label):
  n = s.count(a); assert n == 1, f"{label}: {n}"
  print(f"[gen] {label:48s} ok")
  return s.replace(a, b, 1)

# ---- macros: RED10 + ACC10H2 (insert after the _9 defs) ----
# NOTE: the RED9 def is a 2-line \-continued macro — insert AFTER its full body
# (splitting the continuation = preprocessor breakage).
red9_def = "#define RED9(A0,A1,A2,A3,A4,A5,A6,A7,A8) { \\"
assert red9_def in src
red10_def = ("#define RED10(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9) { \\\n"
             "  _Pragma(\"unroll\") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); A7 += __shfl_down_sync(FULL, A7, o); A8 += __shfl_down_sync(FULL, A8, o); A9 += __shfl_down_sync(FULL, A9, o); } }\n")
_i = src.find(red9_def)
_j = src.find("\n", _i + len(red9_def))          # end of RED9 first line
_k = src.find("\n", _j + 1)                       # end of RED9 body line
assert src[_j+1:_k].lstrip().startswith("_Pragma"), src[_j+1:_k]
src = src[:_k+1] + red10_def + src[_k+1:]

acc9_def_head = "#define ACC9H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8) { \\"
assert acc9_def_head in src
# ACC10H2 = the ACC9H2 text with an X9 ptr decl + an x9/A9 block appended.
i0 = src.find(acc9_def_head)
i1 = src.find("\n\n", i0)
acc9 = src[i0:i1]
acc10 = acc9.replace("ACC9H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8)",
                     "ACC10H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)", 1)
acc10 = acc10.replace("const __half2* x8 = (X8); \\",
                      "const __half2* x8 = (X8); const __half2* x9 = (X9); \\", 1)
x8last = "  { const float2 p = __half22float2(__hmul2(x8[3], w67)); A8 += p.x; A8 += p.y; } }"
assert x8last in acc10, "x8 last line not found in ACC9H2"
x9blk = ("  { const float2 p = __half22float2(__hmul2(x9[0], w01)); A9 += p.x; A9 += p.y; } \\\n"
         "  { const float2 p = __half22float2(__hmul2(x9[1], w23)); A9 += p.x; A9 += p.y; } \\\n"
         "  { const float2 p = __half22float2(__hmul2(x9[2], w45)); A9 += p.x; A9 += p.y; } \\\n"
         "  { const float2 p = __half22float2(__hmul2(x9[3], w67)); A9 += p.x; A9 += p.y; } }")
acc10 = acc10.replace(x8last, x8last[:-3] + "; } \\\n" + x9blk, 1)
src = src[:i1] + "\n\n" + acc10 + src[i1:]
print("[gen] RED10 + ACC10H2 macros added")

# ---- ffn8v10r7 ----
k = extract("ffn8v9r7")
k = srep1(k, "void __launch_bounds__(256) ffn8v9r7(", "void __launch_bounds__(256) ffn8v10r7(", "ffn8v10r7 rename")
k = srep1(k, "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f;\n  float au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f;",
          "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f;\n  float au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f;", "ffn decls a9")
k = srep1(k, "LDH2(xg8, hhx4, DIM, 8, koff)", "LDH2(xg8, hhx4, DIM, 8, koff) LDH2(xg9, hhx4, DIM, 9, koff)", "ffn xg9")
k = srep1(k, "#define R7V9(W, A0, A1, A2, A3, A4, A5, A6, A7, A8) { \\", "#define R7V10(W, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { \\", "R7V10 macro def")
k = srep1(k, "ACC9H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8) }",
          "ACC10H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, xg9, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) }", "ffn ACC10H2 call")
k = srep1(k, "R7V9(wg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8)", "R7V10(wg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9)", "ffn R7V10 wg")
k = srep1(k, "R7V9(wu, au0, au1, au2, au3, au4, au5, au6, au7, au8)", "R7V10(wu, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9)", "ffn R7V10 wu")
k = srep1(k, "#undef R7V9", "#undef R7V10", "ffn undef")
k = srep1(k, "RED9(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8)", "RED10(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8,ag9)", "ffn RED10 g")
k = srep1(k, "RED9(au0,au1,au2,au3,au4,au5,au6,au7,au8)", "RED10(au0,au1,au2,au3,au4,au5,au6,au7,au8,au9)", "ffn RED10 u")
k = srep1(k, "gact4[8*FFN_N + warp] = __hmul(hsilu_hr((__half)ag8), (__half)au8);",
          "gact4[8*FFN_N + warp] = __hmul(hsilu_hr((__half)ag8), (__half)au8);\n    gact4[9*FFN_N + warp] = __hmul(hsilu_hr((__half)ag9), (__half)au9);", "ffn gact row9")
ffn10 = "// ---- ffn8v10r7: gate+up IQ3 r7 GEMVs + silu-mul, M=10 (R8 K=9, port of ffn8v10) ----\n" + k + "\n\n"

# ---- down8nw32v10r7 ----
k = extract("down8nw32v9r7")
k = srep1(k, "void __launch_bounds__(1024) down8nw32v9r7(", "void __launch_bounds__(1024) down8nw32v10r7(", "down8nw32v10r7 rename")
k = srep1(k, "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f;",
          "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f;", "down decl a9")
k = srep1(k, "LDH2(xv8, gact4, FFN_N, 8, koff)", "LDH2(xv8, gact4, FFN_N, 8, koff) LDH2(xv9, gact4, FFN_N, 9, koff)", "down xv9")
k = srep1(k, "ACC9H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8)",
          "ACC10H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)", "down ACC10H2 call")
k = srep1(k, "RED9(a0,a1,a2,a3,a4,a5,a6,a7,a8)", "RED10(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9)", "down RED10")
k = srep1(k, "y4[8*DIM+warp] = hh4b[8*DIM+warp] + (float)((__half)a8);",
          "y4[8*DIM+warp] = hh4b[8*DIM+warp] + (float)((__half)a8);\n    y4[9*DIM+warp] = hh4b[9*DIM+warp] + (float)((__half)a9);", "down y4 row9")
# R8 zero-spill knob: down8nw32 at M=10 spills 8B at unroll 4 (64-reg/1024-thr
# budget); unroll 4->2 — per-row fp order unchanged.
k = srep1(k, "  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds (fp order per row unchanged)",
          "  #pragma unroll 2  // R8: 64-reg budget at M=10 (fp order per row unchanged)", "down unroll 4->2")
down10 = "// ---- down8nw32v10r7: down GEMV IQ3 r7 + residual, M=10 fat-CTA (R8 K=9, port of down8nw32_10) ----\n" + k + "\n"

# audits
for bad in ("ACC9H2(xg", "ACC9H2(xv", "RED9(ag", "RED9(a0"):
  assert bad not in ffn10 + down10, bad
assert ffn10.count("ag9") >= 3 and ffn10.count("au9") >= 3 and "xg9" in ffn10
assert down10.count("a9") >= 3 and "xv9" in down10
assert "ACC10H2" in ffn10 and "ACC10H2" in down10 and "RED10" in ffn10 and "RED10" in down10

src = src.rstrip() + "\n\n" + ffn10 + down10
open(f"{BASE}/r7d.cu", "w").write(src)
print("[gen] r7d.cu extended with ffn8v10r7 + down8nw32v10r7 — AUDITS PASS")
