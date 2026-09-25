# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 K=10: append ffn8v11r7/down8nw32v11r7 (M=11 r7-unit GEMVs) to r7d.cu by
M-extending the _10r7 bodies. Adds RED11/ACC11H2 macro defs (full-def insert —
never split a \-continuation)."""
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

# ---- macros: RED11 + ACC11H2 ----
red10_def = "#define RED10(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9) { \\"
assert red10_def in src
red11_def = ("#define RED11(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10) { \\\n"
             "  _Pragma(\"unroll\") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); A7 += __shfl_down_sync(FULL, A7, o); A8 += __shfl_down_sync(FULL, A8, o); A9 += __shfl_down_sync(FULL, A9, o); A10 += __shfl_down_sync(FULL, A10, o); } }\n")
_i = src.find(red10_def)
_j = src.find("\n", _i + len(red10_def))
_k = src.find("\n", _j + 1)
assert src[_j+1:_k].lstrip().startswith("_Pragma"), src[_j+1:_k]
src = src[:_k+1] + red11_def + src[_k+1:]

acc10_def_head = "#define ACC10H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { \\"
assert acc10_def_head in src
i0 = src.find(acc10_def_head)
i1 = src.find("\n\n", i0)
acc10 = src[i0:i1]
acc11 = acc10.replace("ACC10H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)",
                      "ACC11H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)", 1)
acc11 = acc11.replace("const __half2* x9 = (X9); \\",
                      "const __half2* x9 = (X9); const __half2* x10 = (X10); \\", 1)
x9last = "  { const float2 p = __half22float2(__hmul2(x9[3], w67)); A9 += p.x; A9 += p.y; } }"
assert x9last in acc11, "x9 last line not found in ACC10H2"
x10blk = ("  { const float2 p = __half22float2(__hmul2(x10[0], w01)); A10 += p.x; A10 += p.y; } \\\n"
          "  { const float2 p = __half22float2(__hmul2(x10[1], w23)); A10 += p.x; A10 += p.y; } \\\n"
          "  { const float2 p = __half22float2(__hmul2(x10[2], w45)); A10 += p.x; A10 += p.y; } \\\n"
          "  { const float2 p = __half22float2(__hmul2(x10[3], w67)); A10 += p.x; A10 += p.y; } }")
acc11 = acc11.replace(x9last, x9last[:-3] + "; } \\\n" + x10blk, 1)
src = src[:i1] + "\n\n" + acc11 + src[i1:]
print("[gen] RED11 + ACC11H2 macros added")

# ---- ffn8v11r7 ----
k = extract("ffn8v10r7")
k = srep1(k, "void __launch_bounds__(256) ffn8v10r7(", "void __launch_bounds__(256) ffn8v11r7(", "ffn8v11r7 rename")
k = srep1(k, "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f;\n  float au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f;",
          "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f,ag10=0.f;\n  float au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f,au10=0.f;", "ffn decls a10")
k = srep1(k, "LDH2(xg9, hhx4, DIM, 9, koff)", "LDH2(xg9, hhx4, DIM, 9, koff) LDH2(xg10, hhx4, DIM, 10, koff)", "ffn xg10")
k = srep1(k, "#define R7V10(W, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { \\", "#define R7V11(W, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) { \\", "R7V11 macro def")
k = srep1(k, "ACC10H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, xg9, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) }",
          "ACC11H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, xg9, xg10, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) }", "ffn ACC11H2 call")
k = srep1(k, "R7V10(wg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9)", "R7V11(wg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9, ag10)", "ffn R7V11 wg")
k = srep1(k, "R7V10(wu, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9)", "R7V11(wu, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9, au10)", "ffn R7V11 wu")
k = srep1(k, "#undef R7V10", "#undef R7V11", "ffn undef")
k = srep1(k, "RED10(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8,ag9)", "RED11(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8,ag9,ag10)", "ffn RED11 g")
k = srep1(k, "RED10(au0,au1,au2,au3,au4,au5,au6,au7,au8,au9)", "RED11(au0,au1,au2,au3,au4,au5,au6,au7,au8,au9,au10)", "ffn RED11 u")
k = srep1(k, "gact4[9*FFN_N + warp] = __hmul(hsilu_hr((__half)ag9), (__half)au9);",
          "gact4[9*FFN_N + warp] = __hmul(hsilu_hr((__half)ag9), (__half)au9);\n    gact4[10*FFN_N + warp] = __hmul(hsilu_hr((__half)ag10), (__half)au10);", "ffn gact row10")
ffn11 = "// ---- ffn8v11r7: gate+up IQ3 r7 GEMVs + silu-mul, M=11 (R8 K=10, port of ffn8v11) ----\n" + k + "\n\n"

# ---- down8nw32v11r7 ----
k = extract("down8nw32v10r7")
k = srep1(k, "void __launch_bounds__(1024) down8nw32v10r7(", "void __launch_bounds__(1024) down8nw32v11r7(", "down8nw32v11r7 rename")
k = srep1(k, "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f;",
          "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;", "down decl a10")
k = srep1(k, "LDH2(xv9, gact4, FFN_N, 9, koff)", "LDH2(xv9, gact4, FFN_N, 9, koff) LDH2(xv10, gact4, FFN_N, 10, koff)", "down xv10")
k = srep1(k, "ACC10H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)",
          "ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)", "down ACC11H2 call")
k = srep1(k, "RED10(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9)", "RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)", "down RED11")
k = srep1(k, "y4[9*DIM+warp] = hh4b[9*DIM+warp] + (float)((__half)a9);",
          "y4[9*DIM+warp] = hh4b[9*DIM+warp] + (float)((__half)a9);\n    y4[10*DIM+warp] = hh4b[10*DIM+warp] + (float)((__half)a10);", "down y4 row10")
k = srep1(k, "  #pragma unroll 2  // R8: 64-reg budget at M=10 (fp order per row unchanged)",
          "  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged)", "down unroll 2 held")
down11 = "// ---- down8nw32v11r7: down GEMV IQ3 r7 + residual, M=11 fat-CTA (R8 K=10, port of down8nw32_11) ----\n" + k + "\n"

# audits
for bad in ("ACC10H2(xg", "ACC10H2(xv", "RED10(ag", "RED10(a0"):
  assert bad not in ffn11 + down11, bad
assert ffn11.count("ag10") >= 3 and ffn11.count("au10") >= 3 and "xg10" in ffn11
assert down11.count("a10") >= 3 and "xv10" in down11
assert "ACC11H2" in ffn11 and "ACC11H2" in down11 and "RED11" in ffn11 and "RED11" in down11

src = src.rstrip() + "\n\n" + ffn11 + down11
open(f"{BASE}/r7d.cu", "w").write(src)
print("[gen] r7d.cu extended with ffn8v11r7 + down8nw32v11r7 — AUDITS PASS")
