# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a K=8: append ffn8v9r7/down8nw32v9r7 (M=9 r7-unit GEMVs) to r7d.cu by
M-extending the _8r7 bodies (same laws: R7U uint4 fetch verbatim, per-row fp
order verbatim). Adds RED9/ACC9H2 macro defs."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/r7d.cu").read()

def extract(name):
  i = src.find(f'extern "C" __global__ void __launch_bounds__(')
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

# ---- macros: RED9 + ACC9H2 (insert after the _8 defs) ----
red8_def = "#define RED8(A0,A1,A2,A3,A4,A5,A6,A7) { \\"
assert red8_def in src
red9_def = ("#define RED9(A0,A1,A2,A3,A4,A5,A6,A7,A8) { \\\n"
            "  _Pragma(\"unroll\") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); A7 += __shfl_down_sync(FULL, A7, o); A8 += __shfl_down_sync(FULL, A8, o); } }\n")
src = src.replace(red8_def, red8_def + "\n" + red9_def, 1)

acc8_def_head = "#define ACC8H2(X0, X1, X2, X3, X4, X5, X6, X7, WV, A0, A1, A2, A3, A4, A5, A6, A7) { \\"
assert acc8_def_head in src
# ACC9H2 = the ACC8H2 text with an X8 ptr decl + an x8/A8 block appended.
i0 = src.find(acc8_def_head)
i1 = src.find("\n\n", i0)
acc8 = src[i0:i1]
acc9 = acc8.replace("ACC8H2(X0, X1, X2, X3, X4, X5, X6, X7, WV, A0, A1, A2, A3, A4, A5, A6, A7)",
                    "ACC9H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8)", 1)
acc9 = acc9.replace("const __half2* x7 = (X7); \\",
                    "const __half2* x7 = (X7); const __half2* x8 = (X8); \\", 1)
x7last = "  { const float2 p = __half22float2(__hmul2(x7[3], w67)); A7 += p.x; A7 += p.y; } }"
assert x7last in acc9, "x7 last line not found in ACC8H2"
x8blk = ("  { const float2 p = __half22float2(__hmul2(x8[0], w01)); A8 += p.x; A8 += p.y; } \\\n"
         "  { const float2 p = __half22float2(__hmul2(x8[1], w23)); A8 += p.x; A8 += p.y; } \\\n"
         "  { const float2 p = __half22float2(__hmul2(x8[2], w45)); A8 += p.x; A8 += p.y; } \\\n"
         "  { const float2 p = __half22float2(__hmul2(x8[3], w67)); A8 += p.x; A8 += p.y; } }")
acc9 = acc9.replace(x7last, x7last[:-3] + "; } \\\n" + x8blk, 1)
src = src[:i1] + "\n\n" + acc9 + src[i1:]
print("[gen] RED9 + ACC9H2 macros added")

# ---- ffn8v9r7 ----
k = extract("ffn8v8r7")
k = srep1(k, "void __launch_bounds__(256) ffn8v8r7(", "void __launch_bounds__(256) ffn8v9r7(", "ffn8v9r7 rename")
k = srep1(k, "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f;\n  float au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f;",
          "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f;\n  float au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f;", "ffn decls a8")
k = srep1(k, "LDH2(xg7, hhx4, DIM, 7, koff)", "LDH2(xg7, hhx4, DIM, 7, koff) LDH2(xg8, hhx4, DIM, 8, koff)", "ffn xg8")
k = srep1(k, "#define R7V8(W, A0, A1, A2, A3, A4, A5, A6, A7) { \\", "#define R7V9(W, A0, A1, A2, A3, A4, A5, A6, A7, A8) { \\", "R7V9 macro def")
k = srep1(k, "ACC8H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, wv, A0, A1, A2, A3, A4, A5, A6, A7) }",
          "ACC9H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8) }", "ffn ACC9H2 call")
k = srep1(k, "R7V8(wg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7)", "R7V9(wg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8)", "ffn R7V9 wg")
k = srep1(k, "R7V8(wu, au0, au1, au2, au3, au4, au5, au6, au7)", "R7V9(wu, au0, au1, au2, au3, au4, au5, au6, au7, au8)", "ffn R7V9 wu")
k = srep1(k, "#undef R7V8", "#undef R7V9", "ffn undef")
k = srep1(k, "RED8(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7)", "RED9(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8)", "ffn RED9 g")
k = srep1(k, "RED8(au0,au1,au2,au3,au4,au5,au6,au7)", "RED9(au0,au1,au2,au3,au4,au5,au6,au7,au8)", "ffn RED9 u")
k = srep1(k, "gact4[7*FFN_N + warp] = __hmul(hsilu_hr((__half)ag7), (__half)au7);",
          "gact4[7*FFN_N + warp] = __hmul(hsilu_hr((__half)ag7), (__half)au7);\n    gact4[8*FFN_N + warp] = __hmul(hsilu_hr((__half)ag8), (__half)au8);", "ffn gact row8")
ffn9 = "// ---- ffn8v9r7: gate+up IQ3 r7 GEMVs + silu-mul, M=9 (R7a K=8, port of ffn8v9) ----\n" + k + "\n\n"

# ---- down8nw32v9r7 ----
k = extract("down8nw32v8r7")
k = srep1(k, "void __launch_bounds__(1024) down8nw32v8r7(", "void __launch_bounds__(1024) down8nw32v9r7(", "down8nw32v9r7 rename")
k = srep1(k, "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;",
          "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f;", "down decl a8")
k = srep1(k, "LDH2(xv7, gact4, FFN_N, 7, koff)", "LDH2(xv7, gact4, FFN_N, 7, koff) LDH2(xv8, gact4, FFN_N, 8, koff)", "down xv8")
k = srep1(k, "ACC8H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, wv, a0, a1, a2, a3, a4, a5, a6, a7)",
          "ACC9H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8)", "down ACC9H2 call")
k = srep1(k, "RED8(a0,a1,a2,a3,a4,a5,a6,a7)", "RED9(a0,a1,a2,a3,a4,a5,a6,a7,a8)", "down RED9")
k = srep1(k, "y4[7*DIM+warp] = hh4b[7*DIM+warp] + (float)((__half)a7);",
          "y4[7*DIM+warp] = hh4b[7*DIM+warp] + (float)((__half)a7);\n    y4[8*DIM+warp] = hh4b[8*DIM+warp] + (float)((__half)a8);", "down y4 row8")
down9 = "// ---- down8nw32v9r7: down GEMV IQ3 r7 + residual, M=9 fat-CTA (R7a K=8, port of down8nw32_9) ----\n" + k + "\n"

# audits
for bad in ("ACC8H2(xg", "ACC8H2(xv", "RED8(ag", "RED8(a0"):
  assert bad not in ffn9 + down9, bad
assert ffn9.count("ag8") >= 3 and ffn9.count("au8") >= 3 and "xg8" in ffn9
assert down9.count("a8") >= 3 and "xv8" in down9
assert "ACC9H2" in ffn9 and "ACC9H2" in down9 and "RED9" in ffn9 and "RED9" in down9

src = src.rstrip() + "\n\n" + ffn9 + down9
open(f"{BASE}/r7d.cu", "w").write(src)
print("[gen] r7d.cu extended with ffn8v9r7 + down8nw32v9r7 — AUDITS PASS")
