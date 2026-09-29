# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P10-dense rung 2: append ffn8v5r7/down8nw32v5r7 (M=5 r7-unit GEMVs — the
K=4 EAGLE T=5 probe's FFN/down pair under PF_DR7) to r7d.cu by M-extending the
_v3r7 bodies (same laws: R7U uint4 fetch verbatim, per-row fp order verbatim,
ACC/RED macros extended row-wise — the gen_r7d9 pattern applied v3->v5).
Adds RED5/ACC5H2 macro defs AFTER the full \\-continued existing defs (the
MACRO-DEF INSERTION law)."""
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

if "ffn8v5r7" in src:
  print("[gen] ffn8v5r7 already present — nothing to do"); sys.exit(0)

# ---- macro: RED5 (insert after the complete RED3 def) ----
red3_def = ('#define RED3(A0,A1,A2) { \\\n'
            '  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); } }\n')
assert red3_def in src, "RED3 def anchor not found"
red5_def = ('#define RED5(A0,A1,A2,A3,A4) { \\\n'
            '  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); } }\n')
src = src.replace(red3_def, red3_def + red5_def, 1)

# ---- macro: ACC5H2 (built from the full ACC3H2 def, gen_r7d9's ACC9H2 pattern) ----
acc3_head = "#define ACC3H2(X0, X1, X2, WV, A0, A1, A2) { \\"
assert acc3_head in src
i0 = src.find(acc3_head)
i1 = src.find("\n\n", i0)
acc3 = src[i0:i1]
acc5 = acc3.replace("ACC3H2(X0, X1, X2, WV, A0, A1, A2)",
                    "ACC5H2(X0, X1, X2, X3, X4, WV, A0, A1, A2, A3, A4)", 1)
acc5 = acc5.replace("const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); \\",
                    "const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); \\", 1)
x2last = "  { const float2 p = __half22float2(__hmul2(x2[3], w67)); A2 += p.x; A2 += p.y; } }"
assert x2last in acc5, "x2 last line not found in ACC3H2"
def xrow(xv, av):
  return ("  { const float2 p = __half22float2(__hmul2(%s[0], w01)); %s += p.x; %s += p.y; } \\\n"
          "  { const float2 p = __half22float2(__hmul2(%s[1], w23)); %s += p.x; %s += p.y; } \\\n"
          "  { const float2 p = __half22float2(__hmul2(%s[2], w45)); %s += p.x; %s += p.y; } \\\n"
          "  { const float2 p = __half22float2(__hmul2(%s[3], w67)); %s += p.x; %s += p.y; } }") % (
            xv, av, av, xv, av, av, xv, av, av, xv, av, av)
x34blk = xrow("x3", "A3")[:-3] + "; } \\\n" + xrow("x4", "A4")
acc5 = acc5.replace(x2last, x2last[:-3] + "; } \\\n" + x34blk, 1)
src = src[:i1] + "\n\n" + acc5 + src[i1:]
print("[gen] RED5 + ACC5H2 macros added")

# ---- ffn8v5r7 ----
k = extract("ffn8v3r7")
k = srep1(k, "void __launch_bounds__(256) ffn8v3r7(", "void __launch_bounds__(256) ffn8v5r7(", "ffn8v5r7 rename")
k = srep1(k, "float ag0=0.f,ag1=0.f,ag2=0.f, au0=0.f,au1=0.f,au2=0.f;",
          "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f;", "ffn decls a3/a4")
k = srep1(k, "LDH2(xg0, hhx3, DIM, 0, koff) LDH2(xg1, hhx3, DIM, 1, koff) LDH2(xg2, hhx3, DIM, 2, koff)",
          "LDH2(xg0, hhx3, DIM, 0, koff) LDH2(xg1, hhx3, DIM, 1, koff) LDH2(xg2, hhx3, DIM, 2, koff) "
          "LDH2(xg3, hhx3, DIM, 3, koff) LDH2(xg4, hhx3, DIM, 4, koff)", "ffn xg3/xg4")
k = srep1(k, "#define R7V3(W, A0, A1, A2) { \\", "#define R7V5(W, A0, A1, A2, A3, A4) { \\", "R7V5 macro def")
k = srep1(k, "ACC3H2(xg0, xg1, xg2, wv, A0, A1, A2) }",
          "ACC5H2(xg0, xg1, xg2, xg3, xg4, wv, A0, A1, A2, A3, A4) }", "ffn ACC5H2 call")
k = srep1(k, "R7V3(wg, ag0, ag1, ag2)", "R7V5(wg, ag0, ag1, ag2, ag3, ag4)", "ffn R7V5 wg")
k = srep1(k, "R7V3(wu, au0, au1, au2)", "R7V5(wu, au0, au1, au2, au3, au4)", "ffn R7V5 wu")
k = srep1(k, "#undef R7V3", "#undef R7V5", "ffn undef")
k = srep1(k, "RED3(ag0,ag1,ag2)", "RED5(ag0,ag1,ag2,ag3,ag4)", "ffn RED5 g")
k = srep1(k, "RED3(au0,au1,au2)", "RED5(au0,au1,au2,au3,au4)", "ffn RED5 u")
k = srep1(k, "gact3[2*FFN_N + warp] = __hmul(hsilu_hr((__half)ag2), (__half)au2);",
          "gact3[2*FFN_N + warp] = __hmul(hsilu_hr((__half)ag2), (__half)au2);\n"
          "    gact3[3*FFN_N + warp] = __hmul(hsilu_hr((__half)ag3), (__half)au3);\n"
          "    gact3[4*FFN_N + warp] = __hmul(hsilu_hr((__half)ag4), (__half)au4);", "ffn gact rows 3/4")
ffn5 = "// ---- ffn8v5r7: gate+up IQ3 r7 GEMVs + silu-mul, M=5 (P10-dense K=4 EAGLE, port of ffn8v5) ----\n" + k + "\n\n"

# ---- down8nw32v5r7 ----
k = extract("down8nw32v3r7")
k = srep1(k, "void __launch_bounds__(1024) down8nw32v3r7(", "void __launch_bounds__(1024) down8nw32v5r7(", "down8nw32v5r7 rename")
k = srep1(k, "float a0=0.f, a1=0.f, a2=0.f;",
          "float a0=0.f, a1=0.f, a2=0.f, a3=0.f, a4=0.f;", "down decl a3/a4")
k = srep1(k, "LDH2(xv0, gact3, FFN_N, 0, koff) LDH2(xv1, gact3, FFN_N, 1, koff) LDH2(xv2, gact3, FFN_N, 2, koff)",
          "LDH2(xv0, gact3, FFN_N, 0, koff) LDH2(xv1, gact3, FFN_N, 1, koff) LDH2(xv2, gact3, FFN_N, 2, koff) "
          "LDH2(xv3, gact3, FFN_N, 3, koff) LDH2(xv4, gact3, FFN_N, 4, koff)", "down xv3/xv4")
k = srep1(k, "ACC3H2(xv0, xv1, xv2, wv, a0, a1, a2)",
          "ACC5H2(xv0, xv1, xv2, xv3, xv4, wv, a0, a1, a2, a3, a4)", "down ACC5H2 call")
k = srep1(k, "RED3(a0,a1,a2)", "RED5(a0,a1,a2,a3,a4)", "down RED5")
k = srep1(k, "y3[2*DIM+warp] = hh3[2*DIM+warp] + (float)((__half)a2);",
          "y3[2*DIM+warp] = hh3[2*DIM+warp] + (float)((__half)a2);\n"
          "    y3[3*DIM+warp] = hh3[3*DIM+warp] + (float)((__half)a3);\n"
          "    y3[4*DIM+warp] = hh3[4*DIM+warp] + (float)((__half)a4);", "down y3 rows 3/4")
down5 = "// ---- down8nw32v5r7: down GEMV IQ3 r7 + residual, M=5 fat-CTA (P10-dense K=4 EAGLE, port of down8nw32_5) ----\n" + k + "\n"

# audits
for bad in ("ACC3H2(xg", "ACC3H2(xv", "RED3(ag", "RED3(a0", "R7V3("):
  assert bad not in ffn5 + down5, bad
assert ffn5.count("ag3") >= 3 and ffn5.count("ag4") >= 3 and ffn5.count("au3") >= 3 and ffn5.count("au4") >= 3
assert "xg3" in ffn5 and "xg4" in ffn5 and "ACC5H2" in ffn5 and "RED5" in ffn5
assert down5.count("a3") >= 3 and down5.count("a4") >= 3 and "xv3" in down5 and "xv4" in down5
assert "ACC5H2" in down5 and "RED5" in down5

src = src.rstrip() + "\n\n" + ffn5 + down5
open(f"{BASE}/r7d.cu", "w").write(src)
print("[gen] r7d.cu extended with ffn8v5r7 + down8nw32v5r7 — AUDITS PASS")
