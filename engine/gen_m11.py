# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 K=10: generate m11.cu (M=11/T=11 trunk kernels) from m10.cu (the R8 K=9
state). Same laws as gen_m10 (bounds preserved; every write family gains the
row-10 store; a9->a10; ACC11H2/RED11/IQ3V11; k2s11: t=10 rec->rec11x, conv->
conv11x, t=10 rec_in reads rec10x = the t=9 state — the REC-CHAIN SLOT LAW at
depth 11)."""
import re, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/m10.cu").read()
BS = chr(92)

def rep(pat, repl, expect, label):
  global src
  src, cnt = re.subn(pat, repl, src)
  print(f"[gen] {label:56s} {cnt}")
  if cnt != expect:
    print(f"FAIL: {label}: {cnt} != {expect}"); sys.exit(1)

# ---- 1) RED10 -> RED11 ----
rep(r"#define RED10\(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9\) \{ \\",
    lambda m: "#define RED11(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10) { " + BS, 1, "RED11 def rename")
rep(r"A9 \+= __shfl_down_sync\(FULL, A9, o\); \} \}",
    "A9 += __shfl_down_sync(FULL, A9, o); A10 += __shfl_down_sync(FULL, A10, o); } }", 1, "RED11 A10 line")

# ---- 2) ACC10H2 -> ACC11H2 ----
rep(r"#define ACC10H2\(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9\) \{ \\",
    lambda m: "#define ACC11H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) { " + BS, 1, "ACC11H2 def rename")
rep(r"const __half2\* x9 = \(X9\); \\",
    lambda m: "const __half2* x9 = (X9); const __half2* x10 = (X10); " + BS, 1, "ACC11H2 x10 ptr")
x9blk = "\n".join([
    "    { const float2 p = __half22float2(__hmul2(x9[0], w01)); A9 += p.x; A9 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x9[1], w23)); A9 += p.x; A9 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x9[2], w45)); A9 += p.x; A9 += p.y; } " + BS,
    "  { const float2 p = __half22float2(__hmul2(x9[3], w67)); A9 += p.x; A9 += p.y; } }",
])
x10blk = "\n".join(l.replace("x9", "x10").replace("A9", "A10") for l in x9blk.split("\n"))
if x9blk not in src:
  print("FAIL: x9/A9 macro block not found verbatim"); sys.exit(1)
src = src.replace(x9blk, x9blk.replace("} }", "; } " + BS) + "\n" + x10blk, 1)
print("[gen] ACC11H2 x10/A10 block appended          1")

def acc_call(m):
  xs = m.group(1)
  assert xs[:2] in ("xv", "xg"), m.group(0)
  x10 = xs[:2] + "10"
  a10 = m.group(21)[:-1] + "10"
  g = [m.group(i) for i in range(1, 22)]
  return "ACC11H2(" + ", ".join(g[:10] + [x10] + [g[10]] + g[11:] + [a10]) + ")"
pat = r"ACC10H2\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\)"
n = len(re.findall(pat, src))
src = re.sub(pat, acc_call, src)
print(f"[gen] {'ACC11H2 call sites':56s} {n}")
assert n == 13, n

def red_call(m):
  a10 = m.group(10)[:-1] + "10"
  return "RED11(" + ",".join(m.group(i) for i in range(1, 11)) + "," + a10 + ")"
pat = r"RED10\((\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+),(\w+)\)"
n = len(re.findall(pat, src)); src = re.sub(pat, red_call, src)
print(f"[gen] {'RED11 call sites':56s} {n}"); assert n == 15, n

pat = r"LDH2\((\w+)9, (\w+), (\w+), 9, (\w+)\)"
def ldh(m): return m.group(0) + f" LDH2({m.group(1)}10, {m.group(2)}, {m.group(3)}, 10, {m.group(4)})"
n = len(re.findall(pat, src)); src = re.sub(pat, ldh, src)
print(f"[gen] {'LDH2 row-10 appends':56s} {n}"); assert n == 13, n

# ---- decls ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f,a4=0\.f,a5=0\.f,a6=0\.f,a7=0\.f,a8=0\.f,a9=0\.f;",
    "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;", 11, "compact a-decls")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f, a4 = 0\.f, a5 = 0\.f, a6 = 0\.f, a7 = 0\.f, a8 = 0\.f, a9 = 0\.f;",
    "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, a10 = 0.f;", 2, "spaced a-decls")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f,ag4=0\.f,ag5=0\.f,ag6=0\.f,ag7=0\.f,ag8=0\.f,ag9=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f,au4=0\.f,au5=0\.f,au6=0\.f,au7=0\.f,au8=0\.f,au9=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f,ag10=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f,au10=0.f;", 1, "ffn ag/au decl")

# ---- row-10 stores ----
rep(r"(\w+)\[9\*(\w+)\+(\w+)\] = \(__half\)(\w+)9;",
    r"\g<0> \1[10*\2+\3] = (__half)\g<4>10;", 11, "direct half stores row10")
rep(r"y4\[9\*DIM\+warp\] = hh4b\[9\*DIM\+warp\] \+ \(float\)\(\(__half\)a9\);",
    "y4[9*DIM+warp] = hh4b[9*DIM+warp] + (float)((__half)a9); y4[10*DIM+warp] = hh4b[10*DIM+warp] + (float)((__half)a10);", 1, "down residual store row10")
rep(r"gact4\[9\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag9\), \(__half\)au9\);",
    "gact4[9*FFN_N + warp] = __hmul(hsilu_h4((__half)ag9), (__half)au9); gact4[10*FFN_N + warp] = __hmul(hsilu_h4((__half)ag10), (__half)au10);", 1, "ffn gact store row10")
rep(r"\(OB\)\[9\*\(OS\)\+\(RIDX\)\] = \(__half\)a9;",
    "(OB)[9*(OS)+(RIDX)] = (__half)a9; (OB)[10*(OS)+(RIDX)] = (__half)a10;", 1, "aq3 macro store row10")
rep(r"a9 \+= __half2float\(__hmul\(z4\[9\*6144 \+ \(b<<5\)\+lane\], wh\)\);",
    "a9 += __half2float(__hmul(z4[9*6144 + (b<<5)+lane], wh));\n    a10 += __half2float(__hmul(z4[10*6144 + (b<<5)+lane], wh));", 1, "k3ao z accumulate row10")

# ---- h_embed11: toks[11] + s10 arg ----
rep(r"const int toks\[10\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\], s4\[0\], s5\[0\], s6\[0\], s7\[0\], s8\[0\], s9\[0\] \};",
    "const int toks[11] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0], s7[0], s8[0], s9[0], s10[0] };", 1, "toks[11]")
rep(r"const int\* __restrict__ s8, const int\* __restrict__ s9,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s8, const int* __restrict__ s9, const int* __restrict__ s10,\n    float* __restrict__ x4)", 1, "h_embed s10 arg")

# ---- k0ab11: 11-row decode ----
rep(r"const int t = blockIdx\.x % 10;\n  const int g = blockIdx\.x / 10;",
    "const int t = blockIdx.x % 11;\n  const int g = blockIdx.x / 11;", 1, "k0ab11 11-row decode")

# ---- k2s11 surgery ----
rep(r"void __launch_bounds__\(256\) k2s10\(", "void __launch_bounds__(256) k2s11(", 1, "k2s11 rename")
rep(r"for \(int t = 0; t < 10; \+\+t\)", "for (int t = 0; t < 11; ++t)", 1, "k2s11 t<11 loop")

def srep(a, b, label):
  global src
  n = src.count(a)
  print(f"[gen] {label:56s} {n}")
  assert n == 1, f"{label}: {n} != 1"
  src = src.replace(a, b, 1)

srep("float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ conv9x, float* __restrict__ conv10x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x, float* __restrict__ rec9x, float* __restrict__ rec10x,",
     "float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ conv9x, float* __restrict__ conv10x, float* __restrict__ conv11x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x, float* __restrict__ rec9x, float* __restrict__ rec10x, float* __restrict__ rec11x,",
     "k2s11 sig scratch args")
# REC-CHAIN SLOT LAW at depth 11: t=10's rec_in reads rec10x (the t=9 scratch).
srep("                                   : (t == 9) ? (rec9x + (size_t)h*128*128)\n",
     "                                   : (t == 9) ? (rec9x + (size_t)h*128*128)\n" + " "*35 + ": (t == 10) ? (rec10x + (size_t)h*128*128)\n",
     "k2s11 rec_in t10->rec10x")
srep("(((t == 5) ? rec6x : (t == 6) ? rec7x : (t == 7) ? rec8x : (t == 8) ? rec9x : rec10x) + (size_t)h*128*128)",
     "(((t == 5) ? rec6x : (t == 6) ? rec7x : (t == 7) ? rec8x : (t == 8) ? rec9x : (t == 9) ? rec10x : rec11x) + (size_t)h*128*128)",
     "k2s11 rec_out t5..10->scratch")
srep("(t == 8) ? conv9x : (t == 9) ? conv10x : (conv_b + (size_t)t * (3*CONV_CH));",
     "(t == 8) ? conv9x : (t == 9) ? conv10x : (t == 10) ? conv11x : (conv_b + (size_t)t * (3*CONV_CH));",
     "k2s11 conv dst t4..10")

# ---- renames _10 -> _11 (bounds preserved) ----
REN = {"h_embed10":"h_embed11","k0n10":"k0n11","k0ab10":"k0ab11","q5g8v10":"q5g8v11",
       "op38nw32_10":"op38nw32_11","k3aonw32_10":"k3aonw32_11","ao8nw32_10":"ao8nw32_11","hh10":"hh11",
       "ffn8v10":"ffn8v11","down8nw32_10":"down8nw32_11","aq3k8v10":"aq3k8v11","aq6k8v10":"aq6k8v11","head8v10":"head8v11"}
for a, b in REN.items():
  rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__(\1) {b}(", 1, f"rename {b}")
src = src.replace("#define IQ3V10(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { " + BS,
                  "#define IQ3V11(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) { " + BS, 1)
src = src.replace("IQ3V10(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9)",
                  "IQ3V11(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9, ag10)", 1)
src = src.replace("IQ3V10(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9)",
                  "IQ3V11(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9, au10)", 1)
src = src.replace("IQ3V10(", "IQ3V11(")
src = src.replace("// engine0 R8-K9: T=10 PROBE trunk kernels (M=10; k2s10 slots 0..4 + rec6x t=5..rec10x t=9; conv slots 0..3 + conv5x..conv10x; [48][5] layout + live=slot-4 PRESERVED);",
                  "// engine0 R8-K10: T=11 PROBE trunk kernels (M=11; k2s11 slots 0..4 + rec6x t=5..rec11x t=10; conv slots 0..3 + conv5x..conv11x; [48][5] layout + live=slot-4 PRESERVED);", 1)

# ---- zero-spill knobs (scoped per body) ----
_i0 = src.find("void __launch_bounds__(1024) ao8nw32_11(")
assert _i0 > 0, "ao8nw32_11 sig not found"
_i1 = src.find('extern "C" __global__', _i0)
_seg = src[_i0:_i1]
assert _seg.count("  #pragma unroll 2  // R8: 64-reg budget at M=10 (was 4 at M<=9)") == 1, _seg.count("  #pragma unroll 2")
src = src[:_i0] + _seg.replace("  #pragma unroll 2  // R8: 64-reg budget at M=10 (was 4 at M<=9)",
                               "  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged)", 1) + src[_i1:]
print("[gen] ao8nw32_11 unroll 2 held                       1")
# R8 zero-spill knob: op38nw32 at M=11 spills 8B at unroll 4 (64-reg/1024-thr
# budget); unroll 4->2 — per-row fp order unchanged.
_i0 = src.find("void __launch_bounds__(1024) op38nw32_11(")
assert _i0 > 0, "op38nw32_11 sig not found"
_i1 = src.find('extern "C" __global__', _i0)
_seg = src[_i0:_i1]
assert _seg.count("  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds") == 1, _seg.count("  #pragma unroll 4")
src = src[:_i0] + _seg.replace("  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds",
                               "  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged)", 1) + src[_i1:]
print("[gen] op38nw32_11 unroll 4->2 (body-scoped)    1")
# R8 zero-spill knob: down8nw32_11 (the DR7=0 fallback) spills 8B at unroll 4.
_i0 = src.find("void __launch_bounds__(1024) down8nw32_11(")
assert _i0 > 0, "down8nw32_11 sig not found"
_i1 = src.find('extern "C" __global__', _i0)
_seg = src[_i0:_i1]
assert _seg.count("  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds (fp order per row unchanged)") == 1, _seg.count("  #pragma unroll 4")
src = src[:_i0] + _seg.replace("  #pragma unroll 4  // R5: 64-reg budget at 1024-thr bounds (fp order per row unchanged)",
                               "  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged)", 1) + src[_i1:]
print("[gen] down8nw32_11 unroll 4->2 (body-scoped)  1")
_i0 = src.find("void __launch_bounds__(256) ffn8v11(")
assert _i0 > 0, "ffn8v11 sig not found"
_i1 = src.find('extern "C" __global__', _i0)
_seg = src[_i0:_i1]
assert _seg.count("  #pragma unroll 2  // R8: zero-spill at M=10\n") == 1, _seg.count("  #pragma unroll 2  // R8")
src = src[:_i0] + _seg.replace("  #pragma unroll 2  // R8: zero-spill at M=10\n",
                               "  #pragma unroll 2  // R8: zero-spill at M=11\n", 1) + src[_i1:]
print("[gen] ffn8v11 unroll 2 held                          1")

# ---- AUDITS ----
for bad in ("ACC10H2(", "RED10(", "IQ3V10(", "toks[10]", "for (int t = 0; t < 10", "k2s10"):
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
  if re.search(r"\ba9\b", b) and "float* __restrict__ a4" not in b:
    assert re.search(r"\ba10\b", b), f"{nm} has a9 but no a10"
  if re.search(r"\bag9\b", b): assert re.search(r"\bag10\b", b), f"{nm} ag9 no ag10"
  if re.search(r"\bau9\b", b): assert re.search(r"\bau10\b", b), f"{nm} au9 no au10"
  for arr, s in (("qkv4","10240"),("gate4","6144"),("krow4","1024"),("vrow4","1024"),
                 ("qrow4","12288"),("logits4","VOCAB"),("attn_out4","DIM"),("gact4","FFN_N"),("y4","DIM")):
    if re.search(rf"{arr}\[9\*{s}", b):
      assert re.search(rf"{arr}\[10\*{s}", b), f"{nm} {arr}[9*{s} store without [10*"
assert nk == 14, nk
assert "rec11x" in src and "conv11x" in src
k2s11 = next(b for b in bodies if re.match(r'extern "C" __global__ void __launch_bounds__\(\d+\) k2s11\(', b))
assert "(t == 10) ? (rec10x" in k2s11, "k2s11 t=10 rec_in does not read rec10x"
assert "t < 11" in k2s11, "k2s11 t-loop not extended"
# ---- TLX W4.4: k2s11 textual audits (V-59: exactly 2 barriers, no
# continue/return in the t-loop, rec-chain tail) + sibling existence ----
sys.path.insert(0, BASE)
from rung_manifest import audit_k2s
audit_k2s(src, 11)
for sib in ("accept11k", "acceptsel11k", "lookup11_nw32"):
    p2 = f"{BASE}/{sib}.cu"
    assert os.path.exists(p2), f"rung sibling missing: {sib}.cu (V-52 pairing gap)"
acc_sel = open(f"{BASE}/acceptsel11k.cu").read()
assert "m == 10) src = rec11x" in acc_sel, "acceptsel11k lacks the m==10 rec11x arm"
acc11 = open(f"{BASE}/accept11k.cu").read()
assert "emit[13]" in acc11, "accept11k emit stop must land at word 13 (K+3 law)"

open(f"{BASE}/m11.cu", "w").write(src)
print(f"[gen] m11.cu written ({nk} kernels) — ALL AUDITS PASS")

# ---- TLX W4.4: emit the rung manifest (K=10) from this generator's product ----
from gen_manifest import emit_manifest
emit_manifest(10)
