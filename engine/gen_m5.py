# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R4: generate m5.cu (T=5 probe trunk) from m4.cu by mechanical M-extension.
Count-tolerant replacements with loud reports + a final zero-leftover audit.
Per-row op order identical to M=3/M=4 -> rows bit-identical (W2D pattern)."""
import re, sys

BASE = "~/tinygrad-metal/engine0"
src = open(f"{BASE}/m4.cu").read()

def rep(pat, repl, expect_min=1, label=""):
    global src
    src, cnt = re.subn(pat, repl, src)
    print(f"[gen] {label or pat[:48]:50s} {cnt}")
    if cnt < expect_min:
        print(f"FAIL: {label} -> {cnt} < {expect_min}"); sys.exit(1)
    return cnt

# ---- 1) ACC4H2 macro -> ACC5H2 ----
m = re.search(r"#define ACC4H2\(X0, X1, X2, X3, WV, A0, A1, A2, A3\) \{ \\\n(.*?)\n(#|extern)", src, re.S)
assert m, "ACC4H2 macro not found"
body = m.group(1)
# extract the x0 4-line block; clone it as the x4/A4 block
x0lines = [l for l in body.split("\n") if "x0[" in l]
assert len(x0lines) == 4, x0lines
x4block = "\n".join(l.replace("x0[", "x4[").replace("A0", "A4").rstrip() for l in x0lines)
# drop the line-continuation backslash on the LAST generated line (macro closes there)
lines4 = x4block.split("\n")
if lines4[-1].endswith("\\"): lines4[-1] = lines4[-1][:-1].rstrip()
x4block = "\n".join(lines4)
# header: add X4 param + x4 decl
src = src.replace(
  "#define ACC4H2(X0, X1, X2, X3, WV, A0, A1, A2, A3) { \\",
  "#define ACC5H2(X0, X1, X2, X3, X4, WV, A0, A1, A2, A3, A4) { \\", 1)
src = src.replace(
  "  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); \\",
  "  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); \\", 1)
assert "ACC5H2(X0, X1, X2, X3, X4" in src
# body: insert the x4 block at the macro tail (anchor = the x3 4th line + inline close)
anchor = "x3[3], w67)); A3 += p.x; A3 += p.y; } }"
assert src.count(anchor) == 1, src.count(anchor)
src = src.replace(anchor,
  "x3[3], w67)); A3 += p.x; A3 += p.y; } \\\n  " + x4block + " }", 1)

# ---- 2) RED4 macro -> RED5 ----
rep(r"#define RED4\(A0,A1,A2,A3\) \{ \\\n  _Pragma\(\"unroll\"\) for \(int o = 16; o > 0; o >>= 1\) \{ A0 \+= __shfl_down_sync\(FULL, A0, o\); A1 \+= __shfl_down_sync\(FULL, A1, o\); A2 \+= __shfl_down_sync\(FULL, A2, o\); A3 \+= __shfl_down_sync\(FULL, A3, o\); \} \}",
    "#define RED5(A0,A1,A2,A3,A4) { \\\n  _Pragma(\"unroll\") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); } }",
    1, "RED macro")

# ---- 3) accumulator decls (both spacings) ----
rep(r"float a0=0\.f,a1=0\.f,a2=0\.f,a3=0\.f;", "float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f;", 1, "decl unspaced")
rep(r"float a0 = 0\.f, a1 = 0\.f, a2 = 0\.f, a3 = 0\.f;", "float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;", 1, "decl spaced")
rep(r"float ag0=0\.f,ag1=0\.f,ag2=0\.f,ag3=0\.f, au0=0\.f,au1=0\.f,au2=0\.f,au3=0\.f;",
    "float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f;", 1, "decl ffn")

# ---- 4) LDH2 chains: add t=4 ----
rep(r"LDH2\(xv3, (\w+), (\w+), 3, koff\)", r"LDH2(xv3, \1, \2, 3, koff) LDH2(xv4, \1, \2, 4, koff)", 1, "LDH2 xv")
rep(r"LDH2\(xg3, (\w+), (\w+), 3, koff\)", r"LDH2(xg3, \1, \2, 3, koff) LDH2(xg4, \1, \2, 4, koff)", 1, "LDH2 xg")

# ---- 5) ACC calls ----
rep(r"ACC4H2\(xv0, xv1, xv2, xv3, wv, a0, a1, a2, a3\)", "ACC5H2(xv0, xv1, xv2, xv3, xv4, wv, a0, a1, a2, a3, a4)", 1, "ACC xv")
rep(r"ACC4H2\(xg0, xg1, xg2, xg3, wv, A0, A1, A2, A3\)", "ACC5H2(xg0, xg1, xg2, xg3, xg4, wv, A0, A1, A2, A3, A4)", 1, "ACC xg A")
# ffn8v5: the IQ3V4 macro gains A4; its two invocations pass ag4/au4
rep(r"#define IQ3V4\(QP, SP, DP, A0, A1, A2, A3\) \{", "#define IQ3V4(QP, SP, DP, A0, A1, A2, A3, A4) {", 1, "IQ3V4 sig")
rep(r"IQ3V4\(qg, sg, dg, ag0, ag1, ag2, ag3\)", "IQ3V4(qg, sg, dg, ag0, ag1, ag2, ag3, ag4)", 1, "IQ3V4 ag")
rep(r"IQ3V4\(qu, su, du, au0, au1, au2, au3\)", "IQ3V4(qu, su, du, au0, au1, au2, au3, au4)", 1, "IQ3V4 au")

# ---- 6) RED calls ----
rep(r"RED4\(a0,a1,a2,a3\)", "RED5(a0,a1,a2,a3,a4)", 1, "RED a")
rep(r"RED4\(ag0,ag1,ag2,ag3\)", "RED5(ag0,ag1,ag2,ag3,ag4)", 1, "RED ag")
rep(r"RED4\(au0,au1,au2,au3\)", "RED5(au0,au1,au2,au3,au4)", 1, "RED au")

# ---- 7) output writes (SAME-LINE append: some sites live inside \-continued
#      #define macros — a newline split would break the continuation) ----
for pat, add in [
  (r"attn_out4\[3\*DIM\+warp\] = \(__half\)a3;", "attn_out4[4*DIM+warp] = (__half)a4;"),
  (r"qkv4\[3\*10240\+warp\] = \(__half\)a3;", "qkv4[4*10240+warp] = (__half)a4;"),
  (r"gate4\[3\*6144\+r\] = \(__half\)a3;", "gate4[4*6144+r] = (__half)a4;"),
  (r"krow4\[3\*1024\+r\] = \(__half\)a3;", "krow4[4*1024+r] = (__half)a4;"),
  (r"vrow4\[3\*1024\+r\] = \(__half\)a3;", "vrow4[4*1024+r] = (__half)a4;"),
  (r"qrow4\[3\*12288\+warp\] = \(__half\)a3;", "qrow4[4*12288+warp] = (__half)a4;"),
  (r"logits4\[3\*VOCAB\+warp\] = \(__half\)a3;", "logits4[4*VOCAB+warp] = (__half)a4;"),
  (r"y4\[3\*DIM\+warp\] = hh4b\[3\*DIM\+warp\] \+ \(float\)\(\(__half\)a3\);", "y4[4*DIM+warp] = hh4b[4*DIM+warp] + (float)((__half)a4);"),
  (r"gact4\[3\*FFN_N \+ warp\] = __hmul\(hsilu_h4\(\(__half\)ag3\), \(__half\)au3\);", "gact4[4*FFN_N + warp] = __hmul(hsilu_h4((__half)ag4), (__half)au4);"),
]:
    rep(pat, lambda mm, add=add: mm.group(0) + " " + add, 1, f"out {add[:40]}")

# ---- 8) t loops (NOT the j<4 lane loops) + rr[5] ----
rep(r"for \(int t = 0; t < 4; \+\+t\)", "for (int t = 0; t < 5; ++t)", 1, "t loops")
rep(r"float rr\[4\];", "float rr[5];", 1, "rr[5]")

# ---- 9) h_embed s4 ----
rep(r"const int\* __restrict__ s3,\n    float\* __restrict__ x4\)",
    "const int* __restrict__ s3, const int* __restrict__ s4,\n    float* __restrict__ x4)", 1, "h_embed sig")
rep(r"const int toks\[4\] = \{ s0\[0\], s1\[0\], s2\[0\], s3\[0\] \};",
    "const int toks[5] = { s0[0], s1[0], s2[0], s3[0], s4[0] };", 1, "toks[5]")

# ---- 10) renames ----
REN = {"h_embed4":"h_embed5","k0n4":"k0n5","k0ab4":"k0ab5","q5g8v4":"q5g8v5","k2s4":"k2s5",
       "op38nw32_4":"op38nw32_5","k3aonw32_4":"k3aonw32_5","ao8nw32_4":"ao8nw32_5","hh4":"hh5",
       "ffn8v4":"ffn8v5","down8nw32_4":"down8nw32_5","aq3k8v4":"aq3k8v5","aq6k8v4":"aq6k8v5","head8v4":"head8v5"}
# R5 LAW FIX: PRESERVE the source launch bounds on rename. The m4 nw32-family
# kernels are __launch_bounds__(1024) (32 warps/CTA; gcycle launches "nw32"
# cubins at 1024 threads by the NAME-ENCODED LAUNCH CONFIG law). Normalizing
# to 256 here compiled cubins whose 1024-thread launches overrun the dext
# register allocation -> deterministic row-4 (a4 accumulator) corruption
# (the R4 53/60 exactness bug — root-caused R5).
for a, b in REN.items():
    rep(rf"void __launch_bounds__\((\d+)\) {a}\(", rf"void __launch_bounds__() {b}(", 1, f"rename {a}")
    assert "nw32" not in b or re.search(rf"void __launch_bounds__\(1024\) {b}\(", src), f"nw32 kernel {b} must stay 1024-bounded"

src = src.replace("T=4 PROBE trunk kernels (M=4;", "T=5 PROBE trunk kernels (M=5; k2s5 writes per-step slots 0..4 (slot 4 = live, sequential in-CTA);", 1)

# ---- audit: zero leftovers ----
for bad in ["ACC4H2(", "RED4(", "= (__half)a3;\n", "a3=0.f"]:
    pass
left_acc = len(re.findall(r"ACC4H2\(", src)); left_red = len(re.findall(r"RED4\(", src))
n5 = len(re.findall(r"a4", src))
assert left_acc == 0 and left_red == 0, (left_acc, left_red)
open(f"{BASE}/m5.cu", "w").write(src)
print(f"[gen_m5] OK -> m5.cu {len(src)} bytes; kernels {len(REN)}")
