# P1d — stacked lazy-dequant GEMVs for FFN gate|up: FEASIBLE BUT PERF-NULL (no fix shipped)

Date: 2026-08-25. Stage: P1d of MTP_PLAN.md v3. Verdict: **STOP per stage gate**
(microbench win <20% ⇒ no model.py change).

## What was tested
The RIGHT way to stack ffn_gate+ffn_up (per plan): cat the RAW GGUF IQ3_XXS quant
blocks at byte level BEFORE the lazy dequant expr, so the fused GEMV reads one
contiguous 68MB raw range and dequant is a pure view over it. No fp16
materialization anywhere (the 23.57GB OOM failure mode of the earlier attempt is
structurally impossible with this approach).

Implementation of the probe (/tmp/p1d_stack_bench.py, /tmp/p1d_final_bench.py,
committed under p1d/): targeted GGUF header parse → raw byte tensors for
blk.0.ffn_gate.weight + blk.0.ffn_up.weight (typ=18, dims (5120,17408) gguf =
tinygrad [17408,5120], 348160 blocks × 98B = 34,119,680B each) →
`ggml_data_to_tensor(t8g.cat(t8u), ng+nu, 18).reshape(34816, 5120)`.

## Correctness: EXACT
maxdiff = 0.000000 vs two separate dequant GEMVs, argmax equal. Block axis is the
K (fastest-varying) axis; both tensors share K=5120, so raw-byte concat = row-aligned
stacking [Wg; Wu] as [34816, 5120]. Cat-then-dequant IS a pure view — confirmed.

## Performance: NULL (the reason we stop)
Interleaved trials, one process, BEAM=1, TinyJit, 30 iters × 5 rounds:
  2 separate GEMVs (2×34MB raw):  mean 0.817 ms (steady ~0.810-0.818)
  1 stacked GEMV   (68MB raw):    mean 0.964 ms (steady ~0.801-0.812; r0 warm outlier)
  steady-state delta ≈ ±1% = noise. NOWHERE near the +20% bar.
Stacked really is ONE kernel (to_program hook captured exactly `r_2176_32`, vs sep's
r_2176_32 + E_80_16). Effective raw-byte BW ≈ 84 GB/s both ways.

### Why: the lazy-dequant GEMV is ELEMENT(ALU)-bound, not ramp/bandwidth-bound
Time scales with N·K (elements), not bytes:
  - IQ3_XXS [17408,5120] = 89.2M elems, 34MB raw  → ~0.405 ms  (220 Melem/ms)
  - fp16     [34816,5120] = 178M elems, 356MB     → 0.815 ms  (218 Melem/ms)
  Same element rate despite 5.25× fewer bytes. The 98B→256-elem dequant chain
  (fp16 d scale, u32 scale-word shifts, 128-entry sign LUT gather via even_signs,
  iq3xxs grid gather) costs more than the byte-read it replaces. Stacking removes
  at most one kernel-launch gap (~10µs), which is what we measured.
Consequence for P1c attribution: r_544 at "129 GB/s" is not 129-of-447 wasted
bandwidth — bytes are the wrong denominator. Per-element it already runs at the
same rate as a tuned fp16 GEMV in-model. A bigger grid cannot fix ALU throughput.

## GDN input projections (step 3): MOOT + structurally ill-defined
GGUF dtypes differ per tensor (blk.1): attn_qkv = Q5_K(13), attn_gate = IQ3_XXS(18),
ssm_alpha/ssm_beta = fp32 native(0). Raw-block stacking requires identical block
size/dequant path; a 4-way merge would need per-segment dequant anyway (= separate
kernels again). Not attempted given the null result.

## Implications / next-stage notes
1. Do NOT retry stacking for any IQ3/Q5 lazy-dequant weight. The OOM risk was real
   but irrelevant — even done perfectly it buys ~nothing.
2. The FFN gate/up cost (~33.5ms/tok) only moves via cheaper per-element work:
   better dequant codegen (vectorized LUT-free paths, tensor-core-friendly packing),
   or A3-style custom CUDA fused block kernel. Codegen-level, not layout-level.
3. Pre-dequantizing gate|up to fp16 at load needs 17408×5120×2B×2×48 ≈ 17.1GB —
   dead on VRAM (13.11GB steady today). int8 requant ≈ 8.6GB — also dead.
4. Probe scripts committed: p1d/stack_bench.py (correctness+timing),
   p1d/final_bench.py (interleaved verdict), p1d/probe2.py (kernel-count hook).
