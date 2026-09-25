# P1c attempt 2 — r_544 kernel pair IDENTIFIED (2026-08-24)

## TL;DR
The r_544 pair is **NOT RMSNorms**. It is the **per-block FFN gate/up GEMV pair**
(`ffn_gate`/`ffn_up`, [5120 -> 17408], IQ3_XXS lazy-dequant), 2 kernels/block/token,
128 total, ~33.5 ms/tok (~264 us each). Each reads its own 34,119,680 B raw quantized
weight (= 348160 IQ3_XXS blocks x 98 B, exact match to [17408x5120]) => effective
~129 GB/s vs 447 GB/s proven achievable => ~25 ms/tok theoretical headroom.
No cheap fix within stage scope; see "Why no fix shipped".

## How names decode (fork)
`tinygrad/codegen/opt/postrange.py get_optimized_ast()`: name = ("r" if kernel has a
REDUCE else "E") + "_" + join(SPECIAL vmax+1 sizes + range extents). Digits are
grid/loop sizes, NOT op hashes. RMSNorm(dim) actually renders as `r_256_20`
(~30 us, 129x/tok) — the original suspicion is disproven by measurement.

## Evidence chain
1. Hooked `realize.to_program` during real BEAM=1 JIT=2 decode (p1c_ident.py):
   captured rendered CUDA SOURCE for both variants.
2. Both variants: global (544,1,1), local (8,8,1)/(8,4,4), launch_bounds 64/128.
   Buffers (bytes): out float[17408]; x float[5120]; scalar float[1];
   norm-weight-like uchar[20480] (= [5120] F32, e.g. attn_norm-sized);
   WEIGHT uchar[34119680]; LUT float[1024] + LUT-aux uchar[11141120]
   (dequant lookup tables, indexed along the reduction axis).
3. Weight size exact-matches ffn_gate/ffn_up ([5120,17408] GGUF dims confirmed by
   direct GGUF header parse: blk.N.ffn_gate.weight dims=[5120,17408],
   ffn_down=[17408,5120]; all body tensors IQ3_XXS 98B/256elem).
4. Variant B additionally takes float[17408] input (the sibling branch output) and
   stores `out[i] = sib[i] * gemv[i] * (1/scalar)` — gate GEMV fused with the
   gate*up multiply. That is why TWO near-identical kernels run per block and why
   standalone single-linear probes never reproduce either variant.
5. Standalone probe battery (repro.py): attn_norm/output_norm/ssm_norm/q-normalize/
   qkv/ssm_out/attn_gate/beta/alpha/FFN pieces — NONE emit r_544 (fusion context
   differs outside the model). Norms are all sub-30us kernels.

## Why slow
Grid 544 CTAs x 64..128 threads = 35-70K threads total on 82-SM 3090 (wants
~168K threads for BW saturation); each thread serially walks the whole K=5120
(~62.7KB of weight per CTA). Latency/occupancy-bound, not DRAM-bound.
Beam (BEAM=1) chose LOCAL(32)+GROUP(4/8)+UPCAST(4) — insufficient K-split.

## Why no fix shipped
(a) dtype: inputs already fp16/fp32-appropriate — n/a.
(b) call-site reshape (stack gate|up): Tensor.cat MATERIALIZES both weights to
    fp16 => 2x178MB x 64 blocks ~= 22.7GB => OOM. Empirically proven by the dead
    agent's stack2 run (/tmp/p1c/stack2.log: OOM at 23.57GB before Phase B;
    results invalid). Raw-byte-level concat would need custom GGUF storage +
    dequant-offset surgery — not cheap.
(c) codegen patch: cannot name a broken rule; the schedule is a legitimate beam
    search choice (under-split K), not a bug. A targeted retune (BEAM=2 restricted
    to this shape, or manually pinned GROUPC/SPLIT-K opts) is an overnight-marathon
    class action — out of scope for this stage per instructions.

## Next lever (for P1d/P4)
Retune ONLY this GEMV shape [17408,5120]-IQ3_XXS toward split-K (target >=300GB/s):
-25ms/tok @2k (~100 -> ~75ms floor). Options: pinned applied_opts via KernelInfo
override hack; BEAM=2 with beam pruning to this shape family; or fold into the
A3 custom-CUDA-scan work (same nvcc dispatch path could carry a hand-tuned GEMV).
