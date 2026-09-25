# W2G: CROSS40 round D — all ranked levers REFUTED with evidence; HMMA tensor-core K1 built+validated standalone (a3 −0.08..−0.31ms/launch, probe −8.5..−10.9ms in-graph) but IN-GRAPH-BROKEN (zeros); canonical unchanged 39.03

## TL;DR
Mission: cross 40 tok/s @100k (gap −3.0ms from 71.32). Outcome: **every ranked
lever (1-4) refuted with measurement or decisive analysis; the session's real
finding is the corrected K1 bottleneck model — the dots are 88% of the HFMA2
issue floor, so only tensor cores can win — plus a validated HMMA K1 that is
EXACT standalone (Tier-2 ~1e-3, W2F-QH class) and fast (in-graph probe
64.76 -> 53.84-56.22ms) but produces ZEROS when run inside the engine graph.**
Canonical recipe/env UNCHANGED (`SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352
GEMVV=1 KV8=1 QH=1 PVH=1`, Tier-1 60/60 x2 from W2F stands; HM is additive,
default 0). 40 NOT crossed. The crossing path is now singular and concrete:
fix the HMMA in-graph failure.

## Lever 1 — K1 STAGE PHASE: REFUTED (premise was wrong)
- Built spk_g4pp.cu: software-pipelined stage (raw int4+scale regs held
  across the previous tile's QK/PV compute; per-thread 16B K-or-V split) +
  build_pp.py + test_pp.py. 64 regs 0 spills, symbols OK.
- Speed: a3 1.169->1.209 (+0.04), a1 0.536->0.541 (neutral). NOT WIRED.
- ROOT CAUSE of the miss (measured + arithmetic): K1 reads 205MB but does
  **37 GMAC/launch (QK 18.5 + PV 18.5)**; at the 3090's 35.7T MAC/s HFMA2
  ceiling the math floor is ~1.03ms vs 1.169 measured -> **the kernel runs at
  ~88% of the fp16-vector math ceiling. Stage was already hidden by CTA-level
  overlap; W2F's 0.4/0.25/0.15 phase split was an attribution artifact.**
- a1 cross-check: 0.536ms ~= 0.35 math floor (6 rows) + ~0.2 fixed.

## Lever 2 — T=1 DRAFT K1 WARP RESTRUCTURE: REFUTED (analytic)
a1 is throughput-bound, not warp-starved: splitting rows across more warps
cannot add MAC throughput (same LDS traffic per row, math is the wall). The
26 idle warps already cost nothing.

## Lever 3 — DRAFT GEMV HALF2 PORTS: REFUTED (measured)
- q4vh.cu (ehprojh/dqh/doprojh/ddownh), dkvh.cu, dfguh.cu: all six
  BIT-IDENTICAL on real packed weights (build_dh.py, test_dh.py).
- Speed: ALL NEUTRAL (ehproj -0.0002, dq -0.0003, dkv 0.0000, doproj +0.0002,
  dfgu +0.0093, ddown +0.0013 ms). NOT WIRED.
- Lesson: the draft GEMVs sit at ~170 GB/s LATENCY/L2-bound (not cvt-bound);
  W2D's wins came from the fat-CTA class, which does not transfer here.

## Lever 4 — DRAFT-VOCAB OWN-OUTPUT REBUILD: REFUTED (evidence)
The 100k prompt is a REPETITION: 97,810 tokens, **30 distinct** ids. The
slice already contains 100% of truth tokens ("ref covered: 60/60"): the
30-token slice is an accidentally-optimal truth-containing candidate filter —
restricted-argmax agreement (0.892) >= the draft's intrinsic agreement.
Expanding the slice only adds distractors -> alpha would DROP, not rise.
alpha=0.892 is DRAFT-FIDELITY-bound, not coverage-bound.

## THE REAL LEVER — HMMA TENSOR-CORE K1 (built, standalone-EXACT, in-graph-broken)
- Why: HFMA2 ceiling 35.7T MAC/s vs tensor fp16-f32acc 71T = exactly 2x; the
  kernel is at 88% of the scalar ceiling -> tensor cores are the only 2x-class.
- **spk_g4hm.cu** (engine0): mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  QK + PV. Structure: canonical STAGE (int2) unchanged; QK out-tiles
  (RP/16 m-blocks x 4 key-blocks) with QK_WPT-warp k-splits writing fp32
  partials to an SCP smem plane; row-owner warps (warp=row, lane=key) do the
  plane reduce + the SAME online-softmax op sequence as the scalar kernel;
  PV = A(P 16x16keys) x B(V 16keys x 8dims) into fp32 acc frags, rescaled by
  per-row cor each tile; final pm/ps/pA writes keep the exact partial contract
  (rows < RMAX only). pA written as consecutive-dim pairs from c-frags.
- **Validation (test_hm.py, 100k fixture, distinct buffers)**:
  pm/ps relerr max ~1.2e-3 median ~1.6e-4; pA F-norm max ~1.5e-3 — the exact
  W2F-QH Tier-2 class (fp16 inputs, fp32 acc — numerically BETTER than the
  scalar fp16-chunk accumulators).
- **Speed (standalone, synced)**: a3 1.164-1.168 -> 0.858-1.091 (best -0.31,
  worst-case -0.08 with QK_WPT=1; GPU-thermal spread). a1 consistently SLOWER
  (+0.03..+0.09; RP=16 pads 6 rows) -> **a1 stays scalar; HM is a3-only.**
- **Wiring**: mtp.py HM env (default 0): k1n = "spk_g4nwhm3_100k" if HM else
  (qh3p/qh3 as before). trunk_w1c.py (a1) untouched.
- **IN-GRAPH FAILURE (the open blocker)**: both gates (w100k_hm.log at
  43.4KB smem, w100k_hm2.log at 37.2KB) produced all-zero outputs (0/60,
  alpha=0) while the T=1 stock path stayed 59/59. Deterministic. The kernel
  is standalone-exact, so the failure is graph-context-specific. Even broken,
  the speed showed through: **probe 64.76 -> 53.84 (43KB build) / 56.22
  (36.4KB-class build) ms/cyc = -8.5..-10.9ms — this alone crosses 40**
  (71.32-8.5 -> ~44.6 tok/s at 2.78 tok/cyc).
- Ranked in-graph failure hypotheses for next session:
  1. dext smem carveout granularity: static smem beyond the k2s3-proven
     36.4KB point (my 37,248B is 384B over 36,864) may not be backed ->
     SCP/MS read as zeros -> uniform softmax. TEST: shrink to <=36,864B
     exactly (RP*12=384B is the overage; move MS into the SCP plane tail or
     pack msv/ssv/corv into the aliased P/K region's spare 14KB).
  2. The dext graph-launch path configuring 64-reg/1-barrier launches
     differently than the direct NVProgram path (compare the captured
     launch descriptor fields vs standalone).
  3. Dump pm3/ps3/pA3 after ONE in-graph cycle (add a debug down()) to see
     whether partials are zeros, poison, or garbage — pinpoints K1-vs-K2.
- DEBUG PATH PROVEN: mma_probe.cu/.py (uid+identity decoder) maps the TRUE
  fragment layout in one run. **Banked mapping for m16n8k16 on sm_86:
  a0=(m,k), a1=(m+8,k), a2=(m,k+8), a3=(m+8,k+8); b0={B[k][n=g],B[k+1][n]},
  b1={B[k+8][n],B[k+9][n]}; c0/c1=(m,n)/(m,n+1), c2/c3=(m+8,...); n_b=lane>>2,
  n_c=(lane&3)*2 — the b-n and c-n mappings DIFFER (hardware cross-maps).**

## Gotchas banked this session
- **args-before-up trap (variant of the W2F A/B law)**: capturing P.d[buf]
  objects into an args tuple BEFORE a P.up re-poison of the same name reads
  STALE buffers if up() reallocates -> kernels write orphaned memory, down()
  reads fresh poison, and "BIT-IDENTICAL: True" becomes poison==poison.
  ALWAYS build args AFTER the last P.up, and assert outputs != poison
  (7.7e31 > the usual 1e-20 "active" threshold — it passes activity checks!).
  This bit BOTH test_pp and the first test_hm run.
- Static smem on sm_86 caps at 48KB for ptxas, but the DEXT in-graph appears
  to only back the ~36.4KB class (k2s3) — 43.4KB compiles+runs standalone,
  zeros in-graph.
- mma fragment conventions are self-deceiving: three separate bugs (a1/a2
  swap; QK B-frag transposed row/chunk; PV B-frag using the c-frag column
  mapping) all passed visual audit — only the uid/identity probe settled it.
- KSZ/VSZ are BYTE offsets; indexing a __half* with them doubles the address
  (smem OOR fault on every SM).
- macOS has no `timeout`; `perl -e 'alarm N; exec @ARGV' -- <cmd>` works.
- nvcc shim needs ABSOLUTE source/output paths (docker exec cwd).
- Colima may still be provisioning after boot even when the socket exists;
  retry builds once docker ps answers.

## Files
- engine0/spk_g4hm.cu + spk_g4nwhm{3,1}_100k.cubin (HMMA K1; a3 validated)
- engine0/build_hm.py (builder), engine0/test_hm.py (A/B validation)
- engine0/mma_probe.cu/.py + mma_micro.cu/.py (fragment-layout decoders)
- engine0/spk_g4pp.cu + build_pp.py + test_pp.py (L1 refutation artifacts)
- engine0/q4vh.cu, dkvh.cu, dfguh.cu + build_dh.py + test_dh.py (L3, built
  bit-identical, unwired — neutral)
- mtp.py: HM env flag + a3 k1n selection (default OFF; canonical untouched)
- Logs: ~/w100k_hm.log (43KB gate), ~/w100k_hm2.log (36.4KB-class gate)

## What remains to 40 (ranked, singular)
1. **Fix the HMMA in-graph failure** (hypotheses above; the -8.5..-10.9ms
   probe prize is measured). Then Tier-1 gate (expect the QH-class outcome:
   fp32-acc numerics, zero or near-zero flips; regen T1 ref only if needed —
   note T1 path uses scalar a1 so Tier-1 tests HMMA-spec vs scalar-T1).
2. If smem backing is the issue and 36,864B is a hard cap: pack MS into the
   aliased region, or move row-owner state to registers per 8-row group.
3. a1 HMMA is NOT worth it (slower); the draft keeps the scalar build.
