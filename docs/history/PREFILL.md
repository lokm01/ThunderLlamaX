# PREFILL — the P1-P18 + R2b-R2d + P8 + T2 batched-prefill campaigns

The decode engine (40.35 tok/s @100k at the time, see ARCHITECTURE.md) started life
with a T=1-token prefill: ~21.8 tok/s of prompt ingestion. A fresh 100k-token prompt
took ~17 minutes before the first token. The P-series (18 sessions, 2026-09) rebuilt
prefill as a chunked, batched, graph-captured pipeline and pushed it **17x**; the
R2b/R2c/R2d rungs (2026-09-21) pushed it to **23x, crossing 500 @2k**; the R7b
deciders + P8 rungs (2026-09-22) pushed it to **24x**; the T2 W4A8 ship (2026-09-23,
Tier-2) pushed it to **26x @2k**:

| rung | 2k FRESH | 8k | 100k rebuild | % of 662 ref @100k |
|---|---|---|---|---|
| T=1 chunked prefill (start) | ~114 class | ~106 class | ~21.8 tok/s | 3% |
| P4 (M=16 trunk + DBUF GEMMs + batched fill_draft) | ~165 | 148.4 | — | — |
| P6 (M=32 GEMMs) | 245.0 | 190.8 | 165.3 | 25% |
| P11 (hybrid attention + chunk graphs) | 315.3 | 269.2 | 212.9 | 32% |
| P12 (launch diet) | 319.1 | 278.5 | 218.6 | 33% |
| P15 (M=64 trunk) | 357.1 | 306.3 | 235.0 | 35% |
| P17 (wide-M attention) | 373.5 | 347.3 | 248.2 | 37.5% |
| R2 (PG_SPLIT=4 + pfk_ab16w) | 374.0 | 349.4 | 247.6 | 37% |
| R2b (WY-C32 scan fixed + NC=2 + attnqkv m64 twin) | 401.7 | 371.3 | 255.8 | 39% |
| R2c (decode-r7 shared plane + M=128 trunk) | 494.4 | 451.3 | 314.3 | 47.5% |
| R2d (ring-4 gdnqg + attnqkv g=448) | 503.6 | 457.8 | 317.6 | 48.0% |
| P8 (qg packed5 + o-proj M-grid fold) — Tier-1 ship | 530.6 | 479.4 | 328.0 | 49.5% |
| **T2 (W4A8 packed7-IMMA ffn, Tier-2) — current** | **569.2** | **510.1** | **342.0** | **51.7%** |

The T2 rung is **Tier-2** (authorized numerics change: the ffn GEMM reads the packed7
planes through a linearized int4 codebook view — a 2.4% logits class shift); every rung
below it is the bit-identical Tier-1 path, still the kill-switch default (unset
`PF_W4A8` -> byte-identical, verified line-for-line). Full battery + banks:
docs/history/T2_P8W4.md.

Reference bar: **662 tok/s cold prefill @100k** on the same GPU class (syv-ai vLLM
W4A16 stack, measured on a cloud 3090 during the R0 pre-check); short-ctx Marlin
class is 1-3k. All numbers are prompt-ingestion tok/s including the batched draft
fill; "100k rebuild" = full-context rebuild from scratch. Gate discipline at every
rung: the 2k gate F-metric (max rel logit divergence) must equal the bank value
EXACTLY (9.408e-04 @2k), GATE A / CTRL / D / D2 token classes line-for-line the
bank, 100k rebuild cur=4471 exact. (History note: the banked "8k floor F 4.284e-02
= the M32 reassociation floor" was RETIRED at P8 — it was the M32 r>0 tail feeding
at pos 0, a bug; the fixed 8k gate reads F 9.824e-04 with A 60/60, better than the
old bank and explained by the fix.)

## What shipped (default-on in the daemon env)

1. **Chunked M-row trunk (P1-P6, P15).** The prompt is cut into chunks (2k-class
   windows); every trunk kernel runs batched over M rows with bit-identical per-row
   fp op order (the Tier-1 discipline). M climbed 16 -> 32 (P6: two `m16n8k16`
   tensor-core tiles share one staged W tile = 2x load amortization) -> 64 (P15:
   `PF_M64=1`, +12.5/10.0/7.5% at 2k/8k/100k). Norms/pre-attention/scan/draft stay
   2x16 halves (pos_slot_b, per-half pm/ps/pA, REC ring alternates) — the "seam".
2. **W7 weight repack (P7).** The prefill GEMM family reads a second, wider-tile
   alignment-lawed repack of the quantized weights (`pack_w7.py`, 4.28 GB covering
   160 tensors) — the original decode packs stay resident for the decode graphs
   (both-live budget: full 9.3 GB both-live FAULTS, measured; `PF_G3M_MB` swaps
   budget-ordered).
3. **Chunk-graph capture (P7F-1, `PF_PG`).** The M-chunk program is captured as
   2 QMD-chained graphs; ~700 launches per chunk leave the host entirely.
4. **Dual-graph S13/S26 attention hybrid (P10-P11, `PF_ATTN_HYB`).** Two captured
   attention graph sets; the submit picks by chunk position (THR=8192). The S13
   mid-pos law: at underfilled positions the smaller-CTA variant wins the critical
   path; at full context S26 dominates standalone but loses in-plan via empty-CTA
   partial writes.
5. **Wide-M attention (P17, `PF_ATTNW`).** ROWS=32/64 query-row windows
   (bit-identical to 4x the t32 pair, corr nz=0 over 393216 keys, deterministic x2;
   the 1024-thread w64q variant is register-spill nondeterministic — banked
   negative). w64h low arm + HYB w64 row-swap share one 19968-slot scratch.
6. **The 2k-chunk launch diet (P12).** Norms/embedding row-per-CTA grid merges +
   `TROWS=32` scan/pre kernels: 706 -> 527 launches/chunk with no new cubins.
7. **Interleaved batched fill_draft (P3-P4, `PF_DFILL`).** Draft-state fill became
   a recorded-trunk-hiddens replay amortized into the chunks (7 launches per
   16-position window) — the draft's own attention/o-proj/FFN became dead code
   during fill. This is what makes FRESH prefill numbers include the spec-decode
   seeding for free (alpha after prefill: 2.67 -> 2.68, spec==spec 160/160).
8. **The WY-C32 chunk-level scan (R2b, `PF_SCANC`/`PF_SCANC_N2`).** The per-16-row
   sequential scan chain became a chunk-level WY-representation solve
   (`pf_scanchunk.cu` + the 16-cubin pfc set): one launch triple per 64-row chunk
   (NC=2) or per 128-row chunk (NC=4, R2c), conv writeback at c=NC-1. Three
   stacked kernel bugs were root-caused via a determinism-bisect DBG ladder before
   it could ship: an uninitialized-smem upper triangle (THE in-plan NaN — the
   garbage was launch-history-dependent), a t16/t16l staging race (two-pass
   register staging + barrier), and malformed hi-lo MMA cross terms (independent
   B-operands per pass; M relerr 2.1e-4 -> 1.5e-5).
9. **The shared packed7 plane (R2c, `PF_DR7`).** fg/fu/fd upload as packed7 and the
   six live decode/spec GEMVs were ported to read the SAME plane (`r7d.cu`,
   bit-identical det x2) — the original packed copies are never uploaded, the
   both-live VRAM wall is gone, and prefill runs FULL m64 coverage (288 tensors).
10. **The M=128 trunk (R2c, `PF_M128`).** 128-row chunks: GEMMs are the proven m64
    cubins on 2-M-block M-grids; attention = 2x the shipped w64h windows; norms/
    emb g=128; tail r%128 -> M64 -> M32 (the law-2 tail re-commit: the m128 tail
    must clear BOTH ambient flags or the graph cache replays the M64 plan on the
    M32 tail).
11. **DBUF ring-depth-4 + consolidated attnqkv (R2d, `PF_RING4`/`PF_QKV1`).** The
    gdnqg GEMM got 4 named unit-register sets with loads issued 3 phases early
    (x1.098); attnqkv's 2-M-block pair became ONE g=448 launch (the P7E4 corruptor
    class cleanly retried and retired). Plan: 745 launches/chunk. The ring-depth
    coin-flip law: a ring extension on a mixed-twin `pf_gemm3m` kernel is a
    per-class per-shape bet on the SHARED register allocation — A/B at the exact
    in-plan grid before adopting (fd x0.93, out x0.765 = banked negatives).
12. **The packed5 qkv repack (P8, `PF_P5`).** The gdnqg Q5_K qkv segment — the
    one remaining narrow-load stream the R7 SASS audit found (U8 planes, the
    23 ms qg pool) — is repacked by `pack_w5.py` into `packed5/` (48 qkv tensors,
    6 bits/word units, PURE integer permutation, byte-exact roundtrip) so the GEMM
    reads true 16B units (lo-unit + meta-unit per lane-chunk; `pf_gemm3m -DP5A`
    ring-2). A/B bit-identical nz=0 det-x2; x1.291 on the class (907 -> 703 us,
    -9.8 ms/chunk). VRAM +1.89 GB both-live (raw Q5 stays for decode q5g8v) —
    proven to fit at the full 100k state by the 100k gate.
13. **The o-proj M-grid fold (P8, `PF_OP64`).** The attention o-proj ran as 4x M=32
    g=80 launches per block; the fold (`pfg3_iq3s_m64_nw8k128`, pf_gemm3
    QCLASS=5 classic body at MTILE=64) runs ONE g=160 launch — the same
    M-grid-fold pattern as the R2d qkv consolidation. A/B bit-identical nz=0
    det-x2; x2.060 on the pool (10.3 -> 5.0 ms).
14. **The Tier-2 W4A8 IMMA ffn (T2, `PF_W4A8=1` — the current default).** The ffn
    gate+up GEMM becomes a fused `mma.m16n8k32.s8.s8.s32` kernel
    (`p8_w4ffn7.cu`) fed by a per-(row,128k-chunk) absmax int8 activation quant
    (`pfk_q8.cu`, bit-identical to the numpy quantizer). The W side needs NO new
    planes: the kernel reads the EXISTING packed7 stream through a linearized int4
    view of the IQ3 codebook (the iq3xxs grid is 8 linear levels, delta=4.0507 —
    a 1.49% weight-RMS linearization, 8.4x better than a raw int4 requant; per-32k-
    scale-word rescaling at the mma k-step granularity). Zero new VRAM (a 1 KB LUT),
    full 64-block coverage, decode kernels untouched. Numerics are Tier-2 (relerr
    4.6e-4 vs its fp16 ref; 2k F 1.250e-02 the new Tier-2 bank) with the battery
    green per the P7E7 convention: 2k A 60/60, 8k A 55/60 with D2 160/160 exact,
    100k cur=4471 + rebuilt-state Tier-1 decode 60/60, kill-switch line-for-line.
    The v1 int4-plane variant (`p8_w4ffn.cu` + `pack_w4.py` -> `packed4/`) is kept
    in-tree for partial-coverage experiments but is VRAM-blocked at full coverage
    (the <1.9 GB 100k-state headroom vs +5.88 GB planes).

## What was falsified (with the mechanism evidence)

- **The "overhead myth" (P12).** Isolated-class graph attribution: the sum of
  per-class times == the full chunk (110.0 vs 106.3 ms). The chunk is ~97% kernel
  work; host/window-upload is ~0.2 ms. Launch-count reduction helps only where
  launches actually dominate (the superchunk at 48k+ was ~80% launch-bound).
- **Decode-side ALU as the GEMM wall (P5).** H2 LUT decode = 3x fewer decode
  instructions = 0.99x. The FFN/IQ3 family's 150-167 GB/s plateau is the
  quant-word load stream against the hard 1-CTA/SM limit — 13 validated variants.
- **Sync removal (P5): SYNCW 0.87x. Waves/occupancy: fat CTAs 0.25-0.68x NEGATIVE.**
- **cp.async-style double buffering on multi-plane kernels (P5-P6):** DBUF landed
  +14-23% on 1-plane kernels, flat on the interleaved FFN family (stage_W/mma
  serialize per chunk).
- **Ping-pong smem stages (P16).** Best 1.058x vs the 1.2 gate; kernels
  bit-identical, barrier count irrelevant — the GEMM wall is per-warp issue
  serialization, not stage synchronization.
- **Persistent CTAs (P13-P14).** The mechanism probe answered: 713 GB/s steady at
  82 CTAs, 1/SM; wave/ramp effects are only +17%; it is NOT a stream limit. The
  persistent-FFN ring (w2/w4/w8) measured 0.91-0.93x in-plan — the per-chunk
  decode->mma->sync serial chain (~3.1 us x 40 chunks) is the wall. Killed by the
  1.35x gate; wiring kept env-gated OFF (`PF_PERSIST`).
- **IMMA W8A8 int8 tensor cores (P12) — scope pinned at P8, CORRECTED at T2, shipped.**
  The P12 tile test's mma-issue ceiling was 10% of tile time -> <5% relief, below gate.
  The P8 discriminator (`p8_imma.cu`, W4A8 `mma.m16n8k32.s8.s8.s32` at the census-worst
  ffn shape M=64 K=5120 N=17408) measured "266.6 us/launch vs the shipped fp16 stream's
  1220.3 us -> x4.58" — **a measurement-frame artifact** (T2 correction): the IMMA arm
  ran ONE W plane at M=64 g=272 while the census arm was the TWO-plane M=128 g=1088
  launch; per fp16 unit-of-work the 1220.3 us is ~305 us/one-plane-one-M-block, so the
  true IMMA advantage is **x1.1-1.15 standalone, x1.06 fused-with-full-decode (v2),
  x1.56 fused-with-free-nibbles (v1, VRAM-blocked)**. The law: per-launch speedup claims
  MUST match plane-count and M-grid between arms (the discriminator-frame law). The W
  stream saves only 19% of bytes; the decode/rescale chain (per-32k-group scales!) eats
  the rest. The T2 ship (item 14 above) took the honest +7.3% and closed the 750 route.
- **The "superchunk" fused scan (P7E).** A single fused multi-chunk kernel,
  bit-deterministic and 1.14x faster at 100k — but numerically NOT Tier-1: a 1-2
  fp16-ULP amplifier at block 0 grows ~1.25x/block through the 48 GDN blocks once
  the mature conv-row-2 stream is present, decorrelating state by ~8k real-text
  tokens (0/60 vs baseline). `PF_SUPER=0` stands. The forensic byproduct DID ship:
  hi+lo fp16 splits (>=21-bit, 3-pass MMAs) on the six hot scan carriers, fixing
  the recurrence-seed divergence 60x (block-0 rec 2.31e-3 -> 3.7e-5).
- **Fused last-CTA attention combine (P10 A4):** 2.55 vs 2.40 ms for the pair —
  self-reset atomics work but lose.
- **K-global-direct attention (P9):** 0.91x. Co-residency via wider CTAs is
  reg-capped (1024-thread thread-capped; nw16 at 128 regs regfile-capped).

R-series falsified (docs/history/R2_RUNGS.md): m128-GEMM singles/twins (255 regs +
452-888 B spill = the P18 nondeterminism law); split-plane FFN nw8 (bit-identical,
1.153x slower — the fused nw4 x/smem reuse beats warp count); the w128h window
(bit-identical at pos100224, NONDET-WRONG at low pos — the P17 ROWS-extension
class); pre32x4 (gated bit-identical-exact, perf-neutral — banked behind
`PF_PRE32X4=0`); ring-4 on fd/out/qkv classes (the shared-register-allocation
coin flip above).

R7b falsified (docs/history/R7B_DECIDERS.md — the warp-spec round that priced the
road to 750 and found it closed at this design point):

- **Meet-based warp-spec (`pf_gemm3w.cu`)**: producer/consumer warps with a K=64-stage
  smem ring, bit-identical det-x2 nz=0 on real weights — and x0.82-0.96 at every
  tested shape. With mbarrier dead, the per-stage full-CTA named-barrier meet couples
  producer stores with consumer decode on the critical path; the base's sync-pair +
  register DBUF ring keeps more loads in flight. Do not retry meet-based warp-spec on
  this family; a future retry needs working subset signaling or a persistent-CTA
  megakernel.
- **X-stream uint2 -> uint4 widening**: bit-identical on 5/6 classes but attnqkvi3
  BREAKS, and perf x0.78-0.89 on the nw8 classes — on latency-bound prefill GEMMs,
  halving X-load instruction count by doubling width is NEGATIVE (issue diversity
  beats width). Sources kept as `*.cu.x4banked`.
- **The GEMM census (r7b_classes.py, true M128 shapes)**: the six GEMM classes sum
  to 159 ms of the 256 ms 2k chunk (~62%) — ffn fg+fu 58.6 (37%), gdnqg qg 36.3
  (23%), iq3d fd 30.6 (19%), iq3o out 12.7 (8%), attnqkv 12.4 (8%), o-proj 8.8 (5%).
  The 750 arithmetic needs that pool at ~75 ms (2.1x) — which is why packed5/o-proj
  (P8) took the two cheapest slots and the rest waits on the IMMA tier.

## The ceiling arithmetic (honest statement)

After T2 the platform is at **51.7% of the 662 reference @100k (342.0) and 569.2
@2k (~86%) on the Tier-2 W4A8 path**; the bit-identical Tier-1 path stands at 49.5%
(328.0 @100k, 530.6 @2k). What remains measured and structural: the GEMM pool (~62%
of the 2k chunk, census above) where warp-spec-via-meets and X-widening are both
measurement-falsified AND the W4A8 IMMA tier is now honestly priced at x1.06 fused
(the x4.58 was a frame artifact; the +7.3% v2 ship is the real take) — **750 is
closed on every route**, and the attention growth pool (the 7*CH wave-2-goes-full
step at position 54040) needs persistent-CTA attention or dynamic smem >48 KB (the
ptxas static cap for the raw-cubin path). Full 662 parity is structurally uncertain
on this dext: 1 CTA/SM (except name-gated carveout classes), no cp.async, and wave
quantization cliffs. Realistic endpoint: ~350-450 tok/s @100k Tier-1, plus the
Tier-2 W4A8 class on top. The 100k FRESH wall is no longer prefill-bound (~4.8 min);
the serving answer is the durable prompt cache (SERVING.md / R1), not more prefill
speed.

## Campaign laws (see also DEXT_LAWS.md, P-group)

READOUT-ORDER (gates from the FIRST clean run — timing reps advance GDN state);
the ~0.10-0.12 ms launch floor for <10 MB kernels makes K-split net-negative;
FFN NTILE + W-tile CTA-offset + `8*(lc&3)` qs addressing; the spill-nondet law
(276 B of spill = a nondeterministic kernel); the M64 tail-seam trio (stale
pos_slot / ambient-flag graph-cache key replaying the M64 plan on the 32-token
tail / head post-tail on a stale row); tie-mine gate texts need the both-sides
control; gate harnesses need the FULL decode env or they fault at the first
decode-graph exec; chunked prefill must keep the `stseed_spec(N&1)` parity
contract; the ring-depth coin flip; the M128 tail must clear BOTH ambient flags.

Session-by-session: docs/history/P1_*.md through P18_GROWTH.md, p1[ced]_findings.md,
CTASM_INVESTIGATION.md; the R2b/R2c/R2d rungs in docs/history/R2_RUNGS.md; the R7b
deciders (GEMM census, warp-spec negative, X-widening negative) in
docs/history/R7B_DECIDERS.md; the P8 rungs (packed5, o-proj fold, the P0 stale-feed
fix, the IMMA discriminator) in the P8 campaign record + engine/p0_repro.py,
engine/pack_w5.py, engine/p8_imma.cu.
