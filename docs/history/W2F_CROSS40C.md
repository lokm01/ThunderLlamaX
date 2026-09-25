# W2F: CROSS40 round C — half2 QK/PV dots landed (39.03 tok/s, −6.0ms); 40 NOT crossed (L4 scan-split refuted)

## TL;DR
Mission: >=40 tok/s @100k via (1) half2 QK dots, (2) PV partial-half2,
(3) Lever-B GEMV wiring, (4) scan grid split, (5) remaining GEMV half2 ports.
Outcome: **1+2+3 landed Tier-1 EXACT 60/60 deterministic with ZERO greedy
flips vs every prior sequence, cycle 78.28 -> 71.32 ms (-6.96) = 39.03 tok/s
— NEW CANONICAL.** 4 (k2s grid split) was built, validated BIT-IDENTICAL, and
REFUTED on speed (+1.7ms/probe: the per-t conv+qk-norm prologue is replicated
in every substripe and costs more than the added parallelism wins). 5 not
attempted (<=1ms class, cannot close the remaining 3.0ms; budget). Gap to 40:
71.32 -> 68.3 = -3.0ms; ranked paths at the end.

## NEW CANONICAL (locked, gated)
env: `SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1`
(+ DEV=NV PATH/DOCKER_HOST as usual; `python -u test_w100k.py`, ~/snap100k).
| metric | W2E | W2F (QH+PVH+GEMVV2) |
|---|---|---|
| tok/s @100k (best of 3) | 35.56 | **39.03** |
| cycle | 78.28 ms | **71.32 ms** |
| phases | 5.55 / 70.80 / 1.16 | **5.30 / 64.76 / 1.16** |
| Tier-1 spec==T=1 | 60/60 x2 | **60/60 x2, deterministic** |
| vs fp16-KV T=1 seq | 60/60 | **60/60** |
| vs int8-KV(W2E) T=1 seq | — | **60/60 (zero flips)** |
| engine vs spec_base_100k | 59/59 | 59/59 |
| alpha / tok-per-cyc | 0.892 / 2.78 | 0.892 / 2.78 |
Log: ~/w100k_qh.log (QH only: 37.79), ~/w100k_pvh.log (canonical 39.03).
T=1 reference: ~/snap100k/engine_t1_ref_kv8_qh.npy (test_w100k REFSUF).

## Lever 1 — HALF2 QK DOTS: landed (−4.6ms cycle)
- **spk_preqh.cu** (KPRE + `qw16`): identical math + one extra output
  `qw16[i] = fp16(qw[i])` (qe*0.0625 exact in fp32; single rounding).
  Validated: qw BIT-IDENTICAL to spk_preq, qw16 == fp16(qw) exactly, int8/sc
  stores BIT-IDENTICAL. Cubins spk_pre{1,3}qh_100k (-DKPRE!).
- **spk_g4qh.cu** (canonical spk_g4qh_u8.cu build): QK dot = qw16 half2 +
  K half2 straight from smem: 4 HFMA2 + 1 LDS.128 + 1 LDG.128 per 8 MACs (vs
  8 cvt + 8 FFMA + 2 LDG.128); TWO half2 accumulators (4 independent fp16
  chains), unpacked to fp32 sc every 8 c-iters (=64 elements). `unroll 8`
  variant kills an 8B ptxas spill (same speed; canonical).
- Standalone: K1 1.537 -> 1.267 ms (**−0.27/launch**, −17.5%). Numerics:
  pm/ps relerr max 1.3e-3, pA F-norm 4.4e-4 (fp16-chunk-accumulate class).
- In-graph 100k: Tier-1 60/60, **zero greedy flips** (overlap 60/60 vs BOTH
  the fp16-KV and W2E int8-KV sequences — the fp16 rounding never flipped a
  near-tie at this prompt). Cycle 78.28 -> 73.65 (probe −3.8, draft −0.2).

## Lever 2 — PV PARTIAL-HALF2: landed (−2.3ms cycle, in the combined gate)
- **spk_g4qh2.cu -DPVH** -> spk_g4nw32qh{1,3}p_100k: p packed once per row
  per tile as half2 (broadcast via 32-bit shfl), v half2 from smem, 4 HFMA2
  per row into 4 fp16x2 accumulators, unpacked to fp32 A8 at the 32-row tile
  boundary (FA2-style: fp32 per-tile precision, half2 mults). 6 ops/row vs 17.
- Standalone: 1.267 -> 1.164-1.069 ms (**−0.198/launch** fresh-pair bench).
  Numerics: pA elem relerr mean 3.2e-4 (frac-diff 86%, Tier-2 class; max
  large only under cancellation on near-zero elements). pm/ps unchanged.
- GOTCHA (test infra): comparing two kernels through the SAME P.up-repoisoned
  buffers read bit-identical FALSELY — the second run's args/down can alias;
  ALWAYS use DISTINCT output buffers (pmA/pAA vs pmB/pAB) for A/B kernels.
  First PVH "identical" reading was this artifact; distinct-buffer rerun gave
  the true 3.2e-4 class. 100k gate: still ZERO greedy flips (60/60 vs both).

## Lever 3 — Lever-B GEMV wiring: partially landed (−0.45ms)
- **head8v_3** (head8_3 + LDH2/ACC3H2): BIT-IDENTICAL, 2.28 -> 1.94 ms
  (**−0.34/probe**). Wired under GEMVV=1.
- **aq3k8v_3**: BIT-IDENTICAL, −0.014ms x8 layers (−0.11). Wired (IQ3-q
  layers only; qtype==14 keeps aq6k8_3).
- **aq6k8v_3**: BIT-IDENTICAL but +0.005ms SLOWER — built, NOT wired.
- ACC3H2 order-preservation proof: pairs (2k,2k+1) ascending == j ascending,
  products fp16 exactly like ACC3, accumulation fp32 -> bit-identical
  (verified on real packed weights; V4ROW3/vrow path included).
- GOTCHA (fixture): aq* q/k weights MUST come from engine0/packed/*.npy
  (repacked layout); raw gguf rows produce NaN qrow and a FALSE "diff".

## Lever 4 — SCAN GRID SPLIT: REFUTED (+1.7ms, do not wire)
- Built k2s3v.cu (grid 48 -> 384: h=blockIdx>>3, sub=&7; each warp 2 v-rows
  instead of 16; conv-window shift sub==0 only; prologue replicated) +
  k2z3.cu (the zz RMS-norm + gated z3 tail as its own 48x32 kernel with the
  EXACT original reduction order — cross-CTA core[] dependency forces the
  split since 384 CTAs > 82 SMs at 1-CTA/SM residency makes spin-sync illegal).
- Outputs BIT-IDENTICAL to k2s3 on random fixtures (conv/rec fully-written
  paths; q/core/z3 were all-zero in the fixture — sigmoid saturation — so the
  z3 value path is transcription-verified only). SG=1 wiring exists in mtp.py
  (env-gated OFF).
- **Speed: 0.072 -> 0.107 ms/launch — REGRESSION.** The per-t conv+qk-norm
  prologue is replicated 8x and dominates the state-update parallelism win.
  k2s3 is 48x0.072 = 3.5ms/probe; the split would ADD 1.7ms. Lesson: the
  scan kernel is PROLOGUE-bound, not warp-starved.

## Lever 5 — remaining GEMV half2 ports: NOT ATTEMPTED
Draft-phase T=1 GEMVs (dq/dkv/ehproj/doproj/dfgu/ddown, 5.30ms draft total)
are the remaining fp32-core pool; W2D class says +3-19% each = −0.3-0.7ms
total. Cannot close −3.0ms; skipped for budget (validation-first protocol
would cost 4+ calls per kernel set).

## What remains to 40 (ranked)
Gap: 71.32 -> 68.3 ms/cyc (−3.0). Attention is now ~16x1.07 + combine;
GEMV pool ~38ms issue-bound (W2D); draft 5.30; head 1.9; accept 1.16.
1. **K1 dot phase round 2**: QK ~0.25 + PV ~0.15 + STAGE ~0.4 per launch
   remain. STAGE (BW 545GB/s) is now the biggest K1 component — an
   int8-STAGE with fp16 K-smem WRITE COMPRESSION or S/CH retune could shave
   ~0.1ms/launch. 2-warps-per-row for the T=1 draft K1 (6/32 warps compute).
2. **Draft GEMV half2 ports** (L5 above, −0.3-0.7ms, mechanical).
3. **alpha program** (draft fidelity): unlocks K=3+ machinery (Tier-1-proven,
   parked at W2D) — the ONLY lever that changes tok/cyc (2.78 -> 3+).
4. q5g8v/ffn8v/down8nw32 decode-LUT restructure (the W2D "dead end for this
   format set" — needs smem-LUT class rewrite).

## Gotchas banked this session
- The PVH A/B buffer-aliasing trap (above) — poison-and-reuse via P.up is
  NOT a valid A/B harness; distinct buffers or you'll validate nothing.
- P.down(shape) requires a real shape (None crashes); bench_diag-style
  quick scripts inherit this.
- The -DKPRE name-trap avoided again (cuobjdump -symbols before launch);
  NOTE: cuobjdump only exists INSIDE the container
  (`docker exec cuda-nvcc-persistent cuobjdump -symbols <abs path>`); there
  is no host shim.
- The boot-hook nvcc-container gap hit AGAIN this session (container exists
  but stopped after reboot): `docker start cuda-nvcc-persistent` first.
- head-slice when splitting .cu files drops `#include <cuda_fp16.h>` —
  prepend it or "__half undefined".
- k2s3 prologue is warp-replicated by design (lane-only channel indexing);
  any grid split replicates it per CTA — the reason L4 lost.
- aq6k8 half2 port is NET-NEGATIVE (Q6 decode is not issue-bound the way
  IQ3 is) — keep aq6k8_3.

## Files
- engine0/spk_preqh.cu + spk_pre{1,3}qh_100k.cubin (KPRE + qw16)
- engine0/spk_g4qh.cu, spk_g4qh_u8.cu (unroll-8, canonical qh builds),
  spk_g4qh2.cu (+PVH ifdef), cubins spk_g4nw32qh{1,3}_100k +
  spk_g4nw32qh{1,3}p_100k
- engine0/aq3k8v_3.{cu,cubin}, head8v_3.{cu,cubin} (WIRED, bit-identical);
  aq6k8v_3.{cu,cubin} (built, unwired — slower)
- engine0/k2s3v.{cu,cubin} + k2z3.{cu,cubin} (REFUTED, kept for record;
  SG=1 wiring in mtp.py env-gated OFF)
- engine0/test_h2qk.py, test_l3.py, test_l4.py (standalone validations)
- mtp.py / trunk_w1c.py: QH=1, PVH=1, GEMVV=1 set additions; test_w100k.py:
  REFSUF (_kv8_qh) + both-overlap tier-2 report + 3 timing reps.
- Logs: ~/w100k_qh.log, ~/w100k_pvh.log (canonical).

## Run
cd ~/tinygrad-metal/engine0 && env SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352
GEMVV=1 KV8=1 QH=1 PVH=1 DO_T1=0 DEV=NV PATH/DOCKER_HOST as usual;
python -u test_w100k.py   (DO_T1=1 once regenerates the _kv8_qh reference)
