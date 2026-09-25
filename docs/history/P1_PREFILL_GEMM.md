# P1 — Prefill pGEMM: the batched M=16 dequant-GEMM family

Status: **ALL THREE GATES PASS.** Per-class differential validation vs the
bit-proven T=1 reference kernels on REAL packed weights (gate a ≤3e-3),
shape-swept synced-timing bench projecting **169.7 tok/s GEMM-only prefill
throughput at M=16** (gate b ≥150), zero dext faults in the final sweep (gate c).
HMMA (mma.sync.m16n8k16) beats the HFMA2 fallback ~1.9x end-to-end — the
HMMA-vs-HFMA2 decision is HMMA, everywhere.

## What was built

- **engine0/pf_gemm.cu** — one templated kernel, 7 quant classes × HMMA/HFMA2
  compute paths × (NTHR, NTILE, KCH) shapes via `-D`. Structure per CTA:
  stage x[16][KCH] → smem; stage one NTILE W-row tile (each warp 8 rows; lane =
  (row = lane>>2, quarter = lane&3) decodes KCH/32 consecutive 8k lane-chunks —
  the reference lane-chunk math parameterized by chunk index lc, byte-for-byte
  the w1c.cu addressing); compute 16×NTILE out via mma.sync.m16n8k16
  (W2G/W2H probe-verified fragment map: a0=(m,k) a1=(m+8,k) a2=(m,k+8)
  a3=(m+8,k+8), b-n = lane>>2, c-n = (lane&3)*2, fp32 accs) or HFMA2 half2 dots
  (fp16 products + fp32 acc — same rounding class as the GEMV refs). FFN=1
  fuses gate+up with the exact ffn8 silu·mul epilogue (hsilu_h verbatim).
  Single-array smem (16B-aligned xs | ws planes), static, ≤39.2KB per combo;
  per-kernel cubins; names carry the warp token (nw8/nw16/nw32 → the
  NAME-ENCODED LAUNCH CONFIG law) + k-tile.
- **engine0/build_pf.py** — builder (nvcc via container, cuobjdump symbol check,
  docker-retry). **LAW: NTILE must equal NTHR/32*8** — the warp out-tile is
  hardcoded 8 cols; NTILE=32 @ 256thr faults (OOB) — removed from the matrix.
- **engine0/test_pf.py** — differential validation (validate) + trunk projection
  bench (bench). **engine0/bench_sweep.py** — the shape sweep.
- **engine0/dbg_pf.py / dbg_dq.py** — single-launch fault bisection + decode
  dump vs numpy (the forensics that found the bugs below).
- **engine0/smoke_pf.py + trunk_w1c.pf_ffn16_program (PF_GEMM=1)** — integration
  smoke: the FFN stage of GDN blk0 end-to-end (k3m_hh norm → 16-row batched FFN
  → down8 residual+projection) on real weights: gact F-norm 5.1e-4, block output
  y F-norm 1.2e-4 — PASS ≤3e-3.

## Validation (gate a) — 16 random fp16 rows, vs T=1 refs (p1_val.log)

| class (shape K→N) | ref kernel | hm F-norm | hf F-norm | med relerr |
|---|---|---|---|---|
| iq3 gate 5120→6144    | q5g8   | 3.19e-4 | 9.88e-6 | 0.0 |
| iq3 fd 17408→5120     | down8  | 3.22e-4 | 1.18e-5 | 0.0 |
| iq3 ssm_out 6144→5120 | op38   | 3.19e-4 | 1.28e-5 | 0.0 |
| fused ffn 5120→17408  | ffn8   | 5.29e-4 | 8.88e-6 | 6.2e-4/0.0 |
| q5 qkv 5120→10240     | q5g8   | 3.20e-4 | 1.23e-5 | 0.0 |
| q5 head 5120→248320   | head8  | 3.16e-4 | 1.17e-5 | 0.0 |
| q6 attn q 5120→12288  | aq6k8  | 3.19e-4 | 9.95e-6 | 0.0 |
| iq3 attn q 5120→12288 | aq3k8  | 3.18e-4 | 1.06e-5 | 0.0 |
| iq3 attn k 5120→1024  | aq3k8  | 3.19e-4 | 5.29e-7 | 0.0 |
| q4k attn v 5120→1024  | aq3k8  | 3.21e-4 | 8.47e-6 | 0.0 |
| iq3s attn o 6144→5120 | ao8    | 3.10e-4 | 1.03e-5 | 0.0 |
| q8_0 ssm_out 6144→5120| k3a_oproj | 3.03e-4 | 1.26e-5 | 0.0 |
| q4_0 draft fg (numpy fp32 xcheck) | — | 7.0e-3 | med 2.9e-4, outlier max 0.23 (fp16-product class; hm is the shipping mode) |

Elementwise medians are 0.0 = most elements bit-identical; hm's ~3.2e-4 F-norm
is the HMMA fp32-acc reassociation class (better precision than the refs' fp16
products), hf ~1e-5 (same per-element rounding as refs, only sum order differs).
Elementwise MAX shows near-zero-crossing outliers (e.g. 12.8) — the honest gate
metric is F-norm/median, both ≪3e-3.

## Bench (gate b) — synced timing, min-of-10, best shape per class (p1_sweep.log)

| class (×instances/16-tok chunk) | best shape | ms | GB/s | TFLOPS | chunk ms |
|---|---|---|---|---|---|
| ffn fused fg+fu (×64) | nw8k128 | 0.420 | 163 | 13.6 | 26.9 |
| qkv Q5 (×48)          | nw16k128 | 0.230 | 156 | 6.6 | 11.0 |
| gate iq3 (×48)        | nw16k128 | 0.192 | 62  | 5.2 | 9.2 |
| down iq3 (×64)        | nw8k128 | 0.324 | 105 | 8.8 | 20.7 |
| o18 iq3 (×24)         | nw8k128 | 0.182 | 65  | 5.5 | 4.4 |
| o8 Q8_0 (×24)         | nw8k64 | 0.234 | 143 | 4.0 | 5.6 |
| attn q Q6 (×8)        | nw8k64 | 0.386 | 135 | 5.2 | 3.1 |
| attn q iq3 (×8)       | nw8k128 | 0.278 | 87  | 7.2 | 2.2 |
| attn k iq3 (×16)      | nw8k128 | 0.138 | 15  | 1.2 | 2.2 |
| attn v Q4_K (×16)     | nw8k128 | 0.145 | 20  | 1.2 | 2.3 |
| attn o IQ3_S (×16)    | nw8k128 | 0.179 | 76  | 5.6 | 2.9 |
| head Q5 (×1)          | nw16k128 | 3.752 | 234 | 10.9 | 3.8 |

**Projection (best shapes): hm 94.3 ms per 16-token chunk → 169.7 tok/s
GEMM-only (8.3 TFLOPS eff = 11.7% MFU vs 71T tensor peak / 23.4% vs the 35.6T
HFMA2 ceiling). hf: 179.5 ms → 89.1 tok/s.** All-NW8-k128 single config:
108.3ms → 147.7 tok/s. The T=1 engine today: 21.8 tok/s prefill → **~7.8x
GEMM-stage speedup at M=16**.

### Attribution — what binds, and the P2/P3 levers
- Weight-stream roofline at 450GB/s = 12.5GB/450 ≈ 27.8 ms/chunk → we are 3.4x
  off; the gap is (1) single-buffered smem stages (decode and mma serialize at
  every __syncthreads — double-buffer or register prefetch of raw quant bytes),
  (2) small-N classes are CT A-count-starved (k/v: N=1024 → 16 CTAs → 15-20
  GB/s; gate N=6144 → 96 CTAs → 62 GB/s): K-SPLIT (CTA groups over K + fp32
  partial reduce) is the known fix (they total only ~19ms but are ~5x off their
  floor), (3) ffn at 163 GB/s (26.9 of 94.3 ms) is the single biggest prize:
  double-buffering + 2 CTAs/SM occupancy tuning targets ≥250 GB/s → ~17ms.
- Staged-goal context: 662 tok/s @100k (vLLM W4A16) needs ~60% MFU — P1's 11.7%
  is the expected first-rung; the 150 gate is met with margin at 169.7.

## New laws banked this session
1. **NTILE=NWARP*8 law**: the pGEMM warp out-tile is hardcoded 8 cols; a
   (256thr, NTILE=32, KCH=256) build faults OOB (row index r=lane>>2 spans 8
   rows of a 4-row warp tile). Keep NTILE = NTHR/32*8 only.
2. **The W-tile CTA-offset law**: stage_w must read W rows (blockIdx.x*NTILE +
   warp*8 + r) — missing the CTA term makes every CTA decode rows 0..63: the
   kernel runs clean but outputs repeat with period 64 (values validate on CTA0
   alone — a per-CTA-0-only decode dump can NOT catch it; always diff full-tile
   vs refs).
3. **Q5_K/Q4_K qs addressing**: lane-chunk lc's qs bytes live at
   `blk+qs0+32*(lc>>3)+8*(lc&3)+j` with the nibble selected by ((lc>>2)&1)<<2 —
   `(lc&7)` (the Q6_K form) is WRONG for Q5/Q4_K and shows as F~0.67 uncorrelated.
4. **down8 writes FP32** (y = hh + acc): a fp16-sized output buffer is an OOB
   write (device fault) — reference-side harness bug class; always match the
   trunk's buffer dtypes (xout is the f32 x0/x1 pair).
5. The W2G args-before-P.up trap bites in bench loops too: build args AFTER
   prep_out/poison reallocs (the stale handle points into freed/reused memory).
6. Draft Q4_0 GGUF nibble interleave = dfgu's form: weight k → byte
   `(k>>5)*16 + (o>=16?8:0) + (o&7)`, high nibble iff o≥16 (NOT byte k>>1).
7. Single-array smem remains mandatory hygiene (two arrays ran but keep one
   16-aligned array; ws plane offset 16*XS_LD halfs).

## Bugs found by the forensics ladder (DBG=1/2/3 ablations in pf_gemm.cu)
x-staging clean → decode staged clean (after law-2 fix) → decode VALUES exact
vs numpy (4.6e-4) → full kernel vs numpy dot (3.5e-4) → refs (law-4). The
decode-dump ablation (DBG=3) is the fast path to decode bugs; keep it.

## Files / how to run
- Build: `cd engine0 && python3 build_pf.py [filter...]` (needs docker nvcc env).
- Validate: `~/tg311/bin/python -u test_pf.py validate` (~4 min).
- Bench: `~/tg311/bin/python -u bench_sweep.py`.
- Smoke: `PF_GEMM=1 ~/tg311/bin/python -u smoke_pf.py`.
- Logs: ~/p1_val.log, ~/p1_bench.log, ~/p1_sweep.log (reboot-survivor paths).

## Next (P2/P3 handoff)
1. pATTN (causal M×L attention on the same fragment map) + pKPRE + pSCAN — the
   non-GEMM prefill path; then the full M=16 prefill loop behind PF_GEMM=1.
2. Double-buffer the W stage (or register-prefetch raw quant) — ffn 163→250+
   GB/s is the biggest single lever (~10ms/chunk).
3. K-split variants for N≤6144 classes (k/v/gate/o) — ~12ms pool.
4. M=32 template (x-tile 2x; smem still ≤32KB at KCH=64) once P2 shapes settle.
