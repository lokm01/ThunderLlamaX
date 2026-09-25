# R2d — THE PRICED SCRAPS: 500 @2k CROSSED — ring-4 gdnqg + qkv g=448 SHIPPED: 2k 503.6 / 8k 457.8 / 100k 317.6 (Tier-1 63.08 held; pre32x4 gated exact but perf-neutral, banked off)

Status: **the mission is achieved — 503.6 tok/s @2k clean first-run gate
(bank 494.4, +9.2 = +1.9%). Scrap 1 (DBUF ring-depth-4) and scrap 2 (qkv
2-M-block consolidation) landed individually gated, line-for-line the R2c
banks at every length; scrap 3 (pre32x4) gated bit-identical-exact but is
perf-NEUTRAL at 2k-class (503.3 vs 503.6) — wiring stays in-tree behind
PF_PRE32X4=0. 8k 457.8 (bank 451.3, +6.5; the assert-firing first attempt measured 458.9 same class), 100k rebuild 317.6 (bank 313.6,
+4.0 = 48.0% of the 662 reference), Tier-1 decode 63.08 (bank 63.16 — the
scraps touch prefill only).**

## 1. Scrap 1 — DBUF ring-depth-4 (PF_RING4=1; commits fdfe1c8)

- **The kernel** (pf_gemm3.cu RING4 body + pf_gemm3m.cu RPK branch,
  build_r2d.py): 4 named unit-register sets (chunk k in u[k&3]), manual
  unroll-by-4 group cycle. Each phase issues unit(s+4) into u[s&3] (the slot
  is free — unit(s) was decoded staging chunk s at phase s-1) + x(s+2)
  BEFORE mma(s); decode/mma/k-order/epilogue VERBATIM -> BIT-IDENTICAL by
  construction (r2d_ring4.py G1: nz=0 det-x2=0 ALL 5 classes on real
  weights). Unit loads get ~3 phases of DRAM-latency cover (depth-2 gave
  ~1). NCH%4==0 holds (40/136/48). Guards: REPACK+RING4+FFN -> #error.
- **The adoption is gdnqg ONLY** (r2d_ring4b at the TRUE M128 shapes):
  qg@512 x1.098 (832.4 -> 757.9 us/blk = -3.6ms/chunk), BIT-IDENTICAL.
  **BANKED NEGATIVE (do not retry blind)**: fd x0.930 AND fd128 x0.794 (the
  ring regs 128->180 slow the NCH=136 stream), out WON m64-shape x1.063 but
  LOST the 2-M-block g=160 shape x0.765, qkv-i3 neutral x1.012, qkv-q6
  x0.835 — in the mixed twin kernels the ring-4 register pressure (up to
  182) slows the CLASSIC segments that share the kernel; only gdnqg (where
  nvcc landed 128 regs + 8B stack) won. LAW: **a ring-depth ext on a
  pf_gemm3m mixed twin is a per-CLASS, per-SHAPE coin flip dominated by the
  shared register allocation — A/B at the exact in-plan grid before
  adopting.**
- Gates (r2d_g2k_ring4.log, first clean run): **500.6 tok/s** (chunk med
  257.9 vs bank 261.0), F 9.408e-04 EXACTLY, A 59/60 tie class, CTRL 17/60,
  D 18/60, D2 428/428 — ALL line-for-line the R2c final bank.

## 2. Scrap 2 — attnqkv 2-M-block consolidation (PF_QKV1=1; commit cdafb66)

- The M128 attnqkv arm: ONE g=448 launch on the full xh128/qrow128/
  krow128/vrow128 replaces the 2x g=224 pair (identical cubin, the mb =
  bid/TGRID math already generalizes — the same pattern qg has shipped at
  g=512 since R2c). **The P7E4 corruptor class is CLEANLY RETIRED**: A/B
  BIT-IDENTICAL nz=0 both flavors (i3 + q6), standalone x1.109/x1.098; the
  in-plan 2k gate green line-for-line. Plan 761 -> 745 launches.
- Gates (r2d_g2k_qkv1.log): **503.6 tok/s** (chunk med 256.4), F 9.408e-04
  EXACTLY, A 59/60, CTRL 17/60, D 18/60, D2 428/428.

## 3. Scrap 3 — pre32x4 (PF_PRE32X4, gated exact, banked NEUTRAL)

- Wiring: 4x pfk_pre32_100k on 32-row views, pos_arr128 = 4 per-launch bases
  [p0, p0+32, p0+64, p0+96] (the multi-launch pre base-pos law). The 2k gate
  (r2d_g2k_pre32x4.log): F 9.408e-04 EXACTLY, A 59/60, D2 428/428 — pre32
  rows are bit-identical to pre64 rows (per-row math, pos_arr[0] the only
  position input). Perf: 503.3 vs 503.6 = NEUTRAL (the P15 24-CTA underfill
  does not bite in-graph at 2k-class; pre cost is row-driven, not pos-driven
  — no 100k dividend expected). Default PF_PRE32X4=0.

## 4. Final-stack gates (ship config = R2c canonical + PF_RING4=1 PF_QKV1=1)

- **2k: 503.6** (r2d_g2k_qkv1.log — the first clean run of exactly this
  config; F 9.408e-04 EXACTLY, A 59/60, CTRL 17/60, D 18/60, D2 428/428).
- **8k: 457.8 tok/s** (r2d_final_g8k2.log, the first COMPLETE clean run; F 4.284e-02 floor EXACTLY; A 9/60
  phase-shift, CTRL 13/60, D 6/60, D2 0/390 — line-for-line the bank; the
  first 8k attempt hit the KNOWN P7F1 tie-assert (4649,9956) — run 8k gates
  with PF_GATE_TIEOK=1, harness law, numerics unaffected: F was exact on the
  assert-firing run too).
- **100k rebuild: 317.6 tok/s** (308.0s; cur=4471 EXACT match=True; drift max
  rec 5.320e-3 / conv 3.614e-3 == the bank values; rebuilt-state T1 decode
  60/60; curve 285@0 -> 386@48k-class -> cliff 58k -> 519@97k, the shipped
  shape shifted down ~4-6ms/chunk).
- **Tier-1 decode: 63.08 tok/s** (107.54 ms/cyc; 60/60 x2 det, alpha 2.892
  EXACT, deep=off 40.24 superset intact, deep=on 59.06, tier2 x2 60/60,
  stock 59/59) — decode untouched by construction (prefill-only knobs).

## 5. Ship config (daemon)

```
M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1
PVH=1 HM=1 M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256
LOOKUP_K=7 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1
NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1
PF_M128=1 PF_DR7=1 PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=1 PF_SCANC_N2=1
PF_M64QKV=1 PF_ABW=1 PC_ENABLED=1 PF_RING4=1 PF_QKV1=1
```
Kill-switches: PF_RING4=0 the r7 gdnqg; PF_QKV1=0 the 2x g=224 pair (both
bit-identical worlds, gates above). pcache: PF_RING4/PF_QKV1/PF_PRE32X4 in
_ENV_KEYS (new namespaces by construction).

## 6. New laws banked (R2d)

1. **The ring-depth coin flip** (scrap 1): deeper unit-register rings are
   NOT uniformly good — the pf_gemm3m mixed twins share one register
   allocation across RPK/classic segments; ring-4 pressure (180-182 regs)
   slowed the classic majority segments (qkv-q6 x0.835, out@160 x0.765).
   Only where nvcc kept <=128 regs (gdnqg) did the ring pay (x1.098).
   A/B at the exact in-plan grid, per class, before adopting.
2. **The M-grid consolidation is retired-risk**: a 2-M-block g=2x launch of
   the pf_gemm3m twins is BIT-IDENTICAL to 2x the single-block launch and
   ~10% faster (wave/dispatch amortization) — the P7E4 corruption was
   SC-path-rooted exactly as R2b adjudicated. Any future M-ext should fold
   the M-grid rather than loop launches.
3. **8k gate harness law**: the P7F1 tie-assert (4649,9956 at 8k pos
   ~7690/7722) fires on GREEN numerics — 8k gate runs need
   PF_GATE_TIEOK=1 (the R2b precedent; the F-metric is the truth).
4. pre32x4: pre-kernel cost is row-driven (pos-independent) — the P15
   underfill law does not transfer to the in-graph M128 world.

## 7. Files / logs / commits

- engine0/pf_gemm3.cu + pf_gemm3m.cu (RING4 bodies), build_r2d.py (5 r7q4
  cubins), r2d_ring4.py / r2d_ring4b.py (the A/B + shape discriminators),
  pf_prefill.py (RING4/QKV1/PRE32X4 wiring + graph key), pcache.py (_ENV_KEYS).
- Logs: ~/r2d_ring4.log, ~/r2d_ring4b.log, ~/r2d_g2k_ring4.log,
  ~/r2d_g2k_qkv1.log, ~/r2d_g2k_pre32x4.log, ~/r2d_final_g8k.log (the
  tie-assert first attempt), ~/r2d_final_g8k2.log (TIEOK, the bank),
  ~/r2d_final_r100k.log, ~/r2d_final_t1.log.
- Commits: fdfe1c8 (scrap 1 + banked negatives), cdafb66 (scrap 2), this
  doc (R2d section).

---

# R2c — DECODE-R7 + M=128 SHIPPED: 2k 494.4 / 8k 451.3 / 100k 313.6 (both priced rungs landed, +21-23% over R2b; 500 @2k NOT crossed — 1.1% short, the priced remainder below)

Status: **both remaining R2b rungs are landed, gated line-for-line at every
length, and shipped in the daemon. decode-r7 (PF_DR7=1): fg/fu/fd upload as
packed7 and the 6 live decode/spec GEMVs read the SAME plane (r7d.cu ports,
BIT-IDENTICAL det x2, perf-neutral) — the packed originals are never uploaded,
the both-live VRAM wall is gone, and prefill runs FULL m64 coverage (288
tensors). M=128 (PF_M128=1): 128-row chunks — GEMMs = the proven m64 cubins
(2-M-block M-grid), scan = WY-C32 NC=4 (one triple/chunk), attention = 2x the
shipped w64h windows, norms g=128, tail r%128 -> M64 -> M32. Ladder (clean
boot): 401.7 -> 494.4 @2k (+23.1%), 371.3 -> 451.3 @8k (+21.5%), 255.8 ->
314.3 @100k rebuild (+22.9% = 47.5% of the 662 reference). Tier-1 decode
green on the ship config. 500 @2k = 494.4: NOT crossed; the honest remainder
is ~5-8 tok/s of PRICED, bit-identical-class scraps (below) — no structural
blocker, just gate cycles.**

## 1. Rung 1 — decode-r7 (PF_DR7=1; commit 9b595ca)

- **The 6 kernels** (engine0/r7d.cu -> per-kernel cubins): ffn8r7/down8r7
  (trunk T=1 G_CYCLE), ffn8v3r7/down8nw32v3r7 (the K2 probe, GEMVV),
  ffn8v8r7/down8nw32v8r7 (the K=7 deep probe). The packed7 unit addressing for
  a warp-per-row GEMV: per block b, lane l reads chunk (2b + l>>4), unit
  ((row&7)*4 + ((l>>2)&3)), q u16 @+2*(l&3), sw u32 @+8, d u16 @+12 — the
  words are VERBATIM (pack_w7.py) and 16h+4c+cc == l, so the lane->k mapping,
  per-lane acc order, and shfl tree are UNCHANGED -> BIT-IDENTICAL (r7d_test.py:
  nz=0 det x2 on real weights, all 6; bench 0.92-1.04x — the +30.6% unit-pad
  byte tax is INVISIBLE: these GEMVs are latency-bound and the r7 layout's
  one-64B-span-per-chunk beats the packed 3-stream row).
- **The wiring**: trunk_w1c._upraw diverts fg/fu/fd to packed7 (E._r7native);
  _ffnk/_downk select the r7 kernels in gdn/attn/_build_seqs; mtp.py selects
  them in probe_g + _probe8_seq (asserts GEMVV/K3-off/K in {0,7} — the
  non-ported families would misread the plane). ensure() ALIASES the trunk
  buffers into W7 (zero extra VRAM) -> FULL coverage: 288 tensors (fd64 fg64
  fu64 gate48 out24 q8 k16), 1.43GB uploaded. VRAM net -2.4GB vs the R2b world.
- **Gates** (line-for-line the R2b banks): Tier-1 60/60 x2 @62.9 tok/s (alpha
  2.892 EXACT, deep=off 40.07, stock 59/59, tier2 x2 60/60); 2k 437.5 (F
  9.408e-04 EXACTLY, A 59/60, D2 428/428; chunk med 145.5 = -14.9ms vs 160.4);
  8k 405.6 (F 4.284e-02 floor, A 9/60, CTRL 13/60, D 6/60, D2 0/390); 100k
  274.1 (cur=4471 EXACT, drift 5.29e-3/3.48e-3, rebuilt decode 60/60).

## 2. Rung 2 — the M=128 trunk (PF_M128=1; commits f8f5eb1 + 4c2f398)

- **The plan** (ensure128, 745 launches/chunk): emb/n16/ab16w/hh16 at g=128
  (row-per-CTA, free M-scaling); qkv = 2x the g=224 M64QKV twin; pre64 x2 on
  64-row half-views; attention = 2x w64h+pfcw64h on half-views; qg/ffn/fd/out
  = the m64 cubins at 2-M-block M-grid (g = 2xNGRID, ONE launch per class —
  the P7E4 in-plan-exact set); scan = pfca/pfcb/pfcz_c32_nc4 (ONE triple per
  chunk, grids 192/192/768 per the NC laws); dfill = 8 x 16-row windows (even
  -> REC1, the ring law holds); tail r%128 -> prefill_batch with BOTH ambient
  flags cleared -> M32.
- **The measured M128 value**: chunk med 260.4ms/128r = 130.2 per 64-equiv =
  **-15.3ms vs the dr7 M64 trunk** (more than the seam-only pricing: the NC4
  scan triple + the halved launch/norm/pre/emb seam + the M-grid single-launch
  GEMM classes all compound).
- **THREE bugs found on the way** (all via the gate F-metric, one control each):
  1. **pre64 x2 base-pos**: both halves appended KV at [p0, p0+64) — rows
     64..127 NEVER written (F 2.4e-1). FIX: pos_arr128 = [p0, p0+64], each
     half passes its own 4B view. LAW: a multi-launch pre chain needs a base
     per launch — pos_arr[0] is the ONLY position the kernel reads.
  2. **w128h BANKED NEGATIVE** (r2c_w128h_corr.py): ROWS=128 HRP=1 (48KB-exact
     smem, 128r+44B spill) is BIT-IDENTICAL at pos 100224 but NONDET-WRONG at
     low pos (pos 64 nz=747 det-x2 False; pos 2032 nz=8 nondet) — the P17 w64q
     ROWS-extension register class extends to ROWS=128. The M128 attention =
     2x the shipped w64h (bit-identical, zero risk). NO 128-row attention
     window exists on this dext without a redesign.
  3. **The P15 law-2 tail re-commit**: the m128 tail cleared only _M128ON —
     with _M64ON left True, _pf_graphs keyed (m64=True) built the M64 graphs
     while the 34-row tail ran the M32 path = the M32 chunk replayed the M64
     graph on ids64 (8k F 3.0e-1, first-div-0; the 7680 no-tail control 60/60
     EXACT localized it in one run). LAW: a trunk-chain tail must clear EVERY
     ambient flag of the chain above it.
- **Gates**: 2k 494.4 clean-boot (F 9.408e-04 EXACTLY, A 59/60 tie class, CTRL
  17/60, D 18/60, D2 428/428); 8k 451.3 (F 4.284e-02 floor EXACTLY, A 9/60
  phase-shift, CTRL 13/60 alpha-class == bank, D 6/60, D2 0/390 — P7E7
  criterion PASS); 100k rebuild 313.6 (cur=4471 EXACT match=True, drift max
  rec 5.320e-3 / conv 3.614e-3 == the bank class, rebuilt-state decode 60/60;
  curve 290@0 -> 386@48k -> cliff @58k (7*CH) -> 525@97k, the shipped shape).

## 3. Banked negatives this session (measured, do not retry blind)

- **m128 GEMM singles/twins** (pfg3_*_r7_m128_nw4k128): 255 regs + 452-888B
  spill — the P18 nondet law class AND slow. Root: XTPR (x-staging regs) =
  MTILE*KCH/NTHR = 33 uint2 at nw4; at nw8 the smem wall (52224B > 48KB
  static). M>64 GEMM on this kernel generation is smem+spill DEAD until the
  KCH=64 26B/32B unit layout exists (P7B escape #1 — campaign-class).
- **Split-plane FFN** (fg/fu as single-plane m64-nw8 + pfk_smul): corr
  BIT-IDENTICAL (det x2 nz=0) but 1.153x SLOWER (0.887 -> 1.023 ms/64r/block)
  — the fused nw4's x/smem reuse beats nw8's warp count. The fused nw4 stays.
- **PG_SPLIT=8 at M128**: 493.5 vs 494.7 — neutral; split4 stays.
- **w128h**: see above.

## 4. The 500 statement (honest)

- **494.4 @2k clean-boot** (chunk 261.0/128r). 500 needs -4.4ms/128r (-1.7%).
  The remaining PRICED, bit-identical-class scraps (unclaimed only for gate
  cycles): (a) DBUF ring-depth-4 on the 4 non-FFN m64 GEMM classes (P7F3's
  priced +1-2%; the FFN is spill-blocked at 255r); (b) the qkv 2-M-block
  M-grid consolidation (g=448 one launch; needs its own A/B — the P7E4
  corruptor class was never cleanly retried post-fix); (c) pre32 x4 replacing
  pre64 x2 (the P15 underfill law, ~0.8ms). Sum ~5-8 tok/s — 500 is one
  short session away, NOT a physics wall.
- The STRUCTURAL walls (measured this session): FFN m128 (smem+spill, above);
  attention ROWS>64 (the w64q/w128h nondet class); the P16 issue-serialization
  wall (ping-pong dead); P13/P14 persistent-FFN dead. @100k the wide-attention
  KV-extent pool remains the binding half (313.6/662 = 47.4%).

## 5. Ship config (daemon, verified)

```
M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1
PVH=1 HM=1 M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256
LOOKUP_K=7 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1
NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1
PF_M128=1 PF_DR7=1 PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=1 PF_SCANC_N2=1
PF_M64QKV=1 PF_ABW=1 PC_ENABLED=1
```
Kill-switches: PF_M128=0 the dr7 M64 trunk; PF_DR7=0 the packed originals +
budget-capped coverage (both bit-identical worlds, gates above). pcache:
PF_DR7/PF_FFNSPLIT/PF_M128 all in _ENV_KEYS (new namespaces by construction;
128-boundary nodes are 64-boundaries too — chain_keys compatible).
Verified live: health ok (parked 97810 / cur 4471), FRESH chat coherent,
FOLLOW_UP delta-only (prompt_tokens 60 = the delta; 2.2s turn). Tier-1 on the
SHIP config: 60/60 x2 det, **63.16 tok/s** (107.40 ms/cyc, alpha 2.892 EXACT,
E[m|deep] 7.000 80.0%, deep=off 40.38 superset intact, stock 59/59, tier2 x2
60/60) — decode IMPROVED a hair vs the R5d bank (63.11).

## 6. New laws banked (R2c)

1. **The r7 port pattern**: a warp-per-row decode GEMV reads packed7
   BIT-IDENTICALLY (chunk = 2b + l>>4; unit (row&7)*4 + ((l>>2)&3); q@2cc,
   sw@8, d@12) — the lane->k map and fp op order are untouched. Works for the
   M=3/M=8 half2 batched families verbatim.
2. **The multi-launch pre base-pos law**: pre64/pre32 derive ALL row positions
   from pos_arr[0]; a chunk split into multiple pre launches needs one base
   per launch (pos_arr[2] = [p0, p0+64]) or the later spans never hit the KV.
3. **The trunk-chain tail law** (P15 law-2 generalized): a tail delegating
   from an M=2^k trunk must clear EVERY ambient flag of the chain (m128 AND
   m64) — the graph cache keys the union; one stale flag replays the wrong
   plan's graph on the wrong buffers.
4. **ROWS>64 wide-attention is the w64q register class**: BIT-IDENTICAL at
   high pos, NONDET-WRONG at low pos (w128h: nz=747@pos64) — corr at ONE
   position does NOT clear a wide kernel; corr at LOW pos is the required arm.
5. **XTPR register scaling**: x-staging regs = MTILE*KCH/NTHR uint2s — m128
   at nw4 = 33 = spill-death; budget register headroom BEFORE building M-exts.

## 7. Files / logs / commits

- engine0/r7d.cu + r7d cubins (6), build_r7d.py, r7d_test.py, r7d_bench.py;
  trunk_w1c.py/mtp.py/pf_prefill.py/pcache.py (DR7 wiring); pf_prefill.py
  (M128 machinery + FFNSPLIT knob), build_r2c128.py, pfk_smul{64,128}.cu,
  r2c_ffnsplit_test.py, r2c_w128h_corr.py, r2c_m128_bisect.py, patch_* scripts.
- Logs: ~/r2c_g2k_base.log (399.9 control), ~/r2c_dr7_{t1,g2k,g8k,r100k}.log,
  ~/r2c_m128_g2k{,2,3}.log + g7680.log (the control) + g8k2.log + r100k.log +
  g2k_s8.log, ~/r2c_final_{g2k,g8k,r100k}.log (the clean-boot records).
- Commits: 9b595ca (decode-r7), f8f5eb1 (M128 + banked negatives),
  4c2f398 (the tail fix + the green 8k/100k).

---

# R2b — WY-C32 SHIPPED: the scan-kernel bug root-caused (three stacked defects) + NC2 tier + attnqkv-m64 retry; 2k 401.7 / 8k 371.3 / 100k 255.8

Status: **the blocked rung is landed. The R2 "kernel bug" was THREE stacked
defects in pf_scanchunk.cu's pfca, found via a 5-stage early-return DBG ladder
(determinism bisect + factor dumps). The WY-C32 scan now runs in the M64 trunk
(PF_SCANC=1 PF_SCANC_N2=1, the NC=2 single-launch tier, 3 launches/chunk) and
the P7E4-quarantined attnqkv m64 twins are retried clean (PF_M64QKV=1). Every
A/B is BIT-IDENTICAL where numerics permit, the 8k quality trial passes at the
M32 reassociation floor (the P7E7 degenerate class is GONE), and the ladder is
374.0 -> 401.7 @2k (+7.4%), 349.4 -> 363+ @8k, 247.6 -> 255.8 @100k rebuild
(+3.3%, 38.7% of the 662 reference). 500 @2k NOT crossed — the remaining
priced rungs are decode-r7 (-13ms) and M=128 (-15-25ms), both campaign-class;
arithmetic says ~460-490 with both.**

## 1. The three defects (engine0/pf_scanchunk.cu; commit 19fa971)

1. **THE R2 NaN — the T-solve left Xf's UPPER TRIANGLE as uninitialized SMEM.**
   T = (I+B)^-1 is unit-lower-triangular; the solve writes only i2 >= j. The
   t_g dump and pfcb's d-MMA read the FULL CxC — the upper triangle held
   whatever the SM's previous occupant left: 0.0 on a fresh SM, 7.6e-6-class
   fp16-leftovers after a sibling pfca (self-inheriting = "deterministic"
   across replays), ARBITRARY payload after the trunk's GEMM/attention kernels
   = the in-plan NaN (d consumed launch-history-dependent garbage; R2's
   identical-NaN-on-both-tiers signature = both tiers read the same trash
   class). C=64 "passed" for years on denormal-tiny luck (the P7C fwd32
   stage-4 evidence was real — the pre-hi-lo layout left small-bit leftovers
   there). FIX: zero the full upper triangle after the solve (subsumes the
   old C=64 upper-right-block special case, which only covered the cross-tile
   block — the WITHIN-tile upper triangles were also uninitialized at C=64!).
2. **The t16/t16l staging RACE.** t16l's write region overlaps Xf's head rows
   (512B at C=32, 1KB at C=64) inside one unsynced loop that both reads Xf and
   writes t16l — late Xf readers (the t_g dump + the loop's own stragglers)
   got clobbered values -> nondeterministic T rows 0-3. Caught by the DBG=5
   stage (deterministic through the dumps; breaks the instant the staging is
   added). FIX: two-pass register staging (read ALL C*C Xf floats — exactly
   C*C/NTHR = 2 (C=32) / 8 (C=64) per thread — barrier, then write).
3. **accM's hi-lo passes were MALFORMED.** The ph loop shared ONE khB
   B-operand between accB and accM: accM received (hi,hi)+(lo,lo) — BOTH
   cross terms missing, i.e. M rode bare-fp16 input quality (the P7E5/P7E6
   "amplifier channel that would not close" is this: the z-path's M factor
   never got its hi-lo correction; the rec path never saw it, which is why
   P7E6's rec numbers looked fixed). FIX: an independent khBm B-fragment
   pointer per pass -> M = (hi,hi)+(hi,lo)+(lo,hi); M relerr 2.1e-4 -> 1.5e-5.

## 2. The evidence chain (how they were found)

- fwd32 with the CORRECTLY-SIZED scratch (48*125712 — the P7C-era 50448 in
  pf_fwd32.py/test_p7c.py predates the P7E5/E6 hi-lo rebuild; the stage-4
  "validation" of c32 was stale) FAILS: logits med 4.2e-2 F 1.6e-1, rec drift
  5-21% through blocks — clean reproduction of the R2 in-plan corruption.
- pfca det-x2: 8.3KB differ, t/t_lo quartets; NaNs in meta bg/sg/gend. C=64:
  byte-deterministic (the luck).
- DBG ladder (early-return cubins dumping gz/Tf/Xf into the dead dz zone):
  stages 1-3 (gzone, masks/Tf, solve/Xf) deterministic AND exact vs numpy —
  the corruption enters AFTER the solve. Stage 4 (dumps only): clean. Stage 5
  (+staging): breaks at t row 0 -> defect 2. Removing defect 2 leaves a
  FIRST-LAUNCH-ONLY diff (0.0 vs 7.6e-6 subnormal at stable positions) ->
  uninit-smem read -> defect 1. Factor re-check with defect 3 fixed: M 14x
  better -> defect 3 confirmed as real but silent.
- Post-fix: pfca det x5 byte-identical; test_p7c gate1 O 1.3e-5 rec 1.2e-6
  (was 4.1e-4/3.2e-4), gates 2/3 z med EXACT vs the REAL pfs16, rec 1.1e-6
  (260x under the P7C-era kernels that P7E7 trialed); fwd32 stage-4 PASS
  med 8.2e-4 / F 1.65e-3 (bank 8.7e-4/2.4e-3), argmax 31/32.

## 3. The gates (readout-order; same EFI-clean boot as the 372.9 baseline control)

| gate | R2b WY-C32 | bank (R2/P15-P17) | verdict |
|---|---|---|---|
| 2k (chained) | 384.7, F 9.408e-04, GATE A 59/60 (4649/43614 exact-tie flip at pos 1), D2 428/428 | 372.9 this boot / 374.0 bank, F 1.058e-03 | +3.2% same-boot |
| 2k (N2) | 396.5, F 9.408e-04 | — | bit-identical to chained (0/4334080) |
| 2k (+M64QKV) | **401.7**, F 9.408e-04 | — | bit-identical to N2 (0/4334080) |
| 8k quality (chained) | 363.0-363.4, **F 4.284e-02** (bank 4.282e-02 — the M32 floor), GATE A 9/60 phase-shifted tie-cycle, tokB = the coherent physics-term cycle, **CTRL 13/60 alpha-class IDENTICAL to bank**, GATE D 6/60 (bank 5/60), D2 0/390 | 349.4 | **P7E7 criterion PASS** (the C=64 trial was 0/60 + 1-token attractor + F 5.1e-1) |
| 100k rebuild (N2) | **255.8 tok/s**, **cur=4471 EXACT (match=True)**, drift max rec 5.319e-3 / conv 3.540e-3 (gate <=1e-2; bank 5.348e-3/3.615e-3), rebuilt-state decode **60/60**; chunk curve 222-228 flat -> cliff 286.8 @58k -> 293.6 @97k | 247.6 | +3.3% |
| 8k ship-config (N2+M64QKV) | **371.3 tok/s** (med 178.5), **F 4.284e-02** (== the chained bit-identity), GATE A 9/60 phase-shift class, **CTRL 13/60 IDENTICAL to bank**, GATE D 6/60, D2 0/390 | 349.4 | **+6.3%** |

Why C=32 passes the 8k quality trial where P7E7's C=64-NC4 failed: P7E7
trialled kernels with rec-drift ~1e-3-class (the uninit-triangle + missing M
terms) — that state decorrelated at maturity. The fixed kernels run at
1.2e-6-class rec drift vs pfs16: the 8k end state lands at the SAME F as the
M32 path (4.28e-2), i.e. the reassociation floor, not the amplifier wall.

## 4. Tier + retry details

- **NC=2 tier** (PF_SCANC_N2=1): ONE launch triple per 64-row chunk — pfca
  grid 48*2 computes both C=32 sub-chunks (the c=1 boundary conv reads kvbuf
  fp16 rows directly = the same values the chained tier's pfcz->convlive
  fp32 roundtrip produces), pfcb chains c=0->c=1 in-CTA (state in smem, no
  global roundtrip), pfcz writes conv at c==NC-1. A/B BIT-IDENTICAL
  (0/4334080: logits + rec/conv x5 incl. blk 62/63). Chunk med @2k:
  171.6 (baseline) -> 164.6 (chained WY) -> 160.4 (N2).
- **attnqkv m64 retry** (PF_M64QKV=1): the P7E4-quarantined pfg3m_attnqkv
  twins in ONE launch (g=224 x 256thr, the P7E4 wiring class). A/B vs m32x2
  BIT-IDENTICAL (0/4334080). The P7E4 corruption was SC-path
  launch-history-rooted exactly as R2 adjudicated ("one clean retry due").
  QUARANTINE LIFTED. +5.2 tok/s @2k.
- PF_GATE_TIEOK=1 (pf_gate2k.py): the P7F1 tie-assert class (kernel argmax vs
  np.argsort order on EQUAL fp16 logits) downgrades to a warn — the WY
  landscape has a different tie set (4649/9956 at 8k pos 7690/7722; the 2k
  world still fires the old (4649,43614) pair at pos 1 as a warn).

## 5. The 500 statement (honest)

- Landed: 401.7 @2k (+7.4% vs the R2 ship; the priced WY rung delivered
  ~-11ms/chunk at 2k: 171.6 -> 160.4 including N2+M64QKV).
- Remaining priced: decode-r7 GEMV rewrite (-13ms, frees VRAM for full m64
  fg/fu coverage — campaign-class per P7F3: the w1c decode G_CYCLE graphs pin
  the originals) and M=128 (-15-25ms; the M-extension pattern is oiled —
  gen_m8 audits — and the scan tier composes: C=32 x NC=4 = one triple per
  128-row chunk; needs the 128-row buffers/norms/attention-window set + the
  r%128 tail seam). Both together: ~430-490 @2k. 500 needs both PLUS one more
  scrap class. NOT crossed this session — the budget went to the kernel
  forensics (the DBG ladder + 16 cubin rebuilds + the 5-gate stack).
- @100k: 255.8 / 662 = 38.7%. The P18 arithmetic updates to ~280-290 once
  M=128 lands (WY already in).

## 6. Ship config (daemon)

```
M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1
PVH=1 HM=1 M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256
LOOKUP_K=7 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1
NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1
PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=1 PF_SCANC_N2=1 PF_M64QKV=1 PF_ABW=1 PC_ENABLED=1
```
Kill-switches: PF_SCANC=0 restores the R2 canonical bit-identically;
PF_SCANC_N2=0 the chained tier; PF_M64QKV=0 the m32x2 qkv — all in the
graph-cache key. pcache namespace: PF_SCANC is already an _ENV_KEY — PF_SCANC=1
is a NEW namespace by construction (the WY numerics change); nodes rebuild on
first use (PC_ENABLED=1).

## 7. New laws banked (R2b)

1. **THE UNINITIALIZED-SMEM LAW**: a dump/consumer must never read a matrix
   region the producer leaves structurally-zero but physically-unwritten —
   smem carries the PREVIOUS kernel's payload on this dext (no smem clear
   between launches). Self-inheriting kernels (same cubin back-to-back) look
   deterministic while being wrong; first-launch vs subsequent-launch diffing
   (5-run det matrix) is the detector.
2. **The staging-overlap audit**: whenever a smem region is REUSED with an
   offset shift (Tf -> t16/t16l/bvt), check EVERY new region against EVERY
   still-live READ region — not just the one the comment mentions. The
   register-relay pattern (read-all -> barrier -> write-all) is the universal
   fix when the data fits in registers (C*C/NTHR <= 8).
3. **A shared mma B-operand silently breaks paired hi-lo passes**: when two
   accumulators need DIFFERENT B-sides per pass (B: (hi,hi),(lo,hi),(hi,lo)
   vs M: (hi,hi),(hi,lo),(lo,hi)), load TWO B-fragment pointers per pass —
   the "skipped ph" pattern is the smell.
4. The DBG early-return ladder (dump candidate buffers into a dead scratch
   zone, return at successive phases, det-x2 between stages) localizes
   nondeterminism to an exact phase pair in <1 min/stage — cheaper than any
   amount of source staring (defects 1+2 both hid behind "correct-looking"
   barriers).
5. Gate harnesses must re-derive buffer-size constants from the CURRENT
   kernel macros (or import them) — the stale HC_BYTES dict invalidated
   months of "validated" evidence (R2's indictment of the stage-4 pass was
   right).

## 8. Files / logs

- engine0/pf_scanchunk.cu (3 fixes), build_p7c.py (16 cubins rebuilt 0-spill),
  pf_prefill.py (SCANC_N2 + M64QKV + graph key), pf_gate2k.py (TIEOK),
  pf_fwd32.py + test_p7c.py (scratch/HC_BYTES resize), diag_c32*.py /
  diag_ladder*.py / diag_vals*.py (the forensics harnesses).
- Logs: ~/r2b_g2k_base.log (372.9 control), ~/r2b_fwd32_sc{,2}.log (repro +
  PASS), ~/r2b_g2k_wy.log, ~/r2b_g8k_wy{,2}.log (quality trial),
  ~/r2b_r100k_wy.log / ~/r2b_r100k_n2.log, ~/r2b_ab_n{0,1}.npz /
  ~/r2b_ab_qkv.npz (the A/B state dumps).
- Commits: 19fa971 (kernel fixes + validation), 58088a9 (cubins),
  0860abf (ship wiring).

---

# R2 — THE PRICED PREFILL RUNGS: PG_SPLIT=4 + pfk_ab16w SHIPPED; WY-C32 BLOCKED ON A KERNEL BUG (full forensics banked)

Status: rungs 1 and 4 are gated and shipped (bit-exact by construction,
line-for-line gate banks); rung 2 (the WY-C32 scan in the M64 trunk) is
BLOCKED: the current (P7E5/E6 hi-lo) pf_scanchunk C=32 builds NaN the trunk
state in-plan — BOTH the NC=2 single-launch tier and the fwd32-validated
NC=1 x2 chaining tier, identically — which indicts the kernel, not the
wiring (the wiring mirrors the P7E7-shipped SC integration arg-for-arg).
Rungs 3/5/6 not attempted (budget went to the WY forensics; adjudications
inside). Ladder + the 500 statement at the bottom.

## Rung 1 — PG_SPLIT=4 (the P15 state-unverified bit-gate) — SHIPPED

- P15 measured split4 at -2.7ms/chunk but shipped split=2 for lack of a
  state gate. This session: gated at 2k + 8k + 100k as part of the final
  stack (split4 is numerics-null by construction — pure PfGraph submission
  granularity; the 4 queues chain wait/signal exactly like the 2-queue
  shipped form).
- Gate: see the final-stack bank below.

## Rung 4 — pfk_ab16w: the fat-CTA ab-norm reshape (PF_ABW=1) — SHIPPED

- The P15 attr: nrm_ab = 3.84ms/64-chunk over 48 launches — the pfk_ab16
  grid is 64x13 = 832 tiny CTAs = 10.15 waves at the dext's hard 1 CTA/SM
  (wave/latency-bound, not BW: ~1MB total traffic).
- The reshape (engine0/pf_norms16.cu KSEL=7, pfk_ab16w.cubin): grid
  64x3 = 192 CTAs of 1024 threads (32 resident warps to hide the L2
  latency); the 96 alpha/beta GEMV jobs stay ONE WARP EACH with the math
  VERBATIM (same lane-strided dot, same shfl_down tree, same per-element
  half(x*r*nw) rounding) -> per-job bit-identical outputs; the xh write is
  elementwise-identical (write order is irrelevant on disjoint elements).
  Build: 62 regs, 0 spill, 0 smem (cuobjdump-verified; the P18 spill law
  respected).
- The ORIGINAL 832-CTA kernel stays default (PF_ABW=0); the M32 tail and
  the M32 trunk are untouched.
- Expected: nrm_ab 3.84 -> ~1.5-2.5ms (the 8-warp CTA had only ~8 loads in
  flight/SM; 32-warp CTAs have 4x the memory-level parallelism; the wave
  count drops 10.15 -> 2.34).

## Rung 2 — WY-C32 scan into the M64 trunk — BLOCKED (kernel bug, forensics)

### What was built
- New cubins pfca/pfcb/pfcz_c32_nc2 (build_p7c.py targets added): one
  64-row chunk = 2 C=32 sub-chunks in ONE launch triple. Builds clean:
  pfca 60r/21504B smem, pfcb 80r/32032B, pfcz 36r/0B — all in-graph-legal,
  0 spill.
- The M64 wiring (pf_prefill.py ensure64, behind PF_SCANC): the scwp weight
  plane (48x47200, the ensure_sc-verbatim loader), scscr64 scratch
  (48*2*125712B — HC_BYTES computed from the CURRENT source macros,
  anchored against HC64[64]=263696), sco64 (64x6144 fp32); the plan swap
  replaces pfs64 with the launch triple; graph-cache key += SCANC;
  pcache._ENV_KEYS += PF_SCANC (new cache namespace by construction).
  Grids: pfca 48*NC, pfcb 192, pfcz 48*NC*(C/8) — the pfcz C/8 law (C=32
  -> 4 not 8) verified against the fwd32 wiring.

### The failure (identical on both tiers)
- 8k stacked gate (split4+WY): PF_BATCH prefill RAN clean (no fault,
  373.7 tok/s) but end-state logits = NaN; gate A assert (h[pos-1]=0).
- Eager discriminator (PF_PG=0, PG_SPLIT=2, PF_NPROBE=1 chunk-0 probe):
  [r2probe] rec0 nan=283520/786432 absmax=3.19e4; qkv64/gate64/sco64/z64/
  xA64/xB64 ALL NaN downstream. => the corruption starts IN the scan
  kernels (eager = not a graph/capture issue; split2 = not a PG_SPLIT
  issue).
- Tier triangulation: the NC=2 tier AND the NC=1 x2-chained tier (the
  fwd32 wiring shape, conv-writeback chaining via the in-order queue)
  produce the IDENTICAL NaN signature (rec0 19-36% NaN, absmax ~3.1e4).

### The indictment of the "C=32 passed fwd32 stage-4" evidence
- ALL pfc cubins on disk (incl. c32_nc1) were rebuilt 09-17 15:18 — the
  P7E5/P7E6 hi-lo era. But pf_fwd32.py's harness still allocates the
  PRE-hi-lo scratch: P.poison("scscr", 48 * 50448) vs the current kernel's
  HC_BYTES(C=32) = 125712 per (head,chunk) — 2.5x undersized. The stage-4
  pass recorded in P7C_CHUNKSCAN.md predates the hi-lo rebuild; the current
  C=32 kernel has likely NEVER been validated in any harness (the only
  hi-lo-validated shapes are C=64/NC=4 and C=64/NC=8 on the SC path).
- Next-session plan (the P7C factor-level diag ladder, correctly sized):
  1. Re-size pf_fwd32.py's scscr to 48*125712 and re-run stage-4 with the
     CURRENT cubins — expect the NaN to reproduce in the harness.
  2. diag_p7c-class factor dumps (kh/qe/u/m/t vs numpy) on c32_nc1 with
     REAL trunk states; the P7C laws 3-7 (stride repetition, K-dim split,
     barrier overlap, dump guards, padded strides) are the suspect classes
     for a C=32-only hi-lo regression.
  3. If the C=32 path is deep-broken: the C=64/NC=1 tier is smem-illegal
     in-graph (43520B) — the alternatives are the eager PF_PG=0 chunk path
     (loses the P7F1 capture win) or fixing the C=32 kernel.
- The nc2 cubins + wiring stay in-tree behind PF_SCANC (default 0) for the
  follow-up.

## Rungs 3/5/6 — adjudicated, not attempted (budget)

- Rung 3 (decode-r7 GEMV rewrite -> free ~5GB -> full m64 fg/fu coverage):
  a campaign-sized job per P7F3 (rewrite the w1c decode family to read
  packed7; the decode G_CYCLE graphs pin the originals; the swap
  choreography is prefill-only-process or staged). This session's budget
  went to the WY forensics. Priced value stands: -13ms + unlocks M=128.
- Rung 5 (M=128): builds on 3's full coverage — session-class.
- Rung 6 (qkv/oa m64 retry): the cubins exist (SC_G3); the retry is one
  wiring + one gate cycle; not reached. The P7E4 corruption was
  launch-history-rooted on the SC path; the M64 trunk is a different world
  (the P15 experience says one clean retry is due).

## Ladder (final stack: PF_M64=1 PF_ATTNW=1 PG_SPLIT=4 PF_ABW=1 PF_SCANC=0)

| length | P17 bank | P18 (post-crash boot) | R2 this session |
|---|---|---|---|
| 2k fresh (PF_TRUNC=2048) | 373.5 | 328.6 | **374.0** (5.5s; last-half chunk med 171.4ms; F **1.058e-03 EXACTLY**; tie-mine assert (4649,43614) fires IDENTICALLY; T1/batch end-logits identical top1 29877/15.438 gap 0.0703) |
| 8k fresh | 347.3 | 307.6 | **349.4** (22.1s; last-half chunk med 189.3ms; gates LINE-FOR-LINE the bank: F **4.282e-02**, GATE A **12/60**, CTRL **13/60 alpha 2.67**, GATE D **5/60**, D2 **0/160**) |
| 100k rebuild | 248.2 | 225.6 | **247.6** (395.0s; fill_draft 256.5s unchanged; **cur=4471 EXACT (match=True)**; drift max rec 5.5e-3 / conv 3.6e-3 (gate <=1e-2, the P15/P17 class); rebuilt-state T1 decode **60/60**; chunk curve 223 @0 -> 234 @48k -> cliff 293.9 @58k (the 7*CH wave law) -> 298 @97k — shape identical) |

Gates: ALL GREEN, line-for-line the P17/P15 banks at every length (readout-order law: first clean runs; logs ~/r2_g2k_split4_abw.log, ~/r2_g8k_final.log, ~/r2_r100k_final.log). Decode untouched by construction (no decode kernels changed).

## The 500 statement (honest)

- The mission priced the rung-sum at 498 @2k-class (373.5 base). We land
  **374.0** — the two shipped rungs are worth ~-4.7ms/chunk (split4 -2.7
  measured by P15 + ab16w ~-1.5-2.5; the 2k last-half chunk med 171.4ms vs
  the P15-era ~176-178 in-class) = ~+2.5-3% expected, which cross-boot
  variance (+-5%, the P18 post-crash-boot law) swamps at the ladder level.
  The ladder numbers are the SAME CLEAN-BOOT CLASS as P17.
- The bulk of the rung-sum is NOT landed: **WY-C32 (-9-17ms, ~40% of the
  sum) is blocked on the c32 hi-lo kernel bug** (forensics above — the fix
  is a harness-resize + factor-level diag session); decode-r7 + M=128
  (-21-25ms, ~50%) are campaign-class per P7F3; qkv/oa m64 (-2-3ms) is one
  gate cycle.
- @100k: **247.6 = 37.4% of the 662 reference**. The P18 arithmetic stands
  unchanged: ~290-300 once WY+M=128 land; 400 needs the persistent-CTA /
  dynamic-smem(>48KB) attention redesign (session-class each).
- The honest remaining gap to 500 @2k: ~-55ms/chunk from ~171ms — the
  stacked rungs above (~-30-45ms) plus an attention-class build.

## Ship config (daemon relaunched and verified)

```
M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1
PVH=1 HM=1 M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256
PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1
NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1
PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=0 PF_ABW=1 PC_ENABLED=1
```
Kill-switches: PG_SPLIT=2 / PF_ABW=0 restore the P17 canonical
bit-identically; PF_SCANC=1 is INERT-but-broken (kernel bug) — leave 0.
Log: ~/w100k_serve_r2.log.

## Cache compat

- pcache._ENV_KEYS now includes PF_SCANC — the config fingerprint changes
  (new cache namespace, correct by construction: the WY scan changes GDN
  numerics). The ship line sets PF_SCANC=0 EXPLICITLY so the namespace is
  stable (unset vs "0" hash differently).
- The R1 ingest hooks are untouched (on_chunk fires in prefill_batch_m64
  at 64-boundaries — unchanged by PG_SPLIT/ABW/SCANC wiring).

## Files / commits

- engine0/pf_norms16.cu (+KSEL 7), engine0/pfk_ab16w.cubin,
  engine0/build_p7c.py (+nc2 targets), engine0/pf_scanchunk.cu (untouched),
  engine0/pf_prefill.py (SCANC wiring + buffers + probe + ABW + graph key),
  engine0/pcache.py (fingerprint).
- Logs: ~/r2_g8k_wy_split4.log, ~/r2_disc128.log, ~/r2_g2k_wy_nc1.log,
  ~/r2_g2k_wy_nc1b.log, ~/r2_g2k_split4_abw.log, + the final gates.
