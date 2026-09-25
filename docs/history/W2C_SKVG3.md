# W4/W2C: SKV-G3/G4 occupancy campaign — 33.30 tok/s @100k Tier-1 EXACT (+0.95)

## TL;DR
Goal was attention >=500 GB/s at pos=97810 (probe attention <=13ms, engine >=40).
What actually happened, in order:
1. The fork carveout formula was patched with a name-gated env override
   (NV_SMEM_CFG / NV_SMEM_CFG_NAMES). Forcing 100KB on G2 initially read
   "500.6 GB/s @S=128" — **an ARTIFACT**: the on-disk spk_g2a{1,3}_100k
   production pair was S=256-BAKED (rebuilt post-sweep in W2B; build_g.py's
   job list on disk still says S=128 — the SAME stale-script trap class as
   build_skv.py). Launching it at SKV_S=128 runs HALF the grid (groups 2-3
   never processed, partials stay poisoned). The "484.8/500.6" was 2x
   inflation; it also 0/60'd the first gate attempt of this session.
2. TRUE carveout result at S=256: 281.1 vs 291.6 baseline -> **the dext does
   NOT co-schedule multiple CTAs per SM** (1 CTA/SM hard, carveout override is
   a no-op that only shrinks L1). Occupancy route (a)/(c) of the mission is
   CLOSED on this driver.
3. G3 (P-in-registers via __shfl_sync broadcast + K xor-swizzle pitch 512,
   32.5KB smem, 3-CTA-capable): CORRECT (all contracts), real best 305.6
   pipelined / ~261-283 synced -> ~parity with G2. Shuffle-PV costs MIO
   pressure; LB3 scheduling recovered it to parity, not beyond.
4. G4 = fat-CTA (NW warps/CTA, generalized row->warp loop, RPMAX register
   sets): NW=32 (1024-thread CTAs, 64 regs) benched "692-709 GB/s pipelined"
   -> **283.2 GB/s SYNCED**. The pipelined number is cross-launch overlap:
   back-to-back independent launches of the same kernel RACE on this dext
   (they write the same pm/ps/pA; the bench never checks). **Synced-per-launch
   is the only honest rate.** (G2 synced 271.6 vs its pipelined 291.6 — the
   whole prior ladder's absolute GB/s are ~5-8% optimistic; rankings hold.)
5. Engine integrated G4-NW32 (S=256): **100k gate Tier-1 60/60 bit-exact,
   deterministic, stock 59/59, 33.30 tok/s (82.09 ms/cyc = draft 5.60 +
   probe 75.35 + accept 1.18)** vs W2B 32.35 (probe 77.25). The probe delta
   (-1.9ms) matches the synced K1 delta (23.4 -> ~21.5ms attention).

GATE >=40: NOT MET (33.30). Per mission: report + attribution, no further thrash.

## Sweep table (MODE=bench, CTXK=100352, pos=97810, REPS=30)
| kernel | S | CTA | pipelined GB/s | synced GB/s |
|---|---|---|---|---|
| spk_g2s256a3_100k (W2B ref) | 256 | 256thr | 288.9 | 271.6 |
| spk_g2a3_100k +carveout100 | "128"* | 256thr | 500.6* | — (*S=256-bake half-grid artifact) |
| spk_g2s256a3_100k +carveout100 | 256 | 256thr | 281.1 | — |
| spk_g3a3_100k (LB0) | 128 | 256thr | 238.9 | — |
| spk_g3l3s256a3_100k (LB3) | 256 | 256thr | 305.6 (w/carveout) / 261.2 | — |
| spk_g4nw8s256a3_100k | 256 | 256thr | 232.0 | — |
| spk_g4nw16s256a3_100k | 256 | 512thr | 286.2 | — |
| spk_g4nw24s256a3_100k | 256 | 768thr | 320.1 / 336.1 (carveout) | — |
| **spk_g4nw32s256a3_100k** | 256 | **1024thr** | 692.4 / 708.6 (carveout) | **283.2** |
| spk_g4nw32s128a3_100k | 128 | 1024thr | 673.2 / 708.9 | — |

Production pair: **spk_g4nw32a3_100k / spk_g4nw32a1_100k (S=256, NW=32) +
spk_c3g_100k / spk_c1g_100k**; 2k: spk_g4nw32a{1,3}_2k (S=32) + spk_c{1,3}.

## Validation (test_g.py MODE=py, CTXK=2304/pos=2000/S=32, poison-first)
- g3, g4nw16, g4nw32: relerr vs aattn3 **1.306e-04** each (gate 1e-3; same
  tile-softmax class as G2), numpy-fp64 2.611e-04, empty-split identity
  partials, rerun bitwise, qw1==qw3[0] and ao1==ao3[0] bitwise (T1/T3 contract).
- Row->warp remapping (G4's loop instead of G2's 3-macro unroll) is
  partials-bit-identical: rows are warp-independent, per-row op sequence
  unchanged (dot d ascending 32x8, butterfly 16,8,4,2,1, PV i ascending,
  mask (l<l1)&&(l<=pos+t)).

## 100k gate (test_w100k.py, ~/snap100k, SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352)
| metric | value | W2B reference |
|---|---|---|
| Tier-1 (spec K=2 == engine T=1 greedy) | **60/60 bit-exact, deterministic x2** | 60/60 |
| engine T=1 vs spec_base_100k (shifted) | **59/59** | 59/59 |
| tok/s (best of 2 timed reps) | **33.30** | 32.35 |
| cycle | **82.09 ms** | 84.50 |
| phases | draft 5.60 + probe 75.35 + accept 1.18 | 5.60/77.25/1.16 |
| alpha / tok-per-cyc | 0.867 / 2.73 | 0.867 / 2.73 |
2k gate: ~/w2k_g4nw32.log (SKV=1 SKV_K=g4nw32).

## Attribution (why 500+ is not reachable in this family, honestly measured)
- The wall is per-SM 1-CTA + phase serialization + ~280-320 GB/s synced DRAM
  streaming for ANY smem-staged split-KV variant on this dext.
- Fat CTAs (1024thr) pipeline the STAGE phase much better (pipelined 692 =
  2.4x G2) but that overlap only exists BETWEEN independent launches, not
  inside the dependency-chained probe graph. In-graph the gain is the synced
  delta: ~+4% (271.6 -> 283.2).
- Remaining routes to 40+: (1) K=3 (+3-4 tok/s at alpha 0.87 — still unbuilt:
  amds[4]/dring[2]/probe ROWS=4 trio/draft step-2, see W2_MTP.md); (2) GEMV
  polish (non-attn probe ~54ms dominates now); (3) int8-KV (Tier-2, halves
  bytes; the ONLY known >=2x attention lever left); (4) cross-launch overlap
  as a FEATURE: independent K1 launches (e.g. the 16 layers' K1s are mutually
  independent!) could be submitted unchained to exploit the 692-class
  overlap — needs graph restructuring (QMD dependency surgery), untested.

## Gotchas banked this session
- **STALE-BAKE TRAP (bit twice now)**: build_*.py job lists on disk do NOT
  necessarily describe the on-disk cubins (W2B rebuilt spk_g2a{1,3}_100k at
  S=256 post-sweep without updating build_g.py). NEVER bench/integrate a cubin
  by name without confirming its -D bakes (cuobjdump or rebuild). Symptom of a
  mismatched S: half-grid coverage -> poisoned partials -> 0/60 Tier-1 with
  garbage repeat-loop output, or 2x-inflated bench GB/s.
- **PIPELINED BENCHES LIE on this dext**: consecutive NVProgram launches of
  independent kernels OVERLAP on the device (no implicit serialization by
  buffer aliasing). Any `for rep: launch` + final-sync GB/s is optimistic
  (G2 +8%, NW32 +145%!). Synced-per-launch (wait=True) or in-graph timing
  only. This also re-scores history: K1S 188.6, G2 291.6 were pipelined.
- **The dext is 1 CTA/SM hard**: QMD min/target carveout override to 100KB
  (env now exists: NV_SMEM_CFG=100 NV_SMEM_CFG_NAMES=<substr,substr>) changed
  nothing at true S (281.1 vs 291.6) — multi-CTA co-residency does not happen;
  the override only shrinks L1 (28KB). Name-gated so other kernels keep their
  natural config; default OFF.
- 1024-thread CTAs work fine on this dext (64 regs x 1024 = exactly the
  register file; 1 barrier; graph QMD path accepted (1024,1,1) local_size —
  patched into gcycle.ParityGraph per-program by name substring).
- NW must divide 32 (STAGE iteration count TILE/NW); NW=32 -> RPMAX=1 row per
  warp (18 of 32 warps compute, all stage).
- test_g.py POS2 phase had a buffer-size bug (reused last-swept-S poison
  buffers for the best-S kernel -> OOB -> device fault); fixed (re-poison at
  best S). Also lsz() now maps nw16/nw24/nw32 names -> 512/768/1024 threads.

## Files
- engine0/spk_g3.cu (G3: P-registers + K-swizzle, 32.5KB), spk_g4.cu (fat-CTA,
  NW param), build_g3.py, build_g4.py; cubins spk_g3*, spk_g4* (+ S=128
  combines spk_c{1,3}g128_100k); test_g.py extended (g3/g4nw16/g4nw32 + lsz +
  POS2 fix); trunk_w1c.py / mtp.py: combine name by SKV_S + per-kernel local
  size; gcycle.py: per-program graph local size.
- Fork (~/tinygrad-src): ops_nv.py carveout env override (name-gated).
- Logs: ~/w100k_g4nw32.log (gate), ~/w2k_g4nw32.log (2k), ~/w100k_g2_carve128.log
  (the 0/60 stale-bake repro).

## Run
cd ~/tinygrad-metal/engine0 && env SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352
DO_T1=0 DEV=NV PATH/DOCKER_HOST as usual; python -u test_w100k.py
2k: SKV=1 SKV_K=g4nw32 (S=32 default) test_w2.py
