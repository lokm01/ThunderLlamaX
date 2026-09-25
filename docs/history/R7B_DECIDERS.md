# R7b — THE WARP-SPEC ROUND: prefill 503.6 → target 750 @2k

Status: **750 NOT crossed — and the reason is now MEASURED, not guessed. The
session shipped three decider-grade unlocks (the named-barrier QMD root cause
+ the fork patch; mbarrier DEAD on this dext; the warp-spec build itself,
bit-identical and perf-negative at every tested shape) plus the first full
per-class GEMM pool census at the true M128 shapes. No production prefill
change shipped (all cubins reverted to HEAD, gates re-banked); decode
production was found degraded in the daemon (LOOKUP_K=0 relaunch mistake) and
RESTORED to the R7a canonical (LOOKUP_K=8).**

## 1. THE PER-CLASS GEMM CENSUS (r7b_classes.py, true M128 shapes, min-of-8)

| class | per-launch | pool per 128-chunk | share of GEMM |
|---|---|---|---|
| ffn fg+fu (nw4 g=1088) | 1220.3 us | 58.6 ms | 37% |
| gdnqg qg (r7q4 g=512) | 755.7 us | 36.3 ms | 23% |
| iq3d fd (nw8 g=160) | 638.3 us | 30.6 ms | 19% |
| iq3o out (nw8 g=160) | 263.8 us | 12.7 ms | 8% |
| attnqkv i3+q6 (g=448) | ~770 us | 12.4 ms | 8% |
| o-proj 4xM32 (g=80) | 137.2 us | 8.8 ms | 5% |

- GEMM standalone-sum = **159 ms of the 256 ms 2k chunk (~62%)** — the 750
  arithmetic needs the GEMM pool at ~75 ms (a 2.1x on the whole family).
- qg standalone matches the banked R2d number exactly (755.7 vs 757.9) — the
  harness currency is consistent with history.
- Floor arithmetic: fd W 27 MB + X ~350 MB L2-class per launch → ~120-150 us
  floor vs 638 measured (4-5x off); ffn similar. The room is real; the
  mechanism to take it (below) is what failed.

## 2. RUNG 2a — the barrier smoke (r7b_mb.cu / r7b_mb_run.py): TWO LAWS

1. *** THE QMD BARRIER LAW ***: `bar.sync id>=1` FAULTS on this stack with SM
   "Illegal Instruction Parameter" on every SM → channel wedge. ROOT CAUSE =
   OUR OWN fork QMD: ops_nv.py built every program with `barrier_count=1`
   (only barrier 0 legal). FIX (fork commit 5227123): env-gated
   `NV_QMD_BARRIERS` (default 1 = canonical byte-identical) +
   `NV_QMD_BARRIERS_NAMES` name-scoping. With =16: bar.sync 1..15 are
   checksum-correct at full speed (665 GB/s pipeline arm == the all-warp arm).
   The warp-spec primitive family is UNLOCKED for future kernels.
2. *** MBARRIER IS DEAD ON THIS DEXT ***: init/arrive/test_wait all EXECUTE
   (no fault) but phases NEVER COMPLETE — both producer and consumer
   bounded-spins time out, checksum 0. sm_86 details: the parity operand must
   be a compile-time immediate (register forms are sm_90+; try_wait entirely
   sm_90+). Do not build mbarrier pipelines on this rig; named-barrier MEETS
   are the only subset-sync primitive.
3. (harness) bounded spins are mandatory — the first nbar fault and both
   mbarrier timeouts exited cleanly with flag writes, no reboot.

## 3. RUNG 2b — the warp-spec build (pf_gemm3w.cu): BIT-IDENTICAL, PERF-NEGATIVE

Design (v3): NCONS consumer warps run MMAR VERBATIM per K=64 stage in k order
+ own W-unit register ring + post-mma decode (the base DBUF pattern);
NPROD producer warps continuously stage the X stream (uint4, static ring-2,
loads issued BEFORE the stage meet so DRAM latency hides under consumer mma).
NS=2 smem buffers = 36864 B (in-graph legal). Build: 80-168 regs, 0 spill,
2 barriers, one kernel per cubin (names pfg3w_*_ws{6,10}p{2,3}k64).

- **Correctness: BIT-IDENTICAL det-x2 nz=0 on real weights, all classes**
  (ffn/fd/out). The stage mapping law that makes it work: the base writes
  k-in-chunk at (qc_*4+cc)*8, so a K=64 stage covers lanes qc_ ∈ {2h, 2h+1}
  with ALL 4 cc — chunk-boundary decode writes both future buffers, no warp
  divergence.
- **Perf (true shapes, min-of-8)**: ffn x0.96, fd x0.82 (p2) / x0.90 (p3),
  out x0.87. **NEGATIVE at every shape.**
- VERDICT + LAW: with meet-only handoff (mbarrier dead), the per-K64-stage
  full-CTA meet couples producer smem stores with consumer decode on the
  critical path; the base's sync-pair + register DBUF ring keeps MORE loads
  in flight across barriers than the warp-spec meet schedule. The E1
  "1.8-1.9x latency headroom" does not convert through meet barriers at this
  stage granularity. PARKED — do not retry meet-based warp-spec on the
  pf_gemm3 family; a future retry needs either (a) subset signaling that
  actually works on this dext, or (b) a persistent-CTA megakernel where the
  pipeline lives inside one CTA without meets.

## 4. X-uint4 (the audit's "free" X-wide scrap): BANKED NEGATIVE

uint2→uint4 on all X staging in pf_gemm3/3m (sources kept as
*.x4banked): bit-identical on 5/6 classes but **attnqkvi3 BREAKS (full-output
diff, flags matched p7b exactly)** and perf REGRESSES the nw8 classes 10-22%
(fd x0.78, out x0.81, qg x0.89, qkvq6 x0.84; ffn nw4 neutral x1.01). LAW:
**on latency-bound prefill GEMMs, halving the X-load instruction count by
doubling the width is NEGATIVE — the 2x-more-numerous uint2 stream keeps more
independent loads in flight.** Load-latency structure rewards issue diversity,
not width (consistent with the R7a decode-side "merge is neutral" finding,
stronger here). Reverted; do not retry blind.

## 5. Priced remainders (not built this session, timebox)

- qg packed5 (Q5_K qkv-seg repack): ~8b/w plane, 2-3 wide loads/lane/chunk vs
  ~16 U8 planes; est 1.1-1.25x on a 36 ms pool → -4-8 ms/chunk. VRAM +2.2 GB
  both-live (raw stays for decode q5g8) — needs the headroom check.
- o-proj 4xM32 → one M-grid-folded launch: pf_gemm.cu M32 body has NO mb
  fold — needs the g=320 surgery (R2d scrap-2 pattern); ~-5 ms/chunk.
- The ffn nw4 46 GB/s W-rate (vs iq3d/out nw8) is the single worst
  utilization in the census; NTILE=48/96 warp-splits are grid-illegal
  (17408 % 48 != 0) — an ffn-specific structure (e.g. KDIM-split K=2560
  halves at nw8) is the open idea.

## 6. Gates (final config = R2d canonical, unchanged; READOUT-ORDER this boot)

- 2k (r7b_g2k.log): **501.8 tok/s** PF_BATCH (bank 503.6, same class), F
  9.408e-04 EXACTLY, A 59/60, CTRL 17/60, D 18/60, D2 exact — line-for-line.
- Tier-1 decode @ LOOKUP_K=8 (r7a_gate_r7b_t1.log): **72.02 tok/s BEST**
  (bank 71.51), alpha 3.242 EXACT, E[m|deep] 8.000 (94/94, 78.3% deep-selected),
  deep=off 41.95, stock 59/59, det-x2, tier2 x2 — the fork barrier patch is
  decode/prefill NEUTRAL (default-1 byte-identical path exercised throughout).
- 8k (r7b_g8k.log): **458.7 tok/s** (bank 457.8), F 4.284e-02 floor EXACTLY,
  A 9/60, CTRL 13/60, D 6/60, D2 0-mismatch — line-for-line.
- 100k rebuild (r7b_r100k.log): **317.7 tok/s** (bank 317.6 EXACTLY the class),
  cur=4471 match=True, rebuilt-state T1 decode 60/60.

## 7. Daemon (production) + THE FRESH-PATH STALE-FEED BUG (P0, next session)

- FOUND DEGRADED at session start: the 09-22 post-crash relaunch ran with
  **LOOKUP_K=0** (ctl script mistake — K=8 deep decode OFF) AND a dead dext
  channel (BrokenPipe keepalives). ctl script repaired to the full canonical:
  LOOKUP_K=8 + PG_SPLIT=4 + PC_ENABLED=1 (the latter two were ALSO missing
  from the ctl — the R2d ship line).
- *** THE FRESH-PATH STALE-FEED BUG ***: after restore, the FRESH chat smoke
  answers the PREVIOUS request prompt, deterministically one-request-stale
  (the first smoke after boot answers some boot-era "!!!" content). Reproduced
  with PC_ENABLED=0 (pcache exonerated). The encode path is verified CORRECT
  (render+encode decodes to the exact message; the fed COUNT matches). The
  signature points at the win_up(ids128/ids64/pos_arr) host-window writes
  RACING the async chunk-graph submit (the graph reads the previous content of
  the ids buffer — a cross-channel/ordering issue, plausibly exposed by an
  R5d/R7a-era reordering; last known-green live smoke = R2d era, LOOKUP_K=7).
  Engine numerics are NOT implicated (Tier-1 60/60 x2 + stock 59/59 this boot;
  api_gates (a) greedy-exact internally consistent). NEXT SESSION P0: bisect
  LOOKUP_K 8 vs 7 vs 0 on the serve path; add an explicit device-side ordering
  fence (or move the ids to a device-side copyin) between the ids win_up and
  _pf_submit_chunk in BOTH prefill_batch loops (m64 + m128 paths).
- Posture shipped tonight: daemon on the full canonical env (LOOKUP_K=8),
  health/heartbeat verified; the fresh-path bug is PRE-EXISTING (the code
  state is the R7a tree + the two R7a-era serving fixes, unchanged by R7b)
  and documented here as the known issue.

## 8. Files / logs

- engine0: r7b_mb.cu + r7b_mb_run.py (the smoke), pf_gemm3w.cu + build_r7b.py
  (the warp-spec family + p3 variant), r7b_classes.py (the census),
  r7b_wspec.py (A/B), r7b_x4ab.py + build_x4.py + *.x4banked (the negative).
- Logs: ~/r7b_mb_smoke.log, ~/r7b_classes.log, ~/r7b_wspec{,2,3,4}.log,
  ~/r7b_x4ab.log, ~/r7b_g2k.log, ~/r7b_t1.log, ~/r7b_g8k.log, ~/r7b_r100k.log.
- Fork: commit 5227123 (the QMD barrier patch).
