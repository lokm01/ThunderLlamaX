# W4 SKV-G: smem-staged split-KV attention — G2 291.6 GB/s (1.53x), 100k gate below

## TL;DR
The spec's root-cause model (8x load amplification saturating a per-SM load pipe)
is REFUTED by the G1 discriminator: the redundant loads were L1 hits, not DRAM.
G2 (lane-per-l full-private dots from smem, tile softmax, PV from smem) is CORRECT
(Tier-1 contracts pass) and reaches **291.6 GB/s @ pos=97810 (G2-S256)** vs 188.6
for the W3 K1S — but far below the 600 target. The binding constraint is
**occupancy**: 36.4KB smem/CTA -> the fork's carveout formula (min cfg >= usage,
steps 32/64/100KB) -> **1 CTA/SM**, so DRAM staging and smem compute phases never
overlap. PF (register prefetch) = 279.8 (no gain), D2 = invalid (see T-law).
100k engine gate: see bottom. Both G1 and G2 < 350 -> per mission: report, no
further thrash.

## Sweep (MODE=bench, CTXK=100352, pos=97810, REPS=30, wall-clock)
| kernel | S | TILE | GB/s | note |
|---|---|---|---|---|
| spk_g1s64a3_100k | 64 | 32 | 110.4 | discriminator: WORSE than K1S 190 |
| spk_g1s128a3_100k | 128 | 32 | 124.5 | staging+barriers cost > L1-hit savings |
| spk_g2s32a3_100k | 32 | 32 | 251.9 | |
| spk_g2s64a3_100k | 64 | 32 | 252.7 | |
| spk_g2a3_100k | 128 | 32 | 281.3 | |
| spk_g2pfa3_100k | 128 | 32 | 279.8 | PF ring: no gain (loads not the phase gap) |
| spk_g2s256a3_100k | 256 | 32 | **291.6** | BEST (291.6 on re-run) |
| spk_g2t16s64a3_100k | 64 | 16 | 143.5 | INVALID math (see TILE law) |
| spk_g2d2a3_100k | 128 | 16 | 170.3 | INVALID math (TILE=16) + slower |
| W3 K1S reference | 128 | - | 188.6 | per W2_100K.md |

Best-config pos curve (spk_g2s256a3_100k): pos=97810 -> 291.6; pos=32768 ->
94.3; pos=2000 -> 5.8 (S=256 grid is pos-insensitive ~1.4ms floor; the 100k
production point is the only one that matters).

## Validation (test_g.py MODE=py, CTXK=2304/pos=2000/S=32, poison-first)
- G1: relerr vs aattn3 **2.551e-07** (gate 1e-3) — identical class to the W3
  trio (per-l op sequence verbatim); numpy-fp64 2.611e-04; empty splits (4/32)
  identity partials; rerun bitwise; qw1==qw3[0] and ao1==aoB[0] bitwise.
- G2: relerr vs aattn3 **1.306e-04** (gate 1e-3; tile-softmax reassociation —
  expected ~1e-6, got 1.3e-04, still 7.7x under gate); numpy-fp64 **2.611e-04 =
  identical to the W3 trio's** (both equally close to fp64 truth); empty-split
  identity; deterministic; T1-vs-T3 bitwise (ao1==ao3[0], qw1==qw3[0]).
- T8/T16 builds: **discarded** — the lane-per-l design structurally REQUIRES
  TILE=32 (lanes map 1:1 onto tile rows); TILE<32 reads past the staged tile
  (T8 = smem OOB fault; T16 = in-bounds garbage = wrong math, runs silently).

## DEVIATIONS from the spec (forced by hardware/fork, both documented in-source)
1. **No SM_Q**: spec stages Q into smem (RMAX*256 fp32 = 18KB at T=3) ->
   53.5KB total > the 48KB STATIC __shared__ limit on sm_86; the fork has no
   dynamic-smem path (QMD shared size = cubin .nv.shared section, ops_nv.py
   'shmem_usage'); and 2x53.5KB > the 100KB sm_86 carveout makes the spec's
   "2 CTAs/SM" impossible as written. Q is read from global qw via 16B
   broadcast loads (same fp32 values verbatim, same address all lanes -> one
   L1 transaction; ~2/3 of QK smem-pipe ops but FMA-bound overall). All
   numerics contracts (tile 32 both builds, butterfly 16,8,4,2,1, PV i
   ascending, dot d ascending 0..255, (l<l1)&&(l<=pos+t)) kept -> T1/T3
   bitwise holds.
2. **T64 sweep point impossible** (69.6KB smem); T16/T8 substitutes invalid
   (TILE law above). The tile dimension is NOT sweepable in this design.

## Gotchas banked this session
- **Misaligned smem fault**: `SM + off + (warp*3+ri)*TILE + lane` on a char*
  array is a BYTE offset -> odd lanes = 4k+1 float store = "Misaligned
  Address" warp exception on EVERY SM. Scale by sizeof(float): index a
  `float*` view instead. (This was the G2 fault; __syncwarp is INNOCENT and
  works fine on this dext.)
- **spk_c S-bake clobber (pre-existing, fixed)**: the W3 commit's plain
  spk_c3/spk_c1 cubins were rebuilt S=128-baked at 04:33 (post-goal-run) while
  test_skv.py/2k paths use S=32 workspaces -> K2S reads pm[g*128*18+s2*18]
  far OOB -> "Device fault detected" at the FIRST c3 (looked exactly like a
  new-kernel bug; cost a reboot + false lead). Fixed: spk_c{1,3} = S=32 (2k/py),
  new spk_c{1,3}_100k = S=128 (old 100k path), spk_c{1,3}g_100k = S=256 (G2).
  Loaders (trunk_w1c.py, mtp.py) now suffix the combine by path and select the
  K1 family via **SKV_K env** ("a" = W3 K1S | "g2" = SKV-G).
- nvcc shim needs ABSOLUTE paths (container cwd != host cwd; relative paths =
  "spk_g.cu: No such file"). After a reboot, wait for colima docker (the
  PATH-less non-interactive shell has no docker/colima in PATH).
- Reboot clears the fault-wedge; a faulted dext poisons every later process
  (fresh processes fault at the first wait) — one attempt per boot.

## Files
- engine0/spk_g.cu (G1+G2+PF+D2 in one -D-parameterized body), build_g.py,
  test_g.py (MODE=py differential + MODE=bench sweep).
- Production pair: spk_g2a3_100k / spk_g2a1_100k (S=256) + spk_c3g_100k /
  spk_c1g_100k; 2k: spk_g2a{1,3}_2k (S=32) + plain spk_c{1,3} (S=32).
- Engine: SKV=1 SKV_K=g2 SKV_S=256 SKV_CTXK=100352 (100k) / SKV_K=g2 (2k,
  S=32 default).

## 100k gate (test_w100k.py, ~/snap100k bootstrap, SKV=1 SKV_K=g2 SKV_S=256)
| metric | value | W3 reference |
|---|---|---|
| Tier-1 (spec K=2 == engine T=1 greedy) | **60/60 bit-exact, deterministic x2** | 60/60 |
| engine T=1 vs spec_base_100k (shifted) | **59/59** | 59/59 |
| tok/s (best of 3 timed reps) | **32.35** | 29.63 (+9.2%) |
| cycle | **84.50 ms** (draft 5.60 + probe 77.25 + accept 1.16) | 93.93 (5.30/87.45/1.14) |
| T=1 engine reference @100k | 20.69 tok/s (48.32 ms/tok) | 16.37 |
| alpha / tok-per-cyc | 0.867 / 2.73 | 0.892 / 2.78 |
| GATE >= 40 tok/s | **NOT MET (32.35; also < 38 -> attribution per mission)** | 29.63 |

Attribution: probe 87.45 -> 77.25 (-10.2ms) matches attention 34.8ms ->
34.8*190/291.6 + ~0.7ms combine delta = ~23.4ms (S=256 pA read doubles).
Attention is no longer the #1 probe cost; non-attn probe ~54ms (weights-bound
GEMVs) + attention ~23 is the new floor of this structure. 40 tok/s needs
either the <=16KB-smem redesign (see routes below), K=3 (probe T=4 +8ms is now
net-POSITIVE at 23ms attention: E[N]~3.3 at alpha 0.87 — worth ~+3-4 tok/s,
~35-36 total), or non-attention probe cuts. Log: engine0/w100k_g2_final.log.

## Attribution (why 600 is unreachable in THIS structure)
- G1 < K1S: the 8x warp-redundant loads were L1 hits; smem staging adds a
  round-trip + 2 barriers per tile for zero DRAM savings -> net loss.
- G2 ceiling: 1 CTA/SM (36.4KB smem; fork carveout min>=usage). Per tile the SM
  serializes [32KB DRAM stage] -> [QK+PV smem compute]; ~50% duty ceiling.
  PF proved load LATENCY is not the gap (prefetch ring changed nothing) — the
  barrier-serialized phase structure is.
- Routes that could break 600 (next session's problem, NOT tried per mission):
  (a) <=16KB smem/CTA for 2-3 CTAs/SM (needs TILE=8-class structure, i.e.
  lane-per-l must go: 4 lanes per l-row x 8 rows/tile or a Q-in-registers
  d-split with ONE tree reduce per (row, tile) instead of per (row, l));
  (b) warp-specialized producer/consumer within one CTA (4 warps stage next
  tile into a double buffer while 4 warps compute) — needs the single-array
  smem layout enlarged to 2x(TILE*528+TILE*512) which at TILE=32 = 66.6KB >
  48KB static limit -> TILE=16 + the TILE law fix (lane map 2 l/16 lanes?);
  (c) int8 KV (halves bytes; orthogonal, Tier-2 baseline regen required).
