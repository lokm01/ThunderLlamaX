# W2E: CROSS40 round B — int8-KV landed Tier-1-exact (35.56 tok/s); the attention "BW wall" re-attributed to the DOT phase; 40 NOT crossed

## TL;DR
Mission: >=40 tok/s @100k via (A) QMD-unchain, (B) unwired shaves, (C) int8-KV,
(D) draft-vocab. Outcome: **A is structurally VOID (proof below). C is BUILT,
validated, Tier-1 EXACT 60/60 deterministic, 60/60 token-identical to the fp16-KV
sequence, and lands +0.79 tok/s -> 35.56 (78.28 ms/cyc) — NEW CANONICAL**
(strictly better: faster + 3.1GB VRAM lighter). The mission's attention
21->11ms projection is REFUTED: the K1 kernel is NOT memory-bound — it is
dot-phase-bound (~0.84ms/launch of QK+PV that scales with positions x rows,
invariant to KV bytes). Three micro-optimizations of the dot phase were built
and measured; ALL neutral-to-worse (attribution below). B and D not attempted
(budget consumed by C's kernel work + the K1 floor investigation). Gap to 40:
78.28 -> ~68.3 ms/cyc needed = -10ms; ranked paths at the end.

## NEW CANONICAL (locked, gated)
env: `SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1`
(+ DEV=NV PATH/DOCKER_HOST as usual; `python -u test_w100k.py`, ~/snap100k).
| metric | W2D | W2E (KV8=1) |
|---|---|---|
| tok/s @100k (best of 2) | 34.77 | **35.56** |
| cycle | 78.60 ms | **78.28 ms** |
| phases | 5.62 / 71.27 / 1.17 | 5.55 / **70.80** / 1.16 |
| Tier-1 spec==T=1 | 60/60 x2 | **60/60 x2, deterministic** |
| vs fp16-KV T=1 sequence | — | **60/60 (zero greedy flips)** |
| engine vs spec_base_100k | 59/59 | 59/59 |
| alpha / tok-per-cyc | 0.867 / 2.73 | 0.892 / 2.78 |
| attention-layer VRAM | 16 x 411MB fp16 | 16 x 218MB (**-3.1GB**) |
Log: ~/w100k_kv8.log. NOTE: alpha/tok-per-cyc read 0.892/2.78 this run (same
machinery; the 60-token window's m-distribution — do not over-read the delta).

## Lever A — QMD-UNCHAIN: VOID (structural, no code needed)
The trunk ping-pongs xA/xB sequentially through all 64 blocks (trunk_w1c.py
`_build_seqs`: block i consumes block i-1's xout). Every attention layer's
KPRE/K1/K2S sits at a different trunk position with a REAL data dependency on
the previous block. The 16 layers' K1s are NOT mutually independent -> no
legal cross-layer unchain. Intra-layer, pre->a->c are dependent, and the
splits are already one kernel (4 groups x S=256 CTAs). The W2C 692 GB/s
"cross-launch overlap" only exists between INDEPENDENT launches — there are
none adjacent in the probe chain. Verified by reading the seq construction;
no env/patch written (nothing legal to unchain).

## Lever C — int8-KV: built, exact, and the honest attribution
### What was built (all poison-first validated, engine0/)
- **Format**: biased uint8 kv (`u = q + 128`, u in [1,255]) + per-(row,
  32-channel) fp16 scales `sc [2][4][CTXK][8]` (8 scales/row, syv-ai class).
  Biased encode is load-bearing for the fast dequant (below).
- **spk_preq.cu (KPREq)**: quantize-on-append. qw path BYTE-IDENTICAL to fp16
  KPRE (validated). K stored as fp16(ko) then quantized; V = vrow verbatim then
  quantized. Thread d owns ONE channel -> each WARP is exactly one 32-ch scale
  group -> warp butterfly max, scale = fp16(max/127), q = round(v/scale) clip.
- **spk_g4q.cu (K1 int8)**: same fat-CTA G4-NW32 structure; STAGE loads int2
  (8B/thread/row) + 1 fp16 scale, dequantizes IN REGISTERS to the SAME fp16
  smem layout -> downstream QK/PV math unchanged. Dequant = the **0x6400 PRMT
  trick**: `__byte_perm(w, 0x64646464, sel)` builds half2 pairs {0x6400|u0,
  0x6400|u1} = {1024+u} EXACTLY; HSUB2 by 1152 -> {u-128}; HMUL2 by scale =
  single rounding, bit-identical to float-mul-then-f2h. 4 PRMT + 8 H2 ops per
  8 values, zero I2F/F2FP.
- K2S combines unchanged (partials fp32). Kernels: spk_pre{1,3}q_100k,
  spk_g4nw32qa{1,3}_100k (64 regs, 32KB smem, no spill).
- **Engine wiring (KV8=1)**: trunk T=1 seq + eager, probe T=3 seq, draft chain
  (kv_d + sc_d also int8), restore paths quantize host-side; test_w100k
  quantizes the ~/snap100k KV at load and uses engine_t1_ref_kv8.npy.

### Validation (test_kv8.py, 100k dims, poison-first)
- K1 int8 vs K1 fp16 on identical content: pm/ps relerr max 8-9e-3, pA F-norm
  relerr 6.6e-3 (<=2e-2 storage-quant class ✓).
- KPREq: qw BIT-IDENTICAL; stored int8 dequant within scale/2 bound (0/2048
  out-of-bound per row); untouched rows untouched.
- 100k gate: Tier-1 60/60 x2 deterministic; fp16-sequence overlap 60/60.

### Why attention did NOT drop 21 -> ~11ms (the real finding)
Standalone synced bench (bench_diag.py, POS=97810, 1024 CTAs):
| kernel | ms |
|---|---|
| spk_g4nw32a3_100k (fp16) | 1.315-1.46 |
| spk_g4nw32qa3_100k (int8) | **1.313-1.33** |
| int8, QK loop at 1/8 | 0.757 |
| int8, PV loop at 1/8 | 1.019 |
Decomposition: STAGE ~0.4ms (218MB at ~545 GB/s effective — memory is NOT the
wall!), QK ~0.56ms, PV ~0.28ms. The dot phase is IDENTICAL code in fp16/int8
and dominates -> halving KV bytes cannot help. The W2C "283 GB/s synced wall"
was the FULL kernel's dot-serialized average, not a DRAM limit.
Refuted dot-phase fixes (all built + measured):
1. PRMT 0x6400 dequant (12 ops/8 vals vs ~40): int8 time UNCHANGED (1.33).
2. K tile as FLOAT in smem (48KB: K-float + V-half; zero unpack in QK):
   **1.617 ms — WORSE** (smem traffic doubles; swizzle conflicts).
3. 4-accumulator QK (break the 256-FMA serial chain): **1.431 — WORSE**.
=> the dot phase is bound by an issue/latency mix at 1-CTA/SM that none of
the classic fixes touch. In-graph: probe 71.27 -> 70.80 (-0.47), draft
5.62 -> 5.55. VRAM: 16 x 411MB -> 16 x 218MB (-3.1GB).

## What remains to 40 (ranked, with the new attribution)
Gap: 78.28 -> ~68.3 ms/cyc (-10ms). Attention is ~21ms of dot-phase; the GEMV
pool ~38ms is issue-bound decode (W2D); other ~12ms.
1. **QK/PV dot-phase restructure** (the ONLY big attention lever left):
   (a) half2-accumulated QK: qw stored as HALF (KPRE change: qw16 buffers),
   QK = 1 LDS.128 + 1 LDG.128 + 4 HFMA2 per 8 MACs (~2.5x op cut + 2
   independent accumulator chains). Numerics: fp16 accumulation of 256
   products — Tier-2 class, gate decides. PV must stay fp32 (100k-term
   accumulation would overflow fp16).
   (b) 2-warps-per-row with cross-warp combine (doubles compute warps for
   T=1 where only 6/32 warps compute; for T=3 18/32).
2. **Lever B shaves** (~1-2ms): head8v4-style half2 head (368 GB/s class ->
   +8%?), aq3k8/aq6k8 half2 cores (+15% class) — the M=4 kernels exist as
   patterns (m4.cu), need M=3 ports.
3. **alpha program** (draft fidelity): unlocks K=3+ (machinery exists,
   Tier-1-proven; W2D showed alpha 0.867->0.608 at depth kills it).
4. Draft-vocab rebuild (Lever D, +0.5-1 tok/s class).

## Gotchas banked this session
- **THE NAME-TRAP STRUCK AGAIN (cost ~1h)**: spk_preq.cu's function is
  literally `KPRE` (copied from the substituted original) while I built with
  `-DKNAME=...` -> cubin symbol stayed "KPRE", loaded as TinyELF(name=...) ->
  SM "Illegal Instruction Encoding" on ALL SMs (GSP: Multiple Warp Errors).
  The original build_skv.py renames via `-DKPRE=<name>`. LAW: verify
  `cuobjdump -symbols <cubin> | grep STO_ENTRY` matches the TinyELF name
  BEFORE debugging kernel logic.
- **python-subprocess docker calls fail on this box (script mode only)**:
  build_kv8.py's `subprocess.run(["docker",...])` deterministically got
  "cannot connect to docker API" while the IDENTICAL call from `python -c`
  and from shell worked (same env dict). Workaround: run nvcc DIRECTLY from
  the ssh shell. Also: the nvcc shim needs ABSOLUTE .cu/output paths
  (relative paths -> "No such file" in cc1plus).
- Boot-hook gap: after reboot the cuda-nvcc-persistent container may exist
  but be stopped (`docker start cuda-nvcc-persistent`).
- 64KB static smem is over ptxas's 48KB limit; the K-float+V-half split
  (32+16KB) fits exactly but is SLOWER anyway.
- Engine arg lists: the extra `sc` buffer arg is inserted right after `kv`
  in KPRE/K1 signatures; combine (spk_c*) untouched; all call sites
  (trunk seq, trunk eager, probe seq, draft entries) patched consistently.
- pA element-wise relerr vs fp16 is meaningless under cancellation with
  random V (unnormalized partials scale with s); use per-256-row F-norm.
- The 2k gate remains invalid for attention changes (degenerate region);
  int8-KV kernels are 100k-only builds (assert in code).

## Files
- engine0/spk_preq.cu (+ spk_pre{1,3}q_100k.cubin) — KPRE int8 append
- engine0/spk_g4q.cu (+ spk_g4nw32qa{1,3}_100k.cubin) — K1 int8 (canonical)
- engine0/spk_g4qf.cu (K-float/V-half 48KB experiment, REFUTED, kept for record)
- engine0/spk_g4q4.cu (4-acc QK experiment, REFUTED, kept for record)
- engine0/build_kv8.py (NOTE: preq builds must use -DKPRE not -DKNAME)
- engine0/test_kv8.py (standalone validation), bench_diag.py (dot-phase
  decomposition bench), test_kv8/bisect/rebuild_test scratch removed.
- trunk_w1c.py / mtp.py / test_w100k.py: KV8=1 env-gated wiring.
- Log: ~/w100k_kv8.log.

## Run
cd ~/tinygrad-metal/engine0 && env SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352
GEMVV=1 KV8=1 DO_T1=0 DEV=NV PATH/DOCKER_HOST as usual; python -u test_w100k.py
(T=1 ref cached at ~/snap100k/engine_t1_ref_kv8.npy; DO_T1=1 regenerates.)
