# P5 — FFN wave-structure + pfa16 DBUF: the falsification campaign

Status: **BOTH P4 remaining-work items BUILT, VALIDATED, and FALSIFIED as
binding constraints. The FFN/IQ3 pGEMM family and pfa16 attention are NOT
decode-issue-bound, NOT sync-bound, and NOT wave-quantization-bound on this
dext at M=16 — 13 validated kernel variants all land within 0.85-1.05x of the
shipped kernels. Engine runtime state is BYTE-IDENTICAL to P4 (no production
cubin changed; the pfa16 DBUF rebuild is standalone-certified bit-identical
and parked). The ladder stands at P4: 197.5 tok/s projected @2k-class,
148.4 @8k, 134.1 @100k rebuild vs 662 reference.**

## Item 1 — FFN restructure (the big one): negative, rigorously

Built in `pf_gemm.cu` (all behind `-D`, default builds byte-identical to P1/P4):
- **SYNCW=1**: xs double-buffered + register-prefetched; W stage is warp-local
  (each warp stages and mma's only its own rows) -> ONE full `__syncthreads`
  per k-chunk + one `__syncwarp` (classic needs two full syncs).
- **SUB=n**: multi-subtile warps (NTILE = NWARP*8*SUB) -> fatter CTAs at 1-2
  waves (68/136 CTAs), a-frag smem reads amortize xSUB x 2 planes.
- **H2=1**: IQ3 decode via two 4KB LUTs (appended last arg, args-after-last-up
  law): [0..2048) grid as half2 pairs, [2048..4096) 128 sign-XOR masks (uint4
  per 7-bit sidx, element-7 = parity). w = HMUL2(half2(db), grid_h2) XOR mask:
  4 HMUL2 + 4 XOR + 2 STS.64 per 8 weights vs classic 16 FMUL + 8 F2H +
  8 STS.16 + 7-XOR parity chain ~= 3x fewer decode instructions. The IQ3_XXS
  grid is EXACTLY representable in fp16 (verified) -> only rounding-point
  moves: round(db) then exact dyadic product vs round(db*g) — measured F 7e-4.

Results (standalone, synced min-of-10, real packed weights, vs shipped):
| variant | ffn | vs shipped | note |
|---|---|---|---|
| shipped pfg_ffn_hm_nw8k128 | 0.444-0.465 ms | 1.00x | 147-154 GB/s (bench_sweep: 0.407/167) |
| sw (SYNCW only) | 0.535 | 0.87x | bit-identical (nz=0) |
| h2sw nw8k128 | 0.507 | 0.92x | F 7.0e-4 PASS |
| h2sw nw8k64s2 (136 CTAs) | 0.683 | 0.68x | wider tile LOSES |
| h2sw nw8k32s4 (68 CTAs, 1 wave) | 1.892 | 0.25x | much worse |
| h2sw nw32k32 (68 CTAs, 1024thr) | 1.801 | 0.26x | much worse |
| h2sw nw16k64 (136 CTAs, 512thr) | 0.705 | 0.66x | P1's nw16k64 = 0.541 agrees |
| **h2 (classic + H2, no SYNCW)** | **0.468** | **0.99x** | **the decode-only ablation: NEUTRAL** |
| iq3d h2 / h2sw | 0.283 / 0.302 | 0.96x / 0.91x | F 9.5e-5 PASS |
| iq3q h2 | 0.240 | 0.95x | F 4.4e-4 PASS |
| iq3o h2 | 0.141 | 0.97x | F 4.5e-4 PASS |

**Verdict**: a 3x cut in decode instructions (h2-classic) moves nothing; the
sync halving (sw) moves nothing positive; the wave restructure (68/136-CTA
fat CTAs — the P4 doc's own proposal) is strongly negative. Utilization math:
17408-row N-tiling gives ~83% wave occupancy at EVERY divisor (272/328 =
68/82 = 136/164), so waves were never the loss. P4's DBUF (latency) already
falsified latency-hiding. What remains as the binding constraint: the raw
quant-word load stream itself (strided u16/u32 across 8 rows x quarters)
against the dext's per-SM streaming ceiling at 1 CTA/SM x 256thr = 8 warps.
**The >=300 GB/s FFN target needs a different generation: M=32 rows/CTA
(doubles arithmetic per weight byte), dynamic-smem >48KB launches (if the
dext exposes the opt-in), or persistent CTAs with cross-instance W reuse.
NOT another pass at tiling/decode/sync of the current template.**

## Item 2 — pfa16 @100k DBUF: bit-identical, cost-neutral, parked

`pf_attn.cu` K1 restructured: 4 `__syncthreads` per 32-key tile (was 6) +
K(t+1)/V(t) raw-word register prefetch issued right after barrier (A) so both
load streams hide under QK/owners/PV. NO arithmetic-order change.

Standalone A/B at end-of-100k splits (pos=CTXK-16, S=32, CTXK=100352):
- **pm/ps/pA BIT-IDENTICAL to the shipped kernel (0/12288, 0/12288,
  0/3145728 mismatches)** — the restructure is provably value-preserving.
- Timing 2.910 vs 2.922 ms = **1.00x** (16-layer chunk attn ~46.6-46.8 ms).
  Sync count and exposed load latency are NOT what binds pfa16 either; at
  ~70 GB/s on a fully-coalesced KV stream with ~1us of identifiable ALU/mma
  work per ~19us tile, the binding constraint is again the dext per-SM
  streaming/occupancy ceiling (128 CTAs = 1.56 waves at 1 CTA/SM).
- The rebuilt cubin is **PARKED**: shipped bytes restored (md5-verified);
  to enable: rebuild `pfa16nw32_s32_100k.cubin` from the current pf_attn.cu
  (build line in build_pf2.py comments) — it is a zero-risk swap certified
  bit-identical, but end-to-end gates could not be re-run this session (see
  the machine note) so the proven bytes stay.

## The machine note (important for the next session)

After an early harness OOB (my test_p5a sc buffer sized 8*CTXK instead of
64*CTXK halfs — the sc group stride is in HALF units), pf_fwd16.py developed
a DETERMINISTIC device fault at line 175 (`P.down` of the 2x205MB kv buffers
after the T=1 reference trunk) that SURVIVES TWO CLEAN REBOOTS and reproduces
with the ORIGINAL shipped cubins — i.e. NOT caused by any P5 kernel. Small
workloads run fine (T=1 trunk 16 tokens, test_p5a, all standalone benches).
The fwd16 kv-download fault class is the first open item for the next
session (suspect: dext large-DMA path wedged at a warm-reboot-surviving
level; may need a cold power cycle or the fault is input-position-dependent).
8k/100k gates were NOT re-run because they all route through that harness.

## New laws banked (P5)

1. **THE pGEMM WALL LAW**: at M=16, the pGEMM family's ~150-167 GB/s is
   invariant to decode instruction count (3x cut = neutral), sync structure
   (2->1 barriers = slightly worse), and tile plan (fatter CTAs = worse:
   KCH shrinks as NTILE grows and thinner per-chunk staging + more chunks
   dominates the wave saving). Stop tuning this template; the remaining
   routes are M=32, dynamic smem, persistent CTAs.
2. **Wave-utilization invariance**: for N=17408 and NTILE in {64,128,256},
   ceil-occupancy is ~83% at every divisor — "1-2 waves" is not a lever.
3. **H2 LUT decode is SAFE numerically**: IQ3_XXS grid exact in fp16; only
   the rounding POINT moves (F 7e-4 ffn / 9.5e-5 down) — reusable if a future
   structure ever becomes decode-bound (e.g., M=32 at KCH=64 where decode
   per mma doubles).
4. **Harness laws (re-learned)**: never build NVProgram objects inside a
   timed loop (~10ms/call); `wait=True` per-launch costs ~2ms fixed — always
   async launch + `dev.synchronize()`; the sc (KV-scale) group stride
   (CTXK*8) is in HALF units — the full sc buffer is 64*CTXK halfs, sizing it
   8x small is an OOB device fault.
5. **kill -9 of the parked serve daemon = the 6th data point for the
   shutdown->reboot law** (~9 min cycle, machine self-heals, Tailscale+ssd
   survive).

## Benchmark table (P5 = P4, unchanged; the falsification is the result)

| config | P3 | P4 | P5 |
|---|---|---|---|
| 2k-class chunk (fwd16, pos0) | 83.4-108.9 ms | 81.0 ms | unchanged |
| 2k-class projected prefill | 157.6 | 197.5 tok/s | unchanged |
| 8k-class incl draft fill | ~105.6 | **148.4 tok/s** | unchanged |
| 100k rebuild incl fill | 129.7 | **134.1 tok/s** (134.8 ms/chunk end) | unchanged |
| ffn GB/s / attn @100k GB/s | 163 / ~80 | 163 / ~80 | 163-167 / ~70-80 (measured afresh) |

References: 662 tok/s @100k (vLLM W4A16 Marlin), 1-3k short-ctx Marlin.
MFU: GEMM family ~18-20% of the 71T tensor peak — unchanged.

## The honest parity statement (updated)

The 4.5x gap to 662 @100k decomposes (P4 numbers, P5-confirmed causes):
1. GEMM MFU ~60ms/chunk at 18-20% — P5 falsified every within-template fix;
   the dext-realistic route is M=32 (halves the weight-stream per token,
   doubles the per-byte arithmetic) or persistent CTAs; without one of those,
   ~150-165 GB/s IS the dext bar for this family (~50-60% of reference).
2. Attention @100k ~47ms/chunk at ~70 GB/s — NOT sync/latency (P5 DBUF
   falsification); needs more resident CTAs per layer (S-structure/CH
   retune) or a K1 layout with fewer, longer CTAs.
3. Norm/launch floor ~16ms — untouched (item 3 not attempted; budget).
4. fill_draft — done in P4.

## Files

- engine0/pf_gemm.cu (+SYNCW/SUB/H2, default paths byte-identical),
  pf_attn.cu (K1 DBUF 4-sync restructure, parked),
  build_p5.py, test_p5.py (validate+bench), test_p5a.py (pfa16 A/B),
  pfg_*X*_*.cubin (the 13 variants), pfa16nw32_s32_100k.p4.bak (shipped bytes).
- Logs: /tmp/p5_fwd16*.log (the machine fault), remote test output in session.
