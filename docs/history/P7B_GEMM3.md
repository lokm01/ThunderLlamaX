# P7-B — The W-Resident GEMM generation: repacked weights + M-grid + DBUF ring

Status: **BUILT, VALIDATED BIT-IDENTICAL, BENCHED, AND SMOKE-WIRED
(PF_GEMM3=1 in pf_fwd32, m32 drop-in tier). Every repacked kernel —
singles (ffn/iq3d/iq3o) and merged twins (gdnqg, attnqkvi3) at MT=32 and
MT=64 — is BIT-IDENTICAL to the shipped P6 M32 kernels on real weights
(full-tensor coverage, every q/sw/d word consumed). The offline repack
(pack_w7.py -> packed7/, 288 tensors, 9.3 GB) is a pure byte permutation
(inverse round-trip verified on every tensor). Gains vs classic controls
measured IN THE SAME SESSION: repack+DBUF = 1.10-1.14x at M=32; +M-grid
(MT=64) = 1.21-1.46x total; per-class amortized 248-375 GB/s (gdnqg m64
374.5, iq3d m64 325.3 @ 38.2% MFU, ffn m64 311.7 — ffn is smem-forced to a
thin 4-warp config, the known follow-up). Cross-session vs the P6 record:
ffn 1.46x, iq3d 1.65x, iq3o 1.82x per 32-row equivalent.**

## What was built

### 1. pack_w7.py -> packed7/ (the offline permutation)
- Layout per (8-row warp group, k-chunk of KCH=128): **32 units of 16B**,
  `unit[r*4+c] = {q u16 x NCL (cc order), sw u32, d u16, pad 2B}` — each
  lane owns its WHOLE decode input set for the chunk; the warp's chunk slice
  is 32 consecutive uint4 = **the P7-A 795 GB/s pattern** (strided stage_w =
  284). Words are moved UNCHANGED (u16/u32 verbatim) — decode math and
  per-row k-order untouched.
- Repacked/original byte ratio = 512/392 = **1.3061x** (the 2B pad per unit) —
  the DRAM cost of lane ownership; effective stream ceiling ~607 GB/s.
- Classes: fg/fu/fd (64 each), gate (48), out-iq3 (24), q-iq3 (9 of 16 —
  the other 7 q are Q6_K-packed 4240B rows, auto-skipped), k (16).
  Round-trip inverse verified per tensor (full compare).
- **Mixed-format law**: packed/q*.npy contains BOTH IQ3 (1960B rows) and
  Q6_K (4240B rows) tensors — check rowbytes before repacking.

### 2. pf_gemm3.cu (single class) + pf_gemm3m.cu (merged twins)
- **REPACK=1 body** (IQ3 only): DBUF register ring — the next chunk's unit
  (ONE uint4 per lane, +1 for FFN's second plane) and the next x tile
  prefetch into regs while mma runs; decode (dq_iq3_r VERBATIM from P4) in
  the post-mma bubble; ws/xs staging and the fragment map unchanged.
- **M joins the grid**: grid = (NDIM/NTILE) x (Mrows/MTILE); mb =
  blockIdx.x / NGRID (compile-time NGRID); x/out/res rows offset by
  mb*MTILE. At MTILE=32 the launch shape is IDENTICAL to P6 (drop-in).
- **REPACK=0 body**: classic stage_w VERBATIM (the attribution control).
- Twins: per-segment repack flags (gdnqg = q5-classic + gate-r7; attnqkvi3
  = q-r7 + k-r7 + v-q4k-classic; attnqkvq6 = q6-classic + k-r7 + v-classic),
  M-grid folded into flat blockIdx over TGRID = sum of per-seg grids.
- Epilogues: plain fp16 / RES fp32 / FFN silu-mul (verbatim).

### 3. Build (build_p7b.py, 16 cubins, all symbol-checked)
pfg3_{ffn,iq3d,iq3o}_r7_m32_nw8k128 (drop-in tier); _m64_ (ffn nw4/NTILE=32
— the only smem-legal 64-row FFN shape; iq3d/iq3o nw8/NTILE=64);
pfg3_ffn_cl_m32 / iq3d_cl_m64 (classic controls);
pfg3m_{gdnqg,attnqkvi3,attnqkvq6}_r7_m{32,64}. Regs 80-180, no meaningful
spill (ffn m64nw4 = 255 regs + 48B spill — at the edge, see follow-up).

## Gates (test_p7b.py — ALL BIT-IDENTICAL)

| kernel | vs P6-m32 (rows 0-63) |
|---|---|
| pfg3_ffn_r7_m32 / m64nw4 / cl_m32 | BIT-IDENTICAL x3 |
| pfg3_iq3d_r7_m32 / m64 / cl_m64 | BIT-IDENTICAL x3 |
| pfg3_iq3o_r7_m32 / m64 | BIT-IDENTICAL x2 |
| pfg3m_gdnqg_r7_m32 / m64 | BIT-IDENTICAL x2 |
| pfg3m_attnqkvi3_r7_m32 / m64 | BIT-IDENTICAL x2 |

## Bench (synced min-of-10, this session; classic controls same-session)

| class | config | ms | wGB/s | DRAM GB/s | amort GB/s | TFLOPS | MFU |
|---|---|---|---|---|---|---|---|
| ffn | cl m32 (control) | 0.529/32r | 129.0 | 129.0 | 258.1 | 10.79 | 15.1% |
| ffn | r7 m32 | 0.481/32r | 141.8 | 185.2 | 283.6 | 11.85 | 16.6% |
| ffn | r7 m64 nw4 | 0.876/64r | 77.9 | 101.8 | 311.7 | 13.03 | 18.3% |
| iq3d | r7 m32 | 0.263/32r | 129.9 | 169.6 | 259.8 | 21.71 | 30.5% |
| iq3d | r7 m64 nw8 | 0.420/64r | 81.3 | 106.2 | **325.3** | **27.19** | **38.2%** |
| iq3d | cl m64 (control) | 0.479/64r | 71.3 | 71.3 | 285.2 | 23.84 | 33.5% |
| iq3o | r7 m32 | 0.122/32r | 98.6 | 128.8 | 197.2 | 16.49 | 23.2% |
| iq3o | r7 m64 | 0.194/64r | 62.2 | 81.3 | 248.8 | 20.80 | 29.2% |
| gdnqg twin | r7 m32 nw16 | 0.309/32r | 155.5 | 167.4 | 311.0 | 17.36 | 24.4% |
| gdnqg twin | r7 m64 nw8 | 0.514/64r | 93.6 | 100.8 | **374.5** | 20.90 | 29.4% |
| attnqkvi3 twin | r7 m32 | 0.278/32r | 104.3 | 133.0 | 208.6 | 16.87 | 23.7% |
| attnqkvi3 twin | r7 m64 | 0.380/64r | 76.4 | 97.4 | 305.4 | 24.70 | 34.7% |

(wGB/s counts ORIGINAL weight bytes; DRAM counts the 1.3061x repacked
traffic actually moved; amort = (rows/16)*wb/T, the P6 metric.)

**In-session attribution**: repack+DBUF at fixed M = **1.10-1.14x**;
M64 over M32 = **1.09-1.26x**; combined gemm3-m64 vs classic-m32 =
**1.21-1.46x**. Cross-session vs the P6 record (0.638/0.346/0.177): ffn
**1.46x**, iq3d **1.65x**, iq3o **1.82x** per 32-row equivalent (the
classic control also drifted 0.638 -> 0.529 today — machine-state class,
known from P5; in-session deltas are the honest ones).

## The honest reading vs the ≥380 GB/s / ≥40% MFU target

- **The family is at 248-375 GB/s amortized, 18-38% MFU** — gdnqg m64
  374.5 (at the line), iq3d 325/38.2%, attn 305/34.7%. NOT met uniformly.
- The binding constraint has MOVED: at M=64 with the repacked stream the
  kernels are no longer load-stream-bound (DRAM 100-186 GB/s actual vs the
  795 pattern ceiling) — they are **mma-latency/occupancy bound**: the
  accumulator chains are MTILE/16 per plane (4 plain / 8 FFN at MT=64) vs
  the m16n8k16 dependent-latency depth; the FFN m64 is forced to 4 warps
  (128 thr) by the smem law (MTILE*136 + 2*NTILE*136 <= ~46K halfs-bytes)
  and pays for it (18.3% MFU, 255 regs + spill).
- **The sweep grid collapsed by law** (documented, not skipped silently):
  KCH=96 -> chunks cross 256k block boundaries (the per-lane unit would
  need 2 sw + 2 d = does not fit 16B); KCH=64 -> the unit payload is 10/16B
  = 2.61x disk/traffic blowup (worse than KCH=128's 1.31x) unless a
  26B/32B block-unit layout is added; NTHR=512/1024 at KCH=128 -> smem
  over the 48KB static limit; NTILE=48 -> 17408%48 != 0 (the m64nw6 ffn
  build was INVALID and removed — grid 363 vs NGRID 362 reads x OOB).

## Super-chunk projection (per 256-token chunk, M=64 tier, this session)

Per block (GDN-class): gdnqg 0.514 + out 0.194 + ffn 0.876 + down 0.420 =
2.00 ms; per attn block: attnqkv 0.380 + o (iq3s, P6 ~0.2 est) + ffn/down
1.30 = ~1.9 ms. 48 GDN + 16 attn = **~127 ms of GEMM per 256 tokens** vs
P6-classic ~178 ms (same-session controls) = **1.40x**; at 512 tokens the
M-grid second pass adds the same again (W re-read per M-block — the
"W-resident" budget at M=64 is the mma side now, see above).
**MEASURED SMOKE (~/p7b_fwd32e.log)**: PF_GEMM3=1 PF_G3_BLOCKS=16 (70
tensors swapped, VRAM headroom forced 16 not 32) + PF_G3_SKIPREF=1:
**fwd32 chunk 128.5 ms/32 tok = 249.1 tok/s projected** vs the P6 record
130.6 ms (245.0) with only 16/64 blocks on gemm3 — the ~2.1 ms/chunk delta
matches the standalone per-block repack gains; full-64-block extrapolation
~113.8 ms/32 = **~281 tok/s @2k-class**. NOTE: the in-run merged m32chk
(gdnqg/attnqkvi3 vs the shipped M16 kernels, real engine weights) passed
BIT-IDENTICAL before the timing; the T=1 reference path faulted (channel-
hang class, the P5 fwd16 cousin — machine took a watchdog reboot mid-smoke;
gates were skipped via PF_G3_SKIPREF, covered standalone).

## New laws banked (P7-B)

1. **THE XPR CHUNK-OFFSET TRAP**: a prefetch macro parameterized by chunk
   INDEX must multiply by KCH before pointer arithmetic — passing the bare
   index = a 2-14 BYTE offset on an 8B-aligned uint2 load = "Misaligned
   Address" on EVERY SM (the deterministic all-SM signature of a
   systematic misalignment, not an OOB).
2. **NTILE must divide NDIM when the M-grid folds into flat blockIdx**
   (NGRID = NDIM/NTILE integer-divides; a non-divisor launch grid reads x
   OOB on the phantom extra CTA — 17408%48 != 0 killed the nw6 ffn m64).
3. **The packed/q*.npy set is MIXED-FORMAT** (IQ3 1960B + Q6_K 4240B rows)
   — any repacker must check rowbytes per file (7 of 16 q tensors skipped).
4. **Unit-layout padding economics**: lane-owned 16B units cost
   16/payload per chunk (KCH=128: 14/16 = 1.306x traffic; KCH=64 would be
   2.61x) — the layout buys the 795-pattern stream at a known DRAM tax;
   report DRAM GB/s separately from weight GB/s.
5. **M=64 moves the wall to mma occupancy**: with the repacked stream,
   DRAM sits at 100-186 GB/s (far under the 795 pattern) while MFU caps at
   chains-per-SM — more warps (nw8/nw16) beat more rows per warp; the FFN
   m64 smem corner (2 ws planes) forces nw4 and halves the win. The
   escape routes: 26B/32B block-units at KCH=64 (FFN smem relief), or
   MTILE=48 (47872B, eager-only), or split-plane FFN (two plain m64 +
   silu-mul epilogue kernel — value-identical, extra launch).
6. **Session drift is real**: classic ffn m32 ran 0.529 today vs 0.638 in
   the P6 session — ALWAYS bench a same-session classic control (the
   in-session repack delta 1.10x is the truth; the 1.46x cross-session
   number is optimistic).

## Files (engine0/)

pack_w7.py, packed7/ (288 tensors, 9.3GB), pf_gemm3.cu, pf_gemm3m.cu,
build_p7b.py (16 cubins: pfg3_*_r7_m{32,64}_*, pfg3_ffn_cl_m32,
pfg3_iq3d_cl_m64, pfg3m_* twins), test_p7b.py (gates+bench),
pf_fwd32.py (+PF_GEMM3/PF_G3_BLOCKS smoke wiring). Logs: ~/p7b_pack.log,
~/p7b_test5.log (the clean gate+bench run), ~/p7b_fwd32.log (smoke).

## P7-c follow-ups (ordered)

1. FFN m64 shape escape (law 5): KCH=64 block-unit layout (26B/32B) or
   MTILE=48 nw8 or split-plane — target ffn 311 -> 380+ amort.
2. q5/q6/q4k repacked layouts (the classic segments in the twins + qkv):
   per-(group,block) [lc][r][lo8|qh8] runs — the remaining ~22% of block
   GEMM bytes still on strided staging.
3. iq3s o-proj repack (raw 110B blocks).
4. M-grid super-chunk assembly (P7e): 256/512-row chunks with the m64 tier
   + norms fused into GEMM prologues (the P6 recommendation, unchanged).
