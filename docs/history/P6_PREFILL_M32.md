# P6 — Prefill M=32: the load-stream amortizer lands

Status: **M=32 GEMM generation BUILT, VALIDATED BIT-IDENTICAL, AND INTEGRATED
(env PF_M32=1, default-on). The 32-row chunk runs the six GEMM launches per
block at M=32 (one weight pass per 32 tokens) with the norms/pKPRE/pATTN/
pSCAN/draft-fill stages staying M=16 as 2x16 halves on offset views (the
seam). fwd32 chunk 130.6 ms/32 tok @pos0 = projected 245 tok/s @2k-class (P4:
81.0 ms/16 = 197.5). All correctness gates green: standalone + merged-kernel
outputs BIT-IDENTICAL to the shipped M=16 kernels; the 32-row forward is in
the SAME numeric class as the P3 M=16 gates (logits med 8.4e-4 / F 2.6e-3,
drift 2.2e-4 -> 1.4e-3).**

## The design (what M=32 is)

P5 proved the pGEMM family at M=16 is bound by the quant-word LOAD STREAM vs
the dext's hard 1-CTA/SM (decode/sync/wave/latency all falsified). M=32 is
the priced route: **two m16n8k16 mma fragments share every staged W tile** —
xs holds 32 rows, the b-fragments (the quant-word load+decode stream) are
read ONCE per k-chunk and feed row-groups 0..15 and 16..31. Arithmetic per
loaded weight byte doubles.

- `pf_gemm.cu` +`M32=1` classic-path section (HMMA only, no DBG/KS/DBUF/SYNCW
  combos): stage_x covers 32 rows, the mma loop adds a4..a7 (rows g+16/g+24),
  the epilogue writes m in {g, g+8, g+16, g+24}. stage_w/decode VERBATIM.
- `pf_gemm2.cu` +`M32=1` twin for the merged multi-segment kernels (segment
  select unchanged).
- **Per-row k-order is unchanged** (same fragment map per row) -> per-row
  outputs are BIT-IDENTICAL to the M=16 kernels on identical inputs (proven,
  see gates). K-order documentation: each output row's dot accumulates
  k-ascending in KCH chunks exactly as the M=16 build; M=32 does NOT change
  any row's reduction order.
- smem: `32*XS_LD + (FFN?2:1)*NTILE*WS_LD` halfs — nw8k128 plain 26112 B,
  FFN 43520 B, gdnqg (nw16 k128) 43520 B. All < 48 KB static; these kernels
  launch EAGERLY (the 36.8KB in-graph smem law does not apply; P4's shipped
  FFN already ran 39168 B eagerly).

## The seam (integration)

32-token chunks; per block the plan runs: norm/ab 2x16 halves -> ONE M32
merged GEMM (attnqkv or gdnqg) -> pKPRE/pATTN/pFC per half (half B gets
`pos_slot_b` = pos+16 and its own pm/ps/pA workspace) -> pfs16 scan per half
(sequential state advance = the T=1 semantics) -> M32 o/out GEMM -> hh 2x16
-> M32 FFN -> M32 down-RES. Draft fill: the P4 7-launch window runs PER HALF
(the REC ring alternates per half; seed row = prev half row 15 — verified
against pfk_rec16's recseed semantics); the draft GEMMs stay M=16 (the fill
window is ~1-2 ms/chunk — not the wall). Tail = N % 32 via T=1. Head on row
31. GDN live parity unchanged (32 is even). Files: `engine0/pf_prefill.py`
(PF_M32=1 default-on; PF_M32=0 restores the P4 M=16 plan byte-for-byte).

## Gates (all from the FIRST clean run — readout-order law)

| gate | result |
|---|---|
| standalone bit-identity (ffn/iq3d/iq3o, real packed weights) | **0 mismatches each — BIT-IDENTICAL to the shipped M=16 kernels** |
| standalone bench (synced min-of-10) | ffn M16x2 0.789 -> M32 0.638 ms (**1.24x**, 214 GB/s 32-tok-amortized); iq3d 0.485 -> 0.346 (**1.40x**, 197); iq3o 0.223 -> 0.177 (**1.25x**, 136) |
| merged-kernel bit-identity (gdnqg, attnqkvi3, in-harness) | **BIT-IDENTICAL** |
| fwd32 vs T=1 trunk 32 steps | logits med 8.389e-4 / F 2.617e-3 **PASS** (P3 M=16 class: 9.2e-4/2.6e-3); argmax 31/32; rec drift 2.2e-4 -> 1.4e-3 (M=16: 1.9e-4 -> 1.6e-3) |
| kv slice compare (first 128 rows) | 236/262144 + 389/262144 bytes (int8-quant near-tie class, informational) |
| serving gates (PF_GATE=1, 8k-class) | see P6 gate log — Gate A / CTRL / D / D2 |
| 100k rebuild (PF_GATE100K=1) | **cur rebuilt = 4471 EXACT MATCH**; drift max rec 3.45e-2/conv 6.0e-2 (accumulated class; blk0 6.7e-4); T1-decode vs banked ref 27/60 (6545/9956/4649 tie family) |

## Benchmark

| config | P3 | P4 | P6 (M32) |
|---|---|---|---|
| 2k-class chunk (fwd, pos0) | 83.4-108.9 ms/16 | 81.0 ms/16 | **130.6 ms/32** |
| 2k-class projected prefill | 157.6 | 197.5 tok/s | **245.0 tok/s** |
| ffn GB/s (32-tok amortized) | 163 | 163 | **214** |
| iq3d GB/s (amortized) | 105 | 127 | **197** |
| 8k-class incl draft fill | ~105.6 | 148.4 tok/s | **~190 tok/s** (chunk avg ~168 ms/32) |
| 100k rebuild incl fill | 129.7 | 134.1 tok/s | **165.3 tok/s** (591.6 s; end-of-100k inst 143) |

References: 662 tok/s @100k (vLLM W4A16 Marlin). Ladder: 245 @2k-class / ~190 @8k / 165.3 @100k incl fill
= **25% of the 662 reference @100k** (was 20% at P4); short-ctx Marlin 1-3k
remains 4-12x away. NOTE: gate logs print per-chunk tok/s assuming 16-row
chunks — the M32 numbers are 2x the printed column.

## New laws banked (P6)

1. **THE M32 BIT-IDENTITY LAW**: doubling the m-fragments without touching
   the W path is value-free — every M32 kernel (single + merged, all classes
   tested) is bit-identical to the M=16 builds. M-row scaling that keeps the
   per-row k-order is a ZERO-NUMERIC-RISK generation change.
2. **smem 43520 B static works eagerly on this dext** (FFN + gdnqg M32
   nw8/nw16 k128; < 48 KB hardware static limit). The 36.8 KB law is an
   IN-GRAPH constraint only.
3. **The load-stream amortization is sub-linear**: 1.24-1.40x wall speedup
   for 2x rows — the classic (non-DBUF) kernel serializes stage_W and mma
   per chunk, so the doubled mma/x work adds wall time; the W stream itself
   was not the ONLY cost at M=16. The remaining in-template lever is the
   DBUF-M32 hybrid (decode in the post-mma bubble; P4's DBUF pattern on the
   M32 body) — see P7.
4. **Build-flag trap**: KDIM/NDIM reversal builds a kernel that reads OOB
   (row-stride math wrong) -> device fault. The iq3o class is KDIM=6144,
   NDIM=5120 (o-proj: z[6144] -> out[5120]). Always cross-check against
   test_p5's class table.
5. **kv-slice comparison**: gating on `down_at(kv, 0, 2*4*256*128)` (first
   128 rows) covers everything a 32-row chunk writes and AVOIDS the P5
   fwd16 full-kv-download fault class (2x205MB P.down after the T=1 trunk).
   The P5 open fault did NOT reproduce this session (post-reboot + slice).

## The honest parity statement (updated)

- P6 moves the 2k-class prefill 197.5 -> ~245 tok/s (+24%); the GEMM family
  163 -> 214 (ffn) / 197 (iq3d) GB/s amortized — past the 200 line but under
  the 260-320 hope: the classic template's serialized stage/mma structure
  caps the amortization at ~1.25-1.4x.
- The 4.5x gap to 662 @100k now decomposes: (1) GEMM family needs the
  DBUF-M32 hybrid or persistent CTAs for another ~1.3-1.6x; (2) attention
  @100k ~47 ms/16-chunk (~94/32-chunk) is now the dominant 100k cost;
  (3) norm/launch floor ~16 ms/16-chunk doubles per 32-chunk in launches
  (2x16 halves) — fusing norms into GEMM prologues/epilogues is the known
  route.

## P7 recommendation (ordered)

1. **DBUF-M32 hybrid** (the priced next kernel step): port the P4 DBUF ring
   (raw-word prefetch before mma, decode in the post-mma bubble) onto the M32
   classic body — at M=32 there is 2x the mma work to hide the decode under,
   which P4's M=16 DBUF lacked (its ffn was flat). Target ffn 214 -> ~280+
   GB/s amortized, chunk ~110 ms/32 -> ~270 tok/s @2k.
2. **Attention @100k**: S-structure/CH retune or fewer-longer-CTA K1 layout
   (P5 falsified sync/latency; needs resident-CTA count).
3. **Norm-launch floor**: fuse pfk_n16/ab16/hh16 into the M32 GEMM
   prologues/epilogues (the M32 seam DOUBLED the norm launch count per
   token-row; 2x16 halves pay the launch floor twice per 32 rows).
4. Persistent CTAs (one CTA per SM, cross-chunk W residency) — only if 1-3
   disappoint; bigger restructure, documented in P5.

## Files

- engine0/pf_gemm.cu (+M32 section), pf_gemm2.cu (+M32 twin), build_p6.py,
  test_p6.py (standalone validate+bench), pf_fwd32.py (32-row harness with
  the merged-kernel bit-identity check + kv-slice gates), pf_prefill.py
  (PF_M32 seam). Cubins: pfg_{ffn,iq3d,iq3s,iq3o,q8o}_m32_*,
  pfg2_{gdnqg,attnqkvq6,attnqkvi3}_m32_*.
- Logs: ~/p6_test1.log (standalone), ~/p6_fwd32.log (harness),
  ~/p6_gate.log (serving gates), ~/p6_gate100k.log (100k rebuild).
