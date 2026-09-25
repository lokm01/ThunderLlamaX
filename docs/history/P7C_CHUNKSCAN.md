# P7-C — The chunked delta-rule scan (FLA WY representation)

Status: **BUILT, VALIDATED (all 4 gate stages GREEN), BENCHED, AND WIRED INTO
pf_fwd32 (PF_SCANC=1). The sequential per-token scan (pf_scan16, the oracle)
is replaced at chunk scale by per-64-token WY matmuls on tensor cores
(mma.m16n8k16, LDS.32 fragments) + a tiny sequential inter-chunk state pass.
Gates: single-chunk vs fp64 token-loop O med 4.1e-4 / rec 3.2e-4; 64-tok vs
the REAL pf_scan16 z med 0.0 (EXACT) F 5.6e-4; 512-tok (8 chunks) drift rec
3.1e-4 (budget 3e-2 — 100x under); pf_fwd32 logits med 8.7e-4 F 2.4e-3 PASS
(P6 class 8.4e-4/2.6e-3), argmax 31/32, fwd32 chunk 130.6 -> 121.4 ms/32 tok
= 263.7 tok/s projected. Scan-stage bench @512-tok super-chunk x48 GDN
blocks: 47.5 ms end-to-end (56.3 sum-of-kernels) vs 277.0 ms same-session
sequential control = 5.8x (the mission's 425 ms cross-session number scales
to ~9x; the ~20-25 ms target needs the follow-up traffic fix below).**

## The algorithm (derived FROM pf_scan16's exact op order; vLLM/FLA-equivalent)

Per GDN block, per 64-token chunk (log2-space cumsum g2 = cumsum(lambda*log2e),
lambda = softplus(a+dtb)*ssma, beta = sigmoid(b); q/k L2-normalized with
qn = (1/max(sqrt(qss),EPS_Q))*ISQ128 — VERBATIM pfs16):

- B[i,m] = beta_i (k_i . k_m) 2^(g2_i-g2_m), strictly lower; T = (I+B)^-1
  (blocked forward substitution, two 32x32 tiles + coupling at C=64, fp32)
- U = T(beta V);  Y = K S_-1;  d = U - T(beta*2^g . Y)
- O = M d + Qe S_-1   (M[i,m] = (q_i . k_m) 2^(g2_i-g2_m), m<=i; Qe = q .* 2^g_i)
- S' = 2^g_end S_-1 + K^T (2^(g_end-g) . d)

The ONLY numerics change vs the oracle is this reassociation (fp16 mma
operands, fp32 accumulate + fp32 state). beta sits on the ROW of B (the
current token) in OUR orientation — the mission brief's "beta_j" column form
is FLA's transposed-state convention; identical algebra.

## Kernels (engine0/pf_scanchunk.cu; build_p7c.py; warp-token names)

- **pfca** (K_A, parallel over (head, chunk)): preprocess VERBATIM (4-tap
  causal conv w/ convlive fallback rows <0, silu, norms, lambda/beta, cumsum)
  -> KK^T + QK^T mma -> masks -> T solve (fp32 smem, column-per-lane; the
  32-col tile solves are lane-serial but ~1-2us) -> U = T(beta V) mma (two
  v-halves) -> dumps khat/khatT/qe/u/vraw/m/t/meta to scratch. 512 thr.
  smem: C=64 43520B (EAGER-ONLY; solve phase Tf+Xf+G), C=32 21504B
  (in-graph legal).
- **pfcb** (K_BC, sequential over chunks, grid = 48 heads x 4 v-quarts):
  state slice [128][32] fp32 in smem (16KB); per chunk: Y mma -> d = U -
  T(bg.Y) -> O = M d + Qe S (mma) -> oout fp32 -> S' = 2^g_end S + K^T(sg.d).
  The v-column factorization is EXACT (no cross-quart communication).
  smem 34336B (C=64) — in-graph legal.
- **pfcz** (z epilogue + conv_live writeback): grid = NH*NC*(C/8), one row
  per WARP (v2: the first version's per-thread serial 128-loop was
  latency-bound — 1.382 -> 0.128 ms, 10.8x). Epilogue math VERBATIM incl.
  the half-gate g*sigmoid(g) via hexp2/hrcp.

Scratch per (block, head, chunk), fp16 with padded strides (LDK=136,
LDTC=C+8): khat | khatT | qe | u | vraw | m | t | meta(bg,sg,gend) —
107,024 B at C=64 (41 MB/block for NC=8), 50,448 B at C=32. Weight plane
per block (47,200 floats): convw 40960 | dtb 48 | ssma 48 | snw 128(+pad).
NOTE: **ssm_norm.weight is ONE 128-vector shared by all 48 heads** (pfs16
indexes snw[j], j<128) — not 6144.

## Gates (test_p7c.py; READOUT-ORDER law)

| stage | comparison | result |
|---|---|---|
| 1 | C=64 NC=1 vs numpy fp64 token-loop | O med 4.144e-4 F 4.1e-4; rec med 3.165e-4; conv exact — **PASS** |
| 2 | 64 tok vs 4 chained REAL pf_scan16 | z med 0.0 (EXACT) F 5.56e-4; rec med 3.125e-4 — **PASS** |
| 3 | 512 tok (NC=8) vs 32 chained pf_scan16 | z med 0.0 F 5.51e-4; rec med 3.102e-4 (budget 3e-2) — **PASS** |
| 4 | pf_fwd32 PF_SCANC=1 vs T=1 trunk (real weights) | logits med 8.726e-4 F 2.409e-3 **PASS**; argmax 31/32; rec drift 5.2e-4 -> 1.6e-3; kv bytes 289-377/262K (int8-KV tie-mine class, expected) | 

Multi-chunk drift does NOT accumulate (decay bounds it: 8 chunks drift the
same as 1). The fp64-oracle z comparison shows med ~7e-2 — that is the
HALF-approx gate (hexp2/hrcp are approximate intrinsics; pfs16 uses the
same code path, so mine-vs-pfs16 is med-EXACT and that is the binding gate).

## Bench (synced min-of-N, same-session control)

| kernel | ms/block (8x64 chunks) |
|---|---|
| pfca | 0.520 |
| pfcb | 0.525 |
| pfcz (v2) | 0.128 |
| **sum x48 blocks** | **56.3 ms / 512-tok super-chunk** |
| end-to-end (48 blocks x 3 launches) | **47.5 ms** |
| pfs16 control (48 blocks x 32 launches) | 277.0 ms -> **5.8x** |

fwd32 (C=32 tier, PF_SCANC=1): chunk 121.4 ms/32 tok = **263.7 tok/s
projected** (P6 record 130.6/245.0 — the scan swap nets ~9.2 ms per pass).

## The honest gap vs the ~20-25 ms target — and the fix

The K_A/K_BC split pays an 82 MB/block scratch roundtrip (41 write + 41
read) + 25 MB o/z per block: ~107 MB/block, ~5.1 GB per super-chunk = ~11.5
ms at the 447 GB/s ceiling, and the kernels run at ~100 GB/s effective
(latency/occupancy bound: pfca's serial two-half U staging + transposed
dumps; pfcb's 8-chunk sequential loop with 4 syncs/chunk). The fix (P7-C2):
**fuse per (head) like FLA's shipped kernel — one CTA does ALL chunks
sequentially with the state resident, computing A/T/U/Y/d/O in-place (zero
scratch traffic; only qkv in + z out + rec in/out ~ 30 MB/block -> ~4-6 ms
floor)**, at the cost of 48 CTAs/block occupancy — or keep the split but
stage scratch through smem-tiled K-groups. Also: fold pfcz into pfcb via an
oout-resident second pass (saves the 25 MB o roundtrip).

## New laws banked (P7-C)

1. **THE PFS16 Z-GATE IS g*sigmoid(g)** (a z-swish), computed in HALF via
   hexp2/hrcp approx intrinsics — a fp64 oracle of the epilogue lands ~7e-2
   med; ALWAYS gate against the pf_scan16 cubin (both use the half path ->
   med-EXACT), not a fp64 model.
2. **ssm_norm.weight (snw) is 128 floats shared by all 48 heads** — sizing
   it 6144 silently reads garbage for heads 1-47 in any reimplementation.
3. **Scratch-region pointer chains must repeat EVERY stride** — the
   pfcb m_g = u_g + U_C bug (missing +VR_C) pointed M at vraw and T at M:
   state path still looked "right" (median hides it under strong decay);
   factor-level dumps (kh/qe/u/m/t vs numpy) are the fast detector.
4. **A shared kstep loop for two mmas with DIFFERENT K dims silently
   truncates** (Qe@S needs 8 k-steps at K=128, M@d needs C/16) — split the
   loops.
5. **Phase-reused smem regions need a barrier when a later phase's WRITE
   region overlaps an earlier phase's READ region** (bvt staging vs the Xf
   head; the t16 staging must complete first).
6. **A CTA-per-(head,chunk) dump-slot design gets clobbered by sibling
   heads** — debug dumps must be guarded to one CTA (if (h != 0) return).
7. **Synthetic-scratch probes MUST use the padded strides** (LDK/LDTC) —
   natural-stride puts make every kernel look broken.
8. **The per-thread serial 128-loop reduction is latency-bound** (~18 GB/s
   class): one-row-per-warp + shfl reduction with an 8x-finer grid was
   10.8x. Same law class as the P5 wave-structure findings.
9. **Weight planes: convw is 40,960 floats** (10240 channels x 4 taps), not
   10240 — the plane stride math must count all four taps.

## Files (engine0/)

pf_scanchunk.cu (pfca/pfcb/pfcz + pfcbdbg), build_p7c.py (10 cubins:
c64_nc8, c64_nc1, c32_nc1 tiers + dbg), test_p7c.py (fp64 oracle + gates +
bench), diag_p7c.py / diag_pfcb.py / diag_onehot.py (the factor-level
debugging ladder), pf_fwd32.py (+PF_SCANC wiring; .bak_p7c = pre-edit).
Logs: ~/p7c_fwd32.log (stage-4 gates + timing).

## P7-C2 follow-ups (ordered)

1. Fused per-head FLA kernel (zero scratch traffic) -> the ~25 ms target.
2. Fold pfcz into pfcb (oout-resident second pass).
3. C=128 chunks (T solve 4 tiles) once fused — fewer chunk boundaries.
4. Super-chunk assembly at 256/512 rows (P7e) with the scan swapped in.
