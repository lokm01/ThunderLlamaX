# P4 — Prefill optimization: DBUF ring, merged launches, batched fill_draft

Status: **STAGES 1+2 LANDED AND GATED (commit 78dadac). Chunk 96.8 -> 81.0 ms
@2k-class (-16%); prefill 148.4 tok/s @8k INCLUDING the draft fill (was 143.6 +
a separate 19 s fill); fill_draft for a 100k FRESH collapses from 253 s
standalone to an amortized per-chunk window. All P3 gates reproduce EXACTLY
(GATE A 26/60 + F-relerr 3.958e-2 on the tie-mine text; D2 spec==spec 160/160;
CTRL alpha 2.67 vs batched-fill alpha 2.68 tok/cyc).** Stages 3-4 not attempted
(budget); the Stage-4 question is partially ANSWERED by the DBUF ablation.

## Stage 1a — DBUF software-pipelined register ring (pf_gemm.cu, -DDBUF=1)

Structure per chunk: the RAW quant words for chunk s+1 (iq3: 4 q-ushorts + the
lane-quarter sw word + the d half, per plane per lane; plus the x tile as XTPR
uint2s) are issued BEFORE the mma sequence of chunk s — the global-load latency
hides under tensor-pipe work; the decode ALU + smem stores run in the post-mma
bubble (2 syncs/chunk, ~100 ns each). The ring is two NAMED register sets (E/O)
driven by explicit loop bodies — no runtime-indexed locals (q[cc]/x[t]/w8[j]
are fully-unrolled constants). Zero spills (ffn 125 regs @256thr).

Results (standalone, synced min-of-10, vs P1):
| class | P1 GB/s | DBUF GB/s | chunk ms |
|---|---|---|---|
| ffn fused (x64)   | 163 | 163 (flat) | 26.78 |
| iq3d RES (x64)    | 105 | 127 | 17.23 |
| iq3g (x48)        |  62 |  76 |  7.56 |
| iq3o (x24)        |  76 |  88 |  3.29 |
| iq3q (x8)         |  87 | 100 |  1.92 |

**The FFN did not move — and that is the Stage-4 answer forming**: DBUF removes
load-latency serialization (proven: every 1-plane kernel improved 14-23%), yet
the 2-plane ffn stayed at 163 GB/s => the ffn is bound by the mixed decode+mma
INSTRUCTION STREAM (issue-bound) and by wave quantization: grid 272 CTAs at the
dext 1-CTA/SM limit = 4 waves with a 26/82 (32% idle) tail. The MFU push to
25-40% needs a different structure (wider N-tiles per CTA to cut waves, half2
decode math, or an smem-resident W tile across 2 chunks) — NOT more latency
hiding.

## Stage 1b — small-N K-split: built, validated, REJECTED (banked negative)

KS=4 k-split + fp32 partials + pf_kscomb4 (order-documented): validated F~6.5e-5
vs the P1 kernels. But k/v move only 2 MB per launch: 0.117 ms measured = THE
DEXT LAUNCH FLOOR (synced). K-split cannot beat it and the combine ADDS a
launch => net negative. NOT wired; kernels + pf_kscomb.cu kept for the record.
**LAW: any kernel moving <10 MB is launch-floor-bound on this dext (~0.10-0.12
ms synced, ~70 us effective in-stream); the only lever is FEWER LAUNCHES.**

## Stage 1c — merged multi-segment launches (pf_gemm2.cu)

One launch runs 2-3 weight classes back-to-back (segment by blockIdx.x; the P1
classic body verbatim with per-segment (w, out, NDIM, class) selected at block
scope — uniform branch). All segments must share KDIM/KCH/NTHR and the same x16
input (true for every xh16 consumer in a block):
- GDN: qkv(Q5,80) + gate(IQ3,48) = 128 CTAs nw16 — was 2 launches.
- attn: q(Q6|IQ3,192) + k(IQ3,16) + v(Q4K,16) = 224 CTAs nw8 — was 3 launches.
- 80 launches/chunk removed; chunk 86.6 -> 81.0 ms @2k (in-context launch gap
  ~70 us each). Gates bit-identical (drift tables unchanged).

## Stage 2 — batched fill_draft via RECORDED TRUNK HIDDENS (the 100k wall)

Contract insight: the MTP draft input is x = ehproj([enorm(e(tok)) ||
hnorm(hm)]) where the TRUE hm (MTPLX contract) is the TRUNK pre-final-norm
hidden — during prefill those exist for free (xA16 rows). Feeding recorded
trunk hiddens (instead of the draft self-chain) makes every position
independent => the window batches, AND the draft attention-output/o-proj/FFN/
down become DEAD CODE for the fill (their only product, hd, fed the chain).
Precedent: mtp_spec D8 (trunk hiddens for prompt-fill IMPROVED acceptance
0.65 -> 0.70).

Per 16-pos window, interleaved after each trunk chunk (7 launches):
pfk_rec16 (ring REC0/REC1[17][5120]: rows 1..16 = chunk rows, row 0 = prev
chunk row 15 = next window hm seed) -> pfk_emb16 -> pfd_dnorm16 ->
pfg_ehd_res_hm_nw8k128 (Q4_K ehproj, RES mode = 0+acc fp32, the T=1 ADDHH
class) -> pfk_n16 -> pfg2_dqkv_hm_nw8k128 (q+k+v merged, all Q4_K) ->
pfk_pre16_100k into kv_d/sc_d (the TRUNK KV8 layout — buffers are identical).
Tail r>0: sequential T=1 fill seeded from the last recorded hidden; hd_d1
seed = trunk hidden at the last position. serve.py skips the standalone
fill_draft when the batched path ran (PF_DFILL default-on).

Gates (8k-class, inside the test_w100k host):
| gate | result |
|---|---|
| GATE A (T1 vs PF_BATCH prefill, both T1-decoded) | 26/60, F-relerr 3.958e-2 — EXACT P3 reproduction (tie-mine text class) |
| CTRL (T1 prefill + sequential fill + spec) | 160 tok/60 cyc = 2.67 tok/cyc |
| GATE D (PF_BATCH + BATCHED fill + spec) | 161 tok/60 cyc = 2.68 tok/cyc — ALPHA UNCHANGED |
| GATE D2 (spec-on-batched vs spec-on-T1) | 160/160 EXACT |
| fwd16 chunk gates (S1 kernels) | logits med 9.18e-4 / F 2.62e-3 / argmax 16/16 — bit-identical to P3 |

## Benchmark (synced; fresh-class = prefill INCLUDING draft fill)

| config | P3 | P4 | note |
|---|---|---|---|
| 2k-class (0-2048), chunk w/o draft | 83.4-108.9 ms | 81.0 ms @pos0 (fwd16) | projected 197.5 tok/s |
| 2k-class incl draft window | ~113 fresh-class | ~165 (chunk ~97 ms avg) | |
| 8k-class (0-7714) incl fill | ~105.6 (54.2s + 19s fill) | **148.4** (52.0s) | 5.90x T1 |
| 100k rebuild incl fill | 97.1 fresh-class (754 + 253 s) | see ~/p4_gate100k.log | fill now amortized |
| chunk ms @pos0 / @7.7k (incl draft) | 83.4 / 116.8 | 90.1 / 112.2 | |

References: 662 tok/s @100k (vLLM W4A16), 1-3k short-ctx Marlin. At 148 @8k we
are ~22% of the 100k reference, ~5-7x from short-ctx Marlin.

## The honest parity statement

To reach 662 @100k the remaining gap (~4.5x) decomposes as measured:
1. **FFN/GEMM MFU (the big one)**: the GEMM family in-context is ~60 ms of the
   81 ms chunk at ~18-20% MFU. The DBUF ablation shows the ffn is issue/wave-
   bound, not latency-bound: 272 CTAs / 1-CTA-per-SM dext = 4 waves + 32% tail.
   Fix = wider N-tiles per CTA (fewer waves), half2 decode, 2-chunk W residency
   — a kernel-generation effort (~2-3 sessions), NOT tuning.
2. **Attention @100k (~49 ms/chunk, pfa16 ~80 GB/s sync/compute-bound)** —
   Stage 3 (register-double-buffered K/V tile streaming) untouched this
   session; projected -25..35 ms/chunk @100k.
3. **Norm/launch floor**: ~129 norm launches remain (~16 ms); fusing into GEMM
   prologues/epilogues (sumsq atomics in producers) is the known route (~-8 ms)
   but touches every kernel signature.
4. fill_draft: DONE (this session). A 100k FRESH is now prefill-bound only.

## New laws banked

1. **THE LAUNCH-FLOOR LAW**: kernels moving <10 MB sit at the dext launch floor
   (~0.10-0.12 ms synced, ~70 us in-stream); merge launches or accept it.
   K-split of small-N classes is a documented negative.
2. **The DBUF recipe**: raw-word prefetch BEFORE the mma sequence + decode in
   the post-mma bubble; named E/O register rings (no runtime indexing). Works
   (14-23%) whenever load latency binds; does nothing when issue/waves bind.
3. **The daemon shutdown->reboot law is 5/5**: even a CLEAN RPC shutdown
   (dev.synchronize + os._exit(0)) of the idle, healthy daemon rebooted the
   machine within seconds. Budget a ~10 min machine cycle around any daemon
   stop; the machine self-heals (auto-reboot, Tailscale+ssd survive).
4. **Trunk-hidden draft fill is acceptance-neutral-to-positive** (2.67 -> 2.68
   tok/cyc; D2 160/160) and makes the draft FFN/o-proj dead code during fill.

## Files / env

- engine0/pf_gemm.cu (+DBUF/KS), pf_gemm2.cu (merged segments), pf_kscomb.cu,
  pf_norms16.cu (+KSEL 6 pfk_rec16), pf_prefill.py (draft window + PF_DFILL),
  pf_gate2k.py / serve.py wiring.
- Canonical env additions: **PF_MERGE=1 PF_DFILL=1** (default-on in code;
  PF_DFILL=0 / PF_MERGE=0 disable for A/B).
- Logs: ~/p4_fwd16_dbuf.log, ~/p4_fwd16_merge.log, ~/p4_gate_s12b.log,
  ~/p4_gate100k.log (P1 baseline in ~/p1_sweep.log).
