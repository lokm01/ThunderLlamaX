# P12 — THE 2K-CHUNK DIET + THE IMMA VERDICT (attribution-driven; 2k 315.3 -> 319.1 bit-identical)

Status: **Front 1 LANDED (the N32/PRE32/SCAN32 M32-merge diet, bit-identical,
2k 315.3 -> 319.1 tok/s, chunk med 104.3 -> 101.9 ms); the 20-28ms "gap" was
measured INTO EXISTENCE as an illusion (isolated class sum 110.0 ms == full
graph 106.3 ms; host/win_up overhead ~0.2 ms — there was never a launch/graph
overhead pool to diet); IMMA W8A8 FFN BANKED NEGATIVE on speed arithmetic
(mma-issue ceiling ~10% of the ffn launch < the 1.15x wire gate) — the GEMM
family is now closed on ALL levers by measurement + arithmetic.**

## 1. Front 1a — THE 2K ATTRIBUTION (engine0/pf12_attr.py; ~/p12_attr.log)

Method: full trunk boot (G3M on = ship class), real 2048-token prefill, then
per-class ISOLATED captured graphs (same PfGraph launch machinery as the ship
path — in-graph QMD chaining, min-of-5 synced) + full-graph references.

| class | launches/chunk | ms | us/launch |
|---|---|---|---|
| gemm_ffn (fg+fu) | 64 | 25.83 | 404 |
| scan (pfs16 x2 halves x48) | 96 | 15.89 | 165 |
| gemm_qg (gdnqg) | 48 | 15.56 | 324 |
| gemm_fd (iq3d res) | 64 | 15.14 | 237 |
| attn (pfa32c-t32 s13) | 32 | 12.00 | 375 |
| gemm_og (iq3o/q8o) | 48 | 6.43 | 134 |
| gemm_qkv (attn qkv) | 16 | 4.35 | 272 |
| nrm_ab | 96 | 3.77 | 39 |
| nrm_hh | 128 | 3.18 | 25 |
| gemm_oa (iq3s) | 16 | 2.34 | 146 |
| attn_comb (pfc16t) | 32 | 1.54 | 48 |
| pre (pfk_pre16 x2 x16) | 32 | 1.42 | 44 |
| nrm_n | 32 | 0.82 | 26 |
| dfill (7 kernels x2) | 14 | 1.64 | ~117 avg |
| emb | 2 | 0.13 | 63 |
| **SUM(isolated)** | **706** | **110.03** | |
| FULL graph (PG_SPLIT=2) | 706 | **106.26** | |
| FULL graph (PG_SPLIT=1) | 706 | 108.62 | (WORSE +2.4 — keep SPLIT=2) |
| chunk wall (steady loop) | | **101.6** | |

**THE ANSWER to "where do the 20-28ms go": NOWHERE — the premise pool did not
exist.** The prior kernel-class sum (~73-80 ms) UNDERESTIMATED the real
per-class costs (isolated sum 110.0 >= full 106.3): the GEMM family is ~69.7 ms
at 2k (not 47-50), attention 13.5, scan 15.9, norms 7.8. Host overhead is
~0 (4x win_up + sync = 0.17 ms; the steady-loop wall 101.6 is BELOW the
one-shot full-graph number because the loop pipelines submits). There is no
launch/graph/norm/dFill "overhead pool" worth 20+ ms at 2k — the chunk is
~97% real kernel work.

## 2. Front 1b — THE DIET (PF_N32=1 PF_PRE32=1 PF_SCAN32=1)

The seam launches that CAN merge bit-identically (row-per-CTA kernels, no
hardcoded M; TROWS=32 twins of per-row-verbatim sources):
- **PF_N32**: emb 2->1 (g=32, ids32 one upload), pfk_n16 32->16 launches,
  pfk_ab16 96->48 (g=416), pfk_hh16 128->64. Saves ~128 launches + 1 win_up.
- **PF_PRE32** (pfk_pre32_100k, pf_kpre32.cu TROWS=32): pre16 32->16 launches
  (24-CTA grid unchanged, 32 rows/CTA; per-row quantizer verbatim).
- **PF_SCAN32** (pfs32, pf_scan32.cu TROWS=32 + the (13+row)->(TROWS-3+row)
  final-window fix): pfs16 96->48 launches, halves the per-chunk live-state
  re-read/write traffic.
- dfill windows now slice ids32 (one upload feeds trunk + draft).
- NOT taken: PG_SPLIT=1 (measured +2.4 ms worse); attention merge (partial
  buffers are per-half by structure); dfill batching (1.64 ms pool total).

**Gates** (readout-order, first clean runs):
- 2k (PF_TRUNC=2048): **319.1 tok/s (chunk med 101.9, min 99.3, max 104.3)**
  vs P11 315.3 (med 104.3) = **-2.4 ms/chunk**; **F-relerr 1.058e-03 == the
  banked 2k-class line EXACTLY**; PF_BATCH end logits top1 29877 (15.438)
  top2 9956 (15.367) gap 0.0703 == T1 line identical; the documented
  (4649,43614) 2048-trunc tie-mine assert fires identically (EXIT=1 class).
- 8k: **278.5 tok/s (chunk med 124.4 vs P11 128.3 = -3.9 ms/chunk)**; every
  line == banked EXACTLY: **F 4.282e-02, GATE A 12/60, CTRL 13/60 alpha 2.67,
  GATE D 5/60, GATE D2 0/160** (also proves the S13-below-THR arm carried the
  diet unchanged).

Gotcha hit: **pfk_pre32_100k cubin entry is `pfk_pre32`** — the name-encoded-
entry law (KSYM entry required; loading by filename = garbage-execution device
fault). pfs32's entry matches its filename (no entry needed).

## 3. Front 2 — THE IMMA W8A8 FFN VERDICT: BANKED NEGATIVE (arithmetic)

The mission's own framing contained the answer: the packed IQ3 bytes are the
same — IMMA does not shrink the load stream. Today's attribution quantifies it
decisively:
- ffn launch = 404 us moving fg+fu = 68.2 MB = **169 GB/s isolated** (the
  P5-wall family; 265-298 GB/s amortized with the r7 tier).
- The mma-issue share: 2*32*17408*5120 = 5.74 GFLOP/launch. At the dext's
  measured fp16-HMMA class (P7A: ~142 TFLOPS fp16-class peak, IMMA 293 TOPS),
  HMMA issue floor = 5.74e9 / 142e12 = **40 us = 10% of the 404 us launch**.
  IMMA at 2x issue throughput caps the relief at ~20 us = **~5% of the launch**
  — far below the >=1.15x wire gate, BEFORE adding the W8A8 costs:
  per-128ch activation quantization + the dual-scale (act x IQ3-block)
  epilogue, both strictly extra ALU vs the fp16 path.
- The P5 DBUF evidence already showed the ffn family flat vs staging changes
  (load-stream-bound, not issue-bound); G1 KS2 measured x1.17 sub-gate; G2
  reg-falsified; r7 absorbed the big classes.
**VERDICT: IMMA W8A8 cannot clear 1.15x on a load-bound kernel with identical
bytes. The GEMM family is CLOSED on all levers (P8 ceiling + G1/G2 + this).
The relerr question (3e-3..1e-2 tie-mine risk) is moot at sub-gate speed.**
(P7A's m16n8k32.s8 fragment maps remain banked for any future persistent-CTA
rewrite that restructures the load stream itself.)

## 4. THE LADDER

| length | P10 | P11 | **P12 (diet)** |
|---|---|---|---|
| 2k fresh | 307.2 | 315.3 | **319.1** |
| 8k fresh | 271.0 | 269.2 | **278.5** |
| 100k rebuild | 192.2 | 212.9 | **218.6 (447.4s; cur=4471 EXACT; drift 3.449e-2/5.953e-2 == M32 floor; decode 27/60 tie-mine class)** |

Decode: untouched (no decode kernels changed; daemon boots the same G_CYCLE
graphs — 40.1 tok/s class).

## 5. THE HONEST 400 STATEMENT

- **@2k**: 319.1 = 48.2% of the 662 reference. The diet banked the last cheap
  ms (-2.4). The chunk is ~97% kernel work with pools: GEMM 69.7 (CLOSED —
  see verdict), scan 15.9, attn 13.5, norms 5.4 (post-diet). 400 @2k needs
  -21.4 ms/chunk from ~101.9: the ONLY remaining classes are scan (a megakernel
  restructure) and attention (S-retune at short ctx is worth ~2-4 ms at most;
  S13 dominates at 2k already). **400 @2k requires the persistent-CTA /
  megakernel class — not reachable by launch/graph diet (proven today: the
  overhead pool does not exist).**
- **@100k**: 212.9 + the diet's ~-2.4 ms/chunk class -> ~215-218 expected
  (within the P11-priced 215-218 endpoint). 400 @100k = 139 ms/chunk avg:
  attention ~55-65 ms avg + GEMM ~42-52 floor + scan ~6 + rest — the gap is
  architecture-class only.
- **Persistent-CTA recommendation (with arithmetic)**: the one untested
  architecture class. The case FOR a probe: (a) the dext is hard-limited to
  1 CTA/SM for the shipped GEMM family — a persistent-CTA design with
  dynamic smem >48 KB is the P5-doc's own >=300 GB/s route (vs 265-298);
  (b) at 2k the small-kernel pools (norms 5.4 + pre 0.7 + scan floors + attn
  empty-split overhead ~12 ms with 13 splits over ~2k keys = ~160 keys/split
  underfilling 156 CTAs) collapse into the persistent grid; (c) arithmetic:
  GEMM at 350 GB/s saves ~20 ms/chunk; attention+small-kernel fusion saves
  ~8-12 -> **~70-75 ms/chunk @2k = 430-460 tok/s** — THE 400 cross. The case
  AGAINST: it invalidates the QMD-graph machinery (the launch-diet laws do
  not transfer), Tier-2 numerics across every family, and the P5 evidence
  that stage_W/mma serialize per chunk (the DBUF-M32 hybrid was already the
  P7 answer to that). **Recommendation: probe it as a STANDALONE
  persistent-CTA ffn+norms fused tile first (one class, the 25.8 ms pool) —
  if it clears 350 GB/s amortized in-plan, the architecture pays; else the
  215-220/320-325 endpoint stands as the dext ceiling.**

## 6. Ship config + ops

Daemon line = P11 line + the P12 diet (default-off in code, ON in the env):
`M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1
PVH=1 HM=1 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1
NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1
M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256`
Logs: ~/p12_attr.log, ~/p12_g2k_diet2.log, ~/p12_g8k_diet.log, ~/p12_r100k.log.
New files: engine0/{pf12_attr.py, pf_kpre32.cu, pf_scan32.cu, build_p12.py,
pfk_pre32_100k.cubin, pfs32.cubin}; pf_prefill.py N32/PRE32/SCAN32 env-gated.

## 7. Gotchas (this session)

1. pfk_pre32_100k needs its KSYM entry (pfk_pre32) — the name-encoded-entry law.
2. The gate harness boot needs init_draft+fill_draft before prefill_batch
   (the DFILL buffers are created there, not in ensure()).
3. PG_SPLIT=1 measured WORSE (+2.4 ms) — the 2-queue split is load-bearing
   (inter-queue pipelining beats one big ring).
4. Isolated-class graphs are a VALID attribution tool on this dext (sum within
   4% of the full graph, no PROFILE needed).
