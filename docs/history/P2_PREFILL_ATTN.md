# P2 — Prefill machinery: pKPRE-M / pATTN-M / pSCAN-M / M=16 norms / draft-GEMM classes

Status: **ALL per-piece gates PASS** (pKPRE-M and pSCAN-M and the norms family
BIT-EXACT vs the T=1 canonical; pATTN-M 3.5e-4 vs numpy ground truth and
≤1.05e-3 vs the healthy T=1 chain). The **16-row forward harness runs
end-to-end at 96.8 ms/chunk = 164.8 tok/s projected (7.7x the T=1 trunk)** —
but the full-chain logits gate is **OPEN**: per-stage wiring verifies clean
(~1e-4, dbg3/dbg8), the scan self-consistency is bit-exact, and 2e-4 input
perturbations do NOT amplify on random data (2.7e-4 out), yet on real
embeddings the GDN rec states diverge (0.34 rel at blk0, constant across token
sets — marginally-stable rec-mode signature) and logits decorrelate by
mid-stack. P3's end-to-end greedy agreement is the adjudicating gate (as the
mission itself states). fill_draft M=16 kernels are BUILT (Q4_0 draft pGEMM
classes + pfd_dnorm16) but the replay validation did NOT run this session.

## What was built

- **engine0/pf_kpre16.cu** — pKPRE-M (grid 24, 256thr): per chunk of 16 rows,
  q-norm + partial-RoPE (pairs (i,i+32), theta via freqs table — engine passes
  1e7) + qw16 fp16 rows (0.0625 fold); k-norm + rope + BIASED-UINT8 KV8 append
  with per-(row,32ch) fp16 scales — quantizer math VERBATIM from spk_preqh.cu.
- **engine0/pf_attn.cu** — pATTN-M: `pfa16` (K1: grid 4*S, 1024thr/32 warps,
  all 96 q-rows (16 rows x 6 heads) of a kv-group per CTA; QK+PV on
  mma.m16n8k16 with the W2G fragment map; online softmax per row in registers,
  fixed xor butterflies; causal bound l <= pos + (r&15); empty split -> identity
  partials) + `pfc16` (K2: fixed-order combine over S partials + sigmoid gate
  -> ao16). SMEM = KV 16384 | SCP [96][32]f32 12288 | Pm [96][32]f16 6144 |
  corv 384 = 35200B (<= the 36864 in-graph law). **V is staged OVER the dead K
  region** (K dead after QK+row-owners) — the smem diet that makes M=16 fit.
- **engine0/pf_scan16.cu** — pSCAN-M (grid 48, 256thr): k2s3's T=3 pattern at
  T=16 with the state in REGISTERS (warp owns 16 v_idx rows, lane owns 4 k-dims
  -> sreg[16][4], 128 regs, 0 spills): loaded once from the live slot before
  the t loop, stored once after. Kills the per-step global rec roundtrips
  (4.7GB/chunk at T=16 -> 150MB final writes). Conv window generalized
  (row(t-k) = qkv16[t-k] or live[(t-k+3)*CONV_CH]); intermediate conv slot
  writes are dead in-kernel -> only the final window (rows 13,14,15) written.
  Per-step op order byte-identical to k2s.
- **engine0/pf_norms16.cu** — pfk_emb16 / pfk_n16 / pfk_ab16 / pfk_hh16 /
  pfd_dnorm16 (KSEL-dispatched, one kernel per cubin): the T=1 norm-family
  math verbatim at M=16 (per-warp redundant RMS reductions).
- **engine0/pf_gemm.cu + `-DRES=1`** — fp32 residual-add output mode for the
  down-projection class: out32 = res16 + fp32 acc (matches down8's y = hh +
  acc; NO fp16 rounding of the acc before the residual — the P1 law-4
  contract). RES=0/default path byte-identical to P1 (existing cubins
  untouched).
- **engine0/build_pf2.py** — builder (docker nvcc, cuobjdump symbol check).
  **engine0/test_pf2.py** — per-piece differential validation + timing.
  **engine0/pf_fwd16.py** — the 16-row forward harness. dbg2..dbg8 = the
  forensic ladder.

## Gates (test_pf2.py + dbg2 numpy ground truth)

| piece | ref | result | gate |
|---|---|---|---|
| pfk_emb16 / pfk_n16 / pfk_ab16 / pfk_hh16 | h_embed / k0_norm / k0ab / k3m_hh x16 | **relerr 0.0 (bit-exact)** | 1e-3 PASS |
| pfk_pre16 @pos=0 and @pos=512 | spk_pre1qh_100k x16 T=1 | **kv bytes 0/205520896 mismatches; qw16 0/98304 — BIT-EXACT appends** | bit-exact PASS |
| pfa16+pfc16 @pos=0 / @pos=512 | T=1 chain (pre1qh+g4nw32qh1+c1g) | F 5.7e-4 / 1.04e-3, med 5.7e-4/9.3e-4 | 3e-3 PASS |
| pfa16+pfc16 @pos=0 | **numpy softmax ground truth** | F 3.5e-4 (BETTER than the T=1 ref's 6.5e-4 — fp32-acc class) | PASS |
| pfs16 z/conv/rec @16 steps | trunk k2s x16 T=1 | **relerr 0.0 all three — BIT-EXACT** | 1e-3 PASS |
| 2e-4-input stability | k2s x16 perturbed | rec drift 2.7e-4 (no amplification on random data) | info |

## Timings (synced, min-of-10) and the projection

| kernel | ms | note |
|---|---|---|
| pfk_pre16_100k | 0.161 | 24 CTAs (latency-floor class) |
| pfa16 s32 @pos100320 | 2.86 | ~80 GB/s KV — S sweep 32..256 = 2.57-2.80 (S=128 best, 2.57): NOT BW-bound; same structural class as the decode K1 (compute+sync-bound; 100 FLOP/B at this split shape) |
| pfc16 s32 | 0.209 | |
| pfs16 | 0.277-0.285 | x48 GDN blocks = 13.3 ms/chunk |
| pfk_n16/ab16/hh16 | 0.122-0.128 | launch-floor class |

**16-row forward harness (pos=0, CTXK=100352, KV8+QH path): 96.8 ms/chunk ->
164.8 tok/s projected; T=1 trunk reference 47 ms/tok = 21.4 tok/s; 7.7x.**
Attribution vs P1's 169.7 GEMM-only: GEMMs 94.3 (P1) + scan 13.3 + norms ~16
(129 launches, launch-floor) + attention ~small at pos=0 + embed/head ~4. The
M=16 attention at END-of-100k costs 2.57-2.86 ms/layer x 16 = ~41-46 ms/chunk
(S=128); averaged over a full 100k prefill (quadratic in pos) ~20 ms/chunk —
projected full-prefill average ~16/(96.8+20+drain) ≈ 130-140 tok/s, first
chunks ~160+. Next levers: attention BW (the S-sweep shows the kernel is
sync/compute-bound, not KV-BW-bound — needs double-buffered tiles, the same
P1 lever list), the norm-launch floor (fuse n16+ab16 into the GEMM epilogues).

## The harness correctness story (OPEN gate — evidence trail)

1. Every per-stage comparison is clean: dbg3/dbg8 compare MY chain vs the T=1
   piece kernels INSIDE the harness env (TrunkEngineW1C SKV=1 KV8=1 QH=1
   CTXK=100352): embed/xh/araw BIT-EXACT; qkv/gate 1.9e-4/2.4e-4; z row0
   9.4e-5; attn q/k/v 3.1e-4; appends 23 bytes of quantizer-boundary flips;
   o-proj/hh/ffn/down all 1e-4-class.
2. Self-consistency: replaying the T=1 k2s 16x on MY captured harness block-0
   inputs reproduces my pfs16 rec BIT-EXACTLY (0.0).
3. Stability: a 2e-4 input perturbation -> 2.7e-4 rec drift after 16 steps
   (random data).
4. YET the full harness run: blk0 rec drift 0.34 (|rec| ~119), growing through
   the stack; logits decorrelated (med 1.05); kv appends diverge downstream.
   The drift is ~constant across token sets (random ids vs 100-115) — the
   signature of a marginally-stable rec mode (al ~ 1) that real embeddings
   excite and random data does not. P3's end-to-end greedy agreement on a real
   prompt is the adjudicating gate; if it fails, bisect from the al/be
   distribution per head (instrument exp(softplus(a+dtb)*ssma) magnitudes).

## Latent trunk bugs found (fixed in the harness, NOT committed to trunk)

- **trunk.py E.token() hardcodes local_size=(256,1,1) for EVERY kernel** —
  with SKV+QH the 1024-thread spk_g4nw32qh1 K1 runs at 1/4 threads -> NaN
  partials from token 0 (dbg6/dbg7). The daemon path (graphs) is unaffected.
  The harness works around it by launching the _seq entries with nw32-aware
  local_size (pf_fwd16.t1_token).
- **trunk_w1c.py's KV8+QH combine pairing**: cn = "spk_c1_100k" (S=32 bake)
  paired with the S=256 K1 — partial-layout mismatch = garbage. The healthy
  pairing is spk_c1g_100k (what MTPEngine's KV8 branch uses). The harness
  overrides pr["spk_c1"].
- **trunk_w1c.py TrunkEngineW1C.attn() (the METHOD, not _seq)** has a
  trailing-comma arg bug (`((d["qw16_1"],) if QH else d["qw1"],)` passes a
  tuple-tuple) -> 'tuple' has no va_addr. Dead code in the canonical flow.

## fill_draft batching (STAGED, not gated)

Built: pfg_q4dq/q4dk/q4dv/q4do (draft Q4_0-repacked M=16 GEMMs),
pfg_q4dd_res (draft down, RES mode), pfg_q4eh_res (ehproj, fp32-out),
pfd_dnorm16 (enorm||hnorm cat). The draft fill replay harness (sequential
hd recording + M=16 replay vs kv_d content) did NOT run this session — the
kernels are validated-class (QCLASS=7 addressing = P1's validated q4dn) but
the ≤1e-3 KV-content gate is OPEN. NOTE: the draft chain is autoregressive
(xin_q needs hd_{q-1}) — M=16 batching requires replaying RECORDED trunk/draft
hiddens (the fill's hm source is P3's assembly decision).

## New laws banked

1. **The kernel-SYMBOL vs cubin-FILENAME law**: TinyELF(name=) must be the
   extern-C kernel symbol, not the cubin file name — a mismatched bind
   launches garbage (SM Multiple Warp Errors) with zero python-side error.
   build_pf2.py's builder passes kname separately from the variant-suffixed
   cubin name.
2. **M=16 attention smem diet**: K+V share ONE 16KB region (V staged over the
   dead K after QK+row-owners) — the only way RMAX=96 fits the 36864B in-graph
   law with a [96][32] fp32 SCP plane.
3. **The register-held scan state**: 16 v_idx x 4 dims per lane (64 f32) fits
   at 128 regs / 0 spills — per-step GLOBAL rec slots are unnecessary for
   in-kernel T-loops; write live once (bit-exact vs the slot-chained k2s).
4. **E.token is incompatible with nw32 (1024-thread) kernels** (see bugs) —
   any eager T=1 reference must dispatch local_size by the name-encoded
   launch-config law.
5. The T=1 KV8+QH reference attention triple = spk_pre1qh_100k +
   spk_g4nw32qh1_100k (grid 4*256, LS 1024) + **spk_c1g_100k** (S=256) — the
   bare spk_c1_100k is the S=32 combine.

## Files / how to run

- Build: `python3 -c "import build_pf2; build_pf2.main()"` (import-mode; see
  law below) — 19 targets incl. S=64/128/256 attention variants.
- Validate: `PATH=... ~/tg311/bin/python -u test_pf2.py [norms|attn|scan]`.
- Harness: `SKV=1 KV8=1 QH=1 SKV_CTXK=100352 SKV_S=256 PATH=... ~/tg311/bin/python -u pf_fwd16.py`.
- Transient infra note: after the reboot the cuda-nvcc-persistent container
  was gone and `python3 build_pf2.py` (script-mode) consistently failed with
  docker-API connect errors while `python3 -c "import build_pf2; ..."`
  (import-mode) worked — unresolved; use import-mode.

## Next (P3 handoff)

1. Adjudicate the harness divergence with the end-to-end greedy gate on a real
   prompt (2k first); if it fails, instrument al/be per head (marginal-stability
   hypothesis) and compare against the fp16-KV non-QH T=1 path as a second
   reference class.
2. Assemble the batched prefill loop behind the harness pieces (chunking, KV8
   int8 KV, pos threading) + fill_draft replay (kernels staged above).
3. Attention BW: double-buffered KV tiles + S=128 (the kernel is sync-bound at
   80 GB/s; P1's ffn double-buffer lever applies here too).
4. Norm-launch floor (~16 ms/chunk): fuse the M=16 norms into GEMM
   epilogues/prologues (k0ab16's xh16 write is a natural q5kv prologue).
