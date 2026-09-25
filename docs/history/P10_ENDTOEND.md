# P10 — THE END-TO-END PREFILL PUSH: t32 attention SHIPPED (2k 307.2 / 8k 271.0); A4 falsified; S26 banked

Status: **mission items 1-2 LANDED (item 2 honestly falsified by measurement);
items 3-7 NOT REACHED (budget went to the S13 mid-pos law + S26 rescue attempt).
Ladder: 2k 281.6 -> 307.2 (+9.1%) / 8k 239.6 -> 271.0 (+13.1%) / 100k rebuild
202.5 -> 192.2 (-5.1% — the S13 mid-pos critical-path regression; S26 fixes it
standalone but loses in-plan at 8k; the hybrid is the priced follow-up).
Ship: daemon + api live on the P10 line, verified end-to-end.**

## 1. Item 1 — pfa32c-t32 WIRED + SHIPPED (PF_ATTN32=1)

- **Kernels**: `pfa32c_t32_s13_100k` (entry `pfa32ct`, 64 regs / 104B spills /
  36.9KB smem / 512thr, grid 4*13*3 = 156 CTAs, 2/SM at CFG=100) +
  `pfc16t_s13` (the S13-layout combine, entry `pfc16t`, 40 regs, grid 24x256).
  Partials reuse the pmA/psA/pAA handles (4992 of 12288 slots). Graph-cache key
  now (M32, DFILL, G3M, NT32, ATTN32, A4). Carveout: ship AUTO_NAMES =
  `pfg,pfa32c` — "pfa32c" substring-matches BOTH entry symbols (pfa32c, pfa32ct);
  pfc16t stays min-fitting (the P8 combine-regression law).
- **Standalone corr** (pf10_probe.py corr): t32+pfc16t vs pfa16-ref relerr med
  1.714e-04 max 3.07e-02 (Tier-2 class as priced; maxes = the known
  near-zero-denominator outliers); pfc16t vs numpy-exact 1.7e-4; determinism x2.
  NOTE the probe-law found: the gated-combine row map is q-row = h + 24*t
  (t-MAJOR), NOT h*16+t — head-major slicing in the REFERENCE was the only bug
  (kernel was right all along).
- **Gates** (readout-order, first clean runs):
  - pf_fwd32 (F harness): logits med 8.389e-04, F 2.474e-03 PASS, argmax 32/32,
    drift rec/conv 2e-4..1.4e-3, kv byte-flips 236-357/262144 (int8-KV tie class).
  - 2k (PF_TRUNC=2048): **307.2 tok/s** (chunk med 106.0ms; was 119.6), F
    1.058e-03 == the banked 2k-class line EXACTLY; EXIT=1 = the documented
    (4649,43614) 2048-trunc tie-mine assert.
  - 8k: **271.0 tok/s** (med 127.3ms; was 142.3); GATE A 12/60, CTRL 13/60
    alpha 2.67, GATE D 5/60, **D2 0/160**, F 4.282e-02 (banked 4.26e-2 class).
  - 100k rebuild: **192.2 tok/s** (508.8s), **cur=4471 EXACT**, drift max rec
    3.195e-02 / conv 5.979e-02 (== the M32 reassociation floor 3.45e-2/6.0e-2),
    decode-on-rebuilt tie-mine class, pos97k chunk 176-177ms (was 187.6).
- **Per-launch bench** (synced min-of-5 @pos100336): pfa16 2.83 / t32 2.37 /
  t32+pfc16t pair 2.40 ms = **1.18x/launch, ~-14ms/chunk @100k-end, ~-14ms @2k**.

## 2. Item 2 — A4 atomic last-CTA combine: BUILT, VALIDATED, FALSIFIED

`pfa32ctl` (fused epilogue; self-resetting ticket via atomicAdd, hardcoded NCTA,
__threadfence release/acquire, block-stride epilogue loop) is CORRECT (vs numpy
med 1.686e-04, det x2, ctr==0 after two launches on one counter — the atomic
pattern itself WORKS on this dext) and stays 64 regs/104B spills. But the BENCH
killed it: fused 2.55 vs pair 2.40 ms @100k-end — the separate 24-CTA combine is
only +0.03ms incremental (S=13 partials are 5.1MB, not the S=32-era 12.6MB), and
a single-CTA serial read of 5.1MB (~0.18ms) costs MORE than the launch it saves.
**Verdict: A4 = measured regression, banked, NOT shipped** (PF_A4 machinery
remains wired for future use).

## 3. THE S13 MID-POS LAW + the S26 rescue (the session's key finding)

The 100k rebuild REGRESSED (192.2 vs 202.5): mid-pos chunks (10k-90k) are
+17ms. Root cause (confirmed by standalone bench @pos 39040): S=13 splits are
7720 keys — when only ~5 of 13 splits are active the machine is underfilled AND
the per-CTA critical path is 7720 keys vs pfa16's 3136: **t32_s13 2.31 vs pfa16
1.48 ms @pos39k (57% WORSE)**, while winning at both ends (2k: 1.39 vs 1.50;
100k: 2.36 vs 2.81).
- **S=26 build** (`pfa32c_t32_s26_100k` + `pfc16t_s26`, 312 CTAs = ~2 co-res
  waves, 3860-key splits, still 64 regs): standalone DOMINATES everywhere —
  1.22/1.20/2.36 ms @pos 39040/7714/100336 (beats pfa16 at every pos, beats
  S13 at mid by 1.9x).
- BUT in-plan at 8k it LOSES to S13 (255.0 vs 271.0 tok/s): at low pos S=26 has
  ~300 EMPTY CTAs per launch x 32 launches — each still writes 32KB of identity
  partials (~9.6MB pointless traffic/launch + CTA-slot cost).
- **THE PRICED FOLLOW-UP (not landed, budget)**: dual-graph hybrid — S13 graph
  for pos < ~10k, S26 graph above (both plans in ensure(); _pf_graphs captures
  both; _pf_submit_chunk picks by pos). Expected: 2k/8k stay 307/271, 100k
  recovers to ~203-210 (mid chunks -17ms, end +~1ms). ~25 lines + the 100k
  re-gate. The 8k/2k gates carry over (threshold above their range).

## 4. THE LADDER

| length | P7F3 | P8 | **P10 (S13 ship)** | S26 (banked) |
|---|---|---|---|---|
| 2k fresh | 259.4 | 281.6 | **307.2** | ~305 (empty-CTA cost small @1 wave) |
| 8k fresh | 222.9 | 239.6 | **271.0** | 255.0 |
| 100k rebuild | 191.8 | 202.5 | 192.2 | ~205 est (hybrid gets both) |

Decode: untouched (no decode kernels changed; the daemon boots the same
G_CYCLE graphs — 40.1 tok/s class; health shows parked pos 97810 / cur 4471).

## 5. THE HONEST 400 STATEMENT (mission item 6)

@2k we are at **307.2 = 46% of the 662 reference**; @100k at 192-210 = 29-32%.
What 400 @100k would additionally require (per-chunk arithmetic, current
~166.6ms avg / 3055 chunks):
- attention family is STILL ~60% of the 100k chunk (~70-80ms avg post-t32):
  the S26/hybrid recovers ~10ms; beyond that the remaining routes are the
  persistent-CTA rewrite (P7F3's unpriced 2x-class) or K/V-read restructuring —
  both architecture-class, out of P10 scope by mission rule.
- GEMM family ~42-52ms sits at 265-298 GB/s amortized (the P8 ceiling);
  K-split-2 (G1) addresses a ~6-8ms classic-tail pool (iq3s + out-classic +
  q8o) — the r7 tier already absorbs iq3d/iq3o. Even a perfect G1+G2+pfk_pre64
  lands ~-15ms = ~215-220 @100k.
- MFU arithmetic for 400 @100k: 400 tok/s = 139ms/chunk avg INCLUDING the
  position-dependent attention growth (176ms at end-pos today). With
  GEMM at its measured 298 GB/s family ceiling (~42ms floor = 12.6GB/chunk)
  + scan ~6 + norms ~10 + dfill tail, the attention budget would need to be
  ~60-70ms AVG (currently ~75-85) AND every scrap banked — i.e. 400 @100k
  needs the hybrid + G1 + G2 + a further ~1.5x attention-class win (persistent
  CTAs). 400 @2k-class is closer: 307 -> 400 needs -5.5ms/chunk from the
  remaining ~15ms of priced scraps (G1 tail + ring-depth-4 + pfk_pre64) —
  plausible on this dext.

## 6. Items NOT REACHED (budget): 3 (G1 K-split-2 + TGT3 sweep), 4 (G2
ring-depth-4), 5 margins (pfk_pre64/pfk_pre32 wiring), 7 (IMMA W8A8 probe).
Ranked for the next session by measured pools: (a) the S13/S26 hybrid graph
(+10-15 @100k, machinery 90% built); (b) G1 KS2 on the classic tail (~6-8ms);
(c) G2 ring-depth-4 (~4-5ms, reg-gated); (d) pfk_pre32 one-launch pre (-1ms);
(e) IMMA W8A8 ffn probe (VRAM-neutral s8 materialization; the 293-TOPS path).
The KS scaffolding for M32 does NOT exist yet (pf_gemm.cu M32 guards out
KS/DBUF; the classic-KS pattern with fp32 out32 partials is the template).

## 7. Ops / ship config

Daemon line = P8 canonical + `PF_ATTN32=1` + `NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c`
(full line in ~/w100k_serve_p10.log; api_server on /usr/bin/python3 — NOT the
tg311 venv, which lacks fastapi/jinja2). Logs: ~/p10_g2k.log, ~/p10_g8k.log,
~/p10_g8k_s26.log, ~/p10_r100k.log, ~/w100k_serve_p10.log, ~/api_p10.log.
Verified: health ok (pos 97810 / cur 4471 cached), FRESH turn 5.8s end-to-end,
FOLLOW_UP 2.6s (delta-prefill path). Gotchas hit this session: (1) the
name-encoded-entry law AGAIN (pf_fwd32 prog() loads by cubin filename — the t32
entries are pfa32ct/pfa32ctl; loading by filename = Illegal Instruction
garbage-execution); (2) pf_fwd32 with PF_GEMM3=1 + full KV OOM-faults at the
packed7 upload (run it with PF_GEMM3=0); (3) nc -U hangs after the shutdown RPC
response — the RPC still lands; (4) the gated-combine row map is t-major
(q = 24*t + h); (5) fused-epilogue flat indexing must use /6144 not >>13
(24*256 is not a power of two).
