# P8 — THE CTASM UNLOCK APPLIED TO PREFILL (shipped, gated, +5.6-8.6% ladder)

Status: mission items 1/2/5 LANDED and SHIPPED in the daemon; items 3/4
MEASURED and honestly falsified for the current kernel zoo (attention
co-residency is REGISTER-capped, not carveout-capped; the remaining GEMM
shape re-sweep lands only the ffn NT32 twin). Ladder: **2k 259.4 -> 281.6 /
8k 222.9 -> 239.6 / 100k rebuild 191.8 -> 202.5 tok/s (= 30.6% of the 662
reference, was 29.0%)**. Decode untouched: Tier-1 60/60 x2 deterministic,
Tier-2 60/60, stock 59/59, **69.28 ms/cyc = 40.17 tok/s** (canonical
40.12-40.15) with the ship env active.

## 1. What shipped

- **Fork (fe666db)**: `NV_SMEM_CFG_AUTO_NAMES` scopes the CTASM AUTO
  occupancy carveout to matching program-name substrings. The daemon runs
  `NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg` — every prefill GEMM
  (pfg*/pfg2*/pfg3*, entry symbols) gets the smallest carveout with
  occupancy >= 2; decode/G_CYCLE kernels (k0_*, a_*, spk_*, accept, ...)
  keep the default min-fitting pick (global AUTO costs decode ~2.1%,
  measured in CTASM_INVESTIGATION.md; the NAMES gate sidesteps it).
  Empty AUTO_NAMES = old global-AUTO behavior (back-compat).
- **engine0 (9ff0b7a)**: `PF_NT32=1` (default-on) — the ffn M32 GEMM runs
  the NTILE=32 grid-growth twin `pfg_ffn_m32_nt32_hm_nw4k128` (grid 272 ->
  544 CTAs, 256 -> 128 threads, 43.5KB -> 26.1KB smem), BIT-IDENTICAL to
  the shipped M32 kernel (test_p8 gate, per-element k-order unchanged).
  Graph-cache key now (M32, DFILL, G3M, NT32).
- **Ship line**: canonical serve env + `PF_PREFILL=1 PF_GEMM3=1
  NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg`.

## 2. Per-item results

### Item 1 — prefill-scoped gating (LANDED)
test_p6 (real weights, synced min-of-10): ffn M32 0.615 -> 0.509 ms
(221 -> 265 GB/s amortized, +20%); iq3d/iq3o flat (80-CTA grids — see item
2). Bit-identity: all classes BIT-IDENTICAL (scheduling-only change).
Decode co-resident in the SAME daemon process is untouched by the name gate
(verified by the full decode Tier-1 gate under the ship env, above).

### Item 2 — grid growth (LANDED for ffn only)
Built NTILE=32 twins of the M32 family (build_p8.py). LAWS learned:
- **pf_gemm.cu has the same NTILE/NWARP=8 row-group law as pf_gemm3.cu but
  NO compile guard** — nw8+NT32 builds clean and faults on every SM
  (Multiple Warp Errors, the KSYM-class garbage-execution signature).
  NT32 must be built nw4 (NTHR=128). Cubin names now encode nw4 correctly.
- Measured (AUTO pfg TGT2, all BIT-IDENTICAL vs shipped): ffn nt32 **x1.13**
  (0.519 -> 0.459 ms; chunk-time 33.2 -> 29.4 ms x64); iq3d nt32 x0.92,
  iq3o nt32 x0.96, iq3d_r7/iq3o_r7 nt32 x0.97-0.98 — the 160-CTA nw4 twins
  do NOT beat the 80-CTA nw8 kernels even at 3 CTAs/SM co-residency
  (8-warp ILP per CTA wins over 2x-CTA packing for these shapes).
  => ffn ships NT32; iq3d/iq3o/iq3s stay NT64. TGT3 is worse everywhere
  (100KB carveout L1-shrink; ship stays TGT2 default).
- Small-N classes (iq3k/q4v g=16) not attacked — pool too small (~1-2ms).

### Item 3 — attention co-residency (FALSIFIED for the current zoo)
- pfa16 (1024thr): carveout is a NO-OP — threads-capped at 1 CTA/SM
  (1536 threads/SM on sm_86; 1024x2 > 1536). Measured flat: 2.80 -> 2.83
  ms @pos100336 (noise).
- pfa16ctl_nw16 (512thr, 128 regs — the P7D structure-neutral control):
  carveout fits (smem ~35KB -> 100KB cfg = 2/SM by smem) BUT 128 regs x
  512 threads = 65536 = the ENTIRE regfile -> **1 CTA/SM by REGISTERS**.
  Measured flat: 2.96 -> 2.96 ms. (Same math for pfa8/pfa8t64/pfa32nw16:
  all the P7D/P7F2 kernels sit at 128 regs.)
- Numerics note: ctl partials are NOT bit-exact vs pfa16 on synthetic data
  (pA relerr med 0.0 but max 5.5 — near-zero-denominator outliers), so it
  was never a drop-in anyway.
- **CONCLUSION: the attention pair needs a <=64-reg 512thr (or any
  co-resident) rewrite to use the unlock — the carveout alone cannot move
  it. P7F3's verdict stands: persistent-CTA/low-reg attention is THE
  remaining >=30ms/chunk lever.**
- Side finding: **pfc16/pfc16ctl combine REGRESSES under the carveout**
  (0.148 -> 0.177 ms) — L1 shrink hurts the small combine. This is why the
  ship gate is `pfg` (NOT `pf`).

### Item 4 — DBUF-M32 re-price under the unlock
Covered by the NT32 sweep + TGT2/TGT3 sweeps above: the only class that
wants a new shape is ffn (NT32, landed). The r7 m32 DBUF kernels at NT64
remain the pick for fd/out (r7 228-230 GB/s amortized at TGT2 — the r7
tier already absorbs the iq3d advantage; classic iq3d 199).

## 3. The ladder (readout-order law; first clean runs)

| length | P7F3 | **P8** | delta |
|---|---|---|---|
| 2k fresh (PF_TRUNC=2048, ids8k) | 259.4 | **281.6** (7.3s; chunk med 119.6ms) | **+8.6%** |
| 8k fresh (ids8k 7714) | 222.9 | **239.6** (32.2s; last-half med 142.3ms) | **+7.5%** |
| 100k rebuild | 191.8 | **202.5** (483.1s) | **+5.6%** |

Gates (all == P7F3 line-for-line): GATE A 12/60, CTRL 13/60 alpha 2.67,
GATE D 5/60, D2 0/160, F 4.261e-2 (8k-class) / 1.058e-3 (2k class), 100k
cur=4471 EXACT, drift rec 3.453e-2 / conv 5.998e-2 (the M32 reassociation
floor), decode-on-rebuilt 27/60 (tie-mine class), 12 chunk-graph rebuilds
(same cadence), pos0 chunk 101.8ms (was ~131), pos97k 187.6ms (was 195-197).
The 2k-arm EXIT=1 is the documented 2048-trunc tie-mine assert
(4649,43614) — known artifact, F is the robust signal.

## 4. Decode (must-not-break, verified)

Ship-env canonical run (~/p8_decode_gate.log): Tier-1 60/60 x2
deterministic, Tier-2 60/60 vs W2D+W2E, stock 59/59, alpha 0.892,
**69.28 ms/cyc = 40.17 tok/s** — canonical (40.12-40.15). The NAMES gate
leaves every decode kernel at the default carveout.

## 5. THE HONEST 600 STATEMENT

@100k: **202.5 tok/s = 30.6% of 662**. Per-chunk @100k ~161ms avg =
GEMM ~42-52 (post-P8: ffn 29.4 + iq3d_r7 19 + iq3o_r7 3.6 + small classes)
+ attention-family ~85-95 (unchanged — pfa launches ~45@pos0 -> ~90@pos97k
+ pfk_pre/pfc16/o-proj ~20-25) + scan ~6 + norms/scraps ~10-15.
- The CTASM unlock moved the GEMM family to 265-298 GB/s amortized (from
  221) with bit-identity, but the ATTENTION family — 60%+ of the 100k
  chunk — is REGISTER-CAPPED at 1 CTA/SM in every existing variant. The
  unlock cannot buy the mission's hoped-for ~2x attention overlap without
  a new low-reg kernel class (<=64 regs @ 512thr, or persistent CTAs).
- Realistic endpoint on this dext: **~210-220 @100k (32-33% of 662)** from
  the remaining safe margins (r7 fg/fu VRAM route, ring-depth-4, pfk_pre
  carveout checks). A 2x-class attention rewrite would put **~260-290
  (~40-44%)** in play. Full 662 parity stays structurally out of reach:
  the binding constraint is now the attention kernels' register/thread
  budget at 1 CTA/SM, then the KV read stream — not the scheduler, not the
  dext, not the GEMMs.

## 6. Ops

Daemon relaunched on the P8 line (ENGINE ~/w100k_serve_p8.log, API
~/api_p8.log): canonical serve env + PF_PREFILL=1 PF_GEMM3=1 +
NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg. Health/chat/FOLLOW_UP
verified at relaunch. Logs: ~/p8_g2k.log (8k-class gate), ~/p8_g2k_t.log
(2k rung), ~/p8_r100k.log (100k rebuild), ~/p8_decode_gate.log,
~/ctasm_* (the A/B evidence), ~/p8_attn_*.log class runs via attn_p8.py.
Harnesses: engine0/{build_p8,test_p8,attn_p8}.py. PF_NT32=0 restores the
P7F3 path; dropping the two NV_SMEM vars restores the default carveout.
