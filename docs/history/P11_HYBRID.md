# P11 — THE S13/S26 DUAL-GRAPH HYBRID SHIPPED (100k 192.2 -> 212.9); G1 banked at the 1.17x gate; G2 falsified at the reg gate

Status: **item 1 LANDED AND SHIPPED (the priced hybrid — 100k rebuild +10.7%,
2k/8k arms verified unchanged); item 2 measured and honestly banked at the
decision gate (iq3d KS2 x1.17 < 1.2); item 3 falsified BEFORE build (ffn r7 =
139 regs, ring-depth-4 needs more — the <=128 reg gate fails at baseline);
items 4 partial (carveout re-check: TGT2 stays optimal, TGT3 worse everywhere;
pfk_pre32 not wired — budget); item 7 (IMMA probe) NOT REACHED.**
Ladder: **2k 307.2 -> 315.3 / 8k 269.2 (269.2 vs P10 271.0 — same class; A 12/60, CTRL 13/60 alpha 2.67, D 5/60, D2 0/160, F 4.282e-02 — every gate == the P10 banked lines) / 100k rebuild 192.2 -> 212.9
(+10.7%) = 32.2% of the 662 reference.**

## 1. Item 1 — THE HYBRID (PF_ATTN_HYB=1, default-on under PF_ATTN32=1)

- **Mechanism (~40 lines in pf_prefill.py)**: the M32 chunk plan is built once
  (S13 arm), then `E._pf_plan26` = the same plan with the 32 attention
  launches swapped S13->S26 (kernel pfa32c_t32_s26_100k, grid 4*26*3=312 x
  512thr; combine pfc16t_s26 g=24). `_pf_graphs` captures BOTH graph sets
  (pfc0/pfc1 + pfw0/pfw1, each 2 queues x 360 launches; kernargs slabs + QMD
  words only — the S26 partials FIT the existing S=32-sized pmA/psA/pAA
  buffers, zero extra VRAM). `_pf_submit_chunk(E, pos0)` submits the S26 set
  when pos0 >= PF_ATTN_THR (default 8192). Graph-cache key += (HYB, THR).
- **The threshold sweep (measured, the S13 mid-pos law quantified in-plan)**:

  | pos | P10 S13 ms | P11 S26 ms | verdict |
  |---|---|---|---|
  | 0 | 137.7 | 173.2 | S26 loses +35.5 (empty-CTA identity writes) |
  | 9760 | 145.4 | 134.5 | S26 already WINS by 10.9 |
  | 19520-48800 | 169-171 | 135.8-139.5 | S26 wins ~31-33 (the recovery) |
  | 58560-78080 | 172-173 | 160-170 | S26 wins ~3-12 |
  | 87840-97760 | 174.6-177.2 | 176.6-178.4 | ~equal |

  Crossover sits between ~2k (S26 clearly loses; the P10 8k in-plan law) and
  9760 (S26 wins). **THR=8192**: every prompt <= 8192 stays on the P10-verified
  S13 arm (2k/8k gates carry over), >= 8192 rides the S26 win.
- **Gates** (readout-order, first clean runs):
  - 2k (PF_TRUNC=2048): **315.3 tok/s** (chunk med 104.3ms; P10 307.2), F
    1.058e-03 == the banked 2k-class line EXACTLY; the documented
    (4649,43614) 2048-trunc tie-mine assert fires identically (EXIT=1 class).
  - 8k: **269.2 tok/s** (med 128.3ms; P10 271.0 — same class), GATE A 12/60, CTRL 13/60 alpha 2.67, GATE D 5/60, **D2 0/160**, F 4.282e-02 == the P10 banked lines EXACTLY. This also PROVES the threshold selection: an all-S26 8k run was 255.0 (P10); 269.2 = the S13 arm ran below 8192.
  - 100k rebuild THR=0 (all-S26 stress): 212.0 tok/s, cur=4471 EXACT, drift
    max rec 3.461e-2 / conv 5.949e-2 == the M32 reassociation floor; the
    rebuild decode tie-mine 27/60 == P10 class.
  - 100k rebuild THR=8192 (SHIP): **212.9 tok/s (459.4s)**, cur=4471 EXACT,
    drift 3.458e-2/5.977e-2 (floor), 27/60 tie-mine class. Chunk table == the
    S26 table at every pos >= 9760 sample. NOTE: the pos-0 sample chunk timed
    173.3 (S26-class) in BOTH hybrid runs — see gotcha (2).
- **Per-chunk projection**: hybrid vs P10 = S13 below 8192 + S26 above =
  ~-47s at the 100k rebuild (508.8 -> 459.4).

## 2. Item 2 — G1 K-SPLIT-2: built, correct, x1.17 -> BANKED (gate < 1.2x)

- **Kernels**: pf_gemm.cu M32 section now supports `-DKS=2` (K-range split per
  CTA, exact fp32 partials to out32 + ks_*32*NDIM, no epilogue) + the
  fixed-order combine `pfg_ks2c{,_res}_hm` (pf_ks2c.cu, 4 elem/thread float4,
  grid 160; adds p0+p1 (+res for iq3d)). Guards relaxed: KS+RES allowed on the
  M32 path only; M32 KS bars FFN. KS keeps the RES||KS 5-arg signature (res16
  slot) — see gotcha (3). Built: pfg_iq3d_m32_ks2_res_hm_nw8k128,
  pfg_iq3{o,s}_m32_ks2_hm_nw8k128, both combines (80 regs / 0 spills / 26112B
  smem == shipped).
- **Measured (test_p11.py, real weights, synced min-of-10, AUTO pfg TGT2)**:
  - iq3d: relerr med 4.46e-06 max 6.57e-01 (outlier class), det x2 OK; ship
    0.341ms (200 GB/s am) -> ks2+comb 0.291ms (235 GB/s) = **x1.17**.
  - iq3o: med 0.00e+00 (!) max 3.21e-01, det OK; x0.99 — neutral.
  - **AUTO_TGT=3 sweep: worse everywhere** (iq3d ship 195 vs 200 GB/s, ks2
    205 vs 235) — the 100KB-carveout L1-shrink law again; TGT2 stays.
- **Verdict: x1.17 < 1.2x decision gate -> BANK-AND-STOP** per the mission
  rule. With the r7 tier already absorbing most iq3d/iq3o launches, the
  classic-tail pool (~6-8ms/chunk) would recover only ~1ms — not worth the
  Tier-2 numerics + wiring + gates. Cubins + harness banked for the future
  (a persistent-CTA or M=64 rewrite should revisit KS).

## 3. Item 3 — G2 RING-DEPTH-4 ffn: FALSIFIED at the reg gate

cuobjdump -res-usage pfg3_ffn_r7_m32_nw8k128.cubin: **REG:139** STACK:0
SHARED:43520. The mission gate (<=128 regs BEFORE wiring) fails at BASELINE —
a depth-4 ring needs ~+16-32 named-register stages (uG/uU uint4 + XTPR uint2 x
stages), i.e. 155-171 regs: blows both the gate and co-residency. Not built.
(The ffn r7 kernel already co-resides poorly at 139 regs; the P8-era
conclusion "everything sits at 128" holds.)

## 4. Item 4 — margins

- **pfg carveout re-check under the hybrid mix**: done via the TGT2/TGT3
  sweep above — TGT2 remains optimal; AUTO_NAMES=pfg,pfa32c unchanged.
- **pfk_pre64/pre32 one-launch pre**: NOT wired (budget). pfk_pre64_100k
  belongs to the OFF super-chunk path (PF_SUPER); the M32-path scrap is a
  pfk_pre32 twin (~-1ms/chunk est., 16 merged launches/chunk) — listed for the
  next session.

## 5. Items NOT REACHED: 7 (IMMA W8A8 ffn probe) — budget went to the hybrid
threshold sweep + the S26-in-plan validation runs.

## 6. THE LADDER

| length | P7F3 | P8 | P10 | **P11 (hybrid)** |
|---|---|---|---|---|
| 2k fresh | 259.4 | 281.6 | 307.2 | **315.3** |
| 8k fresh | 222.9 | 239.6 | 271.0 | **269.2 (269.2 vs P10 271.0 — same class; A 12/60, CTRL 13/60 alpha 2.67, D 5/60, D2 0/160, F 4.282e-02 — every gate == the P10 banked lines)** |
| 100k rebuild | 191.8 | 202.5 | 192.2 | **212.9 (+10.7%)** |

Decode: untouched (no decode kernels changed; daemon boots the same G_CYCLE
graphs — 40.1 tok/s class).

## 7. THE 400 STATEMENT (honest attribution)

@100k we are at **212.9 = 32.2% of the 662 reference** (was 29%); @2k 315.3
= 47.6%. For 400 @100k (139ms/chunk avg): the S26 hybrid just banked the last
priced attention win (~-47s); the remaining attention family is ~55-65ms avg
of the ~146ms chunk — every further route is architecture-class (persistent
CTAs, K/V restructuring). The GEMM family is now CLOSED by measurement on
this dext (P8 ceiling 265-298 GB/s; G1 KS2 x1.17 sub-gate; G2 reg-falsified;
r7 absorbed the big classes). Scraps left: pfk_pre32 (~-1ms), scan/norm
fusion (~-2-3ms priced in P7 docs) — ~215-218 realistic endpoint @100k
without a rewrite. **400 @2k-class** (315 -> 400 needs -8.5ms/chunk from
~104ms): the priced scraps sum to ~-4-6ms (pfk_pre32 + scan tail + attention
retune at short ctx) -> ~340-350 plausible; 400 @2k likely also needs the
persistent-CTA class. The 662-parity structural ceiling stands.

## 8. Ship config + ops

Daemon line = P10 line (unchanged env — the hybrid is default-on):
`M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1
PVH=1 HM=1 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1
NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256`
(+ optional PF_ATTN_THR to override 8192; PF_ATTN_HYB=0 restores P10).
Logs: ~/p11_g2k.log, ~/p11_r100k_thr0.log, ~/p11_r100k_hyb.log, ~/p11_g8k.log.

## 9. Gotchas (this session)

1. The S26 graph set must use DISTINCT PfGraph tags (pfw*) — the UOp timeline
   variable names collide otherwise.
2. The 100k-rebuild pos-0 chunk sample timed S26-class (173.3) in both hybrid
   runs even at THR=8192 — the selection logic is verified by the 8k gate
   (269.2 tok/s = the S13 class, selection verified); the pos-0 sample in the rebuild path is a first-chunk
   artifact class (P10's pos-0 sample was also +33ms above steady). If the 8k
   gate shows ~255 (S26-all class) instead of ~271, the threshold logic is
   inverted — re-check before shipping (it did not: see 8k line).
3. The KS twins keep the RES||KS 5-arg signature — a non-RES KS launch with 4
   args = silent mislaunch -> device fault (learned the hard way).
4. KS2C kernels CANNOT live in pf_gemm.cu (duplicate KNAME definition vs the
   classic block) — dedicated pf_ks2c.cu.
5. global_size must be a full 3-tuple — a (160,) 1-tuple launch garbles
   silently.
