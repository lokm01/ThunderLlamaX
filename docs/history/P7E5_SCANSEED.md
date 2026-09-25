# P7-E5 — The nonzero-seed scan bug: algebra EXONERATED, six fp16 carriers split (hi-lo law), the conv-row-2 channel isolated; SC does NOT ship

Status: **the P7E4 root-cause lead was HALF right. The chunked recurrence's
window-0 S_in handling is algebraically CORRECT (full derivation below —
O_inter/Qe, d/T(beta 2^g Y), S'/2^g_end all symbolically verified against the
pfs16 oracle; layouts byte-matched). The failure is NUMERICAL, not structural:
pfca/pfcb carry MATURE-magnitude values (snapshot-scale |S|~30, |v|~40,
|d|~20) through BARE fp16 where the M32 oracle keeps fp32 registers; the 5e-4
relative error amplifies ~20x through the 48-block stack via (v-kd) /
(U - T(beta 2^g Y)) cancellation. SIX carriers were split hi+lo fp16
(>=21-bit effective, 3-pass MMAs, zero smem growth — the HI-LO LAW): the
incoming state S (s16), v (vr_g), U, Y, d (a per-(h,c,vq) dz global zone),
T and M. The REC-SEEDED divergence is FIXED (P2-class 2.4e-3 medrel, was
5.5e-2; block-0 window-0 scan now matches a fp64 sequential oracle at <=1e-3
abs with |v|~30). BUT a SECOND, independent channel survives: divergence iff
convlive ROW 2 (the t-1 halo row) is nonzero — content-independent (row0's
values placed in row2 still break it), magnitude-dependent (x0.01 clean),
NOT in block-0/window-0/tokens-0-15 (probe-verified at quantized-input
level), NOT any of the six split carriers, NOT row-order, NOT the conv-read
addressing (byte-verified identical to pfs16). PF_SUPER STAYS 0 — the gates
still fail. The daemon ships the P6 canonical unchanged.**

## 1. The algebra derivation (mission step 1 — VERDICT: CORRECT)

From pfs16's exact op order (S in R^[K][V], o_t = qhat_t . S_t POST-update):
  S_i = A_i S_{i-1} + khat_i x dl_i,  dl_i = beta_i (v_i - (A_i S_{i-1})^T khat_i),  A_i = 2^(lam_i log2e)
Unrolling with g_i = cumsum(lam. log2e) and S_0 = the INITIAL (snapshot) state:
  u_i = beta_i v_i - beta_i 2^(g_i) y_i - sum_{m<i} B[i][m] u_m,  y_i = khat_i^T S_0,
  B[i][m] = beta_i (khat_i.khat_m) 2^(g_i-g_m)
  => u = (I+B)^-1 (beta V - beta.2^g.Y)  [pfcb: d = U - T @ (beta 2^g Y)]  EXACT MATCH
  o_i = qhat_i 2^(g_i) . S_0 + sum_{m<=i} (qhat_i.khat_m) 2^(g_i-g_m) u_m
      = (Qe_i . S_0) + (M u)_i  [pfcb: O = M @ d + Qe @ S16, UN-decayed S16]  EXACT MATCH
  S_end = 2^(g_end) S_0 + Khat^T (2^(g_end-g) . d)  EXACT MATCH
The window-0 S_in is READ from the live rec buffer (same code path as the
P7E3-proven inter-window carry); layouts (rec[h][v][k], convlive
oldest-first, LDK/LDTC strides) byte-verified against pfs16. The P7E4 lead
"window-0 assumed S_in==0" is FALSE. The bug was precision.

## 2. The empirical discrimination tree (all deterministic, 512-tok SC-vs-M32)

Seed experiments (pf_scdbg5g.py — P0 full snapshot | P1 rec fp16-grid | P2
rec-only; pf_scdbg5c.py — C0 full | C1 row0 | C2 row1 | C3 row2 | C4
all-rows=row0-content | C5 x0.01; rec=0 in all C-variants):

| variant | rec61 medrel | verdict |
|---|---|---|
| P0 snapshot seed (pre-fix) | 5.5e-2 | the P7E4 repro |
| P1 rec grid-rounded | 3.1e-2 | s16 fp16 quantization real but ~half |
| P2 rec-only (conv=0) | 2.4e-3 | rec-seeded path CLEAN after splits (was 2.7e-3) |
| C1 row0 only | 2.5e-3 | clean |
| C2 row1 only | 2.8e-3 | clean |
| C3 row2 only | 4.4e-2 | DIRTY — half of full |
| C4 all rows = row0 content | 4.3e-2 | DIRTY — row-ORDER killed, still breaks => position not content |
| C5 all x0.01 | 2.4e-3 | clean — magnitude-dependent |

Forensics (pf_probe.py — wraps pfcb, snapshots block-0 inputs/scratch/scan
output at first launch; numpy oracle = pfs16 formulas on the dumped qkv):
- pfca preprocess (conv+silu+norms, tokens 0-3, |v|<=30): EXACT vs oracle
  (kh 1e-4 = its own fp16 dump, vr hi+lo 1e-6). The halo reads are byte-
  identical to pfs16's — NOT an addressing bug.
- window-0 scan output (oout rows 0-15 vs fp64 sequential oracle on the
  dumped kh/vr/qe): maxabs 1.5e-6 (t=0) growing to 7.8e-4 (t=15) — the WY
  reassociation is CORRECT; the residual is the amplification floor of the
  remaining fp16 inputs (kh/qh dumps ride bare fp16 — content-independent
  noise x large v) — this residual is what the C3 channel looks like from
  inside block 0, but see 4: it cannot be the whole C3 story (rows 0/1 would
  diverge equally).

## 3. The HI-LO LAW (banked; the six splits)

Law: any value that can carry snapshot-scale magnitude and rides the WY
pipeline must be split hi+lo fp16 (value = hi + lo, both fp16, >=21-bit
effective) and MMA'd as 3 passes (hi.hi + hi.lo + lo.hi; lo.lo ~2^-42
skipped). Implemented via SEQUENTIAL smem reuse (no pfcb smem growth —
34336B law intact; pfca eager-only grew 0B) + a per-(h,c,vq) dz GLOBAL zone
for the d halves (must survive the Qe-phase s16 rebuilds):
  S   (s16)  — rebuilt hi/lo from the fp32 window state, Y and Qe@S 3-/2-pass
  v   (vr_g) — hi/lo dumps; bvt = beta.(v_hi+v_lo) split at build
  U   (u_g)  — 3-pass T@bV; hi/lo dumps; d fixup reads both
  Y   (sy)   — scaled bg.Y stored hi/lo from the fp32 acc (NO raw-fp16
               roundtrip); lo stashed in the dead s16 region, T@sy 3-pass
  d   (dz)   — hi/lo in dz (survives Qe phases); M@d and KhatT@(sg.d)
               3-pass; sg.d built in fp32 from the halves
  T,M (t_g/m_g + t16 smem) — hi/lo dumps; 3-pass in U and d and O
Scratch: HC_BYTES 107024 -> 193024-class growth (193040 final:
+u_lo/vr_lo/m_lo/t_lo/dz). pfca 64 regs, pfcb 114 regs, 0 spill, smem
43520/34336 (laws OK). Zero-seed-class worlds UNCHANGED (C1/C2/C5 logits
8.1-9.4e-4 = the P7E4 zero-seed class 8.1e-4).

## 4. The remaining conv-row-2 channel (NEXT SESSION — sharply scoped)

Facts: dirty iff convlive row 2 (t-1) nonzero; content-independent (C4);
magnitude-gated (C5); rows 0/1 inert; NOT block-0/window-0/tokens-0-15
(probe); NOT the conv-read convention; NOT any split carrier (three fix
rounds moved it 0%); deterministic. Leads, in order:
 (a) extend pf_probe's oracle to tokens 16-63 and windows 1-3 of block 0
     (only 0-15/c=0 checked so far); then block 1/2 window 0 — find the
     FIRST diverging (block, window, token) triple; the probe wrapper
     pattern generalizes (snapshot at each pfcb call k).
 (b) the pfcz writeback -> chunk-2 window-0 boundary (writeback CONTENT
     should be content-independent code, but verify with a 256-token
     C3 run — S1-style single chunk: the ORIGINAL scdbg5n S1 diverged, so
     chunk 2 is NOT required for the failure => prioritize (a) at c=0).
 (c) kh/qh hi-lo (content-independent x |v| amplification — predicted
     ~1e-3-class at t=15 per the probe residual — but it CANNOT explain
     rows 0/1 clean vs row 2 dirty; expect it to lower the floor, not fix).
 (d) an fp32-parity scan harness: run pfcb with all MMAs forced single-
     precision... (not available on sm_86 tensor cores; the hi-lo IS the
     fp32 emulation — already done for the six carriers).

## 5. Ship decision + service

- PF_SUPER=0, PF_G3SC=0 — UNCHANGED (gates fail: the staged A2/B failure
  reproduces through the row-2 channel). P6 canonical: 165.3 tok/s 100k
  stands. Benchmarks NOT run (conditioned on shipping SC).
- Daemon relaunched FRESH on the P6 canonical recipe (M1C + PF_PREFILL=1
  PF_GEMM3=1 PF_SCANC=1 PF_SUPER=0 PF_G3SC=0 MTP_KERNARGS_MB=256 +
  M1A_SERVE=1 + api_server :8080); health + one chat verified.
- Cold-cycle note: TWO dext-degradation fault classes appeared mid-session
  (boot-time faults AFTER clean runs, before any scan kernel) — cleared by
  the scheduled poweron+shutdown cycle per the P7E4 dock-power law.

## 6. Files

- engine0/pf_scanchunk.cu — the hi-lo law implementation (this session)
- engine0/pf_scdbg5g.py (seed variants), pf_scdbg5c.py (conv-row variants),
  engine0/pf_probe.py (block-0 wrap forensics + fp64 oracle) — NEW
- pf_prefill.py — HC64/scscr sizes for the extended scratch
- Logs: ~/p7e5_seedexp.log, ~/p7e5_seedfix*.log, ~/p7e5_convexp.log,
  ~/p7e5_convfix*.log, ~/p7e5_probe*.log, ~/p7e5_zeroreg.log
