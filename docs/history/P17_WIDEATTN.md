# P17 — THE WIDE-M ATTENTION: SHIPPED (PF_ATTNW=1 default-on; bit-identical, 2k 373.5 / 8k 347.3 / 100k 248.2 = 37.5% of 662 @100k)

Status: **The wide-M attention (ROWS=32/64 tokens per attention window) is
SHIPPED. One w64h window (HRP=1, 312 CTAs) replaces the 4x t32 windows at
low pos; the HYB plan26 arm swaps to w64 (HRP=2, RMAX=128) at pos >= 8192.
Per-row math is VERBATIM from the shipped t32 (pf_attn32c.cu) — standalone
corr nz=0/393216 BIT-IDENTICAL det x2 for w64/w64h (w32 nz=0/196608), the
2k/8k/100k gates all reproduce the banked P15 lines EXACTLY (8k gate
line-for-line incl. token sequences and end-logit values; 100k cur=4471
EXACT, drift same class, decode 60/60). Ladder: 357.1/306.3/235.0 ->
**373.5/347.3/248.2** (+4.6/+13.4/+5.6%). The attention pool (P15: 24.7ms
shipped + in-graph growth) is now 0.77ms @2k-class / 8.48ms @100k-end
(standalone) and the 4-window shared-scratch serialization is GONE (128
launches -> 32/chunk).**

## 1. Inherited-work review (the previous agent timed out mid-mission)

Left in the tree (all uncommitted): engine0/pf_attnw.cu (the kernels),
build_p17.py, p17_iso.py / p17_bisect.py (fault-chase probes, superseded),
pf17_probe.py (the corr+bench probe), the pf_prefill.py PF_ATTNW wiring,
11 cubins, and a 100k gate run in flight (~/p17_r100k.log). The pf_prefill.py
comment claimed "corr nz=0/393216 det x2" and bench "w64-S13 beats every S26
shape at pos>=8k: 4.29 vs 4.63 @48k, 8.41 vs 9.07 @100k-end" — **no log of
either existed** (the corr/bisect attempts at 22:44/22:48 had FAULTED at the
control pair — the bare-world first-pair fault on the just-EFI-cold-cycled
machine, boot ~22:41). Both claims were RE-RUN this session and CONFIRMED
(sections 3-4): corr numbers reproduced exactly (nz=0, det x2); bench
reproduced within noise (4.34 vs 4.67 @48k, 8.48 vs 9.05 @100k-end).

## 2. The kernels (engine0/pf_attnw.cu; build_p17.py)

VERBATIM port of the PROVEN pfa32c t32 skeleton with the ROWS/HRP/NW
parameterization that was hardcoded (the P15 Stage-0 law: grid-doubling the
t32 reads OOB because ROWS was baked). Per-row math, online-softmax TILE
grouping, S-split/combine structure UNCHANGED. Key design points:
- ROWS/HRP/NW are -D params (RMAX = HRP*ROWS rows/CTA; NHP = 6/HRP; the
  no-gridDim law keeps NCTA = 4*S*NHP compile-time).
- QK phase LOOPS over QK_TILES (warp stride NW) — at RMAX>32 there are more
  than NW 16x8 out-tiles; each tile still computed exactly once, same mma
  k-order (independent outputs).
- PV db mapping generalized: NW<=16: db=2*warp+(ti&1); NW=32: db=warp.
- corv lives in the K plane (dead after QK of the same tile) — saves RP*4B
  so ROWS=64/RMAX=128 lands EXACTLY at the 48KB static smem cap.
- 1024-thread staging variant (NTHR==1024): one int2 per thread.
- Bit-identity argument: rows beyond a t32 window's 16 are masked by the
  SAME causal law (ka <= pos + t); fully-masked tiles evolve ms/ss/acc by
  *1.0 / +0.0, so a ROWS=64 window at pos P equals the 4 concatenated t32
  windows at P, P+16, P+32, P+48 — PROVEN by corr (below).

| shape | -D | CTAs | thr | RMAX | smem | verdict |
|---|---|---|---|---|---|---|
| pfaw_w32_s13_100k | ROWS=32 HRP=2 NW=16 | 156 | 512 | 64 | 32KB | BIT-IDENTICAL |
| pfaw_w64_s13_100k | ROWS=64 HRP=2 NW=16 | 156 | 512 | 128 | 48KB exact | BIT-IDENTICAL — the HIGH arm |
| pfaw_w64h_s13_100k | ROWS=64 HRP=1 NW=16 | 312 | 512 | 64 | 32KB | BIT-IDENTICAL — the LOW arm |
| pfaw_w64q_s13_100k | ROWS=64 HRP=2 NW=32 | 156 | 1024 | 128 | 48KB | **BROKEN: nz=12065/393216, det x2 FALSE (nondeterministic)** — banked negative, NOT shipped |
| pfcw32/pfcw64/pfcw64h combines | PFC_T32 | 24 | 256 | — | — | t < ROWS loop, same epilogue/op-order as pfc16t |

Scratch law: pmW/psW = 19968 f32 slots, pAW = 19968*256 f32 — EXACTLY
shared by both arms (312 CTAs x 64 rows = 156 x 128 = 19968), so the HYB
swap needs no second scratch set. 2k corr twins built for w32 only.

## 3. Standalone validation (pf17_probe.py corr100k; ~/p17_corr100k.log)

Refs = 4x (t32 attn+combine) at pos 100224/100240/100256/100272
concatenated (64, 6144) on real-scale synthetic kv8; wide = one window at
100224; poison-first, det x2, warm-up pair first (the P17 bare-world law —
see section 6):
```
w32  rows=32: BIT-IDENTICAL nz=0/196608 maxabsdiff 0 | det x2 True | poison 0
w64  rows=64: BIT-IDENTICAL nz=0/393216 maxabsdiff 0 | det x2 True | poison 0
w64h rows=64: BIT-IDENTICAL nz=0/393216 maxabsdiff 0 | det x2 True | poison 0
w64q rows=64: nz=12065/393216 maxabsdiff 2.21 relerr max 7.96e+01 | det x2 FALSE  <- BROKEN
```

## 4. The bench (pf17_probe.py bench; ~/p17_bench.log; ms per 64-chunk-equivalent, synced min-of-5)

| shape \ pos | 2032 | 48000 | 100288 |
|---|---|---|---|
| 4x t32-S13+pfc (shipped low arm) | 1.68 | 9.27 | 9.98 |
| 4x t32-S26+pfc (P11 plan26 high arm) | — | 5.08 | 10.12 |
| 2x w32-S13 | 1.36 | 4.50 | 8.94 |
| 2x w32-S26 | — | 4.67 | 9.05 |
| 1x w64 (SHIPPED high arm) | 1.32 | **4.34** | **8.48** |
| 1x w64h (SHIPPED low arm) | **0.77** | 4.54 | 8.86 |
| 1x w64q | 1.48 | 5.46 | 10.97 |

w64-S13 beats EVERY S26 shape at pos>=8k (the HYB arm is now a ROWS swap,
not an S swap). w64h is the low-pos winner (312 CTAs fix the underfill).
The 1024-thread w64q loses everywhere it is not broken — NW=32 is dead.

## 5. The gates (all green, readout-order law; first clean run = the log)

| gate | P17 wide-M | banked P15 | verdict |
|---|---|---|---|
| 2k (PF_TRUNC=2048) | **373.5 tok/s** (5.5s), med 172.2 | 357.1 (5.7s), med 176.1 | +4.6%; F-relerr **1.058e-03 EXACTLY** banked; the (4649,43614) tie-mine assert fires identically (EXIT=1 class) |
| 8k | **347.3 tok/s** (22.2s), med 190.9 | 306.3 (25.2s), med 227.2 | +13.4%; **line-for-line**: F 4.282e-02, GATE A 12/60 (div@0 6545/22546, gaps 0.03125/0.0390625), CTRL 13/60 alpha 2.67, D 5/60, D2 0/160; end-logits identical to the printed digit (9956 15.141 / 29877 15.023 / gap 0.1172); tokA/B/C/D sequences identical |
| 100k rebuild | **248.2 tok/s** (394.1s; fill_draft 401.8) | 235.0 (416.1s; 379.9) | +5.6%; **cur=4471 EXACT (match=True)**; drift max rec 5.348e-3 / conv 3.615e-3 (gate <=1e-2; banked 5.458e-3/3.547e-3 — same class); rebuilt-state decode **60/60** |

In-model chunk deltas (the honest attribution): at 8k the saving GROWS with
position (-4.9ms @640 -> -9.4 @1280 -> -17.5 @2560 -> -32.3 @5120 ->
**-44.5 @7040**); a ~-16ms CONSTANT at pos 0 (100k run: 224.4 -> 207.9);
100k-end 327.9 -> 300.5 (-27.4). Mechanism = (a) the 4 t32 windows shared
pm/ps/pA scratch -> forced attn,combine,attn,... 8-kernel serialization per
layer (now ONE launch; 128 -> 32 launches/chunk), (b) ROWS=64 amortizes the
KV stream over 4x the rows (the SC-256-class per-token amortization finally
landed in attention). Standalone min-of-5 UNDERSTATES the in-graph t32 cost
(queue/wave-tail effects) — the totals are ground truth.

## 6. Fault postmortem + laws

- The 22:44/22:48 corr/bisect FAULTS were the **bare-world first-pair
  fault** on the just-booted machine (EFI cold cycle ~22:41): the first
  (kv,sc,qw) bare-world control pair faulted. LAW (already encoded in
  pf17_probe.py): warm-up pair BEFORE any corr sequence; every buffer
  poison+up paired, pf10-shaped.
- w64q (NW=32, 1024thr) is nondeterministic-wrong — do not retry without a
  redesign; NW=16 is the proven lane width.
- The scratch-size coincidence (312*64 = 156*128 = 19968) is LOAD-BEARING:
  the HYB row-swap shares pmW/psW/pAW between arms with zero extra VRAM.
- Readout-order law respected everywhere (gates from the first clean run).

## 7. THE 400 STATEMENT (measured arithmetic, no hope)

Ladder: 2k **373.5** (56.4% of 662 ref-class @2k... ref is 662 @100k) /
8k **347.3** / 100k **248.2** = **37.5% of 662 @100k** (was 35.5%).
400 @100k = 244.5s over 97810 tok = **160ms avg per 64-chunk** (incl.
dfill + tail). Current: 300.5ms @100k-end chunk, avg ~258ms.
Pools at 100k-end (P15 attribution + P17 deltas):
- weights-class (GEMM 124.7 + scan 27.6 + norms 5.1 + pre 1.8 + og/qkv/oa/
  comb 29) ~ 161.9ms isolated (P15 sum minus attn 24.7); in-graph ~207.9
  measured at pos 0 WITH wide attn 0.77.
- wide attention: 8.48ms standalone @100k-end.
- **position-growth pool: 92.6ms (pos0 207.9 -> 97k 300.5) of which
  standalone-attn growth is only ~7.7 — ~85ms of in-graph growth is NOT
  standalone attention (dfill-vs-trunk-vs-queue attribution OPEN).**
Rungs, priced with in-graph rates:
1. **M=128+WY** (chunked-scan numerics): GEMM 124.7 -> ~100-112 (m64
   coverage -13 measured-ratio arithmetic + one more m-step at 0.85-0.9x),
   scan 27.6 -> ~18-20, launch/norm floor -5 => **-35 to -45ms** =>
   avg ~215ms => ladder ~275-290.
2. **w128 attention**: -3ms more (8.48 -> ~5-6). Minor.
3. **The position-growth pool (~85ms @97k)**: THE 400 decider. If fully
   recovered, avg -> ~130-160 => **380-420**. Composition unattributed:
   dfill grows with pos (draft replay 243.4s = 62% of the 394.1s wall) and
   in-graph queue growth are the suspects; NEXT MISSION = the 100k-end
   growth attribution (isolate dfill cost vs trunk cost per position
   decile), then attack (dfill restructure / persistent CTAs / queue work).
Honest endpoint: **~280-300 without the growth pool; 400 only if it largely
yields.** The wide-M attention was growth-pool step 1 (-27.4ms of it at 97k).

## 8. Ship record

Canonical daemon env = the P15/P16 line + **PF_ATTNW=1** (requires
PF_M64=1; kill-switch PF_ATTNW=0 restores the 4x t32 windows). Daemon
relaunched on the full line (~/w100k_serve_p17.log), api_server.py on
127.0.0.1:8080; FRESH + FOLLOW_UP verified post-relaunch. Files committed:
pf_attnw.cu, build_p17.py, p17_iso.py, p17_bisect.py, pf17_probe.py,
pf_prefill.py (the wiring), the 11 pfaw/pfcw cubins, this doc.
