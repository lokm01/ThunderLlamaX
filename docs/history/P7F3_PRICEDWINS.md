# P7F-3 — THE PRICED WINS: G3M r7 m32 DBUF tier SHIPPED (+4.4-5.1% ladder); the rest re-priced by measurement

Status: **mission item 1 (DBUF-M32 GEMM hybrid) is LANDED, default-on, bit-identical,
gated at 2k/8k/100k, shipped in the daemon and verified end-to-end. Ladder:
2k 247 -> 259.4 / 8k 212.0 -> 222.9 / 100k rebuild 183.8 -> 191.8 tok/s
(= 29.0% of the 662 reference @100k, was 27.8%). Items 2/3 (fusion, norm diet)
were NOT landed — re-priced DOWN by P7F1 evidence (see verdict); the ring-depth-4
sweep is designed but unbuilt (budget). The 1.3-1.6x P5/P6 pricing for the GEMM
hybrid did NOT materialize at M=32: the honest in-session P7B number (1.10-1.14x
per kernel) is what transferred, AND full coverage is VRAM-blocked (measured fault
at the 9.3GB both-live point — the decode G_CYCLE graphs pin the original weight
buffers resident).**

## 1. What shipped (commit 4c5c869)

- `engine0/pf_prefill.py` — **G3M tier**: when `PF_G3M` (default: follow
  `PF_GEMM3`, =1 in the ship line) and M32, `ensure()` performs a
  **budget-ordered both-live packed7 swap** before building the chunk plan:
  upload r7 tensors until `PF_G3M_MB` (default 4300) is exhausted, priority
  fd > gate > out > q/k > (fg/fu pairs if space). Plan entries switch to the
  P7B-validated bit-identical r7 m32 cubins per membership:
  `pfg3_iq3d_r7_m32` (fd, RES), `pfg3m_gdnqg_r7_m32` (qkv classic + gate r7),
  `pfg3_iq3o_r7_m32` (out), `pfg3m_attnqkv{i3,q6}_r7_m32` (q/k r7, v classic).
  Graph-cache key now `(M32, DFILL, G3M)`. Coverage at 4300MB:
  **160 tensors / 4.28GB = fd:64 gate:48 out:24 q:8 k:16** (~42% of the
  repackable GEMM bytes; the fg/fu FFN twin — another 5.0GB — does not fit).
- `engine0/pf_gate2k.py` — the PF_BATCH prefill-time print moved BEFORE the
  decode arms (the pre-existing 2048-trunc A2 tie-mine assert — P7F1's g2kc
  hit the identical `(4649, 43614)` — no longer eats the ladder readout).

## 2. Gates (readout-order law: first clean runs)

| gate | result |
|---|---|
| **G3M-BIT** (2k, daemon-class world, SCDBG dumps) | **BIT-IDENTICAL (all keys)**: logits(248320) f16 + rec/conv f32 x5 blocks (incl. blk62 with fd+gate+out swapped), PF_GEMM3=1 vs =0 |
| **PG-BIT** (8k, G3M world, PF_PG_BIT=1) | **BIT-IDENTICAL (all keys)** eager vs captured; eager 241 chunks mean 161.0ms vs graph 144.1ms = 1.12x (capture win unchanged) |
| 8k full gate set | F 4.26e-2-class line preserved; GATE A **12/60**, CTRL **13/60 alpha 2.67**, GATE D **5/60**, D2 **0/160** — line-for-line == P7F1 |
| 100k rebuild | **509.8s = 191.8 tok/s**; **cur=4471 EXACT**; state drift max rec 3.453e-2 / conv 5.998e-2 (== P7F1's 3.45e-2/6.0e-2, the M32 reassociation floor); decode-on-rebuilt 27/60 (tie-mine class, unchanged) |
| VRAM ceiling probe | PF_G3M_MB=9300 (full both-live) **FAULTS during upload** (`Device fault detected` at `_copyin`, the OOM-class; machine survived, no lock). 4.28GB clean at 2k/8k/100k + daemon. |

## 3. The ladder

| length | P6 M32 eager | P7F-1 (PG) | **P7F-3 (G3M)** | delta vs P7F-1 |
|---|---|---|---|---|
| 2k fresh | 245.0 | 247 (8.3s) | **259.4** (7.9s; chunk avg 123.4ms) | **+5.0%** |
| 8k fresh | 190.9 | 212.0 (36.4s) | **222.9** (34.6s; last-half chunk med 152.0ms) | **+5.1%** |
| 100k rebuild | 165.3 | 183.8 (532.2s) | **191.8** (509.8s; avg 166.8ms/chunk) | **+4.4%** |

Position profile @100k (G3M): ~131ms @pos0-class -> 152-154 @pos48-58k ->
195-197 @pos78-97k. The GEMM saving is position-independent (~6.3ms/chunk,
measured at 2k: 123.4 vs 129.7 avg); the position growth is attention/KV
(unchanged — not attacked this session).

## 4. Mission items 2/3: re-priced DOWN, not landed (evidence)

- **PRE/ATTN/COMBINE fusion** was priced at ~11.5ms/chunk pre-P7F1, when each
  launch cost ~0.34ms ENQUEUE. P7F1's capture killed enqueue (2 submits/chunk);
  what remains of the pre/combine/norm pool is pure kernel EXECUTION + CTA-wave
  fixed cost. The P7F2 attribution stands: pfa16's ~0.36ms/launch is CTA-wave
  fixed (128 CTAs / 82 SMs at hard 1 CTA/SM), which merging the SMALL kernels
  into the attention grid does NOT remove (same total waves), and the true
  single-kernel pre+attn+combine fusion needs 176 CTAs with cross-CTA
  dependency barriers — at 1 CTA/SM that is the wave-2-waits-wave-1 deadlock
  class (SKEDCHECK-adjacent). Safe merged-half variants (pre 2->1, combine
  2->1 launches) price at only ~1-2ms/chunk. NOT worth the fault-class risk.
- **Norm diet** (folding pfk_n16/pfk_ab16/pfk_hh16 into GEMM prologues) would
  make every GEMM CTA redundantly recompute row norms (RMS reduction order
  would also need per-element-identical restructuring = bit-identity risk) for
  a ~1-2ms/chunk pool of tiny-kernel time. Parked.

## 5. THE VERDICT — the honest 600-gap (evidence-based)

Current @100k: **191.8 tok/s = 29.0% of 662**. Per-chunk avg 166.8ms =
GEMM ~53-63 (post-G3M, 42% coverage) + attention-family ~85-95 (pfa launches
~45@pos0->~90@pos97k + pfk_pre/pfc16/o-proj ~20-25) + scan ~6 + norms/scraps
~10-15. To reach even 300 tok/s needs chunk <= 106ms = -60ms; the two
dominators and their ONLY remaining priced routes:

1. **GEMM (~55ms)**: both-live VRAM caps coverage at 42% (9.3GB full = FAULT,
   measured). Full coverage is worth only ~+2-3ms more. The structural unlock
   is DECODE-SIDE r7 GEMV kernels (rewrite the w1c decode family to read
   packed7, free the originals: net -2.2GB VRAM AND full coverage) — a
   campaign-sized job. Ring-depth-4 (load stream at m32-r7 still ~167-170
   GB/s DRAM vs the 795 pattern; depth-2 prefetch covers only ~512cyc of
   ~1100cyc load latency) is the cheap follow-up: 4 named unit-register sets,
   unroll-by-4 cycle, NCH%4==0 holds for every trunk shape (40/136/48).
   Expected +1-2% ladder.
2. **Attention (~90ms avg)**: P7F2 falsified the dot-engine route (phase-
   structure-bound; a PERFECT mma engine only reaches ~115ms/chunk at pos97k).
   The one remaining structure-class lever is the **persistent-CTA attention
   rewrite** (1 CTA/SM resident, KV streamed, barriers -> warp-local sync):
   unpriced, the highest-variance item. If it lands at the ~2x class ->
   chunk ~125-135ms -> **~240-255 tok/s @100k (~36-38% of 662)**.
3. Everything else priced <= 1-2ms each (fusion/norm diet, re-priced above).

**Realistic endpoint on this dext, with the landed evidence: ~200-210 tok/s
@100k (30-32% of 662) from the remaining safe margins; ~240-255 ONLY IF the
persistent-CTA attention rewrite proves out. Full 662 parity remains
structurally out of reach on this dext (1 CTA/SM, no cp.async, 48KB static
smem, both-live VRAM wall) — consistent with the P5/P6 endpoint estimate,
now sharpened: the attention family, not the GEMM family, is the binding
half of the gap.**

## 6. Ops

Daemon: relaunched on the P7F1 line unchanged (G3M is default-via-PF_GEMM3=1;
`PF_G3M=0` restores the P7F1 path byte-for-byte; `PF_G3M_MB` tunes coverage).
Verified: health ok (parked pos 97810 / cur 4471), [g3m] swap logged in the
daemon, 5.2k-tok FRESH chat 24.2-27.3s (~192 tok/s class), resident TURN-2
1.6s (delta-prefill path), zero faults. One reboot-law cycle at session start
(shutdown RPC); the 9.3GB probe fault cleared on process exit (no cold-cycle
needed). Logs: ~/p7f3_g2k_{on,off,full,t}.log, ~/p7f3_g8k.log, ~/p7f3_r100k.log,
~/w100k_serve_p7f3.log, ~/api_p7f3.log; state dumps ~/p7f3_a{1,2}_on.npz.
