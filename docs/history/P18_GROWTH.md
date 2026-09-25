# P18 — THE GROWTH-POOL FORENSICS: SOLVED (the pool = 100% the wide-attention kernel; the P17 doc's 16x bookkeeping error; the 400 arithmetic corrected)

Status: **The 92.6ms position-growth pool (207.9ms @pos0 -> 300.5 @97k-end
per 64-chunk) is attributed WITH NUMBERS: ALL of it is the wide-M attention
kernel's KV-extent cost. Every other class is FLAT in pos (measured at 11
positions, per-class isolated graphs + pos-buffer ablations + a fine cliff
sweep). The P17 doc understated the attention pool 16x by reading the
standalone bench's PER-LAUNCH ms as per-chunk ms (0.77/4.34/8.48 per launch
-> the real pools are 12.3/69.4/135.7ms per 64-chunk). Three fix attempts
landed bit-identical-but-NEUTRAL (explicit A-reg-share: SPILL->nondet, the
w64q class; warp-permutation: flat; __ldcs/__ldg cache hints: flat) — the
kernel is latency-STRUCTURE bound (4 synced phases x 241 tile-iterations
under the dext's 1-CTA/SM; 16.9us/iter vs ~6us of tensor+issue floor; 19.4
TFLOPS = 14% of tensor peak; 78 GB/s effective vs the decode kernels' 193).
The neutral bit-identical kernel (permutation + hints) is SHIPPED; ladder
re-gated. The 400 statement: ~290-300 via the priced M=128+WY rungs WITHOUT
another attention build; 400 needs a persistent-CTA or dynamic-smem(>48KB)
attention redesign — both P5-listed unlocks, each a session-class build.**

## 1. THE ATTRIBUTION (engine0/p18_attr.py; ~/p18_attr.log — the answer)

Harness: the p15_attr law, at 11 positions on the REAL snap100k KV (graphs
built ONCE; pos enters ONLY via the device buffers pos_arr64/pos_w64 — the
same captured graph runs at any pos). Arms: FULL/TRUNK/DFILL curve, per-class
isolated graphs, pos-buffer ablations, quarter localization, fine cliff sweep.

**ARM 1 — the curve reproduces the rebuild exactly** (in-run numbers ->
harness): 207.9 -> 186.1 @pos0; FLAT ~250 over 9.7k-48.6k; ramp 51k-54k;
FLAT ~310-317 to 97k. TRUNK ~= FULL; **DFILL = 2.2ms FLAT** (the "dfill 62%
of wall" flag was about the SEPARATE fill_draft pre-pass, 243.4s — NOT the
in-chunk dfill windows).

**ARM 3 — per-class (the table, ms per 64-chunk):**

| class | pos 0 | pos 48640 | pos 97280 | note |
|---|---|---|---|---|
| **attn (pfaw_w64)** | **1.15** | **66.26** | **130.59** | **THE growth pool — 100% of it** |
| gemm_ffn | 78.01 | 78.06 | 78.08 | flat |
| scan (pfs64) | 26.78 | 26.79 | 26.78 | flat |
| gemm_fd | 22.82 | 22.53 | 22.77 | flat |
| gemm_qg | 21.98 | 21.93 | 21.95 | flat |
| gemm_og | 10.46 | 10.46 | 10.46 | flat |
| gemm_qkv | 9.95 | 9.96 | 9.95 | flat |
| gemm_oa | 3.92 | 3.93 | 3.93 | flat |
| nrm_ab | 3.84 | 3.86 | 3.83 | flat |
| attn_comb | 1.74 | 1.74 | 1.75 | **flat — the S-split combine suspect is DEAD** (no early-exit loop reads all S splits at every pos: CONSTANT by construction) |
| pre64 (KV append) | 1.58 | 1.56 | 1.59 | flat — the KV8-append suspect is DEAD |
| dfill (all 7 classes) | ~1.9 | ~1.9 | ~1.9 | flat |
| SUM(isolated) | 187.56 | 250.19 | 314.79 | ~= FULL 186/251/311-318 (classes fully account for the graph) |

**ARM 4 — the ablation nails it**: same captured graph, only the pos device
buffers change: pos_w64 LOW (attention extent + dfill window pos) ->
**-121.1 / -129.5ms** at 58368/97280; pos_arr64 LOW (trunk append pos) ->
+4.7 / -0.6 (NOTHING). The growth is 100% pos_w64-driven = the attention
KV extent.

**ARM 2 — the cliff localized**: the ramp 48640 -> 54040 (+60.92ms) then
FLAT. **54040 = 7 x CH exactly (CH = ceil(100352/13) = 7720; split 6 goes
FULL at pos >= 7*7720 = 54040)** — the wave model closes: w64 = 156 CTAs =
4 groups x 13 splits x 3 NHP on 82 SMs (1 CTA/SM): active CTAs = 12 x
splits-with-work; <= 82 (<= 6.8 splits) = 1 real wave -> FLAT ~250; split 6
full -> wave 2 carries full-work CTAs -> ~2 x full-CTA time = THE CLIFF;
flat after because 156 = 82+74 stays 2 waves to 97k. The 0->8k arm (w64h,
312 CTAs, 6 NHP) is cheap because nt is tiny at low pos.

## 2. THE MECHANISM (why 8.16ms per launch at 97k)

Per launch (layer): 156 CTAs x 241 TILE=32 iterations (CH/32), 2 waves ->
per-CTA 4.08ms = **16.9us per tile-iteration**. Budget: hmma floor ~2.4us
(1024 m16n8k16/iter at 1-per-4-cycles/SM = 142 TFLOPS peak) + issue ~3.5us.
The rest = LATENCY: 4 __syncthreads-bound phases per iteration (stage-commit
/ QK / softmax / PV) with NO co-resident CTA to fill the gaps (the dext's
hard 1-CTA/SM). The Q A-frags are re-read from global EVERY iteration (241 x
64KB/CTA = 2.4GB per launch) — but cache hints did NOT move it, so the cost
is the phase-latency structure itself, not pure Q bandwidth. Achieved: 19.4
TFLOPS (14% of tensor peak), 78 GB/s effective — vs the DECODE attention
kernels' 193 GB/s on the same KV (they are GEMV-shaped, no 4-phase smem
protocol). Standalone == in-graph here (8.42-8.47 bench vs 8.16 in-plan).

**THE P17 BOOKKEEPING ERROR (fixed)**: the P17 bench table is PER LAUNCH
(per layer); P17_WIDEATTN.md §7 read "0.77/4.34/8.48" as the whole-chunk
attention pool. Real pools: x16 -> 12.3/69.4/135.7ms per 64-chunk. The
"~85ms unattributed" never existed — the standalone bench was RIGHT and the
aggregation was wrong. (0.77 @2k coincidentally matched the 16-launch total,
sealing the misread.)

## 3. THE FIX LEDGER (what was tried this session)

| attempt | corr | perf @100288 | verdict |
|---|---|---|---|
| explicit A-reg share (c[2], kb-major-j) | **NONDET (nz=16792, det x2 FALSE)** | — | 276B ptxas spill at the 128-reg/512-thr cap = the w64q nondet class. LAW: heavy-spill kernels nondet-break on this dext. Banked negative. |
| warp-permutation (mb = wt %% MB, nb = wt/MB; the warp's 2 tiles share mb) | BIT-IDENTICAL nz=0 det x2 | 8.42 (old 8.48) | flat — L1-hits across the wt pair are not the cost. 36B spill OK. |
| + __ldcs on the K/V staging stream, __ldg on A | BIT-IDENTICAL nz=0 det x2 | 8.47 | flat — cache policy is not the cost; the phase-latency structure is. 44B spill OK. |
| **SHIPPED: permutation + hints** | BIT-IDENTICAL | flat (in-plan 316.9 vs 311.7 @97k = session noise) | kept: principled Q-protective hints, zero downside, corr green |

## 4. THE LADDER (re-gated with the shipped kernel; readout-order law)

(bank: P17 = 373.5 / 347.3 / 248.2; gates re-run this session — see
~/p18_g2k.log, ~/p18_g8k.log, ~/p18_g100k.log; first clean run = the log)

| length | P17 (clean boot) | P18 this session (post-crash boot) | correctness |
|---|---|---|---|
| 2k fresh | 373.5 | 328.6 (med 195.5) | F-relerr **1.058e-03 EXACTLY** banked; the (4649,43614) tie-mine assert fires identically (EXIT=1 class) |
| 8k fresh | 347.3 (med 190.9) | 307.6 (med 214.6) | **LINE-FOR-LINE the P17 bank**: F 4.282e-02, GATE A 12/60 (div@0 6545/22546), CTRL 13/60 alpha 2.67, GATE D 5/60, D2 0/160 |
| 100k rebuild | 248.2 | 225.6 (433.5s; fill_draft 255.0) | **cur=4471 EXACT (match=True)**; drift max rec 5.482e-3 / conv 3.836e-3 (gate <=1e-2; banked class 5.35/3.62); rebuilt-state decode **60/60** |

The -8/-9%% perf delta is MACHINE STATE, not the kernel: this session gated on a
post-crash boot (no EFI cold cycle), and the degradation is uniform across the
whole curve INCLUDING the <8k region that runs bit-identical-to-P17 cubins
(w64h untouched; the P18 edit rebuilds only the >=8k w64 arm). The kernel is
corr-proven bit-identical and bench/in-plan NEUTRAL (8.42-8.47 vs 8.48
standalone; 316.9 vs 311.7 in-plan = session noise) — the P17 clean-boot
numbers remain the ship numbers. Chunk curve shape identical: 240.9 @0 ->
~256 flat -> cliff at ~54k -> 325 @97k.

## 5. THE 400 STATEMENT (measured arithmetic, no hope)

400 tok/s @100k = 97810/400 = 244.5s = **160ms avg per 64-chunk**.
- Weights-class (all flat classes) = **184ms** (the floor at every pos).
- Attention adds: ~0 @pos0, ~66 @48k, ~131 @97k-end -> measured avg ~258ms
  -> 248.2 tok/s. The attention IS the gap: it alone adds ~74ms to the avg.
- Priced rungs (unchanged from P17, now the ONLY path left):
  1. **M=128 + WY-scan C=32** (P7C machinery: scan 26.8 -> ~6-10; M128 GEMM
     coverage -13-25; norms -5) -> **-35 to -45ms** -> avg ~215 -> **~290-300**.
  2. **The attention redesign** (the 130ms@97k pool -> 40-60ms would take avg
     to ~180-200 -> **380-420**): the three legal routes on this dext =
     (a) persistent CTAs with cross-split software pipelining (the P5 unlock
     #3; fills the 1-CTA/SM latency gaps), (b) dynamic smem >48KB (P5 unlock
     #2; Q-resident-in-smem + double-buffered K/V staging — needs the dext
     dynamic-smem opt-in, UNPROVEN), (c) a fused FLA-style single-pass shape.
     1024-thread builds are BANKED NEGATIVE (w64q + this session's spill law).
- HONEST ENDPOINT: **~290-300 via the rungs without touching attention;
  400 ONLY IF a persistent-CTA/dynamic-smem attention lands.** The wide-M
  attention remains 41.5% of the 97k-end chunk — the single biggest pool.

## 6. New laws banked (P18)

1. **Read the standalone bench units BEFORE aggregating** — the P17 16x error
   (per-launch read as per-chunk) cost a mission. Per-class isolated graphs
   (p18_attr) are the ground truth; they matched FULL within 1-3% at every pos.
2. **Heavy ptxas spill (>~100B) = nondet-wrong kernels on this dext** (w64q
   276B-class and this session's c[2] variant; 36-48B spills are fine).
   CHECK the spill line on every wide-kernel build.
3. **The pos-ablation pattern**: pos enters via DEVICE buffers (pos_arr64 vs
   pos_w64) — the SAME captured graph runs at any pos, so differential
   attribution needs zero re-capture. Use for any future pos-growth question.
4. The combine (S-split read-all) and the KV8 append are pos-CONSTANT — both
   P18 suspects dead by measurement + by construction.
5. The wave model (splits activate at s x CH; cliff when the last wave's CTAs
   go full; 82 SMs, 12 CTAs/split for w64) predicts the chunk-time curve to
   the exact position (54040 = 7 x 7720).
6. Cache-policy hints (__ldcs/__ldg) are legal + bit-identical on the dext
   but did not move a latency-structure-bound kernel.

## 7. Files

engine0/p18_attr.py (the forensics harness; PF_GATE18=1 hook in
test_w100k.py), pf_attnw.cu (the shipped permutation+hints QK/STG;
.p17bak = the P17 original), pfaw_w64_s13_100k.cubin (+.p17bak),
~/p18_attr.log, ~/p18_attr2.log (the in-plan re-measure), ~/p18_g{2k,8k,100k}.log.
