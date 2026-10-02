# TLX DRAFTER — Stage-B v3 VERDICT: FAIL (the honest stop)

2026-10-01/02. Agent: GLM (v3 execution dispatch). The decisive v3 iteration
RAN TO COMPLETION on vast 4090 53765059 (destroyed + 0-instance-verified,
~$0.55 total). Init pristine first-party bf16 (sha-gated 15/15), LR 2e-6 AND
3e-6 arms, 1600 steps, canary-CE selection, GPTQ+RTN packing, chain_sim
arbiter. **Verdict: the current engine pack STAYS SHIPPED. The TTT Stage-B
program is closed with a three-fold falsification.**

## 1. The primary result (chain_sim, Mac)

| pack | canary k2 (bar ≥1.05) | canary k4 | r8 k2 (bar ≥0.6) | gsm8k (≥0.764) | prose16k (≥0.862) | code8k (≥0.861) |
|---|---|---|---|---|---|---|
| control = SHIPPED pack | 0.979 | 1.197 | 0.549 | 0.764 | 0.862 | 0.861 |
| bf16 first-party init | 0.987 | 1.208 | 0.608 | 0.521 | 0.829 | 0.826 |
| RTN(bf16) | 0.970 | — | 0.529 | 0.661 | 0.768 | 0.754 |
| **v3 lr2e-6 best_gptq** (CE 1.2504 @1450) | **0.943 FAIL** | 1.077 | **0.451 FAIL** | **0.996 PASS** | **0.759 FAIL** | **0.803 FAIL** |
| **v3 lr3e-6 best_gptq** (CE 1.2467 @700) | **0.913 FAIL** | 1.030 | **0.471 FAIL** | **0.987 PASS** | **0.772 FAIL** | **0.807 FAIL** |

**Curve (canary k2 vs steps, RTN packs)**:
- lr2e-6: 0.896(200) → 0.922(400) → 0.937(600) → 0.934(800) → 0.946(1000) → 0.951(1200); best@1450: RTN 0.937 / GPTQ 0.943 — plateau ~0.95, never reaches RTN(bf16) 0.970.
- lr3e-6: ~0.86(200, partial) → 0.905(400) → 0.925(800); best@700: RTN 0.913 / GPTQ 0.913 — strictly below arm1 at every comparable point.
- The CE-selected best ckpts quantize to BELOW their init class in both quant paths; GPTQ ≈ RTN ± 0.006 (v2's finding holds).

**Reading**: training moves packed canary k2 DOWN from its own init class at
every LR, at every step count, in both quant paths — while canary CE improves
dramatically (2.41→1.25, zero train/canary gap: real generalization in the
trainer's own-chain metric). The trainer-own-chain vs sim-engine-cond law,
previously a selection heuristic, is now the falsification itself: **under
this recipe the two objectives anti-correlate.** The only class that moves up
is gsm8k (+0.22-0.23, reproducing v2's +0.49 profile at half magnitude),
paid for by prose16k −0.09-0.10, code8k −0.054-0.058, r8 −0.078-0.098.

## 2. The three falsifications (the program's exit dossier)
1. **v1 (P1 full run)**: corpus-scale Stage-A TTT — r8-relative gains were
   anchor/exposure artifacts; 100k prose falsified.
2. **v2 (P2 anchor-scale)**: 7.3k diverse serve-position anchors + protective
   mix from the ckpt_6000 lineage — canary 0.979→0.741, prose −0.14, r8 −0.098;
   only asset +0.49 gsm8k.
3. **v3 (this run)**: the last confound — INIT — removed (pristine first-party
   bf16), LR bracketed (2-3e-6), selection on held-out canary CE (no
   memorization, gap +0.001) — and the packed canary k2 still lands BELOW the
   no-training baseline (0.943/0.914 vs 0.970-0.987). **The recipe itself is
   falsified**: the EAGLE-3-style TTT unroll on engine-trace anchors, at
   anchor-scale, with the protective mix, cannot lift packed serve acceptance
   on fresh novel prose — with init, LR, data-scale, and selection all
   exonerated.

## 3. The canary reframe (standing, from v2, re-anchored by v3)
The shipped pack ALREADY sits at 0.979 k2 / 1.197 k4 on held-out fresh-novel
prose at true serve positions (64-99k) — inside the door of the 1.0-1.3 ship
band. The prose 43.7→55-65 projection built on r8-relative training gains is
retired. The remaining headroom evidence (syv-ai 70-80% pos-0 acceptance on
the same dense model + nextn layer) decomposes into precision
workload/exposure-bias/conditioning factors (G1) — none of which this recipe
addresses. K2 remains the carrier at battery ctx; E[m]k2=1.34-class operation
is delivered by the SHIPPED pack, not by retraining it.

## 4. The opt-in gsm8k pack (the v2 asset, v3 cross-check CONFIRMED)
v2's sb2_best_gptq: +0.49 gsm8k (0.764→1.258). v3's GPTQs: **+0.22-0.23
gsm8k (0.996/0.987)** — same profile, half magnitude, from the pristine init.
This is a real, repeatable battery-class asset: short-ctx reasoning loads
(GSM8K-class) gain ~0.2-0.5 E[m]k2 from anchor+mix training while every prose
class pays. It remains viable as a workload-opt-in TLX_DRAFT_PACK swap, NOT
as the default. The current default pack STAYS SHIPPED (its 0.979/1.197
canary + 0.862/0.861 cross profile dominates every v3 pack on 4 of 5 bars).

## 5. Execution record (2026-10-01/02)
- vast 53765059 RTX 4090 24.6GB, pytorch:latest (torch 2.2.1+cu121), 34GB disk.
- Upload 1.74GB zst payload ~3.5 min (the old 2Mbps proxy cap is HEALED).
- setup: extract + HF first-party fetch (range-read safetensors) + sha gate
  15/15 vs weights_v2 manifest.
- b1: sb shards (prose16k 16384 tok / code8k 8192 / gsm8k 30 transcripts) +
  anchors 7361 train cycles / 30 sessions (10.2GB kv) + canary 1468 / 6.
- b2: **24GB adaptation** — session kv PINNED-CPU streamed per chain step
  (identical math; fp16 src, fp32 cast at Kfp) + PYTORCH_CUDA_ALLOC_CONF
  expandable_segments (the first anchor step OOM'd on fragmentation without
  it: 5.8GB reserved-unallocated). Per-arm disk pipeline (pack → cleanup).
  arm 2e-6: full 1600 steps, best CE 1.2504 @1450 (~16.5 min).
  arm 3e-6: early-stop @900, best CE 1.2467 @700 (~10 min).
- b3: RTN curve packs + engine-calib GPTQ best packs per arm.
- ALL packs + hists + logs to Mac; instance destroyed; API 0 instances.
- Spend: ~1.35h × $0.4073 ≈ **$0.55**. Total v3 program marginal cost <$1.

## 6. Permanent assets kept
- ~/drafter/ttt_results/packs_v3/ — 17 packs (2 best_gptq + 15 RTN curve),
  canary_hist_lr{2,3}e6.json, train logs.
- ~/drafter/phase0/score_v3.json — the battery numbers (appended by scorer).
- ttt scripts on the rig (committed): the 24GB adaptation lives in
  train_stageb3.py/train_stageb2.py (backward-compatible patches).
- chain_sim + score_v3.py = the permanent instrument.

## 7. Final battery rows
See section 1 (complete). Full per-session data: ~/drafter/phase0/score_v3_final.json
(the live score_v3.json suffered concurrent-writer clobbering between the 4
parallel scorers; the final table is log-reconciled — [v3score] summary lines
are ground truth).
