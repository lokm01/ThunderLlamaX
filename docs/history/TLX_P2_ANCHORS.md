# TLX DRAFTER Phase 2 — THE ANCHOR-SCALE STAGE B (results)

2026-10-01/02. Agent: GLM (anchor-scale dispatch). Rental: vast 53710032
(A100 PCIE 80GB, 200GB disk) — DESTROYED + API-verified 0 instances.
Spend $3.43 of the ~$10 budget (balance $28.16 -> $24.73). Rig: 2.7h dense
GPU window (boot 15:55 -> ALL DONE 18:44), ends MoE resident /health ok.
Arbiter: chain_sim class-A (r8_prose untouched primary) + THE NEW CANARY
BATTERY (6 held-out fresh-novel sessions at 64k-99.4k).

## 1. The anchor dump (the Phase-1 unlock, executed)

36 FRESH novel-prose sessions (bookcorpusopen shard-0 indie novels; zero
8-gram overlap with the r8 corpus; deterministic crop offsets; chat-header
shape "continue the story"): 6 sessions at each of 64k/72k/80k/88k/96k/99.4k
prompt tokens; 30 train + 6 canary (canary = disjoint books, one per slot).
Per session: serve FRESH prefill (pf_prefill batched + interleaved dfill,
PF_W4A8=0, the r8 capture env verbatim: TLX_EAGLE_K=4 TRIG=0 T1=0 LOOKUP_K=10)
+ 250 forced-K4 decode cycles with per-cycle anchor capture (h_seed BEFORE
each cycle's draft + pos/cur/committed/dring/amds) + kvd/scd int8 base at
decode start + sha'd manifests.

- **8,829 usable anchors** (7,361 train / 1,468 canary), ~245/session,
  positions 64,001..99,901 (engine m-mean 0.85-1.47 per session).
- Harness: engine0/anchor_scale_dump.py (runner ttt/anchor_dump_runner.sh,
  resumable per-session; the daemon's global-cycle-rebuild discipline at
  prefill/decode boundaries — zero faults in the whole window).
- Dumps: rig ~/trace_dump/anchor_scale/ (6.3GB) + Mac mirror
  ~/drafter/anchor_scale_mirror/ (manifest sha256: anchor_scale_manifest.sha256).

## 2. THE HEADLINE FINDING — the canary battery REFRAMES THE BAR

Scoring the CURRENT ENGINE PACK (control) on the new held-out canary:

| battery | control E[m]k2 | control E[m]k4 | a1 | m-dist |
|---|---|---|---|---|
| r8_prose @97.9k (the old primary) | 0.549 | 0.549 | 0.412 | [30,14,7,0,0] |
| **canary fresh-novel @64k-99.4k** | **0.979** | **1.197** | **0.634** | [.37,.29,.18,.10,.06] |

**The r8 anchor UNDERSTATES the engine's novel-prose acceptance by ~0.43 k2.**
The r8 trace (model-voice lighthouse/turbofan continuation on the snap100k
conversation) is an idiosyncratic hard subclass; on representative novel
continuation at serve positions the shipped pack sits at ~0.98 k2 / 1.20 k4 —
already at the door of the E[m]k2 1.0-1.3 ship band. The "43.7 -> 55-65"
projection built on r8-relative gains needs re-anchoring on this battery.
(Canary per-session k2: bk_05 0.879 / bk_11 1.051 / bk_17 0.907 / bk_23 0.825
/ bk_29 1.105 / bk_35 1.105.)

## 3. Stage-B v2 (from ckpt_6000, the mission's spec) — FALSIFIED FOR PROSE

Recipe actually used: anchors 0.55 of steps (8 cycles/step, own-proposal
slice feedback, labels = committed streams, base-only kv conditioning — see
laws) + protective mix 0.45 (prose16k .20 / code8k .10 / gsm8k .15), S=6,
lr 6e-6 cosine, 1200 steps w/ canary-loss early-stop (best @ 800), batch 8.
Trainer chain validated BIT-IDENTICAL to the v1 chain_forward_r8 (same loss
1.5514 on identical inputs). Canary CE 2.22 -> 1.434 (real held-out
generalization; train/canary a1 gap ~0 = NO memorization — the v1 disease
is cured by scale, as hypothesized).

| battery | control | sb2_best_gptq | delta |
|---|---|---|---|
| r8 held-out k2 (primary bar 0.8-1.0) | 0.549 | 0.451 | **-0.098 FAIL** |
| canary k2 (new held-out) | 0.979 | 0.741 | **-0.238 FAIL** |
| canary k4 | 1.197 | 0.837 | -0.360 |
| canary a1 | 0.634 | 0.530 | -0.105 |
| gsm8k k2 | 0.764 | **1.258** | **+0.494** |
| prose16k k2 | 0.862 | 0.722 | -0.140 |
| code8k k2 | 0.861 | 0.775 | -0.086 |

**G2 VERDICT: DO NOT INTEGRATE.** The ckpt_6000 lineage is now falsified for
long prose a THIRD time — with thousands of diverse serve-position anchors
and a clean generalization signal, anchor training from that init still moves
prose DOWN and battery UP. Its only real asset is the +0.49 gsm8k class.

## 4. THE PRICED NEXT ITERATION (v3 — the correct mainline per this data)

**Init from the FIRST-PARTY bf16 originals** (the 0.608-r8 class; on the
canary class it is the ~0.98-1.2 pack to beat), NOT the Stage-A lineage:
same 7.3k-anchor set + protective mix + low LR (2-3e-6 — protect a good
init) + canary early-stop; bar = canary >= 1.05 k2 AND r8 >= 0.6 AND no
cross-class regression. Everything is now LOCAL on the Mac (mirror + weights
fetcher; anchors_train/canary rebuildable via ttt/stageb2_build.py) — the
only cost is the anchor re-upload to a rental (~2GB zstd; the vast proxy
caps at ~2Mbps — budget 2-3h or use the S3-relay path). If v3 also fails to
beat the first-party control on the canary: the HONEST STOP — ship the
current pack, hand the +0.49 gsm8k profile to a battery-class opt-in pack,
and close the TTT program (the instrument: score_v2.py + the canary battery
are permanent).

## 5. Laws / infra earned this phase

- **THE PROXY LAW**: vast ssh proxies cap aggregate upload ~1.5-2Mbps
  regardless of stream count; Mac raw upstream is ~17MB/s (cloudflare probe).
  zstd -6 on int8 kvd = **3.26x** (the quantized KV is far from random).
  Plan relays (S3/presigned) or compression for any >1GB push.
- **THE MID-MIRROR RACE**: never ship a mirrored session on manifest.json
  presence alone — rsync file order can deliver the manifest before a large
  artifact; verify file count/sizes (cost one b1 crash).
- **THE TRAINER/SIM a1 GAP**: the bf16 own-feedback chain shows a1 ~0.04 on
  ckpt_6000 where the packed sim (cond=engine) shows ~0.35 — own-chain
  discipline is the harsher metric; use CANARY CE for ckpt selection (the
  a-mean is too quantized) and the SIM as the only arbiter.
- **THE BASE-AVAILABILITY SEMANTICS**: anchor dumps trimmed to decode-start
  rows must mask rows [base_len, pos) in training (engine's own prior-cycle
  rows absent); chain_sim gained KV8.from_dump(pad=) + auto-pad in
  run_class_a (r8 path byte-identical — control reproduced 0.549 EXACTLY).
- eval_chunk discipline: 48 fp32 [4,99k,256] buffers = ~98GB — chunk evals.
- anchor harness correctness rests on three verified invariants: boot slice
  == r8 stab (array-compared), zstd roundtrip, and the v1/v2 trainer A/B
  (identical loss on identical inputs).

## 6. Artifacts

- Packs: Mac ~/drafter/ttt_results/packs_stageb2/{sb2_best_gptq, sb2_best_rtn,
  sb2_last_rtn} (NOT shippable — prose regressions).
- Scores: ~/drafter/phase0/score_v2.json (+ score_v2.py driver; canary
  battery = ~/drafter/anchor_scale_mirror bk_{05,11,17,23,29,35}).
- Training: ~/drafter/ttt_results/canary_hist.json + stageb2_run3.log.
- Rig repo: engine0/anchor_scale_dump.py + ttt/{prep_anchor_sessions.py,
  stageb2_build.py, train_stageb2.py, run_stageb2.sh, anchor_dump_runner.sh}.
- Corpus: Mac ~/anchor_corpus (books + selection manifest + venv tooling).
