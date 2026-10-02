# TLX DRAFTER — Stage-B v3 RUNBOOK (first-party bf16 init, LR 2-3e-6)

2026-10-01/02. Agent: GLM (v3 dispatch). FINAL STATUS AT SESSION END:
**TRAINING BLOCKED on a vast.ai platform incident — 17 create attempts across
~10 hosts / 4 images (incl. a 30MB ubuntu:24.04 on a V100) over 21:10-00:30
UTC-equivalent, ALL stuck `cur_state=stopped / actual_status=loading /
intended_status=stopped`, ssh refused; "resources_unavailable" on explicit
starts; ~$0.08 total billed (stuck launches don't bill GPU). ALL instances
DESTROYED, API-verified `{"instances_found": 0}`.** This is NOT a scientific
falsification — the v3 experiment did not run. Everything is staged for a
30-60 min execution the moment vast launches instances again (see READY-TO-RUN
below). Local science that DID complete this session is in the table below —
it materially raises the v3 bar (the QUANT-PATH LAW).

Executed in-session despite the block: init-asset verification (weights/
= offset-buggy, weights_v2 = true first-party), the full bf16-init + RTN(bf16)
battery rows, the v3 trainer/runbook/scorer scripts (committed on the rig,
b14124a), the compressed upload payload (1.74GB), and this runbook.

## The v3 experiment (spec)
v2 recipe EXACTLY (anchors 0.55 weight / S=6 / own-slice feedback / base-only
kv conditioning / protective mix 0.45 of steps = prose16k .20 / code8k .10 /
gsm8k .15 — run as dirs prose16k:0.45,code8k:0.22,gsm8k:0.33 inside the corpus
fraction) — CHANGED ONLY:
  1. init = PRISTINE first-party bf16 (`weights_v2` lineage — VERIFIED this
     session: `~/drafter/weights/` is the OFFSET-BUGGY fetch (cos~0 vs
     ref_pack), `weights_v2/` is the true first-party (norm-law cos 1.0000,
     weight cos 0.996+). On the rental: fetch via fetch_weights.py --with-head-emb,
     sha-verified against weights_v2.sha256.)
  2. LR 2e-6 AND 3e-6 (two arms, sequential, same seed)
  3. steps 1600 (v2 used 800), canary-CE early-stop patience 4 evals,
     ckpt-every-100 for the curve.
Selection on canary CE (the trainer-own-chain vs sim gap law), sim as the
arbiter.

## Measured baselines this session (chain_sim, all on the Mac — COMPLETE)
| pack | canary k2 | r8 k2 | gsm8k | prose16k | code8k |
|---|---|---|---|---|---|
| control = SHIPPED pack (ref_pack) | 0.979 | 0.549 | 0.764 | 0.862 | 0.861 |
| bf16 first-party init | 0.987 | 0.608 | 0.521 | 0.829 | 0.826 |
| RTN(bf16) pack (fresh this session) | 0.970 | 0.529 | 0.661 | 0.768 | 0.754 |
| (v2 sb2_best_gptq, for reference) | 0.741 | 0.451 | 1.258 | 0.722 | 0.775 |

**THE QUANT-PATH LAW (new, program-grade)**: the shipped GGUF pack is NOT
bitwise RTN(bf16) — 53% packed-byte equality, dequant cos 0.9978-0.9991 (same
source, different quant path — consistent with an imatrix/rounding-path
difference upstream). Canary class (novel prose @64-99k) is INSENSITIVE to the
quant path (all three within ±0.017 k2); EVERY cross class is HIGHLY sensitive:
the shipped pack beats the pristine bf16 AND the fresh RTN on gsm (+0.14/+0.10),
prose16k (+0.09/+0.03), code8k (+0.11/+0.04). r8 swings ±0.08 too. So the
v3 no-regression cross bars are NOT reachable by re-quantization of the
first-party init alone — the protective mix must actively recover ~0.03-0.10
(and +0.24 on gsm from the bf16 init) while lifting canary +0.06. v2's gsm
1.258 proves the mix has that class of power.

## Bars (the fork)
- PASS: canary k2 >= 1.05 AND r8 k2 >= 0.6 AND cross no-regression
  (gsm >= 0.764, prose16k >= 0.862, code8k >= 0.861) -> G2-cleared: stage ship
  pack, rig G4 battery (Tier-1 60/60 x2 + stock 59/59 + quote >= 74.9 + k4hist
  within -0.15 of sim + GSM8K-30 daemon battery 43.7), env flip + 15-min soak.
- FAIL: current pack stays shipped; verdict doc (3 falsifications + canary
  reframe + the +0.49 gsm8k profile as opt-in battery-class pack).

## Ready-to-run assets (all on this Mac)
- /tmp/v3stage/anchors.tar.zst  (912,515,079 B — 36 sessions, 6.26GiB at 13.6%)
- /tmp/v3stage/traces_mix.tar.zst (866,526,975 B — prose/code/gsm8k + r8 fnw)
- /tmp/v3stage/weights_v2.sha256 (init verification list + tarball shas)
- /tmp/v3stage/setup_rental.sh (apt zstd, extract, HF fetch, sha gate)
- /tmp/v3stage/upload_and_run.sh "<ssh_host> <port>" (upload + verify)
- ~/drafter/ttt/train_stageb3.py, run_stageb3.sh (b1/b2/b3), stageb_build.py
  (+--skip-r8), score_v3.py (Mac scorer: curve|full)
- Payload copy also at: (copy tarballs to ~/drafter/v3_payload/ if /tmp wiped)

## Rental execution (once ssh alive)
1. `bash /tmp/v3stage/upload_and_run.sh <host> <port>`  (~1.7GB upload)
2. `ssh ... 'bash /root/up/setup_rental.sh'`            (~10 min: extract+fetch+sha)
3. `ssh ... 'cd /root/ttt && bash run_stageb3.sh b1'`   (build shards+anchors)
4. `ssh ... 'cd /root/ttt && bash run_stageb3.sh b2'`   (2 arms, ~15-30 min)
5. `ssh ... 'cd /root/ttt && bash run_stageb3.sh b3'`   (RTN curve + GPTQ best)
6. Download /root/packs3/* (test download speed first; the proxy may cap)
7. Mac: score_v3.py curve for the ckpt curve; full for best_gptq finalists
8. Destroy + 0-instance verify: curl -s -H "Authorization: Bearer $vast_api_key"
   https://console.vast.ai/api/v0/instances/

## Rig side (PASS fork only; rig currently MoE resident /health ok)
Disable-first GPU window (launchctl disable + plist-out), verify ops/state
markers ABSENT, TLX_DRAFT_PACK=<pack> in a dense window, G4 battery list from
TLX_DRAFTER_PLAN.md section 3, then env.canonical.d flip + 15-min soak,
sync;sync;sleep 3;sync before any stop, rig ends MoE resident.
