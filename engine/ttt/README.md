# TLX DRAFTER — Phase 1 Stage A: the blk.64 (EAGLE nextn) TTT trainer

ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
SPDX-License-Identifier: MIT
Copyright (c) 2026 lokm01

Retrains the dense checkpoint's own draft layer (blk.64 / `mtp.*`, 424.6M
params) with the EAGLE-3 recipe — training-time-test multi-step unroll +
token-only CE — and writes engine-loadable Q4_0 packs (the weight-swap path;
ZERO new serving CUDA). Serving chain semantics are ported EXACTLY from
`engine/mtp.py` `_draft_entries`/`fill_draft` + the draft kernels.

## Files
| file | role |
|---|---|
| `model.py` | DraftBlock (torch) = the engine chain math; FrozenHeadEmb (frozen trunk emb+head) |
| `engine_ref.py` | numpy replay of the engine chain (fp16 round-trips) — Validation-A ground truth |
| `q4pack_lib.py` | the engine Q4_0 two-region pack layout (dequant/RTN/Q8) — q4v.cu truth |
| `fetch_weights.py` | selective HF fetch of `mtp.*` + emb + head via safetensors range reads |
| `validate_port.py` | Validation A (layout byte-law + structural cross-check) |
| `gen_prompts.py` | prompt pool (gsm8k/metamath/ultrachat/code/docqa/prose/prose_long) |
| `teacher_gen.py` | vLLM teacher generation (greedy 70% / temp 0.7 30%) |
| `dump_features.py` | teacher-forced pre-final-norm hidden dump -> memmap shards |
| `train.py` | TTT trainer + overfit tests + GEMV-input calibration dump |
| `pack_trained.py` | torch -> draft_pack (RTN / GPTQ-calibrated; +Q8_0 variant) |
| `replay_eval.py` | chained-replay acceptance (a_cond, E[m]|K2/K4) on held-out shards |
| `gguf_probe.py` | standalone GGUF metadata reader (proved the norm law) |
| `longcorpus.py` | long-context corpus builder (bookcorpusopen split-windows, abs RoPE offsets) |
| `stageb_build.py` / `stageb2_build.py` | Stage-B shard builders (engine-trace + anchor-scale sets; `--skip-r8`) |
| `train_stageb.py` / `train_stageb2.py` / `train_stageb3.py` | the Stage-B trainer lineage: engine-anchor adaptation -> ckpt_6000 lineage -> v3 first-party bf16 init (LR bracket, canary-CE early-stop, pinned-CPU session-KV streaming for 24GB) |
| `prep_anchor_sessions.py` | fresh-novel session prompt prep (36 sessions, 6 ctx slots, deterministic crops) |
| `score_v3.py` | the v3 Mac scoring driver (curve \| full; zero GPU) |
| `run_stagea.sh`, `run_stageb*.sh`, `anchor_dump_runner.sh` | the rental/rig orchestration scripts (stagea s1-s6; stageb b1/b2/b3; v3 arm) |
| `results_v3/` | the v3 run's result metadata (canary training curves, score tables) |

## THE TWO LAWS DISCOVERED (2026-09-30, law-grade)
1. **NORM LAW**: the HF qwen3_5 checkpoint stores ALL RMSNorm weights
   ZERO-CENTERED (Gemma-style): functional = stored + 1.0. The GGUF + engine
   use the functional form. Proven element-wise (GGUF trunk+draft norms ==
   HF stored + 1.0 EXACTLY; `d_eh` RTN cos 0.9968). `from_hf` applies +1;
   `pack_trained` writes functional norms. THE RIG GGUF **IS** THE FIRST-PARTY
   CHECKPOINT (an earlier "different lineage" reading was a fetcher bug).
2. **SAFETENSORS OFFSET LAW**: `data_offsets` are DATA-SECTION-relative
   (add `8 + header_len`), not absolute file offsets. Range-fetching at the
   raw offset returns shifted garbage that still LOOKS statistically sane.

## Chain semantics (validated: step-1 relerr 7.9e-7 vs engine math)
fill (parallel, teacher-forced h_{q-1}) -> KV cache; anchor step0 =
(tok_t, TRUNK pre-norm h_t) same-position with its K/V REPLACING slot t;
steps >= 1 = (own slice-argmax token, own hidden) appended; loss =
sum_i CE(frozen-head(shared_norm(hd_i)), x_{t+i+1}); feedback argmax is
slice-restricted (the serve 40960-row draft-slice approximation).

## PILOT RESULTS (H100 NVL, $11.5 total spend; 1.83M dumped tokens,
## 1600 steps / 4.9M token-visits, S=4, slice feedback, n=760 eval chains)
| variant | E[m]|K2 | E[m]|K4 |
|---|---|---|
| HF first-party init (bf16) | 1.334 | 2.186 |
| CURRENT engine pack (Q4_0 RTN of first-party) | 1.351 | 2.220 |
| TRAINED bf16 | **1.476** | **2.346** |
| TRAINED RTN Q4_0 pack | 1.425 | 2.247 |
| TRAINED GPTQ Q4_0 pack | 1.441 | 2.274 |

Workload = GSM8K-class short ctx on teacher continuations (NOT the 100k
novel-prose regime where the engine measures a1=0.42 — the sim/G0 is the
arbiter for that). Validation B2: 10k-token slice overfits to loss 0.0000,
acc [1,1,1,1]. GPTQ > RTN at equal format (+0.016 E[m]K2) despite lower
weight-cos (it optimizes output error).

## Full-run recipe (NOT yet launched — waits on G1 + this pilot)
```bash
# on a 96GB rental (H100 NVL ~$2.92/h; total wall ~4-6h, est $12-18):
python3 gen_prompts.py --out data/prompts.jsonl --n-reason 8000 --n-code 4000 \
  --n-chat 6000 --n-prose 6000 --n-docqa 3000 --n-long 1500        # ~28k prompts
python3 teacher_gen.py --prompts data/prompts.jsonl --out data/gen.jsonl
python3 dump_features.py --gen data/gen.jsonl --out data/shards --lmax 2048 --batch 16
python3 train.py --mode train --data data/shards --weights ~/weights --steps 6000 \
  --batch 8 --lmax 2048 --S 6 --lr 5e-5 --warmup 300 --feedback slice \
  --calib-tokens 300000 --out runs/full                            # ~20-40M tokens
python3 pack_trained.py --ckpt runs/full/ckpt_6000.pt --out packs/full_gptq --mode gptq --calib runs/full/calib
python3 replay_eval.py --ckpt pack:packs/full_gptq --data data/shards_eval --tag full_gptq
```
Gotchas (earned): vast gvisor hosts silently stick at "stopped" — retry
other hosts; `pytorch/pytorch:latest` image + `apt build-essential` +
`pip nvidia-cuda-nvcc-cu13` with CUDA_HOME/PATH exported; vLLM needs
`max_num_seqs<=512` (GDN Mamba cache) and `VLLM_USE_FLASHINFER_SAMPLER=0`
(flashinfer JIT header mismatch); huggingface-cli is now `hf download`;
safetensors offsets = data-relative (LAW above); dump batches of 16×1400
OOM-warn but recover on 96GB.

## Sim handoff (what Phase 0 scores first when chain_sim lands)
Score `~/drafter/ttt_results/packs/pilot_gptq` (and pilot_rtn) against the
r8_prose anchor trace through G0-calibrated chain_sim — packed, never bf16.
Also score the CURRENT pack as the second calibration point, and
`hf_init`-style bf16 weights for the G1 precision ladder. The packs swap in
via `TLX_DRAFT_PACK=<dir>` after the pcache draft_pack fingerprint fix.

## Program verdict (CONCLUDED 2026-10-02 — read this before retraining)
This trainer ran to completion three times (Stage A corpus-scale on a
rented H100; Stage-B v2 anchor-scale on an A100; Stage-B v3 clean-slate
bf16-init on a 4090) and the recipe was FALSIFIED each time — the trainer's
own-chain objective anti-correlates with engine-conditioned acceptance
under this recipe, and the SHIPPED pack was measured near this method's
ceiling on representative prose (the canary reframe: 0.979 k2 / 1.197 k4 on
held-out fresh-novel sessions vs 0.549 on the old r8 hard anchor). The
instruments (chain_sim + the canary battery + this pipeline, all validated
bit-identical to the engine chain) are permanent; the one preserved asset
is the GSM8K-class opt-in pack (+0.22-0.49 E[m]k2 on battery workloads,
`TLX_DRAFT_PACK`-selectable). Full verdicts:
[../../docs/history/TLX_P1_RESULTS.md](../../docs/history/TLX_P1_RESULTS.md),
[../../docs/history/TLX_P2_ANCHORS.md](../../docs/history/TLX_P2_ANCHORS.md),
[../../docs/history/TLX_P2B3_VERDICT.md](../../docs/history/TLX_P2B3_VERDICT.md).
