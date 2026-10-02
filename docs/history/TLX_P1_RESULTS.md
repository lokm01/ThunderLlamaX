# TLX DRAFTER Phase 1 — THE FULL-RUN RESULTS (Stage A + Stage B)

2026-10-01. Agent: GLM (Phase-1 dispatch). Rental: vast 53632319 (H100 NVL 94GB,
500GB) + one dead 40GB-overlay host. Spend $41.91 of the $60 cap (GPU $35.2,
storage $3.97, transfer $2.77; balance $70.07 -> $28.16).
Arbiter: chain_sim class-A on r8_prose @97.9k (control = current engine pack
E[m]k2 0.549 / k4 0.549; engine in-vivo 0.588/0.6275).

## 1. The corrected data mix actually built (Stage A)

| dir | windows | tokens | ctx | notes |
|---|---|---|---|---|
| sh_books_long | 429 | 14.06M | 32k windows, abs pos 0-108k | bookcorpusopen, teacher-forced, split-windows, offs+hpre |
| sh_sg_long | 80 | 2.59M | ~6k-38.6k | vLLM greedy self-continuations (own-text class) |
| sh_books_mid | 479 | 3.92M | 8k windows, spread 0-104k | |
| sh_gsm | 6218 | 1.92M | ~600-2k | gsm8k+metamath teacher gens (battery class) |
| sh_chat | 2475 | 1.54M | ~1-4k | ultrachat |
| sh_code | 1976 | 0.96M | ~1-4k | mbpp + CodeAlpaca-20k gens |
| sh_docqa | 1493 | 0.37M | ~400 | squad |
| sh_prose_s | 800 | 1.17M | ~1.5k | book-slice prose continuations |
| **total** | | **26.5M** | | 8 classes, long-ctx = 63% of tokens |

Mix rationale (the G1 fix): 63% long-ctx window tokens / ~47% of supervised
anchors in the 32k-108k absolute-position band (books left-cropped so window
ENDS sit at doc ends; RoPE absolute offsets stored per window; hpre = trunk
hidden at window-start-1 so the fill's first row is serve-faithful). Trainer:
multi-dir weighted sampling, SDPA flash fill, SEGMENT attention (fill rows
[0,t) + own rows; slot-t replacement exact), multi-anchor (4/window, tail
60-95%), S=6, lr 5e-5 cosine, 6000 steps = 217M window-tok / 288k supervised
chain positions. Both chain paths validated BIT-IDENTICAL (0.0 delta) and
overfit-syn/overfit-long/overfit-real all PASS on GPU.

## 2. THE CURVE (packed RTN, chain_sim class-A r8_prose, engine-conditioned)

| ckpt | tokens | E[m]k2 | E[m]k4 |
|---|---|---|---|
| 666 | 25M | 0.392 | 0.431 |
| 2010 | 75M | 0.373 | 0.373 |
| 3344 | 125M | 0.412 | 0.431 |
| 4693 | 175M | 0.451 | 0.471 |
| **6000** | **217M** | **0.471** | **0.510** |

STAGE A FALSIFIED for the 100k-prose regime: even with the corrected mix the
cloud-teacher TTT run stays BELOW the 0.549 control at every checkpoint
(the pilot's negative transfer is not a mix problem alone). Measured root
cause: **greedy-vs-corpus target mismatch = 0.625/0.645 agreement** (prose16k/
code8k engine traces): on human/book text the trunk-greedy next token differs
from the corpus token ~36% of the time — book-token CE trains the drafter
toward "plausible human continuations", not "THIS model's greedy stream" (the
serve target). Battery class transfers fine (gsm8k trace: Stage-A 1.026 k2 vs
control 0.816).

## 3. STAGE B (engine-trace adaptation, from ckpt_6000, 400 steps lr 6e-6)

Three arms, r8_prose E[m]k2:
| arm | data | r8 k2 | verdict |
|---|---|---|---|
| corpus-only (nor8) | engine hiddens prose16k/code8k/gsm8k + GREEDY labels | **0.392** | NO transfer — 16k/8k/2k ctx does not reach the 98k regime (G1 ctx lever, round 2) |
| + r8 anchors | + 51 r8 prose cycles on the ENGINE kv state | **1.412-1.431** | a1=1.0 on all 51 = ANCHOR MEMORIZATION (loss->0.0001); ~70 supervised positions, trivially overfit |
| best pack (sb_ckpt_400_gptq) | | **1.4314 k4 1.6078** | crosses both G2 bars numerically — but see decomposition |

Cross-class for sb_ckpt_400 (vs control): gsm8k +0.32 (1.132 vs 0.816 — real,
engine-trace anchors at battery ctx transfer), prose16k −0.16 (0.797 vs 0.953),
code8k −0.25 (0.742 vs 0.992). m_dist r8 [0,30,13,7,1] a_cond [1.0,.41,.38,.125].

## 4. THE G2 VERDICT: DO NOT INTEGRATE (go/no-go to the user)

The 1.43/1.61 crosses the numeric G2 bars, but the decomposition kills the
generalization case: the entire r8 gain is memorization of the 51 evaluation
anchors (corpus-only control = 0.39; a1 = 1.0). Shipping it would regress
mid-ctx prose/code for a gain that only replays on the memorized trace.
Per the mission ladder this is the "+0.15 over control, best pack + analysis"
branch — the pack and evidence are staged for the next agent/user.

THE PRICED NEXT ITERATION (the actual unlock this wave identified):
anchor-scale Stage B. The instrument that moves the 100k regime is ENGINE
decode anchors AT SERVE POSITIONS (h_seed + engine kv base + committed
labels). We have 51; need ~2-10k from FRESH long-prose sessions (NOT r8):
S1 dump harness (DUMP_HIDDENS + cycles + kvd) over 10-30 new 64k-100k novel
sessions = ~2-4h rig GPU window, then Stage B rerun ~1h/$5, scored HELD-OUT
on r8. If held-out lands >=0.8-1.0 k2 -> integrate (G4 battery after).
Alternative priced: engine-trace Stage A (dump engine hiddens for the whole
long corpus on the rig) = ~1h/4M tokens of rig windows — dominated by the
anchor-scale route unless anchor-scale fails.

## 5. Ops notes + laws earned

- **THE FLA CACHE LAW**: transformers Qwen3.5 GDN falls back to the reference
  chunked path (32k forward = 88GB OOM) whenever use_cache=True w/ DynamicCache
  even with flash-linear-attention installed; use_cache=False engages the fla
  triton kernel (32k peak 70.4GB). pip install flash-linear-attention REQUIRED.
- **THE PADDING-ATTENTION LAW**: right-padded batched forwards force the eager
  attention path ([8,32,8k,8k] score matrices); batch-1 unpadded keeps the
  causal fast path. Long dumps at batch 1.
- vast: disk_space is a QUOTA — one host hard-capped overlay at 40G despite a
  400G request (destroy + retry other host). accelerate missing in
  pytorch/pytorch:latest. gvisor stick-at-stopped didn't recur.
- mbpp sanitized uses test_list (not test); sahil2801/CodeAlpaca is GONE —
  HuggingFaceH4/CodeAlpaca_20K works.
- Trainer numerics: SHORT and LONG chain paths agree to 0.0 on identical
  inputs (the segment-attention rewrite is exact); in-place KV scratch writes
  break autograd (list+stack); GQA einsum must contract groups ("agrd,agtd->agrt")
  or scores blow up [A,NH,NKV,T].

## 6. Artifacts

- Packs (Mac ~/drafter/ttt_results/packs/, rig engine0/ttt/packs_candidate/):
  sb_ckpt_400_gptq (BEST, r8 1.4314/1.6078), sb_ckpt_400_rtn (1.4118/1.5882),
  ckpt_6000_rtn (Stage-A best 0.4706/0.5098). Full curve packs: packs_stagea/,
  packs_stageb/ (Mac). Checkpoints: ~/drafter/ttt_results/ckpt_6000.pt,
  ckpt_400.pt (fp16 .pt, 849MB each).
- Rig repo: engine0/ttt/ updated (train.py v2 multi-dir/long/multi-anchor,
  dump_features.py v2 split-windows/offs/hpre, longcorpus.py, stageb_build.py,
  train_stageb.py, run_stagea.sh, run_stageb.sh, gen_prompts.py fixes,
  teacher_gen.py long support). Curve JSON: ~/drafter/phase0/stagea_curve.json.
- Rental DESTROYED (53632319, verified 0 instances); total spend $41.91/60.
