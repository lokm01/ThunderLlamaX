# LLM benchmarks: Apple Silicon eGPU vs native — tokens per second

LLM tokens-per-second benchmarks for **two models** running on a Mac: an RTX
3090 eGPU in a Thunderbolt 4 enclosure on a MacBook Air M2, driven through
the custom DriverKit dext — with a clearly labeled native-Linux reference on
the same GPU class for the "Apple Silicon vs NVIDIA for LLM inference"
comparison. Every number on this page carries its gate context: what
workload, what context length, what exactness tier, and which campaign
journal paid for it. The honesty is a feature — the walls and the
workload-dependence are measured and published, not footnoted away. Deep
dives: [history/CAMPAIGN.md](history/CAMPAIGN.md) (the condensed ladder),
[history/PREFILL.md](history/PREFILL.md) (prefill),
[history/R8_DECODE.md](history/R8_DECODE.md) (the 75-cross),
[history/MM_PLAN.md](history/MM_PLAN.md) (the multi-model/MoE campaign plan;
its per-phase journals MM_P0..MM_P10 sit next to it),
[history/PERFLOG.md](history/PERFLOG.md) (the running log).

The two models:

| Model | architecture | quant | context |
|---|---|---|---|
| **Qwen3.8-27B** (dense) | 48 gated-delta-net + 16 full-attention blocks | IQ3_XXS body (12.6 GB GGUF) | 100,352 |
| **Qwen3.6-35B-A3B** (MoE) | 30 GDN + 10 full-attention blocks, 256 routed experts top-8 + 1 shared per layer (~35.1B total / ~3B active) | UD-IQ4_XS experts (13.9 GiB packed) | 98,304 |

Reference rig: RTX 3090 24 GB (sm_86) in a TB4 enclosure on a MacBook Air
M2, driven through the DriverKit dext. Decode context = the model's full KV
(97,810-token prompt for the dense model; the 96k split for the MoE),
greedy. Prefill "100k rebuild" = full-context ingestion from scratch.

## How to read these numbers (the gate contract)

**"Tier-1 bit-exact" means what it says.** The decode gate: the speculative
engine must emit, token-for-token, the identical sequence a greedy T=1
(one-token-at-a-time) rollout of the same engine produces — **60/60
bit-exact, deterministic across repeated runs, and identical to the stock
tinygrad greedy baseline (59/59)**. This is not "close enough" exactness;
it holds because every batched M-row kernel preserves the T=1 kernel's
per-row floating-point op order exactly, so acceptance can never flip a
near-tie. Every performance rung in the tables below was gated this way
before it counted.

**Prefill gates** are metric-banked: the 2k gate's max-rel-logit-divergence
F must equal its bank value EXACTLY (9.408e-04 @2k; 9.824e-04 @8k after the
P0 fix retired the old "M32 reassociation floor"), GATE A / CTRL / D / D2
token classes must match the bank line-for-line, and the 100k rebuild's
first token must be 4471 exactly. Timing is synced-only (pipelined benches
on this dext inflate up to +145% — refuted as measurements); gates read out
from the FIRST clean run (timing reps advance model state).

**Tier-2** = the one authorized numerics change (see below). Everything else
in every table is bit-identical to the path it replaced.

## Decode @100k context (RTX 3090, tokens per second)

### Headline

| Config | tok/s | ms/cycle | tok/cycle |
|---|---|---|---|
| **K=10 deep-K LOOKUP + draft-skip, through the W5 fixed stack (R8+W5; hit-class)** | **75.81** (75.80/75.51 reps) | ~118 | 8.92 |
| K=10 at the R8 ship (pre-review-fix stack, same numerics) | 75.56 | 118.00 | 8.92 |
| K=9 (R8) | 74.70 | 110.66 | 8.27 |
| K=8 (R7a/P8) | 71.51-72.02 | 104.65 | 7.48 |
| K=2 MTP (W2H) | 40.35 | 68.98 | 2.78 |
| T=1, no speculation | 21.78 | 45.92/tok | 1.00 |

K=10 phase split: draft 5.33 / probe 59.46 / accept 1.16 ms. Acceptance:
E[m|deep] = 10.000 exactly (92/92 deep cycles accepted ALL TEN proposals),
76.7% deep-selected, alpha 3.958. The deep=off superset (kill-switch) is
Tier-1 exact at 42.06 tok/s on the K=10 config. The 75.81 number is the
same K=10 engine re-validated end-to-end after the five-wave review-fix
campaign (85/85 GPU-free tests, live rig Tier-1 60/60 x2 + stock 59/59 —
[history/FIX_CAMPAIGN.md](history/FIX_CAMPAIGN.md)); the fix waves changed
zero numerics, and the ladder reads ... -> 71.51 -> 74.70 -> 75.56 ->
75.81-through-the-fixed-stack. The K=9/K=10 kernel sets ship in
`engine/` (gen_m10/m11, lookup10/11, the ROWS=10/11 spk sets, manifests).

### The decode ladder (every rung Tier-1 60/60 x2 det + stock 59/59)

| rung | tok/s | the lever |
|---|---|---|
| stock tinygrad (start of program) | 4.17 | fp16 KV, scheduler swarm |
| tinygrad-stack hand kernels | 10.95 | a3-family substitutions |
| engine0, K=2 MTP (W2H) | 40.35 | static buffers + graphs + per-step state slots + HMMA attention |
| deep-K K=4 -> K=7 (R5) | 53.65 -> 56.60 -> 58.71 -> 63.11 | n-gram LOOKUP drafter + audited M-extension generators + per-cycle graph-set selection |
| norms per-row CTAs (R7a) | 68.62 | the CTA-serialization law — a grid change, bit-identical, +5.43 tok/s |
| K=8 + draft-skip (R7a) | 71.51-72.02 | lookup-only draft graph on deep cycles (emit byte-identical) |
| K=9 -> K=10 (R8) | 74.70 -> **75.56** | the same generator discipline; **the K-ladder stops at ten** (increments fell to +0.86 tok/s/rung, hit decay ~-1.6pt/rung) |
| review-fix campaign re-validation (W5) | **75.81** | five waves of serving/security/tripwire fixes + fork tripwires — zero numerics change, re-gated live on the rig |
| adaptive T=1 prose mode (P8+A, `TLX_T1_MODE=1`) | quote 75.35 / **prose 14.67 -> 20.56** | after 4 zero-accept K2 cycles the session runs T=1-only cycles (the alpha-death signal); first 8-gram hit exits straight into a deep cycle. Output stream BIT-IDENTICAL to pure spec (120/120) + Tier-1 60/60 x2 |
| P10-dense rung 1: full-vocab EAGLE draft head (`TLX_DHEAD_FULL=1`) | quote 74.85 (bank ~75.5) / **prose 23.0 -> 43.7 tok/s GSM8K median through the API** (22.9 in-harness @100k) | the 40960-row slice head proposed only ~10% of novel-prose targets (89.8% out-of-slice); proposing over the resident full-vocab head plane (zero new VRAM) + the T=1-entry suppression; Tier-1 60/60 x2 + 59/59 stock |

### P10-dense: the prose rungs (rung 1 — the full-vocab draft head; rung 2 — the K=4 verdict)

The dense model's shipped prose drafter is its own checkpoint MTP layer:
`blk.64` carries the full `nextn.*` EAGLE block
(`qwen35.nextn_predict_layers=1`, block_count=65), chained W2-style —
eh_proj -> draft block (own KV) -> head. Rung 1 established the prose
failure was never the chain — it was the **slice head**: the rank histogram
measured **89.8% of novel-prose targets OUT-OF-SLICE** (top1==target 6.1%)
under the 40960-row prompt-frequency slice, so K2 cycles accepted ~0
(tok/cyc 1.02) and the alpha-death controller parked decode in T=1
(~20.6-23 tok/s — the exit needs an 8-gram hit that never comes in novel
prose).

**Rung 1 (`TLX_DHEAD_FULL=1`, shipped default in the dense env):**

- `sheadf.cu` — the Q5_K slice-head GEMV verbatim at VOCAB=248320: every
  chain step proposes over the SAME resident head plane the trunk probe
  reads (zero new VRAM; the draft scratch is logits3 row 0, dead at draft
  time by timeline order). `samxf.cu` — the single-row full-vocab argmax
  writing the token id directly (row index == id; no id-table gather).
- `fill_draft` skips the head+argmax pair under the knob (prompt fill never
  consumes the draft argmax — boot fill 300 s -> 211 s).
- The **T=1-entry suppression**: under the knob a 4-zero-accept streak is
  noise at acceptance ~0.6+/cycle, and a T=1 episode can NEVER exit on
  novel prose — entering it parks decode at the T=1 rate for the rest of
  the generation. Knob-off keeps the alpha-death controller verbatim.
- Gates: Tier-1 60/60 x2 + 59/59 stock; quote-class 74.85 tok/s (bank
  75.53, within boot variance — deep cycles run the lookup drafter
  untouched; only K2/transition cycles pay the +2.8 ms full head). BATCH_B
  >= 2 refused under the knob (the shared logits3 row-0 scratch would race
  across batch streams).
- **Result: prose 23.0 -> 43.7 tok/s GSM8K median through the API (1.90x;**
  30-problem battery at 93.3% — sample noise vs the 95.0% 100-problem bank;
  TTFT 2.37 s unchanged; 10-problem spot re-check 10/10 at 44.1 after the
  rung-2 scaffold landed). The dense now beats the MoE MTP K=4 (36.7) on
  prose.

**Rung 2 (`TLX_EAGLE_K`, default 2 = rung-1 bit-exact): the K=4 EAGLE
chain — built, Tier-1-proven, honestly falsified on performance:**

- The scaffold: a 4-step draft chain with dedicated per-step hidden buffers
  (hd_d0..hd_d3; `accept5e` maps m to the exact chain hidden for m<=3), the
  T=5 probe (the R4 M=5 family with the FFN/down pair swapped to
  `ffn8v5r7`/`down8nw32v5r7` — the gen_r7d* v3->v5 M-extension, 50/64 regs
  0 spill), the graphs5 probe5/accept5 set, and a prose hysteresis:
  `TLX_EAGLE_PROSE_TRIG` consecutive zero-hit miss cycles ENGAGE the K4
  set, any hit disengages, 0 = K4 always (the gating mode). Quote-class
  deep hits keep the K=10 deep set untouched.
- Gates: the r7 twins A/B nz=0 det x2 vs the packed originals; Tier-1 with
  EAGLE_K=4 (TRIG=0) 60/60 x2 + 59/59 stock, quote-class 72.08 tok/s (the
  always-K4 tax on quote misses, -3.7%); quote knob-off BYTE-IDENTICAL to
  rung-1 (70.11 vs 69.96 ms/cyc, tok/cyc 1.88 EXACT — the scaffold costs
  nothing off).
- **The verdict (prose @100k in-harness, the R8_PROSE anchor): K4 94.67
  ms/cyc, E[m|k4] = 0.633, m-dist [70,30,14,6,0] -> 17.25 tok/s; the K2
  control on the SAME build 69.03 ms/cyc, E[m|k2] = 0.583, m-dist
  [68,34,18] -> 22.93 tok/s. K4 = -25% prose; break-even needed E[m] >=
  1.17.** The 1-layer EAGLE drafter (blk.64 nextn) saturates at m=2 — m=3
  fires 5% of prose cycles, m=4 never (0/120). The two extra chain steps
  (+7.9 ms draft: two more full heads) and the T=5 probe (+19.2 ms; the
  class bisect: ffn 28.8 | attn 22.6 | scan 3.8 | norms 6.9 | head 1.9 ms)
  buy +0.05 tok/cycle. The binding constraint is DRAFTER DEPTH FIDELITY
  (the old alpha-program law: acceptance collapses with depth), not the
  machinery — the K4 graph set stays armed behind `TLX_EAGLE_K=4` for a
  future drafter-quality program (DFlash2-class or a trained nextn head).
- **Provenance note (measurement hygiene)**: rung-1's in-harness log line
  ("prose 46.46 ms / 1.0 tok-cyc") came from an INTERMEDIATE build — its
  rep shows 56 T1-cycles, impossible under the final T=1-entry suppression.
  The shipped K2 world is 69.03 ms / 1.583 tok/cyc = 22.9 tok/s in-harness
  @100k; the daemon's 43.7 is GSM8K-class work at ~600-token contexts.
  Re-derive in-harness numbers on the FINAL build before banking them.

Tools: `engine/p10r2_phase.py` (per-graph phase bench),
`engine/p10r2_bisect.py` (probe5 per-kernel-class bisect), the `[k4hist]`
acceptance histograms in test_w100k. Eval artifacts:
[eval/results/gsm8k_p10_dhead_summary.json](../eval/results/gsm8k_p10_dhead_summary.json),
`gsm8k_p10_dhead.jsonl`, `gsm8k_p10r2_spot.jsonl`.

### Decode: the second model — Qwen3.6-35B-A3B (MoE, MM campaign)

The MoE engine (see [ARCHITECTURE.md](ARCHITECTURE.md) for the kernel
classes) runs the same speculative discipline with a mode mix selected per
cycle by a scan oracle: `n>=8 -> D8` (K=8 n-gram LOOKUP) | `n>=4 -> D2` |
miss with a live chain -> **P5+MTP (the first-party MTP K=4 drafter)** |
miss with a stale chain -> one T=1 re-anchor, then seed. All numbers below
are greedy, Tier-1 gated (`spec == T=1`, 60/60 x2 deterministic on the
60-prompt bank, through the daemon and the API).

**Through the full API surface (the P10 ship, MTP mode default-on):**

| workload class | tok/s | vs T1-only serving (P8S) |
|---|---|---|
| quote-alpha (re-reading long documents) | **97.6** | 98.1 — unchanged (lookup class) |
| quote-code | **104.0** | 96.3 |
| quote-docx2 | **65.9** | 25.4 (+160%) |
| prose-0 (novel text, start) | **40.1** | 19.1 (+110%) |
| prose-1 | **39.1** | 19.2 (+103%) |
| prose-9 (deep into novel text) | **29.9** | 18.9 (+58%) |

**In-harness references** (the P7 folded engine, quieter machine): quote-
alpha 118.6-121.6 tok/s at E[m|hit]=8.0 (the depth law transfers to the MoE
— every deep hit accepts ALL EIGHT), T=1 decode 17.2-17.3 tok/s @64k-96k
(57.8 ms/cycle), MTP-mode prose 42-47 tok/s. The daemon/API numbers above
carry the serving stack's ambient; the RELATIVE gains are the deliverable
(the full G-tables are in [history/MM_P8S_results.txt](history/MM_P8S_results.txt),
[history/MM_P9_results.txt](history/MM_P9_results.txt) and
[history/MM_P10_results.txt](history/MM_P10_results.txt)).

**The MTP acceptance measurements** (the first-party drafter = the model's
own `blk.40` MTP layer, quantized in-pack; the EAGLE/llama.cpp
one-cycle-late chain shape — seed on the committed hidden, then K=4 draft
steps; every KV row the chain reads is true by construction):

| class | P(accept first draft) | E[accepted] @ K=4 | tok/cycle |
|---|---|---|---|
| quote (wiring check) | **0.875** | 2.75-3.08 | 3.75-4.08 |
| prose-0 | 0.875 | 2.58 | 3.58 |
| prose-9 | 0.708 | 1.79 | 2.79 |

Depth-1 acceptance is 0.875 EXACTLY the offline anchor number — the shipped
MTP layer drafts strong on these weights (depth-4 conditional acceptance
still >= 0.74 on quote/prose-0). The chain goes stale on lookup cycles
(quote work keeps the untouched lookup path) and re-anchors with ONE T=1
cycle on the first miss after a hit streak. Long-ctx (96k) MTP alpha is
measured only at the harness level so far — an honest open item.

**Prefill (MoE)**: chunk-256 bit-exact prefill runs 181-204 tok/s @2k-16k
(class honest: the grouped M-GEMM is the 850+ path if it ever lands); a
full 96k context feeds in ~800 s (~122 tok/s end-to-end incl. anchors) and
64k in ~461 s (~140 tok/s). Context-ladder exactness: the engine state
after 4,000/16,288 doc tokens continues greedy-identically to the fp16
anchor (rebase16 16/16 EXACT at every rung — the LONG-HORIZON law; GDN
norms bounded).

**The fusion that didn't ship (P10-B, honestly falsified):** the pairwise
MoE kernel fusion (router+shared-expert, gate+down merged per pair — zero
spill, verbatim bodies) DIVERGED from the unfused engine at token 27 in the
bit-exact gate: NOT shipped (`MM_FUSE2` stays off). Its launch-count arm
(-74 launches for -1.34 ms of 52.1) also corrected a law: fat-kernel graphs
pipeline dispatch behind execution, so the 0.094 ms/launch serialization
slope measured on tiny-kernel graphs does NOT transfer — the T=1 cycle is
kernel-RUNTIME-bound (the marginal launch is ~0.02 ms). Future fusion work
must make the KERNELS faster (fewer weight passes), not the launch count
smaller.

## Batched decode (the B axis — R6, opt-in)

One M=BT probe trunk shares the weight read across B concurrent streams;
stateful kernels run per-stream over sliced scratch + state banks. ZERO new
CUDA kernels; per-stream Tier-1 bit-exact at every rung (60/60 x2 det each).
Full story: [history/R6_BATCH.md](history/R6_BATCH.md).

| Rung (B=2) | config | aggregate tok/s | vs best solo |
|---|---|---|---|
| mixed 100k+8k (3+5) | harness, K2/K4 solos | 66.2 | 1.32x |
| both-8k (3+5) | harness | 69.3 | 1.38x |
| both-8k (5+5, BT=10) | harness | 78.3 | 1.56x |
| **+ batched draft-skip** | harness | **81.33** | **1.61x** |

**The honest serving numbers (Phase 3, through the API with full deep-K solo
baselines):** deep-heavy pair 89.7 solo vs 68.0 batched (0.76x); prose pair
19.6 vs 22.1 (1.13x); live-rig W5 smoke 79.40 aggregate (1.20x service-
measured). The 1.5x+ harness multipliers were measured against K2-only solos;
against the production deep-K DecodeSession the per-stream stateful kernels +
M-trunk widening cost more than the shared weight read saves at B=2. Batch
mode therefore ships DEFAULT-OFF, as a documented opt-in (concurrency/
fairness: two streams at ~57-80% each instead of 100%/queued) — not an
aggregate win on this engine at B=2. The B axis needs M6+/wider kernel
families (the campaign), not more wiring.

## Prefill (fresh prompt ingestion, tokens per second)

| Rung | 2k FRESH | 8k | 100k rebuild | % of 662 ref @100k |
|---|---|---|---|---|
| T=1 chunked prefill (start) | ~114 class | ~106 class | ~21.8 tok/s | 3% |
| P6 (M=32 GEMMs) | 245.0 | 190.8 | 165.3 | 25% |
| P17 (wide-M attention) | 373.5 | 347.3 | 248.2 | 37.5% |
| R2b (WY-C32 chunk scan) | 401.7 | 371.3 | 255.8 | 39% |
| R2c (shared packed7 plane + M=128 trunk) | 494.4 | 451.3 | 314.3 | 47.5% |
| R2d (ring-4 gdnqg + attnqkv g=448) | 503.6 | 457.8 | 317.6 | 48.0% |
| P8 (packed5 + o-proj fold) — Tier-1 ship | 530.6 | 479.4 | 328.0 | 49.5% |
| **T2 (W4A8 IMMA ffn) — current default** | **569.2** | **510.1** | **342.0** | **51.7%** |

All numbers are prompt-ingestion tok/s including the batched draft fill.
26x over the T=1 start @2k.

### Tier-2: the one authorized numerics change

`PF_W4A8=1` (default on, kill-switched): the prefill FFN GEMM reads the
existing packed7 planes through a linearized int4 codebook view of the IQ3
weights (8 linear levels, delta=4.0507 = 1.49% weight-RMS; a 2.4%
logits-class shift) and runs as a fused int8 tensor-core GEMM. Zero new
VRAM; decode stays Tier-1 bit-exact with the planes resident; **unset the
knob and prefill is byte-identical Tier-1 (530.6/479.4/328.0), verified
line-for-line**. Battery + banks: [history/T2_P8W4.md](history/T2_P8W4.md).

## Service timings

| Operation | Time |
|---|---|
| Cached 100k-context restore, dense (prompt cache hit) | **~6.5 s** (vs ~13 min FRESH; 60/60 exact across 4 restarts) |
| MoE prompt-cache node (per 1024 tokens) | ~76 MB (+10.4 KB/token); CACHE_HIT continuation EXACT vs the FRESH arm (G4) |
| FRESH 100k prefill (dense, at 342.0 tok/s) | ~4.8 min |
| FRESH 8k prefill (dense, at 510.1 tok/s) | ~16 s |
| FRESH 96k feed (MoE, chunk-256 + anchors) | ~800 s (~122 tok/s end-to-end) |
| 200-token follow-up turn @2k-class (resident state) | ~2.1 s (vs ~6.1 s FRESH) |
| Daemon boot to ready @100k ctx (dense) | ~6-7 min (weights ~11 s warm + KV quantize ~40 s + warmup + graphs) |
| Model swap (enginectl switch, dense <-> MoE) | the graceful stop + GPU-EXIT reboot + boot of the target model (minutes-class, by design — the reboot IS the transport; the swap intent survives it) |
| Cancellation latency | next cycle boundary (tens of ms) |

**Serving soaks** (the honest reliability numbers): dense W5 soak 15 min /
32 rounds zero faults; MoE P8S soak 148 rounds / 15 min zero failures;
MoE P10 (MTP default) soak 106 rounds / 15 min zero failures — streams +
follow-ups + quote classes + cancels + health polls, dirty never set. The
R3/L7 hardening behind this: 51 review findings fixed in five waves +
the abort-safety protocol (below). Post-P10, the dense soak record is a
**42.7-min / 300-problem GSM8K run with ZERO engine deaths** (death
watcher clean, engine pid unchanged) — the same traffic class crashed
every 13-18 min before the kernargs-slab pool.

## Output-quality eval battery (P9 — the first quality numbers)

All prior gates were bit-exactness; this battery measures what the models
actually answer, through the live serving stack (harnesses + per-item
results in [../eval/](../eval/), narrative in
[../eval/P9_EVAL_RESULTS.md](../eval/P9_EVAL_RESULTS.md), journal
[history/MM_P9E_results.txt](history/MM_P9E_results.txt)).

**GSM8K** (4-shot primer from the paper's train split, FIRST 100 test
problems, greedy, `enable_thinking=false`, max_tokens=320, streaming):

| Model | Accuracy | Decode tok/s (med) | TTFT med | Latency med |
|---|---|---|---|---|
| Qwen3.8-27B dense (K=10 lookup) | **95.0%** | 23.0 | 2.4 s | 8.4 s |
| Qwen3.6-35B-A3B MoE (MTP K=4) | **93.0%** | 36.7 | 9.5 s | 13.2 s |

Zero answer-extraction failures on both legs. Caveats, honestly: the dense
leg rode ~10 engine deaths (the F2 crash-loop, since fixed) via a resume
driver; both legs predate the first-token fix (F1), but the lost token was
always the answer echo's leading word, never the final number — the scores
stand. Post-fix, the first-20 GSM8K rerun was 20/20 identical predictions.
The MoE TTFT is its weak spot (~66 tok/s effective FRESH prefill through
the stack; no cross-conversation cache hits by design in this battery).
A post-P10 300-problem soak run scored 94.0% at 23.2 tok/s median decode —
performance unchanged, zero deaths.

**Perplexity** (MoE, engine-side teacher-forced NLL, 118-135 tok/s scoring
rate; identical corpora + tokenizer for future apples-to-apples regression
deltas):

| Domain | Tokens | NLL/tok | PPL | Greedy next-token acc |
|---|---|---|---|---|
| prose (Pride & Prejudice) | 32,767 | 0.1304 | 1.139 | 96.5% |
| code (the serving layer) | 10,239 | 2.9931 | 19.95 | 53.7% |
| prose_private (fix-campaign journal) | 3,071 | 3.5374 | 34.38 | 38.4% |
| code2 (MoE kernel library) | 8,703 | 1.5712 | 4.812 | 70.8% |

**The memorization caveat**: the Gutenberg classic is verbatim-memorized
(96.5% greedy next-token) — its PPL 1.14 is a CONTAMINATION FLOOR, not a
quality number. The honest domains are code 19.9, code2 4.8,
prose_private 34.4. These are the program's first PPL baselines.

**Dense perplexity** (Qwen3.8-27B, the same engine-side teacher-forced
harness — ppl_dense.py, 193-208 tok/s scoring; the m128 trunk + FP16 head
make dense scoring FASTER than the MoE twin's 118-135; results in
[../eval/results/ppl_dense.json](../eval/results/ppl_dense.json)):

| Domain | Tokens | NLL/tok | PPL | Greedy next-token acc |
|---|---|---|---|---|
| prose (Pride & Prejudice) | 32,767 | 1.4731 | 4.363 | 63.3% |
| code (the serving layer) | 10,239 | 1.5910 | 4.909 | 66.4% |
| prose_private (fix-campaign journal) | 3,200 | 2.9196 | 18.53 | 44.1% |
| code2 (MoE kernel library) | 8,704 | 1.0248 | 2.786 | 77.9% |

**Dense vs MoE, cross-model**: dense WINS every honest domain — code 4.91
vs 19.95, prose_private 18.53 vs 34.38, code2 2.79 vs 4.81 (the 27B dense
reader beats the A3B router on unfamiliar code and private jargon; the
quant difference — IQ3_XXS vs UD-IQ3_S — is not the dominant term at that
gap). On the contaminated prose domain the dense model shows the honest
4.36 while the MoE's 1.14 memorization floor hides its true prose level.
Numerics note: dense logits are the engine-native FP16 head output (the
same values greedy decode decides on); the MoE head emits FP32; NLL
accumulation is float64 on both.

**Long-context needle** (dense; 5-digit code embedded in Gutenberg filler,
chat API, thinking off):

| Battery | Result |
|---|---|
| 61,189-token contexts (60k class), 10 trials @ 5-95% depth | **10/10 clean, 10/10 EXACT retrieval** (post-P10; e.g. code 53177 -> exactly "53177", ~331 tok/s end-to-end on the manual trial) |
| ~20k-token contexts, 10 trials | 10/10 clean, 8/10 exact retrieval |
| Pre-fix (P9, for the record) | 10/10 FAIL at both 60k and 20k — the F3 sysmem exhaustion blocked every long-prompt prefill; one clean manual 61k datapoint demonstrated the capability |

The needle battery is the F3 regression test: it fails loudly the moment
graph-build mapping exhaustion returns.

**Quality findings that became fixes** (full ledger in
[../eval/P9_EVAL_RESULTS.md](../eval/P9_EVAL_RESULTS.md)): F1 first-token
loss (every completion dropped its first token; one-token answers came
back empty — fixed in the API layer), F2 dense crash-loop + F3 long-prompt
mapping exhaustion (both dead via the P10 kernargs-slab pool: `mapfd`
counters freeze at 85 with ka_reused=1916 vs ka_fresh=42 lifetime), F5
dense direct-PPL device fault (RESOLVED — harness bug, one line: the
logits copyout read VOCAB*4B from a VOCAB*2B FP16 buffer, 496KB past the
end; the fix downloads FP16, the engine-native head output, and the full
4-domain battery above ran clean), F7 the staydown-marker law.

## What this rig can and can't do (measured)

**Decode speed is workload-dependent — measured, per model, per mode:**

- **Hit-class** (documents, code, quotes, repetitions — the model re-reading
  text it has): **75.81 tok/s** dense / **97.6-104.0 tok/s** MoE. The n-gram
  drafter fires on 76.7% of dense cycles and every hit accepts all ten
  (dense) / all eight (MoE).
- **Prose-class** (novel text): the honest floor has MOVED three times, each
  time by shipping a first-party drafter rather than a longer lookup window:
  - dense, adaptive T=1 mode (P8+A): **20.56 tok/s** (was 14.67 pure-spec)
    — after 4 zero-accept K2 cycles the session switches to T=1 cycles and
    the per-cycle lookup scan keeps the exit trigger live; the mixed-mode
    output is BIT-IDENTICAL to pure spec (120/120 positions). Superseded by
    the P10 draft head but kept verbatim knob-off.
  - dense, full-vocab EAGLE draft head (P10-dense rung 1, shipped ON):
    **43.7 tok/s** GSM8K median through the API (1.90x over the 23.0
    battery baseline; 22.9 tok/s in-harness @100k) — the checkpoint's own
    blk.64 nextn layer proposing over the full 248320-row vocab instead of
    the 40960-row slice (89.8% of prose targets were out-of-slice).
  - MoE, first-party MTP K=4 chain (P9/P10-A, shipped default): **40.1
    tok/s** prose-0 through the API (was 19.1 T1-only), 29.9 at prose-9.
  An earlier reading on this page bounded prose at "~15 tok/s with
  56-70 tok/s only via a future DFlash2-class drafter" — that ceiling was a
  category error (it priced the K2-cycle weight floor, not the drafted
  modes). The measured answer is 43.7 / 40.1 with the shipped drafters; the
  BIMODAL MATCH LAW still holds (the lookup tier gains nothing on prose —
  the wins above come from the first-party drafters and T=1 mode-switching,
  not from lookup).
- **What is still honestly open on prose**: the dense K=4 chain is built and
  Tier-1-gated, but its 1-layer drafter saturates at m=2 — K=4 measured -25%
  prose (break-even needs E[m] >= 1.17), so a DFlash2-class or trained
  nextn drafter is the unlock; the MoE's long-ctx (96k) MTP
  alpha is unmeasured; prose-9 decay (29.9) is real and unsolved.

**The measured walls (why prefill stops where it stops):**

- The dext is **1 CTA/SM hard**; the single-CTA W-stream ceiling is ~490
  GB/s, and the GEMM wall is **latency-ordering, not issue-rate** (the E1
  issue-law microbench: warp-spec headroom is real at 1.8-1.9x, but
  meet-based conversions of it measured x0.82-0.96 — falsified).
- **48 KB static-smem cap, no cp.async** (cp.async device-faults on this
  dext). The GEMM pool is 159 of the 256 ms 2k chunk; the position-growth
  pool is the wide-attention kernel (41.5% of the chunk at 97k).
- **750 prefill was closed, honestly**: the x4.58 IMMA reading that
  motivated it was a discriminator frame artifact (one-plane M=64 vs
  two-plane M=128 arms); the true fused-shape advantage is x1.06, and the
  x1.56 int4-plane variant is VRAM-dead at 100k state. The shipped take is
  the honest +7.3/+6.4/+4.3%. The remaining measured route is a
  persistent-CTA megakernel where the pipeline lives inside one CTA.

**Decode ceilings**: the dense K-ladder is done (K=10; increments < +1
tok/s/rung); the probe is 59.5 of the 118.0 ms K=10 cycle; deeper-K
continuation was priced offline at K=12-16 optimum ONLY IF per-rung cost
halves (D4). The MoE T=1 cycle (52-58 ms) is kernel-RUNTIME-bound (the
P10-B marginal-launch law) — its compression route is fewer weight passes
per layer, not fewer launches.

## Comparison context: eGPU on a Mac vs native Linux (clearly labeled: NOT this stack)

Reference numbers from a **native Linux stack on the same GPU class**
(cloud 3090, syv-ai vLLM W4A16, measured during the R0 pre-check —
[history/W0_DOSSIER.md](history/W0_DOSSIER.md)):

| Metric | Native Linux reference | ThunderLlamaX (this stack) | Ratio |
|---|---|---|---|
| Greedy decode @100k | 86.05/86.34 tok/s | 75.81 tok/s | **88%** |
| Cold prefill @100k | 662 tok/s | 342.0 tok/s (Tier-2) | 51.7% |
| Prefill @2k | ~660 class | 569.2 tok/s (Tier-2) | ~86% |

From behind a Thunderbolt cable and a userspace driver, with no NVIDIA
stack, at 88% of the reference decode. The remaining decode gap is the
probe's 59.5 ms (measured attribution above); the prefill gap is the
structural wall set (1 CTA/SM, no cp.async, 48 KB smem) — see
[ARCHITECTURE.md](ARCHITECTURE.md) and [history/R7_DECIDERS.md](history/R7_DECIDERS.md).

## Where the numbers come from

Every row above traces to a gate battery in a campaign journal — the
session-by-session record with configs, env lines, and the refuted-approach
ledgers. Start at [history/README.md](history/README.md) (the index), then
[history/CAMPAIGN.md](history/CAMPAIGN.md) for the condensed program
narrative. Baseline token files for the exactness gates live in
`baselines/`.
