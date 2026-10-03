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

The two benchmarked models:

| Model | architecture | quant | context |
|---|---|---|---|
| **Qwen3.8-27B** (dense) | 48 gated-delta-net + 16 full-attention blocks | IQ3_XXS body (12.6 GB GGUF) | 100,352 |
| **Qwen3.6-35B-A3B** (MoE) | 30 GDN + 10 full-attention blocks, 256 routed experts top-8 + 1 shared per layer (~35.1B total / ~3B active) | UD-IQ4_XS experts (13.9 GiB packed) | 98,304 |

A third registry model — an abliterated weight-variant of the dense
checkpoint — runs the same numbers as the dense model (same kernels,
weight-only swap; measured within noise of it: prose wall 34.5 vs 33.3
tok/s, prefill @100k fill 452 vs 440-447 s — see
[DEPLOY_OBLITERATED.md](DEPLOY_OBLITERATED.md)). It is a
serving-capability demonstration, not a new performance class.

**Concluded — the drafter-quality program (verdict: instruments kept, training
recipe honestly falsified, the shipped drafter validated).** The program set
out to retrain the checkpoint's own 1-layer draft block (the shipped one
saturates at ~2 accepted tokens; the K=4 chain built on it measured -25% and
ships off). Three rental-GPU iterations later (~$59 total cloud spend), the
verdict is in: **the shipped pack stays — now better-understood — and the
instruments are permanent.**

- **The canary reframe (the headline).** The old r8 evaluation anchor was a
  hard outlier: an idiosyncratic model-voice continuation trace. Scored on a
  held-out battery of fresh-novel sessions at true serve positions, the
  SHIPPED drafter is already near this method's ceiling:

  | control = shipped pack | E[m]|k2 | E[m]|k4 | first-token acc |
  |---|---|---|---|
  | r8_prose @97.9k (the old hard anchor) | 0.549 | 0.549 | 0.412 |
  | fresh-novel canary @64k-99.4k (36 held-out sessions) | **0.979** | **1.197** | **0.634** |

  The r8 trace understated the engine's novel-prose acceptance by ~0.43 k2;
  on representative prose the shipped pack sits at the door of the
  E[m]k2 1.0-1.3 ship band. **No published benchmark changes** (43.7 tok/s
  daemon GSM8K stays the shipped prose figure) — the canary numbers are
  evidence the real-world prose class is stronger than the hard-anchor
  in-harness figure (22.9 @100k), not a new shipped number.
- **Three falsifications, three sentences each.**
  1. *Corpus-scale Stage A* (26.5M tokens, 8 classes, rented H100): training
     on human text teaches "plausible human continuations", not THIS model's
     greedy stream — greedy-vs-corpus agreement measured 0.625-0.645, and
     every checkpoint stayed below the 0.549 control on r8 prose.
  2. *Anchor-scale Stage B v2* (8,829 engine decode anchors at 64k-99.4k from
     36 fresh-novel sessions, protective mix, clean held-out generalization
     with zero memorization): the Stage-A lineage still trades prose down for
     battery up (canary 0.979 -> 0.741, prose16k -0.14, gsm8k +0.49) —
     falsified for prose a third time.
  3. *Clean-slate v3* (pristine first-party bf16 init, LR 2-3e-6 bracketed,
     held-out canary-CE selection): the packed canary k2 lands BELOW its own
     no-training init at every LR and step count (0.943/0.913 vs 0.970-0.987)
     while canary CE improves 2.41 -> 1.25 — the trainer's own-chain
     objective anti-correlates with engine-conditioned acceptance under this
     recipe. Init, LR, data-scale, and selection are all exonerated; the
     recipe itself is falsified.
- **The preserved asset**: a GSM8K/battery-class opt-in draft pack
  (+0.22-0.49 E[m]|k2 on battery workloads, reproduced across two
  independent training runs while every prose class pays) — selectable per
  deployment via `TLX_DRAFT_PACK`, never the default.
- **What shipped: nothing.** The instruments are permanent infrastructure:
  `engine/chain_sim.py` (the G0-calibrated offline acceptance simulator),
  the 36-session canary battery + anchor-dump harness
  (`engine/anchor_scale_dump.py`), and the TTT trainer pipeline
  (`engine/ttt/`, validated bit-identical to the engine chain). Verdicts:
  [history/TLX_P1_RESULTS.md](history/TLX_P1_RESULTS.md) (Stage A),
  [history/TLX_P2_ANCHORS.md](history/TLX_P2_ANCHORS.md) (the canary
  reframe + v2), [history/TLX_P2B3_VERDICT.md](history/TLX_P2B3_VERDICT.md)
  (the final verdict + runbook).

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

**Prefill (MoE)**: after the Session A+B+C+D+E campaign (below), chunked
prefill runs **982.0 tok/s @2k · 852.9 @8k · 703.8 @16k** (pre-campaign
stock 203.9 / 187.9 / 171.7 — **4.82x cumulative @2k**), a full 96k
context feeds at 254.5 tok/s (was 92.4). GSM8K-class TTFT on a
1,216-token fresh generation: **~2.3 s** class (2.27 s cold measured in
Session D; Session E's env flip costs one cold pcache rebuild on first
boot). Sessions A+B were bit-exact; Sessions C/D/E ship six Tier-2
numerics moves, every one under the per-seat cross-entropy quality gate
(below — Session E's combined gate measured dCE −0.695%, IMPROVED).
Context-ladder exactness:
the engine state after 4,000/16,288 doc
tokens continues greedy-identically to the fp16 anchor (rebase16 16/16 EXACT
at every rung — the LONG-HORIZON law; GDN norms bounded). One honesty
note on provenance: the Session-E ladder numbers are HARNESS-level (each
rung gated on independent boots); the daemon-level re-gate battery is
staged and pending a physical cold-cycle of the rig (see the Session E
section).

### MoE prefill: the Session A+B campaign (203 -> 384 tok/s @2k, bit-exact)

The campaign opened with a five-analyst plan whose central premise was a
**25-30x DRAM amplification wall**: the routed pair-walk re-reads each
touched expert's weights per token, so a 256-token chunk touching ~all 256
experts should be DRAM-bound many times over — and the plan's #1 lever was
traffic reduction (grouped experts), #2 a raster swap. Session A built the
measurement instruments first (G0 truncated-graph bisect, L1 POC kernels,
in-graph probes — `engine/mm/mm_pf_bisect.py`, `mm_l1_poc.py`,
`mm_probes.py`) and **falsified the premise before anything shipped**:

| phase (per 256-chunk @2k) | modeled by the plan | measured (G0 bisect) |
|---|---|---|
| trunk GEMVs | 58-62% | 43.6% |
| routed experts | 20-24% (the wall) | **44.6% — co-equal, not dominant** |
| shared expert + router | — | 7.1% |
| attention | — | 1.8% |
| GDN scan | — | 2.3% |

Three measured kills followed: (1) **the amplification wall does not exist**
— the stock pair-walk kernels stream ~750 GB/s effective in isolation (L2
absorbs the re-reads); the 212.5 GB/s seen in-graph was contention, not the
kernel's ceiling (DRAM read path re-confirmed at 820.6 GB/s); (2) the
**raster-swap lever was killed at 1.03x** (all seven trunk classes measured
0.97-1.04x, bit-exact); (3) the **real expert-load histogram** (in-graph
router-capture probe) is nothing like the plan's Poisson-30: prose/code/
gsm8k chunks put a **maximum of 253/212/217 tokens on a single expert**
(8x the model), top-16 experts carry only 11.6-14.4% — a fat tail that
shaped the grouped-kernel bins. The plan's #1/#2 levers were swapped
accordingly (the seat-loop trunk first, grouped experts second), and the
L1 gate (trunk < 50%) fired as designed.

Session B then shipped the winners — every changed path **bit-exact**:

| rung | 2k | 8k | 16k | 96k feed |
|---|---|---|---|---|
| stock chunk-256 + per-token tail | 201.4 | 187.4 | 172.0 | 92.3 |
| **+ MM_PFG / MM_PFM / MM_PF64 (Session B)** | **384.3** | **331.5** | **284.2** | **117.0** |

- **`mmsort8`** — a deterministic counting sort (atomic-free rank-count
  scatter) turning the router's expert ids into `eoff`/`plist` bin tables;
  det x2 and exact vs the numpy simulation on real routing, all 12 MoE
  layers. The gold router itself stays `rt8e256` VERBATIM (the bit-exact
  top-8 contract).
- **`gxm_*` grouped expert GEMMs** — the campaign's honest detour: the
  v1 design (traffic-reduction-only: smem-staged weight bytes, dequant per
  token) measured **0.86x and was killed** — the pair-walk is
  LATENCY-bound on the dequant+dot chain, not DRAM-bound, so removing
  traffic buys nothing. The v2 winner is the **register-decode** design:
  each warp decodes its row's weights ONCE per (row, block) into registers
  (TS-times less dequant work) and the m-loop is pure FMA over the bin's
  tokens staged in smem — per-(pair,row) operation order kept verbatim, so
  every expert lane is bit-exact vs the pair-walk (measured 2.0-3.6x on
  the up GEMMs, 1.27-1.41x on down).
- **`gvs32*` seat-loop trunk GEMVs** — ports of the winning L1 classes
  only (qkv 1.54x, q 1.57x, z 1.38x, k/v 1.23x); `out`/`o`/`ab` stay
  stock because the k=4096 pair already runs at ~750 GB/s (measured
  0.96-0.97x — nothing to win).
- **`shgu32`/`shdn32`** — the shared expert M-batched (weights read once
  per chunk instead of per seat), 1.55x, bit-exact.
- **The PF64 tail graph** — the TTFT lever: the 1..255-token tail that
  used to run per-token (~52 ms/token) now runs 64-seat chunks, bit-exact
  vs the per-token tail. **GSM8K-class TTFT (n=1000 battery): ~15.5 s ->
  ~5.5 s** (the tail was ~70% of TTFT).
- **The carveout dependency**: the new kernels carry 16.4 KB smem, and the
  default 32 KB config means 1 CTA/SM; `NV_SMEM_CFG_AUTO` at 64 KB gives
  3 CTAs/SM — the measured 2.0-3.6x lever on the grouped up GEMMs.

**The battery** (all through the shipped env, kill-switches restore the
stock sequence byte-identically; the three `MM_*` knobs are config_fp keys,
so the ship cost exactly ONE cold MoE pcache rebuild): G2 sort det-x2 +
numpy-exact; grouped per-pair bit-exact across all four quant lanes on
real routing; **F1b full-chunk compare stock-vs-new: 0 of 524,288 words
mismatched** with router expert-ids bit-exact through the new path; bank60
60/60 `spec == T=1` bit-exact x2 deterministic THROUGH THE DAEMON; pcache
CACHE_HIT at 4,096 tokens with exact restore continuation; an 80-round
soak with zero failures; decode classes unchanged (quote ~97-99, prose
~30-40 tok/s — the campaign touched prefill graphs only).

**Residual attack list (post-B, priced not built):** the trunk k=4096
latency floor (`out`/`o` already at ~750 GB/s — a structural change, not a
port), the routed-up fp32 compute floor (dequant-to-fp32 math per FMA
chain), the `gxm_dn` act-restaging cost that dominates the down GEMMs
(the mma/L1b branch), and the attention share at 96k (55.1% of the chunk
at 96k, growing 15.8 ms per 1k context — the L4 branch). Banked for that
future: Session A's dynamic-shared-memory probe **PASSED at 64 KB and
96 KB** via the QMD-patch route (static >48 KB is refused by ptxas) — the
structural branch (staged weights + deeper bins) is alive and measured.

### MoE prefill: the Session C campaign (384 -> 494.8 tok/s @2k — the residual attack, Tier-2)

Session C attacked that residual list directly. Two kernels shipped, one
was honestly skipped, and one mystery from Session B was adjudicated as a
harness bug:

| rung | 2k | 8k | 16k | 96k feed |
|---|---|---|---|---|
| Session B (bit-exact class) | 384.3 | 331.5 | 284.2 | 117.0 |
| **+ MM_PFT trunk mma + gxm_dnf fold (Session C)** | **494.8** | **409.7** | **339.8** | **125.4** |

(**+29% @2k this session; 2.43x cumulative** over the 203.9 pre-campaign
stock. TTFT on a 1,216-token fresh generation: **3.18 s** x2 — was ~5.5 s
after Session B, ~15.5 s before the campaign.)

- **`pgmq8m32` — the trunk mma M-GEMM (the L1b lever, `MM_PFT=1`).** The
  Session-B verdict on the `out`/`o` k=4096 pair was "already at ~750
  GB/s in isolation, nothing to win" — but in-graph the pair runs at the
  ~212 GB/s contention floor (PB1's corollary working in reverse: the
  isolated number was the kernel's ceiling, not the graph's). The fix is
  the dense engine's M=32 mma design (two `m16n8k16` fragments sharing
  every staged W tile — the `pf_gemm3m` template) ported to the MoE Q8_0
  weights: per 128-wide k-chunk each lane owns exactly one 34-byte Q8_0
  block, activations staged fp32->fp16, weights dequantized
  `d*(float)q`->fp16, `mma.sync.f32.f16.f16.f32`, epilogue
  `hres + acc` (the stock residual contract). **Isolated POC: 8.18x
  (`out`) / 7.40x (`o`)** vs the in-graph stock pair — the session's
  honest lesson, because in-graph the lever delivered **~2x on the pair,
  +29% end-to-end** (mma-tile contention with the rest of the chunk; see
  PC3 in [DEXT_LAWS.md](DEXT_LAWS.md)).
- **`gxm_dnf` — the routed-down act-restaging fold (bit-exact).** Session
  B's `gxm_dn` restaged the 16 KB activation block per (row-block, b) —
  128 stages per item per CTA from L2. The fold stages the FULL 512-wide
  activation ONCE per item into `xsm2[TS][512]` (32 KB smem -> the 100 KB
  AUTO carveout = 3 CTAs/SM) and the row-block loop becomes pure
  dequant+FMA with zero interior syncs. Per-(pair,row) math verbatim —
  **BIT-EXACT by construction**, 1.36x (uniform routing) / 1.32x
  (skewed). Rides `MM_PFG`; no new config key.
- **The honest skip: routed-up fp16 mma.** The same mma trick on the
  routed up-projections would have cost 27x the error class of the banked
  F metric for roughly -30 ms — skipped, and the mma port of the
  register-decode up kernels queued instead (the honest lever).

**The Tier-2 battery (the mma pair is a numerics move — the first MoE
prefill change that is not bit-exact):**

- **The F-metric bank**: isolated relerr vs fp64 — `out` F = 1.48e-4,
  `o` F = 1.41e-4 (the stock pair itself sits at ~1.1e-7; the dense
  engine's M32 class banks 9.4e-4, so this is the better-than-precedent
  class). Full-chunk F1b through the new path: hidden-state F = 0.070,
  seat-255 logits F = 0.115, determinism x2, **0 top-1 flips on the
  9 sampled seats**.
- **THE NEW QUALITY INSTRUMENT — per-seat next-token cross-entropy.** The
  router boundary moves under any reordering-class numerics change:
  **5,453 of 81,920 router slots (6.7%) flip** their expert draws
  (distributed from layer 0 — near-tie boundaries, not a broken layer).
  The question a bit-fidelity gate cannot answer is whether that drift
  matters. The instrument (`mm_pplc.py`): run the SAME 256-token prose
  chunk through the stock and the Session-C prefill paths, then score
  each seat's true next token under both heads — **mean CE 0.9014 (stock)
  vs 0.9018 (new), delta +0.046%, within one standard error of the mean**
  (SEM ~0.100 both arms), **top-1 agreement 252/256** (251/256 excluding
  the last seat). The router-boundary drift is quality-neutral. This CE
  compare is now the standing gate for numerics-class changes — the
  project measures them by next-token cross-entropy, not just
  bit-fidelity.
- **spec == T1 through the daemon**: the 60-prompt bank re-gated x2
  through the live MM_PFT daemon — 60/60 exact warm; the ONE cold-boot
  first-gens mismatch **did not reproduce** (warm x2 + cold-boot
  exact-shape x1) and was adjudicated with the tieprobe as a **rare,
  non-reproducible, degenerate quote-loop near-tie PHASE-FLIP** between
  the am-head and verify-argmax paths — a PRE-EXISTING class (the same
  transient shape hit the OLD numerics daemon, pre-MM_PFT). Documented
  watch-item, not a regression.
- **pcache CACHE_HIT under the new config_fp namespace** (MM_PFT is a
  config_fp key): 7,168 cached tokens restored, second arm 4.61 s.
- **Decode classes untouched** (the mma pair is prefill-only): quote 42.5,
  prose 46.9 tok/s through the daemon; **104-round / 900 s mixed soak,
  zero faults**.
- **The G3-after-G2 mystery, adjudicated.** Session B closed on an
  unresolved FRESH-t1-vs-FRESH-spec divergence at 4k. Session C proved it
  was a **probe-harness double-append bug**: the `gen` RPC collected
  tokens from BOTH the per-cycle events AND the done event, so every
  stream was doubled — and comparing doubled streams manufactures a
  phantom first-mismatch exactly at the real-length boundary. With the
  harness fixed (line-draining RPC reader + de-duplication discipline),
  t1-vs-spec **matched at 1k/2k/3k/3328/3584/3840/4096 + alt-4k** across
  60/200/600-token windows. The residual after the fix is the
  non-reproducible phase-flip class above.

**Target vs actual, honestly**: the session priced ~700 tok/s @2k from
the isolated POC and delivered 494.8 — the 8.18x isolated compressed to
~2x in-graph (tile contention), and the battery, not the target, gated
the ship. The measured 2k ceiling with the queued levers is ~700-900.

**Residual attack list (post-C, from the fresh family bisect at five
context lengths):**

| family | share of the 2k chunk | share of the 96k chunk |
|---|---|---|
| routed experts (grouped) | 37.4% | 9.7% |
| trunk GEMV/GEMM | 36.4% | 9.5% |
| shared expert + router | 11.8% | 3.0% |
| GDN scan | 8.3% | 2.4% |
| attention | 6.4% | **75.9%** |

The Session-D queue, in order: (1) the **L4 wide-attention** — 76% of the
96k chunk, growing 15.9 ms per 1k context (the dense engine's P18
wide-attention playbook applies); (2) **k=2048 trunk mma ports** — the
`pgmq8m32` template generalizes to the seat-loop classes that now carry
the trunk share; (3) the **routed-up mma port** (the register-decode
design, not the fp16 shortcut that was skipped). At 2k the remaining
weight-class split (routed 37.4 / trunk 36.4) prices the ~700-900 ceiling.

### MoE prefill: the Session D campaign (494.8 -> 708.9 tok/s @2k — long context + the final ports)

Session D executed that queue. The wide-attention kernel and the k=2048
mma ports shipped; the routed-up port was honestly not attempted
(time-boxed — see below):

| rung | 2k | 8k | 16k | 49k | 96k feed |
|---|---|---|---|---|---|
| Session C | 495.4 | 411.8 | 341.3 | 202.2 | 125.4 |
| **+ MM_PFW wide attention (Session D)** | 516.6 | 469.8 | 419.7 | 295.2 | 203.4 |
| **+ MM_PFW + MM_PFK mma ports (Session D, shipped)** | **708.9** | **623.6** | **544.4** | **348.2** | **227.6** |

(**+43% @2k this session; 3.48x cumulative** over the 203.9 pre-campaign
stock; the 96k full feed +82% over Session C. TTFT on a 1,216-token fresh
generation: **2.27 s cold** (2.27/2.31 s x2) / **0.74 s** with the prefix
cached — was ~3.2 s after Session C, ~15.5 s before the campaign.)

- **`spkqw4` — the row-grouped wide PF attention (the 96k lever,
  `MM_PFW=1`).** The stock `spkq256s` put one CTA per (seat, head, split)
  — every CTA streamed its whole split slice from L2 as byte loads, which
  is why attention was 76% of the 96k chunk. The rewrite ROW-GROUPS the
  work: grid (P/4, 2S), each CTA covers 32 rows (4 seats x 8 q-heads of a
  kv-group), warp w owns 4 rows, and the causal boundary stays
  warp-uniform. K/V tiles are staged to **33,792 B of DYNAMIC shared
  memory** once per tile (cooperative uint4) and consumed by all warps —
  the KV stream drops from once-per-seat to once-per-4-seats, and per
  position each row pays ONE 8-byte LDS per tensor plus ONE shared dequant
  (identical fp op order to stock), reused across the warp's rows. Each
  (row, split) runs ONE online-softmax chain and writes its single real
  partial into an **NP=S layout** — `spkc256` then merges exactly the S
  real slots per row instead of 8S (4x less partial traffic). POC
  (`mm_l4_poc.py`): **2.01x on the attention pair @96k** (151.5 -> 75.4
  ms/layer-pair), 2.03x @49k, 1.97x @8k, 1.82x @2k; the RW=8 ILP variant
  ties at long context and loses at 2k — **RW=4 shipped**. Numerics: vs
  stock per-layer relerr **1.998e-07**, 216,165/1,048,576 words
  bit-identical, det x2, sentinel 0 — Tier-2 by reassociation WITHIN
  splits only (split boundaries pinned). The dyn-smem-in-graph unlock is
  Session A's QMD patch, now proven in production graphs (the
  exec-snapshots-QMD-at-record-time law, [DEXT_LAWS.md](DEXT_LAWS.md) PE1).
- **`pgmq8k2` — the k=2048 trunk mma ports (`MM_PFK=1`).** The Session-C
  `pgmq8m32` template (two `m16n8k16` fragments sharing every staged W
  tile) at KDIM=2048 for the seat-loop classes `qkv`/`z` (GDN) and
  `q`/`k`/`v` (attention) — ROWS compile-time per class (8192/4096/512),
  1D folded MxN grid. Isolated POC (`mm_p2_poc_d.py`):
  **4.58x/5.15x/4.75x/4.23x/3.73x** (qkv/z/q/k/v), relerr
  2.79-2.99e-4 — the same Tier-2 class as Session C's F ~1.5e-4 bank. In
  graph the 2k chunk went 495 -> 361 ms (PC3's contention tax again: the
  isolated 4.6-5.2x compressed to ~1.37x on the trunk share).
- **The honest miss: the routed-up mma port.** NOT attempted this session
  — time-boxed. The grouped `gxm_up*` kernels consume item descriptors
  `(e|rs|m0)`, and an mma port needs a gathered-row-list design (the M
  dimension is per-expert ragged), priced at ~half a day. Post-D it is
  **54% of the 2k residual — the top post-D item**; the ~900 @2k ceiling
  needs it first.

**The Tier-2 battery (both moves are numerics changes):**

- **The CE gate, both arms and combined.** PFW alone: dCE **+0.202%**
  (SEM ~0.101 both arms), top-1 agreement **253/256**. Combined
  PFW+PFK: dCE **−0.369% — improved**, within SEM, top-1 **248/256**
  (247/256 excluding the last seat). The compounding note
  ([DEXT_LAWS.md](DEXT_LAWS.md) PE4): two individually-marginal Tier-2
  drifts STACK — each was within SEM alone, and the combined top-1 moved
  further (253 -> 248); dCE-within-SEM must always ship with the top-1
  co-metric, and stacked numerics moves need a fresh combined gate, not
  the union of the individual gates.
- **The F bank**: hA relerr after one chunk 0.0387 (PFW) / 0.0727
  (combined) — the expected reassociation-class drift; determinism x2.
- **spec == T1 through the daemon**: the 60-prompt bank x2 — **match on
  both arms** (r0 t1->spec n=70, r1 spec->t1 n=186).
- **pcache CACHE_HIT under the new namespace** (MM_PFW/MM_PFK are
  config_fp keys): 8,192/8,192 cached tokens, second arm 1.54 s.
- **Decode classes untouched** (both moves are prefill-only): quote 45.8 /
  prose 43.2 tok/s through the daemon; MTP-mode 42.6-44.6 tok/s across
  classes through the soak.
- **Kill-switches restore stock verbatim**: the battery's stock arms
  (every MM_* knob off) reproduced the Session-C ladder within noise
  (2k: 495.4 vs 494.8) — the knobs are byte-identical restores, and both
  are config_fp keys (one cold pcache rebuild on the first boot after the
  flip, announced in the env file).
- **121-round / 909 s soak, zero engine faults** — every round "ok"; the
  harness's summary fault flag read inverted (True with all rounds green),
  adjudicated a reporting artifact.

**Post-D residual (the honest accounting, from the fresh 2k-chunk
profile):**

| family | share of the 2k chunk (361 ms) | share of the 96k chunk |
|---|---|---|
| routed experts (the un-ported up+down) | **54%** | dominant with trunk |
| shared expert + router | 17% | — |
| GDN scan | 12% | — |
| trunk remains | 12% | — |
| attention | 5% | **19%** (was 76%) |

The long-ctx wall moved: attention went from 76% to 19% of the 96k chunk,
and the binding constraint at 2k is now the **routed-expert up-projection
mma port** — the honest miss above. The measured ~900 @2k ceiling needs
that port first; beyond it, the 2k residual is shared-router + scan +
trunk-remains, each a smaller family than the routed share was.

### MoE prefill: the Session E campaign — the finale (708.9 -> 982.0 tok/s @2k, the ~900 ceiling crossed)

Session E executed the post-D queue to the end of the campaign. Both
gathered-row-list expert mma ports shipped, the split-row scan shipped,
the shared-expert mma pair was measured — numerically clean in isolation
— and **held out honestly** when the end-to-end CE gate failed
unexplained; the router M-batch was measured, missed its bit-exact
contract, and was dropped for a 2.5 ms pool. Every number below is
harness-level, min-of-9, on independent boots (see the battery note at
the end for the daemon-level status):

| rung | 2k | 8k | 16k | 49k | 96k feed |
|---|---|---|---|---|---|
| Session D (published) | 708.1 | 625.4 | 545.5 | 354.8 | 230.4 |
| **+ MM_PFU gathered-row gate+up mma (item 1)** | 831.7 | 721.7 | 615.8 | 382.3 | 241.8 |
| **+ MM_PFU + MM_PFD + MM_PFS (Session E, shipped)** | **982.0** | **852.9** | **703.8** | **415.9** | **254.5** |

(**+38.7% @2k this session; 4.82x cumulative** over the 203.9
pre-campaign stock. The ~900 @2k ceiling priced at the end of Session D
was crossed. The shipped stack: MM_PFG/PFM/PF64/PFT/PFW/PFK/PFU/PFD/PFS
on, MM_PFR off.)

- **`gxu_gm` — the gathered-row-list routed gate+up mma (item 1,
  `MM_PFU=1`).** The design Session D priced at ~half a day: the M
  dimension is per-expert ragged, so the port makes M **the expert's 512
  W-rows as dense 64-row mma tiles** and gathers the ragged N side — the
  bin's activations walked via `plist` into 16-token tiles, pad rows
  zero-filled (deterministic by construction). A = dequantized IQ3_S
  gate/up weights in smem `[Wrow][k]`; B = the gathered activations in
  smem `[token][k]` — the proven `pgmq8k2` fragment-load code with the
  operand roles swapped. Epilogue scatters `silu(g)*u` to
  `ys[pair*512+row]`, the combine's expected layout untouched. Decode
  volume identical to the stock kernel (multi-tile bins re-decode W per
  tile — the same waste class as stock). Two bugs were found and killed
  with an indicator-x k-sweep (`mm_e1_dbg.py`): both lived in the odd
  128-k half of every IQ3_S block (`kc&127` -> `kc&255` in the q-byte and
  sign bases) — the sweep's `-1` control (all k bad) against per-k arms
  localized them in one run. Gates: relerr **4.13e-4**, maxabs 1.26e-3,
  det x2, sentinel 0; isolated 1.43x; **in-chunk −49.1 ms** (implied
  per-kernel ×1.86 on the up family).
- **`gxd_gm` — the routed-down twin (item 1b, a data-driven addition,
  `MM_PFD=1`).** The post-PFU family table (`mm_e2_poc.json`) showed the
  dn fold family (64.5 ms/chunk) was now the same magnitude as the up
  win — so the same gathered-row-list design ran at K=512, M=2048 dn
  W-rows (16 × 128-row tiles), epilogue writing raw fp16 accumulators
  straight into the combine's `parts[pair*2048+row]` contract. Shipping
  it required nailing the **IQ4_NL nibble-split-by-16 decode law**
  (see [DEXT_LAWS.md](DEXT_LAWS.md) PG3): within each 32-k group, k 0..15
  are the LOW nibbles of bytes 0..15 and k 16..31 the HIGH nibbles of the
  SAME bytes — not sequential byte pairs. With the decode verbatim from
  the stock kernel, relerr **3.84e-4**, det x2, sentinel 0; isolated
  **2.20-2.23x**; **in-chunk −35.0 ms**. The 3 Q6_K layers keep the
  stock path (a different weight layout, not worth a third lane).
- **`k2s36h_{256,64}` + `k2nz36` — the split-row GDN scan (item 3,
  `MM_PFS=1`).** The stock scan is one CTA per head — 32 CTAs on an
  82-SM part, ~60% of the GPU idle, and the t-chain is serial so rows
  are the only legal parallel dimension short of the full WY-C32
  reformulation. The port splits the 128 v-rows across 2 CTAs/head (64
  CTAs, 8 rows/warp) with the per-row math VERBATIM stock — **S is
  bit-exact**; the only numerics change is the output-norm sum regroup
  (`yss = half0 + half1` in fixed order instead of the serial 8-warp
  sum), Tier-2 reassociation of the F-bank/rebase16 class. The
  gated-RMSNorm apply moves to a cross-CTA epilogue (`k2nz36`, same
  formula and op order). Isolated 1.45x, **in-chunk −7.6 ms**; the CE
  bisect arm clean (0.8899, top-1 249/256).
- **The shared-expert mma pair — measured, clean, HELD OUT (item 2,
  `MM_PFR=0`).** `shgm512`/`sdm2048` port the `pgmq8k2` Q8_0 mma
  template onto the two biggest shared-router members (shgu32 15.6 +
  shdn32 13.4 ms/chunk in the family table). Per-layer eager gates on
  **all 40 layers: max relerr 5.1e-4, both deterministic x2**
  (`mm_sh_dbg.json` — zero bad layers); isolated **2.80x / 3.35x**;
  in-chunk **−15.1 ms**. And the end-to-end CE gate fails
  **catastrophically: mean CE 8.47, top-1 5/256** (`mm_ce_bisect.json`).
  The control matrix could not close it: both arms deterministic, the
  pair's own outputs read identical between arms while `hA`/`hnb`/`gates`
  diverge downstream (0.16/0.21/0.11 relerr in the divergence matrix,
  `mm_sh_layer.json`); with no mechanism identified, the pair was **held
  out of the ship** — kernels + full evidence committed, the **top known
  2k lever at −15.1 ms/chunk**, re-open question for the next session
  (the dump-path question is the open lead). This is the discipline the
  CE gate exists to enforce: a kernel that is clean at every measured
  layer can still be wrong at the system level, and the gate wins the
  argument.
- **The router M-batch — measured, contract missed, dropped honestly.**
  `rt8e_m{2,4}` batch 2-4 seats per CTA (the 2 MB router weights read
  once per group instead of per seat). Expert **ids came out bit-exact**
  — but gates/sg ULP-drift, so the GOLD-ROUTER bit-exact contract
  (never mma, never reordered) was not met, and the pool is only
  **2.5 ms/chunk**: dropped, kernels + evidence committed
  (`mm_e2_poc.json`).
- **THE ONE-SYMBOL-PER-CUBIN LOADER LAW** (the session's law-grade
  discovery, ~6 boot cycles to isolate — see
  [DEXT_LAWS.md](DEXT_LAWS.md) PG1): a 2-symbol cubin makes the loader
  mispick program metadata (regs/smem of the FIRST symbol) -> wrong QMD
  -> Out-Of-Range-Register warp faults masquerading as flaky sequencing
  faults. All Session-E kernels are built as per-symbol cubins
  (`mm_build_e.zsh`); the pre-split combined sources
  (`MM_E_shm.cu`, `MM_E_k2sh.cu`) are kept for the record.

**The Session E battery (all three shipped moves combined,
`mm_bat_e2.json`):**

- **CE gate: dCE −0.695% — IMPROVED** (stock 0.8985 vs new 0.8922, SEM
  ~0.100 both arms), **top-1 agreement 249/256**. The per-move bisect
  (`mm_ce_bisect.json`): PFD 0.8925 / top-1 250, PFS 0.8899 / 249 — both
  clean alone and the shipped combination composes clean (the PE4
  compounding discipline: fresh combined gate, never the union).
- **F bank**: hA relerr 7.40e-2 after one chunk — the Session-D class
  (7.27e-2); determinism x2; zero spill on every shipped kernel
  (build-time audit); sentinel 0.
- **Decode untouched by construction** — all Session-E kernels are
  prefill-graph members only; the decode classes carry their
  Session-D numbers.
- **Kill-switches**: every knob off restores the stock sequence
  byte-identically; the battery's stock arm reproduced the published
  Session-D ladder (2k: 708.1 vs 708.9) within noise.
- **pcache namespace**: MM_PFU/MM_PFD/MM_PFS are config_fp keys — the
  first daemon boot after the flip takes ONE cold MoE pcache rebuild
  (announced in the env file).

**THE HONEST BLOCKER — the daemon-level battery is pending a physical
cold-cycle.** At session close the machine entered the spontaneous-reset
regime under GPU load (~10 resets over ~2.5 h, cadence tightening to
~5 min; idle + /health fine between; launchd self-healed every time) —
the R7-class silent reset, this time dock/GPU-class, and every
battery-length launch reset the box before completing. What IS verified:
the daemon **boots clean on the full E env** — daemon_attached,
heartbeats, and /health green, observed across 4+ reset-heal cycles.
What is STAGED, not run: bank60 ×2 through the daemon, pcache CACHE_HIT
under the new config_fp, decode-class spot checks, and the 15-min soak —
`engine/mm/mm_daemon_e.py` runs the full battery and is the first thing
to execute after the cold-cycle. All ladder/CE/F gates above ran at the
harness level on independent boots. (Also banked from the session: the
first real request on the new config_fp takes the announced one-cold
pcache rebuild.)

**Post-E residual (the honest ceiling path):** the 2k chunk is now
260.7 ms. The named levers, in order: the **held-out PFR pair
(−15.1 ms)** once its CE divergence is explained; a **full WY-C32
chunked-scan reformulation (~15-20 ms** — the split-row port took the
cheap rows; the t-chain itself needs the dense engine's WY reformulation)
plus the trunk remains; and at long context the wall is now
**trunk + attention together** (attention 19% of the 96k chunk after
Session D, trunk co-dominant) — further long-ctx gains are
trunk-port and attention-depth work, not expert work. The routed-expert
family, the campaign's starting 44.6%, is spent down to parity with the
trunk: the campaign is closed at **4.82x**.





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
  prose (break-even needs E[m] >= 1.17), so a deeper drafter is the unlock;
  the drafter-quality program then honestly falsified the EAGLE-3 TTT
  retraining recipe on these weights (verdict section above — the shipped
  drafter is near this method's ceiling on representative prose), so the
  remaining route is a different drafter class, not more of this recipe;
  the MoE's long-ctx (96k) MTP
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
