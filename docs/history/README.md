# The lab notebooks — campaign journals

These are the engineering chronicles of ThunderLlamaX, kept exactly as written
on the rig, session by session. They are the source of truth behind every
number in [docs/PERFORMANCE.md](../PERFORMANCE.md): each performance claim in
the user-facing docs traces to a gate recorded in one of these journals. Read
them for the how-we-know, the postmortems, and the long ledger of approaches
that were tried and refuted with measurements.

Style warning: these are working logs, written dense, in the heat of the
campaign. Expect rig-era shorthand, env-knob names, commit hashes, and
"laws" (platform facts paid for with fault-and-reboot cycles — see
[docs/DEXT_LAWS.md](../DEXT_LAWS.md) for the curated compendium).

## The story, in reading order

| era | docs | what happened |
|---|---|---|
| Preflight | W0_DOSSIER, W0_E2_GRAPHBUDGET, W0_E4_BANDWIDTH | measured the driver path before writing any engine (the 880 GB/s finding) |
| The engine is born | W1A_ENGINE0, W1B_TRUNK, W1C_OPT | hand kernels, static buffers, graph replay: 7 -> 25.6 tok/s @2k |
| The 40-goal | W2_MTP, W2_100K, W2B-W2H | MTP K=2, split-KV attention, int8-KV, HMMA: 40.35 tok/s @100k |
| Serving | M1A_SERVING, M1B_SERVING, M1C_STABILITY, SERVING_PLAN | the daemon + OpenAI API + the stability laws |
| Prefill, 17x | P1-P18 | batched chunked prefill: 21.8 -> 373.5 tok/s @2k |
| Decode rungs | R1_PROMPTCACHE, R2_RUNGS, R3_DECODE, R4_DEEPK, R5_DEEPK | prompt cache + the deep-K LOOKUP ladder to 63.11 |
| Deciders | R7_DECIDERS, R7A_DECODE, R7B_DECIDERS | measurement-first day; K=8 + draft-skip: 71.5-72.0 |
| Tier-2 ship | T2_P8W4 (in CAMPAIGN.md's Phase R7+T2 too) | the W4A8 prefill ffn: 569.2/510.1/342.0, 750 honestly closed |
| The 75-cross | R8_DECODE | K=9 + K=10: 75.56 tok/s Tier-1 @100k; the K-ladder stops at ten |
| The B axis | R6_BATCH | B=2 batched decode, per-stream Tier-1, 81.33 harness aggregate + the honest Phase-3 serving tables |
| The audit | FIX_CAMPAIGN | 10 model reviews -> 60-finding ledger -> five fix waves -> live W5 validation (75.81) |
| Condensed ladders | CAMPAIGN.md, PREFILL.md, PERFLOG.md | the campaign summaries (kept here, not in docs/) |
| Earlier lineage | MTP_PLAN, MTP_V3_NOTES, K4BEAM, SCAN_FUSION, CTASM_INVESTIGATION, p1c/p1d/p1e_findings | the tinygrad-stack era before the engine (`lineage/` in the repo root) |

## Index by file

- **CAMPAIGN.md** — the condensed speed ladder 4.17 -> 75.56 tok/s @100k, and the full refuted-approach tables
- **PREFILL.md** — the P1-P18 + R2b-R2d + P8 + T2 prefill campaigns in detail
- **PERFLOG.md** — the running performance log across the whole program
- **R8_DECODE.md** — K=9/K=10 ships (75.56 tok/s; 75.81 re-validated through the W5 fixed stack), the bimodal match law, prose-class honest numbers
- **R6_BATCH.md** — the B=2 batch campaign: rungs, laws (compact-partial slice, R6 VRAM, BT>RM, graph-class budget, kernargs-slab), Phase-3 serving integration + honest tables
- **FIX_CAMPAIGN.md** — the five-wave review-fix record + the W5 live validation + the open-findings ledger (#1-9)
- **T2_P8W4.md** — the Tier-2 W4A8 prefill ship + the honest 750-close
- **R7_DECIDERS.md / R7A_DECODE.md / R7B_DECIDERS.md** — the E1 microbench, SASS audits, K=8 + draft-skip, warp-spec falsification
- **R1_PROMPTCACHE.md** — the durable prompt cache (LongMemory)
- **R2_RUNGS.md, R3_DECODE.md, R4_DEEPK.md, R5_DEEPK.md** — the prefill rungs and the deep-K K-ladder K=4..7
- **M1A_SERVING.md, M1B_SERVING.md, M1C_STABILITY.md, SERVING_PLAN.md** — the serving layer and its stability laws (the canonical daemon env lives in M1C_STABILITY.md + ops/env.canonical.example; W5 = the supervisor mode)
- **P1..P18 (\*.md)** — the eighteen prefill sessions
- **W0_DOSSIER.md, W0_E2_GRAPHBUDGET.md, W0_E4_BANDWIDTH.md** — preflight measurements
- **W1A_ENGINE0.md, W1B_TRUNK.md, W1C_OPT.md** — engine0 construction
- **W2_MTP.md, W2_100K.md, W2B-W2H (\*.md)** — the MTP engine and the 40-crossing
- **MTP_PLAN.md, MTP_V3_NOTES.md, K4BEAM.md, SCAN_FUSION.md, CTASM_INVESTIGATION.md, p1c/p1d/p1e_findings.md** — the tinygrad-stack ancestry
