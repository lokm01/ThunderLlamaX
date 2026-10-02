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
| Multi-model + MoE | MM_PLAN + MM_P0..MM_P10_results | the Qwen3.6-35B-A3B port (four new kernel classes), the R3/L7 hardening era, multi-model serving, the first-party MTP K=4 chain, the honest fusion falsification |
| The third model + the drafter program opens | DEPLOY_OBLITERATED (in docs/) + TLX_ATT_RB_results + TLX_P0_LEVER_RANKING | the abliterated dense variant as a first-class registry model (offline pack pipeline), the row-batched attention verdict (headroom already shipped), Phase 0's calibrated simulator + lever ranking |
| The drafter program closes | TLX_P1_RESULTS + TLX_P2_ANCHORS + TLX_P2B3_RUNBOOK + TLX_P2B3_VERDICT | Stage A falsified (greedy-vs-corpus target mismatch), the anchor-scale canary battery reframes the bar (the SHIPPED drafter at 0.979 k2 on representative fresh prose vs 0.549 on the hard anchor), v3's clean-slate falsification of the recipe itself; instruments + the opt-in battery pack kept, ~$59 total cloud |
| MoE prefill +91% | the engine/mm Session A+B instrument set (no journal .md was committed on the rig; the record is the campaign section in [../PERFORMANCE.md](../PERFORMANCE.md) + the PB laws in [../DEXT_LAWS.md](../DEXT_LAWS.md)) | 203 -> 384 tok/s @2k bit-exact: Session A's G0 truncated-graph bisect falsified the five-analyst plan's 25-30x DRAM-amplification premise (L2 absorbs the re-reads; the stock pair-walk streams ~750 GB/s isolated) and killed the raster swap at 1.03x; Session B shipped `mmsort8` + the register-decode grouped expert GEMMs (the traffic-reduction-only v1 honestly measured 0.86x and was killed) + the seat-loop trunk + shared M-batch + the PF64 tail (GSM8K-class TTFT ~15.5 -> ~5.5 s); the dyn-smem 64/96KB probe PASS banks the structural branch |
| MoE prefill Session C (2.43x cumulative) | the engine/mm Session C instrument set (no journal .md; the record is the Session C section in [../PERFORMANCE.md](../PERFORMANCE.md) + the PC laws in [../DEXT_LAWS.md](../DEXT_LAWS.md)) | 384 -> **494.8 tok/s @2k** (409.7 @8k, 339.8 @16k, 125.4 @96k feed; TTFT 3.18 s): the `pgmq8m32` trunk mma M-GEMM (the dense M=32 template on Q8_0 — the campaign's one Tier-2 move, POC 8.18x/7.40x isolated vs ~2x in-graph) + the bit-exact `gxm_dnf` act-restaging fold (1.32-1.36x); the routed-up fp16 shortcut honestly SKIPPED (27x the error class); the G3-after-G2 divergence ADJUDICATED as a probe-harness double-append bug; the NEW per-seat next-token cross-entropy quality instrument (dCE +0.046% within SEM on 6.7% router-slot drift); 104-round soak zero faults |
| Condensed ladders | CAMPAIGN.md, PREFILL.md, PERFLOG.md | the campaign summaries (kept here, not in docs/) |
| Earlier lineage | MTP_PLAN, MTP_V3_NOTES, K4BEAM, SCAN_FUSION, CTASM_INVESTIGATION, p1c/p1d/p1e_findings | the tinygrad-stack era before the engine (`lineage/` in the repo root) |

## Index by file

- **CAMPAIGN.md** — the condensed speed ladder 4.17 -> 75.56 tok/s @100k, and the full refuted-approach tables
- **PREFILL.md** — the P1-P18 + R2b-R2d + P8 + T2 prefill campaigns in detail
- **PERFLOG.md** — the running performance log across the whole program
- **MM_PLAN.md** — the multi-model campaign plan: the verified Qwen3.6-35B-A3B model spec, the architecture decisions (router/gather-GEMV/combine/split-KV), the Tier-1 contract for MoE, the P0-P10 sequencing
- **MM_P0_results.txt** — the deciders week: tensor map, bit-exact dequant vs llama.cpp, the name law + single-arg cap, the queue-replay fault open
- **MM_P1_results.txt** — the queue-replay root cause (name law + self-deadlock + the pipelining/wait-each law), the repacker, production v0 kernels
- **MM_P2_results.txt + MM_P2_kmix_manifest.md** — the MoE layer is real: gx/mx families, trunk regen (gen_m36), converter laws, both repack tiers
- **MM_P34_results.txt** — the train is real: full 40-layer T=1 green vs the fp32 anchor, the 20-prompt Tier-1 seed bank, small-shape latency law
- **MM_P56_results.txt** — the spec path is real: spec==T1, quote 118.6 tok/s in-harness, the ctx ladder + LONG-HORIZON law, the 60-prompt bank
- **MM_P7_results.txt** — the host fold (0.2 ms/cycle), split-KV ladder, chunk-256 prefill 181-204 tok/s, 96k unblocked; the silent-no-launch + full-sync-copy + GPU-EXIT-extended laws
- **MM_P8S_results.txt** — the serve-conformance bridge: bank60 60/60 THROUGH the daemon, pcache CACHE_HIT exact, soaks, the first MTP measurement (0.875)
- **MM_P9_results.txt** — the MTP K=4 chain (spec==T1 60/60), the KAPool slab-coherence + narrow-load + Q3_K hmask laws, the launch-serialization measurement
- **MM_P10_results.txt** — MTP wired into serving (prose 19.1 -> 40.1 through the API), the PF-only cur fix, the pairwise fusion honestly falsified + the marginal-launch law correction, the G3 first-contact adjudication
- **MM_P9E_results.txt** — the first output-QUALITY battery (GSM8K 93/95%, PPL baselines, needle) + the F1 first-token-loss fix and the F2/F3 exhaustion findings (closed next by the P10 kernargs pool; eval/ in the repo root carries the harnesses)
- **TLX_ATT_RB_results.txt** — the row-batched attention verdict: the dense probe kernels have been row-batched (one KV pass serves all T rows and all 6 q-heads of a group) since W3 — the priced "win" was already banked in every shipped number; mission-rule NO BUILD, with the T=3 bisect decomposition + the row-scaling bench
- **TLX_P0_LEVER_RANKING.md** — drafter Phase 0's verdict: the G0 sim calibration (chain_sim vs engine traces), the G1 battery (exposure bias is the biggest measured lever: true-conditioning recovers deep conditionals to 0.62-0.72), the pilot's negative transfer lesson (short-ctx cloud training REGRESSES 100k prose), and the committed Phase-1 scope (train-first, corrected data mix, Stage-B engine-trace adaptation before any ship decision). The program plan itself lives outside the repo (driver-seat workspace, `~/.zcode/workspace/default/TLX_DRAFTER_PLAN.md`); its instruments are `engine/chain_sim.py` + `engine/ttt/`, and the third-model deploy record is [../DEPLOY_OBLITERATED.md](../DEPLOY_OBLITERATED.md)
- **TLX_P1_RESULTS.md** — Phase 1's full run (rented H100, $41.91): the corrected 26.5M-token 8-class corpus mix, Stage A FALSIFIED at every checkpoint (the measured root cause: greedy-vs-corpus agreement 0.625-0.645 — human text trains "plausible continuations", not this model's greedy stream), Stage B's anchor-memorization decomposition (the 1.43/1.61 G2 crossing was a1=1.0 replay of 51 evaluation anchors), and the priced anchor-scale unlock
- **TLX_P2_ANCHORS.md** — Phase 2's anchor-scale wave: the 36-session fresh-novel anchor dump (8,829 anchors at 64k-99.4k serve positions), THE CANARY REFRAME (the shipped pack scores 0.979 k2 / 1.197 k4 on held-out fresh prose vs 0.549 on the old r8 anchor — the hard-anchor crisis was partly an artifact), Stage-B v2 falsified for prose a third time (generalizes cleanly, still trades prose down for gsm8k +0.49 up), plus the proxy/mirror/a1-gap/base-availability laws
- **TLX_P2B3_RUNBOOK.md** — the v3 execution spec as staged during the vast.ai platform incident: the pristine first-party bf16 init verification (the offset-buggy fetch vs weights_v2), the complete bf16/RTN baseline rows, and THE QUANT-PATH LAW (the shipped pack is not bitwise RTN; canary insensitive, every cross class sensitive)
- **TLX_P2B3_VERDICT.md** — the program's final verdict: v3 ran to completion (LR 2-3e-6 arms, canary-CE selection) and the packed canary k2 lands BELOW its own no-training init at every point — the recipe is falsified with init/LR/data-scale/selection all exonerated; the current pack STAYS SHIPPED, the canary reframe stands, the +0.22-0.49 gsm8k profile is preserved as the opt-in battery pack, and the 24GB pinned-CPU streaming + expandable_segments adaptations are recorded
- **engine/mm/ (Sessions A+B, no journal .md)** — the MoE prefill campaign's instruments and gate records, published from the pinned rig commits: `mm_pf_bisect.py` (the G0 truncated-graph family bisect that falsified the amplification-wall premise), `mm_l1_poc.py` + `MM_A_gvsl`/`gv8k*` (the L1 raster-vs-seat-loop discriminator), `mm_probes.py` + `MM_A_cp4k`/`dsmemp`/`bwread` (the in-graph expert-load histogram, the dyn-smem 64/96KB PASS, the DRAM re-confirmation), `mm_a_graph.py`/`mm_build_{a,b}.zsh`/`mm_a_build_manifest.txt` (the instrument graph+build rig), and Session B's `mm_g2_l2.py`+`mm_g2_l2.json` (the grouped per-pair bit-exact + speedup gates), `mm_f1b_b.py`+`mm_f1b_b.json` (the F1b full-chunk 0/524,288 gate + the ladder), `mm_dbg_b.py`/`mm_dbg2_b.py`/`mm_dbg3_b.py` (the per-class bisect that caught the dropped-gvf32ab wiring), `mm_wire_b.py` + the `MM_B_*.cu` kernels (mmsort8 / gxm register-decode / gvs32 seat-loop / shgu+shdn) + `MM_P7_lib.py` (`build_seq7`, the gated wiring). The narrative lives in [../PERFORMANCE.md](../PERFORMANCE.md); the laws in [../DEXT_LAWS.md](../DEXT_LAWS.md) PB1-PB8
- **engine/mm/ (Session C, no journal .md)** — the Session-C set, published from pinned rig commits c6e4292..784ee94: `MM_C_pgmq8m32.cu` (the trunk mma M-GEMM kernel) + `MM_C_gxm_dnf.cu` (the act-restaging fold), `mm_l1b_poc.py`+`mm_l1b_poc_c.json` (the isolated POC gates: 8.18x/7.40x + the fold 1.32-1.36x bit-exact), `mm_f1b_c.py`+`mm_f1b_c.json` (the Tier-2 F bank + the ladder 203.9 -> 494.8), `mm_pplc.py`+`mm_pplc.json` (THE per-seat next-token cross-entropy instrument), `mm_g3c.py`..`mm_g3c4.py` (the G3 double-append adjudication rounds) + `mm_tieprobe.py`+`mm_tieprobe.json` (the transient class probe), `mm_daemon_c.py`+`mm_daemon_c.json`/`mm_coldbank.sh`+`mm_coldbank.json`/`mm_soak_c.py`+`mm_soak_c.json` (the daemon battery, the cold-boot bank60, the 104-round soak), `mm_pf_bisect_c.py`+`mm_pf_attr_c.json` (the post-C residual family bisect at five context lengths), `mm_wire_c.py` + `mm_build_c.zsh` (the wiring patch + the cubin build), and the `MM_PFT` env knob in ops/env.canonical.d. Laws: [../DEXT_LAWS.md](../DEXT_LAWS.md) PC1-PC4
- **R8_DECODE.md** — K=9/K=10 ships (75.56 tok/s; 75.81 re-validated through the W5 fixed stack), the bimodal match law, prose-class honest numbers
- **R6_BATCH.md** — the B=2 batch campaign: rungs, laws (compact-partial slice, R6 VRAM, BT>RM, graph-class budget, kernargs-slab), Phase-3 serving integration + honest tables
- **FIX_CAMPAIGN.md** — the five-wave review-fix record + the W5 live validation + the open-findings ledger (#1-9)
- **T2_P8W4.md** — the Tier-2 W4A8 prefill ship + the honest 750-close
- **R7_DECIDERS.md / R7A_DECODE.md / R7B_DECIDERS.md** — the E1 microbench, SASS audits, K=8 + draft-skip, warp-spec falsification
- **R1_PROMPTCACHE.md** — the durable prompt cache (LongMemory)
- **R2_RUNGS.md, R3_DECODE.md, R4_DEEPK.md, R5_DEEPK.md** — the prefill rungs and the deep-K K-ladder K=4..7
- **M1A_SERVING.md, M1B_SERVING.md, M1C_STABILITY.md, SERVING_PLAN.md** — the serving layer and its stability laws (M1C_STABILITY.md also carries the R3 live-window ship record: the 51-finding R3 hardening merge, the L1-L5 live fixes, the L7 soak blocker; the canonical daemon env lives in M1C_STABILITY.md + ops/env.canonical.example; W5 = the supervisor mode)
- **P1..P18 (\*.md)** — the eighteen prefill sessions
- **W0_DOSSIER.md, W0_E2_GRAPHBUDGET.md, W0_E4_BANDWIDTH.md** — preflight measurements
- **W1A_ENGINE0.md, W1B_TRUNK.md, W1C_OPT.md** — engine0 construction
- **W2_MTP.md, W2_100K.md, W2B-W2H (\*.md)** — the MTP engine and the 40-crossing
- **MTP_PLAN.md, MTP_V3_NOTES.md, K4BEAM.md, SCAN_FUSION.md, CTASM_INVESTIGATION.md, p1c/p1d/p1e_findings.md** — the tinygrad-stack ancestry
