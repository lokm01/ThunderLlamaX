# CAMPAIGN — the speed ladder to 72 tok/s @100k, and what was refuted

Goal from day one: **>= 40 tok/s greedy, Tier-1 bit-exact, at 100k-token context**
on the TB4 eGPU rig, for a Qwen3.8-27B-class model. Crossed 2026-09-15 (W2H,
K=2 MTP, 40.35). Re-crossed decisively 2026-09-21: **deep-K LOOKUP K=7 at
63.11 tok/s Tier-1-exact** (R5d), with prefill over 500 tok/s @2k in the same
window (R2d). Pushed again 2026-09-22: **K=8 + draft-skip at 71.86-72.02 tok/s**
(R7a/R7b/P8), prefill 530.6 @2k / 328.0 @100k (P8). Closed 2026-09-23: the
**Tier-2 W4A8 prefill ship at 569.2 @2k / 510.1 @8k / 342.0 @100k** (T2), decode
Tier-1 untouched (71.83-72.02) — with the 750 goal honestly closed in the same
round (the IMMA discriminator's x4.58 was a frame artifact; the real fused-shape
class is x1.06). This doc is the condensed
ladder; the full session-by-session logs are in docs/history/ (W0-W2H, P1-P18,
M1A-M1C, R1, R2-R2d, R3-R5, R7/R7A/R7B, T2 + PERFLOG.md), and the tinygrad-stack era
is summarized in lineage/ + PERFLOG.

## Phase 0 — stock floor and attribution (tinygrad stack)

- Stock tinygrad decode, BEAM=1, fp16 KV: 2k 10.17 -> 100k 4.17 tok/s (100k was
  OOM before fp16 KV). Kernel histograms: ~1578 kernels/token, GDN-block swarm
  dominates; the GPU never reaches steady bandwidth between tiny kernels.
- Hard measurement: 12.6 GB weights / 447 GB/s achievable = ~28 ms/pass floor ->
  40 tok/s REQUIRES speculative decode (MTP) + fewer, better kernels.
- Refuted early: layout tricks (KV stacking LOSES: 47.8 GB/s), chunked split-K at
  tensor level (launch tax), requant (IQ3_XXS already fastest supported format),
  element-level fusion attempts (r_544 pair wasn't even the scan).

## Phase A — hand CUDA inside tinygrad (a3 family)

Replaced tinygrad-generated kernels 1:1 via a substitution hook: IQ3_XXS/Q5_K/Q6_K
dequant-GEMVs (warp-per-row, lane-owns-consecutive-bytes, register scale tables)
and the O(L) attention reduces (rowmax/rowsum/pv -> parallel tree reduces).
Ladder @100k: 4.17 -> 10.50 -> **10.95 tok/s** (@2k: 15.38).

## Phase B — tinygrad-stack MTP (mtp_v3, lineage/)

First exact speculative decode: single JIT=1 graph families, explicit-state
dataflow, PROBE_RO (the T>1 probe must not store rejected states), the
emission/accept contract, vocab-sliced draft head. Ladder @100k: 0.12 -> 2.32 ->
3.67 -> 3.75 -> 5.05 -> 5.43 -> 6.18 -> 6.69 -> 6.84 -> 6.87 -> **7.28 tok/s**
(run39). The scheduler swarm (48 GDN blocks x ~25 kernels) is the wall — the
split-graph decode attempt at 100k hit the piece-graph budget wall (W0-E2: every
userspace hypothesis refuted by measurement; GSP-side, opaque).

## Phase C — the engine (engine0/): rewrite the decode loop by hand

Premise (validated by W0/E4): the dext data path sustains **880 GB/s stream /
843 GB/s fp16 GEMV** — 94% of hardware peak. tinygrad's ceiling (~447) was
scheduler/kernel quality, not the driver. So: static buffers, ~470 hand kernels,
graph replay, device-resident control.

| step | tok/s | the lever |
|---|---|---|
| W1A fused GDN block T=1 | (0.61 ms/block) | 10-launch hand block, numerics contract pinned vs stock |
| W1B full T=1 trunk @2k | 20.09 | 452-launch whole-model loop, 60/60 vs stock |
| W1C wide loads + aligned repack + G_CYCLE | 25.59 | THE ALIGNMENT LAW; QMD-chained graphs kill the ~40 ms host floor |
| W2_MTP: engine MTP K=2 @2k | **40.14** | 484-kernel cycles as 4 graphs; per-step GDN state slots; Q4_0 nibble-plane draft |
| W2_100K: split-KV trio + bootstrap | 29.63 @100k | GQA-shared streaming K1; 100k snapshot bootstrap |
| W2B SKV-G smem staging | 32.35 | 291.6 GB/s; discovered 1 CTA/SM occupancy wall |
| W2C G4 fat CTAs + honest timing | 33.30 | 1024-thread CTAs; pipelined-bench lie exposed |
| W2D GEMV polish (half2 cores, fat CTAs) | 34.77 | in-graph truth: ~230 GB/s GEMV class; T-layout refuted |
| W2E int8-KV (PRMT dequant) | 35.56 | biased-u8 + fp16 row scales, Tier-1, -3.1 GB VRAM |
| W2F half2 QK/PV dots | 39.03 | FA2-style fp32-per-tile / half2-mac cadence |
| W2H HMMA m16n8k16 in-graph (name-encode fix) | **40.35** | tensor-core K1; the W2G "broken kernel" was a 256-thread launch |

## The refuted-approach table (each with the measurement that killed it)

| approach | verdict | evidence |
|---|---|---|
| cmdq-ring wrap as the 100k split-capture fault | refuted | ring never wraps (1359/2048 KiB at fault) |
| bigger cmdq ring / fewer mappings / kernargs pool | refuted | W0-E2 elimination ladder |
| cp.async double-buffering | parked | device-faults on this dext |
| G1 smem-staged attention | refuted | redundant loads were L1 hits; staging round-trip loses (110-124 GB/s) |
| carveout override for 2+ CTAs/SM | refuted | 281 vs 291 GB/s — dext is 1 CTA/SM hard |
| T-layout transposed packs (16B lanes) | refuted | ±1% — load width not the wall |
| K=3 with the current draft | refuted | alpha(pos) 0.867 -> 0.608 at depth; 24.16 < 34.77 |
| draft-vocab slice rebuild | refuted | prompt is a 30-distinct-id repetition; slice already 100% truth-covering; alpha is draft-fidelity-bound |
| QMD-unchain across layers | void | attention layers sit at real data-dependency positions |
| draft GEMV half2 ports | refuted | latency/L2-bound; all six neutral |
| scan grid split (k2s) | refuted | +1.7 ms — prologue replication |
| software-pipelined K1 stage | refuted | kernel at 88% of HFMA2 math floor; only tensor cores could win |
| smem diet as the HMMA in-graph fix | refuted | padded-scalar control at 36,864 B passed 60/60 |
| pipelined GB/s benches | refuted as *measurements* | cross-launch racing inflates up to +145%; synced-only |
| fp16-head / int8-head / zero-fp16 head (stack era) | refuted | numerics/quality (degenerate repeats, capture faults) |
| host-pool dchain, DEVCOPY, DTAIL-in-draft | refuted | exact but slower (launch tax) |
| MTP_KV_CHUNK | refuted, NEVER set | numerically wrong at T>1 (the six-hour bug) |

## What actually crossed 40

1. Graphs everywhere (launch tax -> ~1 doorbell/cycle).
2. The alignment law -> aligned weight packs -> 400+ GB/s dequant GEMVs.
3. Bit-identical M=3 batching -> Tier-1 exact MTP at zero numerics risk.
4. Split-KV + GQA-shared reads + fat CTAs + half2 dots + int8-KV + HMMA:
   attention 73 ms/probe (stack era) -> ~21 ms (engine) -> ~13 ms (HMMA).
5. The name-encoded launch-config law (W2H) — the last +1.3 tok/s was pure
   forensics: poison dumps, QMD diffs, and reading the graph builder.

## Phase R — the serving-era campaigns (R1-R5 decode, R2b-R2d prefill, 2026-09-20/21)

After M1 (service) and P1-P18 (prefill), the R-series ran one week of
gate-disciplined rungs on the shipped daemon. Full logs: docs/history/
R1_PROMPTCACHE.md, R2_RUNGS.md, R3_DECODE.md, R4_DEEPK.md, R5_DEEPK.md.

| rung | result | the lever |
|---|---|---|
| R1 durable prompt cache | 100k restore ~6.5 s vs ~13 min FRESH | hash-keyed prefix trie over FED ids (config-fingerprinted); int8-KV + GDN + draft-KV windows on disk; 60/60 across 4 restarts |
| R2/R2b WY-C32 chunk scan | 2k 374.0 -> 401.7 / 100k 247.6 -> 255.8 | chunk-level WY scan solve (3 stacked kernel bugs root-caused via a determinism-bisect DBG ladder: uninit smem upper triangle, staging race, malformed hi-lo MMA passes); NC=2 single-launch scan tier; attnqkv m64 twin (P7E4 quarantine lifted) |
| R3 LOOKUP n-gram drafter | Tier-1 60/60 @K=2, +0.10 ms/cyc | GPU-side n-gram suffix scan replaces the MTP draft on hit cycles (LMIN=8, 83.3% hits, E[m\|hit]=2.000); tok_hist seeding law; D0 smem-LUT decode refuted (1.007x) |
| R4 deep-K selection | K=4 machinery, 46.6 (buggy) | per-cycle K2/deep graph-set selection; M=5/T=5 kernel set; lookup5 (deep-K scan-range law iend=pos-8-K); ROWS=5 attention; the sel-mode Tier-1 bug localized to rows 3-4 |
| R5a exactness fix | **53.65** (K=4) | three stacked gen_m5 M-extension slips fixed (missing row-4 stores, missing a4 accumulate, launch_bounds vs the 1024-thread name-law); localized by the R4_TRACE / R4_DIF / R4_BISECT harness triad (now env-gated in test_w100k.py) |
| R5b/R5c/R5d the K ladder | 56.60 (K=5) -> 58.71 (K=6) -> **63.11 (K=7)** | mechanically generated M=6..8/T=6..8 sets (gen_m6..8.py: launch-bounds-preserved renames + row-store audits); per-K lookup/accept/attention kernels; RP>NW owner bug fix; REC-chain slot law; serve.py seed_hist |
| R2c decode-r7 + M=128 | 2k 494.4 / 8k 451.3 / 100k 314.3 (47.5%) | packed7 shared weight plane (decode/spec GEMVs read the prefill packs; both-live VRAM wall gone, full m64 coverage); 128-row chunks on 2-M-block M-grids; WY-C32 NC=4 scan; the law-2 tail re-commit fix |
| R2d priced scraps | **2k 503.6 / 8k 457.8 / 100k 317.6 (48.0%)** | DBUF ring-depth-4 gdnqg (loads 3 phases early; per-class coin-flip law) + attnqkv g=448 consolidation (745 launches/chunk); pre32x4 gated exact but neutral, banked off |

Deep-K decode ladder @100k (every rung Tier-1 60/60 x2 det + stock 59/59):
deep=off (K2 superset) 40.05-40.24 -> K=4 53.65 -> K=5 56.60 -> K=6 58.71 ->
K=7 63.11 (107.49 ms/cyc, 6.78 tok/cyc; E[m|deep]=7.000 — 96/96 cycles
accept ALL SEVEN proposals; 81.7% hits). Sel-mode beats all-deep at every rung:
per-cycle graph-set selection only pays the T=K+1 probe cost when the drafter hit.

### R-series refuted entries (each with the measurement)

| approach | verdict | evidence |
|---|---|---|
| smem-LUT decode (D0) | refuted | 1.007x, bit-identical — grid gathers are not the wall |
| all-deep mode (always T=K+1 probe) | refuted | loses to sel-mode at every K (K7: 59.03 vs 63.11) |
| 2-consecutive-hits trigger (LOOKUP_TRIG=2) | banked off | guards prose but costs 0.4 tok/s on hit-heavy classes (58.29) |
| w128h attention window | banked negative | bit-identical @pos100224 but NONDET-WRONG at low pos (the P17 ROWS-extension class) |
| m128-GEMM singles/twins | refuted | 255 regs + 452-888 B spill — the P18 nondeterminism law |
| split-plane FFN (nw8) | refuted | bit-identical but 1.153x SLOWER (fused nw4 x/smem reuse beats warp count) |
| ring-4 on fd/out/qkv classes | banked negative | per-class per-shape coin flip (x0.79-0.765); only gdnqg won x1.098 |
| pre32x4 | banked off | gated bit-identical-exact, perf-neutral at 2k-class (503.3 vs 503.6) |
| offline quote-rate as the expected hit rate | refuted as a forecast | in-vivo quote workload: 1.7% hits; the offline 87.9% is the model-quotes-perfectly CEILING |

## What actually crossed 60

1. The n-gram LOOKUP drafter (R3): free proposals from history the model itself
   wrote — hit cycles accept ALL K (alpha 1.0 in-vivo at every depth).
2. Mechanical M-extension with audited row stores (gen_m5..8.py) — the R5a
   postmortem turned "hand-port every kernel per K" into a generator with
   fail-loud audits; K=7 shipped first-build green.
3. Per-cycle graph-set selection: K2 fallback keeps ordinary cycles cheap, deep
   cycles only fire on hits.
4. The shared packed7 plane (R2c): prefill and decode read ONE weight copy —
   full m64 coverage without the both-live VRAM fault.

## Phase R7 + T2 — the deciders, the 70-cross, and the honest 750-close (R7/R7a/R7b/P8/T2, 2026-09-22/23)

One measurement-first day: four decider experiments (no production flips), then
the decode-70 rungs, the warp-spec round, and the P8 prefill/serving session; the
next day the T2 W4A8 round shipped the Tier-2 ffn and closed 750 honestly.
Full logs: docs/history/R7_DECIDERS.md, R7A_DECODE.md, R7B_DECIDERS.md, T2_P8W4.md.

| rung | result | the lever |
|---|---|---|
| R7 deciders (E1/SASS/D3/D4) | measured, nothing shipped | E1 issue-law microbench: the GEMM wall is LATENCY-ORDERING not issue-rate (arms issue at different rates; production mix == pure-load 493 GB/s) -> warp-spec headroom 1.8-1.9x REAL; SASS audit: decode GEMVs read each 16B unit as 3 narrow loads (prefill already 16B except gdnqg-qkv); D3 attribution: the 47 ms deep increment is 69% GEMV-M-extension (ffn_down 37.7% + norms 14.9% + gemv_gdn 11.6%), attention only 22.8% — the "attention-dominated" fingerprint was wrong; D4 histogram: ladder extends past K=9 at <=7 ms/rung, optimum K~12-16 |
| R7a rung-1a (uint4 W-merge) | 63.19 (neutral) | one uint4 load + register extracts on the r7-unit family, bit-identical nz=0 det-x2 — and a LAW: the 3x W-load-instr cut is perf-NEUTRAL (the GEMV critical path is x-loads+ALU); kept as hygiene |
| R7a rung-3 (norms per-row CTAs) | **68.62** | the norms/emb pool was CTA-SERIALIZATION (grid=1 serial-t-loops), not launch count: per-row CTAs (grid 1->8/13->104) with verbatim lane math = bit-identical, -9.75 ms deep cycle, +5.43 tok/s |
| R7a rung-4 (K=8, T=9/M=9 set) | 69.47 | gen_m9.py (all audits pass; aq3k8v9 unroll 5->2 the zero-spill knob), ffn8v9r7/down8nw32v9r7 twins (nw32 64-reg budget held at M=9), lookup9 (iend=pos-17), accept9k/acceptsel9k (REC-CHAIN SLOT LAW held first-try at t=8), spk ROWS=9 set (RMAX=54/RP=64, MAXOWN=2); E[m\|deep]=8.000 94/94, hit decay only -1.7pt |
| R7a draft-skip | **71.51** | on deep cycles (78%) the 2-step MTP draft chain was pure waste — a lookup-only draft graph; emit stream byte-identical (449 emitted, pos_end identical) |
| R7b (warp-spec round) | no prefill ship | the QMD barrier root cause + fork patch (barrier_count=1 was a fork bug — `NV_QMD_BARRIERS=16` unlocks bar.sync 1..15), mbarrier DEAD on this dext, meet-based warp-spec bit-identical but perf-NEGATIVE (x0.82-0.96), X-uint4 widening NEGATIVE (x0.78-0.89; the 2x-more-numerous narrow stream keeps more loads in flight), first full GEMM census: 159 of 256 ms/chunk |
| P8 (packed5 + o-proj fold) | **2k 530.6 / 8k 479.4 / 100k 328.0** | qg packed5 (Q5_K qkv-seg integer repack -> true-16B units, x1.29, +1.89 GB both-live — fits at full 100k state) + o-proj M-grid fold (one g=160 launch replaces 4x m32, x2.06); both bit-identical nz=0 det-x2, kill-switched, pcache-namespaced |
| P8 (the P0 serving fix) | FRESH correctness restored | root cause: the R2c M128 rung broke `_pf_graphs`' ambient-flag inference — M64-context calls submitted M32/M128 graphs reading stale ids; plus the M32 r>0 tail fed at pos 0 (the "8k M32-reassociation floor" was THIS bug — now 60/60 at F 9.824e-04); p0_repro.py harness, all 7 path shapes fresh + pos exact |
| T2 (W4A8 ffn, `PF_W4A8=1`) | **2k 569.2 / 8k 510.1 / 100k 342.0 (51.7%)** | the packed7-reading W4A8 IMMA ffn (`p8_w4ffn7.cu` + act-quant `pfk_q8.cu`): the EXISTING packed7 planes read through a linearized int4 view of the IQ3 codebook (8 linear levels, 1.49% weight-RMS) — ZERO new VRAM, full 64-block coverage, decode Tier-1 untouched (71.83, 60/60 x2 with W4A8 resident); battery green per the P7E7 convention; kill-switch byte-identical; pcache-namespaced |
| T2 (the 750 close) | 750 CLOSED, honestly | the P8 x4.58 was a FRAME ARTIFACT (one-plane M=64/g=272 vs census two-plane M=128/g=1088; apples-to-apples fp16 ~305 us) — true IMMA = x1.06 fused-shape / x1.56 int4-plane (VRAM-dead: <1.9 GB headroom @100k); THE DISCRIMINATOR-FRAME LAW (match plane-count x M-grid between arms) |

Final decode ladder @100k: 63.11 -> 68.62 (norms CTAs) -> 69.47 (K=8) ->
71.51 (draft-skip) -> 71.86 (P8 gate) / 72.02 (R7b-boot best); 7.48 tok/cyc,
alpha 3.242, deep=off 41.78-42.14 intact at every rung.

### R7-series refuted entries (each with the measurement)

| approach | verdict | evidence |
|---|---|---|
| W-load-instruction-count as the GEMV lever | refuted | uint4 merge = 3x fewer W-load instrs, bit-identical, ~neutral at M=3/8 (x-loads+ALU is the critical path) |
| meet-based warp-spec on pf_gemm3 | refuted | bit-identical det-x2 but x0.82-0.96 at every shape — the per-stage full-CTA meet couples producer stores with consumer decode on the critical path |
| mbarrier pipelines | dead on this dext | init/arrive/wait execute, phases NEVER complete (bounded-spin timeouts, checksum 0); parity operand must be a compile-time immediate below sm_90 |
| X-stream uint2 -> uint4 widening | refuted | attnqkvi3 BREAKS (full-output diff); perf x0.78-0.89 on nw8 classes — load-latency structure rewards issue diversity, not width |
| "attention dominates the deep increment" | refuted (attribution) | D3: attention is 22.8% of the 47 ms; ffn_down+gemv_gdn+norms = 64.2% |
| deeper K at the current per-rung cost | parked | D4: at +6.9 ms/rung K=12 reaches only ~76.6; at +12.3 the ladder stops at K=8 — needs the deep-cycle cost work first |
| W4A8 IMMA as the ~750 route | refuted (frame artifact) | the P8 "x4.58" compared one-plane M=64/g=272 IMMA vs the two-plane M=128/g=1088 census launch; apples-to-apples fp16 = ~305 us/unit -> true x1.06 fused (v2) / x1.56 (v1, VRAM-dead) — the shipped take is the honest +7.3% @2k |
| int4 requant of the ffn planes (v1 full coverage) | refuted (VRAM) | 12.6% weight-RMS raw requant + 5.88 GB planes vs <1.9 GB headroom at the full 100k state; partial coverage only — superseded by the packed7-reading v2 |

## What actually crossed 70

1. The norms CTA-serialization law (R7a rung-3): one-CTA serial-t-loop norms cost
   ~Mx their parallel time; per-row CTAs are free and bit-identical. +5.43 tok/s
   for a grid change.
2. K=8 via the audited generator discipline (R5a's postmortem paying off): every
   kernel built first-try-green, REC-CHAIN SLOT LAW included.
3. The draft-skip: skip work the lookup makes redundant — output-lossless by
   construction (amds-verified commits), ~4.1 ms/cycle back.
4. The P0 fix (P8) made the K=8 daemon canonical trustworthy: the "K=8
   daemon-context corruption" era ended with the stale-feed root cause (wrong
   graph set via ambient-flag inference), not K=8's own surfaces.

## Next levers (documented, unbuilt)

Deep-K is bounded by the n-gram hit rate (78.3% gate-class at K=8; arbitrary prose
does not verbatim-continue — R5's honest quote finding), so further K buys ~nothing
without a better drafter (DFlash2-class block drafter = the alpha program) or the
per-rung cost work (hm9's pad rows the first target; D4's K=12-16 optimum needs
<=7 ms/rung). Prefill: the W4A8 IMMA tier is now SHIPPED and honestly priced
(T2: x1.06 fused-shape, +7.3/+6.4/+4.3% at 2k/8k/100k; 51.7% of the 662 reference
@100k) — the 750 arithmetic is closed on every route (frame-artifact correction +
the VRAM wall on the int4-plane variant). What remains measured and structural:
the GEMM pool (159 ms/chunk, ~62%) needs a persistent-CTA megakernel where the
pipeline lives inside one CTA (meet-based warp-spec and X-widening falsified,
W4A8 cashed), plus the attention growth pool (persistent-CTA attention or dynamic
smem >48 KB); decode 72.02 = 84% of the 86 native-stack reference.
