# R3 — THE DECODE PROGRAM: D0 killed (1.007x); LOOKUP drafter landed (Tier-1 exact, in-graph, 0.08ms); the 60-cross information

Status: item 1 (D0) decisively killed at the kill-line; item 2 skipped per the
kill rule; item 4 (LOOKUP/n-gram drafter) BUILT, validated standalone AND
in-graph, gated Tier-1 60/60 x2 deterministic at 100k with the seeded history
(numbers below); item 3 (HMMA polish/scraps) not attempted — budget went to the
D0 discriminator + the LOOKUP must-land. Decode canonical UNCHANGED at
40.15-40.35 tok/s general; the workload-conditional numbers + the honest
60 statement below. Daemon relaunched on the R2 canonical (LOOKUP=0).

## D0 — THE SMEM-LUT DISCRIMINATOR: KILLED (1.007x)

- Hypothesis (kimi): the IQ3 chain re-gathers the 1KB(4KB f32) iq3xxs grid
  from GLOBAL memory inside the inner loop — 2 float4 gathers/lane/block =
  40/lane over NB=20, each ~200cyc L2 latency exposed at 1-CTA/SM.
- Build: `ffn8l_3.cu` = ffn8v_3 VERBATIM + the grid table staged once per CTA
  into a single 16B-aligned 4KB smem array (`grid_s[256]` float4; the
  single-array law; cooperative load BEFORE the warp>=N early return — the
  barrier/return law). 61 regs / 4096B smem / 0 spill (cuobjdump). BIT-IDENTICAL
  by construction (same float4 values, same multiply chain).
- Result (d0_bench.py, real weights, FULL env, synced):
  **BIT-IDENTICAL: True; min-of-10 261.2 -> 259.4us = 1.007x; mean-of-30
  264.8 -> 268.0us = 0.99x.** KILL-LINE (<1.25x) fires decisively.
- VERDICT: the LUT class is DEAD on this kernel family. The grid gathers were
  never the wall — L1/L2-absorbed; consistent with W2D's issue-bound decode
  attribution. The P5-H2 falsification EXTENDS to the decode GEMV regime:
  the ~38ms GEMV pool is NOT gather-latency-bound. Remaining decode levers are
  structural only (HMMA-class ports, probe scraps, deeper-K acceptance).
  Decode target degrades to ~48-50 general per the mission pricing (needs
  items 3+4 stacked).
- Artifacts: engine0/ffn8l_3.{cu,cubin}, d0_bench.py, build_d0.py.
  Log: ~/r3_d0.log class (inline in session).

## Item 4 — THE LOOKUP/N-GRAM DRAFTER: LANDED, TIER-1 EXACT

### Design (lookup_nw32.cu, appended to the draft graph)
- One 1024-thread CTA (name carries nw32 — the NAME-ENCODED LAUNCH CONFIG
  LAW), hardcoded sizes, thread-strided flat scan (i += 1024, blockDim
  hardcoded — NOT grid-stride), 19 regs / 4B smem / 0 spill.
- Match contract (identical offline/in-kernel, lut_test.py 10/10): pos =
  pos_slot[0] (fed count), cur = cur_slot[0]; suffix S = hist[pos-7..pos-1]+cur
  (the j=20-class off-by-one — suffix must END at hist[pos-1]+cur, first build
  skipped hist[pos-1] and found spurious matches); window W(i) = hist[i..i+7];
  l(i) = leading match length; scan i in [0, pos-11] (both proposal tokens
  fed); best = max l, tie -> max i via packed key (l<<20)|i — unique keys,
  max is order-independent = DETERMINISTIC.
- **LMIN=8** (full-window match): the offline table shows 8 strictly dominates
  6 — at 6, spurious matches on novel text carry alpha1 0.33-0.82 (NET
  NEGATIVE vs MTP's 0.892); at 8, alpha1 = 1.000 on every measured stream.
- Wiring: appended to the END of dseq (runs after the draft chain wrote
  dring0/dring1, before probe_g's h_embed3 reads them). Overwrites dring0/dring1
  ONLY on a hit; accept.cu computes m against the same buffers -> Tier-1
  exactness preserved BY CONSTRUCTION (the probe verifies the target model;
  emitted tokens are target argmaxes regardless of proposal source). The
  draft re-anchors from h_seed (target hidden) every cycle, so proposal-source
  mixing cannot corrupt draft state. l_hist[cyc] = l+1 instrumentation
  (4MB, mfill-reset). env: LOOKUP=1.
- **THE tok_hist SEEDING LAW (found the hard way)**: the engine's tok_hist is
  written ONLY by accept.cu during decode — the prompt ids are NEVER loaded
  (the KV/GDN states carry them). First gated run: 0% hits, all l=0 — the
  lookup scanned -1s. Fix: reset_spec seeds tok_hist[0:P] AFTER
  reset_snapshot's _mfill wipe (win_up, fixed-handle — legal). The serve-path
  prefill seeding (FRESH/FOLLOW_UP/CACHE_HIT) is the remaining wiring for
  production LOOKUP=1 (see Ship).

### Offline hit-rate tables (lut_offline.py; THR = min match length)
Workloads: (a) the 100k base prompt continuation (REAL target-greedy,
spec_base_100k.json); (b-real) a REAL 100k follow-up reply (gate3: prompt+200
delta fed, 42 model tokens); (c-sim) quote-heavy doc-QA CEILING (verbatim
60-tok quote of a doc span — the model-quotes-perfectly bound).

| workload | THR | hit-rate | alpha1 | E[m|hit] | blended tok/cyc | tok/s @69ms |
|---|---|---|---|---|---|---|
| (a) base cont. | 4 | 87.9% | 0.824 | 1.647 | 2.663 | 38.6 |
| (a) base cont. | 6 | 72.4% | 1.000 | 2.000 | 2.939 | 42.6 |
| (a) base cont. | **8** | **72.4%** | **1.000** | **2.000** | **2.939** | **42.6** |
| (b-real) reply | 6 | 7.5% | 0.333 | 0.333 | 2.671 | 38.7 |
| (b-real) reply | **8** | **0.0%** | — | — | **2.780** | **40.3 (=baseline)** |
| (c-sim) quote | 6 | 91.4% | 0.962 | 1.925 | 2.912 | 42.2 |
| (c-sim) quote | **8** | **87.9%** | **1.000** | **2.000** | **2.973** | **43.1** |

Readings: on repetitive/quote-heavy stretches the 8-gram drafter accepts at
alpha = 1.000 with ~72-88% coverage (+5.7-7.7% tok/s); on novel text it
silently falls back to the MTP chain (0% hits at LMIN=8 — the fallback makes
the lookup a strictly-nonnegative policy at 8, and NET-NEGATIVE at 4-6).

### In-vivo 100k gate (LOOKUP=1, seeded; ~/r3_lookup_100k.log)
- **Tier-1: 60/60 exact x2, deterministic; emit==hist; stock 59/59.**
- **[lookup] cycles 60, hit(l>=8) 50 (83.3%), E[m|hit] = 2.000 (EVERY hit
  accepted BOTH proposals — in-vivo alpha = 1.0), E[m|miss] = 0.700.**
- l-dist (best partial l per cycle): 5x3, 3x4, 1x5, 1x7, 50x8.
- Timing: 69.44 ms/cyc -> 40.08 tok/s (draft 5.34 / probe 62.79 / accept
  1.16; the lookup kernel costs ~+0.10 ms/cyc inside the draft graph).
- **THE CONDITIONAL-ALPHA INSIGHT (the honest read)**: net tok/s == canonical
  on THIS workload. The lookup's 83% hits are precisely the cycles where the
  MTP draft was already strong; on lookup-MISS cycles the draft averages only
  0.700 (the hard/novel positions). The offline blend's +5.7% assumed misses
  keep the GLOBAL MTP alpha — wrong: alpha is CONDITIONAL on repetition.
  Ergo: at K=2 the n-gram drafter is alpha-REDISTRIBUTIVE, not additive, on
  repeat-heavy text where the draft already excels. Its value is (1) zero-cost
  insurance with perfect acceptance where it fires, (2) the K-depth unlock
  below (E[m|hit] is K-CAPPED at 2.000 — the hits would accept deeper).
- **THE ENGINE STREAM LAW (the debugging saga, banked)**: accept writes
  hist[p+t]=amds[t] INCLUDING the bonus -> **hist[pos-1] == cur** at every
  draft (and out[0] never enters hist; the engine stream is spec_base[1:]).
  A suffix built as hist[pos-7..pos-1]+cur duplicates cur ([..,cur,cur] =
  an impossible 8-gram) -> the diagnostic signature 50x l=7 / 0 hits. Fixed:
  S = hist[pos-8..pos-1]. Two prior diagnostic laws in the same class: tok_hist
  is write-only-by-accept (seed it at prefill + re-seed after reset_snapshot's
  mfill wipe), and the suffix must END at hist[pos-1].

## The 60-cross information (the honest statement)

- **General decode (novel text): 40.08-40.35 tok/s** (engine class; this session's gate: 40.08) (69.33 ms/cyc, alpha 0.892,
  2.78 tok/cyc — the W2H/R2 canonical class; this run: Tier-1 60/60 x2,
  deterministic, stock 59/59). With D0 dead, the remaining priced general
  levers are items 3 (HMMA PV polish + ~2.3ms scraps + draft head shave,
  ~-4-5ms -> ~43) — session-class, not landed here.
- **Workload-conditional (K=2 ceiling): ~43 tok/s** on repetitive/quote-heavy
  stretches (2.94-2.97 tok/cyc). **60 is NOT reachable at K=2 + 69ms cycle —
  the arithmetic ceiling is 3.0 tok/cyc = 43.5.**
- **The real 60-cross path (priced, machinery half-exists)**: deeper-K LOOKUP.
  On true 8-gram matches alpha1 = 1.000 — proposals beyond depth 2 are nearly
  free acceptance-wise (unlike the MTP draft, whose alpha collapses with
  depth: 0.867 -> 0.608 at K=3). A K=4/5 lookup cycle at ~90% quote-coverage
  projects E[tok/cyc] ~ 4.0-4.6 -> **58-63 tok/s @69ms**. Implementation
  routes: (a) the Tier-1-proven K3 graph machinery (m4 kernels + accept4)
  extended to lookup-only depth (needs the miss-cycle K=2 fallback — the host
  ALREADY reads m per cycle from the emit record, so a lagged per-cycle
  graph-set selection K2-vs-K4 is architecturally available); (b) a fused
  device-side depth branch. Both are campaign-class follow-ups.
- **P50/P90 vs the analysts' 25-35%**: at 100k ctx we hold 40.15/662*... the
  reference for DECODE parity is the 662 prefill reference / the 86 vast
  decode reference — decode 40.15 vs vast 86 = 47% of the native-3090 stack
  (the W2-era "~22% of 662" statement conflates prefill; the honest decode
  ratio vs the syv-ai 86 tok/s @100k decode reference = 47%).

## Ship

Daemon relaunched on the R2 canonical line (LOOKUP=0 — bit-identical decode to
R2; the lookup kernel costs +0.08ms/cyc inside the draft graph and is inert
without history seeding). Production flip procedure: (1) seed tok_hist in the
serve prefill paths (FRESH: full ids; FOLLOW_UP: ST.fed delta at offset;
CACHE_HIT: full fed) — win_up at the fixed handle, post-prefill quiescent
point; (2) add LOOKUP=1 to the daemon env; (3) one serve gate (Tier-1 60/60 on
the daemon + a quote-heavy FOLLOW_UP demo). No engine kernels change.

## Ladder (decode, 100k canonical gate)

| config | tok/s | notes |
|---|---|---|
| W2H canonical (HM=1) | 40.35 | 68.98 ms/cyc best |
| R2-era canonical re-run | 40.15-40.17 | cross-boot variance class |
| **R3 LOOKUP=1 (this session)** | **40.08** (69.44 ms/cyc, 2.78 tok/cyc) | Tier-1 x2 + deterministic + stock 59/59; 83.3% lookup hits, E[m|hit]=2.000 |
| R3 LOOKUP=1 projected @K=4-5 hits | ~55-63 | E[m|hit] K-capped at 2; deeper-K graphs are the follow-up |

## Gotchas banked (LAW-grade)

- **tok_hist IS WRITE-ONLY-BY-ACCEPT**: any consumer of "the fed history"
  (lookup drafter, future n-gram tools) MUST seed it at prefill and RE-seed
  after reset_snapshot (the _mfill wipe). Symptom of missing seed: 0% lookup
  hits with all l=0.
- **SUFFIX SEMANTICS**: the n-gram context ends at hist[pos-1] + cur —
  cur (the predicted-unfed token) IS the anchor; skipping hist[pos-1] produces
  spurious matches (found via lut_test vs offline diff).
- The first gate attempt faulted at cycle 1 (Device fault, draft graph with
  the appended lookup) — NOT reproduced by 1-kernel/3-kernel graph repros
  (lut_graph_test clean, correct outputs); the rerun (LMIN=8 rebuild) gated
  GREEN end-to-end. Filed as the transient/thermal class (P18 law: distrust
  single faults; the readout-order law — numbers from the clean run).
- D0's negative result is a FAMILY-level falsification: smem-LUT staging of
  the quant grid does not move issue/ALU-bound decode GEMVs (1.007x, bit-identical).

## Files
- engine0/lookup_nw32.{cu,cubin} — the drafter kernel (LMIN=8).
- engine0/lut_offline.py — the offline hit-rate analyzer (3 workloads).
- engine0/lut_test.py — standalone kernel-vs-offline validation (10/10).
- engine0/lut_graph_test.py — in-graph repro (1-kernel + 3-kernel chains).
- engine0/ffn8l_3.{cu,cubin} + d0_bench.py + build_d0.py — the D0 discriminator.
- mtp.py: LOOKUP env + l_hist buffer + draft-graph append + reset mfill.
- test_w100k.py: reset_spec tok_hist seeding + [lookup] instrumentation.
- Logs: ~/r3_lookup_100k.log (the green gate).
