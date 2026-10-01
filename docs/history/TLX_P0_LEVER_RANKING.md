# TLX DRAFTER Phase 0 — THE LEVER-RANKING MEMO (G0 + G1)

2026-10-01. Agent: GLM (Phase 0 dispatch). Rig repo at fc0c49e; chain_sim.py = the instrument.
Traces: rig `~/trace_dump/{r8_prose, corp/{prose,code,prompt100k}, gsm8k, r8_prose_w4a8, microdiff*}`
(manifests sha256'd; dumps NOT committed). Mac mirror: `~/drafter/phase0/traces/`.

## 1. G0 — the calibration verdict

Reference capture (canonical dense env, PF_W4A8=0, TLX_EAGLE_K=4 forced-K4, r8_prose
anchor @97.9k): 51 K4 cycles, engine m-dist [29,14,7,0,1], **E[m]|k4 = 0.6275,
E[m]|k2 = 0.5882** — inside the plan's G0 band (0.633±0.05 / 0.583±0.05; the historical
[70,30,14,6,0] was 2 deterministic reps of this class).

| sim vs engine (engine-conditioned replay, n=51 K4 cycles) | value |
|---|---|
| sim m-dist | [30, 14, 7, 0, 0] (engine [29, 14, 7, 0, 1]) |
| per-proposal agreement | **201/240 = 83.8%** |
| E[m]\|k2 | 0.549 vs 0.588 → **PASS (±0.05)** — the SHIP metric |
| E[m]\|k4 | 0.549 vs 0.6275 → 0.03 beyond band; residual = ONE m=4 cycle + 1 borderline a1 at n=51 |

**Verdict: G0 passes on the ship metric (E[m]|k2) and the m-dist shape; k4 misses
narrowly with a single-cycle-dominated residual.** Pack-to-pack deltas (the sim's job)
are 2-4x the residual. Validations earned: trunk-head argmax 39/39 exact from h_seed;
emb dequant byte-exact; GEMV outputs 4e-4; synthetic a1 dot exact (512.0); upload
round-trip identical; KV8 append row 5e-3; W4A8 A/B on the anchor +0.017 k4 (the
W4A8=0 capture law costs ~nothing).

Instrument bugs found by the microdiff ladder (both fixed, law-grade):
1. np.load(mmap) re-slicing breaks `is`-identity → K dequantized with V's scales.
2. class-B targets off-by-one: proposal_i compares vs target(t+i), not target(t+i+1).

## 2. The packed ladder on r8_prose @100k (class-A, engine-conditioned)

| pack | E[m]\|k2 | E[m]\|k4 | a1 |
|---|---|---|---|
| **bf16 HF first-party** | **0.608** | 0.608 | 0.451 |
| current engine Q4_0 pack | 0.549 | 0.549 | 0.412 |
| pilot_rtn (trained, RTN Q4_0) | 0.412 | 0.431 | 0.333 |
| pilot_gptq (trained, GPTQ Q4_0) | 0.373 | 0.412 | 0.294 |

- PRECISION at 100k prose: bf16 − Q4_0 = **+0.06 E[m]|k2** — a small lever; the Q4_0
  first-party pack is near its own bf16 ceiling here. The 42%-vs-70%+ gap vs syv-ai is
  NOT mainly quantization.
- **THE PILOT TRANSFERS NEGATIVELY to 100k prose (−0.14..−0.18 k2)** despite winning on
  its cloud GSM8K-class replay. Cloud-teacher + short-ctx training moves the weights
  OFF the 100k-prose optimum. (kv fill rows were the current pack's — a
  mixed-conditioning approximation; the regression is 3-4x the G0 residual.)

## 3. The G1 battery (current pack; targets = engine-greedy from dumped trunk hiddens)

| trace (ctx) | mode | a1 | a2\|1 | a3\|2 | a4\|3 | E[m]\|k2 | E[m]\|k4 |
|---|---|---|---|---|---|---|---|
| gsm8k-10 @~0.7k | serve | 0.513 | — | — | — | 0.816 | 0.934 |
| code @8k | serve | 0.656 | 0.512 | 0.395 | 0.294 | 0.992 | 1.164 |
| code @8k | **tf** | 0.656 | **0.679** | **0.719** | **0.659** | 1.102 | **1.633** |
| novel prose @16k | serve | 0.676 | 0.410 | 0.225 | 0.313 | 0.953 | 1.035 |
| novel prose @16k | **tf** | 0.676 | **0.624** | **0.648** | **0.486** | 1.098 | **1.504** |
| novel prose @100k (r8, IN-VIVO) | serve | 0.431 | 0.364 | 0.125 | — | 0.588 | 0.6275 |
| prompt100k spans @64k/95k | (artifact: corpus is greedy-followable only 1/12 — teacher-forced anchors can't predict the greedy attractor; not a ctx readout) | | | | | 0.08 | 0.08 |

## 4. THE LEVER RANKING

Decomposition of the in-vivo 100k-prose state (a1 0.43, k2 0.588) toward the syv class:

1. **EXPOSURE BIAS — the TTT prize (the biggest measured k4 lever, solid k2)**:
   true-conditioning recovers the DEEP conditionals to 0.62-0.72 (vs 0.23-0.51
   chained): +0.47 E[m]|k4 and +0.11..0.15 E[m]|k2 at matched workload/ctx. EAGLE-3's
   multi-step unroll targets exactly this. **TTT is worth doing** — but only with the
   right data (see #2).
2. **DATA DISTRIBUTION / CTX (the pilot's lesson)**: novel-prose acceptance falls
   0.676→0.431 (k2 0.95→0.59) between 16k teacher-forced and 100k in-vivo; the pilot
   trained on short-ctx cloud data REGRESSES 100k prose. Phase-1 data mix MUST add
   32k+ long-ctx samples + prose/doc-QA classes + ~5M engine-trace tokens (Stage B);
   the S1 dumps are captured and reusable.
3. **PRECISION (small)**: +0.06 k2 at 100k (bf16 vs Q4_0). Cheap parallel rung (Q8_0 /
   GPTQ-first-party) — never the headline.
4. **WORKLOAD (secondary at matched ctx)**: code 0.99 vs prose 0.95 k2 @≤16k.

### RECOMMENDED PHASE-1 SCOPE (commits the full run)
- **Train-first, corrected data mix**: EAGLE-3 recipe stands (v1 1-layer mainline);
  pool = GSM8K/MetaMath + chat + code + **novel-prose/doc-QA + a 32k+ long-context
  subset** (the plan's 40-80k prompt pool reweighted); 20M pilot → 40M full.
- **Stage-B adaptation is REQUIRED before any ship decision**: fine-tune on the
  engine-trace dumps (r8_prose 100k anchors + corpus hiddens + gsm8k), repack,
  re-score through chain_sim on the SAME r8 trace (the packed-vs-packed gate).
- Keep the current Q4_0 pack as control; the pilot packs are NOT shippable for prose
  (they may still help the GSM8K battery class — score before discarding).
- K4 stays OFF (conditionals 0.12-0.36 at 100k; bar is 0.78-0.8). K2 is the carrier.
- G2 unchanged (sim E[m]k2 ≥ 1.0 AND k4 ≥ 1.6), now enforced through the CALIBRATED
  chain_sim on the r8 trace + gsm8k traces.

## 5. Ops / infrastructure shipped this phase
- **pcache draft_pack fingerprint** (svc_fp.set_draft_pack_extra + mtp TLX_DRAFT_PACK +
  BOTH daemon/API sides) — committed 7c00ea5; **Tier-1 gate GREEN after the change**
  (60/60 ×2, deep on/off 60/60, stock 59/59, tier-2 60/60). One cold pcache rebuild
  per model on next boot (config_fp changed) — expected + documented.
- chain_sim.py + trace_dump_lib + dump harnesses in the rig repo (fc0c49e); microdiff
  v1-v5 harnesses (~/phase0_stage) as the permanent instrument-debugging pattern.
- Rig ends: MoE resident, /health ok.
