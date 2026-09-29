# P9 EVAL BATTERY — RESULTS (2026-09-29)

The first output-QUALITY measurement of the program (all prior gates were
bit-exactness). Rig: RTX 3090 eGPU on the engine host, daemon under launchd,
api 127.0.0.1:8080. Models: qwen3.6-35b-a3b-egpu (MoE, MTP K=4 default),
qwen3.8-27b-egpu (dense, K=10 lookup). Harnesses in ~/tinygrad-metal/eval/,
per-item data in eval/results/.

## 1. GSM8K — accuracy + speed (headline)

Protocol: FIRST 100 test problems (deterministic), 4-shot primer from the
FIRST 4 train examples (paper format, <<calc>> annotations kept), greedy
(engine is greedy-always), enable_thinking=false (standard non-thinking
protocol), max_tokens=320, stop=["\nQuestion:","Question:"], streaming for
TTFT/decode split. Answers extracted via "#### N" with last-number fallback
(models end after the computation sentence; ~0% emit "####"; 100% of
responses yielded a number — 0 extraction failures both legs).

| model                    | accuracy   | decode tok/s (med) | TTFT med | lat med | n    |
|--------------------------|------------|--------------------|----------|---------|------|
| MoE 3.6-35B-A3B (MTP K4) | 93.0% (93) | 36.7               | 9.5s     | 13.2s   | 100  |
| dense 3.8-27B (K10 lkup) | 95.0% (95) | 23.0               | 2.4s     | 8.4s    | 100  |

- MTP-vs-T1 toggle for the MoE: NOT run — MM_MTP is a daemon-env kill switch
  (requires restart); MTP≡T1 bit-exactness is already proven 60/60 in-harness.
- The dense leg survived ~10 engine deaths (crash-loop, finding F2); all 100
  problems completed across restarts via a resume driver.
- CAVEAT: both legs ran before the first-token-loss fix (F1) was deployed
  mid-battery. The lost token was always the leading word of the answer echo
  ("Janet's" -> "et's"), never the final number; extraction unaffected.
- MoE TTFT is the weak spot: ~630-token FRESH prompts at ~66 tok/s effective
  through the stack (vs dense ~260+ tok/s). No cross-conversation cache hit
  is possible without prompt_cache_key (by design here — vanilla FRESH).

## 2. Perplexity — teacher-forced NLL (MoE, engine-side direct scoring)

ppl_moe.py: Rig7 + the PF-256 chunk graph (the exact test_moe36.py boot minus
MTP/serve); per-seat trunk hidden -> rmsz2048g -> h6k2048 -> full 248320-logit
download; float64 logsumexp; target = next token. Score rate 118-135 tok/s.

| domain                       | tokens | NLL/tok | PPL   | greedy acc |
|------------------------------|--------|---------|-------|------------|
| prose (Pride & Prejudice)    | 32767  | 0.1304  | 1.139 | 96.5%      |
| code (engine0/serve.py)      | 10239  | 2.9931  | 19.95 | 53.7%      |
| prose_private (FIX_CAMPAIGN) | 3071   | 3.5374  | 34.38 | 38.4%      |
| code2 (MM_P7_lib.py)         | 8703   | 1.5712  | 4.812 | 70.8%      |

- The Gutenberg classic is verbatim-memorized (96.5% greedy next-token): PPL
  1.14 is a CONTAMINATION FLOOR, not quality. Honest domains: prose_private
  34.4 (jargon-dense private docs), code 19.9, code2 4.8.
- These are the program's first PPL baselines for kernel/quant regression
  work; identical corpus + identical tokenizer (verified equal on both GGUFs)
  so future deltas are apples-to-apples.

### 2b. Dense PPL (F5 fixed 2026-09-29b — the full battery ran clean)

ppl_dense.py: the exact test_w100k daemon boot (snapshot + T1 ref + graphs)
then pcache.fresh_prefill with the ingest scorer at every 128-chunk boundary;
per-row pfk_n16 -> head8 -> logits download, float64 logsumexp. Score rate
193-208 tok/s (the m128 trunk + FP16 head make dense scoring FASTER than the
MoE twin's 118-135). Results: eval/results/ppl_dense.json.

| domain                       | tokens | NLL/tok | PPL   | greedy acc |
|------------------------------|--------|---------|-------|------------|
| prose (Pride & Prejudice)    | 32767  | 1.4731  | 4.363 | 63.3%      |
| code (engine0/serve.py)      | 10239  | 1.5910  | 4.909 | 66.4%      |
| prose_private (FIX_CAMPAIGN) | 3200   | 2.9196  | 18.53 | 44.1%      |
| code2 (MM_P7_lib.py)         | 8704   | 1.0248  | 2.786 | 77.9%      |

- Dense-vs-MoE on the HONEST domains: dense WINS all three — code 4.91 vs
  19.95, prose_private 18.53 vs 34.38, code2 2.79 vs 4.81 (the 27B dense
  reader beats the A3B router on unfamiliar code + private jargon; also
  IQ3_XXS-vs-UD-IQ3_S quant is not the dominant term at this gap). On the
  contaminated prose domain dense shows the honest 4.36 while the MoE's 1.14
  memorization floor hides its true prose level.
- Numerics note: dense logits are the engine-native FP16 head output (the
  same values greedy decode decides on); the MoE head emits FP32. NLL
  accumulation is float64 on both rigs.

## 3. Long-context needle (dense signature capability)

BLOCKED as a systematic 10-trial number by the serving-layer sysmem
exhaustion (F3): once the daemon's graph-build pool is drained, every
long-prompt prefill fails with engine "list index out of range" — observed
at BOTH ~60k and ~20k contexts (10/10 trials each, all attempts).
ONE clean manual datapoint (fresh boot, fixed API): 61,189-token context,
needle at ~50% depth, code 53177 -> response exactly "53177" (stop, 5
completion tokens, 185s wall ≈ 331 tok/s end-to-end). Capability
DEMONSTRATED; the 10-trial spread across depths is a next-session item
(after F3's fix).

## 4. Throughput during eval (full serving stack)

- MoE decode (GSM8K math-prose class, MTP K=4): 36.7 tok/s median.
- Dense decode (K=10 lookup): 23.0 tok/s median.
- Dense prefill via API: ~630-token FRESH in ~2.4s TTFT (≈260 tok/s).
- MoE direct PPL scoring (incl. PF prefill + per-token head): 118-135 tok/s.

## 5. FINDINGS (ranked; log evidence in engine logs + eval/*.log)

F1 — FIRST-TOKEN-LOSS IN SERVING (found + FIXED live this session).
  The engine's prefill returns `cur` = the model's FIRST response token
  (predicted at the boundary hidden; held in cur_slot; cycle emits start at
  the SECOND token — serve.py's own header documents the contract). 
  api_server._run_engine_sync consumed the prefill result only for
  mode/cached_tokens and NEVER fed `cur` into the visible stream: every
  completion lost its first token; one-token answers came back EMPTY with
  completion_tokens=0. Evidence: eval/probe_engine.py (prompt "…12345":
  prefill cur=16 ('1'), cycles [17,18,19,20]); GSM8K rows starting mid-word.
  FIX: feed _pr["cur"] through _feed_token before the generate event loop
  (api_server.py, marked "P9 EVAL FIX (first-token loss)"; pre-fix backup at
  api_server.py.p9bak). Verified: "APPLE"->"APPLE", "12345"->"12345",
  needle manual trial exact. LIVE for both models (API-only change; restart
  api, no GPU involvement).

F2 — DENSE-ENGINE HARD CRASH-LOOP under sustained decode load.
  ~10 deaths in ~3h of dense GSM8K traffic; MTBF 13-18 min of serving.
  Whole-process-tree death (no wrapper child_exit records survive), several
  full machine resets (GPU-EXIT class; ResetCounter diags 09:10-09:22).
  Last-op distribution: 3x generate-begin, 1x gen_rebuild, 1x prefill M64
  tail. Cumulative cycle counts at death: 906/911/1206/1457 with rebuild
  fences firing normally throughout — NOT the classic unfenced ~950-cycle
  budget. The MoE served 100/100 back-to-back clean under the identical
  harness: dense-path specific. Prime suspect: host-mapped sysmem
  exhaustion via the documented kernargs-slab leak (every ParityGraph/
  PfGraph build leaks a ka slab; GSM8K's fence cadence — 5-6 gen_rebuilds
  per boot + PF graph churn — accumulates slabs until the dext hard-faults).

F3 — SYSMEM-MAPPING EXHAUSTION, soft variant (blocks long-prompt serving).
  After enough graph builds on one daemon life, any NEW PfGraph build
  raises `IndexError: list index out of range` at
  tinygrad/runtime/support/system.py:383 (`fd = struct.unpack('<i',
  anc[0][2][:4])[0]`) — the dext MAP_SYSMEM_FD RPC returns no fd.
  Path: _pc_prefill -> follow_up -> prefill_batch_m128 -> _pf_submit_chunk
  -> _pf_graphs -> PfGraph.__init__ ka alloc. Repeated long-prompt traffic
  crosses ATTN_THR (=54040) and builds the gs26 graph class; the pool
  drains; every subsequent long FRESH/CACHE_HIT prefill fails. Fresh boot
  clears it (the successful 61k manual needle call). Likely the soft face
  of the same exhaustion as F2's hard face. FIX DIRECTION: ka-slab reuse /
  free-on-build (the known "kernargs leak" open item).

F4 — /health under engine churn: accurately reports engine_down when the
  engine socket is gone (the API process survives engine deaths and serves
  503s with queue semantics); during hard resets clients see connection
  resets instead. Not a stale-path bug — but 503-storm + engine_down
  flapping IS the observable signature of F2/F3 for operators.

F5 — DENSE direct-PPL device fault: RESOLVED (harness bug, one line). The
  fault was the logits COPYOUT reading 2x the buffer: `down_at("logits", 0,
  VOCAB, np.float32)` = VOCAB*4B from a VOCAB*2B FP16 buffer (trunk.py:65
  `("logits", VOCAB*2, np.float16)`) — 496KB past the end, device fault at
  the first copyout (2/2 boots; logs_ppl_dense.log). The eager head launches
  were NEVER the problem: pfk_n16 sync-clean + head8 wait-clean precede the
  fault in the traceback (engine0.py down_at -> _copyout -> err_state).
  Fix: download as np.float16 (the engine's native head output; h_argmax/
  greedy read the same values). Post-fix: the full 4-domain battery ran
  clean end-to-end (§2b; logs_ppl_dense_f5fix.log). The MoE twin worked
  because its logitsb is FP32 and its dn() reads the matching dtype.

F6 — eval-harness accounting bug (client-side): the GSM8K harness's
  post-resume running "acc=" print mixed a session-scoped correct-count
  with the cumulative denominator (printed 0.345 while true running
  accuracy was ~0.94). Per-row JSONL was always clean; all final numbers
  recomputed from rows (gsm8k_*_recomputed.json).

F7 — ops law confirmed live: `enginectl stop`'s staydown marker dies with
  the GPU-EXIT reset it provokes (newly-created-file law) — launchd
  relaunches the engine within ~2-4 min. A PRE-CREATED marker (written
  while the box is stable) survives the reset and holds the engine down
  for standalone-GPU windows. Used successfully twice this session.

## 6. Methodology notes

- Corpora: gutenberg_pap.txt (headers stripped), engine0/serve.py,
  FIX_CAMPAIGN.md, MM_P7_lib.py; pre-tokenized by the API's GGUF tokenizer
  (verified identical between both models' tokenizers); 32k/12k/12k/8.7k
  token slices.
- Teacher-forced scoring convention: no BOS; hidden at pos i scores token
  i+1; final seat of final chunk skipped; the SAME corpora await the dense
  rerun for the dense-vs-MoE PPL comparison (identical text + tokenizer).
- Rig ops during the battery: model swaps via `enginectl switch` (arms
  intent pre-stop with the sync dance — survived every reset); standalone
  scoring windows via pre-created staydown marker + graceful stop; client
  drivers on the driver-seat machine to survive rig resets.

## 7. Artifacts

eval/: gsm8k_eval.py, ppl_moe.py, ppl_dense.py, needle_eval.py, tok_prep.py,
probe_engine.py, diag_boots.py, ppl_run.sh, drive_gsm8k.sh, drive_needle.sh;
eval/results/: gsm8k_{moe_mtp,dense}.jsonl (+ _recomputed.json),
ppl_moe.json, needle_dense20k.jsonl (blocked-run evidence);
engine0/api_server.py.p9bak (pre-F1 backup).

Rig state at close: MoE resident, /health ok, smoke request correct with
first token intact (F1 fix live).
