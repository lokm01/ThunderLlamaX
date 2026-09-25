# Serving Plan — engine0 as an OpenAI/Anthropic-compatible service on the Mac
Synthesized 2026-09-15 from qwen-serving + kimi-kernels + grok-redteam analyses.

## VERDICT (consensus)
- **Two processes**: GPU-owner `engine-daemon` (engine + prefill + tokenizer, unix-socket
  JSON-RPC: status/prefill/generate/cancel/snapshot/tokenize) + stateless FastAPI `api`
  façade (restartable, zero GPU imports). launchd KeepAlive + circuit breaker (≥2 crashes/
  10min → STAY-DOWN + 503; never crash-loop the dext). Lockfile around the GPU.
- **OpenAI-native M1** (chat completions + SSE + /v1/models + /health). Anthropic later as
  a thin text-only ADAPTER outside the engine process (M3) — dual-native doubles SSE
  surface for zero clients we can satisfy without tools. Do not impersonate Claude.
- **Prefill is the product**: T=1 prefill = 21.78 tok/s → fresh 4k ≈ 3min (with progress
  events), but **prefix reuse makes follow-ups ~9s prefill + decode** (conversation state
  stays resident; clients resend full history → longest-prefix-match → prefill delta only).
  Real fix = batched M=16 prefill kernels (pGEMM/pKPRE/pATTN/pSCAN): 2k in ~4.5s v1,
  ~1.5-2.5s v2 — M3 kernel campaign (4-6 sessions).
- **KILL LIST (M1)**: continuous batching, multi-model, tools/function-calling, logprobs,
  n>1, vision, /v1/responses, stock-subprocess-per-request, sampler theater. temp≠0 → 400
  in M1 (greedy honest), sampling kernel in M2 (temp=0 path bit-exact = Tier-1 untouched).

## ENGINE-SIDE (kimi spec, file-referenced)
1. DecodeSession.step() extraction (mtp.py run_cycles → step; per-cycle: submit 4 graphs,
   wait(vf), read 32B emit record — NEW accept.cu outbox {pos,m,tok0..2,stop_flag} +
   Bufs.down_at windowed readback; NEVER the 400KB tok_hist down()).
2. Cancellation = stop submitting after wait(vf) (device quiescent, ≤69ms latency).
   Pause/resume free (carried prev timeline; parked engine is post-flush quiescent).
3. FIXED-HANDLE state (allocate once at boot: kv*, rec4/conv4, kv_d, hd_d*, h_seed,
   pos/cur/m/cyc slots, rings, tok_hist at full CTXK — patch trunk.py CTX=2048 poison);
   mutate only via win_up + device memsets. build_graphs() ONCE. (P.up realloc =
   graph rebuild + ~755MB orphan/reset, OOM after ~4 — the documented trap.)
   FRESH: memset zero-seed (GDN zeros, NOT poison) + prefill. FOLLOW-UP: reset nothing,
   prefill delta. EVICT: FRESH + dhd_seed (NEW: accept.cu stores committed-pos draft
   hidden into dhd_seed buffer; fill_draft seeds from it).
4. Stop conditions: stop_ids in accept.cu (truncate emits, stop_flag; over-commit past
   stop is fine — next turn appends after); string stop-sequences host-side with
   byte-exact incremental detok (_tok2bytes accumulation + UTF-8 holdback, llama.cpp style;
   SimpleTokenizer decode's errors='replace' mangles partial UTF-8 — don't use per-call).
5. Sampling (M2): samp.cu at the amx3 graph slot; params via DEVICE buffer (kernargs are
   static per graph — win_up between requests); temp=0 short-circuit = bit-identical
   amx3; sort-free top-k (k max-iterations) → top-p cumsum cut → Philox draw. Exactness
   contract becomes distribution-exact at temp>0; Tier-1 applies to temp=0.
6. Prefill v2 (M3): amx3 LD8/ACC3 → M=16 templates; x-tile smem 16KB (in the ≤36.8KB
   class); pKPRE M-row; pATTN causal M×L on the banked HMMA m16n8k16 fragment map;
   pSCAN = longer step loop; head M=1 only on last chunk; batched fill_draft (today
   375s@100k launch-bound).
7. Ops: SIGTERM → finish cycle → dev.synchronize() → exit; NEVER pkill -9. Sleep/wake +
   TB unplug → health probe → exit for supervisor (pmset sleep=0 already; closing the lid
   is an outage — document). Per-conversation snapshots (ckpt format + dhd_seed) on turn
   end; lazy reload. /health 503 while loading.

## API-SIDE (qwen spec + grok cut)
- POST /v1/chat/completions (stream + non-stream + stream_options.include_usage),
  GET /v1/models, GET /health. Anthropic adapter LATER (thin, text-only, outside engine).
- Field matrix M1: model=echo; messages (system/user/assistant; multimodal→400);
  temperature/top_p/top_k: only 0/default accepted, else 400; max_tokens default+cap;
  stop sequences; seed ignored (greedy deterministic anyway); tools/logprobs/n>1/
  response_format → 400 with clear message; conversation_id extension header for pinning.
- Queue: 1 active + FIFO cap 4 (beyond → 429 + Retry-After). SSE: one chunk per cycle
  (~2.78 tok / 69ms) + prefill progress as SSE comments; usage chunk at end.
- Qwen chat template from GGUF tokenizer.chat_template (fork cli.py) — bit-stable.
- Advertise ctx cap (default 8-16k; engine can 100k) — reject over-cap with clear error.
- Bind 127.0.0.1 only. No auth/HTTPS in M1.

## BUILD ORDER (gates keep the 100k Tier-1 recipe green at every step)
- M1-A engine: step() + emit record + fixed handles + reset/reuse + dhd_seed
  (gate: Tier-1 60/60 ×2 after FRESH reset; two-turn conversation exact). ~2 sessions.
- M1-B daemon+API: serve.py RPC + FastAPI + template + stops + queue + launchd
  (gate: curl chat stream + follow-up turn does NOT re-prefill; cancel works). ~1-2 sessions.
- M2: cancel-on-disconnect, sampling kernel, snapshot persistence, ctx cap, progress UX. ~3 sessions.
- M3 (optional): batched prefill kernels (the real TTFT fix), Anthropic text adapter,
  tools only with a written client target. 4-6 sessions.
Client reality (grok): works M1 = OpenWebUI, Continue chat, curl/SDKs. NOT without tools:
Cline, Cursor Agent, Claude Code (they are tool runtimes — document "no tools").

## M3 STATUS UPDATE (P4, 2026-09-16): batched prefill + batched draft fill LANDED
- Prefill v2 shipped through P1-P4 (see P4_PREFILL_OPT.md): M=16 chunked trunk +
  DBUF pGEMMs + merged multi-segment launches + INTERLEAVED BATCHED fill_draft
  (recorded-trunk-hiddens contract; alpha 2.67->2.68, spec==spec 160/160).
- FRESH-class numbers incl draft fill: 8k = 148.4 tok/s (52.0 s; was ~105.6);
  2k ~165 (was ~114); 100k fill_draft 253 s -> amortized per-chunk window.
- Daemon env additions: PF_MERGE=1 PF_DFILL=1 (default-on in code).
- TTFT: a fresh 8k prompt now ~52 s; 100k FRESH is prefill-bound (~13 min,
  down from ~17 min). Next TTFT levers = FFN wave-structure + pfa16 @100k
  (P4 doc parity section).
