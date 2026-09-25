# M1-B — daemon fixes + OpenAI-compatible API façade + ops packaging

Mission (from M1A_SERVING.md open issues + SERVING_PLAN.md API-side spec).
Project HEAD after M1-B: see `git log` (engine0/). Host law unchanged: the
daemon is `M1A_SERVE=1 ... python -u test_w100k.py` (HOST-PROCESS BOOT LAW).

## The two M1-A open issues — resolution

### (a) "second FOLLOW-UP after save/load kills daemon+driver"
**Status: NOT REPRODUCIBLE under the v2 daemon — full scripted sequence PASS ×2
(16/16 steps per iteration, plus the strict back-to-back-FU variant ×2).**
- Gate (`engine0/m1b_gate.py` Gate A): park → gen → save → load → gen →
  FOLLOW_UP → gen → **FOLLOW_UP #2** → gen → save#2 → load#2 → gen →
  back-to-back FU (no gen between) ×2 → gen. Ran twice consecutively on one
  daemon: exact (load-resume deterministic A3==A2), clean, alpha intact.
- What changed vs M1-A (confounded, since multiple v2 changes landed at once —
  any/all could be the accidental fix):
  1. **Per-connection listener threads** (v1 served ONE connection to EOF
     before accepting the next — a second client's socket events interleaved
     with a mid-FOLLOW_UP driver could hit the old single-buffer reader).
  2. **10 s keepalive** (dext/TB never idles between RPCs).
  3. mtp.py optional log/prog hooks (no-ops on the canonical path).
- Root cause NOT isolated to a single line: the fault class (silent native
  kill, no crash report) did not recur across 4 full kill-sequences + 8
  back-to-back FUs. The structured logging (`/tmp/m1a_serve.log`, stage
  markers + `dev.timeline_value` + 2 s heartbeat) is now in place — if it
  EVER recurs, the last marker before the heartbeat gap names the faulting
  stage. Post-mortem recipe: `tail -40 /tmp/m1a_serve.log`.

### (b) "first-RPC-after-idle hang"
**Isolation experiment (old v1 daemon): 19 min idle → snapshot_load responded
in 0.14 s.** Pure idle does NOT wedge the engine. The M1-A wedge correlated
with the daemon's HISTORY (that daemon had run save/load + FOLLOW-UP cycles
before idling) — i.e. issue (a)'s damage class, not idle itself.
**Fix shipped anyway (defense in depth): 10 s keepalive** — main loop
`Q.get(timeout=M1A_KEEPALIVE_S)` → tiny eager dposadd probe + sync (the proven
health-probe pattern; NOT a graph; lone-graph law honored). The dext never
sees an idle gap > 10 s.
**Gate: idle 300 s → status = 3 ms; snapshot_load = 0.1 s. PASS.**

## NEW BUGS FOUND + FIXED by M1-B gates
1. **FRESH prefill ran the trunk from STALE GDN state** (P0): serve.py's FRESH
   handler did `reset_fresh` (zeroes SPEC rec4/conv4) + `fill_draft` +
   `prefill_t1` — but never seeded the TRUNK GDN buffers (`rec{i}`,
   `conv{i}_0/1`). The T=1 trunk prefill therefore started from whatever the
   previous conversation left there → nondeterministic FRESH outputs
   (m1b_gate: same 64-token prompt, rep1 cur=41231 vs rep2 cur=6545).
   **Fix: `E.stload_trunk()` right after `reset_fresh()`** (seeds trunk from
   spec slot 4 = zeros; exactly gate23.py's proven `reset → stload_trunk →
   prefill` pattern). This bug would have corrupted EVERY new conversation
   through the API.
2. **v1 daemon listener starved concurrent clients**: one connection was read
   to EOF before the next was accepted → /health during an active generate
   was impossible. Fixed by per-connection reader threads (GPU ops still
   strictly serialized through the single Q).

## NEW LAWS (M1-B)
- **SHUTDOWN-REBOOT LAW (2/2 observed)**: `shutdown` RPC → `os._exit(0)` after
  a session that included snapshot_load cycles → the MACHINE reboots within
  seconds (both M1-B restarts; no crash report; clean state after). Treat
  `shutdown` as a reboot-class op: send it, then WAIT for ssh to return
  (1-3 min), clean stale locks, relaunch. With launchd installed this
  self-heals (RunAtLoad). Do NOT assume the box stayed up.
- im_end for THIS gguf is **248046** (not Qwen2's 151645) — verified via the
  tokenizer (gate G).
- System python3 (3.9) has NO numpy — API/gate code that touches npy files
  runs under `~/tg311/bin/python`; api_server.py itself is numpy-free.

## API SURFACE (engine0/api_server.py, system python3, PORT=8080, 127.0.0.1)
- `GET /health` — 200 {status:ok, engine:{...}, queue_depth} | 503 + Retry-After
  (warming / engine_down / engine_degraded when the circuit-breaker marker exists).
- `GET /v1/models` — single entry `qwen3.8-27b-egpu`.
- `POST /v1/chat/completions` — stream + non-stream.
  - Accepted: messages (system/user/assistant/developer; text-only),
    max_tokens (default 512, clamped to ctx headroom), stop (str | up to 4
    strings; exact special-token strings become stop_token_ids, others are
    string-matched with max(stop)-1 holdback), stream, stream_options.
    include_usage, model (echoed), conversation_id (+ `x-conversation-id`
    header; echoed in response field + header), seed/presence_penalty/
    frequency_penalty/... (ignored, listed in `ignored_params`).
  - 400s: temperature/top_p/top_k ∉ {omitted,0,1} ("sampling not yet supported
    (M2)"), tools/tool_choice, logprobs/top_logprobs, n>1, response_format,
    multimodal content, malformed stop/max_tokens.
  - 429 + Retry-After when queue full (1 active + 4 waiting FIFO; beyond → 429).
  - Response extras: `prefix_mode` (FOLLOW_UP | FRESH), `conversation_id`.
  - SSE: role chunk → per-token content chunks (split from per-cycle m+1
    batches) → finish chunk → optional usage chunk → `data: [DONE]`;
    prefill progress as `: prefill N% (stage)` comments.
  - Tokenizer: VENDORED SimpleTokenizer + GGUF-KV parser (zero tinygrad/GPU
    imports; process restartable). **Gate G: render+encode bit-identical to
    the fork's tinygrad/llm/cli path (103/103 ids, unicode+emoji)**.
    Cache-pickled at /tmp/api_tok_cache_* (0.8 s warm start).
  - Detok: byte-exact streaming — `_tok2bytes` accumulation + UTF-8 holdback
    (never decode() per chunk) + stop-string holdback.
  - Prefix reuse: ids2 must EXACTLY extend the resident fed stream (API keeps
    the mirror; engine cross-check via status conversation_id/fed_tail) →
    FOLLOW_UP with `cur` override = ids2[fed_len] (fed stream == client render
    exactly); else FRESH (also on conversation_id mismatch / API restart).
    Mid-conversation stop over-commit (>max_tokens truncation, stop-batch
    tail) breaks exact extension → next turn silently falls back to FRESH.
  - Disconnect mid-stream → engine cancel (same socket, ≤70 ms class) +
    queue slot released.

## Field matrix (gate f)
| field | behavior |
|---|---|
| model | echoed |
| messages | text-only; multimodal → 400 |
| temperature/top_p/top_k | omitted/0/1 OK; else 400 (M2 sampling) |
| max_tokens | default 512; clamped to ctxk − pos − 32 |
| stop | str/list≤4; special-token strings → stop_token_ids; others string-level holdback |
| seed, presence_penalty, frequency_penalty, user | ignored + echoed in ignored_params |
| tools, tool_choice | 400 (no tool runtimes — Cline/Cursor/Claude Code unsupported) |
| logprobs, top_logprobs | 400 |
| n≠1 | 400 |
| response_format | 400 |
| conversation_id / x-conversation-id | prefix pinning; echoed |
| stream_options.include_usage | usage chunk before [DONE] |

## OPS
- `engine0/ops/engine_daemon.sh` — launchd wrapper: GPU lockfile (/tmp/nv_usb4.lock,
  stale-lock cleanup only when no GPU python alive), crash counter
  (/tmp/llm_engine_crashes.log), circuit breaker (3 exits/10 min → touch
  /tmp/llm_engine_staydown → stay down; /health reports engine_degraded;
  clear via `enginectl clear-breaker`).
- `engine0/ops/api_server.sh` — API wrapper (system python3 + pip3 --user
  fastapi/uvicorn/jinja2 — installed: fastapi 0.128.8, uvicorn 0.39.0, jinja2 3.1.6).
- `engine0/ops/com.<user>.llm-engine.plist`, `com.<user>.llm-api.plist` — NOT
  auto-installed. Install: `engine0/ops/enginectl install` (sudo; loads both
  LaunchDaemons). Uninstall: `enginectl uninstall`. Manual ops:
  `enginectl {start|stop|status|logs} [engine|api|all]`.
- Manual daemon (no launchd): see the env line in M1A_SERVING.md / the launch
  command in this file's git history (DEV=NV SKV=1 SKV_K=g4nw32 SKV_S=256
  SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 M1A_SERVE=1 M1A_KEEPALIVE_S=10
  ~/tg311/bin/python -u test_w100k.py).
- Boot to ready ≈ 7-8 min (weights ~11 s warm + KV int8 ~40 s + T1 warmup ~60 s
  + fill_draft ~240 s + graphs). Watch: `tail -f /tmp/serve_m1b.log`.
- Structured daemon log: `/tmp/m1a_serve.log` (JSON lines: rpc/stage/tl/heartbeat).

## GATE RESULTS (this session; full transcripts in the session log)

ENGINE (`m1b_gate.py`, v3 daemon, canonical 100k ctx):
| gate | result |
|---|---|
| alpha / HOST-PROCESS BOOT LAW (tok/cyc >= 2.2) | PASS 2.35 |
| Gate A it1+it2: park->gen->save->load->gen->FU->gen->FU->gen->save2->load2->gen (32 steps) | PASS 32/32 x2 |
| Gate A strict: back-to-back FUs (no gen between) after load | PASS x2 |
| load-resume determinism (A3==A2) | PASS [6545,9956,6545,...] exact |
| FRESH deterministic x2 (after conv-parity fix) | PASS cur=52448 stable across ALL contexts (cold/after-gen/after-FU/after-park) |
| stop_token_ids (6545 = deterministic first emit at park) | PASS (stops cycle 1, stop=true) |
| idle 5 min -> status | PASS 3 ms |
| idle 5 min -> snapshot_load | PASS 0.1 s |
| Gate G template render vs fork (tg311 reference) | PASS 103/103 ids |

API (`api_gates.py`): health/models PASS; (a) non-stream well-formed OpenAI JSON
PASS (finish=length, usage, echo, ignored_params); (b) SSE sequence + [DONE] +
usage chunk PASS (finish length-capped). The REMAINING API gates (a-exactness
recheck, b-delta-equality, c/d/e/f with the rewritten queue) were GREEN in code
but their full re-run was cut short by the fault class below — rerun
`python3 api_gates.py` when the engine is up.

## OPEN ISSUE (new, M1-C entry): the GATE-STORM DEVICE FAULT
- Reproduced twice: within minutes of the full api_gates sequence (which mixes
  engine-direct RPCs with API requests, a cancelled 4000-token stream, and
  6-way concurrency), a generate faults the dext
  (`RuntimeError('Device fault detected')` at `timeline_signal.wait` in
  sess.step; once ~56 s after a cancelled long stream + idle).
- The v3 daemon now EXITS on device fault (device_fault slog + os._exit(1)) —
  a faulted daemon no longer zombies.
- *** GPU-EXIT/REBOOT LAW (4/4 this session): the GPU python process exiting
  while the dext is alive/faulted COINCIDES WITH A MACHINE REBOOT within
  seconds (3x shutdown-RPC, 1x fault-exit). With launchd installed this
  SELF-HEALS (RunAtLoad relaunch, ~9 min to ready; /health 503 meanwhile).
  Without launchd: manual relaunch required after every fault/shutdown. ***
- NEXT (M1-C): bisect the trigger (cancel->next-generate transition vs
  repeated-FRESH-under-concurrency vs engine-direct+API interleave); instrument
  accept/flush graph submits with slog markers; consider refusing engine-direct
  RPCs while an API conversation is resident.

## M1-C RESOLUTION (see M1C_STABILITY.md)
The GATE-STORM DEVICE FAULT is FIXED (root causes + laws + gates there);
the recommended daemon recipe now includes M1A_GEN_REBUILD_EVERY=256.
