# TLX Serving Runbook (R3 revision)

The single operational reference for the engine daemon + API: the knob
census (R3-49), the config_drift 503 runbook, cancel/resumability semantics,
the conversation-source rules (R3-40; **no gateway — direct API only**, see
the R3 hardening review (the `review-fixes-r3` merge series in the rig
history; predecessor record: `docs/history/FIX_CAMPAIGN.md`) §3.1: our daemon
is launchd-supervised and must stay so; a second supervisor cannot honor the
GPU lock, the breaker/staydown discipline, or the swap intent state machine),
and the alarm delta.

Engine: `engine0/serve.py` (legacy + R6 batch scheduler) via
`ops/engine_daemon.sh` (launchd). API: `engine0/api_server.py` via
`ops/api_server.sh`. Control: `ops/enginectl`. Canonical env:
`ops/env.canonical` (THE single env source; the API drift-checks the daemon
against it).

## 1. Knob census (every os.getenv in serve/api_server/pcache)

### Daemon (serve.py) — set in ops/env.canonical unless noted
| knob | default | what it does / kill-switch |
|---|---|---|
| `BATCH_B` | 1 | engine batch width (2 = the R6 scheduler; **in config_fp** R3-19) |
| `BATCH_REBUILD_EVERY` | 232 | GLOBAL fence-all rebuild cadence (cycles; the ~950-cycle dext budget; **in config_fp**) |
| `BATCH_PF_CHUNK` | 64 | barrier-prefill chunk size (decode interleave between chunks; **in config_fp**) |
| `KV8` | 1 | int8-KV (pcache requires it) |
| `LOOKUP_K` | — | deep-K lookup depth (the R5 ladder; **in config_fp**; LOOKUP_K=0 = the silent slow-path class the drift check exists for) |
| `M1A_CYCLE_SLOG` | 0 | per-cycle slog lines (forensics; rotation-aware R3-47) |
| `M1A_GEN_PAUSE_EVERY` / `M1A_GEN_PAUSE_MS` | 0 / 150 | reference-only quiesce knobs (proven NOT to reset the dext budget) |
| `M1A_GEN_REBUILD_EVERY` | 256 | per-generate rebuild window (TLX_GLOBAL_REBUILD drives it from the GLOBAL counter; **in config_fp**) |
| `M1A_KEEPALIVE_S` | 10 | idle probe cadence (watchdog idle-arm = 3x this, R3-02) |
| `PC_ENABLED` | 1 | durable prompt cache on/off |
| `PF_DFILL` | 1 | interleaved draft-KV fill in prefill (**in config_fp**) |
| `PF_PREFILL` | — | P-series batched prefill path (**in config_fp**) |
| `R6_PF_T1` | — | batch boot PF T1 path (**in config_fp**, R3-19) |
| `PF_G3M_MB` | — | packed7 G3M bank budget (**in config_fp**, R3-19) |
| `TLX_ADMIN_TOKEN` | "" | privileged-RPC ACL (unset = admin methods REFUSED) |
| `TLX_CONN_MAX` | 64 | per-conn listener thread cap (R3-51) |
| `TLX_CYC_RESET` | 0 | reset cyc_slot at rebuild (changes cycle event numbering) |
| `TLX_ENGINE_Q_MAX` | 32 | engine RPC queue cap; loud frame on overflow (R3-13) |
| `TLX_GLOBAL_REBUILD` | 1 | drive rebuilds from the GLOBAL cycle counter (W4.3) |
| `TLX_GLOBAL_CYCLE_REBUILD_EVERY` | 928 | L7 FIX 5: global GRAPH-SUBMIT budget (all classes incl. prefill chunk replays; gcycle.ParityGraph.submit counts, fence/entry rebuild resets; 0 = off) |
| `TLX_LOGS_DIR` | ~/tinygrad-metal/logs | persistent logs + staydown/crashlog |
| `TLX_MAX_LINE_MB` | 4 | per-line byte cap (conn drop past it; V-31) |
| `TLX_MODEL_ID` | qwen3.8-27b-egpu | R3-42 residency identity (pre-MoE) |
| `TLX_PEERCRED_LENIENT` | 0 | escape hatch: accept conns when LOCAL_PEERCRED unavailable (R3-21 default = fail-closed) |
| `TLX_PENDING_MAX` | 8 | batch pending-RPC cap (R3-13) |
| `TLX_SEND_TIMEOUT_S` | 10 | socket send timeout (stalled-consumer protection, V-30) |
| `TLX_SOCKET_UIDS` | "" | extra peer-uid allowlist (comma-separated) |
| `TLX_STEP_TIMEOUT_S` | 120 | watchdog step deadline while an RPC owns the GPU (boot included, R3-04b) |
| `TLX_BOOT_DEADLINE_S` | 900 | wrapper: child alive but not ready -> boot_fail + staydown, child LEFT RUNNING (R3-04a) |
| `TLX_BREAKER_N` / `TLX_BREAKER_WINDOW_S` / `TLX_BREAKER_READY_MIN_S` | 3 / 600 / 300 | breaker: consecutive crash-class exits with ready-uptime < READY_MIN; the window guards boot_fail loops only (R3-05) |
| `TLX_ENGINE_DIR` / `TLX_OPS_ROOT` / `TLX_GPU_LOCK` / `TLX_ENGINE_CMD` / `TLX_PYTHON` | — | wrapper test/ops overrides |

### API (api_server.py)
| knob | default | what it does |
|---|---|---|
| `PORT` / `BIND` / `GGUF` / `ENGINE_SOCK` | 8080 / 127.0.0.1 / models/Qwen3.8… | listeners + model file |
| `TLX_ADMIN_TOKEN` | "" | mirrors the daemon's; gates /health detail (HEADER x-admin-token only, R3-23) |
| `TLX_ALLOWED_HOSTS` | localhost,127.0.0.1,::1 | TrustedHost allowlist (comma-separated) |
| `TLX_BODY_TIMEOUT_S` / `TLX_QUEUE_WAIT_S` / `TLX_GUARD_GRACE_S` | 30 / 300 / 30 | pre-slot body deadline / waiter deadline / stream-guard grace |
| `TLX_CONV_LOCK_WAIT_S` / `TLX_ADMIT_WATCHDOG_S` / `TLX_ADMIT_HEAL_AFTER_S` | 120 / 10 / 90 | L7.1 admission hardening: bounded conv-lock acquire; watchdog heals a leaked permit only when the ENGINE is demonstrably idle (the R3-10 class) |
| `TLX_EVQ_MAX` | 8192 | SSE event queue (coalesce-first + loud-fail, R3-12) |
| `TLX_MAX_BODY_MB` | 10 | request body cap (413 before parse) |
| `TLX_ENV_CANONICAL` | ops/env.canonical | the drift-check env source |
| `TLX_COMPAT_IGNORE_SAMPLING` | 0 | accept in-range sampling params (recorded in ignored_params; greedy unchanged) — R3-33 |
| `TLX_COMPAT_STRIP_TOOL_MESSAGES` | 0 | strip role=tool / tool_calls instead of 400 — R3-38 |
| `TLX_RESIDENT_MAX` / `TLX_RESIDENT_FED_MAX_TOKENS` | 64 / 2,000,000 | resident-mirror LRU caps (R3-14; eviction surfaces x-resident-evicted) |
| `TLX_IDEMPOTENCY_TTL_S` | 120 | Idempotency-Key registry TTL (R3-39) |
| `TLX_TOK_HMAC_KEY` | "" | tokenizer-cache HMAC key (OUTSIDE the cache dir, R3-25) |

### Prompt cache (pcache.py)
`PC_ROOT` (~/prompt_cache) · `PC_QUOTA_GB` (60) · `PC_STRIDE` (1024) ·
`PC_MIN_HIT` (1024) · `PC_PIN_TTL_S` (7d) · `PC_PIN_BUDGET_FRAC` (0.5) ·
`PC_PIN_MAX_NODES` (96) · `PC_PIN_MAX_LIVE` (384) · `PC_FLUSH_S` (30) ·
`PC_HASH_VERIFY` (1). Health surfaces through engine status `pc`
(entries/total_bytes/quota/dropped/writer_errors/meta_save_errors — R3-28).

## 2. config_drift 503 runbook (which env flipped?)

The daemon's `config_fp` hashes: the `_ENV_KEYS` env set + the model-file
identity + the cubin-set digest (R3-19). /health (admin) shows
`config_fp` vs `config_fp_expected`. To diff:

    zsh -c 'set -a; source engine0/ops/env.canonical; set +a; \
      for k in BATCH_B LOOKUP_K PF_PREFILL ...; do print "$k=$k"; done'   # what the daemon booted with
    # vs the file the API parses:
    /usr/bin/python3 -c 'import sys; sys.path.insert(0,"engine0"); \
      import svc_fp; print(svc_fp.parse_env_file("engine0/ops/env.canonical"))'

Common causes: a manual `enginectl start` without the wrapper (stale env),
a knob edited in env.canonical while the daemon runs (restart through the
wrapper picks it up), a rebuilt cubin (deliberate invalidation — one cold
pcache rebuild), a tilde path in env.canonical (R3-20: the parser now
expands it, but keep absolute paths).
**Never** serve through drift: the 503 is the W2 fail-closed tripwire
(e.g. LOOKUP_K=0 boots the slow FRESH path silently otherwise).

## 3. Cancel / resumability semantics

- Client disconnect (stream + non-stream): the receive-owning watcher arms
  the engine cancel; generation stops at the next cycle boundary, mid-prefill
  at the next prog callback (R3-13); the conversation is NOT reusable
  (FOLLOW_UP refused until a fresh prefill re-establishes state).
- Engine-side cancel is scoped to the OWNING connection (R3-21); the legacy
  global flag is armed only by that conn (or the harness teardown).
- A stop-STRING / stop-token truncation: the turn is non-reusable
  (STOP-BATCH OVER-COMMIT law) EXCEPT a length-cap turn (the hidden drain
  tail keeps the R7a exact-prefix contract — the turn stays FOLLOW_UP-able).
- Slots evicted between decide and prefill: one silent FRESH fallback
  (R3-15), surfaced as `x-prefix-mode: FRESH`.

## 4. Deployment posture (direct, no gateway)

The service is a **direct single-endpoint daemon**: clients talk to
`127.0.0.1:8080` (or the bound host) with no proxy tier. Model swap, retries,
routing, and health live entirely in OUR supervisor + API layer. If a reverse
proxy is ever placed in front, it must forward unknown body fields verbatim
(`conversation_id`, `prompt_cache_key`, `user` all survive standard
OpenAI-compatible passthroughs) and must NOT buffer SSE streams.


## 5. Alarm delta (threshold → action)

| # | signal | threshold | action |
|---|---|---|---|
| 1 | watchdog step_timeout with rpc=prefill AND a pc_lookup hit logged | any | healthy restore killed — check beat cadence (R3-01 signature) |
| 2 | config_drift 503 | sustained >2min or any post-kernel-rebuild | env diff (§2); restart through the wrapper |
| 3 | pc_corrupt / emit_seq_violation / device_fault | any | page; pc_corrupt self-heals (quarantine+FRESH) — watch frequency |
| 4 | gen_rebuild_failed / failed_nonfatal | consecutive ≥2 | leading indicator of rebuild_budget_exhausted (R3-06 trips at 3) |
| 5 | crash-loop-after-ready: crashlog ruptime streak | ready_uptime <5min streak | R3-05's detector (breaker trips at N; streak = telemetry) |
| 6 | send_stalled | >5/min | slow consumers; TLX_SEND_TIMEOUT_S is protecting the GPU thread |
| 7 | swap_intent_stalled | age >180s | when the MoE swap lands (scaffold: tests/test_swapfsm_r3.zsh) |
| 8 | 429 rate | queue exhaustion | capacity: permits (BATCH_B) vs MAX_WAITING |
| 9 | pc_quota_hard repeats / eviction storms | >50/min | raise PC_QUOTA_GB or pin the hot chains |
| 10 | breaker trip / staydown present | any | enginectl clear-breaker ONLY after the cause is fixed (flocked, R3-07) |
| 11 | in_flight_duplicate 409s | burst | gateway retry storm — bound retries (§4) |

## 6. enginectl

`status` now shows: last 10 engine slog ops (tail of the persistent log) +
pcache health (entries/bytes/quota/counters via the admin /health view) +
the breaker/crashlog with ruptime attribution (R3-05) and the GPU-lock owner.
`stop` order: graceful socket shutdown (token via stdin) + wait-for-exit
BEFORE any launchd unload (R3-07); `clear-breaker` under the shared flock.

## P8 multi-model knobs (registry + swap lifecycle)
| knob | where | meaning |
|---|---|---|
| TLX_MODEL_PATH | serve.py / engine0.py / svc_fp | THE model file the engine must load (per-model env; the serve.py B.2 boot assert exits 18 on mismatch) |
| TLX_MODEL_ID | engine_daemon.sh -> engine env | the registry id of the resident model (engine status model_id) |
| TLX_MODEL_REGISTRY | api_server.py | registry JSON path override (batteries); default ops/model_registry.json |
| TLX_STATE_DIR | api_server.py | swap-state dir override (batteries); default ops/state |
| TLX_ENV_COMMON | api_server.py | shared env file override (batteries); default ops/env.common |
| TLX_T1_MODE / TLX_T1_TRIG | mtp.py | the adaptive T=1 prose mode (4 zero-accept K2 cycles -> T=1-only cycles; first 8-gram hit exits to deep) |
