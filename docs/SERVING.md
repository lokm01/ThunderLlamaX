# Serving — the OpenAI-compatible service layer

The engine is a **service**, not a benchmark: a GPU-owner daemon (the engine
process, booted once) plus an HTTP façade that speaks the OpenAI chat
protocol. Everything here is live on the rig and gated: the api_gates battery
(25/25 on the W5 stack), the GPU-free mock battery (85/85), 15-minute soaks
(zero faults), and Tier-1 canonical 60/60 x2 after serving stress. Campaign
journals: [history/M1A_SERVING.md](history/M1A_SERVING.md), M1B,
M1C_STABILITY, SERVING_PLAN, R1_PROMPTCACHE, and the five-wave review-fix
record [history/FIX_CAMPAIGN.md](history/FIX_CAMPAIGN.md).

This page has four parts: the **API reference**, the **operational
runbook**, **how the caches behave**, and the **known-issues ledger**.

## Part 1 — API reference

### Two-process design

```
client -> api_server.py (127.0.0.1:8080, OpenAI HTTP/SSE, no GPU imports)
              |  unix socket /tmp/llm-engine.sock, newline-delimited JSON-RPC
              v
         engine daemon = test_w100k.py host (M1A_SERVE=1) + serve.py library
              |  DecodeSession: 4 graph submits per cycle, one wait, 32B emit record
              v
           GPU (dext)
```

### Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | chat completion, stream (SSE) or non-stream; `usage` via `stream_options.include_usage` |
| `GET /v1/models` | model list |
| `GET /health` | liveness (503 while loading) |

**Streaming semantics**: one SSE chunk per engine cycle — each cycle emits
m+1 tokens (up to K+1), so chunks arrive at the cycle rate: ~8.9 tokens /
118 ms on the K=10 hit-class config (~7.5/105 ms at the published K=8
snapshot; ~2.8/69 ms at K=2). The finish window flushes exactly; usage is
reported in the final chunk when requested.

**Field matrix (M1+W1)**: `model` echoed; `messages` system/user/assistant
(multimodal -> 400); `temperature/top_p/top_k` only 0/default (greedy is the
honest M1 sampler — the sampling kernel is M2; temp=0 stays bit-exact
Tier-1); `max_tokens` / `max_completion_tokens` (both honored, capped);
`stop` sequences (deduplicated against the engine's own stop tokens — the
stop-dup fix); `reasoning_effort` accepted with default `medium`;
`tools/logprobs/n>1/response_format` -> 400 with a clear message.

**Think / reasoning_content split**: the Qwen template opens a `<think>`
block per assistant turn. The API strips the reasoning from `content` and
returns it as `reasoning_content` deltas (OpenAI-style); clients that want
the raw stream can pass the template's own controls. `usage` reports
`cached_tokens` (prompt-cache hit) plus the standard counts; the finish
window flushes exactly.

**Session semantics (W1)**: a request HOLDS its engine slot for the full
SSE lifetime — the slot is released only on the finish window, client
disconnect (cancel-on-disconnect arms at the next cycle boundary), or error;
a per-conversation lock serializes same-conversation requests (concurrent
turns on one conversation queue instead of racing the resident state);
mid-stream errors surface as SSE `error` events (never a silent truncation);
the slow-loris and queue-deadline guards bound unproductive waits.

### Conversation pinning and prefix reuse

A conversation stays resident in GPU memory. The client resends the full
history; the server does longest-prefix-match and prefills **only the
delta**:

| turn | FRESH | FOLLOW_UP |
|---|---|---|
| 200-token turn @2k-class | ~6.1 s | **~2.1 s** |
| 200-token turn @100k snapshot | ~10 s prefill class | ~10 s (delta at T=1 rate) |

Extensions: `conversation_id` in the request body (or
`x-conversation-id` header) pins a conversation to its resident engine
state.

FOLLOW_UP = delta `fill_draft` seeded from `dhd_seed` (the committed-position
draft hidden the accept kernel stores) -> trunk state transfer -> T=1 delta
prefill -> spec re-seed (`stseed_spec(nd&1)`). Gate: two-turn RESIDENT vs
FRESH-FULL token streams identical 42/42, cur/pos identical.

Two hard rules learned here: **re-encode, never re-render** (keep a message
mirror; encode only the special-token-boundary tail — re-rendering the full
history through the chat template re-inserts empty `<think>` blocks in prior
turns), and **stop-batch over-commit** (a mid-batch im_end means the next
turn starts FRESH, not FOLLOW_UP).

### Queue and cancellation semantics

- **Queue 1+4**: one active request, FIFO of 4 waiting, `429 + Retry-After`
  beyond that.
- **Cancellation**: `receive()`-watcher + finally-cancel +
  cancel-on-disconnect; takes effect at the next cycle boundary (tens of
  ms). NOTE: on this stack `is_disconnected()` silently fails — the watcher
  on `receive()` is the reliable signal.
- **Bounded cycles**: `max_cycles = max_tokens + 4`, never unbounded (a
  runaway generator once streamed 44 s of post-abort tokens — the API
  runaway law).

### Detokenization

Qwen chat template applied bit-exactly (103/103 tokens vs the reference
implementation). Streaming detok is byte-exact with UTF-8 holdback: the
server accumulates token bytes and never splits a multi-byte character or a
stop sequence across chunk boundaries.

## Part 2 — Operational runbook

### Daemon lifecycle

The daemon MUST boot as `python -u test_w100k.py` with `M1A_SERVE=1` (the
**host-process boot law**: draft acceptance is keyed to the `__main__`
script identity; any other host collapses alpha to garbage while outputs
stay exact — so exactness alone is not a sufficient boot gate, alpha >= 0.85
must be checked).

```sh
# the canonical env (full knob table in GETTING_STARTED.md; this is the same
# line env.canonical carries — LOOKUP_K=10, the shipped K=10 set)
cd engine && env PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH \
  DOCKER_HOST=unix://<colima-socket> DEV=NV M1A_SERVE=1 \
  SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 \
  M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256 \
  LOOKUP_K=10 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 \
  NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 \
  PF_SCAN32=1 PF_M64=1 PF_M128=1 PF_DR7=1 PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=1 \
  PF_SCANC_N2=1 PF_M64QKV=1 PF_ABW=1 PC_ENABLED=1 PF_RING4=1 PF_QKV1=1 \
  PF_P5=1 PF_OP64=1 PF_W4A8=1 NV_SPILL_EXEMPT_NAMES=pf,p8 \
  python -u test_w100k.py &
python api_server.py &     # 127.0.0.1:8080
```

Boot to ready: ~11 s weights (warm cache) + ~40 s KV int8-quantize + warmup +
graphs ≈ 6-7 min at 100k ctx. VRAM at 100352 ctx + KV8 ≈ 17.7 GB; steady
state with the prefill planes ~23.5 GB of 24 GB. `PF_W4A8=1` is the Tier-2
prefill ship — unset it for the byte-identical Tier-1 prefill
(530.6/479.4/328.0 vs 569.2/510.1/342.0), see
[PERFORMANCE.md](PERFORMANCE.md).

### The supervisor (the canonical run mode)

The shipped way to run the service is the launchd supervisor
(`engine/ops/enginectl install`; the one-time env.canonical + plist setup is
in [GETTING_STARTED.md](GETTING_STARTED.md)). What it adds over a manual
boot:

- **The wrapper** (`ops/engine_daemon.sh`) runs the engine python as a child:
  crash-class exits are counted (3 in 10 min -> persistent stay-down marker
  -> refusal, protecting the dext from crash loops); breaker state lives
  under `logs/` (survives the fault-reboots that wipe `/tmp`); the GPU lock
  is pid-liveness-checked and never auto-removed when any GPU process owns
  it.
- **Env single-sourcing (V-27)**: the plists carry NO env dict; the wrapper
  sources `ops/env.canonical` (0600, carries the admin token — generated
  from the published `env.canonical.example`), logs its sha256 digest, and
  the daemon's config fingerprint surfaces in `/health`; the API
  drift-checks it and serves **503 `config_drift`** on mismatch. A launchd
  relaunch can never silently boot a stale env subset.
- **Self-heal across GPU-exit reboots**: `RunAtLoad` + `KeepAlive` bring both
  services back (validated with a kill -9 cycle on the rig).
- `enginectl` subcommands: `status` / `stop` / `restart` / `logs` /
  `install` / `uninstall` / `clear-breaker`.

**Security posture (W2)**: the HTTP API binds 127.0.0.1 only (trusted-host
header enforced, content-type checked, request body capped -> 413/415, the
chat template rendered through a jinja sandbox); `/health` is REDACTED
(liveness + ready + config fingerprint; no paths, no env). The engine socket
is 0600 with peer-uid checking (other local users are refused), and the
privileged RPCs (`shutdown`, `snapshot_save/load`) additionally require the
`TLX_ADMIN_TOKEN` from env.canonical (fail-closed). The tokenizer template
cache is HMAC-integrity-checked (0700; tamper -> clean fallback).

**Boot-time engine tripwires (W4)**: with `NV_GRAPH_ASSERTS=1` (default) the
fork asserts every captured kernel's launch grid against its name-encoded
maxntid and its stack frame against the spill policy — a mismatched boot
fails loudly instead of running miscompiled. The K-rung manifests
(`engine/manifests/manifest_k*.json` + `engine/rung_manifest.py`) assert the
LOADED cubin set at every boot; `engine/w4_census.json` is the 782-cubin
census the policy was calibrated against.

### The batch opt-in (B=2, default OFF)

Two concurrent conversations can share the engine (one M=BT trunk, per-
stream state banks, per-stream bit-exact — the R6 campaign). It is OFF by
default because the honest service-measured aggregate at B=2 is ~1.20x (see
[PERFORMANCE.md](PERFORMANCE.md)); enabling is a documented env block:
LOOKUP_K=7 (the K10 deep-set scratch does not fit with banks), PF_DR7=1,
drop the other PF_* knobs, add `PF_G3M_MB=0 PF_P5=0 R6_PF_T1=1 BATCH_B=2
BATCH_REBUILD_EVERY=232 BATCH_PF_CHUNK=64`. Read
[history/R6_BATCH.md](history/R6_BATCH.md) first — the R6 VRAM law and the
graph-class budget law both bind on batch boots, and 8k FRESH prefill drops
to the T=1 path (~30-45 ms/tok; repeats go through the prompt cache).

### Keepalive and the rebuild-every law

- `M1A_KEEPALIVE_S=10` — socket keepalive.
- `M1A_GEN_REBUILD_EVERY=256` — **the ~950-cycle dext budget**: continuous
  back-to-back 4-graph spec cycles fault the GPU at 850-1025 cycles (6/6
  repro; independent of position/alpha/ring size/duty; idle gaps do NOT
  reset it; interleaved prefill work does). The daemon rebuilds + re-anchors
  graphs every 256 cycles at quiescent points, <1% cost. Never omit this
  knob.
- **Fixed-handle state**: every mutable buffer (KV slabs, GDN rec/conv
  slots, rings, tok_hist at full CTX) is allocated once at boot; the request
  path never reallocs — only windowed uploads (`win_up`) and device
  memsets. Graphs are built once. This is what makes FRESH/FOLLOW_UP resets
  VRAM-flat (verified x5).
- **Snapshots are delta-windowed by law** (rows [base_P0, pos+16) + GDN slot
  4 + seeds; 0.1 s) — a full-KV download is 3.2 GB of host copyouts and
  silently OOM-kills the daemon.

### The P0 stale-feed fix (already in)

FRESH requests once answered the PREVIOUS request's prompt —
deterministically one-request-stale. Root cause was the R2c M128 rung
breaking the prefill graph-set ambient-flag inference (M64-context calls
submitted M32/M128 graphs reading stale id buffers), plus the M32 r>0 tail
re-fed at pos 0; both fixed with explicit m32/m64/m128 graph-set threading
(DEXT_LAWS Q9, `engine/p0_repro.py` for the different-content-prefill
harness). All 7 path shapes (66/65/64/63/130/96/32) verified fresh +
position-exact.

### Engine socket protocol (the RPC under the API)

`{"id":N,"method":...}` newline-delimited JSON on
`/tmp/llm-engine.sock`:

- `status` -> `{ready, ctxk, pos, busy, mode, uptime_s}`
- `prefill {"snapshot":dir}` — park at a bootstrap snapshot
- `prefill {"mode":"FRESH","ids":[...]}` — full prefill from pos 0
- `prefill {"mode":"FOLLOW_UP","ids":[...]}` — delta prefill from resident state
- `prefill {"mode":"AUTO_CACHE","ids":[...],"cache_key":...}` — deepest-chain
  trie restore -> `CACHE_HIT` (cached_tokens=B, M64 tail) or fall to
  FRESH+ingest
- `generate {"max_cycles":K,"stop_token_ids":[...]}` — streams per-cycle
  `{"event":"cycle","cycle":k,"pos":P,"tokens":[m+1 toks]}` then
  `done`/`cancelled`
- `cancel` — side-channel flag, effective at next cycle boundary
- `snapshot_save {"path":dir}` / `snapshot_load` — per-conversation
  (delta-windowed) snapshots
- `shutdown` — drain, synchronize, exit

One cycle emits m+1 tokens (~450 tokens per 60 cycles on the deep-K hit
class; ~167 at K=2); compare emits against `hist[P0:P0+len(emits)]`, never a
fixed window.

## Part 3 — Cache behavior (LongMemory)

Resident conversation reuse only survives while the daemon lives. The
durable prompt cache (`pcache.py`, R1) makes every prefilled context durable
and restart-surviving:

- **Keying**: a content-addressed hash chain over the FED token ids
  (`h_i = sha256(h_{i-1} || ids[64i:64i+64])`), the chain root mixed with
  the engine CONFIG FINGERPRINT (the kernel-set env) — entries are valid
  only under a bit-identical numerics config, by construction. Never keyed
  on rendered text (the RE-ENCODE law).
- **Nodes** (~195 MB per 1024-token node): int8-KV window + scales, the
  DRAFT-KV window (without it a cross-restart restore re-paid fill_draft
  ~255 s @100k), GDN rec/conv states, draft + trunk hiddens, meta. Longest
  COMPLETE chain wins; restore = deepest cached boundary + tail re-prefill
  only.
- **Ingest** at FRESH chunk-quiescent boundaries (GPU download on the daemon
  thread — legal under the ~950-cycle law; decode never pauses) and at clean
  turn ends.
- **Result**: a 100k context restores in **~6.5 s vs ~13 min FRESH**;
  restart-resume 60/60 x4 boots. API surface: `cached_tokens` in the
  response usage, `prompt_cache_key` to pin/share chains.

### tok_hist seeding (R5d)

The deep-K LOOKUP drafter scans the token history; `serve.py` therefore
seeds `tok_hist` on the boot/FRESH/FOLLOW_UP/CACHE_HIT/snapshot_load paths.
Without seeding, the first FRESH generate self-matches a -1 prefix -> -1
drings -> OOB fault (never exercised before deep-K shipped).

## Known issues (the honest ledger — W5 findings, live-rig)

1. **CACHE_HIT continuations from midprefill/turnend pcache nodes are not
   bit-exact vs FRESH** (pre-existing since the R2c M128 trunk; FIX_CAMPAIGN
   finding #5). They are coherent (alpha ~3.5) and deterministic, and the
   boot-node restart-resume path IS exact — but Tier-2-drift vs FRESH. The
   hot path is unaffected: FOLLOW_UP (same-conversation turns) is
   pcache-free and exact. Fix path = the R6 T1-boundary node class.
2. **Stop semantics under launchd** (finding #7): an `enginectl stop engine`
   that falls to the RPC path is effectively a RESTART on this platform —
   the GPU-EXIT reboot law fires seconds after the engine exits, and
   RunAtLoad heals the daemon back (~5 min). A true stop needs the
   disable-then-bootout launchd sequence (open ops item). The staydown
   marker is typically preempted by the reboot.
3. **Batch-config exactness through cache paths** (finding #8): on the
   BATCH_B=2 opt-in, solo FRESH streams are bit-exact but CACHE_HIT-derived
   batch phases fail the bit-exactness gate (finding #5's class extended to
   the batch config). Batch mechanics themselves (concurrency,
   determinism, slot hygiene) are healthy.
4. **The kernargs-slab leak** (R6 P3): every graph build allocates a
   host-mapped kernargs slab that is never released; long-running batch
   service mitigates with rotated fences + non-fatal failed fences. Durable
   fix (ka-slab reuse) is a documented follow-up.

## Not yet (M2/M3)

Sampling kernel (temp>0; distribution-exact contract), context-cap config
knobs, snapshot persistence polish; Anthropic text-only adapter (thin,
outside the engine process). The M3 TTTS fix — batched prefill — landed as
the P-series + R2-series + P8 + T2 campaigns
([history/PREFILL.md](history/PREFILL.md)): FRESH 8k prompts went ~106 ->
479.4 (Tier-1) / 510.1 (T2 W4A8) tok/s class, making FRESH 8k ~16 s and
FRESH 100k ~4.8 min (was ~13 min at P18).
