# M1-C — the gate-storm device fault: root cause, laws, fixes, gates

Mission (from M1B_SERVING.md open issue): make the FULL api_gates mix pass
repeatedly; the mix faulted the dext within minutes (2/2 in M1-B).

## TL;DR

The fault was NOT the API layer, NOT the cancel path, NOT concurrency, NOT the
cmdq ring, NOT thermal, NOT position. It is a **dext-side per-cycle budget in
the spec-decode generate loop**: a single continuous run of back-to-back
4-graph cycles faults the GPU at **850-1025 cycles** (6/6 deaths), regardless
of position, alpha, ring size, or idle gaps. Two independent serving bugs
amplified exposure to it, and one adjacent OOB bug sat right on top. All three
are fixed; the budget is worked around by **periodic graph rebuilds**
(the ~950-CYCLE LAW below).

## The evidence chain (all logs persistent under ~/tinygrad-metal/logs/)

1. **/tmp is wiped on every fault-reboot** (GPU-EXIT/REBOOT LAW turns every
   fault into a reboot) — all prior forensics were destroyed. FIRST fix:
   serve.py now mirrors its structured log to
   `~/tinygrad-metal/logs/m1a_serve.log` (REBOOT-SURVIVOR LOG LAW) and
   per-cycle timeline markers are available via M1A_CYCLE_SLOG=1.
2. Run 2 (full api_gates, instrumented): gate (d)'s 4000-token stream aborted
   client-side at 4.0s — but the engine kept generating for **44 more seconds**
   (868 cycles) and died EXACTLY when the runaway crossed the budget. The
   M1-B "fault 56s after a cancelled stream" observation was this runaway.
3. Engine-direct bisect (repro.py, no API server): fresh-loop / cancel-long /
   cancel-next / cancel-close / gen-stop — **33 sessions, ~990 cycles total,
   ZERO faults** (each session is short and interleaved with prefills).
4. **Pure long generate (longgen.py = prefill + generate(3000), no API, no
   cancel, no disconnect) faults 6/6** at cycles 868/900/945/971/1025/~850
   (pos 956-1035 in short-context runs; pos ~100200 in the snapshot-park run —
   position-INDEPENDENT; healthy-alpha run died the same — alpha-INDEPENDENT).
5. Exonerated by direct test: cmdq ring size (64MB boot died at the same
   cycle count; per-graph submit = 48 bytes — 2MB holds ~8000 cycles);
   thermal/power (100ms idle per cycle = 65% duty reduction died at the same
   BUSY total ~49.5s); idle-reset (250ms quiesce every 25 cycles — same);
   gpfifo entries (65536; ~5 submits/cycle).
6. The fault surface: `wait(vf)` in DecodeSession.step hangs > 200ms, the
   host sleep-poll then sees `dev_impl.is_err_state` -> RuntimeError('Device
   fault detected') (hcq.py:311 -> ops_nv.py:608). The dext flagged the
   channel; the process exits (fault-exit), the machine reboots seconds later.
   In 5/6 deaths the process died before any handler could log (native class).
7. Interleaved prefills reset/clear whatever accumulates (evidence #3 vs #4).

## The three bugs fixed

### 1. THE RUNAWAY (api_server.py) — disconnects never cancelled the engine
- `request.is_disconnected()` (the anyio-cancelled receive trick) is
  UNRELIABLE on this stack: a client that closed mid-stream was not seen for
  44+s. `gen()` also had no GeneratorExit handler, so an abandoned stream
  never armed the engine cancel. And `max_cycles` was 100000.
- FIX (three layers):
  a. a watcher task OWNS `request.receive()` and arms `cancel_flag` on
     http.disconnect (deterministic);
  b. `gen()`'s finally sets `cancel_flag` on early close/GeneratorExit;
  c. **BOUNDED-CYCLE LAW**: the engine is only ever asked for
     `max_cycles = max_tokens + 4` (each cycle emits >= 1 token).

### 2. m_hist[1024] OOB (mtp.py) — deterministic corruption at cycle 1025
- `accept.cu`: `m_hist[cyc_slot[0]] = m; cyc_slot[0]++` with m_hist sized
  1024 — any single generate reaching cycle 1025 writes 4 bytes past the
  buffer into an adjacent allocation -> next cycle faults. (longgen3 died
  exactly at 1025 for this reason.) m_hist is write-only (no readers).
- FIX: m_hist enlarged to 1<<20 int32 (4MB; max_cycles can never exceed ctx).
  No kernel change.

### 3. THE ~950-CYCLE BUDGET (dext-side; worked around, not root-caused)
- A single continuous generate faults at 850-1025 cycles (see evidence).
  What exactly accumulates in the dext across graph-cycle re-submissions is
  not visible from userspace; idle gaps, syncs, and ring size do not reset it;
  prefill-class work between generates does.
- WORKAROUND (M1A_GEN_REBUILD_EVERY=256, default ON in the daemon recipe):
  every 256 cycles at the quiescent point, `E.build_graphs()` +
  `sess.begin()` — fresh queue objects + kernargs (the same work-class as an
  interleaved prefill). Cost ~0.1-0.2s per 256 cycles (<1%). Verified:
  3000-cycle generates clean x2 (12 rebuilds each), then the full gate suite.

## NEW LAWS (M1-C)

1. **REBOOT-SURVIVOR LOG LAW**: fault forensics MUST go to a persistent path —
   /tmp dies with every fault-induced reboot. (serve.py writes both.)
2. **BOUNDED-CYCLE LAW**: never request unbounded engine generation; cap
   cycles at max_tokens + 4.
3. **~950-CYCLE LAW**: no more than ~500 continuous spec cycles without a
   graph-rebuild (or prefill-class interleave). Enforced by
   M1A_GEN_REBUILD_EVERY (256). DO NOT SET IT TO 0 for long generations.
4. **STOP-BATCH OVER-COMMIT LAW**: a stop-token firing mid-batch feeds the
   engine up to 2 tokens the client will never render -> the next turn can
   never exactly extend the fed stream -> deterministic FRESH fallback (not a
   bug; a protocol consequence — a truncate/rewind RPC would be needed to
   reclaim it). Length-capped turns keep client-visible == engine-fed exactly
   and DO get FOLLOW_UP.
5. (observation) the emit copyout host-staging read waits `timeline_value - 1`
   (upstream hcq.py `_copyout`) — an off-by-one-class race window vs the DMA
   in flight; never bit us at 32B, but do not read large buffers this way.

## Secondary fixes (this session)

- **stream-tail drop**: gen() broke on `fut.done()` while text events were
  still queued behind the 20ms poll -> nondeterministic truncated stream
  text (the gate (b) flake). Fixed by draining evq after fut completes.
- **gate (c) rewritten deterministically**: (c) after a mid-batch im_end stop
  the next turn MUST be FRESH (law #4); (c2) a length-capped conversation
  MUST take FOLLOW_UP. Both directions asserted.
- **THE RE-ENCODE LAW (conversation reuse)**: a full-conversation re-render
  can NEVER reproduce the ids the engine was fed, for two independent
  reasons: (a) the model's own token splits are not BPE-canonical (e.g. the
  template's "\n" + the model's " need" re-encodes as one merged token);
  (b) the Qwen template re-inserts an EMPTY `<think>\n\n</think>\n\n` block
  for historical assistant messages (the generation prompt ends with an OPEN
  `<think>\n`). Correct construction (implemented): mirror the MESSAGE
  HISTORY; next turn, render the mirror WITHOUT the generation prompt as the
  text-prefix check, then `ids2 = fed_mirror + TOK.encode(render tail)` —
  the tail begins at a special-token boundary so its standalone encode is
  exact. FOLLOW_UP also requires engine fed_len == mirror length.
- **cancel-latency cycles**: after a client-side bail, the engine can
  complete one more submit before observing the cancel flag. run_chat now
  reads the terminal event (done/cancelled, which carries the engine-side
  full token list) and mirrors those extras; on LENGTH caps they are fed
  into the client-visible reply too (mirror == fed == client exactly ->
  turn stays reusable); on stop-token/abandon they mark the conversation
  non-reusable (FRESH fallback).

## GATES (transcripts under ~/logs/ on the remote)

- minimal repro (longgen 3000, was 6/6 faults): **x2 clean** with
  M1A_GEN_REBUILD_EVERY=256 (12+ rebuilds each, first-ever 3000-cycle gen)
- engine-direct repro ladder (repro.py, 5 modes): 33 sessions clean
- **full api_gates.py x3 consecutive: PASS x3** (all a-f incl. the rewritten
  deterministic c/c2, cancel latency 2.7s, 6-way queue)
- real-client 2-turn smoke (+streamed turn-3): **PASS** — turn-2 FOLLOW_UP
  2.1s (vs 6.1s FRESH), 19 follow_up ops in the engine log
- **15-minute soak: PASS — 32 rounds, faults=0, engine uptime 2565s** (43
  min continuous service: 32 non-stream chats + 32 aborted streams + 32x3
  queued bursts + idle gaps; keepalive ran throughout; zero engine restarts)
- **canonical Tier-1 re-verify (non-serve boot after the m_hist change):
  60/60 exact x2 (167 emitted, pos_end 97977), tier-2 cross-checks 60/60 vs
  W2D fp16-KV and W2E int8-KV sequences, stock 59/59, T=1 ref 21.89 tok/s
  — the engine did NOT drift.**

## R6 PHASE 3 SHIP UPDATE (09-24): batch-capable serving, DEFAULT-OFF

The daemon and API are batch-capable end-to-end (BATCH_B=2: two concurrent
conversations, per-stream bit-exact — every exactness gate green; the W1/W2
surface unchanged at BATCH_B=1, which stays the canonical default). Enabling
is a documented opt-in (R6_BATCH.md "THE BATCH CANONIAL" section): the R6 VRAM
law knobs + LOOKUP_K=7 + BATCH_B=2. Read R6_BATCH.md Phase-3 section before
flipping: the honest perf tables (0.76x/1.13x vs solo through the service),
the GRAPH-CLASS PREFILL BUDGET LAW (T=1 prefills do NOT reset the ~950-cycle
budget — only the rotated fence-all does), and the kernargs-slab leak status.
Admin-token shutdown: unchanged (enginectl stop engine).

**R8 SHIP UPDATE (09-23)**: the canonical daemon env now carries LOOKUP_K=10 (the 75.56 tok/s K=10 deep set — R8_DECODE.md; decode Tier-1-exact, prefill unchanged at 569.2/510.1/342.0).

**R5d SHIP UPDATE**: the canonical daemon env now includes LOOKUP_K=7 (the
63.11 tok/s Tier-1-exact deep-K config; R5_DEEPK.md section 8). Requires the
serve.py seed_hist flip (landed R5d) — never run a LOOKUP daemon without it.

**T2 P8w4 SHIP UPDATE (2026-09-23)**: the daemon env now includes PF_W4A8=1 —
the Tier-2 W4A8 prefill ffn (packed7-reading IMMA, p8q8x + p8w4ffn7; prefill
2k 569.2 / 8k 510.1 / 100k 342.0 tok/s; decode stays Tier-1 bit-exact 71.8).
PREFILL numerics are Tier-2 (linearized-codebook, 2.4% logits class; battery
+ banks in T2_P8W4.md). Kill-switch: unset PF_W4A8 -> the byte-identical
Tier-1 prefill (verified line-for-line). pcache-namespaced via _ENV_KEYS.

## How to run the service (one-liner)

Engine: `cd ~/tinygrad-metal/engine0 && env PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
DOCKER_HOST=unix://~/.colima/default/docker.sock DEV=NV M1A_SERVE=1
SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1
M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256 LOOKUP_K=10 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1 PF_M128=1 PF_DR7=1 PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=1 PF_SCANC_N2=1 PF_M64QKV=1 PF_ABW=1 PC_ENABLED=1 PF_RING4=1 PF_QKV1=1 PF_P5=1 PF_OP64=1 PF_W4A8=1 ~/tg311/bin/python -u test_w100k.py
> /tmp/serve_m1c.log 2>&1 &` then API: `cd ~/tinygrad-metal/engine0 &&
python3 api_server.py > /tmp/api_m1c.log 2>&1 &` (health on 127.0.0.1:8080).
With launchd installed: `engine0/ops/enginectl install` then it self-heals.

## W5 SHIP UPDATE (2026-09-24): SUPERVISOR MODE IS THE CANONICAL RUN MODE

The TLX review-fix campaign (five waves, FIX_CAMPAIGN.md) is merged on main
and live-validated. The shipped mode is now the launchd supervisor:

- **Run**: `engine0/ops/enginectl install` (plists com.tlx.llm-engine /
  com.tlx.llm-api, UserName=%USER%, NO env dict — the wrapper sources
  `ops/env.canonical`, digest logged at every start; config_fp
  e91605105e1a34c0 surfaces in /health and the API drift-checks it).
  Manual runs of `ops/engine_daemon.sh` remain equivalent (slower boot under
  launchd's Background ProcessType: ~14 min vs ~8).
- **Env changes**: ONLY via ops/env.canonical. W5 added
  `NV_SPILL_EXEMPT_NAMES=pf,p8` (the P18 frame-policy scoping — the pf/p8
  Tier-2 prefill families carry 104-592B ptxas FRAMES by design; the hard
  100B law stays for the decode/canon set, max 32B).
- **Expected boot warns** (NV_GRAPH_ASSERTS=1 default): op38nw32_3 8B,
  spk_g4nw32hm11_100k 8B, spk_pre11qh_100k 32B, pfa32ct 104B (exempt class,
  fires at first real-size prefill). k2_scan 8B never warns (legacy
  eager-trunk kernel, not launched canonically). ANY OTHER [NV-GA] line or a
  hard trip = investigate.
- **Tier-1 decode re-verified through the new stack**: 60/60 x2 det, stock
  59/59, 75.81 tok/s BEST (R8 bank 75.56), E[m|deep]=10.000. api_gates 25/25.
- **STOP SEMANTICS (the honest caveat)**: every engine stop (RPC shutdown,
  SIGTERM, kill) is followed by the GPU-EXIT reboot, and launchd RunAtLoad
  heals the daemon back within ~5 min — an RPC shutdown is effectively a
  RESTART. The staydown marker is typically preempted by the reboot (never
  observed writing). A true stop needs disable+bootout sequencing (open ops
  item, FIX_CAMPAIGN.md finding #7). The breaker counts only exits that do
  not take the machine down (finding #6).
- **Known Tier-2 surface (FIX_CAMPAIGN.md finding #5)**: CACHE_HIT
  continuations from midprefill/turnend pcache nodes are coherent but not
  bit-exact vs FRESH on the current trunk (pre-existing since the R2c M128
  trunk; deterministic; G4 boot-node restart-resume IS exact; FOLLOW_UP is
  pcache-free and exact). Fix path = the R6 T1-boundary node class.
