# M1-A — engine0 serving core (DecodeSession, fixed handles, resets, reuse, daemon)

Mission: SERVING_PLAN.md "M1-A engine". Status: STEPS 1-4 implemented; GATES 1-3 PASS;
GATE 4 mostly PASS (generate 60/60 exact @ 40.9 tok/s over the socket, cancel, FOLLOW-UP,
save/load, graceful shutdown) with the open issues listed under GATE 4.

## What changed (engine0/)

- **`mtp.py`**
  - `DecodeSession`: one spec cycle per `step()` — submits draft_g → probe_g →
    accept_g → flush_g timeline-chained on a carried `prev`, waits the cycle signal,
    reads the 32B emit record via `Bufs.down_at("emit", 0, 8)`. Returns
    `{pos_new, m, cycle, stop, tokens[0..m]}`. `begin()` re-anchors `prev` after any
    eager work. Cancellation = stop calling `step()` (device quiescent at return).
  - New fixed-handle buffers: `emit[8]` (outbox), `dhd_seed[5120]` (committed-position
    draft hidden), `fillval`/`filln` (mfill staging).
  - State machine (NO `P.up` anywhere in the request path — P.up reallocs = stale
    graph kernargs + orphan VRAM, the ~755MB class):
    - `reset_fresh(cur0)`: mfill rec4/conv4 ← 0 (fresh GDN state IS zeros), tok_hist
      ← −1, m_hist ← 0, win_up all small slots. kv/kv_d/sc untouched (prefill
      overwrites every row it later reads).
    - `reset_snapshot(snapdir, cur0, p0)`: mfill rec4/conv4 ← 7.7e31 poison (proven
      tier-1 semantics), window-upload slot 4 from bootstrap npys, reset slots.
      Graphs stay valid — no rebuild, no realloc.
    - `load_snapshot_kv(snapdir)`: boot-time prompt KV (fp16 npys → int8+scale when
      KV8) win_up'd into the boot-allocated slabs.
    - `stload_trunk()` / `stseed_spec(par)`: GDN state xfer rec4/conv4 slot 4 ↔ trunk
      `rec{i}`/`conv{i}_par` (new `stxrec`/`stxconv` cubins; grid 1×256, one block
      per launch; 96 eager launches, sync every 64).
    - `prefill_t1(G, ids)`: T=1 trunk prefill from resident pos_slot (win_up
      tok_slot per token; submit parity graph + wait — the proven wait_each pattern).
    - `follow_up(G, delta_ids)`: conversation reuse = delta `fill_draft` seeded from
      `dhd_seed` → `stload_trunk` → `prefill_t1([cur_slot]+delta)` → `stseed_spec(nd&1)`
      → slot resets. Returns (new_cur, pos_new, n_fed).
  - `fill_draft(ids, start_pos=0, seed_hd=None)`: fixed-handle (win_up only — safe
    after build_graphs); delta fill + draft-chain seeding for FOLLOW-UP.
  - `run_cycles(ncyc)` non-phase path now drives DecodeSession.
- **`accept.cu`**: + `emit[8]` outbox `{pos_new, m, tok0..2, stop_flag(0), cyc, rsv}`
  and `dhd_seed[i] = (m==0 ? hd_d0 : hd_d1)[i]` (for m=2 the draft never fed p2, so
  hd_d1 is one step stale — draft is heuristic, no exactness contract). Every legacy
  write bit-identical. Rebuilt per-kernel cubin, STT_FUNC verified.
- **`engine0.py` `Bufs`**: + `win_up(name, off, arr)` / `down_at(name, off, n, dtype)`
  windowed upload/readback into EXISTING buffers.
- **`trunk.py`**: `CTX = SKV_CTXK` (default 2304) — all mutable state allocated once
  at boot at full engine ctx with the RUN dtype (KV8 → uint8 kv + fp16 sc slabs);
  tok_hist CTX+256. The 2048-poison-then-realloc is gone.
- **`test_w100k.py`**: drives DecodeSession everywhere; load/reset fixed-handle;
  graphs built ONCE after fill_draft (were rebuilt every rep — kernargs-slab orphan
  per reset). M1A_SERVE=1 attaches the daemon after build_graphs (see law below).
- **`serve.py`**: daemon LIBRARY (no engine imports; attaches to the test_w100k host
  process). **`gate4_driver.py`/``gate4b_driver.py`/`m1a_persist.py`**: scripted
  socket drivers.
- New cubins `mfill` (device memset; value/count in buffers — no scalar args),
  `stxrec`/`stxconv` (GDN state copies). All 256-thread (no warp-count name token),
  eager-only, never in graphs.

## GATE RESULTS

- **GATE 1 PASS** (canonical `SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1
  KV8=1 QH=1 PVH=1 HM=1 python -u test_w100k.py`, run TWICE, ~2h apart): Tier-1
  60/60 bit-exact ×2 real reps, deterministic; emit record verified == tok_hist
  stream; 3 timing reps 40.30/40.36/40.21 tok/s (BEST 40.36 vs W2H canonical 40.35 —
  regression-free); phase draft 5.27 / probe 62.60 / accept 1.15 ms; T=1 ref
  45.78 ms/tok = 21.84 tok/s; stock cross-check 59/59.
- **GATE 2(a) PASS**: Tier-1 after FRESH reset ×2 — 60/60 exact both, emit==hist.
  **GATE 2(b) PASS**: reset path ×5 consecutive — VRAM perfectly flat (1134 allocs /
  71.750 GB arithmetic sum identical every rep, RSS 0.41GB flat, no OOM), both reset
  kinds interleaved with real 2-cycle decodes.
- **GATE 3 PASS** (two-turn conversation reuse, the product feature): RESIDENT
  (decode 60 cycles → FOLLOW-UP [cur + 200-token delta] → decode 40 cycles) vs
  FRESH-FULL (one-shot T=1 prefill of the identical token stream → decode 40):
  turn-2 token streams identical 42/42, cur identical, pos identical (98113).
  Delta-prefill timing: **10.1s for a 200-token follow-up at the 100k snapshot**
  (≈50 ms/fed-token = 0.6s draft-delta-fill + ~46ms/tok T=1 + state xfers;
  2k-class ≈9s at the same T=1 rate).
- **GATE 4 mostly PASS** (socket daemon, measured end-to-end):
  - status / health-probe boot ✓
  - prefill{snapshot} FRESH at pos 97810 ✓
  - **generate 60: 60/60 exact, 167 tokens / 60 cycles = 40.9 tok/s streamed
    per-cycle over the socket** ✓
  - FOLLOW-UP prefill (201 fed, 0.6s draft-delta) + turn-2 generate ✓ ran clean
    (note: my driver's turn-2 "0/40 vs gate3" comparison was invalid — gate3's
    reference came from a 60-token turn-1 (collapsed-draft era) while the daemon's
    turn-1 is 167 tokens; different conversations. The two-turn exactness property
    is owned by GATE 3.)
  - cancel mid-generate ✓ (cancelled at cycle 6, ≤~70ms latency)
  - cancel-then-new-FRESH-request ✓ clean + exact
  - snapshot_save/load (DELTA-window: kv rows [P0, pos+16) + rec4/conv4 slot 4 +
    h_seed/dhd_seed; 0.1s) ✓; load → regenerate deterministic resume ✓
  - graceful shutdown via RPC ✓ (device synchronized, process exits, no wedge)
  - **KNOWN ISSUES (open, M1-B entry tasks)**:
    1. A SECOND consecutive FOLLOW-UP after a save/load cycle killed the daemon +
       driver silently (no traceback, no macOS crash report, no device lock, GPU
       healthy after). Repro: park → gen → save → load → gen → FOLLOW-UP ×2 → dies
       during the 2nd follow_up's trunk-prefill stage. Suspected dext/fork native
       class.
    2. First-RPC-after-idle hang: a snapshot_load issued ~15 min after the daemon
       went idle parked hung the engine inside dev.synchronize() (process alive,
       0% CPU, no crash; the same load ran fine seconds after prior activity in
       gate4b). Serving needs a keepalive/first-submit-after-idle fix (M1-B).
       NOTE: a daemon in this state is left RUNNING (never pkill live GPU python);
       the machine may need a cold reboot before the next GPU session.
    3. Relaunch-persistence check therefore INCOMPLETE (blocked by #2): save/load +
       deterministic resume are proven in-process (gate4b A3); cross-process reload
       verified only up to the load hang.

## NEW LAWS (LAW-grade, from this session)

1. **THE HOST-PROCESS BOOT LAW (biggest finding)**: draft acceptance is determined
   by WHICH SCRIPT IS `__main__`, not by the calls it makes. `python -u test_w100k.py`
   boots give alpha 0.892 / 2.78 tok/cyc / 40 tok/s. ANY other host (hand-rolled
   boot with byte-identical calls, the same file via runpy from another __main__,
   SKV_K present or absent, split-KV scratch pre-zeroed, T=1 warmup of 3/60 tokens
   before or after fill_draft) collapses the DRAFT to garbage proposals
   (dring0 = 0/45918-class, m=0 every cycle) — while the PROBE stays bit-exact
   (outputs 60/60, only speed halves to ~22 tok/s). Eliminated: env vars, kernel
   selection, scratch contents, execution order, host readbacks, allocation
   differences. Correlates only with argv[0]/__main__ identity. UNRESOLVED —
   suspected address/layout-dependent dext or kernel behavior tied to process
   image. CONSEQUENCE: the daemon MUST be launched as
   `M1A_SERVE=1 ... python -u test_w100k.py` (serve.py is a library attaching at
   test_w100k's handoff gate, after build_graphs). ANY boot change must re-check
   alpha ≥ 0.85 — exactness alone is NOT a sufficient gate.
2. **emit-record semantics**: one cycle emits m+1 tokens; N cycles emit sum(m+1)
   (60 cycles ≈ 167 tokens at alpha 0.89). Compare emits against
   hist[P0:P0+len(emits)], never a fixed window.
3. **snapshot_save must be DELTA-windowed**: the full-KV download is 16×~200MB =
   3.2GB of host copyouts → silent OOM-kill of the daemon. Save only rows
   [base_P0, pos+16) and reload = reset_snapshot(base) + delta win_ups (0.1s save).
4. Health-probe arithmetic: dposadd computes dst=src+1 — expect 124 when seeding 123.
5. VRAM @100352+KV8 ≈ 17.7GB live (weights 12.8 + kv 3.3 + rec4 0.76 + misc);
   resets allocate nothing (verified flat ×5).

## SOCKET PROTOCOL (/tmp/llm-engine.sock, newline-delimited JSON)

- `{"id":N,"method":"status"}` → `{"ok":true,"result":{ready, ctxk, pos, busy, mode, uptime_s}}`
- `{"id":N,"method":"prefill","params":{"snapshot":dir}}` — park at a bootstrap-layout
  snapshot (kv must be the boot-loaded base).
- `{"id":N,"method":"prefill","params":{"mode":"FRESH","ids":[...]}}` — full T=1
  prefill from pos 0 (draft fill + stseed).
- `{"id":N,"method":"prefill","params":{"mode":"FOLLOW_UP","ids":[...]}}` — delta
  prefill from resident state (feeds [cur_slot]+ids; ~50ms/token at 100k).
- `{"id":N,"method":"generate","params":{"max_cycles":K,"stop_token_ids":[...]}}` →
  per-cycle `{"event":"cycle","cycle":k,"pos":P,"tokens":[m+1 toks]}` then
  `{"event":"done",...}` / `{"event":"cancelled",...}`. Host stop-token check per cycle.
- `{"id":N,"method":"cancel"}` — side-channel flag (no ack write; takes effect at the
  next cycle boundary, ≤~70ms). Same connection while generating.
- `{"id":N,"method":"snapshot_save","params":{"path":dir}}` /
  `{"method":"snapshot_load"}` — delta-window per-conversation snapshots.
- `{"id":N,"method":"shutdown"}` — finish cycle → dev.synchronize() → exit.
Single client at a time; listener thread only enqueues + sets the cancel flag;
main loop blocks on queue.get() when idle (NO GPU spin).

## VRAM / perf quick reference

- Canonical decode: 68.97-69.23 ms/cyc, 2.78 tok/cyc, ~40.1-40.4 tok/s @100352 ctx.
- T=1 prefill: ~45ms/tok @100k (21.8-23.0 tok/s) → 200-token follow-up ≈10s.
- Boot: ~11s weights (warm page cache) + ~40s KV int8-quantize + 60s T=1 warmup +
  255s fill_draft + graphs ≈ 6-7 min to ready.
