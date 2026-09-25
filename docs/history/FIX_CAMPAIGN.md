# TLX REVIEW-FIX CAMPAIGN — FIVE-WAVE RECORD + W5 LIVE VALIDATION (ACCEPTANCE)

2026-09-23/24. Branch `review-fixes-w1` (W1 c722a99..639ba35, W2 0d274c8..a1e77bf,
R6-Phase3 3ebbc59..3b9d80d, W3 96c5ac9..175481b, W4 2816c07; fork 03cfc8d) merged to
main (fast-forward from cb0ad21), then LIVE-VALIDATED on the rig in the W5 GPU window.
Ledger: TLX_REVIEW_LEDGER.md on the rig (60 findings, 3 P0 / 23 P1 / 25 P2 / 9 P3 —
the review working file is not part of the published repo; every finding's fix,
or its open status, is visible in this record and the wave commits).

## THE WAVES

- **W1 serving correctness** (V-01..V-23): SSE FIFO slot ownership, mode-aware
  headroom (the 400-lockout kill), per-conv lock, request-scoped cancel,
  stop-dup, think-suppression, max_completion_tokens, dirty-flag force-FRESH,
  SSE error events, non-stream watchers. Mock harness (engine0/tests/) built
  as the proving instrument.
- **W2 security/ops** (V-24..V-38): socket 0600 + peer-uid + admin-token ACL +
  RPC caps, tokenizer cache HMAC under ~/Library/Caches, jinja sandbox +
  HTTP hardening, persistent-breaker supervisor wrapper (engine_daemon.sh),
  env.canonical single-sourcing + config_fp drift 503, enginectl without
  GPU-pkill.
- **W3 pcache hardening** (V-39..V-47): fsync+sha256 durability, transactional
  validate-before-upload restore, lock-held protect + graveyard evict, orphan
  reap, put_nowait/drop backpressure, pin budgets + TTL, manifest hkey trust.
  Root-caused the G1 cur=0 law (M128 xA64-row-63 poison) and added the
  node-hlast-sane gate.
- **W4 engine tripwires** (V-48..V-59): exec-path asserts (launch-vs-maxntid,
  frame policy), timeline-wait deadline, K-suffix rung manifests + audits,
  h_embed token clamp + dring zero-init, gcycle law guards, 782-cubin census.
- **W5 (this document)**: the live GPU validation window — merge, boot through
  the sanctioned path, full battery, supervisor install, batch smoke, soak.

## W5 VALIDATION RECORD (all numbers from the rig, 2026-09-24)

Preflight: **85/85 GPU-free tests green** (46 api-mocks + 16 pcache + 23 w4 —
the 23rd is the W5 dr7-replacement law test). Merge: fast-forward cb0ad21 →
2816c07 (zero conflicts).

**Tier-1 decode @100k, canonical (LOOKUP_K=10)** — the campaign's central
claim (fix-waves + fork tripwires changed ZERO numerics):
- [tier1] rep0 60/60 exact, rep1 60/60 exact, deterministic across reps, emit==hist.
- [stock] T=1[0:59] vs spec_base_100k[1:60]: **59/59**.
- [tier2] vs W2D fp16-KV 60/60, vs W2E int8-KV 60/60.
- [time] 75.81 / 75.80 / 75.51 tok/s (BEST 75.81; R8 bank 75.56). Phase split
  draft 5.33 / probe 59.46 / accept 1.16 ms — R8-identical within noise.
- [deepk] E[m|deep]=10.000 (ALL-TEN), deep-selected 76.7%; deep=off 60/60.

**Tripwire audit (canonical boot)**: `[NV-GA] warn` for op38nw32_3 8B,
spk_g4nw32hm11_100k 8B, spk_pre11qh_100k 32B, + (at first real-size prefill)
pfa32ct 104B warn-only (exempt class). k2_scan 8B NEVER WARNS — the legacy
eager-trunk kernel does not launch on the canonical path (explained deviation
vs the census set). Zero hard trips.

**api_gates: 25/25 PASS** on the new stack (health/fp, (a)-(f) incl. the
over-commit FRESH fallback — after finding #3's fix).

**PC_GATE**: G0 10/10; **G4 100k restart-resume EXACT** (cur=4471, decode
60/60, restore 6.8s). G1/G2/G3 = finding #5 below (FAIL, root-caused).

**Security smokes**: admin RPC without/with-wrong token → refused; status ok.
`sudo -u nobody` socket connect → **REFUSED** (0600 perms layer).

**Supervisor (plists installed, UserName=%USER%, no env dict)**: launchd sees
both services; state paths under ~/tinygrad-metal/logs/; the kill -9 cycle →
GPU-EXIT reboot → **launchd auto-relaunched cleanly** (heal proven); breaker
clear throughout.

**Batch opt-in smoke (env.canonical's documented block, BATCH_B=2)**: boots
through the wrapper (park 5.3ms/tok — no T1-park disaster); /health batch_b=2
fp 7983187bce168adb == expected (drift-check adapts); batch-boot warns add
only spk_pre8qh_100k 32B (the M=8 deep set — expected class). r6_gates spot
(30-cycle class): solo FRESH streams PASS; both slots generating mid-run
(True,True); deterministic ×2; concurrent 38.5 + 40.9 tok/s = **79.40
aggregate (1.20x solo)** — under the 1.5x bar, consistent with the Phase-3
honest table. CACHE_HIT-derived phases (2 and 3) FAIL bit-exactness —
finding #5/#8.

**Soak**: see M1C_STABILITY.md W5 section (15-min interleaved on the final
resting config).

## W5 FINDINGS (live-rig, honest ledger)

1. **[FIXED 510c1a3] rung-manifest DR7 class-swap** — the K=10 canonical boot
   HARD-BLOCKED at assert_rung_wiring demanding base ffn8v11/down8nw32_11 that
   the DR7 loader never loads (M9/10/11 carry only the r7 twins). Fix:
   dr7_cu replaces the ffn8*/down8* trunk class (assert-side only). The W4
   battery's _k10_world modeled the manifest itself, not the loader.
2. **[FIXED 9ad662c] role-key vs program-name union** — the deep spk set loads
   under ROLE keys (pr["spk_pre11"]); the manifest matches cubin NAMES. The
   assert call now passes keys ∪ names.
3. **[FIXED fork 7f3ae82 + 525cde3] pfa32ct 104B frame hard-trip** — the
   canonical FRESH prefill 503'd at the W4 frame-policy line: MIN_STACK_SIZE
   is a FRAME metric, not pure spill; the shipped pf*/p8* families run
   104-592B frames BY DESIGN (P7E7-class maturity gates; the P18 nondet
   datapoint was a 276B SPILL kernel on the decode path). Fix:
   NV_SPILL_EXEMPT_NAMES=pf,p8 (name-scoped, warn-only) in env.canonical +
   fork; the hard 100B law stays for decode/canon (max 32B). The W4 census's
   pf-set missed pfa32c*/pfaw* (flags pf=False) — fixed, census regenerated.
4. **[FIXED 3b0bafe + f86a985] PC_GATE harness races** — W3's async writer
   (put_nowait + fsync) raced the gate's manifest reads/lookups (KeyError
   last_hkey; empty-chain IndexErrors). pc.flush() drains added. The mock
   battery wrote synchronously and never saw this.
5. **[OPEN — the G1 verdict] midprefill/turnend CACHE_HIT restore is NOT
   bit-exact on the current trunk.** cur is CORRECT post-W3 (restored ==
   cur_doc == 9956; the old gate compared vs the DIRTY doc's argmax —
   reference bug fixed) and restores are deterministic (out2==out1), but
   decode diverges from token 0 (72/217 = corpus coincidence) EVEN WITH a
   manual cur override; the tail arm (explicit cur) and G2's follow_up arms
   corroborate; the fingerprint (fp-tolerance) passes. BISECT vs pre-W3
   pcache (cb0ad21): pre-W3 is WORSE (hlast poison 7.7e31, cur=0, alpha 0)
   and the residual 72/217 signature is IDENTICAL → NOT a W3 regression;
   the trunk→spec-slot boundary-state equivalence broke with the R2c-era
   M128/DR7 trunk. W3's hlast fix landed half the distance. G4 (boot-node,
   captured from the parked spec state) restores EXACT. Fix path = the R6
   Phase-3 T1-boundary node class (explicit cur/dhd/hlast). Serving impact
   on canonical: CACHE_HIT continuations are coherent (alpha ~3.5) but
   Tier-2-drift vs FRESH; FOLLOW_UP (the hot path) is pcache-free and exact.
6. **[NOTED] GPU-EXIT reboots preempt the wrapper's exit recording** — a
   kill -9 of the GPU python reboots the machine inside the wrapper's 2s
   poll window: no crashlog entry, no staydown marker. The breaker counts
   only exits that do NOT take the machine down (boot_fails, clean
   exceptions). The reboot itself is the observable.
7. **[OPEN, ops] stop semantics under launchd**: `enginectl stop engine`
   cannot see system-daemon services from the user launchctl domain (takes
   the socket-RPC path instead of launchctl unload); every engine stop on
   this rig is followed by the GPU-EXIT reboot, which RunAtLoad heals — an
   RPC shutdown is effectively a RESTART. `launchctl disable`/`unload -w`
   flags did NOT survive the reboot-resurrection in this session. A true
   stop needs the disable-then-bootout sequence worked out (follow-up).
8. **[OPEN] batch-config exactness through cache paths**: the r6_gates spot
   shows solo FRESH exact, CACHE_HIT replay + cache-derived batch phases
   FAILING — the finding #5 class now also breaks the batch config's
   phase-2/3 gates that passed pre-W3 (yesterday's Phase-3 run). Needs the
   T1-node-restore bisect (pre-W3 pcache on the batch config) as follow-up.
   The batch MECHANICS are healthy (concurrency, determinism, slot hygiene).

Operational notes: /tmp is wiped on every GPU-EXIT reboot — the W5 env
transform script living in /tmp was lost mid-flow (keep op scripts under ~/).

## RESTING STATE

Daemon live on merged main via the launchd supervisor (com.tlx.llm-engine +
com.tlx.llm-api, UserName=%USER%), canonical env (BATCH_B=1, LOOKUP_K=10,
PF_W4A8, NV_SPILL_EXEMPT_NAMES=pf,p8), parked 97810, /health clean,
config_fp e91605105e1a34c0 == expected, drift check on.

## W5 ADDENDUM — THE SOAK CATCH (finding #9, FIXED c5e42d6)

The first 15-min soak on the final resting config FAILED at ~13 min (41
rounds, faults=23): at the FIRST global 256-cycle rebuild boundary,
h_generate's `E.build_graphs()` hit the Phase-3-documented KERNARGS-SLAB
class (ParityGraph ka-slab alloc → alloc_sysmem MAP_SYSMEM_FD returned no fd
→ IndexError) and the RAISED exception poisoned the daemon — ST dirty, every
subsequent generate 503 `engine error: list index out of range`, service dead
until restart (the daemon itself survived; no fault-reboot; breaker silent —
the W1/W2 fault-path isolation worked exactly as designed). The Phase-3
mitigation (failed fences are NON-FATAL) existed only on the BATCH scheduler.
FIX c5e42d6 ports the posture to the legacy path: failed rebuild → slog
`gen_rebuild_failed`, old graphs stay valid (sess holds them), counter
resets, generate continues. Post-fix validation: forced cycle-crossing + the
full 15-min soak re-run (results below).

9. **[FIXED c5e42d6] legacy-path rebuild fatality** — first-rebuild
   alloc_sysmem refusal (fd-less MAP_SYSMEM_FD) killed the canonical service.
   Note: exhaustion-at-FIRST-rebuild differs from the historical 12-rebuild
   M1C runs — the pool pressure deserves its own look when the durable
   ka-slab-reuse fix lands.
