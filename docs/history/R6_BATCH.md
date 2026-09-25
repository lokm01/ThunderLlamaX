# R6 — BATCH-MODE DECODE: multi-stream aggregate throughput (the B axis)

Status: RUNGS 1-4 LANDED — B=2 batched decode, per-stream Tier-1 bit-exact at
every rung, ZERO new CUDA kernels, best aggregate **81.33 tok/s** (both-8k,
both-k4-deep, draft-skip). Harness `engine0/r6_batch.py` (+ a 1-line
`_draft_entries` dd-override in mtp.py, default path byte-identical).

## THE LADDER (all gates = per-stream 60/60 exact ×2 deterministic)

| rung | config | ms/cyc | tok/cyc | aggregate | vs best solo |
|---|---|---|---|---|---|
| solo s0 (8k slice@40000, K2) | 60/60 vs T=1 | 52.9 | 2.18 | 41.2 | — |
| solo s1 (prompt8k, K2) | 60/60 ×2 | 53.1 | 2.67 | 50.1 | — |
| R1: B=2 100k-snapshot + 8k (3+5, BT=8) | 60/60 ×2 BOTH | 107.3 | 7.10 | 66.2 | 1.32× |
| R2: B=2 8k+8k (3+5, BT=8) | 60/60 ×2 BOTH | 93.9 | 6.50 | 69.3 | 1.38× |
| R3: B=2 8k+8k (5+5, BT=10) | 60/60 ×2 BOTH | 106.7 | 8.35 | 78.3 | 1.56× |
| **R4: + batched draft-skip** | **60/60 ×2 BOTH** | **102.7** | **8.35** | **81.33** | **1.61×** |

(s0 solo @100k snapshot = 42.11 — the R1 mixed rung's per-stream retention was
0.62 @100k / 0.80 @8k; R4 retention 0.78/0.84. The stock 59/59 cross-check
passed on the snapshot rung. Boot logs: ~/r6_boot4/7/10/11.log.)

## THE DESIGN (the R6 batching law)

One M=BT probe trunk per cycle where BT = sum of per-stream T rows (B×T ≤ 64 —
the ROWS>64 law). Two kernel classes:

1. **Row-independent M families** (k0n/k0ab norms, q5g8v/aq*q8v GEMVs, k3ao/op38,
   ao8nw32, hh, ffn8v/down8nw32 (+r7 packed7 ports), head8v, amx3): ONE launch
   processes ALL B streams' rows — the weight read is SHARED (this is the whole
   point). Per-row fp order is identical across M (the R4 rowchk law) so each
   stream's rows are bit-equal to its solo run.
2. **Per-stream STATEFUL kernels** (spk_pre/spk_a/spk_c attention, k2s GDN scan,
   h_embed, accept*, acceptsel*, draft chain, lookup): B launches per cycle over
   SLICED row-major scratch + per-stream state banks. Slicing is legal because:
   - all row strides are 16B multiples (the ALIGNMENT law), and
   - the spk partials are COMPACT per-(ROWS) regions (`pb0=(g*S+s)*RMAX`,
     RMAX=6*ROWS — a ROWS=T kernel touches a contiguous 4*S*6*T block), and
   - shared staging (qw3/qw16_3, pm1-class draft partials, q/k/v/core) is
     per-launch TRANSIENT between adjacent serialized launches.

Per-stream state banks (`*_s{s}` bufs, stream 0 = canonical names so every
other path stays byte-identical): kv{i}/sc{i} ×16 layers, rec4/conv4/conv5x
GDN banks, cur/pos/tok/m/cyc slots, tok_hist/m_hist/l_hist, dring0..9, emit,
h_seed/dhd_seed/hd_d0/hd_d1, dpos1/2, kv_d/sc_d. Banks are allocated at boot
(fixed-handle law) and device-zeroed via mfill (no host DMA — DART law).

The EAGER machinery (prefill/fill_draft/trunk T=1) is made stream-aware by the
**swap trick**: temporarily remap P.d[canonical] → P.d[name_s{s}]; all eager
launch sites read P.d at call time. The T=1 reference decoders (GCycleEngine
per stream) build their graphs under the swap, baking per-stream handles.

## RUNG 1 RESULTS — B=2 mixed (s0: 100k snapshot T=3 K2; s1: fresh 8k T=5 K=4-deep)

| gate | result |
|---|---|
| solo-s1 (B=1, T=3, M3 graphs) vs its T=1 ref | **60/60 exact ×2 deterministic** |
| solo-s0 (B=1, T=3, M3) vs BANKED engine_t1_ref | **60/60 exact** (42.11 tok/s, 66.10 ms/cyc) |
| batched B=2 (BT=8, M8) s0 | **60/60 exact vs banked ref (both reps)** |
| batched B=2 s1 | **60/60 exact vs its T=1 ref (both reps)** |
| batched deterministic ×2 (s0, s1) | **True, True** |
| stock cross-check s0[:59] vs spec_base_100k[1:60] | **59/59** |
| s1 lookup (k4 stream) | 60 cycles, 47 hits (l≥8), tok/cyc 4.32 |

Performance (the honest table):

| config | ms/cyc | tok/cyc | per-stream tok/s | aggregate |
|---|---|---|---|---|
| solo s0 (100k ctx, K2) | 66.10 | 2.78 | 42.11 | 42.11 |
| solo s1 (8k ctx, K2) | 53.2 | 2.67 | 50.1 | 50.1 |
| **batched B=2 (100k + 8k)** | **107.3** | 7.10 (2.78+4.32) | s0 25.9, s1 40.3 | **66.2** |

- Aggregate 66.2 = **1.57× the 100k-solo** (42.1), 1.32× the 8k-solo (50.1);
  72% of the sequential ideal-sum (92.2).
- Per-stream retention: s0 0.62 (100k ctx — the per-stream attention tax is
  structural at 100k), s1 0.80.
- THE 2× BAR AT 100K-CLASS IS NOT REACHABLE with this architecture's cost
  model: per-cycle cost ≈ shared-M-trunk + B×(attention over own 100k pool
  ~16ms) + B×(draft chain 5.3ms) + B×accept. The both-8k rung (below) is where
  the 2× class becomes arithmetically possible.

## THE SIZING TABLE (empirical)

Per-stream state at CTXK=100352 pools (the pool stride is COMPILE-TIME in the
spk kernels — every stream allocates full-CTXK pools regardless of actual ctx):
16×kv 3.13GB + 16×sc 0.21GB + rec4 0.755GB + conv4 0.03GB + kv_d 0.20GB + misc
≈ **4.3GB arithmetic per additional stream**.

**THE R6 VRAM LAW (boot 1-3, deterministic 3/3)**: the PF chunk-prefill
machinery (ensure64/ensure128 scratch + G3M packed7 extras + PF_P5 planes) FAULTS
(fault-as-OOM, moving floor: r7 uploads in boots 1-2, scscr128 in boot 3) with
even ONE extra stream's banks resident. Fixes: `PF_G3M_MB=0 PF_P5=0` (skip the
optional prefill planes; fg/fu/fd stay native-packed7-aliased = zero extra) AND
`R6_PF_T1=1` (stream prefills via the T=1 trunk path, ~40ms/tok, instead of the
chunk path). Decode-path perf is unaffected. Batch boots must carry these.

## RUNGS
- RUNG 1 (boot 4): B=2 100k+8k mixed — gates GREEN, 66.2 aggregate (above).
- RUNG 2 (boot 7): B=2 both-8k fresh streams (s0 = 100k-slice@40000) — gates
  GREEN (s0 60/60 ×2 vs in-session T=1, s1 60/60 ×2), **69.25 aggregate**
  (93.86 ms/cyc, 6.50 tok/cyc; s0 23.3 + s1 46.0). Solo refs: s0 41.16 (K2,
  2.18 tok/cyc — the @40000 slice is a harder draft class), s1 50.1.
- RUNG 3 (boot 10): **B=2 BOTH-k4-deep (BT=10, M10 via the BT>RM scratch
  resize) = 78.25 aggregate** (106.72 ms/cyc, 8.35 tok/cyc: s0 4.03 + s1 4.32;
  per-stream 37.8 + 40.5). Gates: both streams 60/60 exact ×2 det. The k2+k4
  control rerun in the same boot: 70.39 (machine-state consistent w/ boot 7).
- RUNG 4 (boot 11): **the batched DRAFT-SKIP = 81.33 aggregate** — a
  lookup-only draft graph (draft_lu, the R7a law batched) selected per-cycle
  when BOTH streams' previous lookups hit (prev emit hit flags, the
  READOUT-ORDER law); skips both draft chains on both-hit cycles (−4.05 ms/cyc:
  106.72 → 102.67; identical 8.35 tok/cyc). Gates: 60/60 ×2 both streams.
- B=3+ rungs: BT=9 (M9, wired) — but the cost model says B=3-k2 < B=2-k4
  (k2 streams bring 2.2-2.7 tok/cyc each vs 4+ for k4 rows); deeper per-stream
  K at B=2 (BT=13/16) needs NEW M-family kernels (M12+ — the register-wall
  class; hm11 already carries the one-cold-spill precedent). The B axis now
  needs the kernel campaign, not wiring.

## THE COST MODEL (measured, why 2× is not reachable at B=2)

batch_cycle ≈ solo_cycle(T=3) + ΔM(3→BT) + Σ_{streams}(attn_s + k2s_s + draft_s
+ accept_s), with ΔM(3→8) ≈ 33ms (the D3 family slopes confirmed: ffn_down
+17.9, norms +7.1, gemv_gdn +5.5, head +1.5, gemv_attn +0.9) and the per-stream
extras ≈ 8-11ms @8k (draft 5.3 dominates; attn@8k ~1-2ms; at 100k attn ~16ms
per stream is the structural tax). The weight-read sharing saves only ~12ms of
a ~106ms sequential pair at M=8 — our decode GEMV families are issue/register-
bound per row, not purely BW-bound, so each extra row costs ~65% of a full M3
stream's GEMV time. The smart money is in the MODE MIX: deep-K rows on a
hit-heavy stream cost ΔM ~+7ms/row but yield +1.6 tok/cyc/row.

## PHASE 3 — SHIPPED (2026-09-24, commits 3ebbc59..a2dafa2 on review-fixes-w1)

The batch engine is wired into the serving layer end-to-end. DEFAULT = OFF
(BATCH_B=1 boots the untouched legacy loop byte-identically); the batch
canonical is a documented opt-in (see below). WHY default-off: the aggregate
>=1.5x-solo bar did NOT survive contact with the production solo path —
measured through the SERVICE (two 8k-class convs, both prefilled, generates
started back-to-back):

| workload (8k ctx, K7) | solo | batched aggregate | vs mean-solo |
|---|---|---|---|
| deep-heavy pair (repeat-class corpus) | 89.7 tok/s | 68.0 tok/s | 0.76x |
| prose pair (natural text, low-hit) | 19.6 tok/s | 22.1 tok/s | 1.13x |

Per-stream retention 0.56-0.58 at B=2: the per-stream stateful kernels
(attention over own pool, k2s, draft chains) + the M-trunk widening cost more
than the shared weight read saves when the solo baseline is the FULL deep-K
DecodeSession (the R6 harness's 1.38-1.61x was measured against K2-only
solos). The batch DOES deliver concurrent streaming (two users at ~57% each
instead of 100%/queued) — a latency/fairness capability, not an aggregate win
on this engine at B=2. The B axis needs the kernel campaign (M6+/wider
families), not more wiring.

GATES (all on the batch daemon, BATCH_B=2 canonical):
- PER-STREAM BIT-EXACTNESS x2: solo == AUTO_CACHE replay == batched, BOTH
  streams, deterministic across reps, both slots generating mid-run (engine
  status verified [True, True]). Through the SERVING path (r6_gates.py).
- W1/W2+R6 mock battery: 45/45 green x2 (8 new batch tests: concurrency
  overlap, per-stream cancel/stop, FOLLOW_UP isolation, same-conv
  serialization, third-waits, generate-without-prefill, legacy fallback).
- api_gates battery: 25/25 PASS on the batch config (two V-07-era gate bugs
  fixed: reasoning-aware (a)/(b) compares; permits-aware (e); the engine
  reference now pairs prefill+generate on ONE conn — the batch protocol
  requires conn->slot binding).
- 15-min soak: PASS after the GRAPH-CLASS BUDGET LAW fix (below) — 125
  rounds, 309 requests, 0 errors, 30 clean cancels, 1177s continuous uptime,
  250 rotated fences, queue always drained.

THE TWO NEW LAWS THIS PHASE:
1. **THE GRAPH-CLASS PREFILL BUDGET LAW**: the R6_PF_T1 prefill (trunk graph
   replays) does NOT reset the dext ~950-cycle budget the way the legacy
   EAGER-class PF chunk prefills did. The 15-min soak wedged DETERMINISTICALLY
   (2/2 repros) at ~4000-4500 continuous mixed graph cycles when the global
   rebuild counter reset on every barrier prefill. Fix: the counter NEVER
   resets on barrier work; only the fence-all rebuild resets it (BATCH_REBUILD_EVERY=232
   global; soak then ran clean with 250 fences).
2. **THE KERNARGS-SLAB LEAK**: every ParityGraph build allocates a nolru
   host-mapped kernargs slab that is never released — a full fence-all (22
   graphs) every 232 cycles exhausted host-mapped memory at ~250 fences
   (alloc_sysmem IndexError; generates cancelled). Shipped mitigation: ROTATED
   fence (one graph set per event, 4-5x) + FAILED FENCES ARE NON-FATAL (old
   graphs stay valid; counter resets). Durable fix = ParityGraph ka-slab
   reuse (documented follow-up).

Other Phase-3 facts: pcache T=1-boundary nodes carry dhd (hd_d1 chain state)
+ hlast (x0 trunk hidden) + cur (tok_slot argmax — free, kills the pfk_n16
dependency); the draft KV fills per-chunk interleaved with the trunk so every
64-boundary has capturable chain state; per-chunk Swap scopes are MANDATORY
(the P.d[emit] name-keyed readout hazard under Swap(1)); THE EMIT-KEY
CONTRACT: DecodeSession returns "cycle" — every emit-dict producer must too
(a mismatch silently drops cycle events inside try/pass sends).

## THE BATCH CANONICAL (opt-in)
The legacy ops/env.canonical stays the shipped default (BATCH_B=1 = the
untouched legacy loop). To enable: LOOKUP_K=7 (NOT 10 — the deep-set scratch
+ banks is a fault-as-OOM), PF_DR7=1, drop every other PF_* knob, add
PF_G3M_MB=0 PF_P5=0 R6_PF_T1=1 BATCH_B=2 BATCH_REBUILD_EVERY=232
BATCH_PF_CHUNK=64 (+ R6_SLICE_EXTRA for deterministic draft coverage of
known prompts). COST: 8k FRESH prefill ~30-45ms/tok via T=1 (vs ~2ms/tok on
the PF chunk path — the VRAM law); repeats go through pcache (CACHE_HIT
restore+tail is fast and bit-exact). LOOKUP_K 7 vs 10: solo deep is K7 (63
tok/s class, not 75.6) — the K10 deep-set scratch does not fit with banks.

## PHASE 3 — the serving integration design (next session, not yet wired)

The daemon queue already holds 1+4. Integration contract:
1. BOOT: the daemon host process (test_w100k.py class) gains R6_B=2: allocate
   stream-1 banks at boot (before build_graphs); build the batched graph set
   alongside the canonical one (BatchSession). BATCH=0 kill-switch = today's
   byte-identical daemon (R6_B unset → no banks, no batch graphs).
2. SCHEDULER: requests drain into batch slots. A slot in FRESH/FOLLOW_UP
   prefill PAUSES the batch (prefills are eager + serialized; a mid-prefill
   stream must not join the cycle graph until its prefill + stseed complete —
   anchor semantics from the harness). Decode cycles then run batched.
3. SLOT REUSE: fixed handles mean the graphs NEVER rebuild on membership
   change — a finished stream's slot is reset_fresh'd + re-prefilled for the
   next request (the swap trick scopes all eager machinery per stream).
4. STOP HANDLING: per the STOP-BATCH OVER-COMMIT law a mid-batch im_end marks
   the slot done (host reads per-stream emit stop flags); the batch keeps
   cycling for remaining streams. Per-stream SSE/finish windows unchanged.
5. MODE SYNC: the graph set is FIXED per boot (the mode mix is baked); a
   request whose class is unknown joins whatever the baked mix serves. The
   per-cycle T-mix selection (k2 vs k4 graph sets per stream) is the
   refinement — needs per-stream graph-set selection like DecodeSession's
   deep flag, but batched (a 2^B set matrix or per-stream subgraphs).
6. The R6 VRAM Law applies to the daemon too: batch daemons carry
   PF_G3M_MB=0 PF_P5=0 (prefill via T=1 path or a VRAM-repriced chunk path).

## NEW LAWS (R6)
1. **THE COMPACT-PARTIAL SLICE LAW**: spk pm/ps/pA partials live in compact
   4*S*6*ROWS regions (pb0=(g*S+s)*RMAX) — per-stream slices at r0*4*S*6*4 are
   collision-free; qw staging is shareable only because pre→a are adjacent
   serialized launches.
2. **THE R6 VRAM LAW**: +4.3GB/stream banks vs the PF prefill machinery's
   ensure-time allocations (~6GB G3M+P5+scratch) = fault-as-OOM with a MOVING
   floor (the faulting op differs per boot: r7 uploads, scscr128, init_draft's
   slice copy); deterministic 3/3. Batch boots carry PF_G3M_MB=0 PF_P5=0
   R6_PF_T1=1; the floor sits within ~300MB of the LK=7-boot world (boot 8:
   LOOKUP_K=9's extra deep-set scratch alone tips it).
3. **THE BT>RM LAW**: a bigger batch trunk does NOT need a bigger LOOKUP_K
   boot — the RM-sized probe scratch can be REPLACED post-boot (pre-graph-build
   = fixed-handle-safe) with BTMAX-row buffers + the M-family cubins loaded
   directly into E.pr; saves the LOOKUP_K=9 boot's +755MB deep-set scratch.
4. h_embed launch counts are PER-CUBIN (3 → launch 3; 5/6/7 → launch 1; 8 → 8):
   never assume grid=T for the M-embeds; copy from the proven seqs.
5. The batched draft chains can SHARE all intra-step transient scratch (e_buf,
   xin_d, ...): the chains are serialized in-graph and only the per-stream
   PERSISTENT state (kv_d/sc_d, hd_d0/d1, drings, h_seed) needs banks.
6. THE GPU-EXIT REBOOT LAW bit the daemon-stop again (the graceful shutdown
   RPC still reboots the machine seconds later — recovered, expected class).

## Harness
`engine0/r6_batch.py` — env: R6_B, R6_SPEC ([{"T","mode","s"}...]), R6_PF_T1,
R6_S0_FRESH (+R6_S0_OFF for the 100k-slice prompt), R6_P8K (ids npy).
Boot env = the canonical daemon line with LOOKUP_K=7 (RM=8/M8 world; 8/9 for
BT=9/10) + PF_G3M_MB=0 PF_P5=0 R6_PF_T1=1 + the R6 knobs. Logs ~/r6_boot*.log.
