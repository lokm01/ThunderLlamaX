# MTP v3 build state (2026-08-29, in progress)

DESIGN (validated by mtp_v3_step05.py):
- probe = ONE TinyJit(_fwd+head, T=K+1=3) at JIT=1 — ONLY graph family. CAPTURE+REPLAY WORKS, no fault.
- head INSIDE the probe graph (model.output(model.output_norm(h).half())) — avoids 2nd family.
- per-step GDN states: fork model.py T>1 scan collects states[t]; store to persistent step_rec/
  step_conv buffers (uop.store chains like existing); partial accept: host reads m (3-int argmax
  copy), then per block b.recurrent_state.assign(b.step_rec[m]).realize() EAGER between replays
  (PROVEN: step0.5 [b] — assign between replays works; cache_kv rand-fill precedent).
- draft: eager single block (copy mtp_spec.py v223 manual composition, lines ~170-200), h seed =
  h_rows[m] from probe output.
- attn KV garbage slots beyond accept: harmless (overwritten before read — established).

BLOCKER FOUND (786ms/cycle replay): the T=3 path does NOT hit ANY A3 kernels — all a3b/a3d/a3e
matchers key x-numel == K (5120); at T=3 x = 15360 -> no substitution -> probe runs at pre-A3
tinygrad speeds. FIX NEEDED: batched-row variants of the GEMV kernels (y numel 3N, x 3K; warp
handles (row, batch) = y-index; x read at x[b*K + e]; same dequant; epilogues per-row). The r_48_2
in_proj kernel at T=3 likewise (its sig also scales). Attention T=3 kernels: r_6 qk etc are
already token-parallel (T=3 = 3x launches of same kernels? check hist of the T=3 pass).
Order: (1) add T=3 GEMV variants (iq3 17408 silu/mul + 5120 down + q5conv qkv + q5head? head is
[3,5120]@[5120,248320] — batched head GEMM, not GEMV — needs its own rule or falls back (one
kernel, ~3x 3ms = 9ms — acceptable initially); (2) re-measure probe replay (target <=120ms);
(3) then assemble the full v3 cycle in mtp_spec.py-style driver + greedy gate 60/60.
Cycle budget @2k: probe ~85 + draft eager ~8 + head ~3 + assigns ~2 = ~98ms / 2.4 tok = 41ms
-> ~24 tok/s @2k; @100k: +attn 27ms -> ~19 tok/s.
Files: mtp_v3_step05.py (step0.5 harness), /tmp/mtpv3_noassign.py copy in project root.

== BATCHED-GEMV IMPLEMENTATION PLAN (exact, do this next) ==
K is W-derived (256*B) — batch comes from X: batch = x_numel // K (1 or 3). N = y_numel // batch.
1. a3b.py _match: reorder — compute B/K from W first; batch from the x candidate (numel in {K,3K});
   N = y[2]//batch; then xs filter a[2]==batch*K, auxs a[2]==batch*N; add batch to cands dict.
2. a3b.py _build_source: grid = ceil(N*batch/wpc); roles.get("batch",1).
3. Template a3b/iq3_gemv_body.cu header:
   if (warp >= A3B_N*%%BATCH%%) return; row = warp % A3B_N; xrow = (warp/A3B_N)*A3B_K;
   W row uses `row`; XLOAD/XNW/XPLAIN strings (_XLOAD_*/_XNW_*/_XPLAIN_* in a3b.py) add xrow
   to the a3b_x offsets (nw NOT batched). Epilogues y[warp]/aux[warp] already flat-batched.
4. _match_q5conv: batch = out[2]//(4*R); require x==5120*batch, inn==3*R*batch; roles.batch.
   Template: warp<N*batch; row=warp%N; xrow=(warp/N)*K; XW uses xrow+...; dot store at
   3*N*batch + warp; passthrough off = warp + lane*(N*batch).
5. _match_iq3swish: batch = out[2]//6144-rel: out[2]*1960 == batch*W; x==5120*batch,
   vec==N*batch; template: row=warp%N; xrow=(warp/N)*K; s48[row>>7]; nwo[row&127]; vec[warp].
6. q5head at T=3 = batched GEMM (no rule; tinygrad fallback ~fine). Attention T=3 = prefill-style
   kernels (no A3c rules; acceptable). GDN scan T>1 = stock swarm (the remaining probe cost).
7. Validate: step0.5 rerun -> target probe <=150ms; then sanity (T=1 path must still match 12/12 —
   batch=1 identical), then bench_ctx full sweep (T=1 regression check), then v3 driver.

== T3 KERNEL LANDSCAPE (from /tmp/t3dump allsigs2 — CRITICAL) ==
At T=3 tinygrad SPLIT behavior:
- FFN gate/up MATERIALIZES: E_10880_4_8_2_4_2_4_4(half* out_89128960=178MB, uchar* W_34119680,
  grid_1024) dequants the FULL weight to fp16 EVERY replay + GEMM kernels (r_7680 reads half).
  128 launches x (34MB read + 178MB write + 178MB GEMM read) ~= 22GB/cycle -> THE 837ms.
- Lazy T=3 GEMV pattern DOES exist for others: (half* out_3N, float* x_3K, float* s_3 PER-ROW
  scales, uchar* nw, uchar* W): r_15520 lm_head (W_874086400, out_744960), r_320 qkv-ish
  (W_36044800, out_61440, NO passthrough arg at T=3), r_768 (W_51609600, out_36864), r_24
  (W_983040, out_144 — NOT 98-divisible, different format).
=> MTP v3 PATHS FORWARD (pick next session):
  A) Fork scheduler: force lazy fused GEMV at M=3 for IQ3 (find where M>1 switches to
     dequant+GEMM in the contraction scheduler; gate by env MTP_T3_LAZY=1). Then the batched
     matchers (this session parts 1-2, committed) + s_3-per-row template make the probe fast.
  B) Substitute the E_10880 dequant with a faster dequant (still materializes — floor ~2x
     weight bytes + fp16 write: FFN pool ~= 50ms/cycle — probe ~200ms — marginal).
  C) K=1 MTP with T=2: check if T=2 ALSO materializes (likely threshold M>1). If scheduler
     keeps lazy at M=2, K=1 gives ~1.7x with less work.
Batched matcher/builder code IS in (a3_t3_1b/2 patches applied, T=1 sanity 12/12 intact).

== T=2 TESTED: PATH C DEAD ==
T=2 probe materializes the SAME way (E_10880 dequant -> 178MB fp16 weight copy, batch-
independent, one per FFN weight per pass). ~50GB/pass traffic floor -> any M>1 probe is
slow without a scheduler fix. PATH A IS THE WAY: find where the fork scheduler abandons the
lazy fused-GEMV contraction for dequant+GEMM when M>1 (likely in the beam/expander cost
model for contractions with dequant expr sources) and gate it with env MTP_T3_LAZY=1.
Then the batched matchers (already landed) + s_3-per-row template finish the probe.
Alternative quick probe-cost check: substitute E_10880 with a warp-per-row DEQUANT
(34MB read, 178MB write, ~0.5ms each, 128/pass = ~27ms just for writes — still 22.8GB GEMM
reads; net probe ~180-250ms — NOT enough). Path A or bust for MTP speedup.

== PATH A ELIMINATIONS (2026-08-29, experiments committed to /tmp/mtpv3_orient*.py) ==
- USE_TC=0 does NOT stop materialization (T=3 probe: E_10880 still present, 1109ms).
- Matmul orientation W @ x.T does NOT dodge it (same E_10880_2_8_16_4_4_2 + r_544 schedule;
  both orientations identical — graph normalizes).
=> The realize-forcing lives in the FUSION rules for multi-row contractions: an elementwise/
   dequant chain feeding a contraction with >=2 output rows is REALIZED instead of fused
   (correct heuristic for big M; wrong for M=3 with our hand GEMVs). FIND: the rule in
   tinygrad/codegen/ (decomp/ or the contraction rewrite) that splits ALU chains from
   CONTRACTION operands by output size — gate it M<=4 via env MTP_T3_LAZY=1. Then rerun
   mtp_v3_step05.py: expect lazy fused GEMVs (batched matchers will fire, s_3-per-row
   template still needed for qkv/lm_head/down at T=3).

== GATE LANDED (fork eeb2bf2) + FAULT CHARACTERIZED ==
MTP_T3_LAZY=1 (rangeify.py buffer_in_reduce gate): standalone M=3 FFN fuses to ONE lazy
r_544 kernel (no materialization) — batched a3b subs fire (down_proj 929->488us).
FULL-MODEL T=3 JIT=1 capture: DEVICE FAULT during capture (30s wait timeout -> watchdog;
zero malformed ASTs reach the parent hook; BEAM=1 also sees an AST-validity assert at
search.py:45 call construction — some lazy-T3 schedule variant is malformed/hangs).
NEXT DEBUG PLAN: bisect by block count (MTP_T3_LAZY + forward only blk[0:n], find smallest
n that faults; then within that block bisect ops — suspect the GDN T>1 scan or attn SDPA
interacting with the gate). Also try: gate ONLY for the FFN/downproj shapes (condition on
buf.shape[1]==5120 && dtype fp16 — the dequant weights) to leave scan/attn schedules stock.
Files: mtpv3_t3iso.py (isolated harness, per-cycle prints).

== T3 GEMVs LIVE (fork 281502e) — NEXT: Q5 T3 variants ==
Probe T=3 JIT=1: ISO PASS, 325ms/cycle steady, all 6 a3b GEMVs substituted (FFN silu+mul
N=17408, down N=5120 x2, scales 12288/1024) via per-row s_3 + no-validate fast path.
REMAINING T=3 kernels to substitute (from /tmp/t3b/allsigs2.txt, dump run flaked but sigs ok):
- GDN qkv: r_320_32_3_20_2_2_2_32(out_61440=3x20480?, x_15360, s_3, nw_20480, W_36044800)
  — out = 61440 = 3*2N with N=10240 rows: TWO outputs/row (or qkv split) — NEEDS SOURCE DUMP
  (regex "^r_320_32_3" — old name changed!). No passthrough arg at T=3.
- lm_head: r_15520_16_2_3_20_8_2_4_2(half* out_744960=3R, x_15360, s_3, nw, W_874086400)
  — batched q5head variant, half output, per-row s_3. Straightforward.
- r_4_8_2_8_20_3...(half* out_4194304, x_15360, s_3, nw, W_2949120) — unidentified dequant.
Also: cycle0 = 8.5s (first-replay rebind — investigate later); BEAM=1 hardening (search.py
_time_program wrap prg.call assert) still TODO; attention T=3 + scan swarm = rest of 325ms.

== T=3 STATE (fork latest): probe 304ms/cycle steady; qkv a3f live; lm_head half-out
variant committed untested (matcher fl-count bug fixed: half* out => fl==2).
JIT=2 graphless T=3 FAULTS (hist route dead) — use JIT=1 DIFFERENTIAL attribution:
generate override subsets (none / a3b-only / +a3f / all) and time mtpv3_t3iso each.
Unattributed ~240ms over T=1 base: suspects = T>1 attention (mask+SDPA over L), scan x3,
in_proj T=3 forms (r_4_8_2_8 half-out 4194304 = dequant-ish 8MB write x48?), cycle-0 8.2s
rebind anomaly (check driver impact). NEXT: verify lm_head T3 fires; differential timing;
per-step GDN states + mtp_v3 driver (correctness M1) once probe <= ~200ms.

== T=3 DIFFERENTIAL ATTRIBUTION (final this session) ==
ALL subs: 298-300ms/cycle | ZERO subs (empty override): 350-351ms | GEMV subs worth 51ms.
Remaining ~249ms = stock T=3 kernels. Ranked suspects (verify next session, in order):
1. in_proj T=3 family: r_4_8_2_8_20_3...(half* out_4194304 = 8MB half write, x_15360, s_3,
   nw, W_2949120) x48 launches + r_24_3_256_2_20(out_144...) — materialize-ish + GEMV;
   extend iq3swish-style matcher to their sigs (dump sources first).
2. T>1 attention: mask+SDPA over L at T=3 (score work x3; T=1 attn was ~4ms) — check cost
   by L-sweep of the probe (2048 vs 8192).
3. GDN scan+elementwise x3 (~45-60ms expected) — inherent; needs fusion not substitution.
Also: cycle-0 8.2-8.5s rebind on first replay — check whether the driver (new toks tensor
each cycle) re-triggers it (if so: feed via assign into a persistent buffer instead).
lm_head T=3 + qkv T=3 a3f BOTH verified live (r_7760 N=248320 grid 186240).

== SEQ-ATTN LANDED (fork latest) ==
MTP_SEQ_ATTN=1 + MTP_KV_CHUNK=8: T=3 attn = per-position chunked passes. Probe: 2k 280ms,
8k 321ms (slope 6.7us/ctx-tok; A3c T=1 is 0.28 — closing this = batched-T A3c kernels,
Phase 3; est @100k probe attn ~80ms if done). Symbolic per-pos slices FAULT — never retry.
REMAINING probe cost @2k ~280ms: ~250ms context-FREE (scan x3, elem x3, in_proj T3,
norms) + ~13ms attn. Next: T=2 differential to split per-row vs fixed; in_proj T3
matchers; cycle-0 8.7s rebind check. Then per-step states + driver (M1).

== ATTRIBUTION CONCLUSION + OOB FIX (fork da42174) ==
nw OOB fixed (batch kernels read nw with batch offset — 2x past buffer; explains flaky
cold-capture faults + JIT=2 SM errors). Full ISO: PASS 277ms @2k stable.
JIT=2 graphless T=3 STILL faults (Misaligned Address — some other bad access in the
graphless launch path; JIT=1 graph path is FINE — park JIT=2 forever).
T=3 cost structure: ~3500 kernels x ~80us effective = launch/execution-bound stock swarm
(T=1: 1194 x 54us). The 240ms stock = small-kernel swarm at 3x work => PHASE 3 swarm
kernels are ALSO the probe speedup (both paths). r_4_8_2_8 dequant GONE from current
schedule (was orientation-era); r_24_3 = tiny norm. BLKLIM new-graph captures remain
coin-flip (cold compile during capture; warm-cache captures pass) — driver must
retry-on-fault at capture.
NEXT: (1) per-step GDN states (model.py) + mtp_v3 driver = M1 correctness; (2) Phase 3
swarm kernels to pull probe 277 -> ~150ms and T=1 65 -> ~50ms together.

== DRIVER END-TO-END (project d57abda) + GATE CORRECTNESS BUG ISOLATED ==
mtp_v3.py runs the full cycle loop (probe jit T=3 JIT=1 + eager draft + step-state select).
Numerics WRONG with AND without substitutions (empty-override run: amd garbage too) =>
the MTP_T3_LAZY gate itself creates schedules that COMPUTE GARBAGE at BEAM=0 (the
buffer_in_reduce keep-rule is a CORRECTNESS guard for range structure — the earlier
"ranges REDUCE not DEVICE" assert at BEAM=1 was catching exactly this; BEAM=0 compiles
the malformed kernel anyway). NOT the substituted kernels, NOT seq-attn necessarily.
NEXT (in order):
1. Confirm: gate-OFF T=3 probe numerics exact (v223-era materialized path was 60/60 —
   one driver run without MTP_T3_LAZY; slow but correctness reference).
2. Fix the gate: in remove_bufferize bypass, only allow when the substitute keeps all
   ranges valid — mirror the PCONTIG>2 partial-contig safety (bufferize reduce-feeding
   operands to LOCAL) instead of skipping the check. Read rangeify.py 226-289 again:
   the rescue path at PCONTIG>2 shows the valid-substitute shape.
3. Driver perf fixes (known): persistent toks buffer (kill 1.6s/cycle rebind), batch the
   96 select realizes into ONE realize (kill 1.8s/cycle), prefill via chunked T=1 or
   accept 6s. After gate correctness: probe ~300ms + draft ~0.3s + sel ~10ms =>
   first REAL M1 speedup measurement.
Landmines: mtp_spec import = 2nd model load; draft sd keys "blk.64."-prefixed.

== DRIVER PERF STATE (run 9): 20/20 held; probe 4.4s/cyc UNSOLVED ==
Tried and NOT the trigger: toks assign (4.8s), stable-copyin toks buffer (4.4s), select
assign->uop.store (4.4s). ISO (no eager between replays) = 280ms. Driver adds: draft/head
eager JIT=2 between replays, select stores, h_seed slices. NEXT: instrument TinyJit
capture count (subclass probe_j or monkeypatch __call__ log); compare per-cycle input
specs. Suspects: (1) something in the eager work invalidates the jit cache (recompile ~4s
matches cold-ish compile); (2) v_sp.bind uop identity per call; (3) flush_step_states
attrs. Fallback if unfixable today: accept 1.6s/cyc plain-tensor path = 1.3 tok/s demo.

== FINAL ISOLATION (this session end) ==
probe in-cycle 4.4s; back-to-back probe 0.46s. Disproved: toks uop swap, state assign,
pool churn (retain test). The eager draft/head EXECUTION between replays makes the next
JIT=1 graph exec 10x slower. Next: (a) profile probe-after-eager (KERNEL_HIST? VIZ?);
(b) test probe at JIT=2 graphless as the fallback (no graphs, no penalty, ~350ms probe,
cycle ~0.96s = 2.3 tok/s); (c) if the graph penalty is resubmission-batching, look at
ops_nv exec_graph / hcq buffer b[] pool interaction with eager copyin path.
CORRECTNESS IS DONE (20/20); this is purely a perf phenomenon with a 10x lever.

== PROBE MYSTERY SOLVED + HONEST STEADY STATE ==
The 4.4s/cycle probe = AMORTIZED CAPTURE (TinyJit 2nd call = compile all ~3500 kernels, 20s,
ONE TIME). Steady probe = 0.317s. Bisected ALL driver components (draft/head/select each
disabled): none affect probe timing. eager_isolation: full T=1 eager fwd between probes
does NOT trigger any penalty (0.286s).
STEADY DRIVER: probe 317 + draft 290 + head 210 + select 110 = ~0.93s/cycle = 2.4 tok/s
(at 2.22 tok/cyc, acc 0.66). vs baseline 15.4 @2k. NOT YET BEATING BASELINE.
RELIABILITY: 60-tok run died cycle ~26 (30s wait timeout — kernel never signaled; the
JIT=1-replay + eager-interleave pattern eventually wedges the channel; matches the
historical cycle-2 fault class).
THE remaining bottleneck for EVERYTHING (probe + draft + head): ~5000 kernels/cycle at
~80us-1ms launch tax each = the graphless floor. HCQ2 (getenv, exists in fork, hcq_compile/
hcq_link at realize.py:312-322 + exec_graph path) batches kernels into graph submissions —
NEVER TESTED on this rig. THAT is the next big lever: HCQ2=1 on the probe.

== HCQ2 TESTED: NOT THE LEVER (final this session) ==
HCQ2=1 ISO: 0.354s steady vs 0.317s graphless — graphs build+run but the per-launch
~80-90us PCIe/driver round-trip dominates regardless. THE FLOOR IS THE KERNEL COUNT.

== STRATEGIC PICTURE (honest, for the 40 tok/s target) ==
Baseline T=1: 65ms/tok = 15.4 tok/s = ~1200 kernels/tok at ~55us effective each.
MTP v3 cycle floor: ~5000 launches (probe 3500 + draft/head 1500) x 80us = ~400ms of
pure launch tax, before any compute. Cycle must be <=144ms (K=2) or <=187ms (K=3).
=> KERNEL COUNT is the ONLY remaining lever: probe must drop from ~3500 to ~600-800
kernels (Phase-3 swarm fusion: the GDN scan chain + norms + elementwise are ~2800 of
the 3500), draft+head to ~400. This is scheduler/fusion-level work in the fork, not
more hand GEMVs (the GEMVs are done and near byte-floor).
The 40 tok/s arithmetic: (600 probe + 400 draft/head) x 80us = 80ms launch + ~40ms
compute = 120ms cycle x 2.6 tok/cyc (K=3) = 2.6/0.12 = 21 tok/s... still short.
=> ALSO need K=4-5 with high acceptance + draft-vocab slice + possibly the
launch-tax itself attacked (the a3a notes: batched q.exec = 6.7us/kernel — THAT rate
makes 1000 kernels = 6.7ms!! The 80us is the PYTHON-side eager launch; batched
graph exec was 12x cheaper. The exec_graph path at HCQ2 should hit this — why is
HCQ2 not faster? NEXT SESSION: verify HCQ2 actually batches (check graph_cache hits,
exec_graph vs exec_kernel call counts in the replay).

== BATCH_EXEC TESTED: NO WIN — PROBE IS GPU-COMPUTE-BOUND (critical insight) ==
Batched q.exec (A3a pattern, one signal+submit for all kernels): 352ms — IDENTICAL to
non-batched 317-354ms. The per-kernel signal+submit overhead was ALREADY hidden by async
submission (CPU submits N+1 while GPU executes N). THE PROBE AT 317ms IS PURE GPU EXECUTION
TIME for ~1200 kernels doing 3x work. Launch tax is NOT the probe bottleneck.

== STRATEGIC RECALCULATION (the honest math) ==
T=1 base: 65ms = GPU compute for ~1200 kernels (55us avg; GEMVs near byte-floor, scan/elementwise latency-bound).
T=3 probe: 317ms = GPU compute for ~1200 kernels doing 3x work (264us avg). WORSE per token than sequential!
MTP cycle: 317 (probe) + 290 (draft) + 210 (head) + 110 (select) = 927ms for 2.2 tok = 2.4 tok/s.
Sequential for 2.2 tok: 143ms. MTP IS 6.5x SLOWER per token.

THE DRAFT ANOMALY: 290ms for ~3 steps of ONE BLOCK (~50 kernels each). Should be ~4ms.
~98ms/step. NEEDS INSTRUMENTATION — suspect: .numpy() sync per step + JIT=2 compile cache
misses + the head GEMV (r_7760 at T=1 ~3ms but maybe recompiling per step?).

THE REAL PATH TO 40 (revised, in order):
1. GDN SCAN FUSION (the one thing that matters): ~900 tiny latency-bound scan/elementwise
   kernels per pass = ~35ms of the 65ms T=1 base. Fuse into 2-3 hand-CUDA kernels/block
   x 48 blocks = ~150 total. Base T=1: 65 -> ~35ms. T=3 probe: 317 -> ~170ms.
2. Draft step fix: 98ms -> ~10ms (find the anomaly).
3. K=3: 3.2 tok/cyc x cycle(170+30+10 = 210ms) = 15 tok/s. STILL SHORT.
4. Deeper scan fusion (1 kernel/block): base T=1 ~25ms, probe ~75ms, cycle 85ms = 38 tok/s.
=> THE GDN SCAN CHAIN IS 80% OF THE REMAINING WORK. All MTP machinery is correct and waiting.

== FUSED-NORM EXPERIMENT: FAILED (important negative result) ==
Computing rms(x) in every GEMV CTA = 4352 CTAs x 20KB = 87MB redundant reads vs 2.6MB
for the separate r_256_20 kernels. Result: 82.6ms vs 64.5ms baseline (+28% WORSE).
LESSON: never fuse a reduction into a multi-CTA kernel — redundancy multiplies.
The separate norm kernels (129 x 32us = 4.1ms) are actually EFFICIENT — the 32us is
the GPU kernel-dispatch floor, not waste.

== THE COMPLETE HONEST PICTURE (all experiments exhausted) ==
T=1 base 65ms = GEMVs 30ms (near 28ms byte-floor, DONE) + scan/elementwise 28ms (14
latency-bound kernels/block x 48 blocks x ~33us each — the GPU kernel-dispatch floor)
+ attention 2ms + head 3ms.

THE ONLY REMAINING PATH TO 40 tok/s:
1. GDN MEGA-KERNEL: fuse the 14-kernel between-GEMV chain (conv→silu→split→norm→
   delta-scan→gating) into 1-2 hand-CUDA kernels per block. Saves ~280us/block x 48
   = 13.4ms → T=1 52ms. This is the single biggest remaining lever.
2. Draft fix: 98ms/step anomaly (1 block should be ~2ms).
3. With both + K=3 MTP: T=1 ~40ms base, probe ~55ms, cycle ~70ms, 3.2/0.07 = 45 tok/s.
The mega-kernel is ~500 lines of CUDA and requires reading/writing the full scan state
(32x128x128 fp32 = 2MB/block through global memory — unavoidable). Estimated 1-2 sessions.

== DRAFT INSTRUMENTATION + FINAL ARITHMETIC (2026-08-30) ==
Draft step = 88ms because EVERY .realize() outside TinyJit = full sync roundtrip (16ms).
Inside TinyJit replay = 90us/kernel. TinyJit for draft fails (input structure mismatch).
THE 40 tok/s ARITHMETIC (K=4): GEMV 28ms (floor) + scan-fused 25ms + draft-fixed 25ms +
head-in-probe 15ms + select 9ms = 102ms cycle x 4.2 tok/cyc = 41 tok/s.
Non-negotiable: GDN mega-kernel (scan 250->25ms) + draft TinyJit (88->5ms/step) + K=4.

== GDN SCAN KERNEL INTEGRATION STATUS (2026-08-30 end) ==
KERNEL: PROVEN CORRECT + FAST. T=1/3/8 all pass (relerr <2e-7). 4.88ms/pass at T=3.
T=1 INTEGRATION: WORKS (sanity 12/12) but no speed change — scan is only ~5ms at T=1.
T=3 INTEGRATION: BLOCKED by ALLOW_DEVICE_USAGE. The .realize() calls inside the model
forward pass (for the scan kernel inputs) trigger compilation which dispatches to the
worker pool where device access is forbidden. Attempted fixes that did NOT work:
- Hardcoding ALLOW_DEVICE_USAGE=1 in worker.py (error is in MAIN process, not worker)
- WORKER_ALLOW_DEVICE env var (not reaching spawned workers)
- Context(ALLOW_DEVICE_USAGE=1) wrapper (auto-indent broke file — needs careful manual fix)
NEXT SESSION FIX PATH (in order of preference):
1. Wrap scan section in Context(ALLOW_DEVICE_USAGE=1) — CAREFULLY, manual indent
2. Pre-realize scan inputs OUTSIDE the forward pass (requires restructuring)
3. Replace .realize() with buffer-level operations (no scheduler involvement)
4. Integrate via A3b substitution (replace first kernel of scan chain with mega-kernel,
   make remaining 17 pass-through no-ops)
The kernel itself is DONE and banked. Only the integration plumbing remains.
