# GDN Scan Fusion — execution package (2026-09-10)

## The prize
Top-5 chain members measured 45-81us each x48 blocks = 13.8ms/probe (26613f3);
~20ms total. Megakernel target 101.6us/block x48 = 4.88ms -> ~15ms/probe
(cycle 337 -> ~322ms, ~7.5 tok/s).

## The code
- Scan loop: fork model.py GatedDeltaNetBlock._attention_mtp lines ~561-568
  (unrolled delta-rule: per t: s1=state*alpha; delta=(v-(s1*k).sum)*beta;
  state=s1+delta*k; outs.append((state*q).sum)) — the ~15-kernel/block swarm.
- Proven kernel: ~/tinygrad-metal/a4/gdn_scan.cu (f57590f; T=1/3/8 relerr<2e-7;
  101.6us/block; fixes already in: typed TinyELF sig, warp broadcast, f64 test data).
- Probe path: mtp_v3._fwd3 -> call_mtp -> _attention_mtp (states explicit args).

## Routes, ranked after this session's analysis
1. 1:1 a3b substitution: IMPOSSIBLE — no single chain kernel carries all five loop
   inputs (q,k,v,beta,alpha); verified from cdump100k signatures.
2. AST-level fusion in _attention_mtp: the serial state dependency forces a reduce
   per step; scheduler breaks there (old P1 verdict, still true).
3. Tensor.custom_kernel op (fork has it: in-place after() semantics): register the
   megakernel as a custom op so it enters the schedule and captures legally.
   The eba2163 blocker ("REDUCE has no ranges") was from wrapping raw CUDA as an
   op pre-custom_kernel — RE-TRY THIS FIRST with the current API.
4. If custom_kernel still fights the renderer: two-kernel split — compute
   delta/coefs per step as now (those kernels are small), but batch the STATE
   UPDATE + output reduce into one kernel per step via substitution of the
   "state*alpha" kernel (its args DO carry state+alpha; q/k/v/beta enter via
   baked VAs like the SKV workspace) — 3 launches/block instead of ~15.
5. Validation ladder: standalone relerr vs the tensor loop (T=3, random states)
   -> 2k gate 60/60 -> 100k gate (MTP_SKV=1 recipe, run39 + scan).

## Gotchas
- The 786432-float args on chain kernels = packed 48-block state VIEWS; the
  megakernel indexes per-block slices internally.
- fp32 discipline: state math is float32 in the tensor code (state = rec_in.float()).
- Hard rules: no gridDim in hand CUDA; poison out-buffers in validation;
  keep harness buffer tensors alive (_KEEP pattern).

## 2026-09-10 correction (post-inspection)
- Route 3 (Tensor.custom_kernel) is DEAD: the API takes a UOps CALLABLE
  (fxn builds a UOp graph; tensor.py:390) — NOT a raw-CUDA hook. The campaign
  already hit "Custom-kernel/UOps megakernel path DEAD (rewriter infinite-loop
  even at T=1)". Do not retry.
- ROUTE 4 IS NOW PRIMARY, concrete design: substitute the step-1 E-kernel
  (s1 = state*alpha; args carry state+alpha) of EACH scan step with a
  full-step megakernel; the remaining step kernels become retro-no-ops
  (machinery exists, proven with the softmax bookkeeping kernels).
  The megakernel needs q,k,v,beta beyond its signature: KEY INSIGHT — q/k/v/
  beta live in GRAPH-INTERNAL buffers whose addresses are STABLE ACROSS
  REPLAYS (same principle as the whole graph capture). The blocker: the a3b
  hook runs at to_program (source level) with NO access to runtime buffer
  VAs. Options: (a) fork surgery to expose the current allocation map to the
  codegen hook; (b) mtp_v3-level: pre-materialize q/k/v/beta into PERSISTENT
  mtp-owned buffers (allocated once, like the SKV ws) that _attention_mtp
  reads inputs from — then their VAs are known and bakeable BEFORE capture.
  (b) is cleaner: _attention_mtp copies q/k/v/beta into persistent bufs
  (small adds ~24KB/block), the megakernel baked-VA reads them, retro-noops
  kill the tensor loop renders. Validation: standalone relerr vs loop, 2k
  gate, 100k gate.

## 2026-09-10 route 4 IMPLEMENTED (carrier design — supersedes 4a allocation-map)
- Fork model.py patched (env MTP_SCAN_MEGA, default OFF): the scan loop is replaced
  by TWO elementwise carrier expressions (e_core over q,k,v,beta,alpha; e_state
  over state+the same), each rendering ONE kernel whose args carry all inputs;
  core/state are then VIEWS/contiguous of the carriers (stock shapes preserved).
  NO assign-in-capture, NO allocation-map exposure needed. Per-launch buffer
  binding serves all 48 blocks (same kernel key, 48 launches).
- REMAINING STEPS (next session, in order):
  1. Run 2k with MTP_SCAN_MEGA=1 + CDUMP_DIR -> capture the two carrier kernel
     sources; decode arg buffer layouts (strides from the views).
  2. Adapt a4/gdn_scan.cu indexing to those layouts (the f57590f math stands;
     T=3; watch the q/k squeeze(-2), v/beta/alpha squeeze layouts).
  3. a3b matchers: core-carrier kernel -> megakernel (writes core + state to a
     per-block ws slot, SKV-pattern baked VA; STRICT per-block ordering makes a
     single slot safe); state-carrier kernel -> trivial copy-from-ws kernel.
     Print [scan-mega] lines; values WRONG if unsubstituted (gate-detectable).
  4. Standalone relerr <1e-6 vs the tensor loop at T=3 (random states; _KEEP
     pattern), then 2k gate 60/60, then 100k gate (run39 + MTP_SCAN_MEGA=1).
- Prize: ~13-18ms/probe (chain 15 kernels/block -> 2 launches/block).

## 2026-09-10 carrier-capture RESULT (cdump_scan, 2k run with MTP_SCAN_MEGA=1)
- The carriers do NOT render standalone: the scheduler FUSED them into adjacent
  GEMV/projection kernels (e.g. r_3_16_8_16_8_128_32_32_128_128_128_128_128_4_4
  = a dequant-GEMV with sigmoid gate that now carries 7 state-view args).
  No clean 6-arg elementwise carrier exists in the dump -> 1:1 targeting fails
  as written.
- FIX (one line, next session): barrier the carriers with .clone() before reuse —
  e_core = (...).clone(); e_state = (...).clone() — a clone renders its own copy
  kernel, breaking the fusion and producing the clean standalone carrier renders
  the matcher needs. THEN re-capture (cdump), decode layouts, adapt a4/gdn_scan.cu,
  wire matchers, validate, gate.
- The 2k scanmega run FAILED the gate as designed (placeholder carriers, no
  substitution) — expected; canonical (gate off) unaffected.

## 2026-09-10 second capture (clone barriers, fork c68032c, cdump_scan2)
- .clone() barriers did NOT change the renders (same 21 multi-state-arg kernels;
  the scheduler fuses through clones). Two capture sets banked: cdump_scan,
  cdump_scan2 (identical). KEY DECODE TARGETS for next session:
  * r_3_16_8_16_8_128_32_32_128_128_128_128_128_4_4 (out_49152 = 3x16384, 7
    state-view args, 366 lines, z=T sections) — likely projection+state composite
    created by the carriers; determine if the recurrence folds in.
  * E_4_16_16_3_16_8_3_2 (out_2359296 = 3x786432) — T-stacked state writer.
  * r_3_16_4_16_4_6_2_128_32_32_32_128_128_4_4 (out_36864, 6 state args).
  Decode workflow = the proven SKV one: read full source, map indices to the
  tensor exprs, verify against numpy, then decide the substitution target.
- Canonical unaffected (gate off). Machine healthy.

## 2026-09-10 DECODE VERDICT (primary targets, write-site analysis)
- r_3_16_8_16_8_128...: writes ONLY data0_49152 (8 sites); the 7 state bufs are
  READ-ONLY inputs (one scalar read each = carrier remnants). It is a projection
  GEMV fused with state reads — NOT the recurrence, BAD substitution target.
- E_4_16_16_3_16_8_3_2: PURE elementwise (zero reduce loops), writes data0_2359296
  at 3 slices of 786432 = the T=3 per-step packed-state stack. THIS is the legal
  injection site for the megakernel state outputs.
- CONCLUSION: NO composite contains the delta-rule recurrence — under MTP_SCAN_MEGA
  the loop never renders (state never evolves — why the gate run is garbage).
  The megakernel must INJECT the recurrence via substitution at (1) E_4 (write
  real per-step states) and (2) the core producer (find its writer next — likely
  another pure-elementwise render consuming e_core; check dumps for a 3x1536/49152
  consumer feeding ssm_norm). Megakernel reads q/k/v/beta/alpha from the E_4 arg
  set if present there, else extend the carrier to pull them into E_4 args
  (state*0 terms did not suffice — use full-tensor terms in e_state).

## 2026-09-10 core-writer candidates FOUND (cdump_scan2 out-numel sweep)
- E_3_8_8_2_16_8: out=49152, PURE elementwise (0 reduces), reads data1_98304
  (48x2048/block activation bundle = q|k sized) + 192 + 65536 — carrier-class.
- E_8_6_4_2_16_3_2: out=36864, PURE E, reads data1_73728 (CORE-sized!) + 144 + 65536.
- E_8192_32_3_3: pure copy 2359296->2359296 (state-stack copy); E_2304_32_32: pure
  fill of the state stack (1 write, no args).
- Sizes: 73728=48x1536 (T3 core packed), 49152=48x1024, 36864=48x768, 98304=48x2048.
- NEXT: read E_3_8_8_2_16_8 + E_8_6_4_2_16_3_2 sources fully, map to e_core/e_state,
  pick the injection kernel (must write core AND/OR state-stack), strengthen the
  carrier so q/k/v/beta/alpha all appear in ITS args, then adapt a4/gdn_scan.cu,
  matchers, standalone relerr, 2k gate, 100k gate.

## 2026-09-10 FINAL DECODE VERDICT — carrier route WALL
- E_3_8_8_2_16_8 and E_8_6_4_2_16_3_2 are GDN PROJECTION elementwise fusions
  (dequant signs + complex-rotation products on core OUTPUT side) — NOT the
  e_core/e_state carriers. The carriers were FULLY ABSORBED by the scheduler
  (their only traces: extra state-view READ args on GEMV composites + the
  step-state stash kernels E_4/E_2304/E_8192). NO standalone carrier render
  exists; .clone() barriers do not prevent the deep fusion.
- CONCLUSION: the carrier-injection route via tensor-expression renders is a
  WALL under this scheduler. Remaining unexplored options for the scan prize:
  (a) .realize() barrier inside _attention_mtp (capture-risk: breaks JIT input
  substitution patterns — needs a careful feasibility check);
  (b) substitute the BIG composite (r_3_16_8_16_8_128... GEMV + state reads):
  re-implement its dequant GEMV AND inject the recurrence in one kernel —
  heavy but 1:1-legal; the kernel is 366 lines, semantics half-decoded;
  (c) park scan (~15ms) and take K1-structural (~20ms) / T=1 draft PV first —
  both are substitution-friendly (established patterns).
- Canonical run39 (6.87, 60/60) unaffected by any of this (gate off).
