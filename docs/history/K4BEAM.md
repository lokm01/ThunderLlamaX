T=4 BEAM=1 FAULT ANALYSIS (2026-09-10): warm-cache DEAD (search still faults); fault-as-OOM in candidate test buffers (palloc+is_err_state, uncatchable post-hoc); routes: (1) MTP_BEAM_MAXGS env capping _time_program max_global_size in codegen/opt/search.py, (2) pre-launch candidate buffer validation, (3) compile-arena headroom. Iterate at 2k (each failure = reboot ~5min); 100k only for the gate. K=3 payoff: tok/cyc 2.86 measured, ~8.5 tok/s projected.
MTP_BEAM_MAXGS RESULT (2026-09-10): cap DID NOT prevent the fault — early-cap captures ran CLEAN (19.2s) then the run died with 219 fault lines during later renders. REFRAMING: the T=4 fault is NOT beam-timing-specific; it strikes any T=4 render at 100k in mtp_v3 context (same family as the historical commit2 T=2-at-100k fault). NEXT: bisect WHICH T=4 render faults (draft T=4? probe T=4 tail? commit?) via the early-cap phase ordering — the fault hit AFTER clean probe calls 0/1; add MTP_NODRAFT/K-gates to isolate; also consider skipping T=4 renders entirely by keeping T=3 probe with K=3 draft-only extension if mtp_v3 supports asymmetric K.

T=4/BEAM FAULT BISECT RESULT (decisive): probe T=4 captures CLEAN (both calls), commit captures CLEAN, select_final CLEAN — the DRAFT CAPTURE at K=3 faults (219 lines start exactly at [early-cap] draft call 0). NOT a T=4 render at all.
PRIME SUSPECT: the skv-t1 kernel (fires in the draft) — its atomicAdd on the baked-VA counter or its 80-CTA launch inside the K=3 draft-capture context (K=2 capture context was fine). NEXT (one run): K=3 BEAM=1 with MTP_SKV_T1=0 — if the draft capture goes clean, the T=1 kernel needs capture-safe handling at K=3 (options: skip last-CTA combine during capture via a capture-detect, or counter-in-ws-base trick, or plain two-kernel fallback for the draft only). If still faults: the K=3 draft render itself (dtail/slice K-variants).

DISCRIMINATOR RESULT: MTP_SKV_T1=0 IDENTICAL fault (219 @ draft call 0) — skv-t1 EXONERATED. The K=3 DRAFT-TAIL RENDER itself faults (draft JIT family at K=3 = the slice/dtail K-variant kernels = the fresh-T-at-100k render class, in the DRAFT not the probe).
ROUTES: (1) REORDER early-cap: capture the draft family FIRST (young allocator) before probe/commit — one mtp_v3.py edit of the early-cap sequence; (2) pre-render the K=3 dtail shapes standalone in a young context then capture (cache-hit path); (3) MTP_DTAIL_JIT=0 at K=3 (eager dtail — check if the eager variant survives at 100k).

REORDER RESULT (c1e192e): draft 0/1 captured CLEAN (1.4s/0.2s) — fault MOVED to probe call 0 (243 lines). BIDIRECTIONAL: only the FIRST fresh-render family survives capture; the second always faults (allocator-age dependent for ANY K=3 fresh render, not family-specific).
NEXT ROUTE (one edit): allocator youth between captures — free_cache()+gc BETWEEN each early-cap family (draft -> drain -> probe -> drain -> commit -> select), watching for the historical free_cache-fault correlation; alternative if drains fault: capture families in SEPARATE process phases via ckpt (draft-phase process, then probe-phase process resuming from the same ckpt).

DRAIN RESULT (a048c76): drains WORK — draft 0/1 CLEAN + probe call 0 CLEAN (18.3s, first time in the K=3 saga). Fault moved to ~probe call 1 (249 lines; last clean line = call 0). The allocator-youth mechanism is CONFIRMED and drainable per-family.
NEXT ITERATIONS: (1) drain between probe call 0 and call 1 as well (finer granularity: MTP_EC_DRAIN between EVERY capture call); (2) if the post-probe drain is itself the trigger (historical correlation), drain BEFORE each call instead of after; (3) once all captures clean -> the advance + steady cycles (new fault surface) -> gate.

PER-CALL DRAINS RESULT (497f893): ALL K=3 captures CLEAN - draft 0/1, probe 0/1, commit 0/1, select_final - the capture wall is FULLY BROKEN. Fault moved to the ADVANCE phase (351 lines during/after chunk prefill - the K=3 draft-fill fresh renders in the old allocator = same mechanism, next surface).
NEXT: drain before chunk-0 of the advance OR pre-render the advance draft-fill shapes during the young capture phase; then steady cycles as the third surface; then the gate. Each iteration ~8min+reboot.

ADVANCE SURFACE LOCALIZED: ALL 5 chunks clean (97280-97792/97810) — the fault is AT THE POST-ADVANCE TRANSITION into steady decode (third surface; the historical run19-era transition-fault point, whose existing drain does not cover K=3).
NEXT: (1) drain at the transition (extend the pre-head free_cache/gc site to K=3 runs + drain before the FIRST probe replay); (2) if the first replay faults: warm it via a probe_j replay DURING the young capture phase (a third cnt at capture time); (3) then the gate.
K=3 BEAM=1 @100k UNLOCKED (780c274): GREEDY 60/60 FULL, 5.54 tok/s, tok/cyc 2.86. All 3 fault surfaces broken (per-call drains + transition drain; advance clean). Recipe: run_100k_k3d2.sh (run39 + MTP_K3=3 MTP_EC_DRAIN=2 MTP_BEAM_MAXGS=8192). Remaining to 8.5 projection = cycle tuning only (probe-t 180, K=3 commit/sel), not faults.

K=3 ECONOMICS (definitive, warm-cache rerun identical): probe-t 179ms = the TUNED T=4 cost (T=3 is 130; +38% row scaling — the 160 projection was optimistic). Cycle 516ms = probe 179 + draft ~81 (3 steps) + head + sel/commit ~200 (the K-scaling commit: 38 percent of cycles pay m<3 re-forwards at alpha 0.62).
K=3 BEATS 7.28 ONLY IF: cycle <= 2.86 x 137.4 = 393ms -> need ~123ms of commit/sel/draft cuts. THE HISTORICAL ITEM IS NOW THE SOLE GATE: commit-cost elimination (per-step states via the megakernel/stash routes, or SELFIN-for-partial-m) benefits K=2 immediately (7.28 -> ~8+) AND unlocks K=3/4/6 after. RECOMMENDED ORDER: commit/sel cuts FIRST (K=2 wins now), then re-test K=3 on the same drain recipe.

STASH ROUTE PROGRESS: drains CURED the stash-at-100k FAULT (SS=1 runs are now clean OOMs, zero device faults). Blocker chain now: (1) stash arena 1.36GB @ 22.44GB used (the historical wall); (2) MTP_HEAD_DROP=1 fires (line prints) but the stash alloc STILL sees 22.44GB — the freed 2.54GB is re-consumed before the stash capture (suspect: lazy-head re-materialization on the SS=1 capture path, or the arena grows to absorb). NEXT: VRAM trace (BIGINIT/BIGBUF prints or allocator used-counter) from head-drop to the stash alloc with SS=1+DRAIN=2+HEAD_DROP — find the 2.54GB re-consumer; then the stash fits with margin and SELFIN extends to all m (commit eliminated; K=2 ~+1 tok/s; K=3/4 unlock).

VRAM TRACE RESULT (823/981 prints live): the drop fires ONCE at the EARLY-CAP site (the pre-capture site guard _HEAD_LAZY_ORIG already consumed -> its DROPPED+trace lines never run). 2.54GB freed there, yet the stash ask still sees 22.44GB => ~2.5GB is allocated BETWEEN the early-cap drop and the stash arena ask by the SS=1 capture path itself (per-step buffers + draft-select-slice + select_final builds).
NEXT: (1) allocation-stack capture at the MemoryError site (BIGINIT-style print_stack in the allocator except-path, env-gated) during SS=1+DRAIN+HEAD_DROP — names the 2.5GB consumer directly; (2) fix the used-counter accessor name (n/a = wrong attr; check MemPool internals); (3) candidates to defer: the select/slice builds could move AFTER the stash capture.

STASH FIT ACHIEVED (78facf0): head_j skip under HEAD_DROP -> the 1.36GB stash arena ALLOCATED (no MemoryError!) — the VRAM wall is BROKEN. New failure = later EXECUTION fault (device fault, not OOM): the headless steady path needs MTP_HEADPROBE=1 (derive cur/h_seed from probe replays — built for exactly this). NEXT COMMAND: sed MTP_HEADPROBE=1 into run_100k_ss3.sh (after MTP_HEAD_DROP=1) and rerun; if the stash captures + steady runs: SELFIN-for-all-m enable (mtp_v3 select path), then the K=2 gate ~8+ and K=3 on the drain recipe.

SS4 (with HEADPROBE): the 1460MB stash ARENA ALLOCATES inside probe call 0 (VRAM solution complete: HEAD_DROP + head_j-skip) — the fault is now IN THE STASH-CAPTURE EXECUTION (the piece-graph stash-store path at 100k; the original historical execution fault, resurfacing now that the allocation fits).
ROUTES for the execution fault: (1) the early-cap RO/SS ordering — try SS=1 WITHOUT DRAIN during the probe call (drain before, not per-call — the allocator state at stash-store exec may matter); (2) capture the stash family SEPARATELY FIRST (a tiny probe-call-0-with-SS-only in the youngest state, like the draft reorder); (3) the megakernel per-step-states (skip the stash entirely — outputs states directly).
STATE: K=2 canonical 7.28 untouched; K=3 drain recipe proven; stash route at the LAST fault (execution).
ROUTE 1 ELIMINATED (ss5): identical stash-execution fault WITHOUT per-call drains (187 lines post-alloc both ways) — this is the ORIGINAL allocation-history-dependent stash-STORE execution fault (the historical young-repro PASSED SS=1 at 100k; the mtp_v3 context delta still unidentified).
ROUTE 2 DESIGN (next): make the SS probe capture the VERY FIRST graph work after model load (before draft, before any warmup that allocates) — env-gated reorder + strip the pre-capture warmup to the minimum (the young-repro equivalence); if the young-context capture passes, the steady replay inherits it.
ROUTE 3 (parallel): megakernel per-step-states (skip the stash; a3b-style substitution of the scan writing states directly).
ROUTE 2 STATUS (276533c): reorder committed (EC_SS-gated block duplication; probe-first under SS). SS6 first run died PRE-early-cap (no early-cap lines; 141 faults during warmup — likely stale state from the pkill of the wedged prior run; NOT yet a clean route-2 datapoint). NEXT (after this reboot): clean rerun of run_100k_ss6.sh; if the pre-early-cap death repeats on a clean boot, bisect the SS-env combo at the warmup (MTP_EC_SS with HEADPROBE+HEAD_DROP during ckpt-restore/warmup = new interaction); the stash-exec fault itself is still the target behind it.

ROUTE 2 CLEAN RESULT: probe-FIRST order confirmed (probe call 0 = first early-cap line; the 1460MB stash arena allocates) — the stash-EXECUTION fault PERSISTS even in the youngest-order context. Route 2 ELIMINATED. The stash-store piece-graph execution at 100k faults deterministically regardless of order/drain/VRAM — the young-repro delta is NOT ordering; it is the mtp_v3 warmup/ckpt-restore code path itself.
REMAINING ROUTE: 3 = MEGAKERNEL per-step states (skip the stash entirely — the scan writes per-step states directly via an a3b-style substitution; builds on the SCAN_FUSION decode work). This is also the scan-fusion prize (~15ms) in one kernel family. THE convergence item for both commit-elimination and scan.
ROUTE-3 SESSION STATE: gdn_scan_ps written (per-step states = the stash bypass, convergence item). BLOCKED at launch-integration: (a) empty signature = silent no-op — now ALSO afflicts the OLD gdn_scan test on the current fork (repo test relerr 1.0; launch semantics evolved since f57590f); (b) typed signature hangs the device (infinite-loop signature; T param or arg binding). NEXT: bisect the launch — strip gdn_scan_ps to 2 args and add back; compare byte-for-byte vs k1t1 (works); check fork TinyELF signature changes since f57590f (git log device.py).
LAUNCH-BISECT COMPLETE (definitive): (1) fork semantics drift ROOT-CAUSED — fill_kernargs binds vals ONLY via signature entries (hcq.py:350 iter_sig over signature[-len(vals):]); empty signature = vals never written = garbage T — explains the old gdn_scan test failing on the current fork. (2) 2-arg typed-signature kernel PASSES (binding mechanism correct). (3) BOTH original gdn_scan AND gdn_scan_ps HANG/FAULT with typed signature on TODAY machine state — the kernel structure itself (smem staging + shfl_down/broadcast + 256thr) faults now though it passed 08-29 (machine has taken ~40 watchdog reboots since; the known post-crash-degradation suspect). -G builds = unsupported reloc 58 (fork ELF loader limit).
NEXT SESSION: (a) cold-power-cycle the eGPU (not just reboot — full unplug; the degraded-state protocol from AGENTS.md) then retry gdn_scan_ps typed; (b) if still faulting, rebuild the kernel without shfl_down (use xor only, the k1t1-proven pattern) and with TILE-smem staging like k1t1; (c) then the validation -> integration path is unchanged.
BISECT ROUND 2 (definitive eliminations): T-PROBE PASSES — 8 buffer args + typed sig + vals=(3,) delivers T=3.0 correctly and the kernel runs (marker written). Machine-state ruled out: T-probe passed THEN gdn_scan_ps hung on the same clean boot. kernargs sizing exonerated (2KB+ buffer). xor-shuffle variant ALSO hangs (shuffles ruled out). ps_smem (staging-only, no scan) faulted.
CONCLUSION: deterministic fault INSIDE the kernel body — remaining suspects: (1) the 2D shared arrays staging pattern (k1t1 uses ONE 2D half array; these use FOUR arrays incl 2x [8][128] float); (2) an smem+blockDim interaction in the staging loop shape.
NEXT BISECT (one compile each, on clean boots): A. gdn_scan_ps with staging REPLACED by direct global reads (no smem at all — keeps the full scan); if PASSES = the staging pattern; if HANGS = the scan loop itself. B. If staging: reduce to ONE shared array (merge k/q into one [8][256]) = the k1t1 pattern exactly. Then validation -> integration unchanged.

*** ROUTE 3 KERNEL VALIDATED (a05730a+): gdn_scan_ps_ns relerr 1.17e-07 (T=3, per-step states + outs + final) at 0.052ms/32-head-set. THE SMEM FAULT: every 4-array smem-staging variant hung; no-smem passes — smem re-add later via k1t1 single-array pattern only if needed for speed. NEXT: INTEGRATION — replace the _attention_mtp scan loop with a call to this kernel via the mtp_v3 dataflow (states explicit): the loop is at model.py _attention_mtp ~561-568; the kernel needs (state, outs-as-core, states_out, alpha, beta, q, k, v, T) — mtp_v3 can call it through NVProgram with persistent buffers between replays; SEE SCAN_FUSION.md routes (the carrier wall does NOT apply to a direct dataflow call from mtp_v3 — only to a3b substitution). Then SELFIN-for-all-m = commit elimination.

*** MODEL-DIMS KERNEL VALIDATED: gdn_scan_m (real dims 16/8/128) relerr 1.47e-07, 48.4us/block = 2.30ms/48-blocks. INTEGRATION DESIGN (next): the capture-safety question is THE gate — _attention_mtp runs inside the JIT-captured probe graph; a raw NVProgram call there = the historical fault. The capture-safe patterns available: (1) run the megakernel EAGERLY between graph replays in mtp_v3 (split the probe graph at the scan boundary: pre-scan graph -> megakernel -> post-scan graph) — costs graph fragmentation; (2) enqueue via the same cmdq path piece-graphs use (study how SKV kernels get captured — they are RENDERED kernels; the megakernel would need an a3b-style substitution which hits the carrier wall for the scan chain); (3) the drains-era discovery may make the old eager-call fault moot — TEST IT: a minimal _attention_mtp variant calling the megakernel directly under JIT=1 at 2k.

CAPTURE-REPLAY VERDICT (decisive, 5-min test): raw NVProgram calls inside TinyJit run at capture (call 0/1 = eager cnt0/1 phase) but are SKIPPED on replay (call 2: b stayed 2.0 when a=3.0). DIRECT-CALL ROUTE STRUCTURALLY DEAD — not a fault question; the JIT never records raw launches. SPLIT-GRAPH IS THE ONLY INTEGRATION: mtp_v3 must split the probe at the scan boundary (pre-scan TinyJit graph producing q/k/v/beta/alpha + state IN -> eager megakernel per block -> post-scan TinyJit graph consuming outs + per-step states). Design: probe_j becomes probe_pre_j + probe_post_j; the megakernel launches between their replays each cycle (48 launches x 48us = 2.3ms); persistent buffers carry the boundary tensors. The per-step states feed select_final directly = SELFIN-for-all-m.

*** SPLIT-GRAPH PRIMITIVES ALL PROVEN (order_test.py): (1) TinyJit .assign() into a persistent buffer = CAPTURED STORE (replays update it: warm values 111/112 tracked input); (2) an ASYNC raw NVProgram launch between two jit replays EXECUTES IN QUEUE ORDER (no host sync needed); (3) the following jit replay reads the FRESH value (replay 0: 115.0 = 5+10+100 exact; replay 1: 117.0 = 7+10+100 exact). NOTE: the script verdict line had an arithmetic bug (checked 217) — the per-line data is unambiguous PASS.
IMPLEMENTATION SCOPE (fully de-risked): the scan of block N depends on block N-1 post-scan (residual chain) => per-block fragmentation: ~48 GDN blocks each need pre-fragment (norms/qkv/conv -> persistent q/k/v/beta/alpha+state bufs) -> gdn_scan_m launch (async) -> post-fragment (reads persistent outs+per-step states -> output proj/FFN); the 16 attn blocks fold into whichever fragment. ~96 TinyJit families (NOBIND made count ~free; capture time ~same total kernels). Persistent buffer set per block: q/k/v/beta/alpha [16,T,128/8], state 16KB, outs [16,T,8], states_out [T,16,8,128] = ~70KB/block => ~3.4MB total. The per-step states feed select_final = SELFIN-for-all-m.
ORDER OF WORK: (1) build the split in mtp_v3 at 2k (start with ONE GDN block split, validate exact, then all 48); (2) 2k gate; (3) 100k + drain recipe; (4) SELFIN-for-all-m; (5) K-ladder.
ONE-BLOCK SPLIT SCAFFOLD (oneblock_scaffold.py): the structure is right (persistent bufs incl. core/states, pre_j/mid_launch/post_j, TinyJit captures) but pre_j MUST copy model.py _attention_mtp lines ~527-577 VERBATIM (the attn_gate/ssm_beta/alpha/log_alpha/conv/qkv/normalize+repeat/transpose chain — the scaffold simplified the q/k normalize and MISSES the num_k_heads repeat) then .assign() the five inputs + state into the persistent bufs. post_j: core = out_pb read (the outs go [nv,T,hv] -> reshape to the [B,T,nv*hv] core the ssm_norm consumes). Validation: run pre_j + mid + post on random x/conv/rec vs the STOCK _attention_mtp scan on the same intermediates — compare core + states exactly. THEN: repeat-per-block in _fwd3 (the 96-family rewrite).

*** ONE-BLOCK SPLIT VALIDATED (this commit): PASS, relerr 5.27e-07 core / 1.33e-07 final state, REAL weights. DIM CORRECTION: nv=48 hv=128 hk=128 (786432 = ONE block state; per-step states for all 48 blocks = 48xTx3MB = 432MB at T=3 — VRAM plan needed, fits in the head-drop freed 2.5GB). All remaining work: per-block _fwd3 rewrite; persistent bufs per block; 2k gate; 100k+drains; SELFIN-for-all-m; K-ladder.
TRUE LAYER DIMS (verified from the loaded model): inner=6144 conv_ch=10240 q_dim=2048; nv=48 hv=128 hk=128 nk=16; ssm_norm(128) ssm_out Linear(6144->5120) attn_gate Linear(5120->6144) FFN 17408. ALL post_j math checks out on paper (core 6144 -> 5120). The (1,3,10240) broadcast is ONE PRINT from found: add shape prints of core/z/attn_out/h in post_j at capture (cnt0 prints survive) — the 10240 must be a stale pool entry (gate stored before a reshape fix in an earlier run created _pool with the wrong shape and the module caches _pool across runs IN-PROCESS — if chain_test ran multiple times in one process, stale) or the pre_j v2-split feeding conv-shaped data. ALSO: the kernel launches had a subtle bug risk — _mid uses P[q].shape[0] for grid; verify nv=48.

2K SPLIT FIRST RUN: probe-split fired at pos=14 (the split fragment chain RAN at 2k),
then a device fault during the chunk-prefill advance (1 chunk logged). The split path
itself executed. The fault is in the INTERACTION: fwd3_split does not write back the
advanced states to the live b.recurrent_state/b.conv_state — the advance/prefill
machinery still owns the stock state flow, so states desync.

ROOT ISSUE: no state write-back from the split pool.
FIX (b) MINIMAL-RISK FIRST GATE: scratch semantics — fwd3_split seeds from live state
each probe, never writes back; the stock commit/select machinery stays 100% unchanged
(the split replaces only the VERIFY forward). This is exactly the chain_test contract
already validated (3.08e-04). FIX (a) later: write-back for the decode fast path.
Also note: the run only logged ONE probe-split call then faulted in the chunk loop —
the MTP_CHUNKED_PREFILL path also runs probe-equivalent forwards? No: pos=14 = the
first verify after prefill; the fault came right after in the advance for cycle 2.
Scratch semantics should eliminate it (no state divergence).
DESYNC SOURCE IDENTIFIED: the eager attn-block fragments advance cache_kv IN PLACE with NO ROLLBACK — the stock probe path wraps cache_kv writes in the captured graph whose rollback machinery mtp_v3 drives on m<K rejects. The split faults right after the FIRST probe because cycle 2 needs the rolled-back KV.
FIXES ordered: 1 = attn-KV snapshot/restore around each split probe — 16 layers small, clone before, assign back per the m/K decision, copy the existing GDN snapshot pattern ~15 lines in the mtp_v3 split branch. 2 = attn fragments through the stock jitted path per block. 3 = T3_LAZY symbolic-sp attention for fragments. FIX 1 is smallest.
2K SPLIT RUN 3: NO FAULTS, 47 probe-split calls, EXIT=0 end-to-end — the split path RUNS in the live pipeline. GREEDY 3/60: outputs wrong after early tokens.
DIAGNOSIS: double KV advance on accepted cycles — the m==K path KEEPS the probe eager KV writes AND the commit re-forwards the same tokens through the stock attention = KV advanced twice. The commit re-forward provides the authoritative advance; the probe eager KV writes are always scratch under the split.
FIX: kv_restore on EVERY cycle under the split gate.
RUN 4 (kv_restore every cycle): identical 3/60 — AND the stock-path 2k control ALSO failed today (1/60 on the same env family) => THE 2K GATE ENV ITSELF IS BROKEN (baseline mismatch: spec_base_2k.json predates today env drift; the base emits [4478,9826,...] while the model produces [1049,...] under this config). The 3/60 is the broken-gate floor, NOT a split verdict.
DECISION: the definitive gate for the split is 100k (canonical env + spec_base_100k.json = the solid baseline, 7.28 verified 3x today). NEXT SESSION COMMAND: sed MTP_SCAN_SPLIT=1 into run_100k_skv.sh -> run -> GREEDY check + cycle split. The split path is END-TO-END CLEAN at 2k (0 faults, EXIT=0, correct cycle mechanics: m=0/2 and m=2/3 decisions flowing, acceptance firing).
ALSO BANKED THIS TURN: FIX-1 complete (4029980: KV snapshot + restore at every cycle c1aee54); SELFIN-skip under split (63dc045 — no probe_scratch states in the split path; the m==K fast path falls to re-forward until the pss-based SELFIN lands).
100K SPLIT GATE: all 5 chunks clean, early-cap clean — the TRANSITION into the split decode hit a clean VRAM OOM (392MB ask at 23.31GB used): the split path allocates per-block persistent pools (48 blocks x ~9.5MB pss+state+io at T=3 = ~460MB) ON TOP of the canonical residents. The headroom needs MTP_HEAD_DROP (frees 2.54GB; requires head_j skip = both already landed this session) or fewer persistent bufs.
NEXT SESSION: run_100k_split.sh + MTP_HEAD_DROP=1 (+MTP_HEADPROBE=1 for the headless steady path) — the VRAM math then fits with ~2GB margin. The gate is ONE ENV VAR away; all mechanics verified clean to the transition.
100K SPLIT +HEAD_DROP RUN: early-cap probe calls 0/1 CLEAN (16.9s/7.5s) then fault at the COMMIT capture (commit call 0) — NOT the split path (the split decodes never started; the probe captures here are the STOCK early-cap calls, the split activates at decode). The HEAD_DROP+HEADPROBE combo under this env faults the commit capture — distinct from the earlier stash-route runs where commit captured clean WITHOUT HEADPROBE.
NEXT DISCRIMINATOR (one run): run_100k_split2 minus MTP_HEADPROBE (HEAD_DROP only) — the earlier SS2/SS3 runs showed HEAD_DROP alone + head_j skip works and the stash arena allocates; if the commit capture passes, the transition OOM is also gone (2.54GB freed) and the split decode starts.
HEAD_DROP-ONLY RUN: identical — early-cap draft/probes CLEAN, fault at COMMIT capture. HEADPROBE exonerated: the fault is the HEAD_DROP + commit-capture interaction (the head_j skip changes what the commit family binds; the commit capture at 100k faults without the head). NOTE: the canonical SS runs (which captured commit clean) had HEAD_DROP only in runs where MTP_EC_SS=1 was ALSO on (the stash context) — the plain-SKV + HEAD_DROP + commit-JIT combo is the faulting configuration.
ROUTES: (1) reorder: capture the commit family BEFORE the head drop (the drop site moves post-commit-capture); (2) or drop the head AFTER all early-captures (move the drop to the transition); (3) or skip HEAD_DROP and shrink the split pool instead (pss buffers only for the blocks needed by select; ~230MB at T=3 if half) — the OOM was 392MB ask; a 230MB pool + free_cache drains may just fit at 23.31-0.23.
FASTEST: route 3 (no new fault surface): reduce pss to fp16? NO (numerics). Reduce to per-block LAZY pss alloc at first probe-split call (after the transition drain) — the pool allocates only when the split first runs = post-transition, where free_cache has run. ~10-line change in fwd3_split._bufs_for.
STOCK CONTROL (current mtp_v3, SCAN_SPLIT off): 60/60 @ 7.25 tok/s — the canonical path is INTACT through all my wiring edits; the commit capture passes. CONCLUSION: the commit-capture fault is specific to the SCAN_SPLIT=1 env being SET — and the only place the env changes capture behavior is... nowhere in the early-cap path directly. PRIME SUSPECT: the SELFIN-skip edit changed the m==K branch condition — but commit capture runs BEFORE cycles. UNLESS: the fault is order-dependent — the split4/5 runs included the fwd3_split POOL DRAINS (d335b6f) which fire at bi%8==0 during the FIRST split probe... no, the split never ran (fault at commit capture).
WAIT — the REAL delta found: run_100k_split*.sh all have MTP_EC_DRAIN unset while stock passes — NO. The actual difference: my split scripts came from run_100k_skv.sh via sed — IDENTICAL except SCAN_SPLIT (+HEAD vars). The env MTP_SCAN_SPLIT=1 itself must gate something in the FORK (model.py! MTP_SCAN_MEGA-style code reads envs at import). CHECK: grep MTP_SCAN_SPLIT in the fork — my earlier MTP_SCAN_MEGA carrier code in model.py reads envs INSIDE _attention_mtp at CAPTURE TIME — if MTP_SCAN_SPLIT trips any fork-side branch, the captured schedule changes.
NEXT: grep -rn MTP_SCAN_SPLIT ~/tinygrad-src/tinygrad/ — find any fork-side reader; if none, the env affects only mtp_v3 (and then the fault must be in an mtp_v3 capture-path read — grep mtp_v3 for SCAN_SPLIT outside probe()).
ROOT CAUSE OF RUNS 2-4 FOUND: MTP_HEAD_DROP=0 is PYTHON-TRUTHY (getenv returns the string 0) — run4 ACTUALLY RAN THE HEAD-DROP (identical to run3). All three commit-capture faults = the HEAD_DROP binding interaction. Run 1 (no HEAD_DROP at all) PASSED all captures and OOMed only at the split-pool — with the d335b6f drains now landed, the TRUE test is run5 = run1 + drains (NO HEAD_DROP VAR AT ALL).
RUN 7 (windowed snapshot): THE SPLIT RAN AT 100k — probe-split pos=97810, 97811 (the first split decodes at the real 100k position!) then faulted at/after the 2nd. The transition PASSED (no OOM — the windowed snapshot fixed it). The split decode executed at least 1-2 cycles at 100k.
FAULT ANALYSIS: 2 probe-split calls logged; the fault is in cycle 2-3 of the split decode — the first split-graph fragment captures at 100k in the post-transition allocator context (the SAME fresh-T-at-100k-render class as the historical T=4/draft faults: 96 new TinyJit families rendering at 100k = a huge fresh-render surface). The per-call drains apply to the POOL alloc but the FRAGMENT CAPTURES (96 families x ~30 kernels each at 100k shapes) hit the known render-fault class.
ROUTES: (1) MTP_EC_DRAIN=2 in the split recipe (the proven per-call drains around the fragment captures — add drains INSIDE fwd3_split between blocks during the FIRST pass); (2) the split first-pass at a YOUNG allocator: force the split decode to initialize its fragments at the transition with a full drain before each block-group (8-block groups = 12 drains); (3) if the render faults persist: the K=3 drain recipe pattern (drain before every capture call) applied to all 96 fragments on first pass.
RUN 8 (capture drains): identical to run7 — probe-split 97810/97811 then fault. The per-group capture drains did NOT extend past cycle 2. ANALYSIS: the fault is NOT the fresh-render allocator class (drains would have moved it); it repeats exactly at the 2nd-3rd split cycle deterministically. NEXT SUSPECT: the mid captures complete but the 2nd CYCLE hits the eager attn fragments REPLAYING with the kv window restore... or the T=3 fragments replay fine but the LOGITS post-hoc path (model.output on the split h — EAGER lazy-head at 100k = the eager-lazy-head arena class from the memory notes!) faults on the 2nd call. CHECK NEXT: which python line faults on cycle 2-3 (add a [split-cycle] print per stage; the last frame before the fault — the tail showed ops_nv sleep so the GPU errored during an async op; the offending launch needs stage prints).
RUN 9 STAGE PRINTS (definitive): the fault is IN model.output_norm(h.half()) — fwd done + synced post-fwd PRINT; norm done NEVER printed. The eager output_norm on the split h faults at 100k on the FIRST split cycle (pos=97810) — the eager-lazy-head/arena class (the memory notes: eager calls on the split-h materialize arenas). The 2 earlier probe-split calls at 97810/97811 in prior runs = the fwd completed twice then died at the same norm.
ROOT: output_norm EAGER on a [1,3,5120] half tensor should be tiny — unless the .half() cast on h or the realize triggers a big arena. NEXT (one edit): reuse the STOCK head path — the early-cap already captured head_j-style machinery; or compute logits INSIDE the last fragment (post_j of the final block + output_norm + head as ONE TinyJit — capture-safe, no eager realize). The cleanest: fold norm+head into a tiny TinyJit(_head_fn) captured on the first split cycle.
RUN 10 (jit-head): head done PRINTED (1x — the first split cycle COMPLETED end-to-end: fwd + norm + head all clean!) then EXIT=1 at/after the first post-split stage (2 probe-split calls logged — cycle 2 started). The jit capture FIXED the eager-norm fault. The next fault is downstream in cycle 2 (the argmax path or kv_restore or the cycle machinery interacting).
PROGRESS CURVE THIS STRETCH: cycle 0 partial (run7/8: fwd only) -> cycle 1 COMPLETE (run10: fwd+norm+head). The fault surface is receding one stage per run. NEXT: stage prints AFTER head (argmax/cycle-2 entry/kv_restore/draft) + check the am path.

RUN 11 (definitive stage map): cycle 1 FULLY CLEAN — fwd, head, post-head sync,
ARGMAX, post-am sync, kv_restore all printed. Cycle 2 STARTED (probe-split
pos=97811) and the fault is INSIDE the cycle-2 fwd3_split (no cycle-2 fwd-done
print). The fault = the SECOND split forward pass — the fragment REPLAYS plus
the 2nd-pass eager attn cache_kv appends.

ANALYSIS: cycle-1 = the capture pass (TinyJit cnt0, everything eager-legal and
clean). Cycle 2 = the first REPLAY pass. Suspects: (a) attn block_j fragments
replaying; (b) eager mid kernels interleaved with graph replays (queue-order
test said OK); (c) assign-store replays racing the attn fragments across the
96-jit + 48-mid chain.

NEXT DISCRIMINATOR: per-block [frag] prints inside fwd3_split every 8 blocks —
localize the cycle-2 fault to a block index, then compare that block's replay
vs capture.
RUN 12 (FRAGDBG, definitive localization): cycle 1 captured all blocks (blk4..blk60 = 16 GDN frag groups logged). Cycle 2 STARTED (probe-split pos=97811) and reached [frag] blk0 start — the fault is AT/INSIDE BLOCK 0 OF THE CYCLE-2 REPLAY (the first fragment chain replay: pre_j(blk0) replay + the gdn_scan_m mid + post_j replay, or the leading attn fragments blk0-3 which come before the first GDN).
BLOCK-0-REPLAY ANALYSIS: blk0-3 are the leading DENSE (attention) blocks — block_j replays of b(x, start_pos). The first replay of an attn block_j at 100k with the kv window RESTORED... suspect: the attn fragment replays append to cache_kv AGAIN (they are eager-python b(x, sp) INSIDE block_j? NO — block_j captured b(x, sp) so replays do NOT re-run python; the graph replays INCLUDING the kv append — and the KV was RESTORED (shrunk back) — the graph re-appends = correct-by-design. OR: pre_j(blk0) is the FIRST TinyJit whose input x changed shape/identity vs capture (cycle-2 x = post_j(blk63) output of cycle 1 = a NEW tensor identity — TinyJit input substitution should handle it...
NEXT: finer prints INSIDE block-0 region (attn blk0-3 block_j call vs first pre_j) — one more run pins attn-vs-GDN; then the fix per class.

RUN 13 (symbolic-sp fix): PROGRESS — cycle 2 now reaches blk8 (was blk0): the
sp fix moved the fault TWO+ fragment groups deeper. Fault surface receding
block-by-block. Current: cycle-2 fwd fault at ~blk8-11 (first GDN group).

PATTERN: each fix reveals the next fault one group deeper — blk0 (attn,
sp-fix) -> blk8 (GDN replay region). GDN replay suspect: the pre_j REPLAY
re-runs .assign() stores into pool bufs while the previous cycle's eager mid
kernel may still be in flight, or the mid launches with stale kernargs.

NEXT: Device[NV].synchronize() between pre_j and _mid in fwd3_split
(serialized but diagnostic); if clear, optimize to queue-ordering later.

RUN 14 (SYNCMID serialized mids): fault moved blk8 -> blk4 — the sync helped
one more group (attn blocks 0-3 cycle-2 replay CLEAN; blk4 = the first GDN
block region). PATTERN CONFIRMED: correctness fixes walk the fault deeper
each run. blk4+ = the first GDN pre_j/mid/post chain replay.

blk4 GDN suspect: the first pre_j REPLAY. Its captured .assign() stores write
pool bufs. conv_in is fresh (the stock commit advances it). The x input =
cycle-1 post_j output = a NEW tensor identity each cycle; TinyJit input
substitution should rebind by slot. The blk4 print fired BEFORE pre_j — the
fault is inside pre_j replay (substitution or store replay) or mid/post.

NEXT: sub-stage prints inside the GDN branch (pre_j-called / mid / post) for
bi < 8 — one run pins the exact sub-stage.

RUN 15 (sub-stage prints, DEFINITIVE): cycle 2 executes gdn1, gdn2 FULLY
(pre/mid/post all print) then [gdn4] pre_j-call + pre_j-done print and the
fault fires at the SYNCMID sync after block 4's pre_j REPLAY (mid-call never
prints). gdn4 pre_j-done appears exactly 2x = once in cycle-1 capture, once in
cycle-2 replay. THE FAULT = the pre_j GRAPH REPLAY of block 4 faults (async,
surfaces at the following sync).

WHY BLK4: gdn4's input x = the output of attn blk3's block_j REPLAY (the
first GDN whose input comes from an attn-fragment replay); gdn1/gdn2's inputs
came from GDN post_j outputs. Suspect: the TinyJit input substitution of
pre_j(blk4) receiving a graph-replay-output tensor (different buffer identity
than capture) — substitution may rebind the buffer but the CAPTURED ASSIGN
targets (pool bufs) are fine; or the blk3 block_j replay output buffer aliases
something pre_j4 captured.

NEXT: (a) force a contiguous realize on x between attn and GDN fragments
(x = x.contiguous().realize() already exists per-block — check ordering); (b)
dump pre_j4.cnt and input shape on cycle 2; (c) if substitution is the issue:
make pre_j take x via a STABLE buffer (copy x into a persistent input buffer
before pre_j — one extra elementwise per block, diagnostic first).

RUN 16 (stable input buffers): IDENTICAL fault position — gdn1/gdn2 replay
clean, [gdn4] pre_j-call + pre_j-done, fault at the post-replay sync. Stable
input buffers did NOT fix it => NOT the TinyJit input substitution of
replay-output identities.

REMAINING SUSPECTS for blk4 pre_j replay specifically:
(1) blk4's pre_j graph is the FIRST fragment whose capture included the
    attn->GDN transition shape/layout: check whether pre_j(blk4) captured a
    DIFFERENT input dtype/layout than the stable buffer provides (x is fp32
    [1,3,5120] from _xbuf assign; capture saw the same => unlikely).
(2) The REAL structural difference: blk4's conv_in = b.conv_state which on
    cycle 2 was updated by the STOCK commit re-forward (T=1 eager) — the
    conv_state BUFFER identity/content changed between capture (pre-commit
    machinery) and replay (post-commit). pre_j captured binding to the
    conv_state buffer captured at capture time; on cycle 2 conv_in is the
    same tensor object but its buffer may have been reallocated by the eager
    commit forward. TinyJit captured the OLD buffer; replay reads STALE/
    FREED memory => fault. THIS FITS: gdn1/gdn2 replays are clean because
    their pre_j captures read conv_state buffers that the T=1 commit
    re-forward rewrites IN PLACE (same buffer); blk4's might differ... OR
    more simply: the eager T=1 commit path FREES pool-adjacent buffers.
NEXT DISCRIMINATOR: also route conv_in and rec_in through stable per-block
buffers (same _xbuf pattern) — if the fault moves again, it's state-buffer
identity; and dump Device allocator state at the fault.

RUN 17 (stable STATE buffers): the fault MOVED — now during CYCLE 1 (only 1
probe-split call; prints end at [gdn6] post-done; gdn7 never printed). The
stable-state buffers changed the capture behavior itself: the per-cycle
_cv/_rc copies (assign+realize eager per block) altered the capture-time
schedule enough to fault at the gdn6-7 capture, OR the .cast(t.dtype) path
created a dtype/layout the capture mishandles.

STATE OF THE WALK: the fault surface responds to every change (proof the
machinery is close), but the state-buffer routing introduced a new capture-time
fault. Options next: (a) make _stbuf dtype-stable (allocate as t.dtype
directly, not fp32-then-cast); (b) revert the state buffers (keep input
buffers) and instead run the discriminator at 2k where cycles are cheap —
debug the blk4 class at 2k (the class is context-independent if it's a
buffer-identity issue; 2k runs take 4 min not 20).

DECISION: (b)+(a): revert state-stable to dtype-direct alloc, and shift the
debug loop to 2k (runs are 5x faster; the class should reproduce at 2k since
cycle 1-2 mechanics are identical).
2K DEBUG RUN (dtype-direct stbuf + SYNCMID + FRAGDBG): THE SPLIT RUNS ALL 47 CYCLES END-TO-END AT 2k — ZERO FAULTS, EXIT=0. The dtype-direct fix (or the 2k context) cleared the gdn6 capture class. GREEDY 1/60 = the same broken-2k-gate floor as the stock control (1/60-3/60; spec_base_2k.json predates env drift). The split is NOW END-TO-END CLEAN at 2k.
MEANING: full multi-cycle split decode (capture + 46 replays + interleaved mids + kv window restores + the whole cycle machinery) runs clean. The remaining questions: (1) does the 100k run pass the same way (the blk4 class may be 100k-specific after all); (2) correctness — needs a VALID baseline. The 2k baseline is broken; the 100k baseline (spec_base_100k.json) is SOLID (7.28 verified 3x).
NEXT: the 100k split gate with the dtype-direct stbuf (one run answers both). If it faults at blk4 again -> the class IS 100k-specific (allocator state), route = the 2k-style serialization at 100k; if it completes -> GREEDY vs the solid baseline.

100K GATE (post-2k-clean): the fault is 100k-SPECIFIC and moved again — the
cycle-1 capture now reaches blk52 (was gdn6 in run17) before faulting; the
cycle-1 prints show gdn6 mid/post CLEAN at 100k too (the dtype-direct fix
cleared the capture class at 100k as well). The remaining fault = deep in the
first-pass captures (~blk52-63 region = late blocks) at 100k only.

PATTERN CONSOLIDATED: 2k = all 47 cycles clean; 100k = first-pass capture
faults deep in the block chain (blk4 -> blk8 -> gdn6 -> blk52 as fixes
landed). The late-block capture fault at 100k = the SAME fresh-T-at-100k
render class the whole session has fought (the 96 families × 100k shapes).
The K=3-proven weapon = MTP_EC_DRAIN-style drains — the capture drains
(d335b6f) fire every 8 blocks ALREADY... but the fault is now PAST blk48.
Try: denser drains (every 4 blocks) at 100k only — or the young-allocator
route (capture the split families FIRST, before the stock early-cap).

THE YOUNG-ALLOCATOR ROUTE (probably right): the split fragments currently
capture at the TRANSITION = the OLDEST allocator state (after prefill + all
stock captures). The historical fix pattern: capture the young families FIRST.
For the split: initialize the fragments during the early-cap phase (a dummy
split call at sp=_start right after the stock captures with drains around it).
That is ~5 lines in mtp_v3's early-cap block.

100K YOUNG-CAP GATE (a6caa05): the split-fragment young capture ran in the
early-cap phase — prints show the fragment chain reached blk52 during the
YOUNG capture too ([frag] blk52 start is the last frag print; gdn6 post-done
the last gdn print = the same region as before). The fault did NOT move =>
the young-allocator hypothesis is WRONG for this class: the fault is NOT
allocation-state; it is deterministic at ~blk52 capture REGARDLESS of
allocator age.

CONSOLIDATED FACTS: (1) 2k: all 96 families capture + 47 cycles clean;
(2) 100k: capture faults at blk52 in BOTH young and old allocator states;
(3) the walk blk4->blk8->gdn6->blk52 tracked the dtype/sp/sync fixes —
those fixed REAL bugs, but the blk52 wall remains. blk52+ = the LAST ~12
blocks. Distinguishing feature of the tail blocks: their fragment renders
happen after ~52 blocks of rendering — a CUMULATIVE effect (graph count,
mapping count, or compile-server state), not per-block. This matches the
HISTORICAL MTP_MAPFD cap class (the ~128 MAP_SYSMEM_FD mapping cap that
NOBIND fixed for the stock) — the split's 96 extra families × per-family
maps may hit a NEW mapping ceiling at 100k shapes (2k shapes map less).

NEXT: measure the mapping count during the young capture (the fork's
MAPFD_DIAG / _MAPFD_TOTAL counter) — if it saturates near the fault, the fix
is NOBIND-everything for the split fragments (they already go through TinyJit
= should be NOBIND... verify MTP_GRAPH_NOBIND applies) or chunked family init
(capture families in 2 processes via ckpt).

MAPFD MEASUREMENT (964668b): FLAT AT 43 the whole way (blk0 through blk52)
=> the mapping-count hypothesis ELIMINATED. NOBIND is holding for the split
families (43 total maps, constant). The blk52 wall is NOT mappings.

ELIMINATED SO FAR for the blk52 100k capture fault: allocator age, input
substitution, state-buffer identity, mapping count, dtype/cast paths, sp
binding, mid-launch races (SYNCMID). REMAINING SUSPECTS: (1) the COMPILE
SERVER state (96 families x ~30 kernels = ~2900 fresh compiles at 100k
shapes through the nvcc container — a compile-server/vrdbg-level resource:
check for compile timeouts or wedged compile workers near blk52); (2) the
HCQ signal/timeline count (each family capture creates timeline signals —
a per-device signal-pool ceiling); (3) cache.db contention.

The compile-server suspect fits the CUMULATIVE pattern best (the 2k shapes
compile fast; 100k shapes are slow/big — the nvcc container has the
immortal-sleep fix but memory growth across ~2900 compiles could wedge it).

NEXT: watch the compile worker/container during the run (docker stats /
nvcc shim log), OR simply CHUNK the family init: init families 0-51 in
process A (ckpt), then 52-95 in process B resuming from ckpt — 2-process
capture sidesteps any per-process cumulative resource. The ckpt machinery
exists (2-min resumes).

COMPILE-SERVER WATCH (87c6405): the nvcc container is HEALTHY AND IDLE the
whole way (mem 1.9MB flat, cpu 0% at every 8-block snapshot through blk48;
the run died before the blk52 snapshot printed). COMPILE-SERVER HYPOTHESIS
ELIMINATED — the container isn't even being hit at capture time (the kernels
come from cache.db / the fork's in-process NVRTC path; the container only
serves my hand-CUDA compiles).

ALSO: cpu=0% at the frag prints means no compiles were in flight at those
moments. The blk52 wall is not a compile resource.

REMAINING SUSPECT (effectively the last): the per-device HCQ SIGNAL POOL /
timeline counter — 96 families x per-family signals at capture time is a
dext-level resource the 2k run doesn't stress the same way (its signals are
smaller-count per capture? both are 96 families... BUT the 100k captures
are LARGER graphs = more piece-graphs per family = more signals).

DECISIVE NEXT MOVE (the directed chunked 2-process init also tests this):
proc A captures 0-51 + ckpt; proc B resumes (2 min), re-renders 0-51 as
cache HITS (no fresh renders), then 52-95 fresh + decode. If the wall is any
per-process cumulative resource (signals, IRAM, graph objects), proc B's
fresh-render count = 44 < 52 = clears the wall. Implementation: MTP_SPLIT_CAP_LIMIT env in fwd3_split (capture families only up to the limit
on pass 1, raise the limit per process); mtp_v3 ckpt-save after the partial
capture and an env to skip re-capture of already-cached families (they're
cache-hits anyway since kernels persist in cache.db).

2-PROCESS CHUNKED INIT RESULT (b182f3e): proc A captured families 0-47 cleanly
and exited (cache.db warm). Proc B (the full gate) STILL FAULTED at the same
position — its frag prints end at blk48 (the same wall; the cache-hit
re-render of 0-47 passed but the fresh captures beyond 48 fault at the same
cumulative count). DEFINITIVE: the wall is a PER-PROCESS GRAPH-CAPTURE-COUNT
ceiling (~48-52 fragment families captured in one process at 100k shapes) —
NOT compile-side (cache hits did not help), NOT allocator, NOT mappings.

This is the SAME class as the historical "one JIT=1 graph family limit" that
NOBIND partially lifted — a dext-level per-process limit on live captured
graphs, now hit at ~50 families x 100k sizes (2k sizes fit under it).

REMAINING ROUTES (hard-scoped):
(1) REDUCE THE FAMILY COUNT: merge fragments per block (pre+post in ONE jit
    per block = 48 families instead of 96 — the mid launches between them
    make this hard) OR capture attention blocks as FEWER jits (chain all 16
    attn blocks into 1-2 families since they have no eager mid between them:
    attn blocks are pure graph -> ONE TinyJit covering all 16 = 96 -> ~64
    families. 64 may still exceed ~52... chain attn into 1 family = 49 total
    (48 GDN pre/post + 1 attn mega-fragment) = UNDER the wall.
(2) The 2k-style per-family process churn is impractical for decode.
(3) Ask the fork maintainer about the graph-count limit at 100k shapes.

ROUTE 1 IS THE MOVE: the 16 attn blocks are consecutive? No — they interleave
with GDN. But attn fragments can be captured as ONE family only if adjacent.
Alternative arithmetic: pre_j+post_j per GDN block are separated by the eager
mid — but post_j(bi) and pre_j(bi+1)... also separated by mid(bi+1). The mids
force per-block boundaries. UNLESS mids also batch: launch all 48 mids for a
GROUP of blocks after their pre_js — the gdn_scan_m for block N only needs
pre_j(N)'s outputs; post_j(N) needs mid(N)'s output. Sequence: pre(0..7) ->
mid(0..7) -> post(0..7)||pre(8..15)? post(N) feeds pre(N+1) — serial. BUT:
capture pre_j for 8 blocks in ONE TinyJit (8 blocks of projections = 1
family), then 8 mids, then post for 8 in ONE TinyJit = 6 groups x 2 = 12
families + attn merged where possible = ~20 total. THIS restructure clears
the wall with margin.

MERGED-V3 RUN (0ab3a75): a clean VRAM OOM, not a device fault — 4.74GB ask at
22.54GB used during the merged young-cap capture (the merged families are
LARGER graphs: each [post+attn+pre] capture materializes more intermediates
at capture time than the small families did). The 100k capture wall did NOT
fire before the OOM — the family-count reduction changed the failure mode to
a budgetable VRAM issue (the same class as the earlier transition OOMs).

VRAM MATH: the merged captures need headroom. Options: (1) MTP_HEAD_DROP
(frees 2.54GB) — but it faulted the commit capture in the SS context... the
split runs differ (MTP_SEL_JIT1 still on); worth one run; (2) move the merged
young-cap AFTER the stock captures with a full drain (it already is after;
add another drain); (3) halve the group size (family count ~74 — back over
the wall). PRIORITY: (1) then (2).

STATUS: the merged structure is COMMITTED and mechanically sound (parses,
loads, reaches the capture); its first 100k run OOMed before testing the
wall. Next session: rerun with MTP_HEAD_DROP=1 (accepting its known
commit-capture risk in this different context) or a pre-merged-capture drain.

MERGED + HEAD_DROP RUN: the HEAD_DROP+commit-capture fault recurred (fault at
'[early-cap] commit call 0' — the known HEAD_DROP interaction from the SS
campaign; head_j skips change the commit family's binding). HEAD_DROP is NOT
viable in this context either.

MERGED-V3 STATE (ef974d0 + this): the merged families structure is committed
and sound; its VRAM needs exceed the no-head-drop budget by ~2GB at capture.
The next options: (a) pre-merged-capture SUPER-drain (multiple free_cache +
gc cycles + a cache purge of the stock-capture intermediates); (b) capture
the merged families in a SEPARATE process (chunked: proc A captures merged
families 1-25 + exits; proc B does 26-50 + decode — the family count per
process halves AGAIN, ~25 each, far under the wall, AND each process's
capture VRAM is fresh); (c) reduce merged group VRAM: the capture
materializes intermediates for post+attn+pre in one graph — splitting the
attn run OUT of the merged jit (attn as its own small families where they
occur in runs) halves the merged capture size at the cost of +16 families =
~66 total... over the wall. (b) is the robust route and reuses the
MTP_SPLIT_ABORT_AT chunking already built.

DECISION: (b) — chunked merged capture: proc A MTP_SPLIT_ABORT_AT=25 (merged
families 0-24 = lead + m0..m23), exit; proc B full (cache-warm ELFs; fresh
captures ~25 = far under the wall; capture VRAM fresh process = no OOM).

MERGED CHUNK-A (ABORT_AT=25): the SAME 4.74GB OOM at 22.54GB — the OOM fires
during the merged young-cap BEFORE reaching family 25 (the merged captures
are each ~2x bigger; the OOM is in the FIRST few merged families' capture
arena, not cumulative). ANALYSIS: 4.74GB = a single allocation (likely a
lazy-head or big-intermediate arena inside the first merged capture). The
merged [post+attn+pre] graph's capture materializes the intermediate residual
chain (attn block outputs) — with the eager .assign stores and stable buffers
creating copies. THE FIX PATH: the merged capture needs the EMB_GATHER-style
discipline applied to its internals OR the intermediate x copies reduced (the
_xbuf/_stbuf stable-buffer copies each add a materialization).

GIVEN THE SESSION ARC (19+ hours, the split is one arena fix from the chunked
gate), THE HONEST CLOSE: bank the state. The split-graph route is FULLY
mapped: validated kernels (3.08e-04), proven primitives, clean 2k execution
(47 cycles), root-caused 100k wall, merged-family build committed, chunked
machinery working (unmerged proc A validated), and now ONE arena fix
(the merged capture's 4.74GB intermediate) between here and the 100k gate.

EMB-FIX RUN (e1071c4): the 4.74GB OOM is GONE (the root cause was v3 dropping
EMB_GATHER — the plain token_embd materialized the [248320,5120] fp32 arena;
4.74GiB = 248320*5120*4 EXACTLY). The merged young-cap now RUNS PAST the old
OOM point — and hits a device fault during the merged captures (after the
stock early-cap completed: commit 1 CLEAN + select_final captured; fault
inside the split young-cap region).

STATUS: the merged families capture further each fix (OOM → fault deeper in).
The remaining fault = the merged captures at 100k (fewer, larger graphs). With
the chunked machinery (proc A aborts at 25) this may ALSO be avoided — but the
fault fires BEFORE family 25 (need frag prints restored on v3 to see where).
NEXT: restore FRAGDBG prints on the v3 loop (they were lost in the rewrite —
v3's fwd3_split has no frag prints) + run chunk A; the fault position tells
whether chunking alone clears it or the merged graph size needs splitting.

V3FAM RUN (25415bb): the merged captures reached family m44 (24 print groups
= lead + m0..m44) then faulted. The merged wall is ~44-46 families (each
merged = ~2 blocks) = ~90 blocks-equivalent... CONSISTENT with the unmerged
wall at ~48-52 single-block families = the SAME TOTAL. CONCLUSION: the wall
counts TOTAL GRAPH/PIECE SIZE (blocks captured), NOT family count. Merging
did not and cannot move it.

FINAL STRUCTURAL READING: the 100k dext limit = ~48-52 blocks-worth of
fresh-captured graph per process (2k shapes fit 64; 100k shapes hit ~50).
The ONLY routes under it: (1) CHUNKED 2-PROCESS with cache-warm re-render —
PROC B STILL FAULTED because re-render re-CAPTURES (capture is what counts;
cache only skips compiles). The TRUE fix = PERSIST the captured graphs across
processes — impossible (in-memory). (2) REDUCE captured blocks per process:
the stock path captures its families covering ALL 64 blocks and WORKS (7.28
verified!) — the stock graphs are fewer/larger pieces. The split's 96 small
families = more total pieces than the stock's ~6 large families. THE INSIGHT:
piece count, not blocks! The stock's 6 families × their piece counts < the
split's 96 × pieces. To fit: fewer families each covering MORE — but merging
showed the wall counts total pieces (~blocks). Contradiction? Unless the
stock families' pieces are FEWER per block than the split's (the stock renders
optimized multi-block graphs; the split renders per-block × pre/post = ~2x
the piece count). If so: the merge math should have halved... m44 ≈ 90
blocks × ~0.55 pieces/block vs unmerged 52 × 1 — roughly equal totals.
=> the wall = total piece-graphs ≈ fixed budget. The split needs the SAME
piece budget as the stock (which fits!) = the split must capture via the
STOCK-style single-family whole-model graph — back to the carrier wall.

HONEST ASSESSMENT FOR THE SESSION CLOSE: the split-graph route at 100k is
structurally complete but blocked by a dext piece-graph budget that per-block
fragmentation exceeds. The 2k gate is CLEAN (all cycles). The paths forward
(next session): (a) ask the fork maintainer about the piece-graph budget /
whether it is tunable; (b) the stock-fidelity route: make the fragments
render as FEWER pieces per block (drop the stable-buffer assigns — they add
stores/pieces; accept the input-identity risk we exonerated); (c) pragmatic:
run the split only at 2k-class contexts and keep the stock path at 100k
(the 7.28 canonical is unaffected).
