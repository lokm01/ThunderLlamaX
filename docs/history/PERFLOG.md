== project baseline ==
ctxbench 2026-08-24: 2k=9.96 4k=9.76 8k=9.15 16k=8.27 32k=7.01 64k=5.18 78k=4.61 82k=4.47 100k=OOM (tok/s, BEAM=1)

== P0 attribution + microbenches 2026-08-24 ==
kern_hist.py (L=2048, BEAM=1 JIT=2 KERNEL_HIST=1, fork hcq patch):
  1578 kernels/tok; kernel-exec sum 115.8ms vs 156.4ms graphless wall (74.1% busy).
  NOTE: dext signal timestamps are MICROSECONDS (fork's PROFILE math assumes ns -> 1000x off);
  eager-paced per-kernel times inflated ~15-30% vs JIT=1 production wall (100.1ms).
  Top kernels/tok: r_544_8_8_4_20_4_2_4 16.8ms(64x), r_544_8_4_4_4_20_2_2_4 16.5ms(64x),
  E_10880_2_8_4_2_4_2 12.6ms(192x), r_320_16_2_2_20_8_2_2_4 9.5ms(48x) -> GDN-block swarm dominates.
gdn_chain_bench.py (BEAM=1, 20 distinct FULL blocks chained in one jit, JIT=2):
  1.95 ms/full-block => ~93.7ms/48 GDN blocks (eager-paced, ~25% inflated);
  single-call eager = 42.7ms/block (confirms block_anatomy.py was launch-tax noise).
kv_bench.py @L=32768 fp32 (16 attn layers x [L,1024] K-cache read = 2.15GB):
  (a) naive per-layer seq 21.95ms = 97.8 GB/s
  (b) ctx-chunked (32/batch, one op) 16.69ms = 128.6 GB/s
  (c) 16-layer stacked contiguous [16,d,L] 44.92ms = 47.8 GB/s (WORSE than strided!)
  In-model KV cost from ctx slope (~48ms @32k) is consistent with the naive rate.
duphead check (vram_check.py, MTP_SKIP_DUPHEAD on/off): after_load 12.63GB / steady 13.11GB
  BOTH ways -> no duplicate head exists in VRAM (GGUF has one output.weight; blk.64 tensors
  never realized; draft shares main head since v223). Patch kept as inert hygiene (drops 15 keys).
gate: mtp_spec spec 20 (K=2 JIT=2 BEAM=0): GREEDY MATCH 20/20, acc 0.61 (11/18, CI +-0.23).
bench_ctx re-baseline (BEAM=1 MEASURE_N=50): 2k 100.09ms 9.99tok/s | 8k 108.71 9.20 | 32k 141.93 7.05
  (baseline was 100.45/109.24/142.75 -> no regression).
TAKEAWAY: P1 GDN scan fusion confirmed top priority (GDN swarm ~67-75ms of the 99.3ms floor,
<10ms unexplained once eager inflation is discounted). P2 KV: stacking LOSES; chunk-split only
reaches 129 GB/s (vs >=300 needed) -> P2 needs fp16-KV first and/or a custom kernel, not layout tricks.
SAFETY: PROFILE=1 with JIT=1 graph-timestamp collection HARD-HUNG the rig (watchdog reset
'wdog,reset_in_1', auto-reboot 2026-08-24 12:45). Do NOT use PROFILE graph instrumentation here;
use KERNEL_HIST=1 (JIT=2 eager) instead.

== P1 2026-08-24 — GDN T=1 A1 scan fusion (window de-uop-store) ==
Change (fork model.py): _attention_t1 fused path for concrete T==1 (env MTP_FUSED_T1, default ON):
pure cat conv-window (no zeros-buffer + 2 window stores), one-expr-tree scan, stores kept on the
proven legacy dataflow chain (trailing-AFTER-only stores BREAK under capture: UOp spec rejects
AFTER(CAST,STORE,STORE), and detaching stores entirely -> garbage 248320-token output). Symbolic/T>1
untouched. NOTE: trailing stores must be .contiguous().uop.after(...) (tensor AFTER spec).
Gates: p1_sanity.py stock-greedy 12/12 MATCH vs spec_base prefix (legacy prefill chunk_size=32 +
fused decode); mtp_spec spec 20 (K=2 JIT=2): GREEDY MATCH 20/20, acc 0.61 (11/18 CI +-0.23).
kernels/tok @2048 (KERNEL_HIST=1 JIT=2): 1578 -> 1482 (-96 = exactly 2/block: win zeros-init + win
store; r_320_16 re-tiled). exec-sum 116.5 -> 113.4 ms. Dominant kernels UNCHANGED: r_544_* pair
64x each 33.5ms = the TWO delta-rule reductions (~0.7ms/block, 5.6MB fp32 state rw); they form a
SEQUENTIAL chain d=sum(s1*k) -> delta -> state_new -> o=sum(state_new*q) => unmergeable at tensor
level. E_10880 (192x, 12.7ms) = gating elementwise, also unchanged.
bench_ctx (BEAM=1 MEASURE_N=50): 2k 98.59ms 10.14tok/s | 8k 108.64 9.20 | 32k 140.49 7.12
  before:            2k 100.09 9.99      | 8k 108.71 9.20 | 32k 141.93 7.05  (-1.5ms @2k, flat rest)
VERDICT: PARTIAL. Window machinery was NOT the main fusion breaker; floor ~unchanged (~97ms),
target >=20 tok/s @2k NOT met. Per-block count ~25, not <=4-6: irreducible = 2 sequential state-wide
reductions + ~4 gating/norm elementwise + GEMVs. A2 (@function CALL units) skipped: changes
submission structure only, cannot change scheduling/fusion => no device-time mechanism.
NEXT LEVER: A3 custom CUDA fused scan kernel (nvcc shim proven) keeping state resident across both
reductions; else accept floor and proceed to P2/P3 where MTP amortizes it.
Artifacts: kernhist_p1_legacy.json (=pre-change rerun w/ MTP_FUSED_T1=0), kernhist_p1_fused.json.

== P1b algebraic scan-reduction merge (2026-08-24) ==
Change: fork model.py GatedDeltaNetBlock._attention_t1 only. Env MTP_ALGEBRA_T1
(DEFAULT OFF — see verdict): replaces the two sequential K-wide delta-rule
reductions with ONE contraction R = s1 @ cat([k_t,q_t],-1) -> (B,H,V,2), then
pure elementwise: delta = (v - R0)*beta; state' = s1 + delta(x); o = R1 + delta*(x.y).
Kill-switches compose: MTP_FUSED_T1=0 full legacy, MTP_ALGEBRA_T1=0 P1 fused path.
GOTCHA (cost one debug round): outer product MUST keep K on the LAST axis
(delta(B,H,V,1) * k(B,H,1,K)); using kt.unsqueeze(-1)(B,H,K,1) silently
broadcasts V==K to (B,H,V,1) and corrupts the state (iso maxdiff 113).
Numerics (fixed form, NV iso): delta 9e-7, state exact, o 4.6e-5 (fp32
reassociation of the x.y term only). Gate: mtp_spec spec 20 (K=2 JIT=2 BEAM=0)
GREEDY MATCH 20/20 vs spec_base.json, acc trajectory identical (0.62 @cyc4,
0.61 @cyc9). First attempt (pre-fix) was systematically broken: acc 0.00,
greedy garbage -> caught by gate, root-caused via iso A/B.
KERNEL_HIST @2048 JIT=2 BEAM=1 (flag ON vs P1-fused):
  exec-sum 113.40 -> 113.91 ms | kernels/tok 1482 -> 1530 (+48)
  r_544_* pair: UNCHANGED (64x each, 16.93+16.52 ms) — THE KEY NEGATIVE RESULT
  removed ~6.0ms of small reduce/E kernels; added ~7.8ms merged-contraction +
  elementwise kernels => net wash.
Production bench_ctx @2048 BEAM=1 MEASURE_N=50: ON 98.94ms/10.11tok/s vs
OFF 99.04/10.10 — statistical tie.
**ATTRIBUTION CORRECTION (supersedes P0/P1 note)**: the r_544_* pair is NOT the
delta-rule double reduction. It survived legacy->fused->algebra byte-identical
in count AND time, i.e. it is per-block-generic (64x/tok each = once per ALL 64
blocks incl. 16 full-attn; two variants = two per-block instances — prime
suspects: the two RMSNorm reduces per block, unconfirmed). The REAL delta-rule
reduction kernels were r_3_16_64_128_2 + r_8_16_3_16_4_4_8 (48x each,
~3.5ms/tok combined) — merging them could win at most ~2ms, and the batched
(48,128,128)@(48,128,2) contraction + elementwise entourage gives it all back
under this scheduler.
bench_ctx sweep (shipping config, flag OFF = P1 behavior), BEAM=1 MEASURE_N=50:
  L=2048 : 98.09 ms/tok 10.20 tok/s   (banked pre-P1b: 100.09 / 9.99)
  L=8192 : 109.34        9.15         (108.71 / 9.20)
  L=32768: 140.78        7.10         (141.93 / 7.05)
  No regression anywhere.
VERDICT: BLOCKED-BY-PREMISE (code correct, target invalid). The ~25-30ms/token
saving P1b promised lives in kernels that are NOT the scan. Neither threshold
(<13 or >=18 tok/s @2k) applies — we sit at 10.2 both ways. The 33.5ms r_544
pair + 12.6ms E_10880 (192x gating elementwise) ARE the floor now (~46ms of
~113ms exec). NEXT: (a) identify r_544 definitively (dump kernel source: disk
cache bypasses codegen.to_program — need NOOPT or cache-bust + source hook),
then fuse/norm-target it if it is the RMSNorm pair (a norm-reduce at 264us ea.
would be a badly-configured grid — cheap fix candidate!); or (b) skip to A3
custom CUDA block-fused scan+norms. Do NOT re-attack the delta-rule reduction.
Artifacts: kernhist.json (=P1b ON run), /tmp/p1b_gate*.log remote,
iso_alg.py (/tmp local+remote). Fork commit: model.py only.

== P1d stacked raw-block dequant GEMVs 2026-08-25 ==
Hypothesis: cat ffn_gate|ffn_up RAW IQ3_XXS blocks pre-dequant (pure view, zero
materialization) -> one 68MB GEMV kernel instead of 2x34MB with two ramps.
Correctness: EXACT (maxdiff 0.000000, argmax equal) -- cat-then-dequant IS a view.
Perf (interleaved, BEAM=1, jit, steady): 2 sep GEMVs 0.810-0.818 ms vs 1 stacked
0.801-0.812 ms => +/-1% = noise; far below the +20% stage bar. Stacked confirmed
ONE kernel (r_2176_32). STOP per gate: no model.py change, fork untouched.
Why: lazy-dequant GEMV is ELEMENT-bound not byte-bound -- IQ3 34MB/89M elems
~0.405ms (220 Melem/ms) ~= fp16 356MB/178M elems 0.815ms (218 Melem/ms); same
element rate at 5.25x fewer bytes. Effective raw-BW 84 GB/s is the wrong metric;
per-element dequant ALU (scale-word shifts + 2 LUT gathers per 256 elems) is the
cost. Stacking can only remove a ~10us launch gap.
GDN note: attn_qkv=Q5_K, attn_gate=IQ3_XXS, ssm_alpha/beta=fp32 -> mixed block
sizes make raw-block stacking ill-defined anyway (moot).
Takeaway: r_544 FFN pair (~33.5ms/tok) only moves via cheaper per-element dequant
codegen or custom CUDA fused block kernel (A3); layout tricks are exhausted.
VRAM dead ends: fp16 gate|up materialization = +17.1GB, int8 = +8.6GB.
Artifacts: p1d_findings.md, p1d/{stack_bench,final_bench,probe2}.py.

== P1e quant-format GEMV shootout 2026-08-25 ==
Question: would a gather-free GGML format (Q4_K/Q4_0/IQ4_XS nibble-scale dequant) beat
IQ3_XXS's LUT-gather chain on the r_544 FFN GEMV [17408,5120]? Measured BEFORE any
download: synthesized raw blocks per fork-supported type, lazy dequant fused into x@W.T,
warm TinyJit BEAM=1, 50 iters (quant_shootout.py).
| type | ms | raw MB | eff GB/s(raw) | Melem/ms | model est GB |
| fp16 ref | 0.773 | 178.3 | 231 | 115 | >55 |
| Q4_0 | 0.811 | 50.1 | 62 | 110 | 15.5 |
| Q4_K | 0.972 | 50.1 | 52 | 92 | 15.5 |
| Q5_K | 1.011 | 61.3 | 61 | 88 | 18.8 |
| Q6_K | 0.913 | 73.1 | 80 | 98 | 22.2 |
| IQ4_XS | 0.820 | 47.3 | 58 | 109 | 14.7 |
| Q8_0 | 1.093 | 94.7 | 87 | 82 | 28.6 |
| IQ3_XXS | 0.748 | 34.1 | 46 | 119 | 10.8 |
VERDICT: NO WINNER - IQ3_XXS is the FASTEST supported format; best gather-free (Q4_0)
is 8% slower, Q4_K 30% slower. LUT-gather hypothesis REFUTED (4KB grid stays cache-hot;
per-element ALU cheaper than Q4_K's d*sc*q-dmin*mn). No download, no e2e run (decision
rule failed at step 2). All formats 2-6x above own byte floor => element-bound across
the board; A3 custom CUDA fused dequant-GEMV (>250 Melem/ms in-graph target) is the only
remaining GEMV lever. Exact inventory from header: 27.32G params, GEMV pool 23.19G elems,
embd 1.27G IQ3_S, output.weight Q5_K (fp16-head note confirmed stale again).
Artifacts: p1e_findings.md, quant_shootout.py. bench_ctx.py got MODEL env plumb.

== P2 2026-08-24 — fp16 KV cache (MTP_KV_FP16=1 default ON) + chunked-KV experiment (MTP_KV_CHUNK, default OFF) ==
Fork: TransformerBlock cache_kv alloc dtype fp16 (env MTP_KV_FP16=1); store casts k/v to cache
dtype, reads cast back to x.dtype so SDPA math stays fp32. MLA block left default (comment only).
Env-gated MTP_KV_CHUNK=8: exact chunked split-K decode attention over full max_context with
-inf masking + zero-init cache — MEASURED REGRESSION @32k: 151.5ms vs 135.9 plain fp16 (~240
extra small kernels/layer-set launch tax > bandwidth gain). Default OFF, kept as documented negative.
Gates: p1_sanity MATCH 12/12; mtp_spec spec 20 GREEDY MATCH 20/20 (acc 0.61, unchanged;
DEV=NV BEAM=0 JIT=2 MTP_K=2). Final-state re-gate after chunk code: sanity 12/12 again.
VRAM @32k steady: 14.99 GB (fp32 equivalent ~17.2) = −2.15GB as predicted. cache.db backed up
pre-run (cache.db.preP2), integrity ok.
bench_ctx BEAM=1 MEASURE_N=50 (100k: N=5), fresh process per L:
| L | fp32 baseline ms/tok | fp16 ms/tok | tok/s | vram GB |
| 2048 | 100.09 | 98.29 | 10.17 | 12.97 |
| 8192 | 108.71 | 107.10 | 9.34 | 13.37 |
| 32768 | 141.93 | 135.94 / 139.48 (repeat, ±2.5% noise) | 7.36 | 14.99 |
| 65536 | (5.18 tok/s, ~193 est) | 197.09 | 5.07 | 17.15 |
| 100000 | OOM (>82k ceiling) | 240.04 | 4.17 | 19.43 |
Takeaway: TARGET PARTIAL. Capacity goal MET — 100k now fits (was OOM >82k; KV bytes halved,
slope fit 1.52e-3 → ~1.25-1.34e-3 ms/tok per ctx-token). Speed goal MISSED (≤125ms @32k not
reached): kern_hist @32k fp32-vs-fp16 shows the three O(L) attn kernel groups are NOT byte-bound
(r_12_2_*: 21.76ms fp32 vs 21.71ms fp16 identical; only r_3_* shrank 8.97→6.17). Halving KV bytes
bought only ~4% wall because the ctx-scaling time is latency/grid-shape bound, not bandwidth bound.
Chunked split-K (the microbench-predicted fix, standalone 97.8→128.6 GB/s) does NOT compose
in-model — same lesson as P1c: standalone probes don't reproduce in-model fused schedules.
Next lever for the remaining ~35ms ctx-tax: grid-shape work on the two big reduce kernels
(attn@v split-K inside the renderer/scheduler, not at tensor-op level), or P3 MTP which amortizes it.
Artifacts: /tmp/p2_b*.log (volatile), hist logs p2_hist32k*.log. Commits: fork 41abe8e, project this commit.

== A3a 2026-08-28 ==
Standalone hand-CUDA IQ3_XXS dequant-GEMV microbench (a3a/): GO for A3 integration.
- gate/up [5120->17408]: 0.092-0.101 ms = 885-971 Melem/ms, 340-372 GB/s raw (tinygrad isolated: 0.748 ms / 119 Melem/ms -> 7-8x)
- down    [17408->5120]: 0.103-0.114 ms = 779-865 Melem/ms, 299-332 GB/s
- qkv     [5120->10240]: 0.063-0.064 ms = 825-828 Melem/ms, 316-318 GB/s
- correctness: dequant math bit-exact vs fork ggml_data_to_tensor; GPU relerr ~1e-7; split-K loses (warp-per-row suffices, 640+ CTAs).
Takeaway: hand kernel is 2x the >=400 Melem/ms GO bar and runs at ~80% of the 447 GB/s
byte floor. Loader for hand cubins via NVProgram BUILT+validated (a3a_findings.md: NTID
cbuf_0[0..2] must be set; per-kernel EIATTR_REGCOUNT via symtab patch or OOR-register
fault; params at c[0][0x160]; batched q.exec chain = 6.7us/kernel vs 165us eager).
Projection: 48 GDN blocks x ~0.4ms GEMVs ~= 19-20ms/token vs ~60ms pool -> ~16-17 tok/s
base before MTP, IF launched inside the JIT=1 graph family (eager per-call tax would eat it).
No model-path change: bench_ctx@2048 sanity 98.85 ms/tok (baseline ~100.1), fork untouched.

== A3c 2026-08-28 — O(L) attention kernels replaced (rowmax/rowsum/pv) ==
Diagnosis from kernel dumps: full-attn decode chain = 5 kernels (qk r_6 token-parallel;
r_12_2 rowmax grid(12)x(2) SERIAL over start_pos; r_8_3 rowsum grid(8)x(3) SERIAL;
E_ normalize token-parallel; r_12_8_16 pv grid(8,12)x(16,2) serial L loop, 3072 threads).
The two reduces ran on 24 THREADS TOTAL (685us/ea @32k); pv uncoalesced-serial.
A3c = 3 new kind-rules in a3b.py: rowmax/rowsum -> grid(24)x(256) float4+shared tree reduce;
pv -> grid(8,12)x(16,2,32) z-split over t, float4 probs quads, shared z-reduce, same
sigmoid-gate epilogue. start_pos scalar kept in _parse; launch via vals. Worker-pool-compiled
kernels substituted parent-side (realize.py store hook). Bugs found by validation probes:
max init must be 0xff800000 (-inf), pv quad v-offset = i*1024, matcher head relation.
Gates: sanity 12/12 JIT=1, mtp_spec spec 20 GREEDY MATCH (acc 0.61 unchanged), all
subst relerr <= 1.2e-6. bench_ctx BEAM=1 production:
  L=2048 : 73.24 -> 72.28 ms/tok  13.84 tok/s
  L=8192 :  81.76 -> 74.70         13.39
  L=32768: 110.92 -> 80.85         12.37
  L=100000: 212.74 -> 98.89        10.11   (was 240.04 / 4.17 this morning)
Context slope 1.42 -> 0.28 us/ctx-token (5x flatter); attn @100k ~= 26.6ms ~ byte-bound
(~9.8GB effective KV+probs traffic at ~370GB/s). Day total @100k: 4.17 -> 10.11 tok/s (2.4x).
Commits: fork 794dc58, project this commit.

== A3d 2026-08-28 (evening, WIP — NOT ACTIVE) ==
Attempted Q5_K warp-per-row GEMVs (GDN qkv r_320 9.5ms bucket + lm_head r_15520 4.9ms).
Kernels written + validated against numpy/gguf.py reference EXACTLY — but tinygrad fused
dequant pairs elements in gguf stack/reshape permuted order, NOT natural byte order;
the generated kernel is the ground truth and its mapping is not fully derived yet.
Scales CONFIRMED standard (sub=k>>5) by unit probes; qs bytes 64..127 probe DEAD (reads
split across +48/+112 byte groups per thread). Rules disabled in override.json; code inert.
Probe infrastructure (a3d_probe_map2.py one-shot 524-row mapping decoder + mapdump.txt)
committed — next session derives the f-table from r_320 source and finishes.
Fork 7c0e7b7, project 5704142. State re-verified: sanity 12/12, bench 72.13ms @2k (13.86 tok/s).

== A3d 2026-08-28 (late) — Q5_K GEMVs LIVE (mapping solved) ==
Element mapping derived from r_320 source (probes confirmed; earlier probe-value confusion
was fp16 rounding of the x encoding): qs byte = 32*(k>>6)+(k&31), nibble = (k>>5)&1;
qh byte = k&31, bit = k>>5; scales standard (sub = k>>5). Two perf landmines found by
bench: (1) iterating elements in k-order gives stride-8 byte loads (4x W amplification,
+13ms REGRESSION) -> lane owns 4 consecutive W bytes, both nibbles each; (2) sc/mn as
dynamically-indexed local arrays spill (+8ms) -> per-lane register scales via byte-group
switch. Passthrough copy must be inside warp<N (else-branch never runs when N%wpc==0);
_validate_q5 now poisons the out buffer between orig/new (was blind to that).
Gates: sanity 12/12, mtp_spec spec 20 GREEDY MATCH (acc 0.61). relerr <= 1.5e-6.
bench_ctx BEAM=1: 2k 69.09ms/14.47 tok/s | 8k 71.12/14.06 | 32k 77.44/12.91 | 100k 95.21/10.50.
Day total @100k: 4.17 -> 10.50 tok/s (2.52x). Remaining base swarm: r_48_2 (7.9ms bucket,
unidentified), norms/elementwise launch tax. Fork this commit; project next.

== A3e 2026-08-28 (night) — GDN in_proj LIVE, bit-exact ==
r_48_2 = in_proj_ba IQ3_XXS GEMV: 6144 rows (=48 blocks x 128), W row-major 1960B/row,
half-product accumulation (x*inv_rms*nw -> half; w fp32 -> half; hmul -> float acc) and
epilogue out[o] = (half)( vec[o] * 1/s48[o>>7] * nwo_f32[o&127] * (float)(h*swish) ) with
hexp2/hrcp (NON-underscore — NVRTC) and the truncated -1.4423828125f constant.
Element mapping IDENTICAL to A3b IQ3 template (natural byte order, lane=2 consecutive
q-bytes, inline parity signs). Validation relerr 0.00 (bit-exact — same rounding chain).
Gates: sanity 12/12, spec 20 GREEDY MATCH (acc 0.61). bench_ctx BEAM=1:
  2k 65.03ms/15.38 tok/s | 8k 67.30/14.86 | 32k 73.62/13.58 | 100k 91.29ms/10.95 tok/s.
DAY TOTAL: @100k 4.17 -> 10.95 tok/s (2.63x); @2k 10.17 -> 15.38 (1.51x).
Remaining known: GDN scan swarm + norms/elementwise launch tax + MTP v3 for the 2.4x.

== MTP v3 M1-CORRECTNESS DONE (2026-08-29, fork 0650210) ==
mtp_v3.py GREEDY MATCH 20/20 (FULL) — first CORRECT spec-decode on the JIT=1 single-graph
architecture. amd=[8280,52237,1118] = exact baseline; acc 11/18=0.61; 2.22 tok/cyc.
The three root causes (pair-bisect + always-validate):
 1. STEP_STATES mid-block stores perturbed scan schedules (0.77 relerr) -> block-boundary
    stash + flush_step_states helper (model.py).
 2. batch-3 N=1024 scale GEMV variant = NAN (s_3 mismap suspect) -> batch>1 kernels now
    ALWAYS standalone-validated, auto-fallback-to-orig on failure. All other T3 variants
    validate 1e-6 (silu 3.8e-6, mul 2.5e-6, adds 1.8e-6, 12288-scale 2.7e-6, a3f qkv/lm_head
    exact via e2e harness).
 3. Harness red herring: T=1 reference must snapshot/restore state baseline (t3_exact.py).
PERF (next, exact numbers from run): cycle 3.3s = probe 4.8s/cyc (!! = per-cycle RECAPTURE
from new toks tensors -> persistent assign buffer), sel 1.9s/cyc (96 eager realizes ->
one batched realize), draft 280ms/cyc (eager tax, v223-parity), head 110ms. After the two
driver fixes: ~0.75s/cyc = 3.2 tok/s; then probe JIT=1 (280ms ISO-proven) is the lever to
M1-speed (needs cycle <=450ms incl. draft: Phase-3 draft/launch work).

== GDN SCAN MEGA-KERNEL (2026-08-30, THE BREAKTHROUGH) ==
Single CUDA kernel replaces ~18 tinygrad kernels per GDN block. T=1/3/8 ALL PASS
(relerr <2e-7). 4.88ms/pass at T=3 vs previous ~250ms = 51x speedup.
Bugs fixed: TinyELF typed signature (empty tuple = silent kernel no-op), warp broadcast
after shuffle reduction, numpy f64 promotion in test data.
Projected: T=3 probe 317→~72ms; T=1 base 65→~40ms. Path to 40: scan(done) + draft + head + K=4

== 2026-08-29 evening session (local ZCode agent) ==
- A5 source transforms landed (fork e5f97d7): serial-reduce tree + flat float4 copy,
  11 kernel families at T=1, sanity 12/12. PERF-NEUTRAL at JIT=1 (eager hist 30us/kernel
  was pacing-inflated; kernels already fast in graphs).
- CRITICAL DECT CONSTRAINT: grid-stride loops HANG the channel — gridDim reads as 0 on
  TinyGPU dext (i += gridDim.x*N = i += 0 -> infinite loop -> 30s wait -> watchdog reboot).
  Isolated standalone (f4/scalar/f2 all wedge; flat one-elem-per-thread fine). ALL hand
  kernels must use flat indexing + bounds guard.
- CRITICAL SUBSTITUTION RULE: a3b_substitute also runs inside compile workers via
  to_program under @Context(ALLOW_DEVICE_USAGE=0) — eager validation launches THERE
  wedge the channel (3 watchdog reboots). Validators gated on ALLOW_DEVICE_USAGE.value==1.
- T3 SUBSTITUTIONS DISABLED (fork c3dca1c): stock T3Lazy kernels are exact; batch>1
  hand-GEMV templates produced wrong probe logits across multiple variant sets.
- *** THE SIX-HOUR BUG: MTP_KV_CHUNK=8 in mtp_v3 required-env line *** — chunked split-K
  attention is numerically WRONG at T>1 (P2 knew: regression, default OFF); every run
  with it gave amd=13395/5324 vs correct 8280. Removed; GREEDY 20/20 restored. NEVER set
  MTP_KV_CHUNK.
- DRAFT_JIT verified 20/20: draft step 88-100ms -> ~21ms/step (one JIT=2 TinyJit,
  stable-buffer inputs, v_sp var). Steady cycle ~395ms = 5.6 tok/s spec (vs 15.4 base
  @2k — still below; needs cycle<144ms).
- Reliability: JIT=1 probe dies deterministically ~cyc 20 (wait 49362, have 49346 — 16
  missing timeline signals: graph-replay vs copy-queue accounting in fork hcq.py).
  JIT=2 probe survives 60 tok but WRONG (25/60). NEXT: fix the signal accounting.
- Infra repaired same day: containerd bolt-DB corruption (rm /var/lib/containerd/* +
  /var/lib/docker/* in VM + image rebuild); cache.db quarantined+restored (proven
  innocent); nvcc docker + shims verified.
- Next levers ordered: (1) hcq signal accounting fix (unlocks long runs), (2) sel_states
  160ms (96 realizes), (3) head 65ms (device argmax), (4) probe 128ms launch tax
  (3500 kernels), (5) K sweep after draft is cheap.

== 2026-08-30 session (continuation): BOTH blockers ROOT-CAUSED AND FIXED ==
- KERNARGS POOL SPLIT (fork 6861bca): eager kernels + graph kernargs shared one 16MB
  wrap BumpAllocator ring; prefill alone pushes ~48MB through it; every wrap clobbers
  the graph kernargs -> replay executes garbage pointers -> channel fault -> 16-56
  orphan timeline signals -> 30s wait death at cycle ~20-26 (deterministic; 24 w/
  draft-jit vs 26 eager = consumption-rate fingerprint). Diagnosed with MTP_SIGTRACE=1
  (reservation-stack ring; all orphans = submitted-never-executed eager kernels).
  Pool now 64MB = 32MB eager wrap + 32MB dedicated non-wrap graph region. Hang GONE.
- PROBE_RO (fork 6861bca, model.py): the T>1 probe scan stored its FINAL state
  (incl. REJECTED tokens) into live recurrent/conv_state EVERY replay. Select-mode
  diverged at token 21 (JUST past the era 20/20 gate — every earlier gate was blind);
  commit-mode at token 12. Fix: probe stores to scratch; states advance via select
  (exact) or commit forwards (exact). GREEDY 60/60 then 200/200 (FULL), acc 0.70-0.79.
- GATE RULE: all greedy gates must be >= 60 tokens (spec_base200.json committed).
- Shavers: head-in-probe 45->20ms; per-m select TinyJits 80->24ms warm; argmax
  everywhere on device (logits never D2H). Steady cycle 178ms = probe 87 + draft 47 +
  head 20 + sel 24 @ 2.56 tok/cyc = 14.4 tok/s @2k-class ctx (base 15.4 parity;
  projected ~12.5 vs base 10.95 @100k = MTP wins long ctx, bench pending chunked
  prefill). Devcopy draft (uop.store hm) REGRESSED (launch tax > 20KB PCIe) — off.
- NEXT LEVERS: whole-draft-chain one-jit (47ms), scan-cubin integration or scheduler
  fusion on the probe (87ms floor), chunked prefill for long-ctx MTP bench, K sweep
  (K=2 optimal at current costs).
