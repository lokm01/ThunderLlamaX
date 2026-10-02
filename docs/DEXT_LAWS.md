# DEXT_LAWS — the DriverKit-GPU gotcha compendium

Every law below was learned the hard way on this rig: RTX 3090 (sm_86) behind a
Thunderbolt-4 dock, driven by the TinyGPU DriverKit system extension
(`org.tinygrad.tinygpu.driver2`) through the tinygrad fork's raw-PCIe NV backend.
"Fault" typically means a device fault that poisons every later process until reboot;
"hang" means a 30 s wait-timeout watchdog reset of the whole Mac. Each law cites the
session doc (docs/history/) that paid for it.

Laws are grouped: L = launch/compile, M = memory/layout, C = concurrency/host,
D = data/format, V = validation discipline.

---

## L. Launch & compile

**L1. `gridDim` and `blockDim` read as 0 in SASS.**
nvcc-compiled SASS reads NTID/NCTAID from constant buffer `c[0][0..8]`, which the
fork does not populate (the QMD raster fields still launch the right CTA count, so
flat-index kernels with `if (i < n)` guards work fine). Any grid-stride loop
(`i += gridDim.x*N`) becomes `i += 0` -> infinite loop -> watchdog reboot. NO
grid-stride loops, ever (W1A-era, isolated 3 ways; W0_E4 found `blockDim.x` is zero
the same way: `blockIdx.x*blockDim.x + threadIdx.x` collapses to `threadIdx.x` in
every CTA — silently wrong, no fault). Workaround if a kernel must know its block:
`tid = (blockIdx.x << 8) + threadIdx.x` for 256-thread CTAs, or set
`prog.cbuf_0[0..2] = local_size` before launch (a4 harness pattern).

**L2. The name-encoded launch config trap.**
The graph builder (gcycle.py) derives the launch LOCAL SIZE from the kernel/cubin
NAME substring (`nw32`->1024, `nw24`->768, `nw16`->512, else 256 threads). A kernel
whose name lacks its warp-count token silently launches with 256 threads in-graph —
standalone launches pass threads explicitly and CANNOT catch it. This produced the
W2G "HMMA in-graph zeros": a 1024-thread kernel launched at 256 threads writes
NOTHING (not even partial rows). Every new engine kernel must carry its warp-count
token in the cubin name; only the in-graph gate can validate it (W2H).

**L3. TinyELF signature must be TYPED.**
`TinyELF(..., signature=(("v",0,dtypes.int32,()),))` — one entry per pointer arg,
ints via `vals=`. An EMPTY signature makes the kernel a silent no-op for every
launch (values never written). Buffer-only kernels must still carry a dummy typed
entry or a dummy int arg (W2H/a4; the a4 megakernel "did nothing" for this reason).

**L4. Arg-count mismatch is a SILENT arg shift.**
Kernel params vs launch buffers off by one -> every arg shifts a slot -> device
fault or garbage. Empty signatures do no validation. Count args on both sides
(W2_MTP gotcha 23).

**L5. Multi-kernel cubins mis-load — THE MULTI-KERNEL CUBIN LAW.**
Identical kernel code faults when loaded from a 10-kernel cubin while passing from
its own single-kernel cubin (4-kernel cubins were fine; the boundary is between 4
and 10). Sharpened at R7 with the mechanism: loading kernel X from a multi-kernel
cubin executes WRONG CODE (per-`.text.<name>` addressing broken) -> "Out Of Range
Register" warp exceptions on every SM -> channel wedge. The repo's one-kernel-per-
cubin convention (`-DKNAME`) is LOAD-BEARING; every standalone harness must build
single-kernel cubins (W1A gotcha 2; R7_DECIDERS).

**L6. TinyELF finds the kernel by cubin SYMBOL name.**
A template .cu must rename its kernel per use (`-DKNAME=...`); a mismatched name =
Illegal Instruction Encoding faults or silent no-ops. Also `do` is a C keyword
(W2_MTP gotcha 22).

**L7. NVRTC (in-model compiles) is stricter than nvcc.**
No `uint4` (custom struct ok), `hexp2/hrcp` NON-underscore forms, a3b args carry
BARE names while the SIG text has `name_numel` (W2-era; pv3/T1_FAULT_NOTES.md).

**L8. The nvcc shim needs ABSOLUTE paths and a live docker env.**
The container cwd is not the host cwd; relative source paths = "No such file".
Non-interactive shells lack both `PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH`
and `DOCKER_HOST=unix://<colima-socket>` — every remote/automated build command
needs the prefix (a3a/build.sh; W2B).

**L9. Lane-strided loops miscompile; keep loops sequential-stride.**
`for(b=lane; b<192; b+=32)` gave wrong sums with all bytes read; sequential
`for(b=0;b<192;++b)` fixed it exactly (W1A gotcha 4).

**L10. The dext adds a 1 KB driver reserve to static smem.**
ELF `smem_size` = ptxas static + 1024. The carveout picker (32/64/100 KB buckets)
works on the INCLUSIVE size; >32 KB static smem is fine in-graph (proven by a
padded-scalar control at 36,864 B) (W2H).

**L11. Kill a GPU-locked python and the host watchdog-reboots.**
pkill of a live GPU process frequently wedges the channel into a reset (~10 min +
/tmp wiped). Let runs finish or rely on the 30 s wait-timeout suicide. After any
reboot, re-wait for the docker VM before building (W2_100K; W1B gotcha 14).

**L12. One attempt per boot.**
A faulted dext poisons every later process (fresh processes fault at the first
wait); reboot clears it (W2B).

---

## M. Memory & layout

**M1. THE ALIGNMENT LAW.**
nvcc merges adjacent narrow loads (2x u16 -> 1 u32) and the merge is legal only at
natural alignment. Quant blocks of odd byte-size (IQ3_XXS = 98 B) put the scale-word
u32 at 2-mod-4 on odd blocks -> SM Multiple Warp Errors, hard fault, empty error
report. RULE: every per-lane multi-byte run must be naturally aligned for EVERY
block parity, or loaded at widths that cannot merge into an unaligned one. This is
why packed layouts exist: IQ3 rows are repacked `[qs 64B][scales 32B][d 2B]` per
block so all runs are aligned. Wide loads themselves (u16/u32/u64/uint4/float4) are
ALL legal — the W1A "uint32 loads fault" note was a misdiagnosis of this law
(W1C_OPT.md; PTX-diff proof).

**M2. One float4 = 8 halves on `__half*` buffers.**
Copying float*-based patterns (which use +4 for second loads) onto half buffers =
4-half OOB reads (W1A gotcha 1).

**M3. Misaligned smem addressing on `char*` arrays.**
`SM + byte_offset` on a char array is a BYTE offset -> odd-lane float stores =
"Misaligned Address" on every SM. Index a `float*` view instead (the G2 fault;
`__syncwarp` is innocent) (W2B).

**M4. The dext is 1 CTA/SM hard.**
QMD carveout override to 100 KB changed nothing — multi-CTA co-residency does not
happen; the override only shrinks L1. Design for it: fat CTAs (1024 threads = 64
regs x 1024 = exactly the register file) work fine and pipeline the DRAM-stage
phase far better than more CTAs would (W2C).

**M5. Static smem limit 48 KB; no dynamic-smem path.**
QMD shared size = the cubin `.nv.shared` section; there is no runtime dynamic-smem
allocation through this fork. Warp-specialized double-buffering plans that exceed
48 KB static are dead on arrival (W2B deviations).

**M6. Single-array smem staging ONLY.**
Multi-array (>=4 separate arrays) static smem staging hangs the dext
deterministically; a single packed char array + typed views passes (Route-3 SMEM
FAULT; re-add staging only via the k1t1 single-array pattern).

**M7. In-flight ceilings (eager ~312-500, graphs ~900-1800 kernels).**
612 real-kernel launches with one final wait = deterministic fault; sync every
~230-250 launches. On 452-kernel graphs: 2 graphs in flight (904 kernels) stable,
8 (3.6k) faults. Pipeline depth 2 max (W1B gotcha 8; W1C gotcha 17).

**M8. Raw-allocator `_copyin` is ASYNC.**
Uploaded numpy arrays must be kept alive (a `_KEEP` list of Tensor refs) or the
source memory is freed and reused mid-upload = deterministic garbage (W1A gotcha 5).

---

## C. Concurrency, submission & host

**C1. A lone-submitted graph never completes.**
Any graph submitted without a follow-up submit already in flight hangs the timeline
(its signal sticks at value-1; full 452-kernel graphs included; not corruption).
FIX: a 1-kernel flusher graph chained after the last real graph each cycle; never
wait a phase unless a later graph is already submitted (W2_MTP gotcha 20).

**C2. Pipelined benches LIE (race-inflation).**
Back-to-back independent launches of the same kernel RACE on this dext (no implicit
serialization by buffer aliasing). `for rep: launch` + final-sync GB/s is optimistic
— G2 +8%, fat-CTA NW32 +145%. Synced-per-launch (wait=True) or in-graph timing only.
This also applies to any A/B bench whose kernels write shared scratch (W2C).

**C3. Orphaned last timeline value.**
Staged-copyin signals can be lost; waits anchor on `timeline_value - 1`. A graph
waiting on `timeline_value` itself deadlocks (isolated repro) (W1C gotcha 16).

**C4. Every host<->GPU sync costs ~17 ms over Thunderbolt.**
The signal roundtrip dominates; the engine is architected around one sync per cycle
boundary and device-side control to avoid it (run33+ campaign).

**C5. Sub-mask shuffles hang.**
`__shfl_xor_sync(FULL_MASK, ...)` inside `if (warp==0 && lane<8)` = mask says 32
lanes, 8 execute -> hang. Sub-group reduces must use mask 0xff with offsets 4,2,1
(W1B gotcha 9). Full-mask broadcasts after reductions (`__shfl_sync(0xffffffff,
val, 0)`) are required — warp shuffles leave results on lane 0 only (a4).

**C6. `HWQueue.exec` tuple grids are SYMBOLIC.**
A tuple grid raises "memoryview invalid type" at submit — pass plain ints (W1C
gotcha 18).

**C7. Submitting a probe graph without the full draft->probe->accept->flush chain
hangs the timeline** (2 signals short). Debug scripts must use the real cycle chain
(W2H).

**C8. macOS host quirks:** no `timeout`/`setsid`; use `nohup ... & disown` + poll.
A killed local ssh takes the remote child with it (W1B gotcha 15). `/tmp` is wiped
on reboot — snapshots live in `$HOME` (W2_100K).

---

## D. Data & format truths (Qwen3.8-27B GGUF on this fork)

**D1. Q4_0 is a NIBBLE-PLANE split, not ggml pairs.**
Within each 32-elem block: elements 0..15 = LOW nibbles of bytes 0..15, 16..31 =
HIGH nibbles. Element e -> byte (e&15), nibble (e>>4). Using ggml consecutive pairs
gives relerr ~1.0 (draft predicts garbage) (W2_MTP gotcha 21).

**D2. Q5_K tinygrad mapping:** qs byte = `32*(k>>6)+(k&31)`, nibble = `(k>>5)&1`
(lane-constant); qh byte = `k&31`, bit = `k>>5` (0..7!); scales standard sub=`k>>5`.
Derived from the generated kernel source, confirmed by probes (A3d).

**D3. IQ3_XXS:** 98 B blocks; grid = `float[256][4]` from int64 words (little-endian
4 magnitude bytes per word); natural byte order with lane = 2 consecutive q-bytes,
inline parity signs (a3b template).

**D4. Q6_K 2-bit chunk index is `(lane>>2)&3`** (32-element granularity), NOT
`(lane>>3)&1` (W1B gotcha 10). IQ3_S field indices are WITHIN the 256-elem block,
not global (W1B gotcha 11).

**D5. GGUF dims are [in, out]** — GEMV rows = dims[1]; a flat-byte reshape hides the
transpose -> silent garbage (W1C gotcha 19). `data_start` must be computed AFTER the
tensor-header table (21 KB shift) (W1A gotcha 6). Open-ended `data[off:]` views
materialize to-EOF phantoms — bound them.

**D6. `Tensor.repeat` = TILE (torch) semantics:** v-head h <- k-head `h%nk`, NOT
`h/(nv/nk)` (W1A gotcha 7). `from_gguf` returns (model, kv) — a SECOND gguf load
for the tokenizer = +12.6 GB VRAM leak (W1B gotcha 12).

**D7. Stock model calls need `start_pos` as a bound UOp variable**; plain ints
recompile per position (~20 s/token at 100k shapes) (W2_100K; W1B gotcha 13).

**D8. Mixed quant types per block position.** 24 of 48 GDN blocks have
ssm_out = IQ3_XXS, not Q8_0; attention q is Q6_K on 8 blocks and IQ3_XXS on the
other 8. Dispatch by tensor type — wrong dispatch = OOB reads = device fault
(W1B model facts).

---

## V. Validation discipline (how the exactness contract is enforced)

**V1. Poison every output buffer before validation.** A no-op kernel then reads as
poison, not as plausible zeros (the W2H one-cycle dump proved the HMMA kernel wrote
NOTHING because outputs stayed at the poison fill).

**V2. Distinct output buffers for A/B kernel comparisons.** Two kernels run through
the SAME re-poisoned upload-repo buffers can read bit-identical FALSELY (second
run's args alias the first). The first PVH "identical" reading was this artifact
(W2F).

**V3. Standalone != in-graph.** Standalone validation passes threads explicitly and
cannot catch name-encoded launch config (L2), stale bakes (V4), or graph-path
faults. The Tier-1 in-graph gate is the only truth; 2k is a degenerate near-tie
region for attention changes — 100k is the healthy-region gate (W2H, W2D).

**V4. The stale-bake trap.** Build-script job lists on disk do NOT necessarily
describe the on-disk cubins (rebakes happen). NEVER bench or integrate a cubin by
name without confirming its -D bakes (cuobjdump or rebuild). Symptom: half-grid
coverage -> poisoned partials -> 0/60 with garbage output, or 2x-inflated GB/s
(W2C, bit twice).

**V5. Bit-identical batching contract.** Every M=3 (and M=4) kernel keeps the T=1
kernel's per-row fp op order exactly — same loads, same mul/add sequence, same
shuffle trees — so batched rows are bit-identical to T=1 and Tier-1 exactness holds
by construction (W2_MTP; the half2 variants prove order-preservation per pair).

**V6. Gates are >= 60 tokens.** Short gates were blind to the probe-state bug that
diverged at token 21; every greedy gate since is 60+ tokens vs a committed baseline
file (baselines/) (2026-08-30 rule).

**V7. Eager validation must not run inside compile workers.** Substitution hooks
run under `@Context(ALLOW_DEVICE_USAGE=0)`; device launches there wedge the channel.
Gate validators on `ALLOW_DEVICE_USAGE.value == 1` (2026-08-29 rule).

---

## The positive discoveries (what the dext does WELL)

- **Data path: 880 GB/s pure-stream (94% of the 3090's 936 GB/s peak), 843 GB/s
  fp16 GEMV** — the driver is not the wall; kernel quality is (W0_E4).
- QMD dependent-pointer chaining: a 452-kernel graph submits as ~40 pushbuffer
  words + one gpfifo entry + one doorbell; the host loop disappears (W1C).
- 1024-thread CTAs, `mma.sync` tensor-core HMMA, `PRMT` byte-perm tricks, and
  `__shfl_sync` butterflies all execute correctly once the laws above are
  respected.
- Per-launch submission floor ~43-49 us pipelined; the graph path makes it
  irrelevant at engine scale.

Cross-references: docs/history/W1A_ENGINE0.md (gotchas 1-7), W1B_TRUNK.md (8-15),
W1C_OPT.md (16-19), W2_MTP.md (20-25), W2B/W2C (attention kernel laws), W2H (the
name-encode law + poison forensics), W0_DOSSIER.md (driver internals: QMD fields,
cbuf layout, cmdq ring), pv3/T1_FAULT_NOTES.md (harness traps).

---

# 2026-09 additions — the serving (M1) and prefill (P1-P18) campaigns

New groups: S = serving/host-process laws, PF = prefill-campaign laws. Each cost a
fault, a silent wrong answer, or a cold-cycle to learn; session docs cited.

## S. Serving & host-process laws (M1A-M1C)

**S1. THE HOST-PROCESS BOOT LAW.** Draft acceptance is determined by WHICH script
is `__main__`, not by the calls it makes. The canonical host boots at alpha 0.892 /
2.78 tok/cyc; a byte-identical call sequence from any other host (runpy, hand-rolled
boot, warmup variations) collapses the draft to garbage proposals while the probe
stays bit-exact (60/60; only speed halves). Correlates only with argv[0]/__main__
identity; suspected address/layout-dependent dext or kernel behavior tied to the
process image. Consequences: the daemon boots AS the canonical host script with
`M1A_SERVE=1`; ANY boot change must re-gate alpha >= 0.85 — exactness alone is NOT
a sufficient gate (M1A_SERVING.md).

**S2. The ~950-cycle dext budget.** Continuous back-to-back 4-graph speculative
cycles fault the GPU at 850-1025 cycles (6/6 repro; independent of position, alpha,
ring size, duty cycle; idle gaps do NOT reset it; interleaved prefill work DOES).
Fix: graph rebuild + re-anchor at quiescent points every 256 cycles, <1% cost
(M1C_STABILITY.md).

**S3. Bounded-cycle law.** Generation loops must be bounded
`max_cycles = max_tokens + 4`. An unbounded loop + a disconnect path that silently
fails (`is_disconnected()` returns false on this stack) once streamed 44 s of
post-abort runaway generation (M1C root cause #1).

**S4. Fixed handles or orphaned VRAM.** No buffer reallocation in the request path:
a realloc means stale graph kernargs plus ~755 MB of orphaned VRAM per reset (OOM
after ~4). State mutates only via windowed uploads and device memsets; graphs are
built once (M1A).

**S5. Snapshots are delta-windowed, always.** A full-KV download is 16 x ~200 MB of
host copyouts -> silent OOM-kill of the daemon. Save rows [base_P0, pos+16) + GDN
slot 4 + seeds (0.1 s) (M1A law 3).

**S6. Re-encode, never re-render.** Re-rendering full history through the chat
template re-inserts empty `<think>` blocks into prior turns. Keep a message mirror
and encode only the special-token-boundary tail (M1B).

**S7. Stop-batch over-commit.** A mid-batch stop token (im_end inside an accepted
batch) leaves the engine past the stop; the NEXT turn must go FRESH, not FOLLOW_UP
(M1B/M1C).

**S8. Emit-record semantics.** One cycle emits m+1 tokens; compare emits against
`hist[P0:P0+len(emits)]`, never a fixed window (M1A law 2).

**S9. GPU-exit/reboot law.** A GPU-holding python process exiting while the dext is
alive triggers a machine reboot within seconds (4/4). Never pkill a live GPU
process; SIGTERM -> drain -> synchronize -> exit; launchd RunAtLoad heals (M1A).

**S10. Reboot-survivor logs.** /tmp is wiped on fault-reboots — anything that must
survive a crash goes elsewhere (M1C).

**S11. m_hist ring sizing.** Per-cycle history arrays indexed by cycle number need
explicit headroom (1<<20); the [1024] version OOBs at cycle 1025 — exactly inside
the S2 window (M1C root cause #2).

## PF. Prefill-campaign laws (P1-P18)

**PF1. READOUT-ORDER.** Correctness gates are read from the FIRST clean run of a
fresh boot; timing reps advance GDN state and are not gate-equivalent (P-series
campaign law).

**PF2. Launch floor ~0.10-0.12 ms for small kernels.** Any split that adds launches
of <10 MB kernels loses (K-split GEMMs net-negative; the P7E7 superchunk at 48k+
was ~80% launch-bound at 2849 launches x ~0.34 ms).

**PF3. The alignment law extends to W-tiles.** The prefill GEMM family needed its
own wide-tile repack (pack_w7) with W-tile CTA-offset addressing and `8*(lc&3)` qs
addressing; FFN has an NTILE law (P6/P7).

**PF4. The CTA/SM carveout is partially unlockable — name-gated.** The fork's
carveout pick owns the 1-CTA/SM pin, not the hardware: configuring the smem carveout
lets 2x 42-KB-smem CTAs co-reside (CTASM investigation). BUT applying it globally
regressed decode 40.35 -> 39.28 tok/s; ship usage is per-kernel-name-gated to the
prefill families only (`NV_SMEM_CFG_AUTO_NAMES=pfg`) (CTASM_INVESTIGATION, P8).

**PF5. Heavy register spill = nondeterministic kernel.** A 276 B explicit-c[2]
A-share spill made the w64q attention class nondeterministic on the dext
(bit-different across identical replays). Zero-spill builds of the same math are
deterministic (P18).

**PF6. Ping-pong smem stages don't break the GEMM wall.** 1-barrier vs 2-barrier
pipelined variants are bit-identical and within noise of each other (best 1.058x
vs the 1.2 gate): the wall is per-warp issue serialization, not stage
synchronization (P16).

**PF7. Persistent CTAs: mechanism answered, in-plan loser.** 713 GB/s steady state
at 82 CTAs (1/SM) with wave/ramp worth only +17% — NOT a stream limit. The
persistent-FFN ring measured 0.91-0.93x in-plan: the per-chunk decode->mma->sync
serial chain (~3.1 us x NCH=40) is the wall; perfect balance would still be
sub-gate (P13, P14_WIRE.md).

**PF8. IMMA/W8A8 relief is capped by mma issue.** The int8 tensor-core path's
mma-issue ceiling is 10% of tile time -> <5% net relief; below every gate. The
quant-GEMM lever is closed on all fronts (P12).

**PF9. 48 KB static smem is the ptxas cap on the raw-cubin path.** Kernels needing
more must go dynamic-smem (a dext capability question) or restructure. Related
compile-path laws: a KSYM file-vs-symbol name mismatch faults as
Illegal-Instruction-Encoding; an f16 half-view can read byte-size-OOB in an
order-dependent way (P7F2).

**PF10. Wave cliffs dominate long-context attention.** All split counts land at
~W/16 waves at 1 CTA/SM; e.g. S=12 -> 65 ms vs S=10 -> 40 ms per probe. The P18
growth pool cliff: cost jumps exactly at position 54040 = 7*CH (the ROWS=64
wave-2-goes-full step) — every other kernel class is flat (P18_GROWTH.md).

**PF11. The alloc-history fault class survives EFI cold-cycle** and reaches INSIDE
a warm host: post-boot fresh device allocations can fault while windowed uploads
and graph-slab reuse paths stay clean. Some fault classes clear ONLY via the
dock-power cold-cycle procedure: `pmset schedule poweron` + `shutdown -h now` ->
EFI-level power-on (full TB dock + GPU power cycle), not warm reboots (P13/P14,
P7E4).

**PF12. The ULP-amplifier law.** A 1-2 fp16-ULP difference at block 0 — invisible
in zero-seed worlds — amplifies ~1.25x/block through the 48 GDN blocks once the
mature conv-row-2 stream is present, decorrelating state by ~8k real-text tokens.
Six hot scan carriers therefore ride hi+lo fp16 splits (>=21-bit mantissa via
3-pass MMAs, sequential smem reuse). Zero-seed gates CANNOT clear a change that
touches scan numerics; seeded rec-seed gates are mandatory (P7E5/P7E6).

**PF13. The M64 tail-seam trio.** Batching by r%64 tails broke three separate ways
at once: a stale pos_slot surviving into the tail chunk; a graph-cache key that
replayed the M64 plan on the 32-token tail (ambient flag); and the head running
post-tail on a stale row. Multi-M chunking needs explicit seam keys (P15).

**PF14. Tie-mine gate texts need the both-sides control.** fp16-resolution top-2
gaps drive cross-class greedy divergence: a gate text that mines near-ties needs an
explicit control run, or a real regression reads as noise (P18 gate class).

**PF15. Gate harnesses need the FULL decode env.** Running a prefill gate without
the complete decode environment (split-KV/quant env vars) faults deterministically
at the first decode-graph execution — an env law, not a code bug (P7E7).

**PF16. Chunked prefill keeps the stseed_spec(N&1) parity contract.** The
speculative-draft re-seeding parity depends on the fed-token count's parity; any
chunk restructuring must preserve it (P3/P4).

## R. Deep-K + R2-series laws (R1-R5, R2b-R2d)

**R1. The M-extension store-audit law (R5a).** Mechanically extending a batched
kernel family M -> M+1 is a bug factory: the three R5a slips were ALL row-4-only
(a missing q-row store, a missing accumulate line, launch_bounds vs the
1024-thread name-law launch). gen_m6..8.py therefore audits EVERY row-write
family for the new top-row store and preserves source bounds through renames.

**R2. The REC-CHAIN SLOT LAW (R5c).** In the deep-K scan (k2s{n}), step t=K must
read rec{n}x scratch (the t=K-1 state), never a live slot: reading slot K-1 of
the [48][5] live layout reads the NEXT BLOCK's slot 0 garbage -> NaN on the top
row. The live slot stays 4 regardless of K (zero trunk/serve surgery).

**R3. The RP>NW owner law (R5b).** When a ROWS=K attention kernel's row-period
doesn't divide the warp-owned row range, single-row owners silently leave the
tail rows unwritten (combine reads 0/0 -> NaN). MAXOWN-generic owner paths; the
fix is codegen-identical for divisible cases.

**R4. The deep-K scan-range law (R4).** The n-gram drafter's history scan for a
K-deep proposal must stop at iend = pos-8-K (the suffix must not overlap the
positions being proposed). Off-by-one here self-matches the proposal prefix.

**R5. The tok_hist seeding law (R3/R5d).** Any LOOKUP drafter needs the token
history seeded on EVERY entry path (boot/FRESH/FOLLOW_UP/CACHE_HIT/snapshot_load)
— an unseeded -1 prefix self-matches -> -1 drings -> OOB fault. Never exercised
before deep-K shipped.

**R6. The uninitialized-smem is launch-history law (R2b).** Uninitialized smem
left by the WY T-solve consumed launch-history-dependent garbage: 0.0 on a fresh
SM, fp16 leftovers 7.6e-6 after a sibling kernel, arbitrary after trunk kernels
= an in-plan NaN that a clean repro CANNOT show. Zero every unwritten triangle.

**R7. The ring-depth coin-flip law (R2d).** A DBUF ring-depth extension on a
mixed-twin pf_gemm3m kernel is a per-CLASS, per-SHAPE bet on the SHARED register
allocation (ring regs slow the classic majority segments): gdnqg won x1.098,
fd/out/qkv lost x0.79-0.93. A/B at the exact in-plan grid before adopting.

**R8. The M128 tail must clear BOTH ambient flags (R2c law-2 re-commit).** With
the M64 ambient flag left on, the graph cache builds the M64 plan while the
34-row tail runs the M32 path — the M32 chunk replays the M64 graph on M64 ids
(first-div-0 at the 8k gate). Multi-M tiers need per-tier ambient keys.

**R9. The P7E4 corruptor was launch-history, not shape (R2b).** The quarantined
attnqkv 2-M-block twin is bit-identical in ONE launch on the M64/M128 trunk —
the P7E4 corruption class was SC-path launch-history. Quarantine lifted with A/B
nz=0; the g=448 consolidation shipped (R2d).

## Q. R7-decider + P8 laws (R7/R7a/R7b/P8)

**Q1. THE QMD BARRIER LAW (R7b).** `bar.sync id>=1` faults with SM "Illegal
Instruction Parameter" on every SM -> channel wedge. ROOT CAUSE WAS OUR OWN FORK:
ops_nv.py built every program QMD with `barrier_count=1` (only barrier 0 legal).
The fork patch (env-gated `NV_QMD_BARRIERS`, default 1 = byte-identical canonical,
+ `NV_QMD_BARRIERS_NAMES` scoping) unlocks named barriers 1..15 — verified
checksum-correct at full speed (665 GB/s pipeline arm == the all-warp arm). The
named-barrier primitive family is open for future kernels. (`NV_QMD_BARRIERS` is
in `patches/tinygrad-fork.patch` as of the R8/W5 republish; the SHIPPED engine
never issues bar.sync>=1 — only the R7b smoke harness and the parked warp-spec
kernel do.)

**Q2. MBARRIER IS DEAD ON THIS DEXT (R7b).** `mbarrier.init/arrive/test_wait` all
EXECUTE without fault but phases NEVER COMPLETE — both producer and consumer
bounded-spins time out, checksum 0. sm_86 details: the parity operand must be a
compile-time immediate (register forms are sm_90+; try_wait entirely sm_90+).
Do not build mbarrier pipelines on this rig; named-barrier MEETS are the only
subset-sync primitive. (Harness corollary: bounded spins are mandatory — the
faults/timeouts then exit cleanly with flag writes, no reboot.)

**Q3. Meet-based warp-spec converts the E1 headroom NEGATIVELY (R7b).** The E1
microbench proved the GEMM wall is latency-ORDERING, not issue-RATE (arms issue at
different rates; the production mix hits 493 GB/s == pure independent loads), so
1.8-1.9x headroom exists — but a producer/consumer warp-spec build with per-K64-stage
full-CTA meets measured x0.82-0.96 at every shape (bit-identical): the meet couples
producer smem stores with consumer decode on the critical path. A retry needs
working subset signaling or a persistent-CTA megakernel (pipeline inside one CTA).

**Q4. The uint4 DCE law (R7).** Consuming only `v.x` of a loaded `uint4` narrows
the LDG.E.128 to 4B component loads — consume all 4 words or the load-width
experiment lies.

**Q5. Load-width laws, both directions (R7a/R7b).** DECODE: cutting W-load
instruction count 3x via a uint4 merge on already-16B-aligned units is ~PERF-NEUTRAL
(the GEMV critical path is x-loads+ALU, not W-load count). PREFILL: doubling X-load
width (uint2 -> uint4) is NEGATIVE on latency-bound GEMMs (x0.78-0.89) — the
2x-more-numerous narrow stream keeps more independent loads in flight. Load-latency
structure rewards issue diversity, not width.

**Q6. The norms CTA-serialization law (R7a).** One-CTA (grid=1) serial-t-loop
norm kernels at M rows cost ~Mx their parallel time; per-row CTAs (t = blockIdx.x)
with verbatim lane loops/shuffle trees are bit-identical and free. Audit every
grid=1 M-row kernel for this — it was worth +5.43 tok/s.

**Q7. The launch floor and the 1-CTA ceiling (R7 E1).** Launch+wait floor on this
stack = 0.198 ms (empty kernel, synced min-of-8); effective SM clock through the
dext ~1.12 GHz (clock64-calibrated). The single-CTA W-stream ceiling at 256
threads/CTA = ~490 GB/s (independent uint4 loads, 1 CTA/SM) — the honest
denominator for all GEMM utilization claims at this occupancy.

**Q8. Big host copyins are a DART fault class (R7).** A host `_copyin` of ~1 GB
(16 MB chunks) faulted mid-way; fill large buffers device-side.

**Q9. The ambient-flag graph-set law (P8 — the P0 stale-feed fix).** The R2c M128
rung made `_pf_graphs`' ambient-flag inference ALWAYS pick the wrong set for
M64-context calls (`m64 = M64ON and not M128ON` is always False under PF_M128=1):
64<=n<128 FRESH/delta requests, m128 r>=64 tails and M32 tails under a resident
plan128 submitted M32 graphs (reading ids32/pos_slot the M64 loop never writes =
STALE content at stale positions) or M128 graphs (reading ids128). Symptom: FRESH
chats answer the PREVIOUS request's prompt, deterministically one-request-stale.
FIX: explicit `which="m32"/"m64"/"m128"` threading through `_pf_submit_chunk` ->
`_pf_graphs`, per-path graph sets (legacy inference as fallback) + the P15
pos-upload law applied to the M32 r>0 tail (which ran from a pos_slot the M32
chunk graphs never advance -> re-fed at pos 0 — ALSO the true identity of the
"8k F 4.284e-02 M32-reassociation floor"; the fixed 8k gate reads F 9.824e-04,
A 60/60). HARNESS LAW: same-content prefill gates CANNOT see stale ids — the
repro is successive DIFFERENT-content prefills vs T1 refs (`p0_repro.py`,
`P0_REPRO=1` in test_w100k.py).

**T1. P.poison takes BYTES (T2).** `P.poison(name, NBYTES, dtype, val)` — passing
element counts for f32/i32 buffers under-sizes the fill 4x -> OOB stores -> SM
fault. A multi-hour fault storm dressed as a "4-arg launch law" was this bug.

**T2. The ternary-negation int-shift packing law (T2).** Composing packed s8 via
`((int)(signed char)(neg ? -l : l) << 8j)` CORRUPTS bytes 1-3 (attenuation ~2^-j
per byte — only-negative values wrong); explicit two's-complement byte composition
(`(unsigned char)(256-level)`) is exact. Isolated by a mixing-matrix probe
(single-k xq vectors regressed against all predicted W columns).

**T3. The per-32k-scale-word law + the 8-linear-levels grid (T2).** The fork's
IQ3_XXS db exponent (sw>>28) is per 32-k SCALE-WORD, not per 256-block — any fused
decode must rescale per k-32-group (conveniently the mma k-step granularity).
And the iq3xxs grid is 8 LINEAR levels [4,12,20,28,36,44,52,62] ~= 4.0507*(1..15):
an int4 linearization at delta=4.0507 costs only 1.49% weight-RMS (8.4x better
than a raw int4 requant) — IMMA-on-packed7 is therefore numerically cheap.

**T4. The discriminator-frame law (T2).** Per-launch speedup claims MUST match
plane-count and M-grid between arms: the P8 "x4.58 IMMA" compared one-plane M=64
g=272 against the census's two-plane M=128 g=1088 launch; apples-to-apples the
true class is x1.06 fused / x1.56 int4-plane. Related dext facts from the same
round: the char4/make_char4 store codegen faults while the identical hand-packed
int STG is clean; the full-100k-state VRAM headroom is <1.9 GB (a 1.93 GB plane
upload faults mid-copyin; 1.2 GB fits) — new-plane ports are
partial-coverage-only (the G3M budget pattern is the right structure).

New cross-references: M1A_SERVING.md, M1B_SERVING.md, M1C_STABILITY.md,
SERVING_PLAN.md (S-group); P1_*.md .. P18_GROWTH.md, p1c/p1d/p1e_findings.md,
CTASM_INVESTIGATION.md (PF-group); R1_PROMPTCACHE.md, R2_RUNGS.md, R3_DECODE.md,
R4_DEEPK.md, R5_DEEPK.md (R-group); R7_DECIDERS.md, R7A_DECODE.md,
R7B_DECIDERS.md (Q-group); T2_P8W4.md, R8_DECODE.md (T/R8-group). Narrative
summaries: docs/SERVING.md, docs/PERFORMANCE.md, docs/history/PREFILL.md,
docs/history/CAMPAIGN.md (Phase R + Phase R7 + T2).

## X. R6-batch + R8-ladder + TLX fix-campaign laws (R6/R8/W3-W5)

**X1. THE COMPACT-PARTIAL SLICE LAW (R6).** The spk attention partials
(pm/ps/pA) live in COMPACT per-(ROWS) regions — `pb0=(g*S+s)*RMAX` with
RMAX=6*ROWS — so a ROWS=T kernel touches a contiguous `4*S*6*T` block and
per-stream batch slices at `r0*4*S*6*4` are collision-free. Shared staging
(qw3/qw16, draft partials, q/k/v/core) is shareable ONLY because pre->a are
adjacent serialized launches (per-launch transients). This is what makes
zero-new-CUDA batched decode legal.

**X2. THE R6 VRAM LAW (R6).** The PF chunk-prefill machinery's ensure-time
allocations (ensure64/ensure128 scratch + G3M packed7 extras + PF_P5 planes,
~6 GB) plus ONE extra stream's state banks (+4.3 GB/stream: 16xkv+16xsc+rec4
+conv4+kv_d pools — pool stride is COMPILE-TIME, every stream allocates
full-CTXK regardless of actual ctx) = fault-as-OOM with a MOVING floor (the
faulting op differs per boot: r7 uploads, scscr128, init_draft slice copy;
deterministic 3/3). Batch boots must carry `PF_G3M_MB=0 PF_P5=0 R6_PF_T1=1`
(prefill via the T=1 trunk path); decode perf is unaffected.

**X3. THE BT>RM LAW (R6 rung 3).** A bigger batch trunk does NOT need a bigger
LOOKUP_K boot: the RM-sized probe scratch can be REPLACED post-boot
(pre-graph-build = fixed-handle-safe) with BTMAX-row buffers + the M-family
cubins loaded directly — an M10 set at an LK=7 boot. Saves the LOOKUP_K=9
boot's +755 MB deep-set scratch (which alone tips the VRAM law).

**X4. THE GRAPH-CLASS PREFILL BUDGET LAW (R6 P3).** T=1 trunk-graph prefill
replays do NOT reset the dext ~950-cycle budget the way legacy EAGER-class
chunk prefills did — a 15-min soak wedged deterministically (2/2) at
~4000-4500 continuous mixed graph cycles when the global rebuild counter reset
on every barrier prefill. Fix: the counter NEVER resets on barrier work; only
the fence-all rebuild resets it (BATCH_REBUILD_EVERY=232 global).

**X5. THE KERNARGS-SLAB LEAK (R6 P3).** Every ParityGraph build allocates a
nolru host-mapped kernargs slab that is never released — a full fence-all (22
graphs) every 232 cycles exhausted host-mapped memory at ~250 fences
(alloc_sysmem fd-less MAP_SYSMEM_FD -> IndexError). Shipped mitigation:
ROTATED fence (one graph set per event, 4-5x) + FAILED FENCES ARE NON-FATAL
(old graphs stay valid, counter resets, generate continues — the W5 finding-#9
posture, ported to both schedulers). Durable fix = ka-slab reuse (open).

**X6. THE EMIT-KEY CONTRACT (R6 P3).** DecodeSession emits under the dict key
"cycle" — every emit-dict producer must too; a mismatch silently drops cycle
events inside try/pass sends (a per-Swap name-keyed readout hazard).

**X7. THE BIMODAL MATCH LAW (R8).** On 100k-token natural-text histories the
engine-selected best-window match length l is BIMODAL {0, 8} (prose: 998x 0,
2x 6, ZERO 7; gate: 22x0/38x8): relaxing LMIN 8->6/7 adds ~0.2% fires, all
spurious. The HyperQwen/llama.cpp graded-lookup tier assumes a populated
l=6/7 middle — it does not exist on this workload class; the sim killed the
build (r8_lutsim.py, three corpora, engine-exact scan semantics).

**X8. THE MACRO-DEF INSERTION LAW (R8).** When appending a new RED/ACC-style
macro after an existing one in a shared .cu, insert AFTER the FULL
backslash-continued definition — inserting mid-continuation splits it and
breaks every downstream kernel split. Python-level string audits CANNOT catch
this; the compile is the only truth.

**X9. THE UNROLL-CREEP LAW (R8).** Each M-rung tips a DIFFERENT kernel family
over the 64-reg/1024-thread budget at unroll 4 (ao8nw32 at M=10;
op38nw32/down8nw32/down8nw32vNr7 at M=11); body-scoped unroll 4->2 / 5->2
fixes each at ZERO per-row fp-order change. Scope the pragma per-kernel-body
(the text is shared verbatim across families).

**X10. THE G1 TRUNK-GENERATION LAW (W3).** A pcache node's boundary "hlast"
capture must read the hlast OF THE TRUNK GENERATION THAT RAN: the midprefill
hlast law read xA64 row 63 — an M64-trunk law; under the M128 trunk the chunks
ping-pong xA128/xB128 and never touch xA64, so the capture read the ensure64
POISON (7.7e31 — FINITE fp32; the x^2 RMS-norm overflow then makes fp16 logits
NaN and h_argmax leaves tok_slot 0). Fix: per-generation source (_trunk_
boundary_hlast: xA128 row 127 under M128, xA64 row 63 under M64) + sane-range
guards at capture AND restore. The residual: the trunk->spec-slot
boundary-state equivalence itself broke with the R2c-era M128/DR7 trunk —
midprefill/turnend CACHE_HIT restores are coherent but not bit-exact
(FIX_CAMPAIGN.md finding #5; the R6 T1-boundary node class is the fix path).

**X11. THE FRAME-VS-SPILL LAW + TRIPWIRES (W4/W5, fork).**
EIATTR_MIN_STACK_SIZE is a FRAME metric, not pure register spill: the shipped
Tier-2 prefill families (pf*/p8*) run 104-592 B frames BY DESIGN, while the
P18 nondet datapoint was a 276 B SPILL kernel on the decode path. The fork's
NV_GRAPH_ASSERTS tripwire policy (default: hard launch-mismatch + hard
>NV_SPILL_HARD_B stack; warn 1..100 B) with name-scoped NV_SPILL_EXEMPT_NAMES
(warn-only) keeps the hard >100 B law where it binds (decode/canon: max 32 B)
without failing the documented-frame families. Calibrated against the
782-cubin census (engine/w4_census.json; rung manifests in engine/manifests/
+ engine/rung_manifest.py assert the loaded set at every boot).

New cross-references: R6_BATCH.md (X1-X6), R8_DECODE.md (X7-X9),
R1_PROMPTCACHE.md W3 section + FIX_CAMPAIGN.md (X10-X11).

## MM. Multi-model / MoE-campaign laws (MM P0-P10; the Qwen3.6-35B-A3B port)

Journals: docs/history/MM_P0_results.txt .. MM_P10_results.txt (+ MM_PLAN.md,
MM_P2_kmix_manifest.md). The campaign ported a second architecture to the
same dext; every law below was paid for in faults or bit-exact gates.

**MM1. THE NAME LAW, SHARPENED (P0/P1).** The TinyELF/program name MUST
equal the cubin's kernel SYMBOL. A mismatch (e.g. a file-prefixed name)
silently skips the .nv.info parse (maxntid/stack never read) and the launch
faults with SM "Illegal Instruction Encoding" on ALL GPCs — 100%
correlation across ~15 boots; it was also the historical queue-replay wedge
("graphs fault, eager clean"). Variant cubins MUST rename their symbol
(the P6 recurrence: h6kam/spkq256m/spka256m all faulted under prefixed
loads).

**MM2. THE SINGLE-ARG SIZE CAP (P0).** Hand-NVProgram buffer args over
~457 MB-1.16 GB fault with the same signature (457 MB x 8 args clean;
8 GB/3.66 GB/2.38 GB/1.16 GB single args fault). Per-layer routed banks
(~373 MB) are legal. VA POINTER TABLES (device-side tables of absolute
64-bit VAs, read on-device) are LEGAL — the earlier "pointer-chase illegal"
reading was this cap biting.

**MM3. THE PIPELINING / WAIT-EACH LAW (P1).** Submitting a second graph
while the first still owns the QMD races SKEDCHECK22_INVALIDATE_ACTIVE_QMD
(timing-dependent, 0-2 cycles of grace at 256-cycle replays). THE
discipline: WAIT every graph before the next submit. (Same root as the P7F1
law; re-learned at MoE graph volumes.)

**MM4. THE CONVERTER LAWS (P2/P34/P9 — bit-exact vs llama.cpp C):**
IQ4_XS nibble map (elements 0-15 of a 32-group = LOW nibbles of bytes 0-15,
16-31 = HIGH nibbles of the SAME bytes); Q3_K's `d` comes LAST
({hmask|qs|scales|d}) and the SAME 32 hmask bytes serve BOTH 128-groups; the
PAIR-VS-POS law (gx up-kernels consume the SHARED hn, pos-addressed — a
per-PAIR copy diverges); mutator-sized buffers (a mutator writes past a
readonly view's extent — the hrot OOB class); the uint8-subtract trap
(numpy (uint8>>s&3)-4 wraps to 255 unless cast first); BF16 is TYPE 30 in
the modern GGUF enum (verify from bytes); the OBJECT-ARRAY TRAP (npz rows
are boxed pointers — denormal logits if read as flat).

**MM5. THE NARROW-LOAD LAW (P9, ptxas load coalescing).** Consecutive
u16/u8 reads from ONE thread get merged into 32-bit loads — MISALIGNED
(fault or silent garbage) in any odd-stride block layout (Q3_K's 110 B
blocks). Inline-PTX `ld.global.u8/u16` are unmergeable. The trunk's IQ3_S
kernel never hit this because its u16s are per-LANE strided, never
consecutive.

**MM6. THE Q3_K SHARED-HMASK LAW (P9).** The hmask bit for (nn, j) is
`1 << (4*nn + j)` — m NEVER resets between the two 128-groups (bits
1,2,4,8 | 16,32,64,128). Using `1<<j` for both halves corrupts lanes 16-31
(the upper 128 elements of every row).

**MM7. THE SILENT-NO-LAUNCH LAW (P7).** A 1-TUPLE grid passed to the graph
builder becomes a malformed 2-tuple global_size and the graph NEVER
EXECUTES — silently (the output buffer is never written; a later graph's
stale-correct output masks it). Normalize 1-tuples to scalars in every seq
builder. Bit EVERY P7 graph on first contact (the mock-vs-rig meta-law
again).

**MM8. THE FULL-SYNC COPY LAW + THE UNCACHED SYSMEM FOLD (P7).** Allocator
_copyin/_copyout each do a FULL dev.synchronize() (the old ~14 ms host path
was six of them). The fix class: cpu_view() over cpu_access=TRUE +
**uncached=TRUE** sysmem buffers — bidirectionally coherent and free
(VRAM-vs-sysmem A/B identical); cpu_access WITHOUT uncached maps VRAM
GPU_CACHEABLE_YES (the stale-read race class — do not use for per-cycle
control). This is the host fold (host residue ~0.2 ms/cycle).

**MM9. THE KAPool SLAB COHERENCE LAW (P9).** MGs carved from a SHARED
kernargs pool slab WEDGE (GSP: SKEDCHECK16_CTA_THREAD_DIMENSION_ZERO — the
GPU reads STALE kernargs/QMDs through the slab's GPU-CACHEABLE mapping) for
graphs built/stepped after the GPU has traffic over the slab; only the
boot-time graph set is safe. FIX: a DEDICATED uncached ka per graph-runner
(sysmem GPU_CACHEABLE_NO — the cpu-fold class). (The dense gcycle's ka-slab
leak once noted here as the follow-up is now FIXED by the P10 pool — see
the KA laws; the MoE path reuses slab slices across rebuilds — audited.)

**MM10. THE GPU-EXIT LAW, EXTENDED TWICE (P7/P8).** (a) The GRACEFUL
socket shutdown (bye:true) ALSO reboots the machine (6/6) — disable-first
+ `sudo sync` is MANDATORY before any stop, and an unsynced launchctl
disable is LOST to the reboot. (b) THE SYNC LAW: a NEWLY-CREATED file +
fsync + parent-dir fsync can still DIE in the GPU-EXIT reset; `sync; sync;
sleep 3; sync` BEFORE the stop makes the swap intent SURVIVE. Asymmetry:
appends to EXISTING files survive unsynced — the killer is dirent creation
vs inode append. This is why swap intent is written (atomic + dir-fsynced)
BEFORE the stop RPC.

**MM11. THE 2D-GRID LAW + GRID NORMALIZATION (P34).** The gv8k2048p/gv8k4096r
families are (rows/32, P) grids — a 1D grid silently runs seat 0 only (the
S8 K-mix smoke was det-but-seat-0-only). Applies to every M-batched family
by construction.

**MM12. THE L>256 SPKQ256 LATENT BUG CLASS (P56).** A parallel-exp softmax
that only covers positions < 256 mixes raw scores into z/out for L in
257..1024 — invisible to any <=52-position battery. Fix: d==0-sequential
(BIT-IDENTICAL at L<=256). Corollary: position-dependent code paths need a
battery that crosses every hard-coded boundary.

**MM13. THE SPLIT-S NUMERICS CLASS (P7).** Split attention at S != S'
differs from itself (~7e-2 state maxdev over 256 tokens); same-S is
bit-exact. Anchors and engines MUST pin the same S per phase (a "state
DIFF" that looked like a chain bug was the S=32-vs-S=8 arm mismatch).

**MM14. THE LONG-HORIZON LAW (P56).** The engine state after 4,000/16,288
ingested doc tokens continues greedy-identically to the fp16 anchor
(rebase16 16/16 EXACT at every rung) — GDN norms bounded (148). Long-context
exactness is MEASURED, not assumed, at every ctx rung.

**MM15. THE FP16-PARTIAL LAW (P34/P56).** Storing the down-proj partials
as fp16 (per pair/rank) is the numerics tier that matches the fp16 anchor
(348/349 vs the f32 accumulation's 347/349 — the residual is the adjudicated
drift class). Priced, adopted.

**MM16. THE SMALL-SHAPE LATENCY LAW (P34).** At P<=9 the MoE kernels are
LATENCY-flat across K=2..K=8 (shape does not matter — launch/latency floor
owns the cycle); per-shape tuning only pays at scale.

**MM17. THE FP-EXTRA CONVERGENCE LAW (P8-SERVE).** The API and the daemon
must derive the SAME per-model config_fp extras (dense -> cubin-set digest;
MoE -> the packed-manifest sha) or EVERY boot drifts. Corollary: a
mid-session repack invalidates the resident daemon the minute the packed
manifest changes (the fp discipline firing live, correctly) — repack +
daemon-restart are ONE operation.

**MM18. THE PF-ONLY CUR LAW (P10).** After a PF-only feed (prompt length %
256 == 0 runs ZERO T1 steps), the last hidden lives in PFB["hA"][255], NOT
the decode graph's rig.hA — eager-head derivations that read rig.hA derive
`cur` from a STALE decode hidden (wrong continuations, INVISIBLE to
spec-vs-T1 gates: both arms share the same wrong cur). Track the feed's
last op; the bit-exact chunk-256 hidden IS the T1 hidden.

**MM19. THE MARGINAL-LAUNCH LAW / LAUNCH-SERIALIZATION CORRECTION
(P10-B).** The ~0.094 ms/launch serialization slope was measured on
TRUNCATED graphs where every kernel is tiny; in real fat-kernel graphs the
dispatch PIPELINES BEHIND EXECUTION — the marginal launch is ~0.02 ms and
the T=1 cycle is KERNEL-RUNTIME-bound. Launch-count-only fusion projections
(the "165 kernels -> 17 ms" class) are INVALID; fusion pays only where it
makes the KERNELS faster (fewer weight passes per layer).

**MM20. THE FIRST-CONTACT PCACHE GATE SHAPE (P10).** A spec-vs-T1 gate run
right after pcache ingest compares FRESH-vs-RESTORED (arm A feeds + ingests,
arm B restores); reruns compare RESTORED-vs-RESTORED (identical inputs) and
always pass. A "transient" first-run mismatch in this shape is a
fresh-vs-restore divergence (or a process-layout ULP residue on degenerate
repetitive eras — the adjudicated G3 class: state bit-proven identical, NOT
the cur, NOT MTP), not noise and not necessarily a bug; label which shape a
gate runs.

**MM21. THE STAYDOWN-DELETER (P10, open).** The daemon's hard-death path
(shutdown RPC -> bye -> death INSIDE dev.synchronize(), no clean_exit slog)
deletes/never-writes the staydown marker and launchd relaunches within
~30-60 s. Holding a GPU window through a shutdown currently needs an
external marker-keeper; root cause open.

**MM22. THE PING-PONG / SCRATCH-STRIDE / CONTAINER-CWD / QQ-WINDOW set
(P34/P56):** build_seq must alternate hin=hA/hB per layer exactly as the
combine reads (a fixed buffer = every layer reads layer-0's partials);
spkq256m scratch = 16*P_MAX*CTX (the 16k rung faulted at 143*16384 >
16*65536 — size scratch by the REAL max, not the current rung); the nvcc
shim's paths must be ABSOLUTE (container CWD differs); drafts follow the
EXTENDED qq match window, not the trimmed one.

## KA. P10 kernargs-pool laws (the F2/F3 fix; P9-eval findings F1-F7)

Journals: docs/history/MM_P9E_results.txt + eval/P9_EVAL_RESULTS.md. The
P9 quality battery (GSM8K/PPL/needle through the live API) is what
surfaced the exhaustion class; the P10 pool
(`engine/ka_pool.py`, `TLX_KA_POOL=1` default-on) is what closed it.

**KA1. THE MAP_SYSMEM_FD CAP / NO-UNMAP LAW (server.c).** The tinygpu
server caps live sysmem mappings at MAX_SYSMEM=128 and has NO unmap RPC —
`g_sysmem[]` holds its own mmap + shm fd until process death, so every
fresh host-mapped kernargs slab burns one server slot for the whole
daemon life. Exhaustion has two faces: SOFT (F3) — MAP_SYSMEM_FD returns
no fd -> `IndexError: list index out of range` at the ancillary-data
unpack (system.py:383), first seen on new PfGraph builds after enough
graph churn; every subsequent long FRESH/CACHE_HIT prefill fails — and
HARD (F2) — whole-process-tree deaths and machine resets under sustained
dense decode (MTBF 13-18 min), NOT the classic ~950-cycle budget (fences
firing normally; deaths at cumulative cycles 906-1457). Fresh boot clears
both. Burn rate accelerates on long prompts: crossing ATTN_THR builds the
gs26 graph class, and prefill class churn rebuilds PfGraph sets.

**KA2. THE KA-SLAB LIFETIME ASYMMETRY (dense vs MoE).** The dense stack
carves a FRESH slab for every ParityGraph/PfGraph build (~6-10 slabs per
fence: gen_rebuild / l7_entry_rebuild -> build_graphs, plus PfGraph sets
on every prefill class change) and simply dropped the retired ones —
never returned. The MoE path never trips the cap because its fence
(MG.rebuild) RE-FILLS kernargs into the SAME slab — zero new mappings.
Same engine family, opposite lifetime discipline: that asymmetry is why
F2/F3 looked dense-specific.

**KA3. THE ZERO-ON-RELEASE DETERMINISM CONTRACT.** A fresh shm mapping is
zero-filled; a pooled slab zero-filled at release is byte-identical at
hand-out, so the GPU-visible bytes are exactly what the new build's
fill_kernargs writes plus the per-submit QMD patches — the same bytes a
fresh slab would carry. This is the whole determinism argument for slab
reuse, and it does NOT contradict MM9's KAPool SLAB COHERENCE law: MM9's
wedges were late CARVES from a traffic'd SHARED arena (the GPU reading
stale kernargs through a GPU-CACHEABLE mapping); recycling whole,
previously-good slabs with a full rewrite at a quiescent point is the
proven-safe shape (the MoE 37-fence soak precedent; the P10 gates:
bit-exact probe trace + GSM8K first-20 20/20 identical).

**KA4. THE BUILD-THEN-SWAP LAW.** Retire the replaced set's slabs only
AFTER the replacement set is live (a failed build must leave the OLD
graphs fully servable — the W5 posture), and only at a quiescent point:
every retirement site already dev.synchronize()s, and ka_release
re-asserts the sync defensively (the V-56 close() discipline). Corollary:
the snapshot-restore reset must retire + clear ALL graph attrs (a handle
set left live on stale buffers is the wrong-answers class).

**KA5. THE STAYDOWN-MARKER / NEWLY-CREATED-FILE LAW (P9 F7).** The
staydown marker written by the stop path DIES with the GPU-EXIT reset it
provokes (a newly-created file + fsync can still vanish — the
MM10b newly-created-file law); launchd relaunches the engine within
~2-4 min. A PRE-CREATED marker (written while the box is stable) SURVIVES
the reset and holds the engine down for standalone-GPU windows. Extends
MM10b from swap-intent files to ops markers generally: pre-create, don't
create-then-die.

## PD. P10-dense prose-campaign laws (the full-vocab draft head + the K=4 verdict)

Journals: PERFORMANCE.md "P10-dense" section; eval/results/gsm8k_p10_dhead*
+ gsm8k_p10r2_spot.jsonl. The prose 23.0 -> 43.7 tok/s run + the
honestly-falsified K=4 EAGLE chain (machinery gated, default K=2).

**PD1. THE SLICE-HEAD PROVENANCE LAW (rung 1).** A draft head over a
prompt-frequency vocab SLICE cannot propose tokens it never saw: 89.8% of
novel-prose targets were OUT-OF-SLICE (top1==target 6.1%) under the
40960-row slice — measured by a rank histogram, not inferred. When a
speculative mode accepts ~0, measure WHERE the targets land in the
drafter's proposal space before touching the chain or the acceptance
machinery. (The fix — the full-vocab head — reuses the resident trunk head
plane: zero new VRAM, and the draft scratch rides a dead logits row.)

**PD2. THE T=1-TRAP / PENALTY-ASYMMETRY LAW (rung 1, serving).** An
alpha-death fallback whose EXIT condition can never fire on the workload
that triggers its ENTRY is a one-way trap: the T=1 exit needs an 8-gram
hit, which novel prose never produces — entering T=1 parks decode at the
T=1 rate for the rest of the generation (the observed 20.6-trap: ~9
ms/cycle saved vs 60-90% throughput lost). Under the full-vocab head
(acceptance ~0.6+/cycle) the 4-zero-accept entry trigger is pure noise —
suppress the ENTRY, keep the controller verbatim knob-off.

**PD3. THE v3->v5 M-EXTENSION GENERATOR PATTERN (rung 2, gen_r7d5).** The
r7 GEMV families M-extend by SOURCE TRANSFORM, not by hand: extract the
_v3r7 body, extend the ACC/RED macros row-wise (ACC3H2 -> ACC5H2, RED3 ->
RED5), keep the R7U uint4 W-fetch and the per-row fp order VERBATIM, and
insert the new macro defs AFTER the full \-continued existing defs (the X8
MACRO-DEF INSERTION law). Gate = A/B vs the packed originals nz=0 det x2
(the r7d_test v5 pairs) + the ptxas audit (50/64 regs, 0 spill — the P18
spill law). First-build-green at M=5.

**PD4. THE INTERMEDIATE-BUILD PROVENANCE LAW (rung 2).** A perf log line
is only as good as the build it ran on: rung-1's in-harness "46.46 ms /
1.0 tok-cyc" prose number came from an INTERMEDIATE build — its rep shows
56 T1-cycles, IMPOSSIBLE under the final T=1-entry suppression (which
forbids T=1 cycles entirely). Falsify a banked number by checking its
BYPRODUCTS against the final code's invariants (cycle-type counts, knob
states), and re-derive every headline number on the FINAL build before
publishing it (the shipped K2 world: 69.03 ms / 1.583 tok/cyc = 22.9 tok/s
in-harness @100k; the daemon's 43.7 is GSM8K-class work at ~600-token
contexts — different regimes, both true).

**PD5. THE AMDS ROW-PINNING LAW (rung 2).** An in-graph argmax over a
multi-row logits buffer must pin its row count to the T the graph PROBES:
under LOOKUP_K=10 the deep probe writes RM=11 rows, so a probe-side amx3
launched at the buffer's full row capacity would argmax STALE rows and
clobber amds[5..] with garbage argmaxes before a later deep cycle rewrites
them. Pin the rows to the active set's M (5 for the T=5 EAGLE probe),
never the buffer's capacity.

**PD6. THE CROSS-CLASS FLIP LAW (R8_PROSE adjudication, 2026-09-29).**
The novel-prose gate's stable 7/60 vs the T=1 ref (first div idx 6,
deterministic x2, identical across K2/K4/K10 and every draft-head variant)
is the tie-mine law at FAMILY-OFFSET scale, not fp16-ulp scale: the M=3
probe kernel family and the M=1 trunk family rank the SAME top-2 pair at
the divergence position but disagree on the order — measured full-vocab at
the flip: T1 margin +1.17 for its pick, probe margin +0.33 for the other,
cross-family logit drift MEDIAN 0.379 / max 4.15 concentrated on the
flipped token (d=1.39 vs 0.11 on the stable one). The weaker margin sits
INSIDE the family drift: WHICH KERNEL FAMILY computes the greedy pick
decides the token. One flip + cascade (novel prose = flat top-2
distributions) means a low N/60 is ONE decision, not N; each stream is its
own family's consistent greedy chain (lossless within its world), so the
class is BENIGN — distinguish it from state divergence by adjudicating
with full logits (R8_PROSE_DUMP=1 in test_w100k: disjoint top-sets or
logits outside the drift envelope = STATE-DIVERGENCE, P0-grade; same
top-2 + margin-inside-drift = cross-class flip).

**PD7. THE SUB-ARGMAX LOGIT WOBBLE LAW (same adjudication).** Run-to-run
replays of the SAME probe cycle wobble ~±0.1 logits on ~237k of 248320
vocab entries (max 0.098) — BELOW argmax resolution at these margins, so
TOKEN STREAMS stay bit-stable across reruns, but a bitwise-LOGIT
determinism check would cry race. The determinism contract is and must be
stated at the stream/argmax level; only a bitwise-logit requirement makes
this class observable. (Same family as the first-launch-vs-rest
uninit-smem detector: quantify before classifying.)

**PD8. THE ANCHOR+SINGLE-STEP MEASUREMENT TRAP (same adjudication).** To
measure the T=1 reference's logits at a divergence position you must
CONTINUE from the same anchor the reference ran from (fd+1 single decode
passes). A single forward step at anchor(prefix) is WRONG: follow_up
builds state via the BATCH-PREFILL kernels — a different accumulation
than the decode path the reference itself ran, one feed-position late —
and its logits are wildly different. Any per-position adjudication that
mixes a prefill-built state with decode-path references measures the state
builder, not the numerics under test.

New cross-references: R6_BATCH.md (X1-X6), R8_DECODE.md (X7-X9),
R1_PROMPTCACHE.md W3 section + FIX_CAMPAIGN.md (X10-X11),
MM_PLAN.md + MM_P0..P10_results.txt (MM1-MM22), M1C_STABILITY.md
R3 live-window section (the L1-L5 ops fixes: pipe-vs-heredoc zsh tokens,
client-side staydown markers, stop-order, dir-fsync), MM_P9E_results.txt +
eval/P9_EVAL_RESULTS.md (KA1-KA5 + the P9 findings F1-F7),
PERFORMANCE.md P10-dense section + eval/results/gsm8k_p10_* (PD1-PD5),
the R8_PROSE_DUMP instrument in engine/test_w100k.py (PD6-PD8).

## Z. Third-model-deploy + drafter-program Phase-0 laws (P11-oblit + P0)

Journals: docs/DEPLOY_OBLITERATED.md (the worked deploy),
history/TLX_P0_LEVER_RANKING.md + TLX_ATT_RB_results.txt. These are
checkpoint/format truths (the D-family lineage) plus the pack-isolation
laws the weight-variant deploy exposed.

**Z1. THE NORM LAW (HF vs GGUF, law-grade).** The HF qwen3_5 checkpoint
stores ALL RMSNorm weights ZERO-CENTERED (Gemma-style): functional =
stored + 1.0. The GGUF and this engine use the functional form. Proven
element-wise: GGUF trunk+draft norms == HF stored + 1.0 EXACTLY (all 7
norm tensors, bit-exact). Any HF->engine weight path must add the +1 (and
any engine->HF path must subtract it), or norms silently run ~1.0x off.
Corollary earned the hard way: the rig GGUF **is** the first-party
checkpoint — an earlier "different lineage" reading was a fetcher bug
(see Z2).

**Z2. THE SAFETENSORS OFFSET LAW.** `data_offsets` in a safetensors
header are DATA-SECTION-relative (add `8 + header_len` for the absolute
file offset), not absolute file offsets. Range-fetching tensors at the
raw offset returns SHIFTED GARBAGE that still looks statistically sane —
which is exactly what made the misfetch read as "different lineage"
above. Always rebase by the header size before seeking.

**Z3. THE GGML ENUM-TRAP (type 21).** ggml tensor-type ids are a
HISTORICAL ENUM, not a stable ABI: current llama.cpp's type 21 is IQ3_S
(110 B/256w block), NOT IQ1_M. Requantizing "to the same map" by id
without checking the current enum wrote the wrong type on the first
obliterated pass — verify by block SIZE from the tensor layout, not by
the numeric id alone.

**Z4. THE MMAP RE-SLICING IS-IDENTITY BREAK.** `np.load(mmap_mode).T`
-style re-slicing of a memmap breaks `is`-identity bookkeeping: two
tensors that should alias the same bytes end up with DIFFERENT views, and
a scale-table lookup keyed on the array object silently returns the WRONG
tensor's scales (observed: K dequantized with V's scales — cos 0.78 vs
the expected 1.0, the class the microdiff ladder isolated). In
differential harnesses, bind scales to tensors BY CONSTRUCTION (one view,
one key), never by object identity after slicing.

**Z5. THE HARDCODED-PACK-DIR STALE-SWAP LAW.** When the runtime hardcodes
shared weight-pack dirs, a weight-only variant swapped via its own GGUF
will boot its trunk on the new weights but serve STALE shared planes
(prefill packs, draft packs) from the base model — silently, with no
fault, at full speed. Pack dirs must be per-model overridable
(`TLX_PACKED`/`TLX_PACKED7`/`TLX_PACKED5`/`TLX_DRAFT_PACK`), defaults
byte-identical when unset. And because draft-pack state persists INSIDE
pcache nodes (kvd/dhd), a draft-pack swap must also change the config
fingerprint (`svc_fp.set_draft_pack_extra` hashing the dir) or a cache
restore resurrects the old drafter across boots.

**Z6. THE OFFLINE PACK_RUNNER PATTERN.** `engine0.py` grabs the dext at
IMPORT (`dev = Device["NV"]`), which couples "pack a new model's weights"
to "own the GPU". The pack_runner replaces tinygrad's Device map with a
lazy stub BEFORE the import: packers run as pure file I/O while another
model is RESIDENT (no GPU window needed for a new model's packs; any real
dev access fails loud). Same pattern applies to any future tool that only
wants parse_gguf/read_raw from the engine module.

**Z7. THE SIM-CALIBRATION CONTRACT (chain_sim).** An offline simulator is
only a pack-selection instrument if its RESIDUAL is small relative to the
decisions it arbitrates: calibrate against a captured engine trace FIRST
(G0: m-dist shape + the ship metric within band), then require
pack-to-pack deltas >= 2-4x that residual before trusting a sim verdict.
Two instrument bugs the microdiff ladder caught on the way: the re-slicing
identity break (Z4) and a class-B target off-by-one (proposal_i compares
vs target(t+i), not target(t+i+1) — a one-token-lookahead definition
error that inflates acceptance and fits no symptom until the diff harness
runs).

**Z8. THE QUANT-PATH LAW (draft packs).** The shipped GGUF draft pack is
NOT bitwise RTN(bf16) of the first-party checkpoint: 53% packed-byte
equality, dequant cos 0.9978-0.9991 — same source weights, a different
upstream quant path (imatrix/rounding-class). The novel-prose canary class
is INSENSITIVE to the quant path (all three packs within ±0.017 E[m]k2);
every cross class is HIGHLY sensitive (the shipped pack beats pristine
bf16 AND fresh RTN by +0.10-0.14 on gsm8k, +0.03-0.09 on prose16k/code8k;
r8 swings ±0.08). Corollary: pack A/B comparisons and no-regression bars
are meaningless unless the quant path is pinned or the deltas dominate it.

**Z9. THE SCORER-CONCURRENCY CLOBBER.** Four chain_sim scorers appending
to ONE shared score JSON corrupted it (concurrent read-modify-write on a
single results file). The [v3score] summary lines each scorer prints to
its own log were the ground truth for reconciliation. One writer per
results file (or per-scorer shards + a merge step) — cheap insurance the
v3 battery paid for retroactively.

**Z10. THE PINNED-CPU SESSION-KV STREAMING PATTERN (24GB-class training).**
The v3 anchor set (7,361 train cycles, 10.2GB of int8 session KV) trains
on a 24GB 4090 by keeping each session's kv on PINNED CPU memory and
streaming it to the GPU per chain step (fp16 source, fp32 cast at the
kernel boundary) — identical math, host transfer fully hidden under the
chain step. For anchor-scale trainer runs, "the anchors don't fit" is a
placement assumption, not a wall.

**Z11. THE EXPANDABLE_SEGMENTS ALLOC CONF.** The first anchor training
step OOM'd with 5.8GB CUDA-reserved-unallocated — allocator fragmentation
under the streaming step's mixed allocation sizes, not a real memory
shortfall. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` removed it
entirely. Rent-by-the-hour 24GB instances running variable-shape training
should set it from the start.

*(Healing note, vast.ai: the ssh-proxy aggregate upload cap measured in
the anchor phase (~1.5-2Mbps regardless of stream count — relay or
compress anything >1GB) was HEALED by the v3 run: the same 1.74GB zstd
payload uploaded in ~3.5 min. Compression stays worth it — zstd -6 on
int8 kvd = 3.26x.)*

## PB. MoE-prefill campaign laws (Session A+B — the 203 -> 384 @2k campaign)

Journals: the pinned rig commits 6c47abc (Session A instruments) and
eb47535 (Session B ship); the instruments and gate records live in
`engine/mm/` (`mm_pf_bisect.py`, `mm_l1_poc.py`, `mm_probes.py`,
`mm_session_a.py`, `mm_g2_l2.py`/`mm_g2_l2.json`, `mm_f1b_b.py`/
`mm_f1b_b.json`, `mm_dbg_b.*`, `mm_wire_b.py`). The campaign's headline
method law is PB1 — it killed the plan's central premise before anything
shipped.

**PB1. THE LATENCY-BOUND-VS-DRAM-BOUND LAW.** Traffic reduction buys
NOTHING on a latency-bound class — and a bandwidth wall cannot be diagnosed
from in-graph numbers. The routed expert pair-walk measured 212.5 GB/s
effective in-graph (119.45 GB/chunk) and the five-analyst plan built a
25-30x DRAM amplification model on top of it; isolated (lone-graph) benches
showed the SAME stock kernels streaming ~750 GB/s — L2 absorbs the expert
re-reads, the in-graph number was contention, and the class is bound by the
dequant+dot DEPENDENCE CHAIN, not bytes. Corollaries: (a) MEASURE EFFECTIVE
GB/s IN ISOLATION before designing around a bandwidth wall; (b) the winning
kernel shape reduces DEQUANT EXECUTIONS, not traffic — the register-decode
grouped GEMMs (decode each row's weights once, FMA across the bin) measured
2.0-3.6x where the traffic-reduction-only v1 measured 0.86x; (c) when a
rewired path diverges, a PER-CLASS BISECT (truncated-graph family ablation)
isolates the culprit class in one run — it caught Session B's
dropped-`gvf32ab` wiring (abb garbage -> all-seat divergence) immediately.

**PB2. THE EXPERT-LOAD HISTOGRAM LAW.** Real router load is FAT-TAILED,
not Poisson: a 256-token chunk puts up to 253/212/217 tokens (prose/code/
gsm8k) on a SINGLE expert — 8x the plan's Poisson-30 — while the top-16
experts carry only 11.6-14.4% of the mass (weak Zipf, fat tail). Bin
tables, smem token staging, and grid sizing must come from the MEASURED
in-graph routing histogram (`mm_probes.py`'s eids-capture probe), never
from an analytic occupancy model.

**PB3. THE 16.4KB-SMEM DEFAULT-CARVEOUT TRAP (PF4 sharpened).** A 16.4 KB
static-smem kernel boots, runs CORRECTLY and BIT-EXACTLY under the dext's
DEFAULT 32 KB carveout config — at 1 CTA/SM, i.e. a third of its speed,
with no warning anywhere. Adding the kernel family to
`NV_SMEM_CFG_AUTO_NAMES` (AUTO picks the 64 KB config = 3 CTAs/SM) was the
measured 2.0-3.6x lever on the grouped up-GEMMs. Never accept a
grouped/smem-kernel speedup verdict without recording WHICH carveout
config booted.

**PB4. THE S=32-T1-VS-S=8-PF HARNESS LAW (MM13 extended).** The valid
exactness compare for chunked-PF tails is T=1-at-S=8 vs the chunk path
(bit-exact, gate-green); an S=32-vs-S=8 comparison across the split
boundary is INVALID by the split-S numerics class and will "fail" a green
kernel. Related gate hygiene: run batch-order-sensitive gates ISOLATED —
a G3-run-immediately-after-G2 showed a batch-order artifact that
reproduced neither isolated nor standalone (pre-existing class, not
adjudicated as a real divergence).

**PB5. THE MGUnc-FOR-LATE-GRAPHS LAW (MM9 sharpened).** EVERY graph built
after the GPU has already run traffic — instruments, bisect phases, PF64
tails — needs the dedicated uncached-ka GraphRunnerUnc; pooled-ka late
graphs fault (the KAPool slab coherence class). Live-proven across the
entire Session A instrument set, which builds graphs per phase on a warm
GPU by design.

**PB6. THE GRIDDIM-STRIDE WEDGE + 1-TUPLE NO-LAUNCH, LIVE AGAIN
(L1/MM7/MM11 corroborated).** The same instrument batch hit BOTH classics:
`MM_A_bwread` v1 walked `i += gridDim.x * BS` (reads 0 in SASS -> stride-0
infinite loop -> the dext WEDGED, a reboot window), and the `cp4k` eids
probe in its 1-tuple grid form silently never launched. Every new probe
kernel needs the two-line guard — hardcoded strides AND normalized
(scalar) grids — plus a bit-check before any timing is interpreted.

**PB7. THE NVCC-SHIM PATH LAW.** The containerized nvcc build shim needs
ABSOLUTE source/output paths (it cannot see the caller's cwd through the
container mount) and `DOCKER_HOST` in URL form `unix:///...` (three
slashes — scheme + absolute path); relative-path builds compile the wrong
thing or nothing. Encode the build root once at the top of every build
script (`mm_build_a.zsh`/`mm_build_b.zsh` are the reference shape).

**PB8. THE DYNAMIC-SMEM UNLOCK, MEASURED (positive).** Static smem >48 KB
is refused by ptxas, but the QMD-patch DYNAMIC-smem route PASSES at BOTH
64 KB and 96 KB per CTA — the structural branch (staged weight slabs,
deeper bins, the future mma/L1b class) is real and measured, not
hypothetical. The DRAM read path re-confirmed at 820.6 GB/s through the
dext.
