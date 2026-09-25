# W2H: HMMA IN-GRAPH FIX — 40.35 tok/s @100k Tier-1 EXACT (40 CROSSED)

## TL;DR
W2G's HMMA tensor-core K1 (m16n8k16 QK+PV) was standalone-exact but produced
deterministic garbage in-graph. **ROOT CAUSE CONFIRMED: the kernel NAME
`spk_g4nwhm3_100k` lacks the substring "nw32", and ParityGraph (gcycle.py)
derives the launch LOCAL SIZE from the kernel name — so the graph launched a
1024-thread kernel with 256 THREADS.** Direct/standalone launches pass the
thread count explicitly, which is why the kernel was exact everywhere except
in-graph. Fix = rename the cubins to `spk_g4nw32hm{3,1}_100k` (one -DKNAME
token). **Result: Tier-1 EXACT 60/60 (fresh T=1 ref regenerated under HM=1),
ZERO greedy flips vs the W2D/W2E sequences (60/60 overlap both), 3 timing reps
all 60/60 agreeing, BEST 68.98 ms/cyc -> 40.35 tok/s @100k** (alpha 0.892,
2.78 tok/cyc; canonical was 39.03 / 71.32 ms/cyc). New canonical recipe =
W2F + **HM=1**.

## The investigation chain (each step evidence-backed)
1. **Smem diet (W2G hyp.1) — applied and REFUTED.** Shrank static smem
   37,248 -> 35,200B (SCP plane [RP][TILE]->[RMAX][TILE] with write guards;
   msv/ssv online-softmax state moved to row-owner registers, only the
   cross-warp corv stays in smem). Standalone relerr unchanged (pm 1.169e-3 /
   ps 1.200e-3 / pA F-norm 1.531e-3 — the exact W2G class); a3 standalone
   1.184 -> 1.049-1.064 ms. **Gate: still deterministic 0/60 zeros at
   35,200B** (probe 55.96 ms — the broken kernel runs "fast").
2. **Clean size A/B — hypothesis killed.** Built `spk_g4qh2pad.cu`: the
   WORKING scalar qh3p kernel with SM_BYTES padded 32,768 -> 35,840B (same
   carveout cfg-17 class; ops_nv QMD: _smem_cfg_kb picks 64KB for any usage
   >32KB; note the dext adds a 1KB driver reserve: ELF smem_size = ptxas
   static + 1024). In-graph gate with the padded scalar: **60/60 EXACT,
   38.96 tok/s** — >32KB static smem is fine in-graph. Size was never the
   issue (also: k2s3 has ZERO smem; W2G's "~36.4KB proven class" claim was
   wrong — the only proven class was 32KB).
3. **One-cycle dump (dbg_hm2.py).** After ONE in-graph cycle (run_cycles(1),
   faithful draft->probe->accept->flush): **pm3/ps3/pA3 are 100% POISON
   (7.7e31) — the HMMA kernel never wrote ANYTHING in-graph.** The gate's
   "zeros" outputs are argmax over poison-NaN downstream. (qw16_3 = all-Inf
   in the debug flow — an artifact of skipping fill_draft; secondary.)
4. **QMD diff (dbg_hm3.py).** 181 QMD fields hm-vs-padded-scalar: only 5
   diffs (program/constbuf addrs, prefetch size, SHARED_MEMORY_SIZE
   36,224 vs 36,864). Nothing structural; carveout fields identical.
5. **Direct launch at the CORRECT local size.** hm kernel direct at
   (1024,1,1) on engine buffers: writes pm3/ps3/pA3 100% (sane values).
   Kernel + QMD + args + direct path ALL fine.
6. **Read the graph builder (gcycle.py ParityGraph)**:
   `ls = (1024,1,1) if "nw32" in nm else (768,1,1) if "nw24" ... else
   (256,1,1)` — **`spk_g4nwhm3_100k` contains "nwhm3", not "nw32" -> 256
   threads.** With 8 of 32 warps, row-owners warp<RP cover rows 0-7 only and
   the launch is degenerate (observed: zero writes — the 256-thread launch
   of this kernel writes nothing at all; confirmed independently by an
   accidental direct launch at LS=(256,1,1) which also left pm3 all-poison).
   W2G's in-graph "speed" (probe 53.84-56.22 with the broken kernel) was the
   no-op kernel's speed, not the HMMA win.
7. **Fix + gate**: rename to spk_g4nw32hm3_100k (build_hm.py / test_hm.py /
   mtp.py k1n selection). Gate: PASS (numbers above).

## Final numbers (w100k_hm4.log)
- [tier1] 60/60 exact, first divergence None (fresh T=1 ref, DO_T1=1)
- [tier2] overlap vs fp16-KV(W2D): 60/60; vs int8-KV(W2E): 60/60 — ZERO flips
- [time] rep0 68.98 / rep1 70.02 / rep2 69.37 ms/cyc -> BEST 68.98 -> 40.35
  tok/s; all reps agree 60/60 (determinism x4: 1 tier-1 run + 3 speed reps)
- [phase] draft 5.24, probe 62.53, accept 1.14 ms/cyc (probe -2.3 vs scalar
  64.76 — matches the standalone -0.119ms x16 launches prediction)
- [stock] T=1 45.92 ms/tok (21.78 tok/s); engine T=1 vs spec_base_100k: 59/59

## Canonical recipe (new)
`SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1`
(HM=1 now DEFAULT-ON-worthy; env still gated, flip when convenient.)

## Gotchas banked (LAW-grade)
- **THE NAME-ENCODED LAUNCH CONFIG TRAP**: ParityGraph (gcycle.py) picks
  local_size by NAME SUBSTRING ("nw32"/"nw24"/"nw16"/else-256). Any new
  engine kernel MUST carry its warp-count token in the cubin name
  (spk_g4NW32hm3...) or it silently launches 256-thread. Direct
  NVProgram.__call__ ignores the name (threads passed explicitly) — so
  standalone validation CANNOT catch it; only the in-graph gate can.
- The dext/driver adds a **1KB driver reserve** to static smem: ELF
  info.cs.smem_size = ptxas smem + 1024 (spk_g4hm 35,200 -> 36,224). The
  ops_nv carveout picks 32/64/100KB on the INCLUSIVE size; >32KB (cfg 17)
  works in-graph fine (proven by the padded-scalar control at 36,864B).
- W2G's "dext in-graph ~36.4KB class (k2s3)" was a mirage: k2s3 has no smem.
- A 256-thread launch of spk_g4hm writes NOTHING (not even partial rows) —
  pm3 stays poison; don't confuse "no writes" with "launch dropped".
- In-flight LAW bit again: eager-replaying a 484k-launch graph seq needs
  dev.synchronize() every <=256 launches (I faulted the device once with
  unlimited async; also p(*a, local_size=LS) in debug scripts must match
  the kernel's NW — the 2k-era LS=(256,1,1) default is wrong for nw32).
- Submitting probe_g without the full draft->probe->accept->flush chain
  hangs the timeline (2 signals short) — always use run_cycles in debug.
- mtp.py now has MTP_K1N_A3 env override (force the a3 K1 cubin; used for
  the padded-control A/B; default off).

## Files
- engine0/spk_g4hm.cu (diet + postmortem note) + spk_g4nw32hm{3,1}_100k.cubin
- engine0/build_hm.py, test_hm.py, mtp.py (renames + MTP_K1N_A3)
- engine0/spk_g4qh2pad.cu + spk_g4nw32qh3ppad_100k.cubin (size-hypothesis
  control — KEPT as refutation evidence)
- engine0/dbg_hm2.py (one-cycle poison dump), dbg_hm3.py (QMD diff +
  direct-1024 proof)
- Logs: ~/w100k_hm3.log (diet-still-zeros), ~/w100k_pad.log (padded-scalar
  60/60), ~/w100k_hm4.log (FINAL PASS 40.35), ~/dbg_hm2.log, ~/dbg_hm3.log

## Remaining to more (not this session)
- a1 HMMA stays unwired (scalar a1 is faster: +0.094ms HMMA).
- Probe 62.53ms attribution: attention ~21 (16x spk_a3 @ ~1.06 + K2), rest
  = GDN/FFN/head GEMV classes per W2G. Next levers unchanged (W2G list).
