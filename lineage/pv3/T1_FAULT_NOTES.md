T=1 draft-PV fault notes (2026-09-10, val_t1.py runs)

SYMPTOM: k1t1 completes clean (its wait=True passes); the fault surfaces at
k2t1's wait. Happened BOTH before and after the k1t1 staging fix (4B->16B),
so staging was never the fault cause.

k2t1 is trivial (24 CTAs x 256 thr; strided ws reads; sigmoid; 3 writes) and
all index bounds check out by hand (ws max index 147455 < 147456; gate max
12287 < 12288; out max 6143 < 6144). Next-session isolation steps (in order):
1. Run k2t1 ALONE (skip k1; pre-fill ws with a Tensor.zeros buffer upload) —
   if it still faults, the kernel/cubin itself is bad: try (a) removing the
   unused `base` variable (compiler warning seen), (b) re-check the ELF loads
   with a different name, (c) rebuild with -G then -O3.
2. If k2t1 alone is clean, the k1->k2 interaction is the issue: check whether
   k1's ws writes race the fault (dump ws via numpy after k1 only).
3. Compare against the WORKING T=3 k2v2 (same style, no int param, vals=(0,))
   — the only structural deltas: k2t1 reads ws with a per-s stride of
   4*6*256=6144 floats and has no `#pragma unroll`; try adding the unroll and
   an explicit local reduction buffer.

Machine rebooted twice for this; canonical 6.87 untouched (this is standalone
harness work only).

## 2026-09-10 STATUS after OOB fix (4467856)
- NO MORE FAULT (OOB was the fault). val_t1: stock=2.841ms, K1=0.494, K2=0.051
  -> pair 5.2x faster; ~4.6ms/cycle prize if correctness lands.
- OPEN BUG: relerr=1.24 (e.g. idx 1089 = h=4 d=65: ref 0.140 got -0.034 —
  sign+magnitude mix => likely wrong-head or wrong-slot mapping, not rounding).
- Verified-by-hand consistent between stock and k1t1/k2t1: P stride PS=100349
  (head-major, heads 2g/2g+1 per stock CTA), V [(h/6)*L*256 + L*1024 + p*256 + d],
  gate [d + h*512 + 256], out [d + h*256], loop bound p < sp+1.
- REMAINING SUSPECTS: (1) the stock P "second stream" offset +100349 might be
  (head+1)*PS only for ODD/even gidx1 — recheck by evaluating the stock kernel
  in numpy on the harness data (exact per-head compare); (2) k1t1 compute reads
  sV[pi] while staging fills rows by SPR=tid>>5 — verify pi<->row identity when
  npos<8 (last tile); (3) dump k1 ws vs numpy per (s,g,h6) to localize the
  wrong-slot head. NEXT: numpy-evaluate stock on harness inputs, diff against
  k1t1 ws partials per head — pinpoints in one run.

## pinpoint round 2 (after acc[6][8] fix — relerr 1.24 -> 0.964, K1 0.40ms)
- ws wrong for ALL 24 heads; per-split shows EVERY split wrong with
  uncorrelated same-scale values (not zero/shifted/scaled) => element-level
  index mismatch (P position mapping or V row identity), NOT partial coverage.
- Everything hand-verified again: chunk ranges, sp bound, staging row<->pos,
  lane+32j vs sdc coverage, head/group/gate/out maps.
- NEXT PROBE (decisive, ~10 min): set S=LMAX+1... simplest: compile k1t1 with
  -DS=1 (single split, C=LMAX) and compare ws[0] against numpy over the full
  range; then hand-check ONE position by zeroing P except p=k for a few k
  (P one-hot) -> ws must equal V[k] rows exactly; whichever one-hot k fails
  reveals the true position map (suspect: initial-prefetch rows are tile-1
  data stored as tile-0 — swap test by disabling the in-loop prefetch).

## ONE-HOT PROBE RESULT (decisive clue)
- P one-hot at head 7 p 5023 -> ws EXACTLY ZERO everywhere. The kernel never reads that point — yet random-P shows all heads non-zero => the true P index map covers a DIFFERENT set than designed.
- NEXT DIAGNOSTIC: one-hot at corner points (0,5), (1,5000), (0,100000), (23,99900) — the slot that lights up for each reveals the actual P(head,p) formula; invert + one-line fix. Suspects: PS constant vs runtime numel; one-hot upload readback; sp val interpretation.

## CORNER-POINT RESULTS (k1t1)
- Slot mapping CORRECT: (h0,p5)->(0,0,0) and (h1,p5000)->(1,0,1) lit the expected slots; p=100000/99900 misses were CORRECT sp-skips (>SP).
- The bug = STAGING CHAIN: lit rows match NO V row (mixed positions); (h0,p5) row only 96/256 dims valid = contiguous prefix = 12 threads worth of stores. smem rows hold mixed/stale data.
- NEXT: (1) print the exact dim mask of the (h0,p5) row; (2) compile a no-prefetch variant (store directly from a fresh global load each tile, sync-only) — if correct, the vp register chain has a hazard; if still wrong, store/load index math differs between staging (sdc contiguous) and compute (lane+32j). 30 percent slower variant acceptable if it validates — optimize after.

## NO-PREFETCH A/B RESULT
- No-prefetch k1t1 (direct load-store-sync per tile) STILL FAILS (0.89, was 0.964) — the register chain was NOT the bug; staging is now trivially correct by construction.
- REMAINING DISCRIMINATOR (next, 2 min): one-hot P at (h1,p5000) through the STOCK kernel — if stock out[h1] == V[g0][5000]*sigmoid(gates) exactly, layouts confirmed and my kernels have a subtle residual bug; if NOT, my decode of the stock P/V layout is wrong and the formulas (not the kernels) need the fix. Also re-run t1_corners on the no-prefetch build: does (h0,p5) now equal V[0,5] exactly?

## DISCRIMINATOR + FINAL STATE (b31ecd0+)
- STOCK one-hot: out[h1] == V[g0][5000]*sigmoid(gate[h1]) EXACTLY (relerr 0.000) — MY LAYOUT DECODE IS PERFECT. Bug is 100 percent inside k1t1.
- Corners on no-prefetch build: (h0,p5)->64/256 dims, (h1,p5000)->224/256, still no V-row match — partial-dim pattern SHIFTS with staging edits => staging<->compute mapping entangled.
- IMAGE-SMEM PROBE (the killer diagnostic, ~15 min): k1dbg = k1t1 with compute replaced by w[h6*256+lane+32j] = sV[pi][lane+32j] (h6=0, first tile only, S=1 build) — run with one-hot V... then ws row(s) ARE the smem image: diff against V[0..8) rows reveals exactly which (row,dim) the staging actually wrote. One compile+run, zero ambiguity.
- Perf if fixed: no-prefetch pair 0.666ms vs stock 2.891 = 4.3x (4.4ms/cycle); prefetch variant 0.44ms = 6.6x.

## SMEM-IMAGE PROBE — RESULT AND PROBE-DESIGN CORRECTION
- Image showed g=0 rows == V[1] rows — BUT the dbg kernel HARDCODED group-0 output slots while all 4 group-CTAs wrote them (last-writer = g=1 explains the match). PROBE ARTIFACT: the staging group-base formula was never wrong (and matches the validated T=3 K1).
- ELIMINATION INVENTORY for the k1t1 bug: layouts (stock one-hot 0.000), slot map (corners), group base (this probe), staging chain (no-prefetch A/B), acc dims (acc[6][8] fix changed values). ALL EXONERATED.
- REMAINING PRIME SUSPECT (by elimination): the P-index position mapping in the compute phase — pi vs p relationship (P read uses p=start+t0+pi while sV row pi holds V position start+t0+pi — the ONLY unprobed link). NEXT PROBE: one-hot P sweep at FIXED position across h (e.g. p=4182+k for k=0..7 within one tile) through REAL k1t1 checking WHICH p lights — isolates the P-position term directly.

## P-SWEEP RESULT (final clue set)
- one-hot p=0: EMPTY. p=7: EMPTY. p=3: row 192/256 dims, no full-row V match. Warps 0,7 contribute NOTHING; warp 3 missing exactly dims [192,256) = lanes 24-31 staging footprint.
- NEXT (first command): check p=3 row dims [0,192) against V[0,3][0:192) EXACTLY — if equal, ONLY the staging trip-count/coverage is broken (some warps/lanes not executing stores — suspect compiler reordering or a guard bug); if not, deeper.
- Note the 192 = 24 threads and warp0/7-dead pattern TOGETHER suggest staging executes for a SUBSET of warps — compare the cubin PTX of the store loop; also try removing the (t0+spr<end) guard with a clamped load as a test.

## PARTIAL-ROW CHECK (final characterization)
- p=3 one-hot: present dims = [64,256) EXACTLY (192 dims, j-slots 2..7; j 0,1 MISSING — earlier 192,256 inference was backwards). Warps executed = {1..6} (0 and 7 dead). BOTH iteration spaces (warp, j) shifted up and truncated identically: {1..6} x {2..7} of 8.
- Present values STILL mismatch V[0,3] (relerr 0.84 both spans) => not coverage-only; the surviving compute pairs wrong (position, dim) data.
- NEXT (determined by this pattern): match the [64,256) segment against V[0,k][64:256) over k — the surviving (warp=3, j) block must equal SOME V row segment; combined with the {1..6}x{2..7} executed-set this nearly determines the true (warp,j)->(pos,dim) map; ALSO compile with -O1/-G to test compiler-miscompilation of the unrolled loops (the shifted-truncated pattern in TWO spaces smells like an unroll/codegen bug — the -O1 build is a 5-minute decisive test).

## -O1 TEST: identical failure (0.878) — compiler optimization EXONERATED. The {1..6}x{2..7} executed-set pattern is deterministic and source-level; suspect the interplay of the u4 16B smem stores with the lane+32j reads under the if(pi<npos) guard — rewrite suggestion: replace strided lane+32j compute indexing with the STAGING-contiguous sdc mapping (compute dims [lane*8+j] like staging stores) so both phases use ONE indexing scheme; that eliminates the two-scheme mapping entirely and is a 10-line rewrite.

## UNIFIED-INDEXING REWRITE: identical failure (0.889) — two-scheme mapping EXONERATED. Eight hypotheses now eliminated; anomaly survives every mapping edit.

## BISECT PLAN (the definitive method; 3-4 builds, 30 min)
Build k1t1 with compute replaced stepwise; validate after each:
 A. w[h6*256 + dims] = pv            (images the P reads; one-hot must light (0,g,0) with value 1.0)
 B. A + multiply by sV value: acc = pv * sV[pi][d]  (adds the V term; one-hot row must equal V[0,p])
 C. B + full h6 loop
 D. C + full j loop
 E. D + real staging (not debug staging)
The first step that breaks localizes the term. All prior probes validated OUTSIDE the kernel; this validates INSIDE it.

## BISECT STEP A RESULT (P-image build)
- Pure P image (acc += pv only): p=0 one-hot LITS 32 dims (one per lane, value=P) in slot (0,0,0); p=3 and p=7 one-hots EMPTY.
- CRITICAL: the executed/reading warp set INVERTED vs the earlier build (was: p0 dead p3 lit; now: p0 lit p3 dead) — the anomaly is NOT a fixed warp set; it shifts when the compute body changes.
- IMPLICATION: the P-position read by each warp is unstable across codegen variants — consistent with UNDEFINED BEHAVIOR in the kernel (the classic dext lesson: same-buffer-for-two-restrict-args or an out-of-bounds access causing arbitrary behavior). NEXT: audit k1t1 for UB with fresh eyes: the smem sV is set-but-unused in this build yet still declared; check the u4 16B smem stores (sV alignment/bank), the P pointer arithmetic types, and TRY: remove sV entirely in step-A build (if P-image then reads correctly for ALL warps => the mere presence of the u4 smem stores corrupts sibling reads = UB signature).

## REMOVE-sV TEST: ALL EMPTY — even the ~30-line trivial P-image kernel (no smem, no syncs, just P-read+acc+ws-write) fails. UB confirmed at the most basic level.
FINAL SUSPECT SET (exhaustive by elimination): (1) the ws EPILOGUE indexing; (2) the launch/arg binding for this specific kernel (signature vals pattern works for k1pf/k2v2/k2t1 — compare byte-for-byte the NVProgram setup); (3) the PS-multiply in the P index evaluated with an unexpected type. DEBUG BY SUBSTITUTE: replace k1t1 body with the EXACT body of the WORKING k2t1-style trivial loop (sum over a known buffer) then morph one line at a time toward k1t1 — the working baseline guarantees the harness/launch is fine and each morph localizes.

## RACE FIX + WS-SIZE STATE (final of this stretch)
- Write race FIXED (per-warp slots); harness ws sizing was the recurring fault (patch had silently no-opd on spacing — now S*4*6*8*256 confirmed on line 30).
- val_t1 now: FAULT-FREE, K1=0.534 K2=0.051 (pair 0.585ms, 4.9x) — but relerr 1.25 (signature CHANGED from 0.88 — the race was real but one residual mapping issue remains).
- NEXT: rerun t1_pin (numpy per-head/per-split) with ws reshape (S,4,6,8,256) — the per-warp dimension now visible: check which (s,g,h6,warp) slots hold correct partials vs numpy; the residual is likely a single mapping axis (suspect: the k2t1 slot stride 8*256 vs k1t1 layout, or warp-tag vs pi in the epilogue).

## INTEGRATION PLAN (post-validation 9301ac4)
- The T=1 stock kernel is ONE fused kernel (P arrives pre-normalized from upstream draft QK/softmax kernels) — unlike T=3 there is no natural second slot for K2.
- ROUTE A (recommended): LAST-CTA COMBINE — fold K2 into K1: atomicAdd a device counter per launch; the final CTA (count == S*4-1) runs the combine for all 24 heads and writes out (the classic last-block pattern). Makes the pair ONE kernel = clean 1:1 a3b substitution of the fused-PV kernel with baked-VA ws + counter. ~40 lines.
- ROUTE B: substitute fused-PV -> K1 and find/neuter a following kernel for K2 (fragile).
- After Route A: standalone re-validate (relerr < 1e-4), a3b matchers (5-arg signature: out_6144, P_2408376, KV_205520896, gate_12288, const int sp), 2k gate, 100k gate (run39 + flag). Prize ~4.6ms/cycle.

## BUG-HUNT LEDGER (closed)
1. ws OOB in k2 (g*6+h6 vs slot-local h6) — the faults.
2. 8-way write race in k1 epilogue (per-warp slots fix) — the wrong values.
3. STALE CUBIN after a batch rebuild (nvcc silent fail) — the phantom residual 1.25.
K1 per-warp partials EXACT vs numpy (0.000 on all probes); K2 1.06e-07; full chain 7.82e-06.
