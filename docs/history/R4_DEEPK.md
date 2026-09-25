# R4 — THE DEEP-K LOOKUP DECODE PROGRAM: machinery built + discriminated; 46.2 tok/s SELECT-mode measured (Tier-1 53/60 — exactness bug localized to the deep row-3/4/state path); the full map for the finish

Status: items 1/2/3 of the mission BUILT (per-cycle graph-set selection live in
DecodeSession; the complete M=5/T=5 kernel set + lookup5 + acceptk/accept5k +
ROWS=5 attention; all gates instrumented). The deep machinery RUNS and delivers
**46.0-46.6 tok/s blended (79.0-80.0 ms/cyc, 3.68 tok/cyc), deterministic x2,
stock 59/59** — but **Tier-1 is 53/60 (NOT shippable as exact)**: a
deterministic divergence class in the deep path, rigorously localized (below)
but not yet fixed at budget end. The K2-side of the new machinery (lookup5 +
acceptk + widened scan) is **60/60 EXACT at 40.2-40.9 tok/s** — a superset of
the R3 config that is bit-safe to ship TODAY. The offline deep-K tables (the
60-statement arithmetic) are complete and decisive.

## 1) THE OFFLINE DEEP-K TABLES (lut_deepk.py — the design driver)

Exact engine contract, LMIN=8 full-window match, most-recent tie-break.

| workload | K | hit-rate | E[m|hit] | m-dist shape |
|---|---|---|---|---|
| (a) 100k base cont. | 2 | 72.4% | 2.000 | all-or-nothing |
| (a) | 3 | 71.9% | 3.000 | all-or-nothing (needs cur as 3rd prop) |
| (a) | **4** | **67.9%** | **4.000** | all-or-nothing |
| (a) | 5 | 67.3% | 5.000 | (needs cur) |
| (a) | 6 | 63.0% | 6.000 | |
| (c-sim) quote ceiling | 2..6 | 87.9% | = K exactly | all-or-nothing |

**THE DEPTH LAW: the lookup's acceptance does NOT collapse with depth —
E[m|hit] = K EXACTLY at every measured depth (alpha = 1.0), unlike the MTP
draft (0.867 -> 0.608 at K=3).** The 8-gram contract proposes verified-true
continuations on both locally-periodic and verbatim-quote text.

**THE DEEP-K SCAN-RANGE LAW (the key offline find)**: on locally-periodic text
the most-recent full-window match sits at the scan EDGE (best-i dist = exactly
10 back = i = pos-11 for EVERY base-workload hit); deeper proposals need
matches FARTHER BACK, so the scan range must WIDEN with K:
`iend = pos - 8 - K` guarantees all K proposal tokens hist[i+8..i+8+K-1] are
WRITTEN ids (<= hist[pos-1] = cur — a real verified token; hist[>=pos] is
unwritten garbage -> embedding OOB). lookup5_nw32.cu implements this for K=4
(iend = pos-12). With a fixed K=2-range scan, deeper K finds ZERO hits on the
base workload (measured) — the naive extension is worthless without this law.

## 2) WHAT WAS BUILT (all zero-spill, cuobjdump-audited)

- **m5.cu** (gen_m5.py: mechanical M-extension of m4.cu with loud asserts):
  14 T=5 kernels — h_embed5/k0n5/k0ab5/q5g8v5/k2s5/op38nw32_5/k3aonw32_5/
  ao8nw32_5/hh5/ffn8v5/down8nw32_5/aq3k8v5/aq6k8v5/head8v5. amx3 reused
  (grid=RM=5, row-generic). Per-kernel cubins via build_k4.py.
- **ROWS=5 attention**: spk_pre5_{2k,100k}, spk_g4nw32a5_{2k,100k},
  spk_c5{,g_100k} (skv_split ROWS-generic) + **spk_pre5qh_100k** +
  **spk_g4nw32hm5_100k** (the canonical QH/HMMA path; spk_preqh.cu's ROWS==3
  ternaries patched to the (ROWS==1 ? 0 : t*stride) generic form —
  bit-identical for ROWS 1/3, legal 5; spk_g4hm.cu already row-generic,
  per-row cap `ka <= pos + t_` verified).
- **lookup5_nw32.cu**: 4-proposal lookup (dring0..3), deep scan range,
  l_hist instrumentation unchanged (9 = hit).
- **acceptk.cu / accept5k.cu / acceptsel5k.cu**: the 16-word emit record
  {pos_new, m, tok0..4, stop, cyc, hit, rsv} — emit[9] = l_hist[cyc] is the
  selection flag; accept5k = m-ladder to 4 + full M1-A emit machinery;
  acceptsel5k = deep state select (see the race fix).
- **mtp.py**: LOOKUP_K env (4 = deep K=4; asserts the canonical W2H env);
  RM=5 shared scratch (both sets; cycles timeline-serialized); dual graph
  sets `graphs` (K2) + `graphs5` (deep: shared draft_g/flush_g, probe5_g,
  accept5_g); DecodeSession per-cycle selection (`deep = 1 if hit >= 9`) —
  the readout-order law respected (decision reads the PREVIOUS cycle's emit);
  DEEP_MODE diag knob (sel/off/on); dring3 slot (init 0 — NEVER poison).
- **test_w100k.py**: R4LOG + [deepk] instrumentation; R4_DIAG discriminator;
  R4_ROWCHK probe-vs-probe5 row comparison; tok_hist seeding for LOOKUP_K.

## 3) THE GATES (~/r4_deepk_100k.log, ~/r4_deepk_diag.log, ~/r4_deepk_fix.log, ~/r4_rowchk2.log)

| config | Tier-1 | ms/cyc | tok/s | notes |
|---|---|---|---|---|
| LOOKUP_K=4 DEEP_MODE=off (K2 probe + lookup5 + acceptk) | **60/60 x2** | 68.0-69.2 | **40.2-40.9** | EXACT — shippable superset of R3 |
| LOOKUP_K=4 DEEP_MODE=on (all-deep) | 21/60 | 84.1-85.1 | 42.6 | deterministic; first div at 21 |
| LOOKUP_K=4 DEEP_MODE=sel (production selection) | 53/60 x2 | 79.0-80.0 | **46.0-46.6** | deterministic; div at 24; stock 59/59; emit==hist |
| rowchk: probe vs probe5 on identical state | rows 0..2 BIT-IDENTICAL (amds + xA maxabs 0.0) | | | the M=5 GEMVs are exact |
| [deepk] sel-mode stats | deep-selected 63.3%, E[m|deep] 3.605, m-dist(deep) [0,8,2,2,64] | | | 64/76 deep cycles accept ALL FOUR |
| [lookup] | 65.0% hits, E[m|hit] 3.462, E[m|miss] 1.238 | | | in-vivo alpha stays ~1 on hits |

Phase (K2 path): draft 5.3 / probe 62.5-62.9 / accept 1.16. Deep cycle
increment measured: +15.5-15.9 ms over K2 (T=3 -> T=5 probe), i.e. ~8 ms/row
in the W2H class — better than the W2D-era 11.1.

## 4) THE EXACTNESS BUG — RIGOROUSLY LOCALIZED (next-session search space)

Facts (all measured, deterministic across reps AND across the k2s5 race-fix
rebuild — the identical divergent stream byte-for-byte):
1. deep=off (K2 probe + ALL new K2-side pieces incl. lookup5's widened scan
   choosing different occurrences, acceptk, RM=5 scratch): **60/60** — the
   K2-side is exact.
2. probe vs probe5 on identical state: **rows 0..2 bit-identical** (amds AND
   the pre-head hidden xA) — every M=5 GEMV/norm/embed/head/amx kernel is
   per-row exact; the shared draft/scratch/emit plumbing is exact.
3. deep=on diverges from emitted position 21 (21/60); sel-mode from 24
   (53/60) — divergences are PATTERN-BREAKING tokens (4471/43614 where the
   T=1 stream says 6545/9956) in the period-3 repeat region, then a
   phase-shift cascade.
4. The k2s5 conv slot-4 RACE (t=4's cross-head strided conv write vs other
   CTAs' t=0..2 live reads — k2s4 never wrote slot 4) was found and FIXED
   (conv5x scratch + acceptsel5k m==4 copy; rec writes are head-local =
   race-free) — the fix is correct-by-construction and KEPT, but the
   divergence is UNCHANGED by it (so the observable bug is not that race).

Remaining suspects (ordered):
- (a) The row-3/4 argmax path on REAL prefixes: k2s5's t=3/t=4 iteration
  (z/core/rec chain), the attention's rows 3/4 (pre5qh append + hm5 K1 row
  cap + c5g combine), or head8v5/amx3 rows 3/4 — something row-3/4-specific
  that the identical-state rowchk cannot see (its seeded prefix is garbage,
  so rows 3/4 had no reference there).
- (b) The post-deep-accept STATE: live slot 4 after m=4 (k2s5 t=4 rec write
  / conv5x -> acceptsel5k copy) or after m<4 — a subtly wrong live state
  drifts every later token (matches the cascade + the all-deep-worse
  gradient). DISCRIMINATOR (cheap, next session): after one deep cycle with
  m=4, run K2 cycles and compare the stream vs a pure-K2 run advanced by the
  same accepted tokens (state-continuity check, no kernel changes needed);
  and/or an m<=2-capped accept5k build (emits at most 3 tokens/cycle — if
  the stream goes 60/60, rows 3/4 argmax are the culprit; if not, state).
- (c) The near-tie reassociation class (W2D law: batch-kernel reassociation
  flips the degenerate repeat region): if (a)/(b) are excluded, the T=5
  rows' association differs somewhere subtle — bisect by substituting the
  ROWS=5 attention set with re-validated builds one kernel at a time.

## 5) THE 60 STATEMENT (honest, per the mission)

- **General decode, exact, TODAY: 40.2-40.9 tok/s** (deep=off config — the
  R3-class engine with the widened-scan lookup; 60/60 x2 deterministic).
- **Deep-K selected mode MEASURES 46.0-46.6 tok/s** (deterministic, stock
  59/59) but is NOT Tier-1-exact yet — the number stands as the performance
  ceiling proof of the machinery, not a shippable config.
- **The 60-cross arithmetic (offline, K=4 measured costs)**: blended
  = hits*(1+4) + misses*(1+0.7) tok per cycle at hits*84 + (1-hits)*69 ms.
  At the quote-heavy 87.9% hit rate: 4.61 tok/cyc @ 86 ms = **53 tok/s**;
  at the base 67.9%: 3.94 @ 74 ms = 47.5. **60 needs K=5-6 on quote-heavy**
  (offline E[m|hit]=K to depth 6: 87.9% hits -> 6.35 tok/cyc @ ~100 ms =
  62-63 tok/s) — K=5 requires the [48][6] rec4/conv4 slot restructure (+
  acceptsel/serve.py offset surgery: the live slot moves 4 -> 5, serve.py's
  (j*5+4) snapshot offsets included) — priced as a follow-up campaign, NOT
  built this session (K=4 kept the [48][5] layout deliberately).
- vs the analysts' 25-35%: decode 46.6/86 native-stack = 54% at the
  performance-proof config; 40.9/86 = 47.5% exact-shippable.

## 6) Ship status

- **Daemon: NOT flipped** (machine cold-cycled after the rowchk fault class;
  relaunch on the R2 canonical per the M1 run line — or with
  `LOOKUP=1 LOOKUP_K=4 DEEP_MODE=off` for the exact 40.2-40.9 superset:
  the serve flip procedure is R3's (tok_hist seeding in FRESH/FOLLOW_UP/
  CACHE_HIT + LOOKUP env) UNCHANGED by R4 — decode_n/step() dict keys are
  backward-compatible; api_gates/soak NOT rerun this session).
- Commit: this file + all kernels/harnesses (green-gate commits per law).

## 7) Gotchas banked (LAW-grade)

- **THE DEEP-K SCAN-RANGE LAW** (section 1) — the lookup's scan range must
  widen with K or deeper K finds nothing on periodic text.
- **THE CONV SLOT-4 CROSS-CTA RACE**: any in-kernel write to conv slot 4
  (live) races other CTAs' live reads (the conv write loop is cross-head
  strided; only REC is head-local). The conv5x+acceptsel5k pattern is the
  legal route for t=T-1 state when T = the slot count.
- **THE PROBE-ONLY HARNESS FAULT**: submitting probe_g without the draft
  graph reads dring = -1 poison -> embedding OOB -> device fault. Seed
  valid dring ids first (rowchk law).
- emit widened to 16 words: legacy accept (LOOKUP_K=0) still writes the
  8-word prefix; DecodeSession reads layout-aware (stop/cyc indices differ).
- dring3 must be initialized to a VALID token id (0) everywhere (boot,
  _reset_slots, restore_mtp) — never poison.
- The row-4 argmax on a seeded garbage prefix is meaningless for exactness
  (2002-class outputs are the model answering a different input sequence).

## Files
- engine0/{m5.cu,gen_m5.py,build_k4.py,build_k4qh.py,fix_k2s5.py} + 14 M=5
  cubins + lookup5_nw32/acceptk/accept5k/acceptsel5k + spk_*5* cubins
- engine0/lut_deepk.py (the offline deep-K analyzer — THE design driver)
- mtp.py (LOOKUP_K wiring), test_w100k.py (R4 diag/instrumentation)
- Logs: ~/r4_deepk_{100k,diag,fix}.log, ~/r4_rowchk2.log
