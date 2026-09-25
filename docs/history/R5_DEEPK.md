# R5 — DEEP-K LOOKUP EXACTNESS FIXED + THE K LADDER TO K=6: 58.71 tok/s TIER-1-EXACT @100K (the 60-cross arithmetic)

Status: the R4 exactness bug (sel-mode 53/60) ROOT-CAUSED and fixed; the deep-K
ladder extended K=4 -> K=5 -> K=6, every rung Tier-1-gated 60/60 x2
deterministic + stock 59/59. **K=6 sel-mode: 58.71 tok/s (101.91 ms/cyc, 5.98
tok/cyc)** on the canonical gate prompt; the quote-heavy class measured HONESTLY
in-vivo for the first time (it does NOT materialize on a random span — the
offline 87.9% was the model-quotes-perfectly CEILING). General-exact fallback
(deep=off) unchanged at ~40.2 tok/s.

## 1) THE R4 BUG — ROOT CAUSE (three stacked gen_m5 M-extension slips)

Method chain (all harnesses env-gated in test_w100k.py, reusable):
- **R4_TRACE** (per-cycle row discriminator): downloads amds[0..K] every cycle
  and compares EVERY emitted row vs the T=1 ref at (pos-P0)+t. VERDICT: rows
  0-3 exact, **row 4 (the BONUS) wrong from the FIRST deep cycle** — the state
  entering deep cycles was perfect; rowchk's garbage-prefix caveat explained why
  R4 couldn't see it.
- **R4_DIF** (isolated single-probe differential): T=5 probe at P0 with
  TRUE-greedy drings vs the bit-proven T=3 probe at P0+2 (GDN re-anchored via
  eager acceptsel5k m=1). Rows 2/3 BIT-IDENTICAL; row 4 frac-neq 1.0, xA maxabs
  60.6 — wholesale corruption, not a tie flip. Cleared: proposal fill (host-seeded
  drings), emit path, KV cache at pos+4 (the T=3 run read those rows exactly).
- **R4_BISECT** (per-kernel partial-graph bisect, T5@P0 row4 vs T5@P0+2 row2,
  same kernel set): block-level binary search + within-block linear with
  FULL-buffer per-row compares. NOTE the harness traps: (a) partial probes
  POISON the GDN live (k2s t=4 rec write lands IN live slot 4) — re-anchor
  before EVERY partial; (b) the anchor must be cached AFTER reset_spec (the
  post-boot slot-4 is fill_draft state, not the snapshot — cost me a detour);
  (c) compare FULL rows (my 2048-element cap gave a false "k2s5 clean").

The three bugs (all row-4-only, all from gen_m5's mechanical M-extension):
1. **aq3k8v5 q-section macro missing the `(OB)[4*(OS)]` row-4 STORE** — qrow3
   row 4 = stale garbage -> wrong RoPE'd Q row 4 -> wrong amds[4] (THE 53/60
   bug; the bisect's first diverging buffer, block 11 attn kernel 2, qrow3-only
   while krow/vrow were fine).
2. **k3aonw32_5 missing the `a4 +=` accumulate line** (attn_out row 4 = 0).
3. **Four nw32-named kernels (op38/k3ao/ao8/down8 _5) generated at
   __launch_bounds__(256) vs their 1024-thread name-law launches** — silent
   overcommit on the dext. (Fixed first; alone it changed nothing bit-for-bit —
   the missing-store bugs dominated. Still a real bug class.)
gen_m6.py/gen_m7.py bake the fixes in: bounds PRESERVED through renames, and a
fail-loud audit that EVERY row-write family gains the new top-row store.

## 2) THE K LADDER (all Tier-1 60/60 x2 det + stock 59/59 @100k)

| config | ms/cyc | tok/cyc | tok/s | E[m|deep] | m-dist(deep) | hits |
|---|---|---|---|---|---|---|
| deep=off (K2 superset) | 69.2-69.5 | ~2.8 | 40.05-40.24 | — | — | — |
| K=4 sel (LOOKUP_K=4) | 82.32 | 4.42 | **53.65** | 4.000 | [0,0,0,0,98] all-4 | 83.3% |
| K=5 sel (LOOKUP_K=5) | 91.58 | 5.18 | **56.60** | 5.000 | [0,0,0,0,0,96] all-5 | 81.7% |
| K=6 sel (LOOKUP_K=6) | 101.91 | 5.98 | **58.71** | 6.000 | [...,0,96] ALL-SIX | 81.7% |
| K=4 all-deep (diag) | 84.9 | — | 52.39 | | | |
| K=5 all-deep (diag) | 97.1 | — | 53.91 | | | |
| K=6 all-deep (diag) | 109.4 | — | 55.31 | | | |

With the row-4 bug fixed, **every deep hit accepts ALL K proposals**
(E[m|deep]=K EXACTLY, alpha=1.0 in-vivo at every depth — the offline depth law
now holds end-to-end). Deep-cycle cost ~ +12-15ms/K-rung (T=7 probe ~109ms
all-deep). NOTE: sel-mode beats all-deep at every rung — selection pays.

## 3) K=5/K=6 CONSTRUCTION LAWS (new, LAW-grade)

- **THE RP>NW OWNER BUG (spk_g4hm.cu)**: row owners were `if (warp < RP)` —
  at ROWS=6 (RP=48 > NW=32 warps) rows 32-35 (heads h%6==5, t>=2) had NO owner:
  pm/ps unwritten -> combine 0/0 = NaN (~512/4096 NaNs per row = exactly 2
  heads — the R5_NAN fingerprint). FIX: MAXOWN-generic multi-row owners
  (ms_r[]/ss_r[] arrays, `r = warp + oi*NW`), codegen-identical for ROWS<=5.
  ANY future ROWS with RP>NW needs this.
- **THE REC-CHAIN SLOT LAW (k2s7)**: with [48][5] slots and t>K-3 writing
  scratch (rec6x/rec7x), the NEXT t's rec_in must read THE SCRATCH, not
  `(t-1)` — slot 5 does not exist; the naive index read the NEXT BLOCK's
  slot 0 (NaN row 6, deterministic). k2s7: t=6 rec_in -> rec6x.
- **Register budget**: every extra accumulator costs regs; the nw32 1024-thread
  family needs <=64 regs. unroll reduction on the b-loop (5 -> 4/2) is the
  zero-spill knob and does NOT change per-row fp op order (bit-compat kept).
  hm6/hm7: ks4-unroll-4 variant; ao8_7/op38_7/down8_7 unroll 4.
- **smem**: hm7 = 38,336B runs fine in-graph (the "~36.4KB class" fear is dead;
  hm6 37,568 and hm7 38,336 both gate green).
- **NO LAYOUT SURGERY NEEDED for K=5/6**: the recNx/convNx per-block scratch
  pattern extends the conv5x race fix — [48][5] + live=slot-4 PRESERVED, zero
  trunk/serve changes. acceptsel6k/7k copy scratch->live on m==K-1/K.
- emit layout: 16-word (K2/K=4 acceptk/accept5k), 17-word (accept6k), 18-word
  (accept7k) — DecodeSession reads layout-aware by LOOKUP_K + which set ran.

## 4) THE QUOTE-HEAVY WORKLOAD — measured honestly (R5_QUOTE harness)

The offline c-sim (lut_deepk.py) class: feed ids + ids[q0:q0+60] (rng(0),
q0=81647) verbatim, assume the model continues verbatim. IN VIVO (R5_QUOTE=1:
follow_up feeds [cur]+quote via the proven T=1 path, tok_hist seeded, T=1 ref
vs spec decode x2): **the model does NOT verbatim-continue a random doc span —
1.7% lookup hits, tok/cyc 1.07**; the spec stream is bit-exact vs T=1
(deterministic x2; window off-by-one in the first harness print — out==ref[1:]
aligned). CONCLUSION: the offline 87.9%-hit quote table is a CEILING bound
(the "model-quotes-perfectly" assumption), NOT an in-vivo workload on this
model/doc. The 60-cross arithmetic therefore needs either (a) a prompt that
actually elicits verbatim reproduction (prompt-engineering, not engine work),
or (b) more K / cheaper deep cycles on the real 81.7%-hit class.

## 5) THE 60 STATEMENT (honest)

- **General gate prompt (the repeat-region class, REAL): 58.71 tok/s Tier-1
  exact** at K=6 — 68% of the 86 vast native-stack decode reference, at 100k
  ctx, through the dext.
- 60 needs: +1.3 tok/s = e.g. K=7 (+~13ms/rung, +0.8 tok/cyc -> ~60.5 if the
  all-accept pattern holds), or the deep-cycle cost down (probe 62.8ms of the
  101.9; attention retune at deeper T, or the K2-cycle 69ms floor shaven), or
  a workload where hits exceed 81.7% (a real quote-eliciting prompt).
- deep=off exact fallback for serving: 40.05-40.24 tok/s (the R3-class engine
  + widened-scan lookup; superset, bit-safe).
- LOOKUP_TRIG=2 (HyperQwen 2-consecutive-hits trigger) implemented
  (LOOKUP_TRIG env); NSTRONG/_AGREE are ARCHITECTURALLY SATISFIED already —
  our LMIN=8 fires ONLY on full 8-gram matches = the "strong match taken
  alone" class; the 6-7-length+agree tier never fires.

## 6) Harness inventory (env-gated in test_w100k.py)

R4_TRACE (row-level per-cycle discriminator), R4_DIF (isolated single-probe
differential + re-anchor), R4_BISECT (per-kernel partial-graph bisect),
R5_NAN (block/kernel NaN scan — mind the anchor+full-row laws in section 1),
R5_DET (determinism probe), R5_QUOTE (in-vivo quote workload with T=1 ref).
Logs: ~/r5_*.log (trace, dif*, bisect*, nan*, det*, gate_k5, gate_k6, quote*,
trig2).

## 7) Ship / daemon

- The daemon was STOPPED for this session (graceful shutdown via the
  /tmp/llm-engine.sock "shutdown" method). Relaunch per the captured env
  (M1C line, NO PF_* knobs, LOOKUP=0) — or with LOOKUP_K=6 for the 58.71
  config (serve.py emit-layout aware via DecodeSession; snapshot offsets
  unchanged).
- Commits: 75fc008 (R5a fix), 95b492e (R5b K=5), 8dc56d9 (R5c K=6), + this doc.

## 8) R5d — K=7 SHIPPED: ***63.11 tok/s TIER-1-EXACT @100K — THE 60 GOAL CROSSED***

The K-ladder extended to K=7 (M=8/T=8 set) with the now-mechanical per-rung
pattern; every gate green on the first build. **K=7 sel-mode: 63.11 tok/s
(107.49 ms/cyc, 6.78 tok/cyc, alpha 2.892)** — the 60-cross, +4.4 over K=6,
73% of the 86 vast native-stack decode reference.

| config | ms/cyc | tok/cyc | tok/s | E[m|deep] | m-dist(deep) | hits |
|---|---|---|---|---|---|---|
| K=6 sel (prev best) | 101.91 | 5.98 | 58.71 | 6.000 | [...,0,96] all-6 | 81.7% |
| **K=7 sel (LOOKUP_K=7)** | **107.49** | **6.78** | **63.11** | **7.000** | [...,0,0,96] ALL-SEVEN | 81.7% |
| K=7 all-deep (diag) | 116.32 | — | 59.03 | | | |
| K=7 deep=off (superset) | 69.21 | ~2.8 | 40.22 | — | — | — |

- Gates: DIF all-8-rows exact (amds[0..7]==ref, rows 2-4 BIT-IDENTICAL vs the
  re-anchored T=3 probe; row-7 healthy — the rec-chain law held first-try);
  Tier-1 60/60 x2 deterministic + stock 59/59; deep=off 60/60 (40.22 — the
  exact superset fallback intact).
- Offline depth-7 pre-check (lut_deepk K=2..7): E[m|hit]=7.000 EXACTLY at
  depth 7 (33/33 all-7), hit rate 62.3% base (-0.7pt vs K=6) — the depth law
  holds to depth 7; in-vivo 81.7% hits with 96/96 ALL-SEVEN accepts.
- Cycle split (K2-phase measure): draft 5.31 / probe 62.65 / accept 1.16 ms.
  The K=6->K=7 deep-cycle increment was only +5.6 ms blended (+6.9 all-deep) —
  BETTER than the +12-15 ms/K-rung trend (hm8 has RMAX=48=RP: zero padding
  rows vs hm7's 42->48 pad; 39,104B smem in-graph OK).
- Construction: gen_m8.py (RED8/ACC8H2/IQ3V8 + row-7 stores audited on every
  write family; bounds preserved); k2s8 (t=7 rec->rec8x, conv->conv8x,
  t=7 rec_in reads rec7x per the REC-CHAIN SLOT LAW; [48][5] live=slot-4
  PRESERVED — zero trunk surgery); lookup8 (iend=pos-15, dring6);
  accept8k (m-ladder to 7, 19-word emit) + acceptsel8k (rec8x/conv8x select);
  spk_pre8qh/spk_g4nw32hm8 (RMAX=48, RP=48, MAXOWN=2, ks4-unroll-4 — 64 regs
  0 spill, same as hm7)/spk_c8g. All 20 cubins cuobjdump-audited 0-spill;
  the nw32 family held at exactly 64 regs.
- **DAEMON FLIPPED to LOOKUP_K=7** (serve.py): the R3-documented tok_hist
  seeding LANDED (seed_hist at boot-park/FRESH x2/FOLLOW_UP/CACHE_HIT/
  snapshot/snapshot_load) — WITHOUT it the first FRESH generate self-matches
  the -1 prefix and writes -1 drings -> embedding OOB fault (the R4
  probe-poison law; serve.py had NEVER seeded — the flip was never exercised
  pre-R5d). Seeding is LK_ACTIVE-gated (LOOKUP=0 daemons byte-identical).
- Files: engine0/{gen_m8.py,build_k8.py,m8.cu,+14 *_8.cu/cubin,lookup8_nw32,
  accept8k,acceptsel8k,spk_pre8qh_100k,spk_g4nw32hm8_100k,spk_c8g_100k},
  mtp.py (LOOKUP_K=7: RM=8, M8_CUBINS, graphs6=_probe8_seq, 19-word emit
  parse, dring6), test_w100k.py (DIF NR=8/dring6/acceptsel8k), serve.py
  (seed_hist). Logs: ~/r5d_dif_k7.log, ~/r5d_gate_k7.log.
- Next margin (unlocked by the 60-cross): K=8+ arithmetic is the same law
  (offline E[m|hit]=K to 7; diminishing hits ~-0.7pt/K), the deep probe now
  dominates (62.65 of 107.49) — attention retune at deeper T or the K2-cycle
  69ms floor are the priced levers.
