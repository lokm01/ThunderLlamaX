# R8 — THE DECODE-75 CAMPAIGN: K=9 + K=10 SHIPPED (75.56 tok/s Tier-1-exact @100k); the graded-LOOKUP tier sim-killed by the BIMODAL MATCH LAW

Status: RUNG A complete — the K-ladder extended K=8 -> K=9 -> K=10, every rung
first-build-green through the full battery (DIF all-rows, Tier-1 60/60 x2 det,
stock 59/59, deep=off superset, perf reps). **THE 75 GOAL CROSSED: 75.56 tok/s
(118.00 ms/cyc, 8.92 tok/cyc, alpha 3.958)** at LOOKUP_K=10. RUNG B (the
relaxed-LMIN graded lookup) was KILLED AT THE SIM: on 100k natural-text
histories the best-window match length is BIMODAL {0, 8} — LMIN 6/7 adds ~zero
fires, so the HyperQwen graded tier has no purchase on this workload class.
Daemon flipped to K=10 and smoked. Commits: 4e5923b (K=9), 4092b21 (K=10).

## THE LADDER (all Tier-1 60/60 x2 det + stock 59/59 + deep=off @100k)

| config | ms/cyc | tok/cyc | tok/s | E[m|deep] | deep-sel | notes |
|---|---|---|---|---|---|---|
| K=8 bank (R7a) | 104.65 | 7.48 | 71.51 | 8.000 | 78.3% | session start |
| **K=9 (LOOKUP_K=9)** | **110.66** | **8.27** | **74.70** | **9.000** (94/94 ALL-NINE) | 78.3% | +3.19 |
| **K=10 (LOOKUP_K=10)** | **118.00** | **8.92** | **75.56** | **10.000** (92/92 ALL-TEN) | 76.7% | +0.86 — SHIPPED |
| deep=off superset (K=10 cfg) | 66.2 | ~2.8 | 42.06 | — | — | exact fallback intact |

- Phase split (K=10 run): draft 5.36 / probe 59.50 / accept 1.16 ms.
- Per-rung cost: +6.0 ms (K=8->9), +7.3 ms (K=9->10) blended — the ladder is
  past its inflection (increment +3.19 -> +0.86 tok/s); offline D4 rows say
  K=11/12 would add ~+0.4 or less. THE K-LADDER STOPS AT 10.
- In-vivo ALL-K acceptance held FIRST-TRY at every new depth (E[m|deep]=K
  EXACTLY, 6 straight rungs now — the offline boundary truncation is the only
  decay the table sees).

## CONSTRUCTION (the mechanical per-rung pattern, 3rd/4th consecutive first-try)

- gen_m10.py / gen_m11.py (m9->m10->m11 M-extension, ALL AUDITS PASS):
  ACC(N+1)H2/RED(N+1)/IQ3V(N+1) macro extensions, call-site arg inserts
  (13 ACC / 15 RED / 13 LDH2 appends EVERY rung), a-decl extensions
  (11 compact + 2 spaced + 1 ffn ag/au), row-N stores on EVERY write family
  (11 direct + y4 + gact + OB-macro + k3ao-z), toks[N+1] + s-arg, k0ab
  %/ decode, renames with bounds PRESERVED.
- k2s10/k2s11: REC-CHAIN SLOT LAW held first-try at t=9 (rec9x) and t=10
  (rec10x); rec_out/conv_dst scratch chains extended; [48][5] live=slot-4
  layout PRESERVED — zero trunk/serve surgery at every rung.
- lookup10 (iend=pos-18, dring0..8) / lookup11 (iend=pos-19, dring0..9);
  accept10k (m-ladder to 9, 21-word emit) / accept11k (to 10, 22-word);
  acceptsel10k (m==9 -> rec10x/conv10x) / acceptsel11k (m==10 -> rec11x/conv11x).
- spk ROWS=10: RMAX=60/RP=64/MAXOWN=2 — 64 regs EXACT, 40,704B smem, 0 spill.
  spk ROWS=11: RMAX=66/RP=80/MAXOWN=3 — 64 regs, 41,536B smem, ONE SASS-audited
  COLD 4B index spill (STL in staging + one predicated LDL in reduce; the 12
  HMMA are STL/LDL-free — far below the P18 >~100B nondet class;
  documented-allow in build_k11.py).
- r7 twins via gen_r7d10/gen_r7d11: ffn8v10r7 68r / down8nw32v10r7 64r;
  ffn8v11r7 70r / down8nw32v11r7 64r — the nw32 64-reg budget HELD to M=11
  via body-scoped unroll knobs (op38nw32_11 + down8nw32_11 + down8nw32v11r7
  4->2; ao8nw32 4->2 at M=10; ffn8v 5->2).
- mtp.py wiring: RM dict, M(N)_CUBINS, poison blocks, spk loads, dring8/9,
  emit alloc 26-word, lookup dispatch + draft_lu_g (draft-skip generalized
  LOOKUP_K >= 8), graphs6 -> _probe10_seq/_probe11_seq (norms per-row CTAs:
  k0n/hh grid M, k0ab grid 13*M), DecodeSession layout-aware emit parse,
  PF_DR7 assert extended. test_w100k: DIF NR dict, dring seeds, trace widths,
  acceptselNk branches, R8_PROSE harness feed.

## NEW LAWS (this session)

1. *** THE BIMODAL MATCH LAW (kills graded-LMIN lookup on this class) ***: on
   100k-token natural-text histories the engine-selected best-window match l
   is BIMODAL — either the full 8-gram recurs (l=8) or nothing >= 6 does
   (prose corpus: 998x l=0, 2x l=6, ZERO l=7; gate: 22x0/38x8; quote-ceiling:
   6x0/112x8). Relaxing LMIN 8->6/7 adds ~0.2% fires, ALL spurious (net
   -0.1 tok/s on prose from wasted deep triggers). The HyperQwen/llama.cpp
   graded-lookup refinement assumes a populated l=6/7 middle — it does not
   exist here. (r8_lutsim.py, three corpora, engine-exact scan semantics.)
2. *** MACRO-DEF INSERTION LAW ***: when appending a new RED/ACC-style macro
   after an existing one in a shared .cu, insert AFTER the FULL \-continued
   definition — inserting after only the first line splits the continuation
   and breaks every kernel split downstream (found by nvcc, not by audits:
   python-level string audits CANNOT catch it; compile is the only truth).
3. *** THE UNROLL-CREEP LAW (M-extension register budget) ***: each M-rung
   tips a different kernel family over the 64-reg/1024-thr budget at unroll 4
   (ao8nw32 at M=10, op38nw32/down8nw32/down8nw32vNr7 at M=11); body-scoped
   unroll 4->2 / 5->2 fixes each at ZERO per-row fp-order change. The knob
   must be scoped per-kernel-body (the pragma text is shared verbatim across
   families).
4. LDH2-row-append generators: `(\w+)8` regex groups EXCLUDE the digit —
   new-name = group + "10", never group[:-1]+"10" (the xv8->x9 chopping bug;
   compile-caught). ACC(N)H2 call patterns need (N X + WV + N A) groups —
   count before asserting.
5. THE NOVEL-TEXT TIE-FLIP CLASS (pre-existing, NOT R8): on novel prose the
   spec stream vs the T=1 GCycle reference diverges at a near-tie
   (deterministic at K=10: 7/60, first div 6; at the SHIPPED K=8 it is the
   same 7-8/60 AND cross-rep unstable). The canonical Tier-1 gate prompt
   (repeat region) is 60/60 at every rung; novel-text cross-config fp16
   top-2-gap flips are the R2-era tie-mine class. K=10 is CLEANER than K=8
   here (deterministic x2 vs a cross-rep flip).
6. The fill_draft 100k replay hit the transient single-fault class mid-run
   (30s wait timeout at 68k/97810; clean on rerun) — the P18 distrust-single-
   faults law holds.

## RUNG B — THE SIM TABLES (killed; banked for any future revisit)

Corpora: (gate) the 100k repeat-region continuation; (prose) a REAL 1046-tok
novel reply (captured live from the daemon, re-encoded at the special-token
boundary — r8_corpus.py) fed as a 100k-ctx follow-up; (quote-ceiling) a
verbatim doc span. Engine-exact scan (iend=pos-8-K-1, suffix=last7+cur, max-l
then newest-i), trigger-chain cycle model (deep_j = prev hit; c2=66.0ms,
cd(K)=104.65+7*(K-8)ms; stale-dring miss m=0.7; k2 baseline 2.78 tok/cyc):

- LMIN 6/7/8 are IDENTICAL on every corpus (the bimodal law): gate 67.0 tok/s
  at K=8-arm (offline m, uncalibrated), quote 82.6, prose 42.0 vs 42.12
  baseline (LMIN=6 is NET -0.1: the 2 spurious l=6 positions trigger deep
  cycles that miss).
- In-vivo prose confirmation (R8_PROSE harness, K=10 config): lookup hits
  0/60 (0.0%), tok/cyc 1.02, ~15.1 tok/s — the K2-path bound on novel text
  (novel-prose MTP alpha ~0; the 40+ tok/s class is repeat-region-only).

## PROSE-CLASS HONEST NUMBERS (the two workload classes, both measured)

- HIT-CLASS (repeat/quote regions — the gate prompt): 75.56 tok/s.
- PROSE-CLASS (novel text, 0 lookup hits): ~15.1 tok/s (1.02 tok/cyc, 66 ms
  K2 cycle) — the MTP K2 draft accepts ~0 on novel text; the deep-K lookup
  never fires. THE PROSE LEVER IS NOT LMIN (bimodal law) — it is draft alpha
  (the DFlash2-class block drafter, R5-banked) or cheaper K2 cycles.

## SHIPPED CONFIG

Daemon (r7_daemon_ctl.sh, flipped 09-23): the R7a line with LOOKUP_K=10
(env = M1C canonical + PF_W4A8=1 PF_P5=1 PF_OP64=1 PF_RING4=1 PF_QKV1=1
PG_SPLIT=4 PC_ENABLED=1 + LOOKUP_K=10). Smokes: health ok (parked 97810),
FRESH lighthouse 64-tok render -> coherent think-stream + natural stop at 145
toks (SAME reply as the K=8-era daemon — cross-config serve consistency),
streaming SSE + finish-window + cancel-on-disconnect clean, FOLLOW_UP with a
synthetic history falls back FRESH per the STOP-BATCH law (prefill green).
Kill-switch: LOOKUP_K=8 restores the R7a config bit-exactly (kernel sets are
env-disjoint); LOOKUP_K=0 the R2-class engine.

## Harness inventory (new, env-gated)

r8_lutsim.py (graded-LMIN sim, 3 corpora), r8_corpus.py (live-reply capture +
re-encode), R8_PROSE=1 (novel-reply feed on the R5_QUOTE harness),
gen_m10/m11.py + gen_r7d10/11.py + build_k10/11.py (the rung generators),
~/r8_dif.sh / ~/r8_gate.sh / ~/r8_prose_gate.sh (gate batteries).
Logs: ~/r8_dif_k9/k10.log, ~/r8_gate_k9/k10.log, ~/r8_prose_k10.log,
~/r8_prose_k8.log (the tie-flip discriminator).

## NEXT (if the program continues on decode)

- The K-ladder is DONE (increments < +1 tok/s/rung, hits decaying ~-1.6pt/rung).
- The remaining decode levers: the ~950-cycle-class probe cost (59.5ms of the
  118.0 — the D3-halving program: ffn_down/gemv_gdn narrow-load families,
  49.3% of the deep increment), K2-cycle 66ms floor (prose-class 15 -> ~20+),
  or the draft-alpha program (prose-class unlock; DFlash2 parked at R5).
