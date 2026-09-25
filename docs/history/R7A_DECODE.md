# R7A — THE DECODE-70 CAMPAIGN: 68.62 tok/s SHIPPED / 71.51 GATED (K=8 parked on a daemon-context bug)

Status: rungs 1a + 3 SHIPPED and live in the daemon (LOOKUP_K=7 canonical);
RUNG 4 (K=8) fully built + Tier-1-gated 60/60 x2 at the snapshot (69.47; 71.51
with the draft-skip) but PARKED from serving on a park-dependent daemon-side
corruption the snapshot harness cannot see (repro + discriminations below).
Commits: 1d16d30 (rung-1a), a1f27ee (rung-3), 3f70fd8 (rung-4 K=8 + draft-skip),
+ the api/pf serving fixes.

## THE LADDER (all Tier-1 60/60 x2 det + stock 59/59 + deep=off @100k)

| config | ms/cyc | tok/cyc | tok/s | notes |
|---|---|---|---|---|
| K=7 bank (R5d) | 107.49 | 6.78 | 63.11 | session start |
| + rung-1a (uint4 W-merge) | 107.35 | 6.78 | 63.19 | bit-identical, ~neutral |
| + rung-3 (norms per-row CTAs) | 98.86 | 6.78 | **68.62** | **SHIPPED** |
| + K=8 (T=9/M=9 set) | 107.73 | 7.48 | 69.47 | gated; E[m|deep]=8.000 94/94 |
| + K=8 + DRAFT-SKIP | 104.65 | 7.48 | **71.51** | gated; emit byte-identical |
| deep=off superset (shipped cfg) | 66.0 | ~2.8 | 42.1 | intact at every rung |

## RUNG 1a — the uint4 W-merge on the r7-unit family (1d16d30)
R7U now does ONE uint4 load + register extracts (q from {u4.x,u4.y} by
lane&3 >> 16*(cc&1), sw=u4.z, d=u4.w) — the 3 narrow loads (qr U16 + sw U32 +
dr U16) eliminated; all four u32 lanes consumed (the DCE law); bytes+order
VERBATIM. A/B nz=0 det-x2 all 6 kernels (r7a_test.py); SASS: pure LDG.E.128
W stream, 0 spill, nw32 held 64 regs.
*** LAW BANKED: the 3x W-load-instr cut is ~PERF-NEUTRAL at M=3/8 — the GEMV
critical path is the x-loads+ALU, NOT W-load instruction count. The R7-D2
"cheapest big lever" verdict was WRONG on magnitude; the merge is kept as
free bit-identical hygiene. Re-prices op38/k3ao/q5g8 width work (rungs 1b/2
BANKED, not attempted — same class, same expected ~0). ***

## RUNG 3 — norms/emb per-row CTAs (a1f27ee) — THE REAL DECODE LEVER
The D3 norms_emb pool (130 launches, 11.6ms deep-cycle) was CTA-SERIALIZATION,
not launch count: h_embed8/k0n8/hh8 ran at grid=1 (ONE CTA chewing an 8-row
serial t-loop), k0ab8 at grid=13. Now: t = blockIdx.x (grid 8); k0ab8
(t,g)-per-CTA grid 104 (w=(g-1)*8+warp covers [0,96) exactly once); the _3
twins grid 3/39. Per-row lane loops/shfl trees/stores VERBATIM -> bit-identical
(r7a_norm_test.py ALL 8 OLD==NEW nz=0 det-x2). Deep cycle 116.10 -> 106.35
(-9.75ms); deep=off 40.30 -> 42.14; sel-mode +5.43 tok/s.
*** LAW: one-CTA serial-t-loop norms cost ~Mx their parallel time; per-row
CTAs are FREE and bit-identical. Audit every grid=1 M-row kernel for this. ***

## RUNG 4 — K=8 (3f70fd8): built, gated, and the daemon bug
Construction (all mechanical per R5d; every build 0-spill): gen_m9.py (m8->m9,
ALL AUDITS PASS; aq3k8v9 needed unroll 5->2 — the R5 zero-spill knob, 48 regs);
gen_r7d9.py (ffn8v9r7 67 regs / down8nw32v9r7 64 regs — the nw32 budget HELD
at M=9); lookup9_nw32 (iend=pos-17, dring0..7); accept9k (m-ladder to 8,
20-word emit); acceptsel9k (m==8 -> rec9x/conv9x); spk ROWS=9 set (hm9
RMAX=54/RP=64, 10 pad rows, MAXOWN=2: 64 regs EXACT, 39,936B smem).
- Offline depth-8 pre-check: E[m|hit]=8.000 EXACT (31/31 ALL-8), reach law 100%.
- DIF gate: all 9 rows == ref; rows 2-4 BIT-IDENTICAL vs re-anchored T3; row 8
  healthy (REC-CHAIN SLOT LAW held first-try at t=8 -> rec8x).
- 100k Tier-1: 60/60 x2 det, stock 59/59, deep=off 60/60 @41.78, E[m|deep]
  8.000 (94/94 ALL-EIGHT), deep-selected 78.3% (hit decay -1.7pt vs K=7).
- Rung cost +8.9ms blended (+12.0 all-deep): hm9's pad rows + M=9 streams —
  worse than the +6.9 trend; 71.51 needs the draft-skip.

### THE DRAFT-SKIP (bit-identical lever, QUARANTINED)
On deep cycles (prev emit hit, 78%) the 2-step draft chain was pure waste
(lookup overwrites dring0/1) — draft_lu_g = lookup-only graph, ~4.1ms/cycle
saved; emit stream byte-identical to pre-skip in the harness (449 emitted,
pos_end 98259 both) -> 71.51. BUT in the DAEMON (FRESH path, unseeded park
state) it corrupts: see the bug section — skip OFF does NOT cure the daemon
corruption, so the skip itself is exonerated-ish and re-quarantined with it.

### *** THE K=8 DAEMON-CONTEXT BUG (PARKED; the blocker for 69.5-71.5 serving) ***
REPRO (deterministic, ~1 min): daemon at LOOKUP_K=8, run the api_gates suite
(its (c)/(d)/(e) multi-turn + cancel turns), then FRESH "Write a vivid
paragraph about a lighthouse in a storm." (renders to exactly 64 tokens) ->
generates ' 4, 5' (stale-counting continuation) and natural-stops in 5 tokens
instead of the correct think-stream.
DISCRIMINATIONS (all measured):
- K=7 daemon, SAME parks/prompts: correct think-stream (twice) -> the K=7 set
  is clean; the K=8 SET is guilty (not the api layer).
- K=8 + R7A_DSKIP=0 (draft-skip off): lighthouse CORRECT after a pineapple
  turn, but STILL WRONG inside the api_gates sequence -> NOT the draft-skip;
  some park left by the gates' multi-turn/cancel traffic poisons a later FRESH
  generate at K=8 only.
- K=8 fresh-boot (100k snapshot park): lighthouse CORRECT (think-stream) ->
  the leak is prior-conversation residue in buffers reset_fresh does NOT
  cover; K=8's new surfaces = dring7, rec6x..9x/conv5x..9x scratches, RM=9
  planes, the 20-word emit path, k2s9 t=8.
- The 100k Tier-1 harness NEVER sees it (test_w100k seeds dring0..7 with ref
  ids + resets between reps) — the harness's seeding masks the daemon path.
PRIME SUSPECTS (unproven): stale rec9x/conv9x read via acceptsel9k on an
m==8 accept whose k2s9 t=8 write was skipped/short-circuited; dring7=-1/0
class reaching h_embed9 on a deep cycle; the K2-graph accept's m-slot vs the
K=8 sel path. NEXT SESSION: R4_TRACE/R4_DIF on the DAEMON socket path (not
the seeded harness) with LOOKUP_K=8; bisect the (c)/(d)/(e) gate turns that
poison the park; then re-run the A/B/A lighthouse discriminator per fix.

## THE SERVING FIXES SHIPPED TODAY (pre-existing bugs, K-independent)
1. api_server finish-window: a stop token AT/BEYOND the client cap (deep-K
   overshoot can cross im_end past max_tokens) is invisible to the client ->
   finish=length + tail mirror (stays reusable). (c2-class.)
2. *** pf_prefill m64 r==0 BUG (n%64==0 FRESH prompts): the m64-own final
   head (_pf_last64 row 63 -> pfk_n16/head8/h_argmax) produced tok_slot=0 ->
   instant im_end; NEVER GATED (all banked prompts have r>0 tails or run the
   M128 trunk). FIX: when r==0 (BEFORE the chunk loop — the loop bound is nc),
   keep the last 64 tokens out of the M64 chunk loop and delegate to the M32
   path exactly like the r>0 tail. Verified: n=64 lighthouse now think-
   streams deterministically. M128's identical head shape is left as-is (the
   2k gate = 2048%128==0 ran it green). ***
3. r7_daemon_ctl.sh: api_server interpreter path fixed (CommandLineTools
   python3; the homebrew python3.9 was removed at some point).
4. build_r7a_norms.py: the split-law fix (the zero-width lookahead split's
   empty first element shifted ALL bodies by one -> 960-byte EMPTY cubins ->
   SKEDCHECK06_REGISTER_COUNT faults; build_r7d.py's `bodies = [b for b in
   bodies if b.strip()]` is LOAD-BEARING).

## SHIPPED DAEMON (canonical)
LOOKUP_K=7 + PF_* ship line unchanged (M1C + R7A run line in r7_daemon_ctl.sh);
logs: w100k_serve_r7a.log / api_r7a.log. Decode 68.62-class; prefill 503.6
@2k unchanged. K=8 config ready to re-flip the moment the daemon bug falls.

## BANKED / NEXT
- K=8 daemon bug (above) — THE gate to 69.5-71.5 serving.
- The draft-skip (R7A_DSKIP env, default 1, currently moot at K=7) re-enables
  after the K=8 fix; +3ms more there.
- op38 packed7 port + k3ao Q8 transpose + q5g8 width: re-priced ~+0-1ms total
  by the rung-1a law — LOW priority.
- K-split deploy (rung 5): untouched.
- The 90-decode remaining arithmetic: K=9+ per D4 needs the per-rung cost
  down (hm9 pad rows the first target).
