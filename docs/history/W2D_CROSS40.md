# W2D: GEMV polish + K=3 — 34.77 tok/s @100k Tier-1 EXACT (+1.47); 40 NOT crossed

## TL;DR
Mission: >=40 tok/s @100k greedy, Tier-1 exact, via (1) weights-GEMV polish,
(2) K=3, (3) int8-KV. Outcome: **Lever 1 landed +4.08ms probe (34.77 tok/s,
Tier-1 60/60 bit-exact x2, deterministic, stock 59/59 — NEW CANONICAL)**.
Lever 2 (K=3) was fully built + Tier-1 EXACT 60/60 but NET-NEGATIVE (24.16
tok/s): the draft's per-position alpha COLLAPSES with depth (0.867 -> 0.608;
tok/cyc 2.22 < 2.73) — the mission's "alpha holds at depth" assumption is
refuted at this draft quality. Lever 3 (int8-KV) not attempted: projected
78.6-10 = 68.6ms -> 39.8 tok/s — still under 40 even if perfect; budget spent
on the two structural levers instead.

## NEW CANONICAL (locked, gated)
env: `SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1` (+ DEV=NV PATH/
DOCKER_HOST as usual; `python -u test_w100k.py`, ~/snap100k bootstrap).
| metric | W2C | W2D (GEMVV=1) |
|---|---|---|
| tok/s @100k (best of 2) | 33.30 | **34.77** |
| cycle | 82.09 ms | **78.60 ms** |
| phases | draft 5.60 / probe 75.35 / accept 1.18 | 5.62 / **71.27** / 1.17 |
| Tier-1 spec==T=1 | 60/60 x2 | **60/60 x2, deterministic** |
| engine vs spec_base_100k | 59/59 | 59/59 |
| alpha / tok-per-cyc | 0.867 / 2.73 | 0.867 / 2.73 |
Log: ~/w100k_gemvv.log.

## Lever 1 — what worked, what didn't (all kernels BIT-IDENTICAL, poison-first validated)
Synced+pipelined bench of every M=3 probe GEMV (bench_g3.py) revealed the TRUE
in-graph rates: ffn8_3 234-302 GB/s, down8_3 161-236, q5g8_3 236-340, op38_3
64-133, ao8_3 62-109, k3ao3 154-270, head8_3 368. The "380-445 GB/s" prior was
pipelined-artifact-class; in-graph reality ~230.
1. **T-layout transpose (pack_t.py, 16B lane-contiguous uint4 loads — the E4
   winner pattern): NO GAIN (slightly worse).** Load WIDTH is not the wall.
2. **half2-packed cores (m3v2.cu: ACC3H2 — hmul2/half22float2 elementwise ==
   hmul/half2float, per-acc add order preserved -> bit-identical): +3-19%.**
   q5g8v_3 406 vs 340 GB/s pipe (+19%); ffn8v_3 +4-5%; down8v_3 +8%.
3. **Fat CTAs (1024thr, one row/warp, "nw32" names auto-LS in gcycle.py):
   helps SMALL-grid kernels only** — down8 (640 CTAs) +11-15%; op38/k3ao/ao8
   similar class; ffn8 (2176 CTAs) and head8 (31040) ZERO — the win is
   warps-in-flight at 1-CTA/SM-class residency, not raw CTA count.
Winner set wired via GEMVV=1 in mtp.py: q5g8v_3 (2048g), ffn8v_3 (2176g),
down8nw32_3 (160g/1024t), op38nw32_3, k3aonw32_3, ao8nw32_3 (160g/1024t).
Every one validated BIT-IDENTICAL vs the m3.cu original on real weights
(test_t.py / test_v3.py). Probe 75.35 -> 71.27ms (-4.08).

### Why the E4 843 GB/s class is NOT reachable here (attribution)
E4 was a wide-load GEMV without quant DECODE. The IQ3/Q5/Q6/Q8 decode chains
(sign extraction, 2x float4 grid gathers per block-lane, per-element
f2h/hmul/h2f round-trips) make these kernels ISSUE/ALU-bound at ~60-80% of
DRAM: cutting load instructions (T-layout) changed nothing; cutting ALU
(half2) bought the observed few %. The remaining ~38ms GEMV pool would need
either a decode-in-smem LUT restructure or dequant-to-fp16 with 5x DRAM
(843x/5 < current) — dead end for this format set.

## Lever 2 — K=3 fully built, EXACT, and NET-NEGATIVE (banked)
Built: m4.cu (15 M=4 kernels — h_embed4/k0n4/k0ab4/q5g8v4/k2s4/op38nw32_4/
k3aonw32_4/ao8nw32_4/hh4/ffn8v4/down8nw32_4/aq3k8v4/aq6k8v4/head8v4 + amx3
grid-4 reuse), accept4.cu (m-ladder to 3), skv_split.cu ((ROWS==3?..) ternary
fixes -> ROWS=4-legal) + spk_pre4/spk_g4nw32a4/spk_c4{,g} at 2k(S=32)+100k
(S=256), mtp.py K3=1 mode (4-row scratch RM, 3-step draft with dpos2/dring2,
accept4, qw3/pm3/ps3/pA3 at 6*RM rows). BUG FOUND+FIXED: k2s4 t=3 conv window
read live[3*CONV_CH] = OOB into the next block's conv slot (correct term =
qkv4[(t-3)]); symptom row-3 garbage.
- 2k: agree 0/60 vs T=1 — the KNOWN degenerate near-tie repeat region
  ([3204,40224] vs [248044,198] — same flip signature the W3 trio had); not
  diagnostic at 2k.
- **100k: Tier-1 60/60 EXACT x2, deterministic, stock 59/59 — machinery
  CORRECT.** But alpha(pos) 0.867 -> **0.608**, tok/cyc 2.73 -> 2.22, cycle
  91.75ms (draft 8.34 + probe 82.37 + accept 1.16) -> **24.16 tok/s**.
  Verdict: the draft (Q4_0 blk.64 + 40960-slice head) loses acceptance fast
  with depth; E[N]=3.27 assumed alpha holds — it doesn't (matches W2_MTP's
  warning "investigate m=0 at depth 0 before K=3"). K=3 parked (K3=1 env-gated
  OFF, default K=2). The alpha program (draft fidelity: DFlash2-class block
  drafter, trunk-hidden resync) is THE prerequisite — that is model-side work,
  not kernel work.
Log: ~/w100k_k3.log, ~/w2k_k3.log.

## What remains to 40 (ranked, for next session)
1. **int8-KV** (halves attention ~21 -> ~11ms): with GEMVV=1 projects to
   ~68.6ms = **39.8 tok/s — TANTALIZINGLY at the line but under**; needs the
   K=2+L1 base plus one more ~1ms shave (head8v4-style half2 head, k0ab/hh
   merge, or the aq* half2 cores not wired) to cross.
2. **alpha program** (draft fidelity): unlocks K=3+ (machinery now EXISTS and
   is Tier-1-proven; only the draft holds it back).
3. Attention ~21ms is 283 GB/s synced G4-NW32 — the W2C wall stands.
4. Remaining GEMV pool ~38ms at issue-bound decode; only structural decode
   changes move it.

## Files
- engine0/m3v2.cu (v2 half2 + fat-CTA kernels; split per-kernel cubins by
  build_t.py), pack_t.py + packed_t/ (T-layout, KEPT for the record — no win),
  bench_g3.py, test_t.py, test_v3.py, test_v2.py.
- engine0/m4.cu + accept4.cu + build_k3.py + skv_split.cu (ROWS=4-legal skv).
- mtp.py: GEMVV=1 (L1 winner swap) + K3=1 (K=3 mode) + RM-sized scratch.
- Logs: ~/w100k_gemvv.log (canonical gate), ~/w100k_k3.log, ~/w2k_k3.log.

## Gotchas banked
- Remote-file editing through nested ssh heredocs with single quotes in the
  payload silently corrupts (zsh glob/quote breaks): ALWAYS scp local-written
  files instead.
- m4 kernel paren typo class: hrcp/hexp2/hmul nesting needs 5 closes.
- k2sN conv-history generalization: first-term source = (t < 3) ? live[t*CC]
  : qkv[(t-3)] — the live window is exactly 3 rows; anything else is OOB into
  the NEXT BLOCK's slots (silent garbage, not a fault).
- skv.cu KPRE/K2S are ROWS==3-hardcoded via ternaries; skv_split.cu fixes to
  (ROWS==1 ? 0 : t*stride). spk_g4.cu is fully ROWS-generic (RMAX=6*ROWS).
- 2k Tier-1 vs T=1 is NOT a valid gate for new attention/batch kernels (the
  degenerate repeat region flips [248044,198]<->[3204,40224] on any
  reassociation); 100k is the healthy-region truth.
- The E4 843 GB/s number does NOT transfer to dequant-GEMVs (issue-bound
  decode); pipelined benches still overstate in-graph GEMV rates ~20-30%.

## Run
cd ~/tinygrad-metal/engine0 && env SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352
GEMVV=1 DO_T1=0 DEV=NV PATH/DOCKER_HOST as usual; python -u test_w100k.py
