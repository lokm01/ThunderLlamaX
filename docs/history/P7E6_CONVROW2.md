# P7-E6 — The conv-row-2 channel DECODED: two components (kh/qh/qe/kht fp16 system perturbation FIXED; residual z-ULP -> content-dependent trunk amplification OPEN); SC stays unshipped

Status: **the P7E5 "conv-row-2 channel" is now fully explained as TWO
separate mechanisms, one fixed this session. (1) THE STATE CHANNEL (FIXED):
SC's preprocess outputs kh/qh/qe/KhatT rode BARE fp16 (the P7E5 hi-lo law
explicitly exempted "normalized tiles") — the fp16 quantization perturbs the
WHOLE WY system CONSISTENTLY: SC solves the fp16-kh model exactly (1.9e-5 vs
a fp64 sequential oracle on the SAME fp16 kh dumps) but that model deviates
from the true fp32-kh recurrence by ~2.3e-3/block at |S|~30. Block-0 rec
maxabs 2.31e-3 -> 3.7e-5 after the fix (60x; now the CLEANEST block).
(2) THE AMPLIFIER CHANNEL (OPEN): SC-vs-M32 z outputs differ by 1-2 fp16
ULP at block 0 in EVERY seeded world (C1 AND C3 alike — implementation-floor
rounding flips from o-cancellation in fast-decay heads), and the trunk
amplifies that ULP noise EXPONENTIALLY (~1.25x/block, measured: z-noise
2e-3@blk6 -> 2.9e-2@blk15) ONLY in worlds whose stream carries mature
row-2 content (C0/C3/C4 dirty: rec61 4.4e-2-1.0e-1; C1/C2/C5 bounded:
2.4-2.9e-3). The row-2 "specificity" of P7E5 was a THRESHOLD ARTIFACT:
row 2 (t-1) puts mature magnitudes into TOKEN 2, the deepest o-cancellation
regime; rows 0/1 land at tokens 0/1 (little cancellation) -> sub-threshold.
PF_SUPER stays 0. P6 canonical 165.3 tok/s stands.**

## 1. The forensic funnel (mission step 1 — all measured this session)

- pf_probe2.py (per-block rec, SC vs M32, C3/C1/C0 seeds, 256 tok single
  chunk): the pre-fix C3 profile grows SMOOTHLY 1.7e-4 (blk0 medrel) ->
  5e-2 (blk47) — no single guilty block; block 0 carries maxabs 2.31e-3
  (10-20x the 1.5e-4 floor of blocks 1-5). C1/C0 runs: block-0 maxabs
  2.52e-3 / 3.18e-3 — THE P7E5 "rows 0/1 inert" WAS WRONG AT BLOCK 0
  (their C1/C2 finals were clean only because the AMPLIFIER is separate).
- pf_probe3.py (z per 16-token M32 window): z outputs CLEAN (<=4.9e-4 =
  1-2 fp16 ULP) at blocks 0-2 pre-fix — the divergence lives in the STATE,
  not the outputs.
- THE ARBITRATION ORACLE (the session's key instrument): fp64 sequential
  recurrence of the pfs16 op order on dumped inputs. TWO P7E5-era oracle
  bugs found+fixed en route: (a) decay is exp(softplus(a+dtb)*ssma) —
  NATURAL exp (pfs16 verbatim), not 2^lam (the missing log2e factor made
  the old oracle garbage on fast-decay heads, |lam|~1.24); (b) M32's
  per-window states match the exact oracle on ITS OWN inputs to 3.9e-5
  (pf_probe4.py per-launch captures) — M32 IS EXACT; SC deviates 2.3e-3
  AND matched the oracle to 1.9e-5 when fed SC's OWN fp16 kh/qe dumps =>
  the fp16-quantized INPUT MODEL was the whole state error. Inputs
  bit-identical between worlds (qkv/araw/braw maxabs 0.0).
- THE CORRECTED ORACLE VERDICT TABLE (block 0, C3, per head):
  SC-vs-exact 1.9e-5 max | M32-vs-exact 2.3e-3 max (vs SC's fp16 dumps) —
  after the fix: SC block-0 rec 3.7e-5, M32-vs-SC 3.7e-5 (both at floor).

## 2. THE FIX (engine0/pf_scanchunk.cu, P7E6 HI-LO LAW EXTENSION)

kh, qh (phase-1 operands), qe, KhatT now split hi+lo fp16 (>=21-bit):
- pfca: preprocess dumps kh_lo/qh_lo from REGISTERS (lo needs the fp32
  value); the kh/kht/qe dumps moved PRE-phase-1 with hi+lo (qe lo computed
  in fp32 from qh*2^g); phase-1 = 3 passes ((hi,hi),(lo,hi),(hi,lo)) with
  smem reloads between passes (kh_ smem reloaded from global; B-operand
  khB read from global) — NO smem growth (43520B unchanged). M (qh x kh)
  accumulates only passes 0 and 2 (ph1 is B-only); B (kh x kh) all three.
- pfcb: Y = Khat@S16 -> sph{x}khp 3-pass (s16 rebuild once for lo); O's
  Qe@S16 -> 3-pass; S' = KhatT@(sg.d) -> 3-pass with the sg.d HI halves
  RECOMPUTED (fp32 from dz halves) for pass 2 (spass1's lo-swap clobbers
  sy). smem unchanged 34336B; pfcb_c64_nc4_nw8 = 72 regs, 0 spill.
- Layout: new LO region appended AFTER meta: kh_lo|qh_lo|qe_lo|kht_lo
  (+70656B); HCB 193040 -> 263696; pf_prefill.py HC64 + scscr updated.
- TWO self-inflicted bugs found by the funnel in the first build (banked):
  phase-1 M double-counted (hi,hi) [ph1 skip added]; S' pass-2 read
  (lo,lo) [sy-hi recompute added].

## 3. Post-fix measurements (all deterministic, 256/512 tok)

| variant | rec0 (block-0 state) | rec61 final medrel | logits medrel |
|---|---|---|---|
| C0 full snapshot | 7.6e-5 (was 3.2e-3) | 9.6e-2 DIRTY | 1.9e-2 |
| C1 row0 | 7.3e-5 | 2.6e-3 clean | 9.6e-4 |
| C2 row1 | 7.4e-5 | 2.8e-3 clean | 8.6e-4 |
| C3 row2 | 6.8e-5 (was 2.3e-3) | 4.4e-2 DIRTY | 8.5e-3 |
| C4 row0-in-all | 8.1e-5 | 5.3e-2 DIRTY | 1.3e-2 |
| C5 x0.01 | 7.4e-5 | 2.4e-3 clean | 1.1e-3 |
=> the STATE channel is closed everywhere; the C0/C3/C4 finals are now
purely the AMPLIFIER channel (z-ULP -> trunk growth).

## 4. The amplifier channel (OPEN — next session)

Facts: z diff (SC vs M32) = 1-2 fp16 ULP at blocks 0-2 in BOTH C1 and C3
(same source noise: o-cancellation in fast-decay heads — e.g. h2:
dtb=18.4, ssma=-0.059 => |lam|~1.2 => g spans -1.6 bits/token, o is a
~1e4-1e6-cancellation dot for 21-bit inputs => ~1e-3-absolute o noise =>
ULP flips after the z RMSnorm). The trunk then amplifies: z-noise 2e-3
@blk6 -> 2.9e-2 @blk15 (~1.25x/block, C3; C1 stays bounded ~2e-3).
The growth is CONTENT-DEPENDENT (same noise, different worlds) — the
mature row-2 stream sits in a chaotic/ill-conditioned regime (the WY
solve / delta-rule cancellations amplify input noise; M32 amplifies its
own equal-magnitude noise identically, so SC-vs-M32 separates).
NEXT LEADS: (a) quantify the o-noise floor per head (which heads flip);
(b) the only closure for exact gates = SUB-ULP z agreement — candidates:
compute z's pre-quantization value to >=24-bit for the cancellation
heads (o in fp64? split the M@d accumulation further), OR round SC's o
to M32's z quantization semantics bit-exactly (hard: M32's z comes from
fp32 core + half() — SC must reproduce the fp32 value to <0.5 ULP);
(c) accept a tolerance gate (greedy-token match at reduced beam) — a
PRODUCT decision, not an engineering one.
(d) re-audit whether the 100k serving world (real snapshot states) sits
in the bounded regime like C1 — the staged gates are the arbiter.

## 5. Rig/ops notes this session

- The P7E5 "boot-time fault class" (scdbg5r/5n + first runs this session)
  was an ENV bug, NOT a machine class: the harness resets upload
  2*4*100352*256 zero bytes into kv{i} — buffers sized by SKV_CTXK
  (default 2304 => 10MB) => OOB copyin => "Device fault". ALL engine
  harness runs need the canonical env: DEV=NV SKV=1 KV8=1 QH=1
  SKV_CTXK=100352 (+ PF_DFILL=0 for scdbg5*). A cold-cycle was performed
  unnecessarily (harmless; poweron schedule works).
- The daemon shutdown RPC + nv_usb4.lock clear worked as documented.

## 6. Ship decision + service

- PF_SUPER=0 PF_G3SC=0 UNCHANGED (C0 logits 1.9e-2 => staged gates would
  fail). Daemon relaunched on the P6 canonical recipe. P6 165.3 stands.
- Benchmarks NOT run (conditioned on shipping SC).

## 7. Files

- engine0/pf_scanchunk.cu — the P7E6 hi-lo extension (+bugfixes)
- engine0/pf_prefill.py — HCB 263696 + scscr size
- engine0/pf_probe2.py (per-block rec funnel + seed env), pf_probe3.py
  (z-per-16-token-window + M32 inputs + seed env), pf_probe4.py (M32
  per-launch rec/qkv/araw/internals capture) — NEW forensic instruments
- Logs: ~/p7e6_funnel*.log, ~/p7e6_m32cap.log, ~/p7e6_5c_fixed.log
- npys: ~/p7e6_funnel.npy, p7e6_funnel2.npy, p7e6_m32cap.npy,
  p7e6_oracle2.npy

## 8. END-OF-SESSION RIG STATE (read before any GPU work)

The serving daemon could NOT be relaunched: DETERMINISTIC device fault at
the FIRST decode-graph execution (engine loads, gcycle w1c0/w1c1 build OK,
fault inside G.run_tokens timeline wait) — 4/4 launches, surviving TWO
scheduled poweron+shutdown cold cycles (3-min and 10-min). NOT the SC
changes (PF_SUPER=0; SC cubins never loaded in this path; the morning
daemon ran the identical serving config for hours). Matches the P7E3
dock-power law severe form (dock stays powered across host shutdowns).
NEXT OPERATOR ACTION: PHYSICAL dock power-off (unplug 30+s), then:
  cd ~/tinygrad-metal/engine0 && nohup env DEV=NV SKV=1 KV8=1 QH=1 SKV_CTXK=100352 PF_PREFILL=1 PF_GEMM3=1 PF_SCANC=1 PF_SUPER=0 PF_G3SC=0 MTP_KERNARGS_MB=256 M1A_SERVE=1 ~/tg311/bin/python -u test_w100k.py > ~/w100k_serve.log 2>&1 &
then python3 api_server.py (:8080) if not running; health + one chat.
Logs: ~/w100k_serve_p7e6{,b,c,d}.log (same frame). Benchmarks NOT run (no
ship). Sec-5 env LAW applies to all harness runs.
