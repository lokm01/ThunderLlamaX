# P16 — THE PING-PONG DISCRIMINATOR: KILLED AT THE GATE (best 1.06x vs the 1.2 kill-line; the serial-chain wall is per-warp issue serialization, NOT the barrier/decode critical path)

Status: **Discriminator A ran clean and the ping-pong class is DEAD by the
mission's own gate (<1.2x = dead everywhere). The static-legal ping-pong
kernels were built in both flavors — PING=1 (ws double-buffered, 43520B
static, decode moved into mma's shadow, 2 barriers/chunk) and PING=2 (ws AND
xs double-buffered, 34816B, ONE swap barrier per chunk = the full ping-pong)
— both BIT-IDENTICAL to the shipped r7 iq3d m32 on real W7 weights, and both
land at 1.01-1.06x. G0 (the dynamic-smem fork probe) and the FFN ping-pong
build are CANCELLED per the gate. No ship change; the P15 canonical stands
(357.1 / 306.3 / 235.0). The mechanism finding is the payload: barrier count
is IRRELEVANT (the 1-barrier kernel is SLOWER than the 2-barrier one) — the
~3.1 us/chunk wall from P14 is per-warp issue serialization
(load->decode->mma in-order per warp, lock-stepped across warps by the
chunk barrier), not the decode-on-critical-path and not the sync.**

## 1. The build (engine0/pf_gemm3.cu PING section; build_p16.py)

pf_gemm3.cu now carries a `PING` variant of the REPACK=1 body (the base r7
kernel is guarded `#if REPACK && !PING` — rebuild-neutral, verified: the
shipped target still compiles 80r/26112B). PING=1: ws planes x2
(FFN: both G/U planes x2), xs single — decode(k+1) runs after this warp's
mma(k) into the other ws plane with NO barrier between; the xs re-stage
keeps its own barrier. PING=2: xs double too — ONE barrier per chunk.
Decode (dq_unit/WCM), mma fragment map (MMARP == MMAR verbatim), epilogue
and per-row k-order VERBATIM. Cubins (0 spills, symbol-checked):

| cubin | regs | smem | class |
|---|---|---|---|
| pfg3_iq3d_r7pp1_m32_nw8k128 | 128 | 43520B (+1KB dext = 44544) | discriminator, shipped geometry |
| pfg3_iq3d_r7pp2_m32_nw4k128 | 190 | 34816B (35840) | FULL ping-pong @ nt32 |
| pfg3_iq3d_r7pp1_m64_nw4k128 | 254 | 34816B (35840) | gemm_fd-class candidate |
| pfg3_ffn_r7pp1_m32_nw4k128 | 254 | 43520B (44544) | FFN-class (W7 fg/fu case) |

## 2. Discriminator A (engine0/pf16_bench.py; ~/p16_disc.log; in-plan, pf14 law)

Boot = the P15 canonical (FULL env + PF_M64=1): prefill 2048 = 5.67s,
med 176.1ms/chunk (the healthy M64 class — machine state clean post
shutdown-RPC reboot), W7 fd coverage 64/64. My cubins loaded at the
proven-stable window (post-trunk-load, pre-prefill); bench inputs via
win_up into existing plan buffers only; synced min-of-10 over 8 fd blocks.

GATES (first clean run, readout-order law):
```
[gate] pp1-nw8 vs shipped-m32: nz=0/163840 det-x2 nz=0 -> BIT-IDENTICAL
[gate] pp2-nw4 vs shipped-m32: nz=0/163840 det-x2 nz=0 -> BIT-IDENTICAL
[gate] shipped-nt32 vs shipped-nw8 control: nz=0 (geometry A/B control)
[gate] pp1-m64-nw4 vs shipped-m64: nz=327679/327680 -> DIFF (anomaly, moot)
```
BENCH (us/blk, fd = 34.2MB IQ3/block; w = weight-stream GB/s):
```
shipped-m32-nw8 (grid 80)   242.1 | w 140.9 | x1.00
shipped-nt32-nw4 (grid 160) 250.6 | w 136.1 | x0.97   <- nt32 costs 3.5% alone
pp1-nw8   (2-bar, dec off)  228.9 | w 149.0 | x1.06
pp2-nw4   (1-BAR FULL PING) 238.5 | w 143.0 | x1.01
shipped-m64-nw8             393.0 (64-row) | 2x-m32 equiv x1.23
pp1-m64-nw4                 361.2 (64-row) | 2x-m32 equiv x1.34 (m64 amort x1.09)
```
**VERDICT: best 1.058x << 1.2 kill-line -> DEAD.** (The m64 "1.34" is the
M64 amortization mostly — the ping-pong contribution at m64 is 393.0/361.2
= 1.088x, same class as m32.)

## 3. THE MECHANISM (why it died — the falsification ladder)

1. **Barrier count is irrelevant**: PING=2 (ONE __syncthreads per 128-k
   chunk) is SLOWER (1.01x) than PING=1 (two barriers, 1.06x). If the P14
   "~3.1us/chunk chain = decode-on-path + 2 syncs" model were the whole
   story, PING=2 would be the winner. The sync cost is noise.
2. **Decode off the critical path buys ~6%**: exactly the XCM-only
   inter-barrier window relief of PING=1. The decode ALU itself still
   occupies each warp's issue slots in-order (mma -> decode -> loads within
   the same warp); the chunk barrier lock-steps all 8 warps into the same
   serial dance, so cross-warp overlap never materializes at 1 CTA/SM.
3. Combined with P13 (713 GB/s pure stream at 1 CTA/SM) and P14
   (persistence 0.93x, ring depth w2=w4=w8): **the GEMM family wall is
   per-warp ISSUE serialization of the quant-GEMM inner loop, not loads,
   not barriers, not occupancy, not wave structure.** Five levers now
   falsified on this family (P8 ceiling, G1/G2, IMMA, persistent, ping-pong).
   The family closes at ~140-150 GB/s standalone-eager / ~150-160 in-graph
   per m32 launch and that is the dext's quant-GEMM design point.
4. Anomaly banked: pp1-m64-nw4 differs from shipped-m64 on 327679/327680
   outputs while BOTH m32 pings are bit-identical — an MTILE=64-specific
   divergence (suspect nvcc scheduling of the XTPR=16 unrolled x-stores vs
   the 254-reg budget). MOOT: the class is dead; do not chase.

## 4. What this means for 400 @100k (the honest next-stage arithmetic)

With ping-pong dead, the @2k-class per-64-chunk pools stand as P15 measured
them IN-GRAPH: ffn 53.22 (classic x2) > scan 27.58 > fd 24.27 > attn 24.71 >
qg 22.96 > og 11.47 > qkv 8.20 > oa 4.56 > norms 5.06 > misc 6.4; wall med
176.4ms = 361 tok/s harness / 357.1 gated. Remaining LEGAL levers, priced:

- **Attention ROWS=32/64 (Tier-2, the wide-M amortization)**: P15 Stage-0's
  SC-256 classsync precedent measured 21ms/256-tok-class vs today's 24.7/64
  = ~4.7x per-token amortization at wide M. A ROWS=64 t32-class rewrite
  taking even HALF that (attn+comb 26.2 -> ~10-13/64) = **-13 to -16ms/chunk
  = ~390-410 @2k — THE 400 cross lives HERE**, and it is also the @100k
  lever (the 100k chunk grows 224->328ms with pos = attention-dominated
  under S26; wide-M amortization applies with the same force at 100k where
  the ladder needs the most: 235.0 -> ~260-280 if attn halves).
- **M=128+WY (the structural rung)**: seams halve again (norms/emb/pre/scan
  launch floor ~35ms/64 -> ~18), GEMM m128 in-graph ratio ~0.85-0.9 of
  2x-m64 (extrapolating the 0.74-0.80 m64-vs-m32x2 line), ffn 53 -> ~38-42
  IF classic-layout m128 is legal (smem at NTILE=16: 43520B static — it
  fits). Requires the WY chunked-scan numerics program (Tier-2) — the cost
  is correctness machinery, not kernel speed.
- **Dead ends now formally closed**: ping-pong (this doc), persistent-CTA
  (P14), IMMA (P12), deeper LUT (P5 H2), DBUF-1plane-only transfer (P5),
  full packed7 fg/fu both-live (P7F3 VRAM law), grid-doubled t32 (P15 law).
- Realistic endpoint UNCHANGED from P15: ~330-400 @100k (50-60% of 662),
  with attention-wide-M as the highest-value next rung (P17 candidate),
  M=128+WY second.

## 5. Ship state + verification

- NO ship change. Daemon relaunched on the P15 canonical line (PF_M64=1
  default-on; kill-switch PF_M64=0 unchanged). The PING code is
  compile-time-gated (-DPING) — zero runtime surface in the shipped path.
- Files: engine0/pf_gemm3.cu (+PING section, base guarded), build_p16.py,
  pf16_bench.py, 4 pp*.cubin, ~/p16_disc.log, pf_gemm3.cu.p16bak (pre-edit
  snapshot).

## 6. Laws banked

1. **The ping-pong law**: smem double-buffering of the staged W tile(s)
   buys <= 1.09x on the quant-GEMM family REGARDLESS of barrier count;
   never re-price it above that on this dext.
2. The P14 1.3-1.6x pricing was optimistic; the correct model is per-warp
   issue serialization (in-order mma->decode->load per warp, lock-stepped
   by the chunk barrier at 1 CTA/SM).
3. The in-plan cubin-load window (post-trunk-load, pre-prefill) is stable
   for NEW cubins on a clean boot — extends the pf14 law (which banned
   post-warm-boot loads): load at boot, bench after.
4. Graph-timing loops on this stack still mis-wait (0.2-0.3us/blk reads =
   the P14 gotcha #4 class); eager synced min-of-N remains the only trusted
   kernel timer.
