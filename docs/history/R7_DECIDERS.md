# R7 — THE FOUR DECIDERS (measurement-only; production untouched)

Status: all four decider experiments ran to a verdict. No production kernel
changes, no config flips. The daemon was stopped for the GPU blocks (graceful
shutdown RPC) and is to be relaunched with its exact captured env (see run
line at the bottom). New session-class laws found on the way (the multi-kernel
cubin law above all). Harnesses: engine0/{r7_sass_audit2.py, r7_lut_hist3.py,
r7_lut_proj.py, r7_d3_attr.py (+ R7_D3=1 hook in test_w100k.py, default-off),
r7_e1d.cu / r7_e1d_fill.cu / r7_e1d_run.py (single-kernel cubins)}.
Logs: ~/r7_sass_audit.log, ~/r7_d4_hist3.log, ~/r7_d3_attr4.log,
~/r7_e1_final2.log, ~/r7_e1_verdict (appended in final2).

## DECIDER 1 — E1 issue-law microbench: **WARP-SPEC ALIVE (latency, not issue-rate); headroom 265 -> ~490 GB/s = 1.8-1.9x**

Arms (82 CTAs x 256 thr = 1 CTA/SM; synced min-of-8; launch floor 0.198ms
measured by an empty-kernel arm; SM clock ~1.12 GHz via clock64):

| arm | wall | exec | GB/s | warp-instr/cyc/SM |
|---|---|---|---|---|
| A independent 16B loads | 0.527ms | 0.329 | 489.7 | 0.071 |
| B FFMA chains (ref) | 0.210ms | ~0.012 | - | >=2.2 (caveat: near-floor) |
| C HMMA m16n8k16 stream | 0.406ms | 0.208 | - | 0.44 |
| D production mix (2L+ALU+1M) | 0.523ms | 0.325 | 493.5 | 0.195 |
| E pure copy (4L+4S) | 0.527ms | 0.329 | 489.7 (r+w) | - |
| F dependent 16B loads | 0.570ms | 0.372 | 905 apparent (L2) | - |
| G empty (launch floor) | 0.198ms | - | - | - |

- Arms do NOT issue at the same rate -> the GEMM wall is NOT a hard
  N_instr/R_issue law -> the P16 "per-warp issue serialization" is a
  LATENCY-ORDERING effect, not an issue-RATE law.
- Arm D (the production inner-loop shape) hits 493.5 GB/s = the SAME as pure
  independent loads: decode-ALU + HMMA hide completely under load latency.
  The production GEMM at 265-298 GB/s in-graph = 54-60% of the achievable
  single-CTA ceiling -> **producer/consumer warp-spec (or deeper independent
  load pipelining) can recover ~1.8-1.9x on the W stream (265 -> ~490)**.
  Beyond that needs 2-3 CTAs/SM (the CTASM carveout route) — the ceiling
  itself moves, not just the utilization.
- GATES WHAT: the 4-6 day warp-spec/DBUF-M32-hybrid build is GO (expected
  payoff re-priced to ~1.8x on the GEMM family, not the 1.3-1.6x of P14 or
  the dead 1.06x of P16 ping-pong).

## DECIDER 2 — SASS load-width audit: **NARROW LOADS ARE REAL AND HOT — but split decode vs prefill**

(cuobjdump -sass + .cu hot-loop census; static counts == unrolled loop body,
verified exactly on ffn8v3r7.)

- PREFILL main W stream (pfg3_ffn/iq3d/iq3o REPACK=1): ALREADY true 16B
  (uint4 packed7 units) — the "prefill W narrower" suspicion is REFUTED for
  these classes. Exceptions: pfg3m_gdnqg's qkv seg (Q5_K dq_c2: U8 planes,
  48 launches/64-chunk = the 22.96ms qg pool) and the attnqkv v-seg (dq_c4);
  X-tile staging is uint2 (8B) everywhere (X = 16KB/chunk/CTA vs W = 4KB).
- DECODE W stream: NEVER unit-repacked — narrow everywhere:
  - r7-unit family (ffn8v3r7/ffn8v8r7, down8nw32v3r7/v8r7, op38nw32_3/_8;
    62+62+34+34 launches/cyc): each 16B-ALIGNED packed unit read as THREE
    narrow loads (qr U16 + sw 4B + dr U16 = 8B payload). ZERO-REPACK fix
    available: one uint4 fetch instead of 3 = 3x W-load-instr cut (the units
    are already 16B-aligned in memory).
  - q5g8v_3/q5g8v8 (48+48/cyc): Q5 = 2x 8B (LDG.E.64) + U8 scales.
  - ao8nw32_3/_8 (16+16): IQ3_S classic 4B+U8.
  - k3aonw32_3/_8 (14+14): Q8_0 = pure LDG.E.U8 (1B/lane!) + U16 scale —
    the narrowest hot stream; a uint4-packed Q8 = up to 8-16x load-instr cut.
- GATES WHAT: composes with D3 (below): the deep-cycle increment is
  GEMV-dominated, and the GEMV increment is dominated by exactly these
  narrow-fetch kernels -> the decode-side load-width merge is the cheapest
  big lever (the ffn/down r7-unit merge needs NO repack).

## DECIDER 3 — deep-cycle phase attribution: **the 47ms is GEMV-M-extension-dominated (69%), NOT attention (23%)**

(The R5d K=7 deep increment 116.32-69.21 = 47.1ms, never split before.
Method: per-family isolated ParityGraphs from the probe8/probe3 seqs, the
p18_attr law; R4 probe-poison law respected (drings seeded valid).)

| family | deep ms | k2 ms | delta | share |
|---|---|---|---|---|
| ffn_down (ffn8v8r7+down8nw32v8r7, 128 launches) | 42.21 | 24.26 | +17.95 | 37.7% |
| attn_rows (spk_pre8/a8/c8, 48) | 26.82 | 15.94 | +10.88 | 22.8% |
| norms_emb (k0n8/k0ab8/hh8/h_embed8, 130) | 11.59 | 4.51 | +7.09 | 14.9% |
| gemv_gdn (q5g8v8/k3ao/op38, 96) | 16.85 | 11.33 | +5.52 | 11.6% |
| k2s8 rec-chain (48) | 6.82 | 3.06 | +3.76 | 7.9% |
| head_amx (2) | 3.34 | 1.84 | +1.50 | 3.1% |
| gemv_attn (aq*q8v8+ao8, 32) | 4.48 | 3.58 | +0.90 | 1.9% |
| accept (accept8k+acceptsel8k) | 0.64 | 1.25 | -0.61 | -1.3% |
| SUM | 112.10 | 64.51 | +47.59 | (vs full-graph delta 48.37; attr err < 2.5%) |

Calibration: probe3 63.07 (R5d 62.65), draft 5.08 (5.31), deep cycle ~117.3
(116.32), k2 ~69.5 (69.21). The analysts' "attention 30-40%" fingerprint is
WRONG — attention retune at deeper T caps at ~23% of the increment. The
deep-increment-halving build should target ffn_down + gemv_gdn (49.3% = the
DECIDER-2 narrow-load families) and norms_emb's 130-launch floor (14.9%).

## DECIDER 4 — continuation histogram: **the ladder extends PAST K=9 if the per-rung cost stays <=7ms; optimum K~12-16**

(Offline, zero GPU; lut_deepk laws preserved — LMIN=8, reachA i<=pos-8-K,
newest-match, index over the GROWING tok_hist (the v1 bug: gate-class hits
are SELF-matches inside the model's own repeat loop).)

- The gate-class continuation is one long period-2 loop: depth does NOT decay
  within it (offline E[m|hit,7]=6.30 vs in-vivo 7.000 — boundary truncation
  of the 60-token sample; in-vivo ALL-SEVEN holds). The binding constraint is
  the REACH law: in-range hits 87%@K7 -> 82.6%@K9 -> 78.3%@K12 -> 69.6%@K16
  -> 60.9%@K20 (46-hit sample, +-5pt).
- Projections (f_deep(K) varying, in-vivo anchors):
  - measured +6.9ms/rung: K=8 +3.1 tok/s, K=12 76.6, peak K=16 79.9, K=20 declines.
  - trend +12.3ms/rung: K=8 +0.6, then flat/negative — ladder STOPS at K=8.
  - fudged +4.0ms/rung (if the D3-informed retune lands): K=16 87.8 tok/s.
- GATES WHAT: build K=8 next (positive in ALL cost scenarios); extend to
  12-16 only after the deep-cycle cost work (D3's ffn_down/gemv/norms pools)
  pins the real per-rung cost. 90-decode campaign arithmetic: the K-ladder
  and the GEMV load-width merge are the same money.

## NEW LAWS (this session)

1. *** THE MULTI-KERNEL CUBIN LAW ***: loading kernel X from a multi-kernel
   cubin executes WRONG CODE on this fork/dext (per-.text.<name> addressing
   broken) -> "Out Of Range Register" warp exceptions on every SM -> channel
   wedge (fresh process clears; warm reboot if stale). Bisected clean. The
   repo's -DKNAME one-kernel-per-cubin convention is LOAD-BEARING — every
   future standalone harness MUST build single-kernel cubins.
2. nvcc uint4 DCE law: consuming only v.x narrows LDG.E.128 to 4B component
   loads — consume all 4 words or the load-width experiment lies.
3. Host _copyin of ~1GB (16MB chunks) faulted mid-way — the DART danger
   class re-confirmed; fill large buffers device-side.
4. Effective SM clock through the dext ~1.12 GHz (clock64-calibrated); the
   launch+wait floor on this stack = 0.198ms (empty kernel, synced min-of-8).
5. The single-CTA W-stream ceiling at 256thr/CTA = ~490 GB/s (independent
   uint4 loads, 1 CTA/SM) — the honest denominator for all GEMM utilization
   claims at this occupancy.

## Daemon restart (exact captured env — unchanged from the R2d-era process)

```
zsh ~/r7_daemon_ctl.sh start    # = cd engine0 && env PATH/DOCKER_HOST + \
  DEV=NV M1A_SERVE=1 SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 \
  QH=1 PVH=1 HM=1 M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256 \
  LOOKUP_K=7 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 NV_SMEM_CFG_AUTO=1 \
  NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1 \
  PF_M128=1 PF_DR7=1 PF_ATTNW=1 PF_SCANC=1 PF_SCANC_N2=1 PF_M64QKV=1 PF_ABW=1 \
  PF_RING4=1 PF_QKV1=1 ~/tg311/bin/python -u test_w100k.py >> ~/w100k_serve_r2d.log
  + api_server.py on 8080
```
The R7_D3 hook added to test_w100k.py is env-gated default-off — the daemon
path is byte-identical when R7_D3 is unset.
