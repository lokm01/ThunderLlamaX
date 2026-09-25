# P14 — THE PERSISTENT-FFN MEASUREMENT: bit-identity PROVEN, speed KILLED (0.93x; the serial chunk chain is the wall, not the stream)

Status: **The P13 persistent-CTA fused FFN tile was measured IN-PLAN (the
standalone route is dead on this dext — see the fault-class law below) and
the verdict is KILL by the mission's own gates: pf13ffn_w{2,4,8} (grid 82,
NBLK=1 per-block drop-in) are BIT-IDENTICAL to the shipped
pfg3_ffn_r7_m32_nw8k128 on real packed7 weights (all ring depths, 8 blocks,
determinism x2, poison-first) but run at 0.91-0.93x the shipped kernel's
speed: 498-511 vs 465 us/blk, amort 267-274 vs 293 GB/s — under the 350
gate AND under parity. No ship change; the P12 canonical stands
(319.1 / 278.5 / 218.6 = 33.0% of 662 @100k). The 400 cross does NOT happen
on this route; the mechanism arithmetic below is the ceiling statement.**

## 1. The same-boot controls (post-EFI-cold-cycle)

- `test_p7b.py`: ALL BIT-IDENTICAL, ffn m32r7 0.478 ms/32r amort 285.6 —
  the documented control, machine class healthy (run twice this boot).
- `pf13_ext2.py` / `pf13_ext.py` / `pf13_ffn.py`: ALL FAULT at the first
  fresh-buffer launch after the weight-upload flow — the P13
  allocation-history fault class SURVIVES the EFI cold-cycle. **LAW
  (final): standalone GEMM-class probing is dead on this dext, period. The
  in-plan host process is the only measurement route.**
- **NEW LAW: the fault class reaches INSIDE the proven host process** —
  after the warm boot (engine + graphs + prefill), ANY fresh device-buffer
  allocation (P.up / P.poison of new names) faults at the next sync. The
  pf12_attr pattern is the law for post-boot probing: win_up into EXISTING
  plan buffers + PfGraph kernargs slabs ONLY (pf14_bench.py v2).

## 2. The in-plan measurement (engine0/pf14_bench.py; ~/p14_bench.log)

Boot = full trunk (PF_PERSIST=1: W7 = fg/fu-only, PF_PERSIST_MB=6000 ->
FULL 64/64 block coverage, 5.70 GB — it FITS), 2048-token prefill WITH the
persistent launches live in the captured chunk graphs (64 chunks clean,
med 106.8-107.6 ms — 4096 persistent in-graph launches, zero faults: the
graph-capture question is answered, persistent kernels QMD-chain fine).

NBLK=1 drop-in cubins (build_p13.py `_n1` targets): pf13ffn_w2_n1
(128r + 8B spill), w4_n1 (128r + 52B spill), w8_n1 (182r, 0 spill).

Gates (readout-order, first clean run):
```
[gate] pf13ffn_w2_n1: vs r7 nz=0/557056 det-x2 nz=0 -> BIT-IDENTICAL
[gate] pf13ffn_w4_n1: vs r7 nz=0/557056 det-x2 nz=0 -> BIT-IDENTICAL
[gate] pf13ffn_w8_n1: vs r7 nz=0/557056 det-x2 nz=0 -> BIT-IDENTICAL
[gate] w8 all 8 bench blocks: nz=0 -> BIT-IDENTICAL
```

Bench (synced min-of-10, 8 real blocks, per-block launch = the plan
pattern; w = original bytes, amort = 2x the P6 def, dram = x512/392):
```
R7-CONTROL (grid 272):   465.1 us/blk | w 146.7 | amort 293.5 | dram 191.6
PERSIST w2  (grid 82):   511.1 us/blk | x0.91 | amort 267.0
PERSIST w4  (grid 82):   497.5 us/blk | x0.93 | amort 274.3
PERSIST w8  (grid 82):   498.7 us/blk | x0.93 | amort 273.7   <- best
```
350-amort gate: FAIL. 1.15x gate: FAIL (slower than shipped). Ring depth
w2≈w4≈w8 within 2.6% — **the load ring was NEVER the limit** (latency fully
hidden at every depth => not load-bound). Eager-vs-in-graph note: the eager
r7 control (465) vs P12's in-graph 404us carries ~15% enqueue overhead;
the relative verdict is enqueue-symmetric (both benched in the same loop)
and the 0.93x deficit cannot invert into 1.15x in-graph.

## 3. THE CEILING STATEMENT (the mechanism arithmetic)

- Per parcel (one 64-col n-tile of one block): the kernel is a sequential
  KCH=128-chunk chain (NCH=40): XPR2/XCM2 (x stage) -> WCM2 (IQ3 decode to
  smem) -> MMAR (m16n8k16 mma) -> 2x __syncthreads per chunk. Measured
  chain rate: ~3.1 us/chunk -> ~125 us/parcel.
- The persistent launch: 272 parcels / 82 CTAs = 3.32 avg but 26 CTAs do 4
  (tail) -> 4 x 125 = ~500 us = the measured 498. The shipped 3.3-wave
  kernel: 465 us / 3.32 waves = 140 us/wave — THE SAME per-parcel chain.
  Wave pipelining and persistence hide it equally (barely): the chain, not
  the schedule, is the wall.
- Perfect balance (an NBLK>=8-class launch, impossible in-plan — blocks
  are sequential) would give ~415 us = amort ~330: STILL under the 350
  gate. **The tail is not binding; the serial chain is.**
- The P13 713 GB/s stream is unreachable through this chain: the WRING
  hides DRAM latency completely (w2=w4=w8) yet nothing speeds up — the
  smem round-trip (decode -> sync -> mma -> sync) per chunk gates at
  ~3.1 us, i.e. ~45% of the per-chunk stream time budget.
- **The one unexplored lever: eliminate the per-chunk __syncthreads via
  smem ping-pong (ws x2 + xs x2 = ~87 KB <= 100 KB sm_86 limit, needs the
  100KB smem config). That is a P15-class architecture change, priced at
  maybe 1.3-1.6x on the ffn pool (25.8 -> ~16-20 ms @2k => ~92-96 ms
  chunks => ~340-360 tok/s @2k) — still not the 400 cross without also
  winning qg/fd. The honest dext endpoint remains the P12 pricing:
  ~320-325 @2k / ~215-218 @100k (48-50% of 662).**
- IMMA W8A8 (P12) + persistence (P14) both falsified: the GEMM family is
  now closed on FOUR levers (P8 ceiling, G1/G2, IMMA, persistent-CTA).

## 4. What shipped (and what did not)

- NO ship change. PF_PERSIST stays default-OFF in pf_prefill.py (env-gated
  wiring kept for the record: W7 fg/fu-only candidates under PERSIST_MB,
  the g=82 plan line, all-three-variant cubin loads, graph-key extended).
- Regression gate (~/p14_g2k.log, PERSIST off, 8k-class default prompt):
  EVERY line == the banked P12 8k gate EXACTLY — tokA/tokB/tokC/tokD
  token-for-token identical, F 4.282e-02, divergence line identical,
  alpha 2.67, spec 169 tok/60 cycles. ZERO numerics regression.
  Perf: 257.1 tok/s (med 133.3) vs banked 278.5 (124.4) = +9ms/chunk
  uniform — attributed to degraded post-fault boot state (5 fault runs +
  one 30s wait-hang this boot; the AGENTS perf-drift law), NOT to code:
  the edits are env-gated off in this run and every output byte matches.
- Daemon: relaunched on the P12 canonical line (unchanged env), after a
  warm reboot (the session's 5 fault runs degraded that boot: pre-reboot
  8k gate ran 257.1 tok/s with ALL lines byte-exact; post-reboot the
  daemon's own 8k-class FRESH = 18.5s/5243 tok = 283 tok/s == the banked
  278.5 class -> the drift was machine state, not code).
  Verification: health ok (parked 97810, cur 4471, keepalive 10s);
  FRESH @1519-tok 7.4s (first-after-boot class); FOLLOW_UP 2.26s (==
  the M1C ~2.1s class, delta-prefill pinned conversation); 8k-class
  FRESH 5243 tok in 18.5s, greedy cur 1596 consistent across prompts.

## 5. Artifacts + gotchas

- engine0/: pf14_bench.py (the in-plan measure harness), pf13ffn_{w2,w4,w8}_n1.cubin,
  build_p13.py (+3 `_n1` targets), pf_prefill.py (PF_PERSIST wiring +
  E._pf_W7 attach).
- Logs: ~/p14_bench.log (the measurement), ~/p14_ext2.log, ~/p14_g2k.log.
- Gotchas banked this session:
  1. The allocation-history fault class SURVIVES EFI cold-cycles —
     standalone GEMM probing is permanently dead on this dext.
  2. Post-warm-boot fresh device allocations fault even in the proven
     host — win_up/PfGraph-slabs only (the pf12_attr law).
  3. No `timeout(1)` on this macOS — police runtimes from the driver seat.
  4. My graph-timing loop's prev-value bookkeeping was wrong (0.5us
     artifacts + a final wait-timeout); the boot's 4096 clean in-graph
     persistent launches are the authoritative graph-answer. The process
     exited cleanly; no reboot was triggered.
  5. docker/nvcc: /opt/homebrew/bin/docker + `docker start
     cuda-nvcc-persistent` after the boot (the boot hook warms Colima but
     not always the container).
