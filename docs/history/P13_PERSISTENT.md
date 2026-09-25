# P13 — THE PERSISTENT-CTA PROBE: mechanism ANSWERED (713 GB/s steady state), fused tile BLOCKED by a launch-instability fault class

Status: **Kernel A (the mechanism question) answered DECISIVELY: the per-SM
steady-state stream at exactly 82 CTAs / 1 CTA / SM is ~713 GB/s aggregate —
the shipped GEMM family's 169-265 GB/s is NOT a per-SM stream limit and NOT
primarily wave/ramp (multi-wave adds only +17%). Kernel B (the persistent
fused FFN tile) is BUILT (3 cubins, decode/mma-verbatim, bit-identity
expected by construction) but could NOT be measured: a dext launch-path
instability — affecting the SHIPPED r7 kernel identically under the same
harness conditions — faults standalone GEMM-class launches on today's machine
state. The GO/KILL gate (≥350 GB/s amortized measured) is therefore UNPROVEN,
not failed. The remaining route is the in-plan wiring (PF_PERSIST=1 inside
the proven pf_fwd32 host process), which was always the required ship path.**

## 1. Kernel A — the per-SM steady-state stream microbench (pf13_stream.cu / pf13_stream.py; ~/p13_str.log)

Grid = literal NCTA (82 = one CTA/SM, exactly one wave, no rotation), each CTA
streams a contiguous parcel (2.62 MB) with a RINGD-deep uint4 register ring,
XOR consume (no smem, no syncs, no mma), synced min-of-10, 215 MB working set
(≫ L2). Signature: (ptr src, ptr sink, i32 nu16) — **the i32 scalar needs
TinyELF signature=(INT,) with vals=(NU16,); an empty signature silently drops
the val and the kernel no-ops (impossible-GB/s numbers are the tell).**

| variant | grid x thr | ms | GB/s aggregate | GB/s/SM |
|---|---|---|---|---|
| nw8 d4 | 82 x 256 | 0.302 | 711.1 | 8.67 |
| nw8 d8 | 82 x 256 | 0.310 | 693.9 | 8.46 |
| nw8 d16 | 82 x 256 | 0.304 | 706.7 | 8.62 |
| nw16 d8 | 82 x 512 | 0.302 | 712.8 | 8.69 |
| nw32 d4 | 82 x 1024 | 0.302 | 711.4 | 8.68 |
| **nw8 d8 (272-CTA ref)** | **272 x 256** | **0.855** | **833.8** | 3.07 (3.3 waves) |

**THE MECHANISM ANSWER:**
- Steady state at 1 CTA/SM = **~713 GB/s** (86% of the 834 multi-wave rate,
  76% of the 936 DRAM peak class). Ring depth ≥4 saturates (d4=d8=d16);
  thread count 256-1024 irrelevant.
- The shipped 3.3-wave structure buys only **+17%** over the one-wave steady
  state → **wave/ramp is a MINOR term.** The ffn family's 169-265 GB/s
  deficit vs 713 is the kernel's INNER SERIALIZATION (the stage_W→decode→mma
  →syncthreads chain per chunk — the P6 "stage_W/mma serialize" law), not
  launch structure and not a per-SM stream cap.
- **Headroom for a persistent kernel: 3.8x over the shipped ffn DRAM rate**
  (89.1 MB repacked/block at 713 GB/s = 125 us/block pure-stream vs 479 us
  shipped). The 350-GB/s-amortized gate needs only 1.23x over shipped — the
  stream path supports it IF the ring keeps loads ahead of the mma chain.

## 2. Kernel B — the persistent fused FFN tile (pf13_ffn.cu / build_p13.py)

Built per the mission spec: grid literal 82 (compile-time parcel map, parcel
p → CTA p%82, sequential stride-82 loop — the gridDim=0 law), packed7 stream,
WRING-deep unit register ring (loads lead decodes by WRING-2 chunks — legal
ONLY because persistent 1-CTA/SM frees the ≤128-reg constraint that falsified
P11-G2), x re-staged per chunk (L2-hot), decode + mma + epilogue VERBATIM from
pf_gemm3.cu REPACK=1 FFN m32 → bit-identity expected (same math order; the
identical body was P7B-proven bit-identical). smem 43520 B = the shipped m32
profile.

Cubins (build_p13.py, all symbol-checked): pf13ffn_w2_nw8 (128r + 8B spill),
pf13ffn_w4_nw8 (128r + 52B spill), pf13ffn_w8_nw8 (**182r, 0 spill** — the
deep-ring variant the launch-bound world could never afford).

**NOT MEASURED — blocked by the fault class below.** The gates (bit-identical
vs pfg3_ffn_r7_m32_nw8k128 on real weights) and the ≥350 amortized bench are
PENDING the in-plan run.

## 3. THE FAULT CLASS (the session's hard finding — banked as a LAW)

Standalone GEMM-class launches fault on TODAY's machine state, and the
instability is NOT specific to the new kernel:

- `test_p7b.py ffn` passes 100% (5+ runs across 5 warm reboots; e.g. 0.479
  ms/32r, amort 285.1 same-boot control) — its flow allocates ~15 buffers
  THEN launches, and re-uses those buffers throughout.
- Every probe harness that allocates NEW buffers after a warm flow (or
  .offset()-view launches into large buffers) faults at the next sync —
  including the SHIPPED pfg3_ffn_r7 and P6-classic kernels (pf13_ext2,
  p13_run2/3: fault at first new-buffer launch; iso A-G; pf13_ffn.py).
- In-process ladder (p13_plus2/3, ~/p13_v.log): p7b-buffer launches CLEAN
  (base AND offset views on early buffers); fresh-buffer BASE launches CLEAN
  (V2c); fresh-buffer + offset-view second launch FAULT (V2); then even a
  4-buffer minimal persistent launch faulted on the next boot (p13_run3)
  while the same-boot canary passed.
- Cross-boot nondeterminism + allocation-history dependence + affects
  proven cubins ⇒ the documented dext "allocation-history dependent" fault
  family (the 2026-09 stash-fault class), NOT a kernel bug. The P5-P12
  standalone probes all ran fine in their sessions ⇒ today's dext state is
  DEGRADED relative to those sessions; warm reboots do NOT clear it.
- **NEXT SESSION LAW: run the EFI COLD-CYCLE PROCEDURE (pmset schedule
  poweron + shutdown -h, AGENTS 2026-09-16) BEFORE any standalone GEMM-class
  probe; if the instability persists after cold-cycle, standalone probing of
  this class is dead on this dext and the in-plan path is the only route.**
- Harness law discovered en route: **TinyELF scalar args REQUIRE
  signature=(INT,) + vals=(...)** (empty signature = silent no-op).

## 4. The honest verdict + the remaining arithmetic

- **Mechanism: GO-class evidence.** 713 GB/s steady state at 1 CTA/SM kills
  the "per-SM stream limit" kill-switch condition (the ~265 steady-state
  scenario is FALSIFIED). Persistence headroom is real: 3.8x stream margin
  over the shipped ffn DRAM rate; the 350-amortized gate needs 1.23x.
- **Fused tile: UNMEASURED.** The ≥350 gate is unproven, not failed. No ship
  change (P12 canonical stands: 319.1 / 278.5 / 218.6 = 33.0% of 662 @100k).
- **The remaining route (priced):** wire pf13ffn_w8 behind PF_PERSIST=1 in
  pf_fwd32's proven host process (where every shipped GEMM launch is stable):
  per-chunk persistent launch over the 64-block parcel map (weights already
  resident per-class in packed7 plan), x per block, out to the ffn staging
  buffers; then the Tier ladder (standalone-class relerr → 2k F → 8k set →
  100k cur). If ≥350 amortized holds in-plan: GEMM 69.7 ms → ~47 ms class,
  chunk ~102 → ~79 ms @2k ≈ **~410 tok/s = THE 400 CROSS** (P12 arithmetic).
  If it lands 265-330: bank the ceiling statement at ~330-345 @2k.
- The w2/w4 spill variants are the fallback ladder if w8's 182 regs misbehave
  in-graph (43.5 KB smem + 182r at 1 CTA/SM = legal by occupancy math, but
  in-graph capture is the untested interaction — measure both paths).

## 5. Artifacts + logs

engine0/: pf13_stream.cu, pf13_ffn.cu, build_p13.py, pf13_stream.py,
pf13_ffn.py, pf13_iso.py, pf13_ext.py, pf13_ext2.py, p13_plus2/3.py,
p13_run2/3.py, p13_name.py, p13_plus.py, cubins pf13s_* (6) + pf13ffn_* (3).
Remote logs: ~/p13_str.log-class output in-session, ~/p13_ffn_w2.log,
~/p13_canary.log, ~/p13_ext.log, ~/p13_ext2.log, ~/p13_v.log, ~/p13_v3.log,
~/p13_run.log, ~/p13_run2.log, ~/p13_run3.log, ~/p13_name.log, ~/p13_plus.log.
Rig state at close: daemon DOWN (shutdown RPC), machine needs EFI cold-cycle
before further GPU work.
