# P7F-2 — THE IMMA INT8 ATTENTION: BUILT, VALIDATED, **NO-SHIP** (the pair is not dot-engine-bound)

Mission was: replace fp16 QK (m16n8k16) with m16n8k32.s8 on the already-int8
kv8 KV, target attention 204.9 -> <=40 ms/chunk. The kernels were built and
are bit-deterministic, but measurement says the premise is false on this dext:
**the attention pair is phase-structure-bound (barriers + staging + softmax +
PV), NOT tensor-core-bound. The int8 QK swap is perf-neutral-to-negative.**

## 1. What was built (all in engine0/, committed)

- `pf_attn8.cu` — three kernels + attribution knobs:
  - KSEL=1 `pfa8` (cubin pfa8nw32_s32_100k): pfa16 skeleton, QK on
    m16n8k32.s8 (P7A-validated fragments, verbatim maps), Q per-(row,32ch)
    s8 + f32 scales, K staged as raw u8 XOR 0x80 -> s8 (no dq8 for K), s32->f32
    per-group epilogue (sq[row][g]*sk[pos][g]), 2 barriers/tile (K/V separate
    smem regions), PV unchanged fp16.
  - KSEL=3 `pfa8t64` (pfa8t64nw32_s32_100k): TILE=64, K B-frags DIRECT from
    global (LDG.32 ^ 0x80808080, l1-guarded), V dq8-staged f16, SCP->Pm
    IN-PLACE f16 (single 12KB plane), sk staged f32, 3 barriers/64 keys
    (the PV->V-stage race fixed), single-pass 64-wide owners softmax.
  - KSEL=2 `pfk_q8` (pfk_q8nw8): Q quantizer, warp/row, group-amax via
    xor-1,2 butterflies, `__float2int_rn`, grid 48x256thr; **qsc bit-exact
    vs the numpy reference; qs8 differs only in tie-rounding bytes**
    (deterministic kernel-side; q per-(row,32ch) is tighter than the dp4a
    probe per-(row,128ch)).
  - ATTR=1..4 cubins (attn8a{1..4}): phase-skipped variants for attribution.
- `pfq8_probe.py` — standalone harness (modes base|a8|a8b|attr): real-scale
  synthetic kv8/qw, poison-first, synced min-of-N bench, relerr vs the fp16
  pair, determinism x2.

## 2. Ground truth (standalone, pos=100336, grid 128x1024thr)

| kernel | ms/launch | KV GB/s | TOPS | notes |
|---|---|---|---|---|
| pfa16 (shipped) | **2.81** | 73.0 | 14.0 | deterministic x2; ~1.0ms fixed overhead |
| pfa8 v1 (TILE32) | 3.07 | 67.0 | 12.9 | STACK:72 spills; relerr below |
| pfa8t64 v3 | **2.86** | — | — | correct after race fix; still >= pfa16 |

pos=32768: 1.45 / pos=2048: 1.00 (pfa16) — ~0.36ms is CTA-wave fixed cost
(128 CTAs / 82 SMs at hard 1 CTA/SM = 2 waves) + identity-partial writes.

**Phase attribution (v1 skeleton, pos=100336):** empty-loop 0.36 | +staging
0.98 | +QK 1.50 | +owners+PV 0.51. No phase dominates; ~1.6us per
__syncthreads at 1024 threads; the dot engine is a minority shareholder.
Halving barrier count (v3) bought ~0.2ms; killing K staging bought ~0.1;
the mma count halved on a 2x-rate pipe bought ~nothing. QED: structure-bound.

Numerics (v1, synthetic data): pA relerr med 7.25e-3 / ps med 6.3e-3 vs the
fp16 pair at pos=100336 — ABOVE the 2e-3 Tier-2 bar on synthetic
distributions (softmax amplifies flat-score regions; real-text may be tamer,
but with no perf win there is no reason to carry Tier-2 risk). **PV verdict:
PV stays fp16 — the sv[pos][chan-group] scale varies along the contraction
index, so an int8 PV cannot factor scales out of the epilogue without 8x
re-quantized P variants (measured-out by design analysis, not attempted).**

## 3. New laws (hard-won, keep)

1. **48KB STATIC smem cap**: ptxas rejects >0xc000 static smem regardless of
   carveout; >48KB needs dynamic smem + func-attr opt-in = NOT reachable via
   the raw-cubin launch path on this dext. The 47488B v3 fits only via the
   in-place SCP->Pm trick. (43520B m32 GEMMs remain the largest proven.)
2. **KSYM law (fault signature)**: loading a cubin under its FILE name when
   the entry symbol differs => GSP "SM Warp Exception: Illegal Instruction
   Encoding" + Multiple Warp Errors = garbage execution, NOT memory. Always
   TinyELF(name=<entry symbol>) (pfq8_probe.py KSYM map).
3. **Offset-view byte-size law**: half-views of f16 buffers are 512 B/row —
   a 256 B/row view = silent 98KB OOB read that faults ONLY when the
   neighbor page is unmapped (order-dependent; the P5 allocation-history
   fault class).
4. Global B-frag reads must be l1-guarded (tile padding past l1 reaches
   past CTXK at pos~100k).
5. Owners softmax over >32 keys must be single-pass over the full width
   (two-pass 32+32 with a stale max = inf/NaN in Pm).
6. TILE=64 with in-place SCP->Pm needs the PV->V-stage barrier (3/tile).

## 4. Why no-ship (and what the 600 path actually needs)

- Attention FAMILY at pos97k = 204.9 ms/chunk; the pfa launches are ~90 ms
  (32 x ~2.8ms); the rest is pfk_pre16 + pfc16 + o-proj + scraps. Even a
  PERFECT dot engine (0.3ms/launch = KV-read floor) only reaches ~115 ms.
- Wave math at hard 1 CTA/SM: wall ~ ceil(4S/82) x W/S => every S in the
  divisor set of 100352 (=2^11*7^2) lands at ~W/16. Only more ROWS per CTA
  break the wall, and M32-rows needs PV acc = 48 regs/thread > the 64-reg
  cap at 1024thr (acc would spill to death; PV acc in smem f16 = 96KB > cap).
- **Next levers by measured magnitude**: (1) DBUF-M32 GEMM hybrid (P6-priced
  1.3-1.6x on the 60-70ms/chunk GEMM quant-load floor — the biggest single
  lever); (2) fuse pfk_pre16+attention+combine per block (48 launches ->
  16, kills per-launch fixed ~0.36ms x 32 = ~11.5ms/chunk); (3) persistent-
  CTA attention (one CTA/SM resident, KV streamed, barriers replaced by
  warp-local sync) if attention itself is attacked again; (4) norm-GEMM
  fusion. The IMMA machinery (validated fragments, quantizer, probe) is
  banked and ready if a persistent-CTA rewrite makes the dot engine matter.

## 5. Ops

- NO config change: the daemon stays on the P7F1 canonical line (PF_PG=1
  default-on). Daemon was shut down for the session (shutdown RPC; reboot
  law fired once, clean) and relaunched + verified after the campaign.
- Harness: `cd ~/tinygrad-metal/engine0 && DEV=NV ~/tg311/bin/python -u
  pfq8_probe.py base|a8|a8b|attr` (full env not needed — raw-cubin probe).
