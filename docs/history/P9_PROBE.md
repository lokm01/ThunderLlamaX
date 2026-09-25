# P9 — THE CO-RESIDENT ATTENTION PROBE (pfa32c): built, validated, **1.20x — RECORD & STOP**

The three-analyst adjudication question: can a co-resident attention kernel
exist at <=64 regs (512thr x 64 x 2 CTAs/SM = the regfile), and does it beat
the shipped pfa16 (2.81 ms/launch @pos100k) by >1.3x? Built per kimi's spec,
grok's kill-switch executed. **Answer: YES it exists (64 regs, 0 spills);
NO it does not beat 1.3x — best variant t32 = 1.20x (2.36 ms). Verdict band
1.0-1.3x = record and stop; no ship, no full build.**

## 1. GATE 1 — register feasibility: PASS (decisively)

ptxas via -Xptxas -v, launch_bounds(NTHR, MINB) pressure (build_p9.py):

| variant | regs | spills | smem | thr | CTAs/SM legal |
|---|---|---|---|---|---|
| **pfa32c_t64** (K global-direct, TILE=64) | **64** | **0** | 42112B (42.1KB, spec-exact) | 512 | **2** (64x512x2 = regfile; 84.2KB<=100KB carveout) |
| pfa32c_t64 unhinted (diagnostic) | 126 | 0 | 42112B | 512 | 1 (confirms P8's "everything sits at 128") |
| **pfa32c_t32** (K+V smem-staged, TILE=32) | 64 | 104B | 36992B | 512 | 2 (73.8KB<=100KB) |
| pfa32c_16c (R=16, 256thr) | 80 | 0 | 37952B | 256 | regs legal at 3 (61,440<=65,536) BUT smem 113.9KB>100KB -> smem-capped at 2 |

The rewrite-class premise is TRUE: a 512thr attention kernel at exactly 64
regs with zero spills exists and co-resides 2/SM under the CTASM unlock.
Barrier counts: t64/16c = 3/tile (warp-private V staging makes the PV/V-stage
race structurally impossible — each warp stages only the channels its own PV
tiles read); t32 = 4/tile (linear staging, extra "PV done" barrier).
CH = ceil(CTXK/S) — S=13 does NOT divide 100352 (2^11*7^2); floor-div drops
the last 5 key positions at pos=100336 (the shipped kernels only ever used
divisor S).

## 2. Correctness @2k (CTXK=2048, pos=2032, poison-first, vs pfa16 pair)

Combined (flash) output over splits vs pfa16 on identical real-scale
synthetic kv8/qw (P7F2 distributions):

| variant | relerr med | relerr max | den relerr med | determinism x2 |
|---|---|---|---|---|
| t64 | 7.45e-07 | 1.97e-01 | 8.3e-08 | bit-exact |
| t32 | 7.07e-07 | 2.02e-01 | 7.5e-08 | bit-exact |
| 16c | 6.79e-07 | 2.57e+00 | 7.4e-08 | bit-exact |

Med is ~3000x under the 2e-3 bar (fp32-rounding class); maxes are the known
near-zero-element outlier class (P7F2's pfa16ctl note). All four kernels ALSO
ran correct + deterministic at pos=100336. NOTE: not bit-exact vs pfa16
(different TILE/S => different softmax renorm grouping) — Tier-2 class if
ever wired; a pfc16 combine variant for S=13/S=10 would be required.

## 3. GATE 2 — synced min-of-5 bench @pos=100336 (the P7F2 protocol)

| kernel | default carveout | AUTO 100KB (2/SM) | best vs pfa16 2.82 |
|---|---|---|---|
| pfa16 (shipped, grid 128x1024) | 2.89 | 2.82 | 1.00x |
| pfa32c_t64 (grid 156x512) | 3.29 | 3.10 | **0.91x — LOSES** |
| **pfa32c_t32 (grid 156x512)** | 2.60 | **2.36** | **1.196x** |
| pfa32c_16c (grid 240x256) | 4.02 | 3.72 | 0.76x |

**Attribution (t32): 2.82 -> 2.60 = structure (K smem-staging f16, 512thr,
156-CTA 1-wave at 1/SM); 2.60 -> 2.36 = co-residency (the carveout, ~10%).**
Co-residency is REAL (+10% on t32/t64/16c alike) but compounds with — not
multiplies past — the structure gain. The K-global-direct idea (t64, the
spec's main variant) is REFUTED: per-frag u16 loads + on-the-fly dq8 from
global cost more than the smem staging they replace (echoes P7F2's pfa8t64
verdict). 16c: more CTAs x less work each = launch/wave-floor bound, loses.

L2 note: the (g,s,hp) hp-fastest ordering contribution is not separable from
this data (no profiler); the wave math (156 CTAs = 1 wave at 2/SM vs
128 = 2 waves at 1/SM) is the dominant structural effect we can attribute.

## 4. Verdict + what it would be worth

- **GATE 1 PASS, GATE 2 = 1.20x (t32) — the 1.0-1.3x band: RECORD AND STOP.**
  The >1.3x GO (full build: in-plan wiring + gates) did NOT trigger. The
  L2/re-read + co-residency theory delivered real but sub-threshold gains.
- If ever revisited: t32 at 2.36ms x 32 launches ~ -15ms/chunk @pos97k
  (~90ms pfa pool -> ~75ms) ~ chunk 161 -> ~146ms ~ **~223 tok/s @100k**
  (from 202.5) — ABOVE P8's 210-220 safe-margin endpoint, but Tier-2
  numerics + S=13 combine wiring + gates are the entry cost, and the
  protocol priced that at >1.3x only.
- Artifacts: engine0/{pf_attn32c.cu, build_p9.py, pf32c_probe.py} + cubins
  pfa32c_{t64,t32,16c}_s{13,13,10}_{2k,100k} + pfa16r_s8_2k (fresh 2k
  reference — the on-disk pfa16nw32_s8_2k FAULTS at 2k buffer sizes; suspect
  a misnamed CTXK=100352 build, the name-encoded-config law; left untouched).
- No ship changes. Daemon relaunched on the P8 canonical line after the probe.
