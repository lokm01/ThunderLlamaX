# CTASM INVESTIGATION — Root-cause of the "1 CTA/SM" limit (2026-09-18)

MISSION: root-cause the platform observed hard 1-CTA-per-SM property to its owning
layer, patch if ours, verify (co-residency microbench + decode Tier-1 gate + prefill
GEMM A/B), or document immovability.

## VERDICT (one paragraph)

The "1 CTA/SM hard limit" DOES NOT EXIST at the platform layer. Multi-CTA/SM
co-residency works through the full stack (fork QMD -> dext -> GSP-RM -> GA102):
measured 12 CTAs/SM (threads-limited), 5, 3, 1 — textbook resource-based occupancy.
The 1-CTA pin on our prefill GEMM family is OUR OWN fork-userspace carveout pick:
ops_nv.py NVProgram.__init__ chooses the MINIMUM SM carveout config in {32,64,100}KB
that fits the kernel per-CTA smem, which caps co-residency at floor(cfg/usage) = 1
CTA/SM for ANY kernel with >16KB smem (32KB cfg) or >32KB smem (64KB cfg). All P5/P6
production GEMM kernels (22.2-43.5KB smem) were pinned at exactly 1 CTA/SM by this
line. Fixed env-gated (NV_SMEM_CFG_AUTO); ffn M32 GEMM moves 0.617->0.514 ms
(221->265 GB/s amortized, +20%) with ZERO kernel changes. The dext, server.c,
GSP-RM, and the QMD free_cta_slots_empty_sm field are all exonerated by direct
experiment (below).

## 1. Layer survey (evidence)

| Layer | File:line | What it does | CTA-residency content |
|---|---|---|---|
| Fork QMD build | ~/tinygrad-src/tinygrad/runtime/ops_nv.py:313-345 | builds QMDV03 per program: shared_memory_size=shmem_usage(+1KB law), register_count_v from ELF EIATTR_REGCOUNT(0x2f), min/target/max_sm_config_shared_mem_size | THE OWNER: line ~322 min-fitting carveout pick |
| Fork launch | ops_nv.py:150-171 NVComputeQueue.exec | PCAS + dependent-QMD chain; grid bound to cta_raster_width/height/depth | none (intra-kernel CTA distribution is HW) |
| Dext | extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp (198 ln) + TinyGPUDriverUserClient.cpp (191 ln) | IOBuffer/DMA mapping only | NONE (no sched/CTA/channel-class code) |
| Server | extra/usbgpu/tbgpu/installer/Shared/server.c (285 ln) | BAR map, sysmem FDs, RPC forward | NONE |
| GSP-RM | ops_nv.py:617-700 NVDevice init | channel group ENGINE_TYPE_GRAPHICS, ctxshare SUBCONTEXT_ASYNC, AMPERE_CHANNEL_GPFIFO, NVC6C0 compute obj, GPFIFO_SCHEDULE | channel-level only; CTA residency proven working through it |
| QMD free_cta_slots_empty_sm (bits 664-671, nv_570.py:14528) | never set by the fork (=0) | probe: 0 = HW-auto, NOT a 1-slot gate (5 CTAs/SM observed with field=0; explicit =1 does NOT clamp to 1) | cleared |

## 2. The co-residency microbench (direct measurement)

Kernel engine0/ctaswz.cu + runner engine0/ctaswz_run.py: per CTA records
[smid, clock64_start, clock64_end] (tid0), all 128 threads spin a bounded 3M cycles;
host computes max overlapping intervals per smid = max simultaneously-resident CTAs/SM.
GA102 here reports NSM=84 but 82 SMs take CTAs (2 fused off — matches 3090 spec).

| config | smem/CTA | carveout (QMD min_sm_cfg) | MAX CTAs/SM | floor(cfg/smem) | note |
|---|---|---|---|---|---|
| ctaswz 128thr | 1.0KB | 32KB (9) | 5 (grid-limited; 12 at 16x grid) | 32 | 12 = threads cap 1536/128 |
| ctaswz8k | 9.2KB | 32KB (9) | 3 | 3 | exact resource math |
| ctaswz32k | 33.8KB | 64KB (17) | 1 | 1 | the "wall" reproduced |
| ctaswz32k + NV_SMEM_CFG=100 | 33.8KB | 100KB (26) | 3 | 3 | wall 7.7->3.2 ms (2.4x) |
| ctaswz32k + NV_SMEM_CFG=64 (control) | 33.8KB | 64KB | 1 | 1 | control: model holds |
| ctaswz8k + NV_FREE_CTA_SLOTS=1 | 9.2KB | 32KB | 3 | 3 | field does NOT clamp |
| ctaswz grid 16x (1344 CTAs) | 1.0KB | 32KB | 12 | 32 | threads-limited 12 |

Conclusion: occupancy is ordinary HW resource math. "1 CTA/SM" only appears when
smem x 2 exceeds the carveout — which the fork default pick guarantees for every
kernel with >half-carveout smem.

## 3. Why every P5/P6 GEMM was pinned (production kernels, measured)

| kernel | regs | smem | default cfg | CTAs/SM default | possible @100KB (with regs cap) |
|---|---|---|---|---|---|
| pfg_ffn_m32_hm_nw8k128 | 128 | 43.5KB | 64KB | 1 | 2 (regs 128x256thr = exactly 2) |
| pfg_iq3d_m32_res | 80 | 26.5KB | 32KB | 1 | 3 (regs allow 3) |
| pfg_iq3o_m32 | 80 | 26.5KB | 32KB | 1 | 3 |
| M16 twins (39.2/22.2/22.2KB) | | | | 1 | 2-3 |

## 4. The patch (fork userspace only — no dext rebuild, env-gated, default OFF)

ops_nv.py NVProgram.__init__:
- NV_SMEM_CFG_AUTO=1: pick the SMALLEST carveout in {32,64,100}KB whose
  smem-occupancy floor(cfg/usage) >= NV_SMEM_CFG_AUTO_TGT (default 2; TGT=0 -> max).
  Existing NV_SMEM_CFG/NV_SMEM_CFG_NAMES manual override still wins (elif).
- NV_FREE_CTA_SLOTS=N: writes QMD free_cta_slots_empty_sm (experiment knob; probe
  shows HW ignores it for residency — kept for future span-list experiments).
Default path is byte-identical behavior (canonical unchanged when env absent).

## 5. Verification

### (a) microbench — see table above (2-3 CTAs/SM achieved via patch; 2.4x wall).

### (b) prefill GEMM A/B — test_p6.py (synced, real packed weights), patch ON vs OFF:
| class | baseline M32 | AUTO=1 TGT2 | delta |
|---|---|---|---|
| ffn M32 | 0.617 ms / 221.3 GB/s amort | 0.514 ms / 265.3 GB/s | 1.20x / +20% |
| ffn M32 TGT0 | | 0.512 ms / 266.5 | same (L1 loss not yet binding) |
| iq3d M32 | 0.345 ms / 198.0 | 0.341 / 200.0 | flat — grid is 80 CTAs < 82 SMs: co-residency irrelevant as launched; needs grid growth (NTILE/split-K), not carveout |
| iq3o M32 | 0.170 / 141.3 | 0.168 / 143.0 | flat — same 80-CTA grid reason |

THE P5 WALL MOVED: the ">=300 GB/s requires M=32 rows / dynamic smem >48KB /
persistent CTAs" verdict was derived under the hidden 1-CTA pin. ffn now at 265
with zero kernel work; iq3d/iq3o need >=164-CTA grids to even exercise 2/SM.

### (c) decode canonical gate — clean same-day A/B (canonical env + NV_SMEM_CFG_AUTO=1 vs off):

| config | Tier-1 | ms/cyc | tok/s |
|---|---|---|---|
| patch OFF (default; ~/ctasm_gate_base.log) | 60/60 x2 deterministic, tier-2 60/60 vs W2D+W2E, stock 59/59 | **69.33-69.37** | **40.12-40.15** |
| patch ON global (~/ctasm_gate.log) | 60/60 x2 deterministic, tier-2 60/60 vs W2D+W2E, stock 59/59 | 70.83-70.92 | 39.25-39.30 |

Exactness: ZERO impact (scheduling-only change; bit-exact in both configs).
Perf: global AUTO costs decode ~2.1% (bigger carveout = smaller L1 for decode-path
kernels with >16KB smem that cannot use the 2nd slot — threads/regs-capped or
grid-limited). PRODUCTION POSTURE: keep default OFF; apply carveout for prefill
GEMM classes via NV_SMEM_CFG_AUTO=1 in prefill-scoped processes, or surgically via
the existing name gate: NV_SMEM_CFG=100 NV_SMEM_CFG_NAMES=pfg_ffn.

## 6. Corrections to prior laws/docs

- W2C_SKVG3.md "the dext does NOT co-schedule multiple CTAs per SM (1 CTA/SM
  hard...)" — WRONG. The carveout-100 no-op there was kernel-specific (G2 @S=256
  latency-bound and/or stale-bake era), not a platform property. Keep the carved
  lessons (pipelined benches race-inflate; name-gated override exists) but the
  "occupancy route CLOSED" conclusion is REVERSED.
- P5 "THE WALL" doc: the load-stream wall = the 1-CTA pin from the carveout pick;
  re-derive the >=300 GB/s routes under 2-3 CTA/SM co-residency.

## 7. P8 implications (priced, not done)

1. Re-price the DBUF-M32 hybrid under 2 CTAs/SM (load/compute overlap across CTAs
   on-SM now possible): ffn 265 baseline today.
2. iq3d/iq3o: grow grids to >=2x82 CTAs (smaller NTILE per CTA or split-K) — they
   currently cannot use a second slot per SM even though resources allow 3.
3. P7F2 attention phase-serialization: with smem<=~48KB and carveout 100KB, 2 CTAs
   of staging/QK/PV can co-reside — the phase-structure bound may lift.
4. Decode path unaffected by default (all decode kernels <=16KB smem; attention
   fat CTAs are 1024-thread = threads-capped at 1/SM anyway; spk_g4nw32 smem=1KB).

## Run log
- ctaswz: cd ~/tinygrad-metal/engine0 && env PATH/DOCKER_HOST/DEV=NV ~/tg311/bin/python -u ctaswz_run.py
  (env vars: CTASWZ_KERNELS, CTASWZ_GRID_MULT, NV_SMEM_CFG*, NV_FREE_CTA_SLOTS)
- GEMM A/B: same env + ~/tg311/bin/python -u test_p6.py
- Gate: canonical env + NV_SMEM_CFG_AUTO=1 + test_w100k.py -> ~/ctasm_gate.log
