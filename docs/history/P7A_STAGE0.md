# P7-A — Stage-0 microbenches: the four verdicts (all measured 2026-09-16)

The EVERYTHING-gates for the 600-tok/s prefill campaign. Each probe: build →
standalone exact-validate → synced min-of-N bench (async launch +
dev.synchronize; no pipelined races). Machine: post-shutdown-reboot, exclusive
GPU, all dext laws honored (hardcoded sizes, sequential loops, lane-in-address
only, single aligned smem array, full-warp masks, per-kernel cubins +
cuobjdump symbol check, nw-token names).

## VERDICT TABLE

| # | probe | result | verdict |
|---|-------|--------|---------|
| 1 | IMMA (int8 tensor cores) | m16n8k32.s8 **EXACT**; 151.8 TOPS dep-chain / **293.1 TOPS @ILP4** | **PASS** (≥35.5 needed) |
| 2 | Repacked-W stream | strided 284 GB/s → repacked **795.0 GB/s** (2.8x) | **PASS** (≥400) |
| 3 | Int8-KV stream (pfa16 pattern) | **716.5 GB/s** (u16-scale full pattern, exact) | **PASS** (≥300) |
| 4 | LDMATRIX | works, roundtrip EXACT; **LDS.32x4 3.3x FASTER** (5.59 vs 18.24 ns/warp-iter) | **PASS + NOTE** |

## DECISION

**(a) FULL CAMPAIGN including the int8 tier. 600 tok/s @100k stays the median
path.** IMMA PASS + both stream benches PASS with huge margins. Concretely:

- The int8 tier is UNLOCKED: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
  executes CORRECTLY (bit-exact vs numpy int32 on 4 independent warps) and at
  ~103% of the 284-TOPS dense int8 peak at 1 CTA/SM (293 TOPS measured at
  ILP4/82 CTAs; the task's 142 figure is the fp16-class peak — s8 dense peak
  is 2x that; either way ≥6x over the ≥25% bar). The compute side of
  600-tok/s attention (needs ~58-115 TFLOP-equivalent against a 300+ GB/s KV
  stream) is now proven feasible ON TENSOR CORES.
- The memory side is UNLOCKED: the P5 "pGEMM wall" is a LAYOUT wall, not a
  dext wall — offline repacking of the same real FFN-gate IQ3_XXS bytes into
  warp-contiguous 16B runs streams at **795 GB/s vs 284** for the shipped
  stage_w strided pattern (both checksum-exact over every byte). The
  int8-KV slab streams at **716 GB/s** with the full pfa16-style consumption
  (per-32B u16 scale loads + dp4a), 2.4x over the ≥300 bar and 9x over the
  shipped pfa16 ~70-80 GB/s.
- pf_gemm3 fragment feeding: use the **4x LDS.32 path** (measured 5.59
  ns/warp-iter vs 18.24 for ldmatrix.x4 on this dext — the dext penalizes the
  ldmatrix opcode; on real HW it's the reverse).

## The numbers (detail)

### 1. IMMA — engine0/p7a_imma.cu (cubins p7a_imma16832{,_i4,_u8}_nw4, p7a_imma8816_nw4)
- Validation: fragments host-packed per the PTX ISA maps (a0/a1 = k 0-15 rows
  gid/gid+8, a2/a3 = k 16-31; b0/b1 likewise; c = (gid,2tig),(gid,2tig+1),
  (gid+8,...)); iters=1; 4 warps x distinct random s8 data in [-7,7];
  **EXACT vs numpy int32 (all 512 c-frags)**.
- Bench (82 CTAs x 128 thr, 20000 iters): ILP1 dependent-chain **151.77 TOPS**
  (= 53% of peak from ONE accumulator chain — the mma pipe is deep);
  ILP4 **293.11 TOPS**. Ops = 2*16*8*32 per mma.
- m16n8k32 **.u8** variant: executes, benches 291.7 TOPS, but outputs do NOT
  match the unsigned reference (misexecutes vs documented layout — do not use).
- m8n8k16.s8 fallback: same class — executes (~248 TOPS at the corrected
  2*8*8*16 MAC count) but misexecutes vs the documented fragment map.
- **The ONE shape that matters (m16n8k32.s8) is exact and full-rate.**

### 2. Repacked-W — engine0/p7a_repack.cu (p7a_wstrd_nw{8,16}, p7a_wrep{8,16}_nw{8,16})
- Real weight bytes: `packed/fg0.npy` (FFN gate IQ3_XXS, 17408x5120, 34.1 MB).
- Strided baseline = the shipped stage_w access pattern verbatim (lane r/c
  decode, u16 quant words + u32 scales + d u16): **284.1 GB/s (nw8/272 CTAs)**,
  276.5 (nw16/136) — checksum-exact.
- Repacked = offline permute to [8-row group][49 u16 units/block][row] so each
  warp's whole stage slice is contiguous 16B runs (980 uint4 per group, 15680B
  = exactly the 8 rows' bytes, zero padding); lane-distributed uint4 loads
  (sequential step loop, lane in address): **795.0 GB/s (rep8_nw8)**, 788-786
  for nw16/ring16 — checksum-exact over every u16.
- The 2.8x gap with byte-identical data is the Stage-1 repack prize.

### 3. Int8-KV — engine0/p7a_kv.cu (p7a_kvs_nw32, p7a_kvs_nw16; ctxk=50176)
- Layout = pfa16 production layout: kv[g][row][256B], g 0..7 (K 0-3, V 4-7),
  sc[g][row][8] u16 scales. CTXK=50176 → 103 MB slab + 6.4 MB scales (the
  original 205 MB run was deferred by an unrelated smem-overread bug since
  fixed; both sizes exceed L2 by >15x, so DRAM-bound either way).
- Full consumption pattern: uint4 slab loads + per-32B u16 scale load + 4x
  dp4a per uint4; contiguous per-warp ranges; grid 82/164 x 512/1024 thr.
- **Checksum-exact (every slab byte + every scale read accounted)** and
  **646.4-716.5 GB/s** (best: nw32/grid82). An sc32 (u32-pair scale batching)
  variant was slower (648) and its checksum formula mismatched — discarded;
  the u16-scale variants are the shipped-pattern result.
- Context: 600 tok/s attention needs 124-247 GB/s sustained — 3-5x headroom.

### 4. LDMATRIX — engine0/p7a_ldmx.cu (p7a_ldmx_nw8)
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16`: **executes and is bit-exact**
  (512B smem region = 4 tiles x 8 rows x 16B; lane L supplies tile L>>3 row
  L&7 addresses, 16B-aligned — 8B rows fault the dext, banked as a law below).
- Rate (82 CTAs x 8 warps, 100k iters): ldmatrix.x4 **18.24 ns/warp-iter**
  vs plain 4x `ld.shared.u32` **5.59 ns/warp-iter** for the same 512B/warp
  fragment set → **the dext executes ldmatrix ~3.3x SLOWER than 4 LDS.32**.
- NOTE for pf_gemm3: feed HMMA fragments with LDS.32 (the current pf_gemm
  pattern is already right); do NOT adopt ldmatrix.

## New laws / gotchas banked (P7-A)

1. **WARP-UNIFORM LOAD TRAP**: a warp-loop whose load address lacks a lane
   term makes all 32 lanes issue the same address (L1 broadcast) — checksums
   come back 32x the truth mod 2^32 while bandwidth looks plausible. Lane
   distribution must be sequential-outer + lane-in-address (stage_w pattern).
2. **ldmatrix rows must be 16B-aligned on this dext** (8B-stride row
   addresses = SM "Multiple Warp Errors" device fault; in-bounds but illegal).
   An m8n8 b16 tile is 8 rows x 16B = 128B, NOT 64B.
3. **Scalar kernel args** need TinyELF signature entries
   `(None, 4, dtypes.int32, ())` per int + `vals=(...)` at launch; empty
   signature treats every arg as a buffer (`int has no va_addr`).
4. **nvcc shim needs ABSOLUTE source paths** (container view); relative paths
   = "No such file or directory" from cc1plus.
5. **smem reduce tail**: shfl_down leaves the total on lane 0 only —
   broadcast (`__shfl_sync(mask, acc, 0)`) before per-warp smem slots; and
   size the slot array for the index math actually used (an `[i*32]` read
   over a `[NW*64]` array was our one device fault — "Out Of Range Address"
   from SM 0 of every GPC, machine survives, fresh process clean).
6. **u8 / m8n8k16 s8 IMMA shapes misexecute** on this dext (run but return
   non-documented layouts/values). Use m16n8k32.s8 only.
7. The shutdown-RPC wire protocol is NEWLINE-terminated JSON
   (`{"id":N,"method":"shutdown"}\n`) on /tmp/llm-engine.sock.

## Files (engine0/)

p7a_imma.cu, p7a_repack.cu, p7a_kv.cu, p7a_ldmx.cu (sources; per-kernel
cubins p7a_*_nw*.cubin), build_p7a.py, test_p7a.py (validate+bench suite),
p7a_kvbis.py + p7a_kvpure/kvdp/kvsc_nw32.cubin (the kv bisection ladder:
410.6 pure / 358.1 +dp4a / 273.1 +u16-scales at warp-uniform issue — superseded
by the lane-distributed kernels), p7a_kvdbg.cu (the ×32-trap forensic),
p7a_ldmxrun.py, p7a_results.json. Logs: /tmp/p7a_imma.log, /tmp/p7a_rk3.log.

## What Stage-1 inherits (the campaign skeleton)

- **W-repack tier**: offline repacker (numpy, 49-u16-unit blocks per IQ3 block
  — zero padding) + a repacked-layout pGEMM generation. Budget from probe 2:
  795/284 = 2.8x on the W stream; at M=32 amortization that is the difference
  between the current 214 GB/s class and a ~600 GB/s class.
- **IMMA tier**: int8 GEMM with s8 operands (quantized activations + int8
  weights at the repacked layout), m16n8k32 only. 293 TOPS measured = the
  600-tok/s compute budget is real. Fragment packing maps proven in
  p7a_imma.cu/test_p7a.py.
- **KV tier**: pfa16 K1-class kernels can restructure around contiguous
  per-warp slab ranges (716 GB/s proven) instead of the current ~70-80.
- **Fragment loads**: stay on LDS.32.
