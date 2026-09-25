# P7-D/E — Attention widening verdict + super-chunk assembly (the honest state)

Status: **P7-D = MEASURED NEGATIVE on widening (banked with attribution) +
pfk_pre64 KEEPER (bit-identical, 4x fewer append launches). P7-E =
ASSEMBLED AND FAST AT POS 0 (~300-330 tok/s projected harness-class vs P6
245) BUT BLOCKED BY A POST-REFERENCE ALL-NaN (late OOB overwrite,
layout/allocation-history dependent) — SC path shipped env-gated OFF
(PF_SUPER=0 default); the daemon stays on the P6 M32 canonical until the
blocker is solved. P7-F spec below (NOT built this session, per mission).**

## P7-D — the widening campaign (all measured, synced min-of-10, same session)

Kernels (pf_attn2.cu, KSEL 1/2): pfaW = R q-rows/CTA (R=6*ROWS), TILE=16 keys,
NW warps, warp-LOOP QK, ms/ss in a per-warp PRIVATE smem plane (the reg law),
DBUF ring + 4 syncs/tile + P2 V-over-K kept. pfcW combine loops ROWS.

| config | regs | spill | per-token vs shipped pair |
|---|---|---|---|
| pfa16ctl nw16 T16 (R96 control) | 128 | 0 | ~0.97x (structure neutral) |
| pfa24 nw16 (R144, 24 tok) | 128 | 0 | 0.87-0.90x |
| pfa32 nw16 (R192, 32 tok) | 128 | 134B st/140B ld | 0.80-0.83x |

Bench (per 32-tok layer, shipped pair = pfa16x2+pfc16x2, L=100352):
pair 5.85ms (13.5 TFLOPS, 19% MFU, ~70 GB/s counting 2x KV reads) vs
pfa32-s128 6.73ms (11.1 TFLOPS, 29 GB/s 1x-KV) vs pfa24-s32 4.76ms/24tok.
**Widening is FALSIFIED per-token on this dext**: the PV accumulator wall
(acc = 8*RMAX/NW regs) blows the 64-reg/1024-thread limit — R192 needs ~121
regs -> NW=16 (512thr) where half the warp parallelism + 96-reg acc chains
(24 dependent mma tiles/warp with per-tile rescale) dominate; the kernel is
mma-latency bound at 29 GB/s DRAM (7% of the 447 ceiling), NOT load bound.
The P7-A int8-KV 716 GB/s probe does NOT transfer to this HMMA structure.
- Gates that DID pass: pfa32 partials pm/ps/pA BIT-EXACT vs pfa16 per row on
  identical inputs (online-softmax re-batching at TILE=16 is exact); final ao
  med 0.0 / F 6.6e-5; determinism bit-stable; pfk_pre64 vs 4x pfk_pre16:
  kv/sc/qw ALL BIT-IDENTICAL (the quantizer math verbatim x4 rows).
- Escape routes for >=25 TFLOPS (P7f-class work): (a) fp16-ACC pfaW (2-reg
  c-frags -> 48 acc regs at 1024thr, the one un-tried shape; numerics gate
  vs the pair must clear 3e-3 med); (b) int8-KV + dp4a QK outside mma; (c)
  the P7f IMMA tier below.

## P7-E — super-chunk (PF_SC=256): built, benched, blocked

Assembly (pf_prefill.py, PF_SUPER=1 to enable): emb/norms M-wide (pfk_n16
g=SC, pfk_ab16 g=13SC, pfk_hh16 g=SC) -> GEMMs (gemm3-m64 x SC/64 m-passes
on packed blocks; classic m32 twins x SC/32 views elsewhere) -> [attn:
pfk_pre64 ONE flat launch (24*NC CTAs, pos_arr per 64-row window) -> the
SHIPPED pfa16 pair per 16-row half (P7d verdict) -> iq3s o-proj] / [GDN:
qg -> pfca/pfcb/pfcz c64_nc4 chunked WY scan on LIVE trunk rec/conv (in-place
safe: pfcb slices are per-(head,v-quart) disjoint; pfca reads conv pre-pfcz,
stream-ordered) -> o -> FFN] -> head on last row; tails delegate to the M32
path; dfill 16 windows with pos_w[k]=pos0+16k.

- Timing (harness, pos 0, PF_DFILL=0): **256-tok chunk 780-860ms = 298-330
  tok/s projected** vs P6 fwd32 130.6ms/32tok = 245. ~1900 launches per
  super-chunk (vs ~6400 P6-class per 256 tok).
- THE BLOCKER: with a T=1 reference pass BEFORE the SC run, the SC output is
  ALL-NaN; without it (fresh process), the SAME code+inputs run CLEAN
  end-to-end. Forensics: block 0 fully clean (probed launches 1-8 incl. its
  scan); NaN appears at launches 269-275 (~block 15-16 boundary, SAME onset
  with PF_GEMM3=0 — gemm3 falsified as the cause); rec0 verified ZERO
  post-reset AND clean after launch 6, yet NaN at the end => a LATE OOB
  WRITE clobbers live state — the allocation-history/layout-dependent dext
  class. Falsified fixes: sync-after-uploads, 32-launch pacing, first-8
  warmup syncs, fixed-handle resets (win_up instead of P.up). NEXT SESSION:
  (1) bracket the clobbering kernel by writing a canary pattern across the
  rec/conv/x buffers and binary-searching launch windows (the probes are in
  place: PF_SC_STEPS); (2) suspect list: pfa16's pAS at S=32 on the SC
  layout, the classic m32 twin VIEWS at p*32 strides, pfca scscr NC stride;
  (3) the gemm3 m64 M-grid at M=256 (mb=1..3) is ALSO still only
  standalone-validated at M=64 — validate at M=256 vs 4x m32 before
  re-enabling PF_GEMM3 in the SC plan.

## Laws banked (P7-D/E)

1. **TinyELF symbol derivation**: `name.split("_")[0]` MANGLES
   multi-underscore symbols (pfk_pre64_100k -> "pfk") => loads a garbage
   entry => "Illegal Instruction Encoding" on EVERY SM (deterministic
   all-SM signature, empty fault report). Keep an explicit KSYM map.
2. **qw16 stride is 6144 halves/token** (24 head-slices x 256) — NOT the
   12288 qrow stride; harness views/compares at 12288 read/write garbage
   (the "rows 8..31 wrong" illusion was a stride bug, not a kernel bug).
3. **P.down size overflow = device fault**: reading (64,12288) from a
   (64,6144)-halves buffer faults the channel — size every down to the
   buffer.
4. **The reg wall for widened attention**: acc = 8*RMAX/NW regs/thread;
   R>~124 is impossible at 1024thr (64-reg cap); at 512thr it executes but
   loses to warp parallelism (measured, table above).
5. **Standalone harness kv/sc must replicate the ENGINE allocation recipe**
   (np.zeros count-style; trunk.py sc = 2x nbytes) — a differently-sized sc
   + poisoned kv correlated with launch hangs in the early P7d debugging;
   engine-replica buffers were stable.
6. **The SC post-reference NaN is REAL and layout-dependent** (see P7-E):
   never assume syncs fix dext corruption — probe canaries at launch
   milestones (PF_SC_STEPS infra landed).
7. pfk_pre64's quantizer math is row-position-independent: appends are
   BIT-IDENTICAL regardless of window packing (the x4-row law).

## The 100k benchmark (honest)

NOT re-run this session (blocked on the SC gates). P6 record stands: 165.3
tok/s rebuild. The SC harness projection at pos 0 (298-330 tok/s, 2k-class)
is a HARNESS number, not the 100k number: at 100k the attention stage
(shipped pair, 2x KV reads) dominates the super-chunk late in the context —
with the P7d falsification, the attention wall at 100k is THE remaining
structural blocker for >=300 @100k (see P7f).

## P7-F spec (the int8 tier — 600 median path; DO NOT BUILD, spec only)

Target: >=2x the HMMA ceiling via `mma.sync.aligned.m16n8k32.s8` (P7-A
probe: EXACT + 293 TOPS @ILP4; the .u8 and m8n8k16 shapes MISEXECUTE —
forbidden).
1. **Classes to convert (order)**: (a) FFN gate+up and down (the 127ms GEMM
   pool -> int8 W8A8): per-tile activation quantization s8 with per-128-
   channel scales; (b) GDN qg + attn qkv twins; (c) o-projs; (d) attention
   QK on int8-KV via dp4a (skip mma) or IMMA with per-(row,32ch) KV scales
   (the shipped sc layout); (e) head LAST (int8 head DEGENERATED in the W0
   era — quality gate first).
2. **Activation quantization scheme**: dynamic per-tile absmax over 128-
   channel groups (match the W group structure so epilogue dequant is one
   fma): a_s8 = rn(x/scale_a), scale_a = absmax/127 in fp32, computed in the
   GEMM prologue from the x smem tile (reuse the DBUF ring bubble). KV-side:
   the EXISTING biased-uint8 + sc u16 layout — dp4a needs (u8-128) bias
   correction: precompute K' = K-128 at append time (new kpre variant) to
   make QK' = sum dp4a(q, K') exact against dequant scales.
3. **Accumulate + epilogue**: s8xs8->s32 mma; per-out-channel dequant
   epilogue: out_f32 = acc_s32 * (scale_a * scale_w[col-block]) + (RES?
   residual read) — one fma per fragment pair, fp32 out, silu-mul for FFN in
   the same epilogue (H2 law: approximate intrinsics safe).
4. **Weight repack**: extend pack_w7.py: per-(8-row warp group, k-chunk)
   units already exist; add an s8-quantized plane (offline per-128-channel
   absmax; keep the q/sw/d IQ3 decode path as the W8 fallback tier).
5. **Numerics gate plan**: per-kernel relerr vs the shipped fp16 kernels on
   real weights (med <= 1e-2 class for int8 activations; F <= 3e-2), then
   pf_fwd32 logits med <= 3e-3 / F <= 1e-2 class, then the serving gates
   (GATE A/CTRL 60-token text, D2 160/160, tie-mine control) — the P3
   framework verbatim; acceptance (alpha spot) MUST be re-measured (int8
   prefill changes kv bytes).
6. **Do not**: int8 head without the quality gate (W0 law); .u8 IMMA (law);
   grid-stride anywhere; runtime-indexed locals.

## Files (engine0/)

pf_attn2.cu + build_p7d.py + test_p7d.py (pfaW/pfcW/pfk_pre64: gates+bench);
pf_kpre64.cu; pf_prefill.py (+PF_SUPER/PF_SC/ensure_sc/prefill_batch_sc,
default OFF); pf_fwdsc.py (the SC gate harness + probes + PF_SC_STEPS
milestone instrumentation); build_p7c.py (+nc4 tier cubins); diag_p7d*.py
(the fault forensics). Logs: ~/p7d_*.log, ~/p7e_g*.log.
