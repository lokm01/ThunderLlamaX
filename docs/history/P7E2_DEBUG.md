# P7-E2 — The super-chunk NaN forensics: THREE root causes found, one fixed-and-validated, SC still multi-chunk-blocked

Status: **the P7E "post-reference all-NaN" is DECOMPOSED into three real bugs.
(1) pfk_pre64 window-indexing [FIXED, cubin rebuilt, single-chunk SC now
EXACT vs M32]. (2) gemm3-m64 at M=256 corrupts memory [SC now runs
PF_GEMM3=0; m64-at-M256 validation still pending — the P7E doc's warning was
RIGHT and its "gemm3 falsified" conclusion was an artifact of bug (1)].
(3) pf_fwdsc's T=1 reference used `E._seq[0]` instead of the GDN parity
ping-pong `E._seq[t & 1]` [FIXED] — the harness ref computed garbage, making
the P7E-era gates unreadable. A FOURTH issue remains OPEN: multi-chunk SC
(chunk >= 2) device-faults / diverges — THE next-session blocker. The daemon
ships the P6 M32 canonical (PF_SUPER=0). MTP_KERNARGS_MB=256 added to the
fork (eager kernargs ring 32->128MiB) as launch-volume safety.**

## The forensic chain (how the NaN was run down)

1. Milestone-diff harness (pf_scdbg.py: crc+nan snapshots at launch
   milestones, fresh vs post-ref, dense 41-96) -> first divergence at
   launch 52 = the FIRST pfk_pre64 (qw buffer), all inputs/outputs
   bit-identical.
2. No-probe truncation ladder (SCDBG_TRUNC=K, end-state dumps) on the FIXED
   pre64: qw NaN appears deterministically at launch 226 (block-10 pfcz)
   ONLY when gemm3-m64 kernels are in the plan; skip-tests + bit-histograms:
   qw = uniform 0x7FFF fp16 fill (3MB, exactly zsc-sized), re-upload test
   proved the mapping intact -> a REAL write at a wrong VA, size-matched
   (3MB qw <- pfcz-zout-shaped record; 6MB qrowsc <- sco-shaped). Signature:
   kernels executing with STALE/foreign pointer sets at super-chunk launch
   volumes. gemm3-m64@M256 is the trigger family (its qrow/qkv writes at
   M=256 were observed landing elsewhere: NaN counts identical across the
   launches that should have rewritten them).
3. With pre64 fixed AND PF_GEMM3=0: single-chunk SC end state CLEAN (rec/conv
   0 NaN, logits 0 NaN) in fresh AND post-ref processes.
4. Stage-diff vs the P6 M32 path (pf_scdbg3.py, same world, same ids):
   **SC == M32 at every stage (emb/x/xh/qrow/qw/ao/attn_out/hh/qkv/gate/z/
   gact all medrel ~1e-3 fp16-class); logits med 5.2e-4 / F 5.6e-4; argmax
   271 == 271.** The SC math is CORRECT at one chunk.
5. pf_fwdsc gates still "FAIL" (med 2.7) -> the T=1 REFERENCE was wrong:
   t1_token ran E._seq[0] every token (no GDN conv ping-pong parity);
   pf_fwd32 (P6, passing) uses E._seq[t & 1]. Fixed. (The fixed 256-token
   in-harness ref still diverges after ~32 tokens in the fwdsc world —
   harness-context artifact; the SERVING T=1 is the proven reference.)
6. Serving gates (PF_GATE=1, PF_SUPER=1 PF_GEMM3=0, 7714-pos prompt = 30 SC
   chunks + 54 tail): GATE A 0/60, end-state logits F 5.07e-1, decode
   degenerate (22525 loop) — the 30-chunk state is WRONG. GATE D 60/60,
   D2 0-mismatches/160 (spec machinery consistent). Bench: 30 chunks in
   34.6s = 222.7 tok/s prefill-class (execution real, state wrong).
7. Multi-chunk isolation: N=512 fresh-process SC-only -> chunks run (2.27s)
   then the post-run DOWNLOAD faults (P5-class); N=512 after an M32 pass in
   the same process -> device fault INSIDE chunk 2. Multi-chunk SC is
   unstable independent of reference/history -> BANKED as the blocker.

## Bug (1) detail — pfk_pre64 missing window offsets (the certain fix)

pf_kpre64.cu indexed qrow16/krow16/vrow16 reads AND the qw16 write by the
LOCAL window row t; kv/sc appends correctly used pos_arr[ywin]+t. Effect at
NC=4: all four window-groups raced on qw rows 0..63 (PCIe interleaving ->
nondeterministic garbage incl. NaN bit patterns) and rows 64..255 were NEVER
written (zeros). pfa16's per-row q-norm turns zero rows into rsqrt(0)=inf ->
0*inf = NaN -> aosc -> x -> every GDN rec from the next block on. Blocks 3/7
"survived" only because their pairs ran before the late garbage landed.
Fix: `const int g = ywin * TROWS;` + (g+t) on the four sites. Rebuilt via
build_p7d.py (symbol check clean, 40 regs, 3072B smem, 0 spill). NOTE: the
P7d standalone validation passed because it ran grid (24,1) = ONE window —
the multi-window path was never exercised (doc law #7 over-generalized).

## Bug (2) detail — gemm3-m64 at M=256

Evidence: with PF_GEMM3=1 the deterministic launch-226 wrong-VA 3MB 0x7FFF
write + qrowsc writes not landing; with PF_GEMM3=0 both gone. The m64 kernels
were standalone-validated at M=64 ONLY (P7B); the SC calls them with
MP64=4 m-passes over full 256-row buffers. SC plan now must run
PF_GEMM3=0. (The M32/fwd32 serving path keeps gemm3 — P7B-gated at M=32.)
Separate env for the SC side recommended next session (PF_G3SC), then
validate m64 at M=256 vs 4x m32 (the P7E TODO stands).

## The MTP_KERNARGS_MB knob (fork, ops_nv.py)

Eager kernargs ring was 64MiB total = 32MiB eager (wrap-bump, NO in-flight
tracking). SC-scale launch volumes (a 256-token T=1 reference = ~12.8k
launches + 2485-launch trains) can push the host one ring-past the GPU while
the 2MB cmdq still permits ~20k launches of lead — the exact MTP-era
"kernargs wrap clobbers pending records" class. Not proven to be THE bug-2
mechanism (MTP_CMDQ_DIAG showed 0 cmdq wraps), but the margin is wrong at
these volumes: `kernargs_size=int(getenv("MTP_KERNARGS_MB","256"))<<20`
(eager 128MiB ~ 52k launches > any cmdq-permitted lead). Default 256;
service launched with it explicit.

## Benchmark table (P7E2 state)

| class | P6 record (shipped) | SC P7E2 | note |
|---|---|---|---|
| 2k harness pos-0 | 245.0 tok/s | **298-300 tok/s** (858ms/256-tok chunk, GEMM3=0) | single chunk, EXACT vs M32 |
| 8k-class gate bench (7714 pos, incl. tails) | 188.7 tok/s (40.9s) | 222.7 tok/s (34.6s, 30 chunks) | STATE WRONG — speed only |
| 100k rebuild | **165.3 tok/s** | not run (blocked) | P6 stands |

SC chunk ms by position (gate bench): 870ms @0 -> 1170 @2.5k -> 1220ms
plateau @4k+. Attribution per 256-tok chunk: non-attention stack ~800ms
(GEMMs classic-m32 x8 + M-wide norms + chunked scan + launches ~1900/chunk);
attention (shipped pfa16 pair) grows with L — P7D: 5.85ms/32tok/layer at
L=100352 => 749ms/chunk at 100k => projected ~1.55s/chunk = ~165 tok/s at
100k (matches the P6 record — the model is consistent).

## P7f attention spec numbers (updated with P7E2 attribution)

At 100k ctx, 256-tok chunks, non-attention ~800ms (GEMM3=0; ~650-700ms if
m64-at-M256 is fixed):
- **300 tok/s**: chunk <= 853ms -> attention budget ~50-200ms depending on
  the GEMM tier => the HMMA pair (749ms) must drop 4-15x => IMMA/dp4a int8-QK
  (P7f spec item d) is REQUIRED; realistic IMMA 3x => ~250ms attn => ~244
  tok/s at today's non-attn; with m64-fixed ~285.
- **400 tok/s**: chunk <= 640ms => attention <= ~150ms AND non-attn <= 490ms
  => P7f classes (a)-(c) (int8 W8A8 FFN/gdnqg/o-proj + m64) + IMMA QK.
- **600 tok/s**: chunk <= 427ms => attention <= ~80ms (IMMA at >=9x the pair,
  i.e. the full m16n8k32 route with kpre int8 bias-corrected KV) + non-attn
  <= 350ms (full int8 tier).
Priority per this session's numbers: fix m64-at-M256 first (free 150ms/
chunk), then IMMA QK (the 100k binder), then the int8 GEMM tier.

## Service state

Daemon on the P6 canonical: M1C recipe + PF_PREFILL=1 PF_GEMM3=1 PF_SCANC=1
PF_SUPER=0 MTP_KERNARGS_MB=256. PF_SUPER stays default-OFF in pf_prefill.py
until the multi-chunk blocker is solved.

## Next session (ordered)

1. Multi-chunk SC: bisect INSIDE chunk 2's plan (truncation ladder with
   pos0=256 — chunk 2's own launch 226-equivalent); suspects: pfk_pre64 at
   pos_arr=[256..448] (kv/sc append at row>=256 with the CTXK slab stride),
   the classic m32 twin windowed views at chunk-relative offsets (they are
   chunk-relative already — verify), pfca/pfcb/pfcz on live rec/conv across
   chunk boundaries (state carry), pfa16 pair at pos_w=256+16k over kv rows
   256+.
2. gemm3-m64 at M=256 validation (P7E TODO) + PF_G3SC split env.
3. fwdsc's 256-token T=1 ref divergence after ~32 tokens (harness artifact —
   understand or cap the harness ref at 32 + reuse fwd32's).
4. Then the full gate ladder + benchmark re-run on the SC.

## Files

- engine0/pf_kpre64.cu (+ .bak_p7e2) + pfk_pre64_100k.cubin (FIXED)
- engine0/pf_fwdsc.py (t1 parity fix)
- engine0/pf_scdbg.py / pf_scdbg2.py / pf_scdbg3.py (forensic harnesses:
  milestones, truncation ladder, skip-tests, end dumps, M32 stage-diff)
- ~/tinygrad-src/tinygrad/runtime/ops_nv.py (MTP_KERNARGS_MB)
- Logs: ~/p7e2_*.log, scdbg_*.jsonl(+.plan), tr_*.npz, sd512.log
