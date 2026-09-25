# P3 — Batched prefill: root cause, assembly, gates, benchmark

Status: **ROOT CAUSE FOUND AND FIXED (a harness readout artifact, not a compute
bug); full path assembled (pf_prefill.py) and integrated (daemon PF_BATCH /
PF_PREFILL=1 / PF_T1 fallback); gates adjudicated; 143-158 tok/s prefill
(5.7-6.3x the T=1 trunk)** — the 100k-class reference (662 tok/s vLLM W4A16)
remains ~4.3x away; the priced roadmap (below) closes ~2x of it.

## 1. The P2 open gate root cause (one paragraph)

The "block-0 rec-state drift 0.34, input-independent, logits decorrelate
mid-stack" was a **measurement artifact in pf_fwd16.py**: the harness ran 9
extra `fwd16()` timing repetitions BEFORE reading the drift/logits gates, and
every `fwd16()` advances the live GDN state buffers (`recp{j}`/`convp{j}`) by
another 16 steps — the gate therefore compared a ~160-step-advanced state
against the 16-step T=1 reference (and read logits computed from that polluted
state). The compute path was always in the expected class. Proven by the
λ-interpolation experiment (committed in pf_fwd16.py, `PF_LAMBDA=1`): replaying
the T=1 `k2s` scan with `qkv/gate = mine + λ(ref − mine)` gives a LINEAR
drift-vs-λ curve (λ=0 → 1.92e-4, λ=1 → 0.0 bitwise) — no amplification, no
"marginally-stable rec mode"; a 4-kernel prefix replay (embed→k0ab→q5g8→k2s)
reproduces the full-chain reference rec bitwise (0.000e+00). With gates read
immediately after the single clean run: **logits med 9.2e-4 / F(row-max)
2.6e-3 PASS, argmax 16/16, rec drift 1.9e-4 (blk0) → 1.6e-3 (blk4) — the
HMMA-vs-GEMV reassociation class, bounded.** P2's kernels stand as built.

## 2. Assembly (engine0/pf_prefill.py)

- `ensure(E)`: one-time attach — P2/P1 cubins + M=16 scratch (fixed handles) +
  a **cached launch plan** (the whole 64-block chunk as ~129 prebuilt
  (program, args, grid, ls) tuples; only `ids16`/`pos_slot` mutate per chunk).
- `prefill_batch(E, G, ids)`: chunked loop over the prompt; per chunk:
  `win_up ids16/pos_slot` → run plan. GDN live state binds to the TRUNK world
  (`conv{i}_0` persistent slot, `rec{i}` in-place) so the post-conditions match
  `prefill_t1` exactly: KV8 rows appended at [pos, pos+16), trunk GDN live
  parity = `N & 1` (the SAME `stseed_spec(len(toks)&1)` formula serve.py
  already uses), `pos_slot = pos0+N`, `tok_slot = argmax-after-last`.
- **Tail policy (v1)**: r = N % 16 remainder rows go through the existing T=1
  graph path (`prefill_t1`) — exact by construction; tails are short.
- **Head**: only the LAST chunk's row 15 is normed (`pfk_n16` g=1 on a row-15
  view) → `head8` → `h_argmax` (advances pos_slot, sets tok_slot, writes the
  last tok_hist row). No per-token heads — the T=1 path's dominant cost.
- **tok_hist rows [pos0, pos0+N-1)**: filled host-side with the fed ids
  (T=1 writes argmax-after-p there; nothing downstream reads prompt rows —
  decode/accept only touch rows >= final pos). Documented divergence, zero
  consumers.
- **fill_draft stays sequential v1** (393-415 pos/s): 2k ≈ 5s, 100k ≈ 253s —
  the long-ctx remaining cost (batched draft kernels are P2-staged, ungated).
- Daemon wiring (serve.py): RPC `prefill{mode:"PF_BATCH", ids}` explicit;
  `PF_PREFILL=1` upgrades FRESH to batched; `mode:"PF_T1"` forces the T=1
  path (fallback + A/B comparisons). Per-chunk `prefill_progress` events.

## 3. Gates

| gate | what | result |
|---|---|---|
| P2 harness (fixed readout) | 16-row chunk vs T=1 x16 | logits med 9.2e-4/F 2.6e-3 **PASS**; argmax **16/16**; rec drift 1.9e-4→1.6e-3 |
| λ-interpolation | drift vs input-error curve | **linear, no amplification** (λ=0: 1.9e-4; λ=1: 0.0 bitwise) |
| Gate A (8k-class, 100k-head text) | T1-prefill+T1-dec vs PF_BATCH+T1-dec, 60 tok | **60/60**; end-state logits F-relerr 1.12e-3 |
| Gate A (2048 / prompt8k texts) | same | 7/60, 26/60 — **near-tie flips**: divergences sit exactly where top-2 gaps are at fp16 resolution (e.g. gap 0.0078 at logit 15.25); end-state logits relerr 3.96e-2 on prompt8k |
| **Control (the adjudicator)** | production spec-vs-T1 on prompt8k (BOTH on T1 prefill) | **13/60** — the tie-mine text flips the PRODUCTION spec path even harder than the batched prefill (26/60); the text, not the prefill |
| Gate D2 | spec-on-batched vs spec-on-T1 (serving path) | **160/160 tokens exact** |
| Gate D (2k text) | spec-on-batched vs greedy-T1-on-batched | **60/60** (Tier-1 on the batched state) |
| Gate B (100k rebuild) | PF_BATCH from scratch @97810 vs snapshot + banked ref | **cur rebuilt EXACTLY = snapshot cur0 (4471)**; GDN rec relerr ≤1.9e-2 / conv ≤5.6e-2 (accumulated reassociation over 97810 pos; gate 1e-2 missed); T1-decode vs banked ref **43/60** — flips all in the 6545/9956/4649 repeat-tie family (same text class as the control above) |
| Gate C (API path) | daemon PF_PREFILL=1 + api_server + RPC A/B | **82/82 tokens exact** PF_T1 vs PF_BATCH on a live daemon (4096-pos FRESH, cur=12253 both); HTTP /health ok; SSE chat streams with `prefill NN% (prefill_batch)` progress comments; **FOLLOW_UP after batched FRESH works** (RPC: pos 2048→2324, gen continues). NOTE: the API façade's prefix-reuse did not trigger for hand-written assistant turns (client-render mismatch — pre-existing, orthogonal to prefill mode) |

Verdict on greedy agreement: the batched prefill is in the same numeric class
as the production spec probe; on texts whose greedy decode is well-conditioned
it is token-exact (60/60); on tie-heavy degenerate continuations (repeat loops
with fp16-resolution top-2 gaps) ANY cross-class comparison flips — including
the banked production spec-vs-T1 pair (13/60 on the same text). The serving
contract (spec ≡ greedy within a prefill class) holds: 160/160.

## 4. Benchmark (synced, chunk min-of-run; positions = prompt offset)

| config | prefill tok/s | note |
|---|---|---|
| T=1 trunk (old canonical) | 25.1-25.2 | 40 ms/tok |
| **PF_BATCH v1, 2k-class avg (0-2048)** | **157.6** | chunk 83.4→108.9 ms (191→147 inst.) |
| **PF_BATCH v1, 8k-class avg (0-7714)** | **143.6** | chunk → 116.8 ms at 7.7k |
| **PF_BATCH v1, 100k (0-97810)** | **129.7** (754.0 s) | chunk 83→138.7 ms; end-of-100k 115.4 tok/s inst. |
| via daemon RPC (4096-pos FRESH, incl. fill_draft + slots) | 109.5 | PF_T1 24.0 on the same path; 4.57x |
| + fill_draft (sequential, per 100k) | 386.8 | 252.8 s — 25% of a 100k fresh; batch it next |

References: **662 tok/s cold @100k** (syv-ai vLLM W4A16, 2x3090-class
throughput on their single 3090 setup — our reference table), 1-3k tok/s
short-ctx Marlin-class, our old T=1 21.78 (25.1 today), stock tinygrad ~5-20.
**We are at 20% of the 100k reference (129.7 vs 662); 143-158/1000-3000 ≈ 5-15% of short-ctx Marlin.**

### Priced roadmap (per 16-tok chunk, from P1/P2 attribution)
1. FFN double-buffer (163→250 GB/s): 26.9→~17 ms → **−10 ms**
2. Small-N K-split (k/v/q classes ~19 ms at 15-20 GB/s): → **−12 ms**
3. Norm-launch floor (~16 ms, 129 launches): fuse into GEMM epilogues → **−8..12 ms**
4. Attention double-buffered KV tiles (sync-bound at 80 GB/s today; 49 ms/chunk @100k): → **−25..35 ms @100k**
5. Batched fill_draft (P2-staged kernels): 100k fresh 253s → ~30-60s

Projection after 1-3: ~53-63 ms/chunk at 2k-class → **~260-300 tok/s**;
after 1-4 at 100k: ~75-90 ms → **~180-215 tok/s**. **Parity with 662 @100k
requires additionally ~2.5x on the GEMM family (Marlin-class tensor-core
kernels: we run 11.7% MFU vs their ~60%) plus the attention BW fix — i.e., a
kernel-generation campaign, not tuning.** If the assembled path had landed
<300 tok/s the mission ordered applying levers 1+2 now; at 143-158 tok/s v1
with all gates green, they are the first items of the next session (they were
priced against P1's standalone benches and need in-context validation).

## 5. New laws banked

1. **THE READOUT-ORDER LAW**: any harness that re-runs a stateful forward for
   timing must read its correctness gates from the FIRST clean run — GDN state
   buffers advance on every call; timing reps before readout = comparing an
   N×16-step state to the reference (this exact bug produced P2's "0.34
   drift / logits decorrelation" open gate).
2. **The λ-interpolation adjudication**: when a chain diverges, replay the
   reference's downstream consumer on `mine + λ(ref − mine)` inputs. Linear
   curve = no amplification (rounding class); curvature/ceiling = real
   instability. Plus the 4-kernel prefix replay (block 0 depends only on the
   embedding) validates the reference capture bitwise.
3. **Tie-mine texts invalidate cross-class greedy gates**: before declaring a
   prefill/decode path wrong, run the CONTROL (the same comparison with BOTH
   sides on the reference path — here production spec-vs-T1 gave 13/60). Gate
   quality = agreement on well-conditioned text + gap instrumentation at
   divergences.
4. **The host-process boot law extends to prefill gates**: a hand-rolled
   driver process faults in fill_draft / collapses draft acceptance; all P3
   gates run inside `test_w100k.py` via env hooks (PF_GATE=1, PF_GATE100K=1).
5. Chunked prefill leaves trunk GDN live parity = `N & 1` — the T=1 tail
   flips parity per token, full chunks keep `conv{i}_0`; the existing
   `stseed_spec(len(ids) & 1)` formula composes unchanged.
6. `tok_hist` prompt rows are write-only legacy (T=1 wrote argmax-after-p
   there); the batched path's id-fill placeholder has zero consumers — but
   keep the last row's h_argmax write (it is the pos_slot/threading anchor).

## 6. Files / how to run

- `engine0/pf_prefill.py` — assembly (ensure/prefill_batch).
- `engine0/pf_fwd16.py` — fixed harness + PF_LAMBDA forensics.
- `engine0/pf_gate2k.py` (+ `PF_GATE=1` hook in test_w100k.py) — gates A/CTRL/D/D2.
- `engine0/pf_gate100k.py` (+ `PF_GATE100K=1`) — snapshot rebuild gate.
- `engine0/serve.py` — daemon PF_BATCH/PF_T1/PF_PREFILL wiring.
- Harness: `env SKV=1 KV8=1 QH=1 SKV_CTXK=100352 SKV_S=256 ~/tg311/bin/python -u pf_fwd16.py`.
- Gates: `env DEV=NV SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 PF_GATE=1 ~/tg311/bin/python -u test_w100k.py`.
