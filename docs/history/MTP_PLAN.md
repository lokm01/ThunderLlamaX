# Qwen3.8-27B — 10x Plan (v3, 2026-08-24) — unified from 3 planner agents
### GLM-5.3 + Kimi-K3 + ox-alpha consensus · target ≥50 tok/s @32k, ≥25 @100k

**Reference to beat (user's M5 Max, unified memory, MLX/llama.cpp kernels):**
30 tok/s @32k · 15 tok/s @100k. The 3090 (936 GB/s peak, 447 proven) should
beat this; today it does not (7.0 @32k, OOM @100k) — the gap is software.

**Ctx-length benchmark (BEAM=1, 2026-08-24, bench_ctx.py, synthetic-fill method):**
| L | 2k | 4k | 8k | 16k | 32k | 64k | 78k | 82k | 100k |
|---|---|---|---|---|---|---|---|---|---|
| tok/s | 9.96 | 9.76 | 9.15 | 8.27 | **7.01** | 5.18 | 4.61 | 4.47 | OOM |
Fit: ms/tok = 99.3 + 1.52e-3·L (R²≈0.999). Ceiling ~82k fp32 KV (needs ~25.7GB
at 100k vs 23.62 pool). KV slope = 5.4× ideal ⇒ attn KV GEMV ~83 GB/s effective.

**The decisive math (ox-alpha):** 50 tok/s @32k = 20ms/tok, which is BELOW the
22ms weight-read floor (9.7GB @447GB/s). **Only speculative decoding crosses
it** — one weight-read per cycle amortized over 2.4-3 accepted tokens. Every
phase below either shrinks the cycle or raises accept length.

Current cost model @32k (142.75ms): GEMV floor ~22ms + GDN kernel swarm ~67ms
(subtraction-attributed!) + KV read ~48ms @83GB/s eff (5.4x ideal) + misc.

---

## P0 — Direct attribution measurement [1d] — ALL 3 AGENTS INSIST THIS IS FIRST
Per-kernel timeline histogram of one steady decode token (HCQ signal timestamps
→ kern_hist.py): sum-of-kernel-time vs inter-kernel-gap split, top-10 kernels,
true attn-KV times. Plus two same-day microbenches: (a) KV GEMV naive vs
split-K/stacked @32k (need ≥300GB/s to hit P2 numbers); (b) one hand-fused GDN
block T=1 elementwise chain (sets P1 floor; target ≤150µs/block).
EXIT: pie chart of the 99.3ms floor with <10ms unexplained. If swarm <30ms,
P1 de-prioritized and plan pivots to KV/custom kernels first.

## P1 — GDN T=1 scan fusion [4-8d] [floor 99→36-40ms ⇒ ~26 tok/s @2k, 11-16 @32k]
`model.py GatedDeltaNetBlock._attention` (L281-343):
1. Kill `win` uop.after/store window → `conv_state.cat(rows)` (proven pattern
   in `_attention_mtp`); move conv_state store off the critical dataflow.
2. T=1: collapse recurrence to ONE expr tree (no mid-chain .contiguous()/
   .float() splits; fp32 casts hoisted); recurrent_state store LAST.
3. Target ~23 → ≤4 kernels/block (~1100 → ~200 total).
Fallback ladder: A2 per-block @function CALL units → A3 custom CUDA scan
kernel (nvcc shim proven; dispatch external cubin via dext ioctl) +5d.
Gate: greedy-identical vs banked baseline. BUDGET: overnight BEAM marathon
per phase (cache invalidated; backup cache.db first — 2 corruptions already).

## P2 — KV read tax [2-4d] [@32k ≈ 44-47ms ⇒ 21-27 tok/s; ctx ceiling ~130-178k]
1. **fp16 KV** (syv: PPL-neutral) — halves 128→64KB/token; 100k fits.
2. **Kill the 83GB/s kernel**: split-K over ctx chunks (GLM/Kimi) AND/OR stack
   the 16 layers' K/V into one contiguous big GEMV (ox-alpha: 2×1.07GB @
   ≥350GB/s ≈ 6ms @32k). P0(b) microbench decides which.
3. **Drop the duplicate 2.54GB MTP head** (draft already shares main head) —
   free VRAM, do this in P0 alongside (glm53: trivial, zero risk).
Gate: greedy-match; logprob drift check for fp16 KV.

## P3 — MTP v3 no-commit, ONE graph family [5-8d] [@32k 43-52 tok/s]
- Extend P1's fused scan to emit **per-step states as kernel OUTPUTS** (stacked,
  never mid-scan stores — resolves review D6); rollback = feed states[m+1]
  tensors as next jit INPUTS (D9-safe, no uop.store restores).
- ONE JIT=1 TinyJit family containing draft chain + T=K+1 verify (v212-proven
  single-family viability; replay overhead irrelevant at 0.072ms).
- Concrete-T everywhere (symbolic slices proven broken); pad-to-T_max if needed.
- Draft: blk.64 0.85GB ≈ 1.9ms/step in-graph; 40k vocab slice (97.5% coverage)
  → head 0.42GB ≈ 0.9ms/step. Draft ≤3-5ms/step is THE number that decides 43 vs 52.
- Cycle @32k K=3: verify 38+8+2 ≈ 48ms + draft ~12 + heads ~4 ≈ 61ms; accept
  len 2.4-3.0 ⇒ **42-51 tok/s**. @100k: +16ms fp16-KV ⇒ cycle ~66ms ⇒ **~25-28 tok/s**.
- RISK (HIGH, ox-alpha): three unproven mechanisms (per-step outputs, symbolic
  pos in family, single-family with 96 state args) → iso_cycle harness FIRST.
- Add temp>0 exact rejection sampling (Leviathan) or quality drifts silently.
- Boot-time assert: count live HCQGraphs == 1 (stray @function qualname
  collision has bitten before → MAP_SYSMEM_FD IndexError).

## P4 — Cost shavers, each env-gated [1-2d each] [@32k 55-65 tok/s cumulative]
1. int8 g128 lm_head requant (2.54→1.27GB, −2.8ms/cycle)
2. K-sweep 2-5 at new cost profile (K=4 + slice ⇒ len ~3.2)
3. fp16 GDN states (~150MB traffic halved)
4. sort-free top-k sampler
5. lookup/prompt-replay drafting (doc-grounded regimes)

## Trajectory @32k: 7.0 → 11-16 (P1) → 21-27 (P2) → 42-52 (P3) → 55-65 (P4)
## @100k: OOM today → fits after P2 → ~25-28 after P3+P4.
Checkpoints vs M5 Max: @32k beat it after P3 (42-52 vs 30); @100k beat it after
P3+P4 (~25-28 vs 15) — and only once 100k fits (P2).

## Fallback if P1 A1/A2 stall entirely
A3 custom CUDA scan (in-plan). Beyond that: hardware move to Linux box
(VeroFess llama.cpp: 64 tok/s proven on this model) — plan-B only.

Consensus divergences kept: kimik3's int8 head/embed requant (traffic 10.1→8.3GB)
folded into P4; ox-alpha's stacked-KV variant as P2 option; glm53's head-skip
promoted into P0. All three agreed P0 attribution must run before any P1 code.
