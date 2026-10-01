# Architecture — the engine, the cycle, the kernels

*Plain language first, then the deep end.* Each section below opens with a
paragraph a smart developer can follow without GPU expertise, then keeps the
full technical record. For the flavor names: the engine is the DraftHorse,
the driver layer is ThunderSilicon, the prompt cache is LongMemory — the real
component names are `engine0` (published as `engine/`), the dext, `pcache`.

The engine (`engine/`, originally `engine0/`) is a from-scratch decode loop
that now serves **three models**: the original **Qwen3.8-27B** hybrid (48
gated-delta-net blocks + 16 full-attention blocks + IQ3_S embedding + Q5_K
output head + one MTP draft block), since the multi-model campaign the
**Qwen3.6-35B-A3B MoE** (30 GDN + 10 full-attention blocks at hidden 2048;
every layer routes top-8 of 256 experts plus one shared expert — ~35.1B
parameters total, ~3B active; its own `blk.40` MTP layer is the shipped
first-party drafter), and an abliterated weight-variant of the dense
checkpoint (same architecture, weight-only swap through the registry —
[DEPLOY_OBLITERATED.md](DEPLOY_OBLITERATED.md)). All three bypass tinygrad's scheduler entirely: static raw
buffers, ~330 (dense) + ~60 (MoE) hand-CUDA kernel sources compiled to
per-kernel cubins, and a graph-replay submit path. tinygrad is used only as
the *driver runtime* (NVProgram/TinyELF/NVComputeQueue from the fork — see
`patches/`). On top of the decode loops sit the deep-K LOOKUP speculative
layer, the batched-prefill pipeline, the durable prompt cache and the
multi-model serving layer (sections below; [history/PREFILL.md](history/PREFILL.md)
and [SERVING.md](SERVING.md); the MoE campaign plan is
[history/MM_PLAN.md](history/MM_PLAN.md)).

## How eGPU LLM inference works on Apple Silicon

*The driver story in one section — because "is there an NVIDIA eGPU driver
for macOS?" is the question this whole repo answers, and almost nothing is
published about what replaces the missing stack.*

On Linux, an NVIDIA GPU is driven by a stack roughly three layers deep: a
kernel-mode driver, a userspace driver (libcuda etc.), and on top the CUDA
runtime that frameworks call. macOS has none of these for modern NVIDIA
GPUs — no web driver since 2018, no CUDA toolkit target, and no eGPU support
at all on Apple Silicon. Thunderbolt correctly wires the card's PCIe into
the Mac; nothing on the macOS side knows what to do with it.

What this stack substitutes, layer by layer:

- **The kernel-adjacent layer → a DriverKit system extension (the "dext",
  the ThunderSilicon layer).** DriverKit is Apple's sanctioned
  userspace-driver framework. The dext maps the GPU's PCIe BARs into the
  engine's address space and exposes a queue RPC: the *userspace* builds
  QMDs (queue meta descriptors = work items), pushbuffers, and semaphore
  releases, writes them into device-mapped rings, and rings a doorbell. No
  kernel extension, no NVIDIA code, fully inspectable.
- **The userspace-driver + runtime layer → the tinygrad fork's NV backend.**
  NVProgram loads the hand-written kernels' cubins as bare ELF objects
  (TinyELF); NVComputeQueue owns the command queue, submits through the
  dext, and does the timeline-signal waits. There is no CUDA runtime
  anywhere — no `libcuda`, no `cudaLaunchKernel`, no context management
  beyond what the engine itself does.
- **What the kernels are: ordinary CUDA, compiled ahead of time.** The
  hand-written `.cu` sources compile with `nvcc -arch=sm_86 -cubin` inside a
  Docker container (build-time only). What runs on the Mac at inference time
  is the cubin ELF + this driver path — the same SASS a Linux box would
  execute.

Two measured facts that define the platform (full laws in
[DEXT_LAWS.md](DEXT_LAWS.md)): through this driver path the 3090 sustains
**880 GB/s streaming and 843 GB/s GEMV — 94% of the card's peak** (the
driver is not the bottleneck; every host roundtrip costs ~17 ms of
Thunderbolt latency, which is why the engine is built to touch the host
~once per token *cycle*, not once per kernel), and the dext imposes its own
hardware-adjacent laws — hard 1 CTA/SM, 48 KB static smem cap, cp.async
faults, name-encoded launch dims — that shape every kernel design decision
in the rest of this document.



## The MTP cycle

*Plain language: the engine guesses K tokens ahead, checks all guesses in one
batched GPU pass, and keeps however many were right. Because the checking
pass computes each row exactly like a single-token pass would, the guesses
can never corrupt the output — the emitted stream is identical to plain
greedy decoding, just faster.*

Canonical numbers @100k, K=2 MTP (W2H: 68.98 ms -> 40.35 tok/s):

| phase | ms | kernels | what happens |
|---|---|---|---|
| draft (2 steps) | 5.24 | 29 | eh_proj([enorm(emb(prev)), hnorm(hm)]) -> draft block (own KV) -> head_norm -> 40960-row slice head -> slice argmax -> dring |
| probe T=3 | 62.53 | 452 | feed [cur, dring0, dring1] at [pos, pos+1, pos+2] through the M=3 trunk; per-row argmax -> amds |
| accept | 1.14 | 2 | m = longest prefix of dring matching amds; emit amds[0..m]; pos += m+1; cur = amds[m]; h_seed = probe hidden row m; GDN state slot m -> live slot |
| flusher | ~0 | 1 | 1-kernel graph that keeps the submit chain legal (lone-graph law) |
| total | 68.98 | 484 | 4 graph submits; zero host work inside the loop |

alpha(pos) = 0.892, 2.78 tokens/cycle. T=1 (non-spec) reference: 45.92 ms/tok.

**Emission contract** (identical to the tinygrad-stack mtp_v3 semantics): each
cycle emits the accepted probe argmaxes `amds[0..m]`; the bonus token
`amds[m]` becomes the next cycle's `cur` and is never fed twice; the state
invariant is `states == f(tokens[0..pos-1])` at cycle boundaries. Draft KV for
rejected positions is always overwritten by the next chain (m+K >= K).

**Tier-1 exactness proof**: every M=3 kernel replicates the T=1 kernel's
per-row floating-point op order exactly (same loads, same mul/add chains, same
shuffle trees) -> probe rows are BIT-IDENTICAL to what T=1 would compute ->
the m-prefix selection can never flip a near-tie -> spec output == greedy T=1
output, 60/60, deterministic. Additionally: engine T=1 matches the stock
tinygrad greedy baseline 59/59 at 100k, and each optimization (int8-KV, half2,
HMMA) was gated for ZERO greedy flips vs the previous sequence.

## The deep-K LOOKUP cycle

*Plain language: when the model is re-reading text it has already seen —
documents, code, quotes — you don't need a neural network to guess what comes
next. The engine scans its own token history for the current 8-token suffix
and, if it recurs, proposes the actual continuation from history, up to K=10
tokens at once. A batched verify pass accepts or rejects; on this workload
class, hits accept ALL K proposals (measured, six rungs in a row). On novel
prose the drafter never fires and the engine falls back to the K=2 cycle —
whose draft head is full-vocab since P10 (next section).*

K=8 shipped config (104.65 ms -> 71.51-72.02 tok/s @100k); K=9/K=10 rungs
(74.70/75.56 tok/s, same construction — kernel sets pending the next engine
republish, see [history/R8_DECODE.md](history/R8_DECODE.md)). The K=2 MTP
cycle above remains the fallback path; the shipped config selects PER CYCLE
between it and a deep-K cycle:

- **The n-gram drafter (`lookup*_nw32.cu`)**: on a device-side scan of the
  token history (`tok_hist`) for the current suffix (LMIN=8, deep-K
  scan-range law `iend = pos-8-K`), a hit fills the ENTIRE draft ring with
  the continuation — no draft-model forward at all. A hit flag rides the
  per-cycle emit record.
- **Per-cycle graph-set selection**: `DecodeSession` submits either the K2
  set (draft_g -> probe_g -> accept_g) or the deep set (lookup_g ->
  probeT{K+1}_g -> accept{k}K_g) based on the previous cycle's hit flag.
  Sel-mode beats all-deep at every K (K=7: 63.11 vs 59.03) — the T=K+1 probe
  is only paid when the drafter actually hit (76.7% of cycles at K=10; every
  hit accepts ALL K: E[m|deep]=K exactly, alpha 1.0 in-vivo at every depth).
- **The draft-skip (R7a)**: on deep cycles (previous emit was a hit) the
  2-step MTP draft chain is pure waste — the lookup overwrites dring0/1 — so
  the deep set runs a lookup-only draft graph. Output-lossless by
  construction (amds-verified commits; the K2/miss cycles keep the full draft
  graph); the emit stream is byte-identical to the pre-skip engine.
- **The M=5..9/T=5..9 kernel sets** (`gen_m5.py`..`gen_m9.py` generators;
  the rig's K=9/K=10 rungs extend the same pattern to M=11/T=11): each deep
  rung adds mechanically generated batched variants of every trunk kernel —
  launch-bounds-preserved renames, audited row-stores on every write family
  (the R5a bug class: three stacked gen_m5 M-extension slips, all row-4-only),
  per-K `accept{k}k`/`acceptsel{k}k` (layout-aware emit word count),
  ROWS=5..9 attention (`spk_pre{n}qh`, `spk_g4nw32hm{n}` RMAX pad rows,
  MAXOWN owner paths), and `k2s{n}` scan with the REC-CHAIN SLOT LAW (t=K
  reads rec{n}x scratch, never a live slot).
- **The norms CTA fix (R7a rung-3)**: the norms/emb kernels (h_embed8/k0n8/
  hh8, k0ab8 + the _3 twins) run per-row CTAs (grid 8/104/3/39) with
  verbatim lane math — the one-CTA serial-t-loop versions cost ~Mx their
  parallel time (the CTA-serialization law, +5.43 tok/s, bit-identical).
- **Host-side**: `serve.py` `seed_hist` seeds `tok_hist` on
  boot/FRESH/FOLLOW_UP/CACHE_HIT/snapshot paths — without it the LOOKUP
  drafter self-matches a -1 prefix and faults (the tok_hist seeding law, R5d).
- **Exactness**: deep=off (K2 superset), deep=on, and sel modes are each
  Tier-1 60/60 x2 deterministic + stock 59/59 at every K rung; the
  R4_TRACE/R4_DIF/R4_BISECT harness triad (env-gated in `test_w100k.py`)
  proves all probe rows bit-identical to the T=3 path.

## The EAGLE prose drafter (P10-dense: the full-vocab draft head + the gated K=4 chain)

*Plain language: the dense checkpoint ships its own one-block draft model —
a 65th layer that exists purely to predict ahead. The engine was already
running it, but its proposals were filtered through a 40,960-row
"most-likely-tokens" shortcut head that misses ~9 of 10 novel-prose
targets. Pointing the drafter at the real, full-vocabulary output head —
the same weights the verifier reads, zero new memory — doubled prose speed.
A 4-deep version of the same chain was built and verified but ships off:
the 1-layer drafter can only reliably predict 2 tokens ahead, so the deeper
verify pass costs more than it recovers.*

- **The EAGLE nextn finding**: the dense checkpoint carries a first-party
  MTP layer — `blk.64` holds the full
  `nextn.{eh_proj,enorm,hnorm,shared_head_norm}` block plus a complete
  attn+FFN block (`qwen35.nextn_predict_layers=1`, block_count=65). The
  W2-era 2-step EAGLE chain (`_draft_entries` + the kv_d draft KV + the
  fill_draft prompt-fill pass) had been running it all along; prose
  acceptance was zero because of the HEAD, not the chain — the rank
  histogram measured **89.8% of novel-prose targets OUT-OF-SLICE**
  (top1==target 6.1%) under the 40960-row prompt-frequency slice, so K2
  cycles accepted ~0 and the alpha-death controller parked decode in T=1.
- **Draft-head plane reuse (zero new VRAM)**: `sheadf.cu` is the Q5_K
  slice-head GEMV verbatim at VOCAB=248320 — every chain step proposes over
  the SAME resident `W[("head",0)]` plane the trunk probe head reads; the
  draft scratch is logits3 row 0, dead at draft time by timeline order (the
  probe head overwrites rows 0..RM-1 before any probe-side read).
  `samxf.cu` is the single-row full-vocab argmax writing the winning row
  index DIRECTLY as the token id (full vocab — no id-table gather). Under
  the knob (`TLX_DHEAD_FULL=1`, dense default) the alpha-death T=1 entry is
  suppressed: a T=1 episode can never exit on novel prose (the exit needs
  an 8-gram hit), so entering it parks decode at the T=1 rate for the rest
  of the generation. `fill_draft` skips the head+argmax pair (prompt fill
  never consumes the draft argmax). Result: prose 23.0 -> 43.7 tok/s GSM8K
  median through the API; quote-class untouched (~75).
- **The K=4 graph set (`TLX_EAGLE_K=4`, gated; default 2 = rung-1
  bit-exact)**: a 4-step draft chain over DEDICATED per-step hidden buffers
  (hd_d0=h(cur), hd_d1=h(dring0), hd_d2=h(dring1), hd_d3=h(dring2) —
  `accept5e` maps m to the exact chain hidden for m<=3), the R4 M=5 trunk
  probe (T=5; FFN/down via the `ffn8v5r7`/`down8nw32v5r7` r7 twins — the
  packed7 plane under PF_DR7, 50/64 regs 0 spill), `accept5e`/
  `acceptsel5k` accept, and a prose hysteresis in DecodeSession:
  `TLX_EAGLE_PROSE_TRIG` consecutive zero-hit miss cycles ENGAGE the K4
  set, any hit disengages, 0 = K4 always (the gating mode). Quote-class
  deep hits keep the K=10 deep set untouched; the K2 and deep sets are
  untouched by the scaffold (knob-off is byte-identical to rung-1).
  Tier-1 proven (60/60 x2 + 59/59 stock) — and honestly falsified on
  performance: the 1-layer drafter saturates at m=2 (m=3 fires 5% of prose
  cycles, m=4 never), so K=4 measured -25% prose and ships OFF pending a
  deeper drafter (verdict + break-even math in
  [PERFORMANCE.md](PERFORMANCE.md#p10-dense-the-prose-rungs-rung-1--the-full-vocab-draft-head-rung-2--the-k4-verdict)).

## The MoE engine (Qwen3.6-35B-A3B — four new kernel classes)

*Plain language: a mixture-of-experts layer is a router that picks, for
every token, eight small expert networks out of 256, plus one always-on
shared expert. Doing that on this dext meant four genuinely new kernel
shapes — a deterministic router, a gather-GEMV that walks expert slabs in a
fixed order, a combine that sums partials in a fixed rank order, and a
split-KV attention retargeted to the model's head geometry. Determinism is
the load-bearing constraint throughout: the router must pick exactly the
same experts, in the same order, every time — or bit-exactness dies.*

The MoE port reused the dense engine's laws and machinery (split-KV, the
GDN scan family, accept/commit, the graph discipline, pcache) and added
four kernel classes plus a first-party MTP chain (campaign journals
[history/MM_P0_results.txt](history/MM_P0_results.txt) ..
[history/MM_P10_results.txt](history/MM_P10_results.txt)):

- **Router — `rt8e256.cu`**: the 2048->256 router GEMV in fp32 with a
  FIXED-ORDER top-8 (8 masked-argmax passes, tie -> lower id — no runtime-
  indexed locals), softmax -> top-k -> renorm -> cast in the template's
  exact op order, plus the shared-expert sigmoid gate. Emits the canonical
  (expert-id, row, gate) pair list — NO sort anywhere.
- **Gather-GEMV — `gx8e256up`/`gx8e256dn` (+ `up4`/`dn6` quant variants)**:
  grouped GEMVs that walk the selected pairs FLAT and SEQUENTIALLY (no
  grid-stride — the dext's gridDim=0 law), reading 16B-aligned per-expert
  packed slabs through a device-side expert-pointer table (VA pointer
  tables are legal on this dext — the P0 law). The up kernel carries the
  SiLU*gate epilogue; the down kernel stores fp16 partials per (pair, rank).
  The measured gather tax vs dense GEMV is ~0 — the MoE decode weights
  stream at the same class of bandwidth.
- **Combine — `mx8e256cmb`**: fixed-rank-order fp32 combine of the routed
  partials x gates PLUS the always-resident shared-expert FFN — a fixed
  summation order is what keeps the MoE layer bit-stable.
- **Split-KV attention — `spkq256`/`spka256`/`spkc256`** (with the
  warp-per-position `spkq256m`/`spkq256s` variants): retargeted to the
  model's head_dim 256 / GQA 16:2 geometry with the output sigmoid-gate
  fused into the combine epilogue; the P7 warp-per-position rewrite removed
  the O(L) score wall (4x at 16k).

Around them: the `k2s36`/`gconv36` GDN scan at the 2048-hidden dims (the
"36" = the 30-GDN+6-attn slot layout of the rescaled state), `h6kam` (the
fused head-argmax that folds the vocab projection and argmax into one
kernel), `acc36`/`selc36` (the ON-DEVICE accept + IN-GRAPH commit — the
host fold, below), `embg248` (the 248320-vocab embedding gather with the
h-rot clamp), the RMSNormZeroCentered `(1+w)` norm variants, and
`rowcp2048` (the MTP chain's hidden carry).

**The host fold (P7)** — the reason the MoE engine feels like the dense
one: accept (`acc36`), commit (`selc36`) and the per-cycle control/emit
readback all run on-device; the host sees the engine through an UNCACHED
sysmem `cpu_view` window (bidirectionally coherent, free by A/B). Host
residue is ~0.2 ms/cycle (was ~14 ms through six full-sync copies — the
FULL-SYNC COPY law).

**The MTP K=4 chain (P9/P10)** — the first-party drafter: the model's own
`blk.40` MTP layer (eh_proj + a full-attn+MoE layer + the SHARED output
head), repacked into the engine pack (routers dequantized to F32; experts
kept at their Q3_K/Q4_K quants with verbatim-ggml dequant ports). The chain
runs one cycle late (the EAGLE/llama.cpp shape): after each probe's accept,
`rowcp2048` copies the committed hidden into the chain's H0/H1
double-buffer, a TRUE-pair seed step runs at the committed position (no
head — its c1 already knows the boundary), then four head steps draft the
next cycle's proposals. The KV invariant: every row the chain reads was
written by a true seed, an accepted draft's own step, or the chain's own
earlier steps — rejected rows are overwritten by the next seed before any
read. Lookup cycles mark the chain STALE (the quote path stays untouched);
the first miss after a hit streak pays one T=1 re-anchor. Acceptance:
depth-1 = 0.875 exactly; E[acc]@K4 = 2.58-3.08 (see
[PERFORMANCE.md](PERFORMANCE.md)).

**The mode mix**: a scan oracle picks per cycle — `n>=8 -> D8` (K=8
lookup), `n>=4 -> D2`, miss+live-chain -> P5+MTP, miss+stale -> one T=1
re-anchor then seed. `force_mode` (`t1`|`spec`|`mtp`) overrides for gates.

**What was tried and honestly falsified on the MoE path**: the pairwise
kernel fusion (P10-B — bit-exactness diverged at token 27; `MM_FUSE2` stays
off, and its launch-count arm corrected the launch-serialization law — see
[DEXT_LAWS.md](DEXT_LAWS.md)); the K=8-variant graph budget at 96k (the
<=8-variants-per-model law from the plan held); the 96k CPU anchor (the
16 GB host RAM wall — anchors stream now).

## Kernel inventory (engine/*.cu, the load-bearing families)

| kernel(s) | role | key technique |
|---|---|---|
| `h_embed*`, `h_argmax` | embedding gather + tie-safe argmax + token/pos slot write | IQ3_S gather-dequant from device token slot; block argmax; pos_slot++ on device |
| `k1_q5`/`q5g8*` (q5g8v_3) | GDN qkv (Q5_K 10240x5120) + gate fused | warp-per-row dequant GEMV; lane owns 4 consecutive W bytes; register scale tables |
| `k1_iq3`/`ffn8*` (ffn8v_3) | FFN gate+up (IQ3_XXS 2x17408x5120) | aligned-repacked IQ3 rows; half2 cores (ACC3H2) |
| `k3c_down`/`down8*` (down8nw32_3) | FFN down (17408->5120) | fat 1024-thread CTAs ("nw32" name token) |
| `k2s*` (k2s3/k2s4) | GDN conv window + delta-rule scan + z gate, all T rows | single fused kernel replacing the ~18-kernel tinygrad scan chain (the 51x scan win); per-row sequential op order |
| `k0ab*`, `k0n*` | RMSNorm + alpha/beta fp32 GEMVs | redundant per-warp norm to skip a barrier |
| `op38*`/`ao8*`/`k3ao*` | o_proj / attn out (Q8_0 or IQ3_XXS by block) | type-dispatched (24 of 48 GDN blocks are IQ3_XXS, not Q8_0) |
| `a_q6`, `a_kv`, `a_attn`, `a_o` | attention block GEMVs + T=1 attention | qk-RMSNorm + PARTIAL RoPE (rope_dim=64, theta 1e7) + fp16 KV append + gated online softmax, GQA h->h/6 |
| `spk_pre{1,3}` (+`qh`,`q` variants) | split-KV phase 1: q-norm + partial RoPE + int8 KV quantize-on-append | biased u8 = q+128 + per-row 32-ch fp16 scales; qw16 fp16 mirror output |
| `spk_g4nw32qh{1,3}` / `spk_g4nw32hm{3,1}` | split-KV phase 2 (THE attention kernel): QK + online softmax + PV | G4 fat-CTA (1024 thr, 32 warps), S=256 splits, GQA-shared KV (one read serves 6 q-heads x 3 rows), smem-staged tiles, half2 QK/PV dots (HFMA2) or `mma.sync.m16n8k16` HMMA; int8 dequant via `PRMT 0x6400` trick |
| `spk_c{1,3}` | split-KV phase 3: fixed-order combine + sigmoid gate | sequential per-split merge, fp32 partials |
| `head8*` (head8v_3) | output head GEMV (Q5_K 248320 rows) | 874 MB at memory floor; half2 |
| `m3*/m4*` family | the full M=3/M=4 probe trunk (batched variants of every trunk kernel) | bit-identical per-row op order (the Tier-1 contract) |
| `q4v` (ehproj/dq/doproj/ddown), `mtpd` (dnorm2/dfgu/dkv/aattn_d/shead/samx/dposadd) | the draft chain (all Q4_0) | nibble-plane Q4_0 decode; two-region aligned pack; 40960-row slice head with prompt-frequency table (knob-off path) |
| `sheadf`/`samxf` (P10-dense, `TLX_DHEAD_FULL=1` default) | the FULL-VOCAB draft head + argmax (the shipped prose drafter) | slice-head GEMV verbatim at VOCAB=248320 over the resident trunk head plane (zero new VRAM; logits3 row-0 scratch, timeline-ordered); argmax writes the row index AS the token id |
| `accept`, `accept4`, `accept5e`, `acceptsel` | accept/commit: m-ladder, state slot copy, h_seed select | device-resident; no host rollback; accept5e maps m to the K4 chain's dedicated per-step hiddens (hd_d0..d3) |
| `lookup_nw32`, `lookup{5..9}_nw32` | deep-K n-gram drafter: suffix scan over tok_hist -> fill the whole draft ring on hit | LMIN=8; deep-K scan-range law iend=pos-8-K; hit flag in the emit record |
| `k2s{5..9}`, `accept{k}k`, `acceptsel{k}k` | deep-K probe trunk scan + accept/commit per K | REC-CHAIN SLOT LAW (t=K reads rec{K}x scratch); layout-aware emit word count (16w K2 -> 20w K8) |
| `spk_pre{5..9}qh`, `spk_g4nw32hm{5..9}`, `spk_c{5..9}g` | deep-K attention (ROWS=5..9 windows) | RMAX pad rows / MAXOWN owner paths (the RP>NW owner bug fix); 64 regs 0 spill in-graph |
| `r7d.cu` family (`ffn8r7`, `down8r7`, `ffn8v3r7`, ..., `ffn8v9r7`, `down8nw32v9r7`, `ffn8v5r7`/`down8nw32v5r7`) | decode/spec GEMVs reading the shared packed7 plane (the v5r7 twins serve the T=5 EAGLE probe) | bit-identical ports; kills the both-live VRAM wall, full m64 prefill coverage; R7a added the uint4 single-load W-fetch (all four u32 lanes consumed — the DCE law) |
| `gen_m{5..9}.py` | generators for the M=5..9 batched kernel sets | launch-bounds-preserved renames + fail-loud row-store audits on every write family (the R5a bug class) |
| `pack_w5.py` -> `packed5/` | the packed5 Q5_K qkv repack (48 tensors, 6 bits/word units) | pure integer permutation, byte-exact roundtrip; the gdnqg prefill GEMM reads true-16B units |
| `p8_imma.cu` | the W4A8 IMMA discriminator (info-only) | mma.m16n8k32.s8.s8.s32, per-chunk fp32 rescale; its "x4.58" was a discriminator FRAME ARTIFACT (one-plane M=64 vs two-plane M=128 arms) — the honest fused-shape class is x1.06 (T2 correction; match plane-count x M-grid between arms) |
| `pfk_q8.cu` + `p8_w4ffn7.cu` (`PF_W4A8=1`) | the SHIPPED Tier-2 W4A8 prefill ffn: per-(row,128k-chunk) absmax int8 act quant + fused gate+up IMMA GEMM with silu*u epilogue | reads the EXISTING packed7 planes through a linearized int4 view of the IQ3 codebook (8 linear levels, delta=4.0507 = 1.49% weight-RMS; per-32k-scale-word rescale at the mma k-step); zero new VRAM, full 64-block coverage; +7.3% @2k; kill-switch unset = byte-identical Tier-1 |
| `p8_w4ffn.cu` + `pack_w4.py` -> `packed4/` | the v1 int4-plane W4A8 variant (kept for partial-coverage experiments) | 12.6% weight-RMS raw requant; +5.88 GB planes vs <1.9 GB 100k-state headroom — full coverage VRAM-dead; superseded by the packed7-reading v2 |
| `p8q8_v{0..9}.cu`, `p8q8_w{a,b,d,e}.cu`, `p8q8_time.py` | the W4A8 act-quant/IMMA bench+dbg ladder (the discriminator genealogy) | the v0..we iteration trail that isolated the ternary-negation packing law, the per-word scale law, and the char4-store codegen fault |
| `k1_*var`/`amx3` | argmax + misc fused epilogues | in-graph argmax (dtype-safe via max/where/arange) |
| **MoE: `rt8e256.cu`** | the router GEMV + fixed-order top-8 + renorm + shared gate | 8 masked-argmax passes (tie -> lower id), canonical pair list, NO sort |
| **MoE: `gx8e256up`/`dn` (+`up4`/`dn6`)** | grouped gather-GEMV over the selected experts (up = gate+up w/ SiLU·gate epilogue; dn = down, fp16 partials) | flat sequential pair walk (no grid-stride); device-side expert-pointer tables (VA tables legal); 16B-aligned expert slabs |
| **MoE: `mx8e256cmb.cu`** | fixed-rank-order fp32 combine × gates + the resident shared-expert FFN | fixed summation order = bit-stable MoE layer |
| **MoE: `spkq256`/`spka256`/`spkc256` (+`m`/`s` variants)** | split-KV attention at head_dim 256, GQA 16:2, output sigmoid-gate fused | warp-per-position score loop (the O(L)-wall fix, 4x @16k) |
| **MoE: `k2s36`/`gconv36`/`h6kam`/`embg248`/`rmsz2048`** | GDN scan + gated conv at 2048-hidden, fused head-argmax, 248k-vocab embedding gather, (1+w) norms | the dense families rescaled; `h6kam` folds head GEMV + argmax |
| **MoE: `acc36`/`selc36`/`rowcp2048`** | on-device accept, in-graph commit, MTP-chain hidden carry | the host fold (~0.2 ms/cycle host residue) |

## Graph & submit model

*Plain language: the ~480 kernels of one token cycle are pre-recorded into a
GPU-side program (a graph); running a cycle is one small write and one
doorbell. The CPU does essentially nothing per token — which is the only way
to hit these speeds over Thunderbolt.*

One `NVComputeQueue` per conv-parity; cycle = draft_g -> probe_g -> accept_g
-> flusher_g. Kernels chain via **QMD dependent pointers** (the fork's exec
active_qmd path): the pushbuffer per graph is ~40 words (memory barrier +
timeline wait + 2 SEND_PCAS + release semaphore on the last QMD). Per
token-cycle: patch 2 timeline values, one small pushbuffer copy into the
shared cmdq ring, ONE gpfifo entry, ONE doorbell = ~50-100 us host. No
`bind()` (per-graph hw_pages are unproven; the shared-ring MTP_GRAPH_NOBIND
path is the safe one). The 2 MB shared cmdq ring never wraps in steady state
(1359 KiB process-lifetime peak measured). In-flight ceiling: pipeline depth
2 on 452-k graphs (904 kernels stable, 3.6k faults). Local sizes are
NAME-ENCODED (`nw32`/`nw24`/`nw16` tokens — see
[DEXT_LAWS.md](DEXT_LAWS.md) L2).

## Memory map (steady @100k, ~23.5 GB of 24 GB with the packed5 both-live plane)

| region | size | notes |
|---|---|---|
| packed weights (engine/packed/) | 7.6 GB | aligned-repacked IQ3_XXS/Q6_K + raw Q5_K/Q4_K/Q8_0/IQ3_S blocks (12.6 GB GGUF total) |
| packed5 qkv plane (engine/packed5/) | +1.89 GB | P8 packed5 Q5_K qkv repack for the prefill gdnqg GEMM (raw Q5 stays for decode q5g8v; fits at full 100k state) |
| int8-KV cache | 16 x ~218 MB = 3.5 GB | biased u8 + per-row fp16 scales; replaced 16x411 MB fp16 (-3.1 GB) |
| draft pack | 168 MB | Q4_0 blk.64 (two-region aligned) + 40960-row Q5_K head slice + id table |
| GDN state slots rec4/conv4 | [48][5]... | slot 4 live, 0..2 per-step drafts (M=3 scratch) |
| workspaces, qw16/pm/ps/pA, token ring, logits | ~1 GB | split-KV partials at S=256; 40960-slice draft logits |

## The weight planes (packed7 / packed5)

*Plain language: the quantized weights are re-laid-out on disk once, offline,
so that every load the kernels make is wide and aligned — and so decode and
prefill share ONE resident copy instead of two.*

- `pack_w1c.py` -> `packed/`: alignment-lawed repack of the IQ3/Q6 GEMV
  families (`[qs 64B][scales 32B][d 2B]` per block — the M1 law made
  mechanical).
- `pack_w7.py` -> the packed7 plane (4.28 GB, 160 tensors): wide-tile
  alignment for the prefill GEMM family — AND, since R2c, the decode/spec
  GEMVs read the SAME plane through the `r7d.cu` family, so the original
  packed copies never upload (net -2.4 GB VRAM, full m64 coverage, 288
  tensors).
- `pack_w5.py` -> `packed5/`: the Q5_K qkv segment as true-16B units for the
  prefill gdnqg GEMM (integer permutation, byte-exact roundtrip).
- `q4pack.py` -> `draft_pack/`: the Q4_0 draft block in nibble-plane layout
  (two-region aligned).

## Kernel-lineage history: 190 ms -> 75 tok/s

| stage | result | verdict |
|---|---|---|
| stock tinygrad decode @100k | 4.17 tok/s (fp16 KV) | baseline |
| tinygrad-stack a3-family substitutions | 10.95 tok/s @100k, 15.38 @2k | worked (IQ3/Q5/Q8 hand GEMVs, O(L) attention trees) |
| tinygrad-stack MTP (mtp_v3, graphs) | 7.28 tok/s @100k | worked — the correctness contracts + graph machinery; but scheduler-swarm-bound |
| **engine0 W1A-W1C** (hand kernels, G_CYCLE graphs @2k) | 20.09 -> 25.59 tok/s T=1 | worked — static buffers + wide loads + graph replay killed the launch tax |
| **W2_MTP** (engine MTP K=2 @2k) | 40.14 tok/s | worked — 484-kernel cycles, per-step state slots |
| **W2_100K** split-KV trio + 100k bootstrap | 29.63 tok/s @100k | worked — GQA-shared streaming K1 (but 160-190 GB/s plateau) |
| W2B SKV-G smem staging | 32.35 | worked — 291.6 GB/s pipelined; occupancy-bound (1 CTA/SM) |
| W2C G4 fat CTAs + honest synced benches | 33.30 | worked — 283 GB/s synced; pipelined benches refuted |
| W2D GEMV half2/fat-CTA polish | 34.77 | worked; T-layout transpose REFUTED (load width not the wall) |
| W2E int8-KV (PRMT dequant) | 35.56 | worked, Tier-1, -3.1 GB; "attention is byte-bound" REFUTED (dot-phase-bound) |
| W2F half2 QK/PV dots | 39.03 | worked (-6.0 ms); scan grid-split REFUTED (+1.7 ms) |
| W2G HMMA standalone + all levers refuted | 39.03 (unchanged) | stage-phase pipelining, draft half2, draft-vocab rebuild, warp restructure: all refuted with measurement |
| **W2H name-encoded-launch fix + HMMA in-graph** | **40.35 tok/s** | worked — the 256-thread in-graph launch was the whole W2G "broken kernel" |
| R3 LOOKUP n-gram drafter (K=2) | 40.2-40.9 class, +0.10 ms/cyc | worked — hit cycles draft for free (83.3% hits, E[m\|hit]=2.000) |
| R4/R5 deep-K graph-set selection, K=4 -> K=7 | **63.11 tok/s** | worked — sel-mode K=7; the R5a row-store bug postmortem + gen_m6..8 audits |
| R7a norms per-row CTAs + uint4 hygiene | **68.62 tok/s** | worked — the CTA-serialization law; the W-merge kept as free bit-identical hygiene |
| R7a K=8 + draft-skip | **71.51-72.02 tok/s** | worked — gen_m9 first-build-green; lookup-only draft graph on deep cycles (emit byte-identical) |
| R8 K=9 -> K=10 | **74.70 -> 75.56 tok/s** (75.81 through the W5 fixed stack) | worked — gen_m10/m11 first-build-green; the K-ladder stops at ten (increments < +1 tok/s/rung) |
| P8+A adaptive T=1 prose mode (dense) | prose 14.67 -> **20.56 tok/s** (+40.2%), output bit-identical | worked — the alpha-death signal switches to T=1 cycles; quote classes never trigger it (75.35 canonical with the mode ON) |
| MM P0-P2 MoE decoders + converter | bit-exact dequant vs llama.cpp C; both repack tiers (16.553/13.945 GiB) | worked — the four kernel classes (router/gather-GEMV/combine/split-KV) at production bandwidth; ~15 converter laws banked |
| MM P3-P6 the MoE train + spec path | full 40-layer T=1 GREEN vs the fp32 anchor; spec==T1; quote 118.6 tok/s in-harness | worked — GDN/attn ports, the 60-prompt Tier-1 bank, K-mix graph sets |
| MM P7 host fold + chunk-256 + 96k | host residue ~0.2 ms/cycle; 181-204 tok/s prefill @2k-16k; 96k ship config unblocked | worked — acc36/selc36 on-device, the uncached-sysmem cpu_view, S-split anchors |
| P8 multi-model serving | registry + swap + per-model pcache live; full swap round-trip through the reboot | worked — dense <-> MoE promotion tracked across /v1/models |
| MM P8-SERVE the MoE serving bridge | bank60 60/60 THROUGH THE DAEMON; pcache CACHE_HIT exact; soak 148 rounds 0 fail | worked — serve.py as an imported library; 29/29 mock battery |
| MM P9/P10-A the first-party MTP K=4 | prose 19.6 -> 42-47 (harness) / **19.1 -> 40.1 through the API**; mtp==t1 60/60 det x2 | worked — the blk.40 drafter at 0.875 depth-1 acceptance; the PF-only cur fix; soak 106 rounds 0 fail |
| MM P10-B pairwise MoE fusion | DIVERGED @ token 27; NOT shipped | honestly falsified — and it corrected the launch-serialization law (fat kernels pipeline dispatch; the T=1 cycle is kernel-RUNTIME-bound) |
| P10-dense rung 1: full-vocab EAGLE draft head | dense prose 23.0 -> **43.7 tok/s** GSM8K median through the API (1.90x); quote ~75 unchanged; Tier-1 60/60 x2 + 59/59 | worked — the failure was the slice head (89.8% of prose targets out-of-slice), not the blk.64 nextn chain; proposing over the resident head plane = zero new VRAM |
| P10-dense rung 2: the K=4 EAGLE chain | Tier-1 60/60 x2 + 59/59 with EAGLE_K=4; prose K4 17.25 vs K2 22.93 tok/s in-harness (-25%) | honestly falsified — the 1-layer drafter saturates at m=2 (m=4: 0/120); machinery ships gated (`TLX_EAGLE_K`, default 2) for a future deeper drafter |
| R2c decode-r7 shared packed7 plane | prefill +22.9% @100k | worked — one resident weight copy serves decode + prefill |
| P8 packed5 + o-proj M-grid fold | prefill 530.6 @2k / 328.0 @100k (Tier-1) | worked — true-16B qkv units + one g=160 o-proj launch; both bit-identical |
| T2 W4A8 packed7-IMMA ffn (`PF_W4A8=1`) | prefill **569.2 @2k / 510.1 @8k / 342.0 @100k** (Tier-2) | worked — the existing packed7 planes read through a linearized int4 codebook view; zero new VRAM, decode Tier-1 untouched; the honest IMMA verdict x1.06 (the banked x4.58 was a discriminator frame artifact — 750 closed) |

The pre-engine tinygrad-stack lineage (mtp_v3 et al.) lives in `lineage/` —
it is the correctness-contract ancestor of the engine and reached 7.28 tok/s
@100k on the same gates.

## What was tried and refuted (do not re-learn)

grid-stride loops (hang); smem-staged G1 attention (L1 hits, staging
round-trip loses); carveout override for 2 CTAs/SM (dext is 1 CTA/SM hard);
T-layout transposed weight packs (load width not the wall); K=3 at this draft
quality (alpha collapses 0.867->0.608 with depth, net-negative); the dense
K=4 EAGLE chain at the CURRENT 1-layer drafter (machinery Tier-1-proven and
gated on; drafter saturates at m=2, m=4 never fires — -25% prose, ships off);
QMD-unchain
across layers (structurally void — real data deps); draft GEMV half2 ports
(latency/L2-bound, neutral); draft-vocab slice expansion (the 100k prompt is
a 30-distinct-id repetition; the slice already covers 100% of truth — alpha
is draft-fidelity-bound); scan grid split (prologue replication costs more
than the parallelism); software-pipelined K1 stage (kernel already at 88% of
the HFMA2 math floor — only tensor cores could win, and they did); cp.async
(device-faults on this dext — parked); symbolic-slice TinyJit replays
(offsets don't rebind); chunked split-K attention at tensor level (launch
tax > bandwidth gain, and numerically wrong at T>1 in one form).

## The serving layer

*Plain language: the engine runs as a daemon that owns the GPU forever; an
HTTP façade speaks the OpenAI chat protocol on top. Conversations stay
resident, so a follow-up question only pays for its new words.*

Full reference: [SERVING.md](SERVING.md). Two processes over a unix socket
(`/tmp/llm-engine.sock`, newline-delimited JSON-RPC):

- **Engine daemon** — the canonical `test_w100k.py` host booted with
  `M1A_SERVE=1`; `serve.py` attaches as a library at the post-`build_graphs`
  handoff gate. The request path is `DecodeSession.step()`: submit draft_g ->
  probe_g -> accept_g -> flusher_g timeline-chained on a carried `prev`, one
  wait, then read the 32-byte emit record (`{pos_new, m, tok0..2, stop_flag,
  cycle}`) via a windowed 8-word readback — never the 400 KB tok_hist
  download. Cancellation = stop submitting (the device is quiescent at each
  step return, tens of ms latency).
- **Fixed-handle state discipline** — every mutable buffer is allocated once
  at boot at full engine context (KV int8 slabs, GDN rec4/conv4 slots, token
  ring, tok_hist, seeds); requests mutate state only via windowed uploads and
  device memsets, never reallocation (a realloc means stale graph kernargs
  plus orphaned VRAM — the ~755 MB/reset trap). FRESH reset = zero-seed
  memset + prefill; FOLLOW_UP = nothing reset, delta prefill only.
- **HTTP façade** (`api_server.py`, 127.0.0.1:8080) — OpenAI
  `/v1/chat/completions` (SSE + non-stream + usage), `/v1/models`, `/health`;
  Qwen chat template applied bit-exactly; byte-exact streaming
  detokenization with UTF-8 + stop holdback; queue 1 active + 4 FIFO (429
  beyond); cancel-on-disconnect; `conversation_id` prefix pinning;
  longest-prefix-match reuse so a follow-up turn prefills only its delta.
- **Stability laws** — the ~950-cycle dext budget (continuous 4-graph spec
  cycles fault at 850-1025 cycles; graphs rebuild + re-anchor every 256
  cycles at <1% cost); bounded generation (`max_cycles = max_tokens + 4`);
  snapshots are delta-windowed (a full-KV copyout silently OOM-kills the
  daemon); re-encode message history from a mirror, never re-render it
  through the chat template.
- **Ops** (`engine/ops/`) — launchd plists for both processes, `enginectl`,
  circuit breaker (3 crashes/10 min -> stay-down + 503, state persistent
  across fault-reboots), GPU lockfile (pid-liveness-checked); SIGTERM drains
  and synchronizes, never kill -9 a live GPU process. The supervisor wrapper
  single-sources the environment from `ops/env.canonical` (generated from
  the published `env.canonical.example`; sha256-digest logged, config
  fingerprint surfaced in `/health` and drift-checked to 503 by the API).
- **The review-fix stack (W1-W5)** — the serving layer carries a five-wave
  hardening campaign: per-request slot ownership + per-conversation locks +
  stop/think semantics (W1); the socket trust boundary (0600 + peer-uid +
  admin-token ACL on privileged RPCs), jinja-sandboxed template rendering,
  HTTP hardening, redacted `/health` (W2); prompt-cache durability —
  fsync+sha256 nodes, transactional validate-before-upload restore, back-
  pressure, corruption quarantine with clean FRESH fallback (W3); and
  boot-time engine tripwires — every captured kernel's launch grid asserted
  against its name-encoded maxntid, stack frames against the spill policy
  (name-scoped exemptions for the documented Tier-2 prefill families), and
  the loaded cubin set asserted against K-rung manifests (W4). The GPU-free
  mock battery in `engine/tests/` (85 tests) is the regression net;
  `engine/w4_census.json` is the calibration census. Full record:
  [history/FIX_CAMPAIGN.md](history/FIX_CAMPAIGN.md).
- **Batch serving (R6, opt-in)** — `engine/r6_batch.py` + `r6_serve.py`
  batch B=2 conversations into one M=BT probe trunk (shared weight read;
  per-stream state banks `*_s{s}`; the swap trick makes the eager machinery
  stream-aware). Per-stream bit-exact at every rung; ships default-off with
  an honest cost model ([history/R6_BATCH.md](history/R6_BATCH.md)).
- **The durable prompt cache (R1, `pcache.py`)** — every prefilled context
  becomes an on-disk, hash-keyed prefix trie over FED token ids (chain root
  mixed with the engine config fingerprint, so entries are valid only under
  a bit-identical numerics config; never keyed on rendered text — the
  RE-ENCODE law). Nodes at 1024-token stride carry the int8-KV + GDN +
  DRAFT-KV windows plus draft/trunk hiddens (~195 MB/node); a 100k restore
  lands at the deepest cached boundary in ~6.5 s vs ~13 min FRESH (60/60 x4
  boots). Ingest fires at chunk-quiescent boundaries (the ~950-cycle law
  untouched); the API surface exposes `cached_tokens` / `prompt_cache_key`.
- **tok_hist seeding (R5d)** — `serve.py` seeds the token history on
  boot/FRESH/FOLLOW_UP/CACHE_HIT/snapshot paths; the deep-K LOOKUP drafter
  self-matches a -1 prefix without it and faults.

## Multi-model serving (P8): the registry, the swap, the per-model caches

*Plain language: the service now hosts a model REGISTRY — several models
declared with their engine host, env file, context cap and cache root — and
swaps which one owns the GPU through a deliberate, crash-safe lifecycle.
Only one model is resident at a time (the 24 GB card holds one); the API
makes that explicit instead of lying about it.*

- **The registry** (`engine/ops/model_registry.json`): per model — id,
  display name, arch class, `engine_host` (the python entry point that owns
  the GPU: `test_w100k.py` dense / `test_moe36.py` MoE), `model_path`,
  per-model env file, `ctxk`, `max_output_tokens`, pcache root + quota GB.
  The `default_model` boots when no state says otherwise; `ops/state/`
  (`current_model`, `next_model`, `swap_in_progress`) carries the swap
  intent.
- **The split env** (`ops/env.common` + `ops/env.canonical.d/<model>.env`,
  generated from the published `env.common.example`): shared secrets/ops
  knobs in env.common, per-model numerics/paths in the per-model env. The
  dense union is variable-identical to the old monolithic
  `ops/env.canonical` (config_fp unchanged — pcache continuity); the
  monolithic file remains the pre-P8 rollback source.
- **The swap FSM** (wrapper `engine_daemon.sh` + `enginectl switch <id>`):
  validate (registry entry, env file, engine host, model file) -> drain
  check -> write `next_model` atomically (tmp+rename+fsync+DIR-fsync — the
  L5/GPU-EXIT-sync laws; a system `sync` before the stop) -> graceful
  shutdown RPC -> the GPU-EXIT reboot the stop provokes IS the transport ->
  the wrapper's boot selection promotes `next_model > current_model >
  default`, clears a swap-authored staydown (never a breaker-authored one),
  quarantines invalid intents (keep current + refuse, exit 17), exports
  `TLX_MODEL_ID` to the engine. Auto-swap on request is DELIBERATELY
  refused (thrash); the API answers 409 `model_not_resident` + a switch
  hint instead.
- **Model identity is load-bearing**: the engine loads `TLX_MODEL_PATH`
  (was hardcoded) and `serve.py` boot-asserts loaded == fingerprinted
  (exit 18 on mismatch). The API and the daemon derive the SAME per-model
  config_fp extras (dense -> cubin-set digest; MoE -> the packed-manifest
  sha) — the FP-EXTRA CONVERGENCE law, or every boot would look like
  config drift.
- **Per-model prompt caches**: `PC_ROOT`/`PC_QUOTA_GB` per model env
  (30 GB dense / 26 GB MoE); nodes are config-fp-scoped by construction,
  so the two caches never cross. The MoE daemon additionally runs its own
  `pcache_moe.py` bridge (r1moe1 nodes; ~76 MB per 1024-token node; the
  G4 gate proves a CACHE_HIT continuation EXACT vs the FRESH arm).
- **The API surface** (`/v1/models` registry-driven with
  resident/loadable/unavailable + swap states; `model` field 404 unknown /
  409 not-resident / absent = resident; swap-in-progress -> 503 +
  Retry-After, file-derived). Conversations are model-scoped by the
  (model_id, conversation) key — a cross-model follow-up is FRESH by law.
- **The MoE daemon** (`engine/serve_moe.py`): imports the dense `serve.py`
  protocol primitives as a library (framing, send locks, validation, admin
  ACL, watchdog patterns) and swaps the module state for the MoE engine —
  the dense daemon is untouched. `test_moe36.py` is the engine host;
  `engine/mm/` carries the campaign workbench (harness, kernels, banks).
- **The R3 + L7 hardening** both daemons carry: the R3 review campaign
  (51 findings, five waves: W-A liveness — restore beats, idle-arm
  watchdog, watched shutdown, boot deadline, the consecutive breaker with
  ready-uptime attribution, the rebuild budget; W-B event loop — the
  promote-vs-cancel grant release, evq coalesce-first, conv-lock held for
  the stream lifetime, encode-once-off-loop, bounded residency; W-C
  integrity — one env parser, fail-closed peercred, 0600/0600 cache
  perms, header-only admin, /health redaction, HMAC key placement; W-D
  conformance — reasoning_effort tiers, sampling compat, error enums,
  reasoning tail, honest max_tokens 400s, idempotency keys,
  cached_tokens, tvars; W-E the real-listener harness + knob census) and
  the L7 abort-safety protocol (mid-prefill aborts fence, invalidate and
  force FRESH — the unfenced-abort class that panicked the box; the global
  graph-submit budget `TLX_GLOBAL_CYCLE_REBUILD_EVERY=928`; admission
  hardening with an idle-engine-heals-only permit watchdog). The knob
  census and drift runbook live in `engine/docs/SERVING.md` (the R3
  serving runbook).

## The prefill layer

*Plain language: eating a fresh prompt is a different job than generating —
so it gets different kernels. The prompt is cut into 128-row chunks pushed
through batched versions of the whole model, recorded into two GPU graphs
per chunk. It went from 21.8 to 569 tok/s @2k — a 26x ladder, every rung
bit-identical to the slow path until the one labeled Tier-2.*

Details and the full ladder: [history/PREFILL.md](history/PREFILL.md),
numbers in [PERFORMANCE.md](PERFORMANCE.md). The shape of it:

- **M=128 trunk (R2c)**: every trunk kernel has a batched variant (`pf_*`/
  `pfs64`/`pre64`/`m64` twins, 2-M-block M-grids of the m64 cubins)
  processing up to 128 prompt positions per launch with bit-identical
  per-row fp op order — GEMMs as `m16n8k16` tiles sharing one staged W
  tile; tail r%128 -> M64 -> M32 (both ambient flags cleared — the law-2
  tail re-commit).
- **The WY-C32 chunk-level scan (R2b)**: a chunk-level WY-representation
  solve (`pf_scanchunk.cu` + the pfc cubin set) replaces the per-16-row
  scan chain; one launch triple per chunk (NC=2 at M=64, NC=4 at M=128).
- **The shared packed7 plane (R2c)**: decode/spec GEMVs read the prefill
  weight packs (`r7d.cu`) — the original packed copies never upload; full
  m64 coverage with no both-live VRAM fault.
- **Chunk graphs**: the M-chunk program is captured as 2 QMD-chained graphs
  (~527 launches per chunk after the P12 diet; 745 at M=128 with the R2d
  consolidations), submitted with no host work.
- **Attention**: two captured graph sets (S13/S26) picked per chunk position
  at THR=8192; wide-M ROWS=32/64 attention windows (`PF_ATTNW`); attnqkv as
  ONE g=448 launch on the M128 trunk (R2d; the P7E4 corruptor class
  retired).
- **DBUF ring-depth-4 (R2d)**: the gdnqg GEMM issues unit loads 3 phases
  early through 4 named unit-register sets (x1.098 on that class; per-class
  coin-flip law for the mixed twins).
- **packed5 + o-proj fold (P8)**: the gdnqg Q5_K qkv segment reads a
  packed5 integer repack (true-16B units, x1.29 on the class); the attention
  o-proj is ONE M-grid-folded g=160 launch instead of 4x m32 (x2.06 on the
  pool).
- **The Tier-2 W4A8 ffn (T2, `PF_W4A8=1`)**: a fused
  `mma.m16n8k32.s8.s8.s32` IMMA kernel (`p8_w4ffn7.cu`) reading the
  EXISTING packed7 planes through a linearized int4 view of the IQ3
  codebook with a per-row absmax int8 activation quant (`pfk_q8.cu`) — zero
  new VRAM, full 64-block coverage, +7.3% @2k; the honest IMMA verdict is
  x1.06 fused-shape (the banked x4.58 was a discriminator frame artifact),
  so 750 was closed, not crossed.
- **Batched draft fill**: the speculative-draft state fill is amortized into
  the chunks as a recorded-trunk-hiddens replay — the draft's own
  attention/o-proj/FFN are dead code during fill (alpha after prefill
  2.67 -> 2.68).

## The drafter-quality program instruments (chain_sim + ttt/; in flight)

*Plain language: the next performance axis is the DRAFTER itself — the
1-layer EAGLE draft block saturates at ~2 accepted tokens, so K=4 measured
-25% prose and ships off. Raising that means retraining the checkpoint's
own draft layer (blk.64) and scoring candidate packs OFFLINE against real
engine traces before anything touches the GPU. Two pieces of permanent
infrastructure landed for that program — nothing from it has shipped into
the serving defaults yet.*

- **`engine/chain_sim.py` — the ENGINE-CALIBRATED offline acceptance
  simulator.** The engine dumps a full cycle trace (`TLX_TRACE_DUMP` in
  `test_w100k.py`: the anchor state before each draft cycle, the engine's
  own proposals, probe argmax rows, the decode-start int8 KV + GDN state,
  emb/head planes); chain_sim then replays the EXACT cycle sequence in
  numpy — the real draft kernels' math (fp16 round-trips, the Q4_0
  two-region pack dequant, the 40960-row slice argmax) against a
  swappable draft pack. A candidate pack is scored (E[m] at K=2/K=4,
  per-position conditionals) without a GPU boot. Calibration (G0) holds:
  sim m-dist [30,14,7,0,0] vs engine [29,14,7,0,1], 83.8% per-proposal
  agreement, E[m]|k2 0.549 vs engine 0.588 — pack-to-pack deltas are 2-4x
  the residual, which is the resolution the pack-selection gate needs.
- **`engine/ttt/` — the EAGLE-3 trainer pipeline (external, rented GPU).**
  The checkpoint's own blk.64 draft layer (424.6M params) is retrained in
  plain PyTorch on a rented H100 with the EAGLE-3 recipe (training-time-
  test multi-step unroll + token-only CE over a frozen trunk), then packed
  back into the engine's Q4_0 two-region draft-pack layout
  (`pack_trained.py`, RTN or GPTQ-calibrated) and swapped in via
  `TLX_DRAFT_PACK=<dir>` — a pure weight swap, zero new serving CUDA.
  `validate_port.py`/`engine_ref.py` prove the port (step-1 relerr 7.9e-7
  vs the engine chain math; pack norms bit-exact vs the first-party
  checkpoint).
- The Phase-0 verdict and the ranked levers (exposure bias via TTT
  training is the big one; data distribution/ctx second; precision a
  distant third) are in [history/TLX_P0_LEVER_RANKING.md](history/TLX_P0_LEVER_RANKING.md);
  the program plan lives with the driver-seat workspace, not this repo.
