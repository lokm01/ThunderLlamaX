<p align="center">
  <img src="docs/media/logo.png" alt="ThunderLlamaX" width="256">
</p>

# ThunderLlamaX

**LLM inference on an eGPU, hitched to a Mac.**
*(Or, if you prefer: the Mac that thinks it's an Ultra.)*

Run a local LLM on your Mac — any Apple Silicon Mac with Thunderbolt — by
attaching an NVIDIA RTX 3090 eGPU over Thunderbolt 4. macOS supports no eGPU
on Apple Silicon and ships no NVIDIA driver, so this project brings its own:
an open DriverKit driver that talks raw PCIe to the GPU (no CUDA runtime, no
CUDA driver, no NVIDIA userspace), a hand-written kernel engine, and an
OpenAI-compatible API on top. The result: **three models served from one
service** — Qwen3.8-27B dense at **75.81 tokens per second** decode
(bit-exact against greedy decoding, 100k-token context, 569/510/342 tok/s
prefill; **43.7 tok/s** on novel prose via its own EAGLE draft layer),
Qwen3.6-35B-A3B MoE at **97.6-104 tok/s** quote-class /
**40.1 tok/s** prose-class, and an abliterated (uncensored) weight-variant
of the dense checkpoint served as a first-class registry model — the same
pipeline turns any Qwen3.8-architecture checkpoint into a swappable model —
with a durable prompt cache that restores a
100k context in ~6.5 seconds, speculative decoding that drafts from the
document being read AND from the model's own MTP layer, and a model
registry that swaps residents through a crash-safe lifecycle. Local
inference, on a Mac, at workstation speed.

The X reads as *ten*: K=10 speculative depth is the signature number
(75.81 tok/s, bit-exact, see below).

The cast, if you like flavor names: the **DraftHorse** engine (the hand-kernel
engine and its draft-and-verify speculation — `engine/`), **ThunderSilicon**
(the DriverKit dext/driver layer), and **LongMemory** (the durable prompt
cache). The technical docs keep the real component names: `engine0`, the
dext, `pcache`.

## What you get

| Capability | Detail |
|---|---|
| OpenAI-compatible API | `/v1/chat/completions` (stream + non-stream + usage), `/v1/models` (with per-model residency), `/health` on 127.0.0.1:8080 |
| Models | **Qwen3.8-27B** dense hybrid (48 gated-delta-net + 16 full-attention blocks, IQ3_XXS) · **Qwen3.6-35B-A3B** MoE (30 GDN + 10 attn blocks, 256 routed experts top-8 + shared, ~3B active, UD-IQ4_XS) · **Qwen3.8-27B-OBLITERATED** (an abliterated weight-only variant of the dense checkpoint — any same-architecture GGUF becomes a first-class model via the offline pack pipeline, [docs/DEPLOY_OBLITERATED.md](docs/DEPLOY_OBLITERATED.md)) |
| Context | 100,352-token KV (dense) / 98,304 (MoE); gates run at a 97,810-token prompt |
| Decode speed, quote-class @100k | **75.81 tok/s** dense (K=10 LOOKUP) · **97.6-104.0 tok/s** MoE (K=8 LOOKUP) — bit-exact in both |
| Decode speed, prose-class | **43.7 tok/s** dense (the checkpoint's own EAGLE draft layer behind a full-vocab draft head, GSM8K median through the API) · **40.1 tok/s** MoE (first-party MTP K=4) |
| Speculative decoding | n-gram LOOKUP from the document being read (K up to 10) **plus the models' own MTP layers as first-party drafters** (dense: EAGLE chain + full-vocab draft head; MoE: K=4 chain, 0.875 depth-1 acceptance) |
| Prefill speed (dense) | **569.2 tok/s @2k · 510.1 @8k · 342.0 @100k** (Tier-2 default; bit-identical Tier-1 path one env away); MoE chunk-256 prefill 181-204 tok/s @2k-16k |
| Multi-model serving | A model registry (`model_registry.json`), per-model env files + caches, `enginectl switch` — one resident at a time, 409 `model_not_resident` with a switch hint |
| Batch mode (opt-in) | B=2 concurrent streams, per-stream bit-exact; 81.33 tok/s engine-class harness aggregate / honest 1.20x service-measured — see honesty section |
| Prompt cache | A 100k context restores in **~6.5 s** vs ~13 min fresh; survives restarts; per-model roots + quotas |
| Follow-up turns | Resident conversation state: a 200-token turn @2k-class in **~2.1 s** vs ~6.1 s fresh |
| Supervised operation | launchd supervisor + circuit breaker + env single-sourcing + boot-time kernel tripwires; survives GPU-fault reboots |
| Exactness | **Bit-exact by default** — every speculative mode emits the identical token stream a non-speculative greedy rollout produces (Tier-1) |
| Output quality | **GSM8K 95.0%** dense (K=10) / **93.0%** MoE (MTP K=4) through the full serving API, 4-shot greedy · **61k-context needle: 10/10 exact retrieval** |
| Long-prompt reliability | Long prompts ≥20k tokens now serve reliably (a graph-kernargs pool recycles the driver's limited sysmem mappings); **42-min sustained-load soak with zero crashes** (300-problem GSM8K run, was a crash every 13-18 min before the fix) |

## Qwen on a Mac with an RTX 3090: the benchmark numbers

All decode numbers are greedy, at full model context (97,810-token prompt
for the dense model; the 96k split for the MoE), on the reference rig
(RTX 3090 24 GB in a TB4 enclosure, MacBook Air M2). "Tier-1" means
bit-exact (see FAQ). Full context and the complete ladders:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

**Qwen3.8-27B (dense):**

| Metric | Value |
|---|---|
| Decode, K=10 deep-K LOOKUP (hit-class workload) | **75.81 tok/s** (~118 ms/cycle, 8.92 tok/cycle; R8 + the W5 re-validation, ladder ... -> 71.51 -> 74.70 -> 75.56 -> 75.81-through-the-fixed-stack) |
| Decode, K=9 / K=8 / K=7 / K=2 rungs (same engine) | 74.70 / 71.5-72.0 / 63.11 / 40.35 tok/s |
| Decode on novel prose (full-vocab EAGLE draft head, shipped) | **43.7 tok/s** GSM8K median through the API (1.90x; was 20.56 adaptive-T1 / 14.67 pure-spec; quote-class unchanged at ~75) |
| T=1 engine, no speculation, @100k | 21.78 tok/s (45.92 ms/token) |
| Prefill FRESH @2k / @8k / 100k rebuild — Tier-2 W4A8 (default) | **569.2 / 510.1 / 342.0 tok/s** (+7.3/+6.4/+4.3% over Tier-1) |
| — same, Tier-1 bit-identical path (kill-switch) | 530.6 / 479.4 / 328.0 tok/s |
| Batch B=2, both-8k + draft-skip (engine harness, per-stream Tier-1) | 81.33 tok/s aggregate (1.61x the K2 solo; honest service-measured 1.20x — see below) |
| Cached 100k-context restore | ~6.5 s vs ~13 min FRESH; 60/60 exact across 4 restarts |
| Tier-1 gate: speculative == greedy T=1 rollout | **60/60 bit-exact, x2 deterministic; 59/59 vs stock tinygrad** |
| Deep-K acceptance (K=10) | E[m\|deep] = 10.000 — 92/92 deep cycles accept ALL TEN; 76.7% deep hits |
| Quality: GSM8K (4-shot, greedy, through the API) | **95.0%** (first 100 test problems, 0 extraction failures; decode median 23.0 tok/s at the P9 battery -> **43.7** with the P10 draft head, spot-verified 93.3% / 10-of-10; 2.4 s TTFT) |
| Quality: long-context needle @61k | **10/10 exact retrieval** (61,189-token contexts, code at 5-95% depth; 20k control 10/10 clean, 8/10 exact) |

**Qwen3.6-35B-A3B (MoE, through the full API, MTP mode default):**

| Metric | Value |
|---|---|
| Decode, quote-alpha / quote-code (K=8 LOOKUP class) | **97.6 / 104.0 tok/s** |
| Decode, quote-docx2 | 65.9 tok/s (+160% over the T=1-only daemon) |
| Decode, novel prose (first-party MTP K=4 chain) | **40.1 tok/s** prose-0 / 39.1 prose-1 / 29.9 prose-9 (was 19.1 T1-only) |
| MTP acceptance (depth-1 / E[acc]@K=4) | **0.875** / 2.58-3.08 accepted drafts per cycle |
| T=1 decode @64k-96k | 17.2-17.3 tok/s (57.8 ms/cycle) |
| Prefill, chunk-256 (bit-exact class) | 181-204 tok/s @2k-16k; full 96k feed ~800 s end-to-end |
| Tier-1 gate through the daemon | 60/60 mtp == t1 bit-exact, x2 deterministic (60-prompt bank) |
| Prompt-cache hit | continuation EXACT vs the FRESH arm (G4); ~76 MB per 1k-token node |
| Quality: GSM8K (4-shot, greedy, through the API) | **93.0%** (first 100 test problems; 36.7 tok/s median decode, 9.5 s TTFT — TTFT is the MoE's weak spot, ~66 tok/s effective FRESH prefill) |

**Audited, not just benchmarked.** Before this snapshot was published, the
whole stack went through repeated external-model review: a 10-review audit
that became a 60-finding ledger and five fix waves — serving correctness
(SSE slot lifetime, per-conversation locks, stop/think semantics),
security/ops (socket trust boundary, admin-token ACL, config-fingerprint
drift 503s, a persistent-breaker launchd supervisor, single-sourced
environment), prompt-cache hardening (fsync+sha256 durability, transactional
restore, backpressure), engine tripwires (boot-time launch-grid and
spill-frame asserts, K-rung manifests, a 782-cubin census) — then a live
re-validation (Tier-1 decode re-gated at 75.81, 15-min soak, zero faults).
A second 51-finding round (R3) hardened the serving layer again
(liveness/watchdogs, event-loop admission, integrity, OpenAI-conformance
error enums, a real-listener test harness), L7 closed the abort-safety
class that crashed the box under cancelled prefills, and the multi-model
campaign carried its own gate battery end-to-end (the full record,
including the findings that remain open, is
[docs/history/FIX_CAMPAIGN.md](docs/history/FIX_CAMPAIGN.md) +
[docs/history/MM_PLAN.md](docs/history/MM_PLAN.md) and the MM_P* journals).
A first output-quality battery ([eval/](eval/)) then closed the loop between
speed and correctness — GSM8K through the live API, perplexity baselines,
61k-context needle retrieval — and caught two real serving bugs on the way:
a first-token emission loss (every completion silently dropped its first
token; fixed) and a graph-kernargs mapping exhaustion that crashed
long-prompt serving (fixed with a slab-recycling pool: long prompts ≥20k
now serve reliably, and a 42-minute sustained-load soak ran with zero
crashes where the same traffic class used to die every 13-18 minutes).
Two later closures: the dense-PPL scorer fault was a one-line harness bug
(a 2x-oversized logits download — fixed, the full dense perplexity battery
now runs clean), and `sudo enginectl switch` state files are now always
written daemon-readable, so mixed sudo/non-sudo invocation can't silently
skip model promotion.

## Can you use an eGPU with Apple Silicon?

**Natively? No. Through this stack? Yes — that's the whole project.**

The background, because it's the question every Mac owner hits: macOS *did*
support external GPUs on Intel Macs (Thunderbolt 3, AMD cards only — Apple's
last NVIDIA web drivers date from 2018, and NVIDIA's CUDA toolkit dropped
macOS not long after). **Apple Silicon dropped eGPU support entirely**: plug
an NVIDIA (or any) eGPU into an M-series Mac and macOS simply has no driver to
hand it to. That's why "eGPU for AI on a Mac" has been a dead end — the
hardware link works over Thunderbolt, but the software stack stops at the
door.

ThunderLlamaX replaces the missing door:

- **A custom DriverKit system extension (the "dext")** maps the GPU's PCIe
  BARs into userspace and submits work directly — QMDs (queue descriptors),
  pushbuffers, doorbells, the works. No kernel extension, no NVIDIA code; the
  driver is open and inspectable. For everyone searching for a *macOS eGPU
  driver*: this is one, and it's the load-bearing wall of the project.
- The surprise that made the whole project worth doing: measured through this
  path, the GPU sustains **880 GB/s streaming and 843 GB/s GEMV — 94% of the
  card's peak**. The driver wasn't the wall; the software stack above it was.
  The PCIe/Thunderbolt link doesn't throttle decode because the weights are
  GPU-resident and the host only sends doorbells per token cycle.
- The dext has laws of its own (alignment, 1 CTA/SM, zeroed launch dims...)
  — forty-odd of them, each learned the hard way:
  [docs/DEXT_LAWS.md](docs/DEXT_LAWS.md).

## Does CUDA work on macOS?

**Not as a runtime, no — and this stack doesn't need it to.** If you've seen
`torch.cuda.is_available()` return `False` or hit "CUDA is not available" on a
Mac: that's permanent, not a setup problem. There is no CUDA runtime for
modern macOS and no NVIDIA driver to host one, so the usual stack — PyTorch,
vLLM, Ollama-with-NVIDIA — cannot use an external NVIDIA GPU on a Mac at all.
(They run on the Mac's internal GPU via Metal/MLX instead; see the
[eGPU vs unified memory](#egpu-vs-unified-memory-which-is-faster-for-llms)
section for that comparison.)

What this project does instead:

- **No CUDA runtime, no CUDA driver, no NVIDIA userspace at run time.** The
  GPU is driven entirely through the DriverKit dext above.
- Every kernel in the hot loop is **hand-written CUDA compiled ahead of time
  to raw cubins** (an `nvcc` container is used at *build* time only), loaded
  by the engine as bare ELF objects and launched by the dext's queue path.
  The kernels are the same PTX/SASS you'd write on Linux; it's the *driver
  stack* that macOS lacks, and that's the part this repo replaces.

## How it works (the engine, in plain language)

**A hand-written kernel engine instead of a framework.**
The decode loop is not built on a tensor framework's scheduler. It is ~390
hand-written CUDA kernels (dequant-GEMVs, a fused gated-delta-net scan,
split-KV flash attention with int8 KV and tensor-core dots, and — for the
MoE model — the router / gather-GEMV / combine / split-KV quartet), static
pre-allocated buffers, and the whole per-token program captured into a handful
of GPU graphs — so one token cycle costs about one doorbell ring of host work
(the MoE engine folds accept/commit fully on-device: ~0.2 ms host per cycle).
Weights live in offline-repacked, alignment-lawed planes (`packed7`/`packed5`;
expert-major slabs for the MoE) that serve decode and prefill from one
resident copy.

**Speculative decoding that drafts from the document being read — and from
the model itself — while verifying bit-exactly.**
While ingesting or revisiting long documents, the model's own context is the
best drafter: a GPU-side n-gram scan finds where the current 8-token suffix
last occurred and proposes the next K tokens from history (K has climbed
2 -> 10; that's the X). On novel prose — where there is nothing to copy —
the engines now ship their own answers: the dense model runs its
**checkpoint-shipped EAGLE draft layer behind a full-vocab draft head** (the
old 40960-row slice head could only propose ~10% of novel-prose targets;
proposing over the full vocabulary doubled prose to 43.7 tok/s), and the MoE
model runs a **first-party MTP K=4 chain**: the model's own multi-token
prediction layer (the `blk.40` head the checkpoint ships for exactly this)
drafts four tokens ahead of the verifier, accepting 0.875 of first drafts.
Speculation-depth machinery is likewise proven to K=4 on the dense engine
(Tier-1 gated behind `TLX_EAGLE_K`) pending a deeper drafter.
In every mode the load-bearing trick is the same: **every batched kernel
preserves the single-token kernel's floating-point op order exactly, so
acceptance can never flip a near-tie** — the speculative stream is
bit-identical to plain greedy decoding, at 8.9 tokens per cycle on
hit-class work.

**A second architecture: the MoE engine (four new kernel classes).**
The Qwen3.6-35B-A3B port (256 routed experts + 1 shared per layer, ~3B of
~35B parameters active per token) needed four genuinely new hand-kernel
shapes on this dext: a **deterministic router** (fp32 top-8 with
tie-to-lower-id, no sorts, no runtime-indexed locals), a **gather-GEMV**
that walks the selected experts' packed slabs in a fixed order through
device-side pointer tables, a **fixed-rank-order combine** with the shared
expert folded in, and a **split-KV attention** retargeted to the model's
head geometry. Accept and commit moved fully on-device (the "host fold" —
~0.2 ms of host work per cycle). Full story:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) + the
[docs/history/MM_PLAN.md](docs/history/MM_PLAN.md) campaign.

**Multi-model serving: a registry, not a lie.** One 24 GB card holds one
model, so the service says so: `ops/model_registry.json` declares the
models (engine host, env file, context cap, cache root), `/v1/models` shows
which is **resident / loadable / unavailable**, a request naming a
non-resident model gets a clean **409 `model_not_resident`** with a switch
hint, and `enginectl switch <id>` performs the deliberate swap — intent
written atomically before the stop, the platform's GPU-exit reboot used AS
the transport, the target promoted on the way back up. Each model keeps its
own prompt-cache root and quota. [docs/SERVING.md](docs/SERVING.md).

The third quiet idea: **honesty as a feature**. Numbers come with their gate
context, the walls are measured and published, and the workloads where this
rig does and doesn't shine are written down (see FAQ and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md)).

## What you need (Mac eGPU hardware requirements)

| Component | Requirement | Reference rig |
|---|---|---|
| Mac | Apple Silicon with Thunderbolt 4 (MacBook Air/Pro, Mac mini, Mac Studio — the Mac just steers) | MacBook Air M2, 16 GB |
| GPU | NVIDIA sm_86 class in a TB4 eGPU enclosure; **24 GB VRAM for the 100k-context config** (~21.6 GB live) — a used RTX 3090 is the price/performance pick for local LLM work | RTX 3090 24 GB |
| Cable/dock | TB4 end-to-end (TB3 will throttle) | TB4 dock |
| Host RAM | >= 16 GB | 16 GB |
| macOS + dext | TinyGPU DriverKit system extension (`org.tinygrad.tinygpu.driver2`) installed and active | — |
| Toolchain | Python 3.11 + numpy; tinygrad fork (patch in `patches/`); Docker (Colima) with a CUDA 12.8 `nvcc` image for cubin builds (build-time only) | — |
| Model files | Qwen3.8-27B GGUF (IQ3_XXS body) and/or Qwen3.6-35B-A3B UD GGUF (IQ4_XS/IQ3_S) — **not included**; the offline repackers build each engine's weight planes | — |

Full details, paths, and gotchas: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## How to run an LLM on a Mac with an NVIDIA eGPU (5 steps)

Details in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

```sh
# 1. prerequisites: tinygrad fork + patches/tinygrad-fork.patch, Python 3.11
#    venv, Colima nvcc image, dext active (docs/GETTING_STARTED.md)
# 2. build the kernels (nvcc shim on PATH)
cd engine && python build_kernels.py && python build_w2.py && python build_hm.py
# 3. pack the weights from your GGUF (one-time; produces packed/ packed5/ ...)
python pack_w1c.py && python q4pack.py && python pack_w7.py && python pack_w5.py
# 4. bootstrap a context snapshot (stock prefill -> engine-layout state)
python bootstrap_w1b.py            # 2k gate state
# 5. run the 2k Tier-1 gate, then bring up the service
python test_w2.py                  # expects Tier-1 60/60 + ~40 tok/s class
#    service (the sanctioned mode = the supervisor; see GETTING_STARTED):
#    generate engine/ops/env.canonical from env.canonical.example, then
#    engine/ops/enginectl install    # launchd-supervised daemon + API
curl 127.0.0.1:8080/v1/chat/completions -d '{"model":"qwen","stream":true,
     "messages":[{"role":"user","content":"hello"}]}'
```

## eGPU vs unified memory: which is faster for LLMs?

The honest, measured answer for *this* workload class (a 27B hybrid model at
100k-token context):

- **Apple unified memory (MLX / llama.cpp / Ollama on the internal GPU)** is
  the capacity champion: a 64-96 GB Mac can *hold* huge models that no 24 GB
  card can. But the internal GPU's compute and bandwidth are the ceiling, and
  a 27B model at 100k context runs far below the numbers above on it.
- **A discrete eGPU (RTX 3090, 24 GB GDDR6X)** is the speed champion: ~940 GB/s
  card bandwidth driven at 94% through the dext, tensor cores, and 24 GB of
  VRAM that exactly fits the 100k-context config (~21.6 GB live). This stack
  exists to make that card usable from macOS.
- The Thunderbolt link is **not** the bottleneck people assume: weights upload
  once and stay GPU-resident; per token the host sends a doorbell, not the
  model. The measured decode gap vs a native-Linux stack on the same GPU
  (75.81 vs 86 tok/s @100k) is attributed in
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md), not hand-waved.

Rule of thumb: if your model fits in 24 GB and you want speed, the eGPU wins;
if you need a 70B-class model resident and can wait, unified memory wins.

## Performance

The condensed tables are above; the full story — decode ladder 40 -> 75.81
(dense) and the MoE/MTP numbers, prefill ladder 21.8 -> 569.2, the benchmark
methodology (what "Tier-1 bit-exact" means as a gate contract), cache
timings, and the measured walls — is [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
For comparison context: the same GPU class running a tuned native-Linux
vLLM stack does 86 tok/s decode / 662 tok/s prefill @100k on the dense
model — ThunderLlamaX's decode is at 88% of that reference from behind a
Thunderbolt cable and a userspace driver, and the gap that remains is
measured and explained, not hand-waved.

## FAQ

**Is CUDA available on macOS?** No — there is no CUDA runtime or NVIDIA
driver for modern macOS, which is why PyTorch, vLLM, and friends can't see an
NVIDIA eGPU on a Mac. ThunderLlamaX replaces the missing driver stack with a
DriverKit dext (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) and compiles
its kernels to raw cubins at build time.

**Does macOS support eGPU in 2026?** On Intel Macs, yes (Thunderbolt 3, AMD
cards). On Apple Silicon, no — eGPU support was dropped with the transition,
and Apple has announced nothing since. This project is the workaround: a
custom open driver that makes an NVIDIA eGPU a first-class local-inference
device on an M-series Mac.

**Can a MacBook Pro / Mac mini / Mac Studio use an external GPU for AI?**
With this stack, yes — any Apple Silicon Mac with Thunderbolt 4 works as the
host, from an M1-generation machine to M4 Max / M5-class laptops and desktops
(the reference rig is a 16 GB MacBook Air M2). The M-chip's speed barely
matters: the Mac does almost none of the compute, it steers the GPU.
TB3-only ports throttle the data path.

**Will eGPU support ever come to Apple Silicon natively?** No public sign of
it. This project stopped waiting and wrote the driver instead.

**Can Ollama or vLLM use an eGPU on a Mac?** No. Ollama on macOS runs on the
internal GPU via Metal; vLLM has no macOS GPU backend at all (it needs CUDA).
That's exactly the gap ThunderLlamaX fills — and for scale reference, the
same 3090 under a tuned native-Linux vLLM does 86 tok/s decode @100k vs our
75.81 through the Mac.

**How does this compare to MLX?** MLX is Apple's framework for *internal*
Apple Silicon GPUs — great for models that fit and tolerate its speed; it
can't touch an external NVIDIA GPU. MPS is PyTorch's Metal backend, same
story. This stack is a third path: the discrete NVIDIA card, driven from
macOS. See [eGPU vs unified memory](#egpu-vs-unified-memory-which-is-faster-for-llms).

**Is it exact?** Yes — by default, *bit-exact*. The Tier-1 contract: the
speculative engine must emit, token for token, the identical sequence a
greedy one-token-at-a-time rollout of the same engine produces — 60/60
bit-exact, deterministic across repeated runs, and identical to the stock
tinygrad greedy baseline (59/59). This holds because every batched kernel
replicates the single-row kernel's floating-point op order exactly, so
acceptance can never flip a near-tie.

**What about the quality-changed mode?** One knob, `PF_W4A8=1`, is the single
authorized Tier-2 numerics change: the prefill FFN GEMM reads the existing
weight planes through a linearized int4 codebook view (a 2.4% logits-class
shift) for +7.3/+6.4/+4.3% prefill speed. Decode is always Tier-1. Unset the
knob and prefill is byte-identical again — verified line-for-line. It ships
on (with a kill-switch) because prefill changes no emitted token stream in the
gates; see [docs/PERFORMANCE.md](docs/PERFORMANCE.md#tier-2-the-one-authorized-numerics-change).

**75 tok/s sounds too good — what's the catch?** The deep-K speedup comes
from the model re-reading text it already has (documents, code, quotes,
repetitions). On the gate workload (a 100k prompt with a repeat region) the
dense engine sustains 75.81; on novel prose the n-gram drafter never fires —
and that's exactly where the two shipped prose modes kick in: the dense
engine drafts from its own EAGLE layer through a full-vocab head (43.7
tok/s through the API, output bit-identical), and the MoE engine
runs its first-party MTP K=4 chain (40.1 tok/s through the API, 0.875
first-draft acceptance). When a mode doesn't pay, that gets published too:
the dense K=4 chain is built and Tier-1-proven but ships **off** — its
1-layer drafter saturates at two accepted tokens, so K=4 measured **-25%
prose** and the default stays K=2 until a deeper drafter exists (break-even
math in PERFORMANCE.md). All of these numbers are measured and published —
see "What this rig can and can't do" in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md#what-this-rig-can-and-cant-do-measured).

**Can it serve more than one model?** Yes — that's the model registry:
declare models in `engine/ops/model_registry.json`, and the service exposes
all of them through `/v1/models` (with residency status). One model is
resident at a time (one 24 GB card, one engine); `enginectl switch <id>`
swaps residents through a crash-safe lifecycle, each model keeps its own
prompt cache, and requests naming a non-resident model get a clean 409 with
a switch hint instead of a wrong-model answer. Three ship today: the
Qwen3.8-27B dense engine, the Qwen3.6-35B-A3B MoE engine, and an
abliterated weight-variant of the dense model — deployed through the same
recipe any Qwen3.8-architecture checkpoint can take (requantize to the
engine's exact tensor-type map, pack offline, register;
[docs/DEPLOY_OBLITERATED.md](docs/DEPLOY_OBLITERATED.md)).

**And the batch number?** The B=2 batched-decode mode (R6) is opt-in and
honest about what it buys: 81.33 tok/s aggregate in the engine harness
(against K2-class solos; per-stream bit-exact at every rung), but measured
through the full service against production deep-K solos it lands at
~1.20x — a concurrency/fairness capability (two streams at ~57-80% each
instead of 100%/queued), not an aggregate multiplier at B=2. The measured
cost model (per-stream stateful kernels + M-trunk widening vs the shared
weight read) is published in
[docs/history/R6_BATCH.md](docs/history/R6_BATCH.md).

**Can I use two GPUs / a dual-3090 setup?** No — the engine targets exactly
one sm_86 card. Multi-GPU is not on the roadmap.

**Apple Silicon only?** The driver layer is macOS DriverKit, so yes — this is
the point. The GPU must be NVIDIA sm_86-class (kernels are tuned for it;
other arches need re-tuning, not just recompiles). Only the RTX 3090 is
tested; treat anything else as an experiment.

**Why not just use a Linux box?** Because a Linux box is easy. This project
exists to prove the Mac route: full-speed eGPU LLM serving on macOS with an
open, inspectable driver stack — and to publish everything it took. (For
reference, the same GPU on a tuned native-Linux stack does 86 tok/s decode
@100k; we measure, publish, and explain the remaining gap.)

**Do I need the model weights?** Yes, separately — Qwen3.8-27B in GGUF
(IQ3_XXS body; per-layer quant mix fixed in the engine) and/or
Qwen3.6-35B-A3B in a UD GGUF tier (IQ4_XS ships; IQ3_S measured). Weights
are governed by their own license and are not part of this repo. The
repackers (`engine/pack_*.py`, `engine/mm/pack36.py`) build everything
else.

**Sampling / temperature?** Greedy only, so far — temp=0 keeps the bit-exact
contract trivially. The sampling kernel is roadmap (M2).

## Project status + roadmap

**Status (2026-09-29):** the service is real and daily-driven — daemon +
OpenAI API + prompt cache under the launchd supervisor, with the environment
split per model (`env.common` + `env.canonical.d/<model>.env`, generated
from the published examples) and the kernel set asserted against rung
manifests at every boot. **Three models serve from the one registry**: the
dense Qwen3.8-27B (K=10 decode finished at 75.81; the full-vocab EAGLE
draft head doubled novel-prose to 43.7 tok/s through the API; the K=4
chain is built, Tier-1-gated, and honestly falsified at -25% until a
deeper drafter exists), the MoE Qwen3.6-35B-A3B (K=8 quote at 97.6-104 tok/s;
first-party MTP K=4 prose at 40.1 — the multi-model campaign MM P0-P10,
including its honestly-falsified fusion, is
[docs/history/MM_PLAN.md](docs/history/MM_PLAN.md) + the MM_P* journals),
and the abliterated dense variant (a third registry entry proving the
weight-only path: any Qwen3.8-architecture checkpoint packs offline and
swaps in at runtime — [docs/DEPLOY_OBLITERATED.md](docs/DEPLOY_OBLITERATED.md)).
The serving layer carries two full review campaigns (the W1-W5 fix waves +
the 51-finding R3 hardening) and the L7 abort-safety protocol; the GPU-free
battery is ~180 tests across serving/api/pcache/L7/P8/swap-FSM. Prefill
sits at its Tier-2 ship with the "750 tok/s" question honestly closed
(full story in [docs/history/T2_P8W4.md](docs/history/T2_P8W4.md)). Batch
serving (B=2, per-stream bit-exact) ships as a documented opt-in.

**Roadmap (roughly ordered):**
- The pcache T1-boundary node class — restore bit-exactness for
  midprefill/turnend CACHE_HIT continuations (FIX_CAMPAIGN finding #5, the
  one open exactness item; FOLLOW_UP, the hot path, is unaffected).
- MoE decode compression via fewer weight passes per layer (the corrected
  P10-B route: the single-CTA expert quartet doing one weight sweep), not
  launch-count fusion.
- MoE long-ctx (96k) MTP alpha battery; the dense first-party-MTP question
  (its shipped drafter is vocab-slice-limited — a full-vocab head retest is
  priced).
- M2: the sampling kernel (temp>0, distribution-exact contract).
- Persistent-CTA GEMM megakernel — the remaining measured prefill route
  (159 ms/chunk GEMM pool; meet-based warp-spec and X-widening are falsified).
- The B=3+ batch axis needs M6+/wider kernel families (the register-wall
  campaign), not more wiring.
- CTA/SM unlock investigation at the driver level (the 1-CTA/SM limit is the
  structural tax behind the remaining walls; the QMD construction is ours).
- Other GPUs (sm_86 is the one tested class) + hardening for non-reference
  hosts; the engine as a more generic Mac-eGPU framework.
- Anthropic-style thin adapter (outside the engine process).

## Repository layout

```
engine/      the DraftHorse engine: ~330 python driver files + ~390 hand-CUDA
             kernel sources (.cu) — trunk, MTP + deep-K LOOKUP cycle (K=2..10
             kernel sets + rung manifests), graph replay, batched prefill,
             prompt cache, serving daemon + api_server, the B=2 batch engine,
             kernel builders + gen_m5..11 generators, decider harnesses;
             serve_moe.py/pcache_moe.py/test_moe36.py = the MoE daemon bridge
engine/mm/    the MoE campaign workbench: the router/gather-GEMV/combine/
             split-KV kernel sources, the repacker (pack36), the MTP chain
             (MM_P9_mtp), the harness libs + Tier-1 banks
engine/tests/ the GPU-free mock battery (~180 tests: serving, api, pcache,
             manifest/tripwire laws, L7 abort-safety, P8 multi-model +
             swap-FSM, the real-listener harness)
engine/ops/  serving ops: enginectl (+ model registry / switch), launchd
             plists, the supervisor wrapper, env.common.example +
             env.canonical.example + per-model env.canonical.d/
engine/docs/ the R3 serving runbook (knob census, drift 503s, alarms)
tools/       standalone probes: dext bandwidth bench, graph-budget probe,
             bootstraps, sync-cost microbench
lineage/     the pre-engine tinygrad-stack MTP work (historical; needs the fork)
docs/        GETTING_STARTED · ARCHITECTURE · SERVING · PERFORMANCE ·
             DEXT_LAWS (the laws of this platform) · DEPLOY_OBLITERATED
             (the third-model worked example)
docs/history/ the lab notebooks — every campaign journal, indexed
baselines/   greedy baseline token files for the exactness gates
patches/     tinygrad-fork.patch — the fork deltas the engine requires
```

## License + credits

MIT — see LICENSE. The tinygrad fork patch is provided under tinygrad's MIT
license. Model weights are governed by their own license (Qwen) and are not
part of this repo.

The designs borrow liberally from the open LLM-inference literature:
per-position state slots and the draft-vocab slice follow the syv-ai/vLLM
3090 recipe; split-KV flash-decoding follows FlashInfer/FA2; softmax cadence
and the MTP wiring follow FlashAttention-2 and the DeepSeek-V3 MTP layout as
implemented in MTPLX; the n-gram LOOKUP drafter is the REST/prompt-lookup
idea rebuilt for this engine. The dext forensics would not exist without the
tinygrad fork's readable NV backend.
