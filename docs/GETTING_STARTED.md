# Getting started — from bare metal to a running service

This guide takes you from "a Mac and a spare GPU" to a running ThunderLlamaX
service. It absorbs the old SETUP.md. Time budget for a first setup: an
afternoon (weight packing and kernel builds are one-time).

If anything faults along the way, read
[DEXT_LAWS.md](DEXT_LAWS.md) before debugging — the failure signature almost
always matches a known law, and a couple of the laws say "reboot first, then
think."

## 1. Hardware requirements

| Component | Requirement | Notes |
|---|---|---|
| Mac | Apple Silicon with **Thunderbolt 4** | developed and gated on a 16 GB MacBook Air M2; >= 16 GB host RAM |
| GPU | NVIDIA **sm_86** class (Ampere) | developed and gated on **RTX 3090 24 GB** — the reference rig |
| VRAM | **24 GB for the 100k-context config** (~21.6 GB live) | a 2k-context setup fits comfortably in less; see the memory map in [ARCHITECTURE.md](ARCHITECTURE.md) |
| Enclosure | TB4 eGPU box or dock | TB4 end-to-end (TB3 links throttle the data path) |
| Power | the usual for the GPU class; a 3090 wants ~350 W+ headroom | see thermals below |

**Which GPUs are tested?** Exactly one: the RTX 3090 24 GB (sm_86). The
kernel set is sm_86-specific — block sizes, `__shfl_sync` trees, `PRMT` int8
dequant, register budgets — so other Ampere cards will likely work with
re-tuning rather than recompiling, and non-Ampere arches need real porting.
Treat anything but a 3090 as an experiment.

**Power and thermals.** The eGPU enclosure needs to feed the card; under
sustained prefill the 3090 pulls its full board power. Two operational notes
from the rig: (a) a *dock-power fault class* exists — if the GPU vanishes or
faults deterministically at the same step after running fine, physically
unplug/replug the dock power; an EFI-level cold cycle (below) handles the rest;
(b) the Mac itself stays cool — nearly all compute is on the GPU; the host
just steers.

## 2. Software prerequisites

1. **The TinyGPU DriverKit system extension**
   (`org.tinygrad.tinygpu.driver2`) installed and activated. This is the hard
   requirement: the whole engine talks to the GPU over raw PCIe through this
   dext — there is no CUDA runtime anywhere. macOS developer tools are needed
   to build/enable it.
2. **The tinygrad fork** — the engine uses the fork's NV backend as its driver
   runtime (NVProgram/TinyELF/NVComputeQueue):

   ```sh
   git clone https://github.com/tinygrad/tinygrad
   cd tinygrad && git apply /path/to/ThunderLlamaX/patches/tinygrad-fork.patch
   ```

   The patch (50 commits, 19 files) carries the compile-server loop-read fix,
   the eager/graph kernargs pool split, the PCIIfaceBase munmap fix, the
   shared cmdq-ring graph submit path, the name-gated smem-carveout override
   (`NV_SMEM_CFG`), MTP_* env-gated hooks, and debug instrumentation.
3. **Python 3.11** venv with `numpy`, plus the tinygrad fork installed
   editable.
4. **Docker (Colima) with a CUDA `nvcc` image** — cubins are built by an
   `nvcc` shim that execs into a `cuda-nvcc:12.8` container. Build the image
   with tinygrad's `extra/setup_nvcc_osx.sh`, then verify:

   ```sh
   nvcc -arch=sm_86 -cubin ...   # any smoke source; see the shim docs
   ```

   After every reboot, wait for the docker VM before building.

### Path placeholders

Historical scripts and run lines use the original rig's layout. Replace
before running:

| placeholder | replace with |
|---|---|
| `~/tinygrad-metal` | your clone of this repo (scripts assume the original flat layout — for `lineage/` scripts, run from inside `lineage/`) |
| `~/tinygrad-src` | your patched tinygrad fork checkout |
| `~/tg311/bin/python` | your python3.11 venv interpreter |
| `unix://<colima-socket>` | your docker socket, e.g. `unix://$HOME/.colima/default/docker.sock` |
| `PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH` | wherever your nvcc shims live |
| `~/snap100k`, `~/w1b_state_2k.npz` | your bootstrap snapshot locations |
| `$TMPDIR/nv_usb4.lock` | the GPU device lock — `sudo rm -f` it if a killed run leaves it stale |

Every GPU command needs `DEV=NV` (else tinygrad falls back to Metal and OOMs
the Mac — harmless but wasted).

## 3. Model preparation

- **GGUF**: `Qwen3.8-27B-IQ3_XXS` (bartowski). The engine hard-codes this
  model's exact per-layer quant mix (documented in
  [docs/history/W1B_TRUNK.md](history/W1B_TRUNK.md)). Weights are NOT in this
  repo — place the file under `models/` (gitignored) or point the packers at it.
- **Offline repack** (one-time; regenerates everything the gitignore excludes):

  ```sh
  cd engine
  python pack_w1c.py     # aligned IQ3/Q6 packs  -> packed/
  python q4pack.py       # Q4_0 draft pack       -> draft_pack/
  python pack_w7.py      # prefill W7 wide-tile plane (shared with decode as packed7)
  python pack_w5.py      # packed5 Q5_K qkv plane -> packed5/
  ```

- The a3b override JSONs (`lineage/a3b/override*.json`) reference their
  template `.cu` files relative to the `lineage/` directory.

## 4. Build the kernels

```sh
cd engine
python build_kernels.py && python build_w2.py && python build_hm.py
# (the build_*.py scripts enumerate the rest of the families; nvcc shim
#  must be on PATH and the docker VM up — see prerequisites)
```

## 5. Bootstrap a context snapshot

The engine resumes from a post-prefill snapshot in engine layout:

1. Run the stock tinygrad prefill over your prompt to the resume position
   (checkpointed; `MTP_CKPT=1` infra) — or use `tools/bootstrap_100k.py` /
   the `MTP_SNAP100K=1` hook in `lineage/mtp_v3.py`, which resumes from a
   checkpoint, finishes the chunked prefill to P=97810 and writes `~/snap100k/`
   (48 GDN conv/rec states + 16 fp16 KV caches + ids/theta).
2. The draft KV is prompt-filled once at engine start (`fill_draft`,
   ~6 min at 100k; the slice table is built from the real prompt's frequency).

For the 2k gate, `python bootstrap_w1b.py` produces the 2k bootstrap state.

## 6. First launch — the canonical environment

The canonical 100k gate/bench (the recipe that produced the published
numbers):

```sh
cd engine && env SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 \
  QH=1 PVH=1 HM=1 DO_T1=0 DEV=NV \
  PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH \
  DOCKER_HOST=unix://<colima-socket> \
  python -u test_w100k.py
```

Expected: `[tier1] 60/60 exact`, phase splits draft ~5 / probe ~60 / accept ~1.
The 2k gate is `python test_w2.py`; kernel-level differentials live in
`test_g.py` / `test_t.py` / `test_v3.py` (poison-first, vs T=1 references).

### The service — the supervisor mode (the sanctioned way)

The shipped way to run the service is the launchd supervisor (the TLX W2/W5
campaign hardened it: a persistent circuit breaker, env single-sourcing, and
self-heal across GPU-fault reboots). One-time setup:

```sh
cd engine/ops
# 1. generate your env.canonical (the live file is NOT in the repo — it
#    carries the admin token; the example carries a placeholder):
cp env.canonical.example env.canonical && chmod 600 env.canonical
TLX_TOK=$(openssl rand -base64 24)
sed -i '' -e "s|REPLACE_WITH_openssl_rand_base64_24|$TLX_TOK|" \
          -e "s|~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf|/path/to/your/Qwen3.8-27B-IQ3_XXS.gguf|" \
          env.canonical
# 2. personalize the launchd plists (they ship with a %USER% placeholder):
sed -i '' "s/%USER%/$USER/" com.tlx.llm-engine.plist com.tlx.llm-api.plist
# 3. install + start (engine boot ~6-7 min; watch it come up):
./enginectl install
tail -f ~/tinygrad-metal/logs/llm-engine-launchd.log
```

What this buys over a manual boot:

- **The wrapper** (`engine_daemon.sh`) runs the engine python as a child so
  exits are counted (3 crash-class exits / 10 min -> stay-down marker ->
  refusal, protecting the dext from crash loops); breaker state persists
  under `logs/` (survives the fault-reboots that wipe `/tmp`).
- **Env single-sourcing**: the plist carries NO env of its own — the wrapper
  sources `ops/env.canonical`, logs its sha256 digest, and the daemon's
  config fingerprint surfaces in `/health`; the API drift-checks it (503
  `config_drift` on mismatch). A relaunch can never silently boot a stale
  env subset.
- **Self-heal**: `RunAtLoad` + `KeepAlive` bring both services back after the
  GPU-exit reboots this platform occasionally takes (see the GPU-EXIT law in
  [DEXT_LAWS.md](DEXT_LAWS.md)).
- `enginectl` subcommands: `status` / `stop` / `restart` / `logs` /
  `clear-breaker` / `install` / `uninstall`.

Notes: manual runs of `ops/engine_daemon.sh` are equivalent (just slower to
boot under launchd's Background process type). Expected boot warns with
`NV_GRAPH_ASSERTS=1` (the default tripwire mode): `op38nw32_3 8B`,
`spk_g4nw32hm11_100k 8B`, `spk_pre11qh_100k 32B`, and `pfa32ct 104B` (the
exempt Tier-2-frame class) — any OTHER `[NV-GA]` line or a hard trip means
investigate. Ops paths can be relocated via the `TLX_OPS_ROOT` /
`TLX_LOGS_DIR` / `TLX_ENGINE_DIR` envs.

### The service — the legacy manual env-line (kept for reference)

Before the supervisor existed the daemon was booted by hand with the full env
line. It still works (the wrapper just wraps exactly this); kept here as the
reference form of every knob in one place:

```sh
cd engine && env PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH \
  DOCKER_HOST=unix://<colima-socket> DEV=NV M1A_SERVE=1 \
  SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 \
  M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 MTP_KERNARGS_MB=256 \
  LOOKUP_K=10 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 \
  NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 \
  PF_SCAN32=1 PF_M64=1 PF_M128=1 PF_DR7=1 PF_ATTNW=1 PG_SPLIT=4 PF_SCANC=1 \
  PF_SCANC_N2=1 PF_M64QKV=1 PF_ABW=1 PC_ENABLED=1 PF_RING4=1 PF_QKV1=1 \
  PF_P5=1 PF_OP64=1 PF_W4A8=1 NV_SPILL_EXEMPT_NAMES=pf,p8 \
  python -u test_w100k.py &
python api_server.py &     # 127.0.0.1:8080
```

(`LOOKUP_K=10` is the shipped K=10 deep set — the kernel sets for K=2..10 all
ship in `engine/`. `LOOKUP_K=0` gives the plain K=2 engine. The env knobs are
explained below; `env.canonical.example` is the same line as a file.)

Boot to ready: ~11 s weights (warm cache) + ~40 s KV int8-quantize + warmup +
graphs ≈ 6-7 min at 100k ctx. VRAM at 100352 ctx + KV8 ≈ 17.7 GB before the
prefill planes; steady state with all planes is ~23.5 GB of 24 GB.

### The GPU-free test battery

`engine/tests/` carries the mock-engine battery (85 tests: serving
correctness, api mocks, pcache hardening, manifest/tripwire laws) — it runs
without a GPU and is the fastest sanity check after touching the serving
layer:

```sh
cd engine && python3 tests/test_api_mocks.py && \
  python3 tests/test_pcache_w3.py && python3 tests/test_w4_engine.py
```

### What the knobs mean

| Knob | Meaning |
|---|---|
| `DEV=NV` | use the raw-PCIe NV backend (else Metal fallback) |
| `M1A_SERVE=1` | boot the engine as the service daemon (unix socket) — the host-process boot law applies, see [SERVING.md](SERVING.md) |
| `SKV=1 SKV_K=g4nw32 SKV_S=256` | split-KV attention: family `g4nw32` (1024-thread fat CTAs), S=256 splits @100k (S=32 @2k) |
| `SKV_CTXK=100352` | KV context capacity (100k + margin) |
| `GEMVV=1` | the half2/fat-CTA dequant-GEMV winners |
| `KV8=1` | int8 KV cache (biased u8 + per-row fp16 scales; -3.1 GB, bit-exact) |
| `QH=1 PVH=1` | half2 QK / PV dot cadences |
| `HM=1` | HMMA `m16n8k16` tensor-core attention phase |
| `M1A_KEEPALIVE_S=10` | socket keepalive |
| `M1A_GEN_REBUILD_EVERY=256` | graph rebuild + re-anchor cadence — the ~950-cycle dext budget law (never omit) |
| `MTP_KERNARGS_MB=256` | kernargs pool size for superchunk launch volumes |
| `LOOKUP_K` | deep-K n-gram drafter depth: 0 = K=2 MTP only; the kernel sets for 4..10 all ship in `engine/` (10 = the 75.81 config) |
| `PF_PREFILL=1` | enable the batched chunked prefill pipeline (everything `PF_*` below is a prefill rung; all kill-switched) |
| `PF_N32/PF_M64/PF_M128` | M-row trunk widths (32/64/128-row chunks, tail folds M128 -> M64 -> M32) |
| `PF_GEMM3/PF_ABW/PF_RING4/PF_QKV1/PF_M64QKV/PF_ATTN32` | batched GEMM families, wide loads, DBUF ring-4 pipelining, consolidated attnqkv launch |
| `PF_DR7=1` | decode/spec GEMVs read the shared packed7 plane (one weight copy for decode + prefill) |
| `PF_ATTNW=1` | wide-M (ROWS=32/64) attention windows |
| `PF_SCANC=1 PF_SCANC_N2=1` | the WY-C32 chunk-level scan (NC=2; NC=4 under M128) |
| `PG_SPLIT=4` | prefill graph split factor |
| `PF_P5=1 PF_OP64=1` | packed5 qkv repack + the o-proj M-grid fold |
| `PF_W4A8=1` | the Tier-2 W4A8 IMMA prefill ffn — the ONE authorized numerics change (unset = byte-identical Tier-1 prefill) |
| `PC_ENABLED=1` | the durable prompt cache (LongMemory) |
| `NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c` | dynamic smem carveout for the big prefill kernels |
| `NV_SPILL_EXEMPT_NAMES=pf,p8` | fork tripwire scoping — the Tier-2 prefill families run 104-592 B stack FRAMES by design (warn-only for these names; the hard >100 B spill law stays for the decode/canon set) |
| `TLX_ADMIN_TOKEN` / `TLX_MODEL_PATH` | ops: the privileged-RPC token + the model path feeding the config fingerprint (live in env.canonical, not the repo) |
| `BATCH_B=2` (+ R6 knobs) | the opt-in batch serving mode — see [SERVING.md](SERVING.md) and [history/R6_BATCH.md](history/R6_BATCH.md) before flipping |
| `DO_T1=0` | (bench harness) skip the T=1 reference pass |

## 7. Verifying it works

```sh
# daemon health (also via the socket: see SERVING.md)
curl -s 127.0.0.1:8080/health

# first chat completion, streamed
curl 127.0.0.1:8080/v1/chat/completions -d '{
  "model": "qwen", "stream": true, "max_tokens": 64,
  "messages": [{"role": "user", "content": "hello"}]}'
```

Sanity checks that all pass on a healthy boot: `/health` returns ready (503
while loading); the completion streams coherent text and stops naturally;
`usage` reports `cached_tokens` (0 on a fresh conversation, >0 when the
prompt cache hit); a second identical request returns in seconds (resident
prefix reuse). Full API reference: [SERVING.md](SERVING.md).

## 8. Troubleshooting

**Read [DEXT_LAWS.md](DEXT_LAWS.md) first.** The failure signature almost
always matches a known law — alignment, name-encoded launch size, stale bake,
lone graph, in-flight ceiling, multi-kernel cubin, one-attempt-per-boot.

The short version of the fault playbook:

- **A faulted dext poisons every later process.** Fresh processes fault at
  the first wait after a device fault. Reboot, remove a stale device lock
  (`sudo rm -f $TMPDIR/nv_usb4.lock`), wait for the docker VM, then rebuild.
- **Never `kill -9` (or pkill) a live GPU python.** It frequently wedges the
  channel into a watchdog reset (~10 min + `/tmp` wiped). Let runs finish,
  SIGTERM (drains and synchronizes), or rely on the 30 s wait-timeout suicide.
- **The EFI cold-cycle procedure** (clears dext fault classes that warm
  reboots can't — deterministic faults at weight download, dock-power-linked
  fault classes):

  ```sh
  sudo pmset schedule poweron "MM/DD/yy HH:MM:SS"   # ~3 minutes out
  sudo shutdown -h now
  # ... the Mac EFI-level powers back on (full dock + GPU power cycle);
  #     ssh back in ~3-4 min, re-wait for docker, rebuild
  ```

- **Numbers look ~10% slow after a crash-boot?** Post-crash machine state
  degrades perf without changing numerics (documented in
  [docs/history/P18_GROWTH.md](history/P18_GROWTH.md)). Cold-cycle before
  trusting perf numbers.
- **`git status engine/` after adding cubins** — the name-clobber trap; never
  trust an on-disk cubin without confirming its `-D` bakes.
- **Daemon ops**: use `engine/ops/enginectl` (`status` / `stop` / `restart` /
  `logs` / `install` / `clear-breaker`). The circuit breaker (3 crashes/10
  min -> stay-down + 503) protects the dext from crash loops. Under the
  supervisor, logs and breaker state persist under `logs/` (the older
  `/tmp/*.log` paths are wiped on fault-reboots — the reboot-survivor law).
- **FRESH replies answer the previous request's prompt?** That was the P0
  stale-feed bug — fixed (see [SERVING.md](SERVING.md)); if you see it again,
  `engine/p0_repro.py` is the different-content-prefill harness.
