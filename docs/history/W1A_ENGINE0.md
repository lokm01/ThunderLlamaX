# W1-a: engine0 — fused GDN block (T=1) on real weights

First organ of the thin engine: static raw NV buffers + hand CUDA kernels issued
through the fork launch path (NVProgram/TinyELF), no tinygrad scheduling, no graphs.

## E1 — sync/launch costs (e1_sync.py, trivial kernel, wall-clock)
| measurement | cost |
|---|---|
| launch + immediate wait, per op | 101.2 us |
| pipelined per-launch (N=8..512, flat) | ~49 us |
| back-to-back launch+wait on completed timeline | 143 us |
=> fixed sync ~50us; submission floor ~49us/launch. 10 launches/block = ~0.49ms
submission per block — the engine loop must stay GPU-bound or move to graphs.

## Validation (test_w1a.py) vs stock GDN block 0 on REAL weights, T=1
GREEDY-path reference (MTP_FUSED_T1 stock _attention_t1), same x/conv/rec inputs:
| output | relerr | gate |
|---|---|---|
| hidden y [5120] | **1.25e-04** | 1e-3 |
| recurrent state [48,128,128] | **1.28e-04** | 1e-3 |
| conv state [3,10240] | **2.06e-04** | 1e-3 |
PASS. Stage-level: xh 0.0, qkv 2.1e-4, gate 3.7e-4, alpha/beta 1.2e-4,
scan/core 1e-7 (self), z 6e-4, gact 6.9e-4, down 1.6e-4.

## Timing (block-0 weights x48, pipelined, one wait)
- **0.612 ms/block** end-to-end incl. 10-launch submission; kernel-attribution
  sum 0.75ms of which ~0.16ms is per-launch floor on 4 tiny kernels ->
  GPU-only ~0.59 ms/block ~= the 0.6 gate. 183.5 MB weights/block -> ~300 GB/s
  effective across the whole block.
| kernel | us | GB/s | bytes |
|---|---|---|---|
| k1_q5 (qkv Q5_K 10240x5120) | 89 | 404 | 36.0 MB |
| k1_iq3 (gate 6144x5120) | 53 | 228 | 12.0 MB |
| k1_ab (alpha/beta f32) | 38 | 51 | 2.0 MB |
| k2_scan (conv+norm+scan) | 87 | 77 | state 6MB+ |
| k3a_oproj (Q8_0 5120x6144) | 75 | 445 | 33.4 MB |
| k3b_ffn (gate+up 2x17408x5120) | 180 | 380 | 68.2 MB |
| k3c_down (17408->5120) | 111 | 307 | 34.1 MB |
| k0/k2b/k3m (tiny) | ~40 each | launch floor | |

## Numerics contract (stock tinygrad replication)
half(x*rms*nw) inputs; quant GEMVs = hmul(half x, half w) fp32-acc, HALF outputs;
alpha/beta fp32 GEMVs; conv/scan fp32; qk L2 normalize = x/max(||x||,eps), q *= 128^-0.5;
alpha = exp(softplus(a+dtb)*ssm_a); z = half(ssm_norm(core) * hsilu(gate));
hsilu via hexp2/hrcp const -1.4423828125; y = hh + half(down-acc).

## NEW DEXT GOTCHAS (each cost a fault+reboot cycle — all now banked)
1. **One float4 = 8 halves** on __half* buffers (a3e patterns with X as float*
   use +4 second loads — copying them to half buffers = 4-half OOB read).
2. **Multi-kernel nvcc cubins mis-load**: identical k1_q5 code faults from a
   10-kernel cubin while passing from its own cubin -> per-kernel cubins
   (build_kernels.py). E4 4-kernel cubin was fine; somewhere between 4 and 10.
3. **uint32 (4B) vector load on the Q5 weight buffer faults** ("Out Of Range
   Register"); byte loads pass. uint16 (scw) loads pass. float4 on xh passes.
4. **Lane-strided loop starts miscompile**: `for(b=lane; b<192; b+=32)` gave
   wrong sums (kernel != numpy by 1.04 relerr, full bytes read); sequential
   `for(b=0;b<192;++b)` fixed it exactly. Keep loops sequential-stride.
5. Raw-allocator _copyin is ASYNC: uploaded numpy arrays must be kept alive
   (Bufs._keep) or freed-and-reused source memory = deterministic garbage uploads.
6. GGUF data_start must be computed AFTER the tensor-header table (21KB shift).
7. Tensor.repeat = TILE (torch) semantics: v-head h <- k-head h%nk, NOT h/(nv/nk).

## Fork-layout truths decoded (Qwen3.8-27B-IQ3_XXS)
- Q5_K qs: byte = (k>>6)*32 + (k&31), nibble = (k>>5)&1 (lane-constant!),
  qh byte = k&31 bit = k>>5, scale sub = k>>5 (truth-fitted vs stock dequant).
- reshape(*reversed(dims)) = FLAT reinterpret of disk bytes, NOT transpose.
- IQ3_XXS 98B blocks exactly per a3b _IQ3SW_BODY; grid = float[256][4] from
  int64 words (little-endian 4 magnitude bytes per word).
- GDN block0 types: attn_qkv Q5_K, ssm_out Q8_0, alpha/beta/norms/conv1d F32,
  attn_gate/ffn_* IQ3_XXS. 183.5 MB/block.

## W1-b needs
1. k1_q5 at 404 GB/s vs k3as 445 and E4s 843 (uint4): retry 16B loads on the
   Q5 buffer (uchar4/uint32-after-dext-investigation) -> qkv ~89->45us.
2. k1_iq3 228 GB/s weakest: grid 768 CTAs (6144 rows / 8) — try wpc=4 (1536 CTAs)
   or 2 rows/warp; k3b 380 GB/s (try splitting gate/up kernels).
3. Merge tiny kernels (k0+k1_ab?, k2b into k2 via single-array smem + syncthreads
   test) to cut launches 10->6; then submission ~0.29ms/block.
4. Engine host loop: chain y->x across 48 blocks (ping-pong x/y buffers), then
   the attention blocks (16) + head — the W1-b milestone.
5. The multi-kernel-cubin boundary (4 OK, 10 bad) and the uint32-load fault
   class deserve a fork-maintainer question.
Code: engine0/ (engine0.py runtime, gdn_block.cu master + per-kernel .cu/.cubin,
build_kernels.py, test_w1a.py, debug_*.py, e1_sync.*). Run:
`cd ~/tinygrad-metal/engine0 && PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
DOCKER_HOST=unix://<colima-socket> DEV=NV
~/tg311/bin/python test_w1a.py`
