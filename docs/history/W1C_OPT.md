# W1-c: wide-load GEMVs + aligned repack + G_CYCLE graph replay — engine0 @2k T=1

Gate target: >=30 tok/s @2k T=1 (stretch), hard floor >=24 after step 1.
Result: **25.59 tok/s (39.08 ms/token), GREEDY 60/60 EXACT vs stock baseline**
(graph replay, sync_every=2, unroll-5 GEMVs). Step-1 gate (>=24) PASSED; the
30-gate missed by ~4.4 tok/s -- attribution + remaining levers below (all
kernel-GPU-time now; the host floor is gone).

## Headline ladder
| config | ms/tok | tok/s | agree |
|---|---|---|---|
| W1-b baseline (452 py launches) | 49.00 | 20.41 | 60/60 |
| + wide-load GEMVs (step 1) | 48.24 | 20.73 | 60/60 (host-bound: ~85us/launch floor) |
| + G_CYCLE graph replay, sync_every=1 | 40.42 | 24.74 | 60/60 |
| + G_CYCLE sync_every=2 | 39.36 | 25.41 | 60/60 x3 reps |
| + unroll-5 GEMV loops | **39.08** | **25.59** | 60/60 x3 reps (kval 13/13 bit-equal) |

## What was built
- `w1c.cu` -> 8 wide-load kernels (per-kernel cubins, build_w1c.py): `q5g8`
  (Q5 qkv + IQ3 gate), `head8`, `ffn8`, `down8`, `op38`, `aq6k8`, `aq3k8`, `ao8`.
  All 13 output tensors BIT-EQUAL vs the W1-b kernels on real weights (kval).
- `pack_w1c.py` -> `engine0/packed/*.npy` (7.64GB): aligned-repacked IQ3_XXS
  (gate/fg/fu/fd/iq3-ssm_out/q-iq3/k) + Q6_K q. Q5_K/Q4_K/Q8_0/IQ3_S stay raw.
- `trunk_w1c.py` -> TrunkEngineW1C (loads packed tensors, launches wide kernels).
- `test_w1c.py` (MODE=kval|trunk), `gcycle.py` + `test_gcycle.py` (graph replay).

## THE ALIGNMENT LAW (new dext gotcha, root of the W1A 'uint32 fault')
**nvcc merges adjacent narrow loads into wider ones (2x u16 -> 1 u32) and the
merge is legal ONLY if the merged address is naturally aligned.** IQ3_XXS blocks
are 98B -> on odd blocks the scale-word u32 lands at 2-mod-4 -> SM Multiple Warp
Errors (hard fault, empty error report). This is what actually faulted in W1A
("uint32 loads fault on Q5 buffer") -- width was never the problem; ALIGNMENT of
compiler-generated merged loads is. PTX-diff proved it (`ld.global.nc.u32`
present in the faulting build, absent in the passing one, same source semantics).
RULE: every per-lane multi-byte run must sit at its natural alignment for every
block parity, or be loaded at widths that can never merge into an unaligned one.
Wide loads themselves (u16/u32/u64/uint4/float4) are ALL legal on this dext
(probe_w1c.py; the W1A note was a misdiagnosis of the alignment issue).

Packed layouts (both keep decoded values bit-identical):
- IQ3_XXS row (98B*NB, SAME SIZE): [qs 64B/blk][scales 32B/blk][d 2B/blk]
  -> qs u16 @64b+2lane, scale u32 @32b+4*s, d u16 @2b -- all natural-aligned.
- Q6_K row (212B*NB, +1%): [pad2][d2][sc16][lo128][qh64] -> 2x u32 runs/lane.
- Q5_K (176B blocks) + Q4_K (144B): already 8B-aligned -> RAW + u64 runs/lane
  (8 qs bytes + 8 qh bytes each = ONE u64 load per run).

## G_CYCLE architecture (gcycle.py)
One NVComputeQueue per conv-parity (2 graphs total) containing the ENTIRE token:
h_embed -> 64 blocks (48 GDN + 16 attn) -> head -> argmax = 452 kernels.
Kernels chain via **QMD dependent pointers** (the ops_nv exec active_qmd path) --
the pushbuffer (_q) is only ~40 words: memory_barrier + timeline wait + 2 nvm
(SEND_PCAS_A/PCAS2_B for the first QMD) + release semaphore bound to the last
QMD. Per token: patch 2 timeline values -> submit (one small _q copy into the
shared cmdq ring + ONE gpfifo entry + ONE doorbell) = ~50-100us host.
All data flow is device-resident (argmax -> tok_slot -> next token's h_embed,
pos from pos_slot) -- ZERO host work inside the loop; NO bind() (hw_page path
unproven on this dext; the MTP_GRAPH_NOBIND shared-ring path is the safe one).
Launch count per token: 452 kernels, 1 gpfifo entry, 1 doorbell.

### New fork quirks found (banked)
16. **Orphaned last timeline value**: staged-copyin signals can be lost; the
    fork's synchronize()/HCQGraph both anchor waits on `timeline_value - 1`.
    A graph waiting on `timeline_value` deadlocks. (Reproduced in isolation.)
17. **In-flight kernel ceiling applies to graphs too**: ~900 kernels queued
    ahead is OK (sync_every=2 = 904 in flight, 3x60 stable); ~3.6k (sync=8)
    faults with Device fault detected. Pipeline depth 2 max on 452-k graphs.
18. `HWQueue.exec` treats a tuple grid as a SYMBOLIC value -> TypeError at
    submit ("memoryview invalid type") -- pass plain ints.
19. GGUF tensor dims are [in, out]: GEMV rows = dims[1] (packer assert caught it;
    flat-byte reshape hides the transpose -> silent garbage).

## Attribution (per token; eager per-phase includes ~10ms py+sync overhead that
the graph removes; graph total = 39.36ms)
| phase | eager ms/tok | note |
|---|---|---|
| GDN x48 | 32.88 | was 38.85 (-5.97: wide GEMVs) |
| attention x16 | 14.10 | was 14.87 (-0.77) |
| head+argmax | 1.91 | was 2.05 (head8 ~at 470 GB/s = wall) |
| embed | 0.15 | |
| graph total | 39.36 | host floor GONE (was ~40ms of submits) |

## Remaining levers to 30+ (W2/W1-d queue, ranked by est. prize)
1. **k2s scan grid split** (~2ms): 48 CTAs = 12K threads on 82 SMs -- the scan
   phase is 87us/block latency-bound; split to 384 CTAs (one warp-group per CTA
   recomputing the per-warp prep) -> ~45us. Needs conv/z phase restructure.
2. **GEMV ILP/occupancy**: GDN GEMVs now ~70-75% of the 445 GB/s wall; try
   unroll 5 (full 20-block unroll) / 2 rows per CTA on q5g8+ffn8 (~2-3ms).
3. **a_attn widening** (24 CTAs latency-bound; ~1-2ms at 2k, grows with ctx).
4. Pipelining ceiling measured: K=1 (452 in flight) and K=2 (904) stable,
   K=4 (1808) and K=8 fault -- in-flight ceiling ~1000-1800 kernels on graphs.
   unroll-5 bought only +0.18 tok/s (GEMVs are latency/BW-bound, not ILP-bound).
5. 100k ctx: attention kernels (a_attn KV loop + ao) are the only ctx-scaling
   part; the graphs are ctx-static (pos from pos_slot device-side) so the G_CYCLE
   architecture carries over unchanged -- only CTX-sized buffers change.

## Run
`cd ~/tinygrad-metal/engine0 && SYNC_EVERY=2 DEV=NV PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH ~/tg311/bin/python test_gcycle.py`
(needs ~/w1b_state_2k.npz + packed/ from pack_w1c.py; kval: MODE=kval test_w1c.py)
