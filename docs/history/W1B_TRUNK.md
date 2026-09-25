# W1-b: full T=1 decode trunk @2k — engine0 whole-model proof

Gate target: **>=25 tok/s sustained T=1 @2k over 60 tokens + agreement report vs stock.**
Result: **20.09 tok/s (49.77 ms/token), GREEDY 60/60 EXACT vs stock baseline.**
Gate MISSED on speed (attribution + blocker below); correctness contract fully met.

## Headline numbers (final run: ~/w1b_trunk.log, 2026-09-13)
| metric | value |
|---|---|
| tok/s (best of 3 x 60-token runs) | **20.09** (49.77 ms/tok) |
| agreement vs stock 2k baseline | **60/60 exact** (engine == stock greedy token-for-token) |
| baseline | regenerated stock greedy from stock-prefilled state (prompt8k.txt truncated to 1988 tok; stock locks into [3204,40224] repeat loop — engine reproduces it exactly) |
| attention-block validation (blk.3, real weights, T=1, pos 1500, poison-first) | y relerr **5.08e-4**, K[pos] 9.46e-4, V[pos] 3.78e-4 (gate 1e-3) PASS |

## Attribution (per token, 60x pipelined, per-rep sync)
| phase | ms/token | launches | note |
|---|---|---|---|
| GDN x48 | 38.54 | 336 (7/block) | ~0.80 ms/block; W1A kernel BWs hold (qkv 404 GB/s, iq3 228, ffn 380) |
| attention x16 | 14.86 | 112 (7/block) | q(Q6_K|IQ3)+k(IQ3)+v(Q4_K) fused GEMV; per-head online-softmax kernel |
| head + argmax | 2.06 | 3 | 874 MB Q5_K GEMV ~= memory floor (100% BW) |
| embed | 0.16 | 1 | IQ3_S gather+dequant, device-resident |
| total measured | 49.77 | 452 | +1 mid-token dev.synchronize (see gotcha 8) |

GPU-side ideal (weights @447 GB/s): ~26 ms -> engine runs GDN at ~50%, attn at ~36% of
achievable BW. Host submit ~85-100 us/launch measured (E1's 49 us was 3-arg trivial
kernels; real kernarg sets cost more), so 452 launches ~= 40-45 ms host floor — BOTH
walls are near 45-50 ms today; kernel BW is the bigger prize (see queue).

## What was built (engine0/)
- `attn_block.cu` -> `a_q6` (Q6_K GEMV, 12288x5120), `a_kv` (k IQ3_XXS + v Q4_K fused),
  `a_attn` (qk-RMSNorm + PARTIAL RoPE + fp16 KV append + gated online-softmax attention,
  24 CTAs = q-head, GQA h->h/6), `a_o` (IQ3_S 6144->5120)
- `head_block.cu` -> `h_embed` (IQ3_S 248320-row gather+dequant from device token slot),
  `h_argmax` (tie-safe block argmax; writes tok_slot + tok_hist[pos]; pos_slot++), `k3a_iq3`
- `merge.cu` -> `k0ab` (norm + alpha/beta GEMV w/ redundant per-warp norm), `k2s`
  (conv+scan+z fused), `a_qkv_q6`/`a_qkv_iq3` (q+k+v one launch), `k1_q5g` (qkv+gate one launch)
- `trunk.py` — TrunkEngine: all 64 blocks + head/embed static buffers (~12.8 GB weights),
  cached per-parity launch sequences, device-resident greedy loop (argmax -> tok_slot ->
  embed; pos from pos_slot; ONE host sync per 60 tokens + mid-token split)
- `bootstrap_w1b.py` — stock prefill (T=1 x1987), snapshot (48 conv/rec + 16 fp16 KV +
  ids/theta), stock greedy 60-token baseline -> ~/w1b_state_2k.npz (291 MB)
- `test_w1b.py` (MODE=attnval|trunk), `build_w1b.py`, `trunk_dbg.py`, `q6dbg.py`

## Model facts decoded this session (law for future kernels)
- attn blocks: 24 q-heads x 256, 4 kv-heads x 256, **attn_output_gate** ([q|gate]
  interleaved per head in the 12288 q-row), qk-norm 256 = head_dim (post-reshape),
  **RoPE PARTIAL: rope_dim=64** (first 64 dims only, pairs (i,i+32), 32-entry freq
  table, theta=1e7), GQA h->h//6, scale 1/16, fp16 KV [2][4][2048][256]
- attn weight types per block: q = Q6_K (8 blocks: 3,7,19,31,43,55,59,63) or IQ3_XXS
  (8: 11,15,23,27,35,39,47,51); k = IQ3_XXS; v = Q4_K; o_proj = **IQ3_S** (256/110B
  blocks — engine0.QUANT[21] said (128,110): WRONG, fixed at import)
- **24 of 48 GDN blocks (8,9,12,13,...,53) have ssm_out = IQ3_XXS (12 MB), NOT Q8_0
  (33.4 MB)** — k3a_oproj on those reads OOB -> deterministic device fault at block 8.
  Dispatch by tensor type in trunk (k3a_oproj | k3a_iq3)
- token_embd = IQ3_S (546 MB, 2200 B/row); output head = Q5_K (874 MB)
- stock RMSNorm = fp32 math -> cast back to input dtype (half) -> * fp32 weight;
  **gate sigmoid must be computed in fp32 then rounded** (half-sigmoid cost 1.3e-3
  relerr on y; fp32 version passes at 5e-4)

## NEW DEXT/GOTCHAS (each cost a debug cycle — banked)
8. **Pipelined-launch ceiling ~312-500 on real kernels**: 612 launches/token with one
   final wait = deterministic device fault (hang); W1A's E1 "512 flat" was 3-arg trivial
   kernels. Mid-token dev.synchronize() at launch ~230-250 keeps both halves safe.
   Attribution loops must sync per rep too (30k un-synced launches fault).
9. **__shfl_xor_sync(FULL,...) inside `if (warp==0 && lane<8)` HANGS the dext** (mask
   says 32 lanes, only 8 execute). Sub-group reduce must use mask 0xff and o=4,2,1.
10. **Q6_K 2-bit chunk index is (lane>>2)&3** (chunk = i>>5, 32-element granularity) —
   NOT (lane>>3)&1. Cost relerr 1.05 on all q GEMVs; numpy lane-view brute force found it.
11. IQ3_S field indices are WITHIN the 256-elem block (g=(e&255)>>2 etc) — using global
   e (embed across 20 blocks) reads in-row but WRONG bytes: plausible-looking garbage,
   0/60 agreement. Symptom: values look sane, tokens diverge from token 0.
12. `from_gguf` returns (model, kv) — take the tokenizer kv from it; a separate
   gguf_load() for SimpleTokenizer = the known +12.6 GB second-load OOM (bit me again).
13. stock model() calls need **start_pos as a bound UOp variable**
   (UOp.variable("start_pos",0,2047).bind(sp)) like generate(); plain ints work for the
   first call then JitError "args mismatch" on replay. The eager `model.forward(...)`
   path takes plain ints (used for the 60-token baseline).
14. `pkill` of a live GPU python wedged the machine into a watchdog REBOOT (~10 min
   lost + /tmp wiped). npz snapshots now live in ~ not /tmp.
15. macOS has no `timeout`/`setsid`; `nohup ... & disown` + separate poll calls is the
   reliable remote-launch pattern (a killed local ssh takes the remote child with it).

## Optimization log (this session)
| step | ms/tok | tok/s | note |
|---|---|---|---|
| first full run | 54.4 | 18.4 | 612 launches, mid-sync, 0/60 (embed bug) |
| embed block-local fix | 54.8 | 18.3 | **60/60 agreement** |
| launch-seq caching | 54.8 | 18.3 | no gain — per-launch cost is in ops_nv path, not glue |
| merges (k0ab, k2s, a_qkv, k1_q5g) -> 452 launches | 49.8 | **20.09** | 60/60 held throughout |

## Queue (W1-c)
1. **16B/uint4 weight loads** (the W1A #1 item, now decisively the top blocker):
   k1_q5g qkv 89->~45 us, k3b_ffn 380->~560 GB/s, k3c_down, a_qkv_q6 — worth ~9-11 ms
   -> ~39-41 ms/token ~= 24-26 tok/s. Then arg-count-independent submit (or graphs).
2. Submission: 85-100 us/launch on the ops_nv path; either drop to ~350 launches
   (deeper fusion) or capture the seq into a graph (W2 direction — MTP arrives there).
3. k1_iq3 occupancy (228 GB/s weakest GEMV; wpc=4 or 2 rows/warp).
4. spec_base_2k.json cross-check: the bootstrap "short" tag crashed (rollout_jit arg
   mismatch after the 2k run captured the jit with t_full slices) — rerun short-tag in
   a fresh process if the old-baseline comparison matters.
5. ckpt the bootstrap state per prompt (the 10-min stock prefill is the awkward part;
   reuse MTP_CKPT infra or accept it).

Run: `cd ~/tinygrad-metal/engine0 && MODE=trunk DEV=NV ~/tg311/bin/python test_w1b.py`
(needs ~/w1b_state_2k.npz from bootstrap_w1b.py; attn validation: MODE=attnval)
