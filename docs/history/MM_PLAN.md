# ThunderLlamaX MULTI-MODEL PLAN — Qwen3.6-35B-A3B (the unified plan; 4-analyst + dispatcher-verified)
# Mission: add MoE model support. Targets (HONEST): decode 140 = quote-class headline / P50 60-110 spec-mix @64-128k;
# prefill 850-1400 with ported machinery (3,300 needs chunk>=256 + perf campaign); ctx 96k default / 128k opt-in / 262k NO.

## VERIFIED MODEL SPEC (ground truth: HF config.json + transformers modeling_qwen3_5_moe.py + llama.cpp source)
- 40 layers = 30 GDN(linear_attention) + 10 full-attn (full_attention_interval=4); hidden 2048; vocab 248320 (Qwen2Tokenizer family, SAME as Qwen3.8); eos <|im_end|>
- GDN/layer: in_proj_qkv 2048->8192, in_proj_z ->4096, a/b ->32, conv1d(8192,k=4), out_proj 4096->2048 ≈ 33.7M → 1.01B total
- GDN state: (32 v-heads, 128 k, 128 v) fp32 = 2MiB/layer → 60MiB/slot, 660MiB @ K+1=11 slots. ⚠️ 32× larger per-layer state than Qwen3.8 — k2s rescale is NOT a dims change (HMMA-ize the S(I−βkᵀ)+vkᵀ update over [32,128,128] tiles may be needed) — WEEK-1 BENCH
- Attn/layer (10): 16Q:2KV @ head_dim 256 (matches our spk K-loop depth ×2); q_proj DOUBLED 2048->8192 chunk->(Q,gate), out = attn_out × sigmoid(gate) → FUSE INTO attn_c EPILOGUE; qk-RMSNorm per-head-256; fp32 softmax; partial RoPE 64 dims/theta 1e7 IDENTICAL to ours; mrope = text no-op; 27.27M/layer → 273M
- MoE ALL 40 layers (incl. GDN layers): 256 routed experts top-8 + 1 shared @ moe_inter 512; 3.146M/expert → routed 32.21B; router 2048->256; softmax fp32 → top8 → RENORM top-8 to 1 → cast; shared gated by sigmoid(Linear(2048,1))
- ⚠️ RMSNormZeroCentered: out = x_norm × (1.0 + w) — OUR NORM KERNELS MUST CHANGE or every block drifts
- MTP: 1 layer, shared embed/head (0.83B) — DROP at repack (−0.4-0.5GB); mmproj separate — NEVER LOAD
- Totals: 35.1B w/ MTP; active compute ~2.06B (+embed/head ≈ 3B card number); llama.cpp arch = qwen35moe, MoE via ggml_mul_mat_id; artifacts: unsloth UD-IQ4_XS 17.4GB (primary) / UD-IQ3_S 14.6 (aggressive) / UD-IQ3_XXS 13.4 / Q8_0 36GB = eval oracle only

## THE ARCHITECTURE DECISION (kimi design (d) + mimo N1-N5; the only shape legal under our graph laws)
DECODE (per layer, 4 kernels): rt8e256 (router GEMV fp32-acc + fixed-order top-8 [8× masked-argmax passes, tie→lower-id; no runtime-indexed locals] + renorm + sigmoid shared-gate; writes canonical-order pair list pairs[K*8]=(eid,row,gate) row-major/rank-major — NO SORT) → gx8e256_up (grouped GEMV, FLAT sequential pair walk [NO grid-stride], expert-pointer table, per-expert 16B-aligned packed slabs [gate/up 2×512×2048], SiLU·gate epilogue → moe_act[pair][512]) → gx8e256_dn (down [2048,512] → moe_part[K][8][2048] fp16 partials) → mx8e256_cmb (fixed-rank-order fp32 combine × gates + shared-expert FFN [dense q5g8v-class, always resident]).
- Expert-major layout [E][3 mats][N,K] packed7/packed4, 16B-aligned; 1.67MB/expert slab → trivial addressing base + e×1.67MB
- ~160 MoE + ~120 dense kernels/cycle ≈ 280 nodes — ONE graph family, 256-cycle GLOBAL rebuild, depth-1 pipelining (650 nodes × 2 > 904 in-flight)
- Bytes/cycle: K=2 ≈2.2GB / K=8 ≈4.9 / K=10 ≈5.7 (expected-unique experts 15.6/57/68 of 256)
- Decode model @240-300 GB/s gather: K=2 ~78 tok/s / K=8 ~181-207 / K=10 ~215-238 (hit-class); the 140 mix = deep-K on quote classes

PREFILL: chunks of 256-512 (THE CHUNK LAW: ≥100 tokens touch ~all 256 experts → full ~16-17GB routed read per chunk regardless; 64-chunks cap at ~3,400 tok/s from re-reads alone; compute is NOT the binder — 4.3 GFLOP/tok active vs our 30.7 TFLOPS achieved): rt8e256_m128 (router M-GEMM + deterministic counting sort — ATOMIC-FREE rank-count scatter [per-(token,rank) O(8) loop over prior same-expert draws] → eoff/plist) → gxm128e256_up/dn (CTA i walks bins i,i+G sequentially; M-subgroup slim M≤32 W4A8 IMMA per bin; zero-pad masked deterministic) → combine fixed-order.
- M=128 for shared expert + attn trunk (existing M-families retarget 5120→2048 hidden via gen_m*.py)

## REUSE MAP
Ports verbatim: n-gram K=10 LOOKUP (model-agnostic — hit rate is a text property; verify with repeat-class corpus gate), accept/acceptsel, ParityGraph cadence, serve.py session layer, pcache (per-model namespace), int8-KV (10 layers → 10.3KiB/tok → 1.26GiB @128k), partial RoPE, device-resident control.
Ports w/ rescale: k2s GDN (30L, [30][5] slots, ⚠️ 32× state — bench first), spk attention (head_dim 256 ✓, GQA 16:2 map, output-gate fuse), WY-C32 scan, GEMV families (2048-hidden regen), embed gather (clamp; image_token id harmless).
NEW (the actual port): rt8e256 + gx8e256_up/dn + mx8e256_cmb + gxm128e256 family + counting-sort + output-gate epilogue + (1+w) norm variant. ~15-25 new sources.

## WEEK-1 DECIDERS (kill-cheap-first; stop at NO-GO)
- D1 [2h] GGUF header dump + tensor map (R1 residue) — confirm arch/exact tensor names vs modeling.py
- D2 [1-2d] ⭐ R2 REPACK: one layer's experts UD-IQ4_XS + UD-IQ3_S → our repacker → packed ratio + 16B-alignment of 512-elem down rows → VRAM verdict (kimi table 22.5GB@128k/4.25bpw vs mimo 11.5GB/IQ3S+0.6× ratio — R2 decides) + tier choice
- D3 [1-2d] ⭐ G2 GATHER-GEMV POC: one cubin, nw32 name-encoded, ROWS≤64, smem≤36KB, spill=0, 1 CTA/SM: y[row]+=W[eid[row]]@x over 8GB packed bank; KILL <100 GB/s; GO ≥180 grouped / ≥120 scattered, bit-exact vs CPU, in-graph 256×
- D4 [1d] K4 MUTATING IDS: graph router→topk→gemv replays 256+ cycles with device-written eids; 4000 mixed cycles; no host recapture; no wedge
- D5 [0.5-1d] K2S BENCH at new dims T=1..11 (the 32×-state sleeper) — if latency-bound blowup → HMMA-ize plan priced
- D6 [0.5d CPU] OCCUPANCY MONTE CARLO: router logits (uniform lower-bound + real if Linux box) → E[distinct|M] tables → spec policy (K-mix) + decode envelopes
- D7 [4-8h Linux] QUALITY ORACLE: llama.cpp PPL/KL of UD tiers vs Q8/Q6 + top-k membership vs fp16 router (G5) — locks the tier

## TIER-1 CONTRACT (rewritten for MoE — grok/kimi consensus)
Bit-exact = (a) our frozen router is gold (fp32 logits, tie→lower-id); (b) expert bodies bit-exact GIVEN ids; (c) spec ≡ T=1 same-engine 60/60 ×2 det + stock-class vs llama.cpp-CUDA greedy bank (59/59-class tolerance). Quality gates (separate): router-ID set-equality vs llama.cpp dump on 1000 prompts; PPL Δ ≤ +1.5%; top-k membership ≥95% vs fp16 router; tie-margin counter logged. Bit-exactness traps: RMSNormZeroCentered (1+w)!, router epilogue order (softmax→topk→renorm→cast), sorted expert-sum, SWIGLU-in-epilogue order, output-gate order, sort determinism.

## SERVING MULTI-MODEL (qwen design + dispatcher corrections)
- ITEM #0 FIRST: engine loads TLX_MODEL_PATH (today hardcoded GGUF!) + boot assert loaded==fingerprinted
- ops/model_registry.json + env.common + env.canonical.d/{model}.env; model_id in status (+fail-fast); per-model drift EXPECTED_FP dict; /v1/models resident/loadable/unavailable
- Swap = durable next_model intent (atomic+fsync BEFORE shutdown RPC; reboot law) → wrapper boot selection (next_model > staydown > current); enginectl switch = validate→drain(API admin)→intent→stop; verify/clear by the RELAUNGED api (file-derived swap state, not memory); single launchd unit; swap 503s w/ ETA; DEFAULT 409 model_not_resident (auto-swap = thrash, NO)
- pcache per-model roots (migration: mv ~/prompt_cache → /qwen3.8-27b-egpu subdir; relocatable=verified) + quotas 30/26GB; model-scoped conversations (model_id,cid) — cross-model = FRESH by law; per-model tokenizer lazy-loader (vendored preset qwen35moe EXISTS); per-model caps min(registry, live ctxk); usage/echo = SERVED model
- Keep: permits = max(1,batch_b) structure (NOT qwen's batch_b+waiters); stop-ids derivation (thread tokenizer only); engine RPC conformance suite as the MoE daemon contract

## SEQUENCED WORK (merged kimi 36d + mimo 42-66d; +10d contingency; ~6-9 weeks)
P0 research/deciders D1-D7 [5-7d] → P1 MoE repacker (expert extraction, 16B bases, drop MTP+mmproj, tier measure) [4-6d] → P2 dense trunk retarget 2048-hidden + regen [2d] → P3 MoE decode kernels (rt/gx/mx families + SWIGLU epilogues + R4_TRACE-class discriminators + expert-slab checksums) [8-12d] → P4 GDN rescale (D5-informed) + attn port (output gate, GQA map, head_dim 256) + (1+w) norms [7-10d] → P5 T=1 bring-up + Tier-1 bank + ctx ladder [3d] → P6 spec path (deep-K graph sets; K-mix per D6) [2d] → P7 prefill grouped GEMM + chunk 256 + sort [5-9d] → P8 serving multi-model (item#0 → registry → swap → pcache → API; parallelizable with P3-P7) [10-15d] → P9 VRAM audit @128k + integration + eval battery (PPL/tasks/banks 2k-128k/thinking ON+OFF/B=2) [5-8d] → P10 perf bring-up (140-decode quote-class / prefill ladder / K sweep) [4-6d]
LAUNCH SHIP: 96k default (128k opt-in post-audit), UD-IQ4_XS-or-IQ3_S per D2/D7, n-gram spec K-mix, batch B=1 (B=2 phase 2), MTP dropped.

## RISK LEDGER (top): 1. gather-BW scatter collapse (D3 kills cheap) 2. VRAM/repack ratio (D2) 3. k2s 32×-state (D5) 4. 16B alignment on 512-elem rows (D2) 5. sort/routing determinism (gates) 6. graph-node/950-cycle (depth-1 + one family) 7. graph-variant budget ≤8/model (K=8 deep set not K=10 at 96k) 8. long-ctx numerics fp32-state+int8-KV (Tier-2 agreement contract).
