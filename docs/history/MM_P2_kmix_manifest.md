# MM P2 — K-MIX GRAPH-VARIANT MANIFEST (design only; P3 builds it)
Qwen3.6-35B-A3B @ iq4_xs tier | per MM_PLAN D6 policy + the <=8-variant/model law

## THE POLICY (D6, fixed)
- PROSE (default): K=2 (P=3). Deeper K is NET-NEGATIVE on prose at every BW (D6 envelope).
- QUOTE/LOOKUP-triggered: K=8 (P=9). Trigger = LOOKUP_TRIG=2-class: TWO consecutive
  saturated n-gram steps -> switch; hysteresis back after N=4 unsaturated steps
  (switch only at a quiescent boundary -- the R6 rotated-fence/GLOBAL rebuild law).
- K=10 rejected: +15% over K=8 on quote, not worth a variant slot at 96k (D6).

## GRAPH SETS (4 of the <=8 budget; 4 held in reserve)
| set | shape | kernels/cycle | notes |
|-----|-------|---------------|-------|
| D2  | decode K=2, P=3 | ~500 nodes | prose default; always resident |
| D8  | decode K=8, P=9 | ~510 nodes | deep set; swap at rebuild boundary |
| PF  | prefill chunk 256/512 | gxm128e256 family | P7; M-grouped grouped-GEMM, counting sort |
| B2  | batch B=2 opt-in | (phase 2) | reserve |
Node budget: MoE 280 (40L x [2 norm + rt + shexp + up + dn + cmb]) + GDN 30L x ~5
(gv8qkv/z + conv1d + k2s(T) + gv8out) + attn 10L x ~5 (gv8q/k/v + core + gv8o) +
embed 1 + head 1 + accept/sel ~3 + draft ~8 = ~500-520. Depth-1 ONLY (650 x 2 > 904
in-flight law); wait-each per train (the P1 SKEDCHECK22 law).

## CUBIN INVENTORY (distinct programs; per-layer INSTANCES in parens)
MoE (P2-landed): rmsz2048 (80) | rt8e256 (40) | shexp8 (40) | gx8e256up (39) +
gx8e256up4 (1, L39) | gx8e256dn (37) + gx8e256dn6 (3, L34/38/39) | mx8e256cmb (40)
Trunk GEMV (P2-landed): gv8k2048 | gv8k4096 | gv8k512 | embg248 | h8i2048
  gv8k2048 serves: GDN qkv [8192x2048] + z [4096x2048] (attn-named map), attn q
  DOUBLED [8192x2048] (rows 0..4095 = Q, 4096..8191 = gate; consumer splits -- the
  sigmoid-gate fuse belongs to the P4 attn epilogue, not the GEMV), attn k/v
  [512x2048]. gv8k4096: ssm_out + attn_output [2048x4096]. gv8k512: shexp down.
P4 ports (not yet built): gdn conv1d k=4 [8192ch], k2s scan per-T (T=3, T=9 per D5
  cubins), spk attention core (head_dim 256, GQA 16:2, output-gate epilogue),
  q/k-norm 256, partial RoPE (64d, theta 1e7 -- port).
K-variant cubins (the 27B engine pattern): accept{K}/acceptsel{K} + draft chain
  per K + k2s per T. K=2 and K=8 only.

## KERNARGS SLAB BUDGET
~520 nodes x ~88B avg (5-7 ptr args + headers) ~= 46 KB/variant build; D2+D8
~92 KB host-mapped. The KERNARGS-SLAB LEAK law (every ParityGraph build leaks a
host-mapped ka slab) is the real line: 256-cycle GLOBAL rebuild cadence ->
~1 leak/build -> the ka-slab-reuse open item MUST land before P5 soak-class runs
(rotated-fence mitigation ships as default).

## VRAM LINES (iq4_xs @ 96k, from D2/P1 tables)
weights 16.553 GiB + KV 0.95 + GDN states 9 slots (K=8 max) 0.53 + act/ovh ~2.0
= ~20.1 GiB (2.9 headroom at 23.0). GDN slots: allocate the K=8 MAX once; the
K=2 set reuses a subset (no dual allocation). The K-mix itself adds ~0 (graphs
share weights; activation buffers are KBs).

## TRIGGERS + REBUILD INTERACTION
- Variant switch ONLY at the 256-cycle GLOBAL rebuild boundary (fence-all; the
  GRAPH-CLASS PREFILL BUDGET law: only fence-all resets the ~950-cycle dext budget).
- The deep-K set's first build happens at boot (cold) OR lazily on first trigger
  (warm); lazy = +1 rebuild stall on the first quote burst -- acceptable v0.
- TLX_STEP_TIMEOUT_S watchdog unchanged; step budget per variant identical.

## P3 OPEN ITEMS (the decode graph assembly)
1. 40-layer graph: MoE nodes now exist; wire GDN (30) + attn (10) trunk kernels
   (P4 ports) + embed/head + draft/accept per K -> the ~520-node D2/D8 trains.
2. h layout: fp32 [P][2048] trunk-wide (the P2 slice contract); fp16 only at the
   cmb store + KV boundaries.
3. ka-slab reuse (the leak) + M1A_GEN_REBUILD_EVERY=256 cadence validation at
   520 nodes x 256 replays (the S3g 7-node x 256 x 2 clean run is the floor proof).
4. Per-layer ptbl tables (40 x 256 entries x 8B = 80 KB device) built at boot from
   manifest.json; L39's split banks = two tables.
