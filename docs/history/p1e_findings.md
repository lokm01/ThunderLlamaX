# P1e — Quant-format GEMV shootout (2026-08-25)

Question: is IQ3_XXS's LUT-gather dequant chain the reason the r_544 FFN GEMV pair
(~33.5ms/tok) is element-rate-bound (220 Melem/ms)? Would a gather-free format
(Q4_K/Q4_0/IQ4_XS scale-nibble arithmetic) dequant meaningfully faster?

Method: quant_shootout.py — synthesize raw block bytes for [17408,5120] (=89.08M elems)
per fork-supported type, lazy ggml_data_to_tensor dequant fused into x@W.T, warm TinyJit,
BEAM=1, 50 iters, DEV=NV, fresh random-but-valid blocks. VRAM delta per type <=95MB raw
(no dequant materialization; cast stays lazy inside the GEMV kernel).

## Results (isolated jit replay; relative comparison valid, absolutes carry ~0.4ms replay overhead)

| type    | ms     | raw MB | eff GB/s | Melem/ms | model est (GB) |
|---------|--------|--------|----------|----------|----------------|
| fp16    | 0.773  | 178.3  | 231      | 115      | 55+            |
| Q4_0    | 0.811  |  50.1  |  62      | 110      | 15.5           |
| Q4_K    | 0.972  |  50.1  |  52      |  92      | 15.5           |
| Q5_K    | 1.011  |  61.3  |  61      |  88      | 18.8           |
| Q6_K    | 0.913  |  73.1  |  80      |  98      | 22.2           |
| IQ4_XS  | 0.820  |  47.3  |  58      | 109      | 14.7           |
| Q8_0    | 1.093  |  94.7  |  87      |  82      | 28.6 (>20GB)   |
| IQ3_XXS | 0.748  |  34.1  |  46      | 119      | 10.8           |

(eff GB/s = raw-bytes rate; ALL formats sit 2-6x above their own 447GB/s byte floor
=> element/ALU-bound everywhere, exactly as P1d concluded.)

## Verdict: NO WINNER — IQ3_XXS is already the fastest supported format
Decision rule (>=30% GEMV win AND <=20GB) fails for every candidate; best gather-free
format (Q4_0) is 8% SLOWER than IQ3_XXS, Q4_K 30% slower. The LUT-gather hypothesis is
REFUTED: the 256-entry x4B iq3xxs grid (4KB) sits hot in cache; its per-element ALU
(mul by db*scale, sign flip) is cheaper than Q4_K's d*sc*q - dmin*mn chain or Q8_0's
wider int8 reads. Dequant ALU cost is roughly uniform (~80-120 Melem/ms isolated) and
IQ3_XXS tops the band while also having the FEWEST bytes.

Corollaries:
- Requantizing the model to any other GGML format buys NOTHING in GEMV time and costs
  quality (IQ3_XXS -> Q4_K would be a quality upgrade but a speed REGRESSION).
- In-model context: kern_hist shows in-model r_544 execs ~0.26-0.35ms (vs 0.75ms
  isolated) — in-graph pipelining beats any isolated number; format choice is not
  the lever either way.
- The only remaining GEMV lever is **A3: custom CUDA fused dequant-GEMV kernels**
  (nvcc shim infra exists). Spec from this table: target >250 Melem/ms sustained
  in-graph for IQ3_XXS blocks (2.3x current in-model ~110 effective), i.e. approach
  the 447GB/s byte floor of 19.9ms pool-wide; realistic goal halves the ~33.5ms
  r_544 pair + similar gains on qkv/down GEMVs.

Model-size facts (exact, from GGUF header): 27.32G params total; main-model per-token
GEMV pool 23.19G elems; token_embd 1.27G IQ3_S; output.weight 1.27G Q5_K (NOT fp16 —
the old 2.54GB-fp16-head memory note was stale, reconfirmed dead).
