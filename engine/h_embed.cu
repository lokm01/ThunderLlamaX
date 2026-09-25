// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-b: embed (IQ3_S gather+dequant), argmax over half logits.
// The head GEMV reuses k0_norm (output_norm -> xh half) + k1_q5 (Q5_K 248320-row
// GEMV, grid 31040) from W1-a cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define VOCAB 248320

// ---- h_embed: read token id from device slot, gather+dequant its IQ3_S row.
// grid=(1,), 256 threads; row = 2200B = 20 blocks x 110B. Output fp32 x[5120]. ----
extern "C" __global__ void __launch_bounds__(256) h_embed(
    const unsigned char* __restrict__ emb, const float* __restrict__ grid512,
    const int* __restrict__ tok_slot, float* __restrict__ x)
{
  const int tid = threadIdx.x;
  // W4.5 (kimi F4): clamp token ids into vocab before the row pointer —
  // a stale/garbage slot (allocator-recycled rows at conversation starts)
  // must never wild-index the embed table. Identity for in-range ids.
  const int tok = min(max(tok_slot[0], 0), VOCAB - 1);
  const unsigned char* row = emb + (size_t)tok * 2200u;
  #pragma unroll
  for (int i = 0; i < 20; ++i) {
    const unsigned char* blk = row + i*110;
    const int e = i*256 + tid;             // element within the 5120-row
    // IQ3_S fields are WITHIN-BLOCK (256 elems): q g=e&255>>2, scale e&255>>5, sign byte e&255>>3 bit e&7
    const float d = __half2float(*((const __half*)blk));
    const int g = tid >> 2, j4 = tid & 3;
    const unsigned int q = (unsigned int)blk[2 + g] + ((((unsigned int)blk[66 + (g>>3)] >> (g&7)) & 1u) << 8);
    const int s8 = tid >> 5;
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (s8>>1)] >> ((s8&1)<<2)) & 0xF);
    const float sgn = ((blk[74 + (tid>>3)] >> (tid & 7)) & 1) ? -1.f : 1.f;
    x[e] = d * sc * grid512[(q << 2) + j4] * sgn;
  }
}

// ---- h_argmax: greedy pick over half logits; grid=(1,), block-wide reduce.
// Tie-safe (lower index wins on equal values). Writes tok_slot and tok_hist[pos],
// bumps pos_slot. All device-resident (no host roundtrip in the decode loop). ----
