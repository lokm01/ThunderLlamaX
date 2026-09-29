// MM PB1 (Part B, step 2): gxdn8 -- the FUSED expert GEMV pair
// (gx8e256up[IQ3_S gate+up] + gx8e256dn[IQ4_XS down] merged VERBATIM, one
// launch instead of two; the launch-serialization law prices this at
// ~0.094ms/layer/cycle). The two bodies are per-PAIR self-contained (the
// dn half reads ONLY its own pair's 512-float act), so the act passes
// through SHARED MEMORY instead of the global actb round-trip -- same fp32
// bits, same accumulation order => BIT-EXACT by construction. smem 2KB.
// -fmad=false; watch the spill audit (must stay 0).
#include <cuda_fp16.h>
#define KB 8
#define ROWB (110*KB)
#define NROW 512
#define GATE_B 450560           // bytes of the gate mat per expert (IQ3_S 512x2048)
extern "C" __global__ void __launch_bounds__(1024) gxdn8(
    const unsigned long long* __restrict__ ptbl_up,  // per-layer gate+up tables
    const unsigned long long* __restrict__ ptbl_dn,  // per-layer down tables
    const unsigned short* __restrict__ eids,         // [NPAIR]
    const float* __restrict__ xs,                    // [P][2048]
    const float* __restrict__ gridf,                 // IQ3_S grid [.. x 4]
    const float* __restrict__ iq4nlb,                // [16] kvalues_iq4nl f32
    __half* __restrict__ parts)                      // [NPAIR][2048]
{
  const int pair = blockIdx.x;
  const int pos = pair >> 3;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __shared__ float t[512];       // the act handoff (was the global actb row)
  const float* x = xs + (size_t)pos*2048;
  // ---- gx8e256up half: gate+up dots + SWIGLU -> t[512] ----
  {
    const unsigned char* base = (const unsigned char*)(size_t)ptbl_up[eids[pair]];
    const unsigned char* gp = base;
    const unsigned char* upp = base + GATE_B;
    for (int r0 = 0; r0 < NROW; r0 += 32) {
      float ag = 0.f, au = 0.f;
      #pragma unroll
      for (int b = 0; b < KB; ++b) {
        const float x0 = x[(b<<8)+(lane<<3)+0], x1 = x[(b<<8)+(lane<<3)+1];
        const float x2 = x[(b<<8)+(lane<<3)+2], x3 = x[(b<<8)+(lane<<3)+3];
        const float x4 = x[(b<<8)+(lane<<3)+4], x5 = x[(b<<8)+(lane<<3)+5];
        const float x6 = x[(b<<8)+(lane<<3)+6], x7 = x[(b<<8)+(lane<<3)+7];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const unsigned char* rowp = (mi == 0 ? gp : upp) + (size_t)(r0+warp)*ROWB;
          const unsigned char* blk = rowp + b*110;
          const float d = __half2float(*((const __half*)blk));
          const int g0i = lane*2, g1i = lane*2 + 1;
          const int sraw = lane >> 2;
          const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
          const unsigned int sg = blk[74 + lane];
          const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
          const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
          const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
          const float* gr0 = gridf + (size_t)qb0*4;
          const float* gr1 = gridf + (size_t)qb1*4;
          float w0 = d*sc*gr0[0]; if (sg & 0x01u) w0 = -w0;
          float w1 = d*sc*gr0[1]; if (sg & 0x02u) w1 = -w1;
          float w2 = d*sc*gr0[2]; if (sg & 0x04u) w2 = -w2;
          float w3 = d*sc*gr0[3]; if (sg & 0x08u) w3 = -w3;
          float w4 = d*sc*gr1[0]; if (sg & 0x10u) w4 = -w4;
          float w5 = d*sc*gr1[1]; if (sg & 0x20u) w5 = -w5;
          float w6 = d*sc*gr1[2]; if (sg & 0x40u) w6 = -w6;
          float w7 = d*sc*gr1[3]; if (sg & 0x80u) w7 = -w7;
          if (mi == 0) {
            ag += w0*x0; ag += w1*x1; ag += w2*x2; ag += w3*x3;
            ag += w4*x4; ag += w5*x5; ag += w6*x6; ag += w7*x7;
          } else {
            au += w0*x0; au += w1*x1; au += w2*x2; au += w3*x3;
            au += w4*x4; au += w5*x5; au += w6*x6; au += w7*x7;
          }
        }
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        ag += __shfl_xor_sync(0xffffffffu, ag, o);
        au += __shfl_xor_sync(0xffffffffu, au, o);
      }
      if (lane == 0) {
        const float g = ag, u = au;
        t[r0+warp] = (g / (1.0f + __expf(-g))) * u;
      }
    }
  }
  __syncthreads();
  // ---- gx8e256dn half: down dots (x = the smem act) -> parts fp16 ----
  {
    const unsigned char* base = (const unsigned char*)(size_t)ptbl_dn[eids[pair]];
    __half* y = parts + (size_t)pair*2048;
    for (int r0 = 0; r0 < 2048; r0 += 32) {
      const unsigned char* rowp = base + (size_t)(r0+warp)*272;
      float a = 0.f;
      #pragma unroll
      for (int b = 0; b < 2; ++b) {
        const unsigned char* blk = rowp + b*136;
        const float d = __half2float(*((const __half*)blk));
        const unsigned int sh = *(const unsigned short*)(blk + 2);
        const unsigned int sl = *(const unsigned int*)(blk + 4);
        const int nby = 8 + (lane & 15);
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
          const unsigned int nib = (lane & 16) ? (blk[16*ib + nby] >> 4) : (blk[16*ib + nby] & 0xFu);
          const unsigned int ls = ((sl >> (8*(ib>>1) + 4*(ib&1))) & 0xFu) | (((sh >> (2*ib)) & 3u) << 4);
          const float dl = d * (float)((int)ls - 32);
          const float w = dl * iq4nlb[nib];
          a += w * t[(b<<8) + (ib<<5) + lane];
        }
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) y[r0+warp] = __float2half(a);
    }
  }
}
