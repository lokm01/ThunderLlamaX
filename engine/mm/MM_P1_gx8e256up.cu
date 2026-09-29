// MM P1 PRODUCTION v0: gx8e256up -- grouped GEMV, gate+up mats + SWIGLU epilogue.
#include <cuda_fp16.h>
// POC lane pattern (gx8e256nw32) ported: one CTA per PAIR (position P = blockIdx.x>>3,
// rank = blockIdx.x&7 -- all 8 ranks of a position share xs row P); 1024 threads =
// 32 warps; 512 rows in 16 x 32-row sweeps; per row TWO dots (gate mat at slab+0,
// up mat at slab+GATE_B), each = per-lane partials (b asc, j asc) + xor-shfl tree;
// epilogue y[r] = silu(g)*u  (silu = g/(1+expf(-g))).
// IQ3_S rows (110B per 256-elem block, 8 blocks per 2048-row) -- the modal class.
// nvcc -fmad=false (dot part bit-exact vs the D2-validated dequant ref order);
// the expf epilogue carries ULP tolerance vs np.exp (harness checks allclose).
#define KB 8
#define ROWB (110*KB)
#define NROW 512
#define GATE_B 450560           // bytes of the gate mat per expert (IQ3_S 512x2048)
extern "C" __global__ void __launch_bounds__(1024) gx8e256up(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,   // [NPAIR] values
    const float* __restrict__ xs,              // [P][2048] (P = NPAIR/8)
    const float* __restrict__ gridf,
    float* __restrict__ ys)                    // [NPAIR][512] moe_act
{
  const int pair = blockIdx.x;
  const int pos = pair >> 3;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[pair]];
  const float* x = xs + (size_t)pos*2048;
  float* y = ys + (size_t)pair*512;
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
        // RUNNING sums per mat (b asc, j asc) -- bit-exact vs the dequant-ref
        // accumulation order (P0's gx_ref); NO per-block partialization.
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
      y[r0+warp] = (g / (1.0f + __expf(-g))) * u;
    }
  }
}
