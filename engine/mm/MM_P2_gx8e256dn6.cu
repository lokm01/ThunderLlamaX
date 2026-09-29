#include <cuda_fp16.h>
// MM P2: gx8e256dn6 -- grouped GEMV, down mat (Q6_K lane; layers 34/38/39).
// Same CTA/pair shape as gx8e256dn; rows 420B = 2 blocks x 210B.
// Q6_K block: ql[128] | qh[64] | scales int8[16] | d half@208 (ggml layout).
// Lane l covers elems {b*256 + h*128 + l + {0,32,64,96}} per the verbatim
// dequantize_row_q6_K order; mult order (d*sc)*q; RUNNING sums (b,h,jj asc);
// -fmad=false; fp16 store. Down base via ptbl (single-bank L34/38: slab+901120;
// L39: the split dn bank base+0).
extern "C" __global__ void __launch_bounds__(1024) gx8e256dn6(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs,              // [NPAIR][512] moe_act
    __half* __restrict__ parts)                // [NPAIR][2048]
{
  const int pair = blockIdx.x;
  // pos = pair >> 3 (unused: dn consumes pair-addressed moe_act)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[pair]];
  const float* x = xs + (size_t)pair*512;   // moe_act is PER-PAIR [NPAIR][512] (pos*512 was the P2 run-4 bug)
  __half* y = parts + (size_t)pair*2048;
  const int is = lane >> 4;
  for (int r0 = 0; r0 < 2048; r0 += 32) {
    const unsigned char* rowp = base + (size_t)(r0+warp)*420;
    float a = 0.f;
    #pragma unroll
    for (int b = 0; b < 2; ++b) {
      const unsigned char* blk = rowp + b*210;
      const float d = __half2float(*((const __half*)(blk + 208)));
      const signed char* sc = (const signed char*)(blk + 192);
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int qb = 64*h + lane, hb = 128 + 32*h + lane, sb = 8*h + is;
        const unsigned char qhb = blk[hb];
        const unsigned int ql0 = blk[qb], ql32 = blk[qb + 32];
        const float x0 = x[(b<<8) + (h<<7) + lane + 0];
        const float x1 = x[(b<<8) + (h<<7) + lane + 32];
        const float x2 = x[(b<<8) + (h<<7) + lane + 64];
        const float x3 = x[(b<<8) + (h<<7) + lane + 96];
        float t = d * (float)sc[sb + 0];
        float w = t * (float)((int)((ql0 & 0xFu) | ((qhb & 3u) << 4)) - 32);
        a += w * x0;
        t = d * (float)sc[sb + 2];
        w = t * (float)((int)((ql32 & 0xFu) | (((qhb >> 2) & 3u) << 4)) - 32);
        a += w * x1;
        t = d * (float)sc[sb + 4];
        w = t * (float)((int)((ql0 >> 4) | (((qhb >> 4) & 3u) << 4)) - 32);
        a += w * x2;
        t = d * (float)sc[sb + 6];
        w = t * (float)((int)((ql32 >> 4) | (((qhb >> 6) & 3u) << 4)) - 32);
        a += w * x3;
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[r0+warp] = __float2half(a);
  }
}
