#include <cuda_fp16.h>
// MM P2: gx8e256up4 -- grouped GEMV, gate+up mats (IQ4_XS lane; layer 39 only).
// Same structure as gx8e256up (IQ3_S lane) but rows 1088B (8 x 136B blocks,
// k=2048); GATE_B4 = 557056 (the IQ4_XS 512x2048 mat); SWIGLU epilogue.
// IQ4_XS lane (same as gx8e256dn): per lane elems {b*256 + ib*32 + l},
// ib = the 32-elem group (one nibble per group per lane); RUNNING sums
// (b asc, ib asc) per mat; -fmad=false; fp32 store moe_act.
extern "C" __global__ void __launch_bounds__(1024) gx8e256up4(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs,              // [P][2048]
    const float* __restrict__ iq4nlb,          // [16]
    float* __restrict__ ys)                    // [NPAIR][512]
{
  const int pair = blockIdx.x;
  const int pos = pair >> 3;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[pair]];
  const float* x = xs + (size_t)pos*2048;
  float* y = ys + (size_t)pair*512;
  const unsigned char* gp = base;
  const unsigned char* upp = base + 557056;
  const int nby = 8 + (lane & 15);   // elems 0..15 = LOW nibbles of bytes 0..15, 16..31 = HIGH
  for (int r0 = 0; r0 < 512; r0 += 32) {
    float ag = 0.f, au = 0.f;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      #pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const unsigned char* blk = (mi == 0 ? gp : upp) + (size_t)(r0+warp)*1088 + b*136;
        const float d = __half2float(*((const __half*)blk));
        const unsigned int sh = *(const unsigned short*)(blk + 2);
        const unsigned int sl = *(const unsigned int*)(blk + 4);
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
          const unsigned int nib = (lane & 16) ? (blk[16*ib + nby] >> 4) : (blk[16*ib + nby] & 0xFu);
          const unsigned int ls = ((sl >> (8*(ib>>1) + 4*(ib&1))) & 0xFu) | (((sh >> (2*ib)) & 3u) << 4);
          const float dl = d * (float)((int)ls - 32);
          const float w = dl * iq4nlb[nib];
          const float xv = x[(b<<8) + (ib<<5) + lane];
          if (mi == 0) ag += w * xv; else au += w * xv;
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
