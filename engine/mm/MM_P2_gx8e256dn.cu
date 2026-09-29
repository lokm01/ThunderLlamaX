#include <cuda_fp16.h>
// MM P2: gx8e256dn -- grouped GEMV, down mat (IQ4_XS lane, the 37 modal layers).
// One CTA per PAIR (pos = pair>>3); 1024 thr = 32 warps; 2048 rows in 64 sweeps
// of 32 rows; per row dot k=512 = 2 IQ4_XS blocks (272B rows). Per lane per
// block: elems {b*256 + ib*32 + l}, ib 0..7 (nibble at qs[8 + 16*ib + (l>>1)]);
// RUNNING sums (b asc, ib asc) = the dequant-ref order; nvcc -fmad=false.
// fp16 store -> moe_part[pair][2048], rank-addressed for mx8e256cmb.
// The down-mat base offset is baked into ptbl on the host (per-layer-bank).
extern "C" __global__ void __launch_bounds__(1024) gx8e256dn(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,   // [NPAIR]
    const float* __restrict__ xs,              // [NPAIR][512] moe_act (pair-addressed)
    const float* __restrict__ iq4nlb,          // [16] kvalues_iq4nl f32
    __half* __restrict__ parts)                // [NPAIR][2048]
{
  const int pair = blockIdx.x;
  // pos = pair >> 3 (unused: dn consumes pair-addressed moe_act)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[pair]];
  const float* x = xs + (size_t)pair*512;   // moe_act is PER-PAIR [NPAIR][512] (pos*512 was the P2 run-4 bug)
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
      const int nby = 8 + (lane & 15);   // elems 0..15 = LOW nibbles of bytes 0..15, 16..31 = HIGH
      #pragma unroll
      for (int ib = 0; ib < 8; ++ib) {
        const unsigned int nib = (lane & 16) ? (blk[16*ib + nby] >> 4) : (blk[16*ib + nby] & 0xFu);
        const unsigned int ls = ((sl >> (8*(ib>>1) + 4*(ib&1))) & 0xFu) | (((sh >> (2*ib)) & 3u) << 4);
        const float dl = d * (float)((int)ls - 32);
        const float w = dl * iq4nlb[nib];
        a += w * x[(b<<8) + (ib<<5) + lane];
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[r0+warp] = __float2half(a);
  }
}
