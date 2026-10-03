// MM SESSION E (item 1b): gxd_gm -- the GATHERED-ROW-LIST mma routed
// DOWN-projection (the dn twin of gxu_gm). gxm_dnf measured 64.5ms/chunk
// (family table mm_e2_poc.json) -- the same magnitude as the up win.
//   M = the expert's dn W rows (2048 dense, MROW=128 per CTA = 16
//   row-tiles/expert), N = gathered 16-token tiles (plist, pads zeroed),
//   K = 512 (act dim; 2 blocks/row). W layout [2048][272] IQ4_NL
//   (ROWB_DN = 2*136): per 256-k block: d half[0], sh u16[2] (8x2-bit
//   hi scales), sl u32[4] (8x4-bit lo scales), 256 nibbles bytes[8..136)
//   (nibble g at byte 8+(g>>1), low if g even); w = d*(ls-32)*iq4nl[nib]
//   (VERBATIM the gxm_dnf decode). A-operand = dequant W smem [Wrow][k];
//   B = gathered actb [token][k] (fp32->fp16). Epilogue: RAW acc ->
//   parts[pair*2048 + row] __half (the cmbz2048 fp16 contract).
//   Warp partition (the pgmq8k2 MRG=2 structure): ns = warp&1, rg pair
//   = (warp>>1)*{2,3}.
// grid 4096 static (256 experts x 16 row-tiles); no runtime vals.
// NUMERICS: Tier-2 (fp16 operands + mma order) -> F-bank + CE gates.
#include <cuda_fp16.h>

#define GN_DN 4096
#define NTOK 16
#define MROW 128
#define KDIM 512
#define KCH 128
#define NTHR 256
#define LD (KCH + 8)
#define KSTEPS (KCH / 16)
#define MRG2 2                 // m16 groups per warp
#define NRG 2                  // n8 slices
#define ROWB 272               // 2 blocks x 136 (IQ4_NL)

__device__ __forceinline__ void hmma16816(float &c0, float &c1, float &c2, float &c3,
                                          const unsigned a0, const unsigned a1,
                                          const unsigned a2, const unsigned a3,
                                          const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void stage_wtile_dn(
    const unsigned char* __restrict__ base, const int row0,
    const int kc, const float* __restrict__ iq4nl,
    __half* __restrict__ wsm)
{
  // rows 128 per CTA: warp*16 + (lane>>1); lane covers 64 k (2x 32-slices)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = warp * 16 + (lane >> 1);
  const int ks = (lane & 1) * 64;
  const unsigned char* blk = base + (size_t)(row0 + r) * ROWB + (kc >> 8) * 136;
  const float d = __half2float(*((const __half*)blk));
  const unsigned int sh = *(const unsigned short*)(blk + 2);
  const unsigned int sl = *(const unsigned int*)(blk + 4);
  const int kib = kc & 255;
  #pragma unroll
  for (int half32 = 0; half32 < 2; ++half32) {
    const int ks32 = ks + half32 * 32;
    const int kbase = kib + ks32;                  // block-relative k base
    const int ib = kbase >> 5;                     // 32-k scale group
    const unsigned int ls = ((sl >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xFu)
                          | (((sh >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)((int)ls - 32);
    __half* dst = wsm + (size_t)r * LD + ks32;
    // IQ4_NL packing: within each 32-k group, k_local 0..15 = LOW nibbles
    // of bytes [8+16ib .. +15], k_local 16..31 = HIGH nibbles of the SAME
    // bytes (the gxm_dnf (lane&15)/(lane&16) law -- NOT sequential pairs).
    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      const int kloc = sub * 4;                    // k offset within the group
      const unsigned char* nb = blk + 8 + 16 * ib + (kloc & 15);
      if (kloc < 16) {
        *(__half2*)(dst + kloc) = __halves2half2(
            __float2half(dl * iq4nl[nb[0] & 0xFu]),
            __float2half(dl * iq4nl[nb[1] & 0xFu]));
        *(__half2*)(dst + kloc + 2) = __halves2half2(
            __float2half(dl * iq4nl[nb[2] & 0xFu]),
            __float2half(dl * iq4nl[nb[3] & 0xFu]));
      } else {
        *(__half2*)(dst + kloc) = __halves2half2(
            __float2half(dl * iq4nl[nb[0] >> 4]),
            __float2half(dl * iq4nl[nb[1] >> 4]));
        *(__half2*)(dst + kloc + 2) = __halves2half2(
            __float2half(dl * iq4nl[nb[2] >> 4]),
            __float2half(dl * iq4nl[nb[3] >> 4]));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(NTHR) gxd_gm(
    const unsigned long long* __restrict__ ptbl,
    const int* __restrict__ eoff,
    const unsigned short* __restrict__ plist,
    const float* __restrict__ xa,          // [NPAIR][512] fp32 act
    const float* __restrict__ iq4nl,       // [16] LUT
    __half* __restrict__ parts)            // [NPAIR][2048] fp16
{
  __shared__ __align__(16) __half wd[MROW * LD];
  __shared__ __align__(16) __half xt[NTOK * LD];
  __shared__ int xoff[NTOK];
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int ns = warp & 1;
  const int rg0 = (warp >> 1) * 2;         // this warp's m16-group pair
  const int e = blockIdx.x >> 4, mi = blockIdx.x & 15;
  const int eb0 = eoff[e], bin = eoff[e + 1] - eb0;
  if (bin <= 0) return;
  const int row0 = mi * MROW;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[e];

  const int ntiles = (bin + NTOK - 1) / NTOK;
  for (int t = 0; t < ntiles; ++t) {
    const int m0t = t * NTOK;
    const int nt = min(NTOK, bin - m0t);
    if (tid < NTOK)
      xoff[tid] = (int)plist[eb0 + m0t + tid];   // act row = pair id (gxm_dnf xoff law)
    for (int i = tid; i < NTOK * KCH; i += NTHR)
      xt[(i >> 7) * LD + (i & 127)] = (__half)0;
    __syncthreads();
    float c0a = 0.f, c1a = 0.f, c2a = 0.f, c3a = 0.f;
    float c0b = 0.f, c1b = 0.f, c2b = 0.f, c3b = 0.f;

    #pragma unroll 2
    for (int kc = 0; kc < KDIM; kc += KCH) {
      #pragma unroll
      for (int i = 0; i < (NTOK * KCH / 2 + NTHR - 1) / NTHR; ++i) {
        const int lin2 = tid + i * NTHR;
        if (lin2 < NTOK * KCH / 2) {
          const int n = lin2 >> 6, kk = (lin2 & 63) << 1;
          if (n < nt) {
            const float2 f2 = *(const float2*)(xa + (size_t)xoff[n] * 512 + kc + kk);
            *(__half2*)(xt + (size_t)n * LD + kk) = __float22half2_rn(f2);
          }
        }
      }
      stage_wtile_dn(base, row0, kc, iq4nl, wd);
      __syncthreads();
      #pragma unroll
      for (int s = 0; s < KSTEPS; ++s) {
        const int kb = s * 16;
        const __half* br = xt + (size_t)(ns * 8 + g) * LD + kb + tp;
        const unsigned bq0 = *(const unsigned*)(br);
        const unsigned bq1 = *(const unsigned*)(br + 8);
        const __half* ar = wd + (size_t)((rg0 + 0) * 16 + g) * LD + kb + tp;
        hmma16816(c0a, c1a, c2a, c3a,
                  *(const unsigned*)(ar), *(const unsigned*)(ar + 8 * LD),
                  *(const unsigned*)(ar + 8), *(const unsigned*)(ar + 8 * LD + 8),
                  bq0, bq1);
        const __half* ar2 = wd + (size_t)((rg0 + 1) * 16 + g) * LD + kb + tp;
        hmma16816(c0b, c1b, c2b, c3b,
                  *(const unsigned*)(ar2), *(const unsigned*)(ar2 + 8 * LD),
                  *(const unsigned*)(ar2 + 8), *(const unsigned*)(ar2 + 8 * LD + 8),
                  bq0, bq1);
      }
      __syncthreads();
    }

    // epilogue: raw acc -> parts[pair*2048 + row] fp16 (c0/c1 = (m, tp/tp+1)...)
    #pragma unroll
    for (int dn = 0; dn < 2; ++dn) {
      const int n = ns * 8 + tp + dn;
      if (n < nt) {
        const unsigned short pair = plist[eb0 + m0t + n];
        parts[(size_t)pair * 2048 + row0 + rg0 * 16 + g] = __float2half(dn ? c1a : c0a);
        parts[(size_t)pair * 2048 + row0 + rg0 * 16 + g + 8] = __float2half(dn ? c3a : c2a);
        parts[(size_t)pair * 2048 + row0 + (rg0 + 1) * 16 + g] = __float2half(dn ? c1b : c0b);
        parts[(size_t)pair * 2048 + row0 + (rg0 + 1) * 16 + g + 8] = __float2half(dn ? c3b : c2b);
      }
    }
    __syncthreads();
  }
}
