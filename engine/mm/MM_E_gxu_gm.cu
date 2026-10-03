// MM SESSION E (item 1): gxu_gm -- the GATHERED-ROW-LIST mma routed
// gate+up GEMM. Replaces gxm_up's rs-stripe x 8-row-sweep structure with
// 64-row mma tiles: M = the expert's W rows (dense -- no M-side ragged
// waste), N = the bin's gathered activation rows (16-token tiles via
// plist, pad rows zero-filled -> deterministic), K = 2048.
//   A-operand = dequantized IQ3_S gate/up weights, smem [Wrow][k] (the
//     pgmq8k2 xs addressing pattern); B-operand = gathered x, smem
//     [token][k] (the pgmq8k2 ws addressing pattern). c[m][n] =
//     sum_k W[m][k]*x[n][k] -- the template fragment math with the
//     operand roles swapped (both slots consume [row][k] row-major smem,
//     so the proven load code transfers verbatim: A/B-frag n=g,k=tp-based;
//     C-frag c0/c1=(m,n),(m,n+1), c2/c3=(m+8,n),(m+8,n+1)).
//   Decode volume = ceil(bin/16) x 512 rows x 4096 per expert -- IDENTICAL
//   to gxm_up (TS=16): items (expert, rowtile) fixed 2048, token-tiles
//   looped in-kernel. Multi-tile bins re-decode W per tile (the same
//   waste class as gxm_up's token-chunks; Poisson-8 bins -> ~1 tile).
//   Epilogue: silu(g)*u scattered to ys[pair*512 + row] -- cmbz2048's
//   expected [pair][512] layout UNTOUCHED.
// NUMERICS: Tier-2 (fp16 operands + mma reassociation vs gxm_up's fp32
// FMA trees) -> the F-bank + CE gates. det x2 (every smem slot written
// from a fixed lane computation; pads zeroed).
// grid 2048 static (256 experts x 8 row-tiles); NO runtime vals (eoff
// drives everything). gridDim.x reads 0 on this dext -- grid==GN baked.
#include <cuda_fp16.h>

#define GN 2048
#define NTOK 16           // tokens per tile (N)
#define MROW 64           // W rows per item (M)
#define KDIM 2048
#define KCH 128
#define NTHR 256
#define LD (KCH + 8)
#define KSTEPS (KCH / 16)
#define MRG (MROW / 16)   // 4 m16 groups
#define NRG (NTOK / 8)    // 2 n8 slices
#define ROWB 880
#define GATE_B 450560

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

// stage ONE 64xKCH weight tile into wsm. Partition: row = warp*8+(lane>>2),
// kslice = lane&3; per lane 8 q-bytes (k = kc + ks*32 + sub*4, 4 weights
// each from gridf[qb*4..+3]). 8 subs unrolled for load ILP.
__device__ __forceinline__ void stage_wtile(
    const unsigned char* __restrict__ base, const int row0,
    const int kc, const float* __restrict__ gridf,
    __half* __restrict__ wsm)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = warp * 8 + (lane >> 2), ks = lane & 3;
  const unsigned char* blk = base + (size_t)(row0 + r) * ROWB + (kc >> 8) * 110;
  const float d = __half2float(*((const __half*)blk));
  const int s = ((kc >> 5) + ks) & 7;
  const float sc = 1.0f + 2.0f * (float)((blk[106 + (s >> 1)] >> ((s & 1) << 2)) & 0xF);
  const float ds = d * sc;
  const int kib = kc & 255;                        // k offset within the 256-k block
  const int g0 = (kib >> 2) + ks * 8;              // first q-byte index in block
  const int sgn0 = 74 + ((kib >> 3) + ks * 4);     // first sign byte index
  __half* dst = wsm + (size_t)r * LD + ks * 32;
  #pragma unroll
  for (int sub = 0; sub < 8; ++sub) {
    const int g = g0 + sub;
    const unsigned int qb = (unsigned int)blk[2 + g]
        + ((((unsigned int)(blk[66 + (g >> 3)] >> (g & 7)) & 1u) << 8));
    const float4 gr = *(const float4*)(gridf + (size_t)qb * 4);
    const unsigned int sg = (unsigned int)blk[sgn0 + (sub >> 1)]
                              >> (((sub & 1) << 2));
    float w0 = ds * gr.x; if (sg & 0x1u) w0 = -w0;
    float w1 = ds * gr.y; if (sg & 0x2u) w1 = -w1;
    float w2 = ds * gr.z; if (sg & 0x4u) w2 = -w2;
    float w3 = ds * gr.w; if (sg & 0x8u) w3 = -w3;
    *(__half2*)(dst + sub * 4) = __halves2half2(__float2half(w0), __float2half(w1));
    *(__half2*)(dst + sub * 4 + 2) = __halves2half2(__float2half(w2), __float2half(w3));
  }
}

extern "C" __global__ void __launch_bounds__(NTHR) gxu_gm(
    const unsigned long long* __restrict__ ptbl,
    const int* __restrict__ eoff,
    const unsigned short* __restrict__ plist,
    const float* __restrict__ xs,          // [P][2048] fp32
    const float* __restrict__ gridf,       // [512][4] fp32 lattice
    float* __restrict__ ys)                // [NPAIR][512] fp32
{
  __shared__ __align__(16) __half wg[MROW * LD];
  __shared__ __align__(16) __half wu[MROW * LD];
  __shared__ __align__(16) __half xt[NTOK * LD];
  __shared__ int xoff[NTOK];
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int rg = warp >> 1, ns = warp & 1;
  const int e = blockIdx.x >> 3, mi = blockIdx.x & 7;
  const int eb0 = eoff[e], bin = eoff[e + 1] - eb0;
  if (bin <= 0) return;                    // uniform per CTA
  const int row0 = mi * MROW;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[e];
  const unsigned char* baseu = base + GATE_B;

  const int ntiles = (bin + NTOK - 1) / NTOK;
  for (int t = 0; t < ntiles; ++t) {
    const int m0t = t * NTOK;
    const int nt = min(NTOK, bin - m0t);   // >= 1
    if (tid < NTOK)
      xoff[tid] = (int)(plist[eb0 + m0t + tid] >> 3) * KDIM;
    for (int i = tid; i < NTOK * KCH; i += NTHR)
      xt[(i >> 7) * LD + (i & 127)] = (__half)0;   // pads stay 0 (det)
    __syncthreads();
    // this warp's accumulators: (m16-group rg = warp>>1, n8-slice
    // ns = warp&1) -- scalars, NOT acc[rg][..]: a runtime-indexed array
    // keeps all MRG slots live (32 regs); scalars = 8.
    float cg0 = 0.f, cg1 = 0.f, cg2 = 0.f, cg3 = 0.f;
    float cu0 = 0.f, cu1 = 0.f, cu2 = 0.f, cu3 = 0.f;

    #pragma unroll 2
    for (int kc = 0; kc < KDIM; kc += KCH) {
      // x stage: gathered rows fp32->fp16 pairs (n >= nt stay zero)
      #pragma unroll
      for (int i = 0; i < (NTOK * KCH / 2 + NTHR - 1) / NTHR; ++i) {
        const int lin2 = tid + i * NTHR;
        if (lin2 < NTOK * KCH / 2) {
          const int n = lin2 >> 6, kk = (lin2 & 63) << 1;
          if (n < nt) {
            const float2 f2 = *(const float2*)(xs + (size_t)xoff[n] + kc + kk);
            *(__half2*)(xt + (size_t)n * LD + kk) = __float22half2_rn(f2);
          }
        }
      }
      stage_wtile(base, row0, kc, gridf, wg);
      stage_wtile(baseu, row0, kc, gridf, wu);
      __syncthreads();
      #pragma unroll
      for (int s = 0; s < KSTEPS; ++s) {
        const int kb = s * 16;
        const __half* ar = wg + (size_t)(rg * 16 + g) * LD + kb + tp;
        const unsigned a0 = *(const unsigned*)(ar);
        const unsigned a1 = *(const unsigned*)(ar + 8 * LD);
        const unsigned a2 = *(const unsigned*)(ar + 8);
        const unsigned a3 = *(const unsigned*)(ar + 8 * LD + 8);
        const __half* br = xt + (size_t)(ns * 8 + g) * LD + kb + tp;
        const unsigned bq0 = *(const unsigned*)(br);
        const unsigned bq1 = *(const unsigned*)(br + 8);
        hmma16816(cg0, cg1, cg2, cg3, a0, a1, a2, a3, bq0, bq1);
        const __half* au = wu + (size_t)(rg * 16 + g) * LD + kb + tp;
        const unsigned a0u = *(const unsigned*)(au);
        const unsigned a1u = *(const unsigned*)(au + 8 * LD);
        const unsigned a2u = *(const unsigned*)(au + 8);
        const unsigned a3u = *(const unsigned*)(au + 8 * LD + 8);
        hmma16816(cu0, cu1, cu2, cu3, a0u, a1u, a2u, a3u, bq0, bq1);
      }
      __syncthreads();
    }

    // epilogue: silu(g)*u -> ys[pair*512 + Wrow]. c-frag map:
    // c0/c1 = (m=rg*16+g, n=tp/tp+1), c2/c3 = (m+8, n/n+1).
    {
      const int wr1 = row0 + rg * 16 + g, wr2 = wr1 + 8;
      #pragma unroll
      for (int dn = 0; dn < 2; ++dn) {
        const int n = ns * 8 + tp + dn;
        if (n < nt) {
          const unsigned short pair = plist[eb0 + m0t + n];
          const float gf = dn ? cg1 : cg0, uf = dn ? cu1 : cu0;
          ys[(size_t)pair * 512 + wr1] = (gf / (1.0f + __expf(-gf))) * uf;
        }
      }
      #pragma unroll
      for (int dn = 0; dn < 2; ++dn) {
        const int n = ns * 8 + tp + dn;
        if (n < nt) {
          const unsigned short pair = plist[eb0 + m0t + n];
          const float gf = dn ? cg3 : cg2, uf = dn ? cu3 : cu2;
          ys[(size_t)pair * 512 + wr2] = (gf / (1.0f + __expf(-gf))) * uf;
        }
      }
    }
    __syncthreads();   // smem reuse guard before the next token-tile
  }
}
