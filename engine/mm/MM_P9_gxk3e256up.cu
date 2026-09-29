// MM P9: gxk3e256up -- grouped GEMV, gate+up mats + SWIGLU epilogue for the
// MTP layer's Q3_K experts (blk.40 routed bank; gate|up both Q3_K 512x2048).
// Structure VERBATIM gx8e256up (one CTA per PAIR: pos = blockIdx.x>>3, rank =
// blockIdx.x&7; 1024 thr = 32 warps; 512 rows in 16 x 32-row sweeps; per row
// TWO dots, per-lane partials + 5-level xor tree; epilogue silu(g)*u).
// Q3_K row = 8 blocks x 110B; block = { hmask[32] | qs[64] | scales[12] | d[2] }
// (d LAST -- the K-quant layout law). Dequant VERBATIM the validated numpy
// port (mm_mtp_anchor.dq_q3_k_np, itself verbatim ggml-quants.c):
//   scales unpack: a0,a1,tmp = u32 @ blk[96:108];
//     scale[s] (s = 8g' + ... form): k = s&3, g = s>>2; src = (g&1)?a1:a0;
//     sc = ((src >> (8k + (g>=2?4:0))) & 0xF) | (((tmp >> (8k + 2g)) & 0x3) << 4)
//     read as int8 (>=128 -> -=256);  value uses (sc - 32)
//   elem e (within the 256-block) = qs[nn*32 + e] >> (2*j) & 3, ONE elem per
//     byte; nn = e>>7, j = (e>>5)&3; hmask bit = hmask[e&31] & (1<<j) (the
//     SAME 32 hmask bytes serve both 128-groups)
//   w = d*(sc-32) * (q2 - (hmask_bit ? 0 : 4))
// Lane l covers block elems l*8..l*8+7 = 8 consecutive qs bytes, all one
// (nn, j, scale). -fmad=false. ZERO-SPILL contract.
#include <cuda_fp16.h>
#define KB 8
#define ROWB (110*KB)            // 880 B per Q3_K row (k=2048)
#define NROW 512
#define GATE_B 450560            // gate mat bytes per expert (Q3_K 512x2048)

// THE NARROW-LOAD LAW (ptxas load coarsening): consecutive u16/u8 reads from
// ONE thread get merged into 32-bit loads -- at odd blocks (110B stride =
// 2 mod 4) those are MISALIGNED (fault or silent garbage on this dext).
// Inline-PTX narrow loads are unmergeable. (The trunk's IQ3_S kernel only
// survives because its per-LANE strided u16s are never consecutive.)
__device__ __forceinline__ unsigned int ld_u8(const void* p) {
    unsigned int v;
    asm volatile("ld.global.u8 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned int ld_u16(const void* p) {
    unsigned int v;
    asm volatile("ld.global.u16 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

extern "C" __global__ void __launch_bounds__(1024) gxk3e256up(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,   // [NPAIR] values
    const float* __restrict__ xs,              // [P][2048] (P = NPAIR/8)
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
  // lane's 8 elems within each 256-block: e = l*8..l*8+7 (block-internal)
  const int o0 = lane << 3;                       // 0..248, multiple of 8
  const int nn = o0 >> 7;                         // 128-group 0/1
  const int jj = (o0 >> 5) & 3;                   // 32-segment within group
  const int e0 = o0 & 31;                         // 0/8/16/24 (byte idx in seg)
  const int sh = jj << 1;                         // 2-bit shift
  // THE SHARED-HMASK LAW: the SAME 32 hmask bytes serve both 128-groups;
  // m's 8 bits SPAN them (nn=0: 1,2,4,8 | nn=1: 16,32,64,128) -- the numpy
  // port's m never resets between groups.
  const unsigned int hm_mask = 1u << ((nn << 2) + jj);
  const int sc_s = nn*8 + (jj << 1) + ((e0 & 16) ? 1 : 0);  // lane's scale idx
  const int sc_k = sc_s & 3, sc_g = sc_s >> 2;
  const int qb = nn*32 + e0;                      // first qs byte
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
        const unsigned char* blk = (mi == 0 ? gp : upp) + (size_t)(r0+warp)*ROWB + b*110;
        const float d = __half2float(__ushort_as_half((unsigned short)ld_u16(blk + 108)));
        // THE 110-BYTE BLOCK ALIGNMENT LAW: odd blocks sit at 2 mod 4 -> NO
        // u32 loads anywhere in a Q3_K row; the scale words assemble from
        // UNMERGEABLE u16 pairs (all even offsets).
        const unsigned int a0 = ld_u16(blk + 96) | (ld_u16(blk + 98) << 16);
        const unsigned int a1 = ld_u16(blk + 100) | (ld_u16(blk + 102) << 16);
        const unsigned int tmp = ld_u16(blk + 104) | (ld_u16(blk + 106) << 16);
        const unsigned int src = (sc_g & 1) ? a1 : a0;
        const int losh = (sc_k << 3) + ((sc_g >> 1) << 2);
        const int tsh = (sc_k << 3) + (sc_g << 1);
        int scv = (int)(((src >> losh) & 0xFu) | (((tmp >> tsh) & 0x3u) << 4));
        if (scv > 127) scv -= 256;
        const float dl = d * (float)(scv - 32);
        const unsigned char* qseg = blk + 32 + qb;   // 8 consecutive qs bytes
        const unsigned char* hseg = blk;             // hmask[32]
        float w[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
          w[i] = dl * (float)((int)((ld_u8(qseg + i) >> sh) & 3) - ((ld_u8(hseg + e0 + i) & hm_mask) ? 0 : 4));
        if (mi == 0) {
          ag += w[0]*x0; ag += w[1]*x1; ag += w[2]*x2; ag += w[3]*x3;
          ag += w[4]*x4; ag += w[5]*x5; ag += w[6]*x6; ag += w[7]*x7;
        } else {
          au += w[0]*x0; au += w[1]*x1; au += w[2]*x2; au += w[3]*x3;
          au += w[4]*x4; au += w[5]*x5; au += w[6]*x6; au += w[7]*x7;
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
