// MM P6: selc36 -- the spec partial-state COMMIT: copy GDN slot m (contiguous
// 30 layers x 2MiB) + conv slot m (30 x 96KiB) into the LIVE state buffers.
// m read from a device int (host copyin after the accept decision). Pure copy
// -> deterministic. Grid: ceil((NGDN+NCS)/2048) CTAs x 256 thr x 8 elems.
#define NGDN 15728640   // 30*32*128*128
#define NCS  737280     // 30*8192*3
extern "C" __global__ void __launch_bounds__(256) selc36(
    const float* __restrict__ slots,     // [T][30][32][128][128]
    const float* __restrict__ cslots,    // [T][30][8192][3]
    const int* __restrict__ m_slot,      // device int m
    float* __restrict__ S_live,          // [30][32][128][128]
    float* __restrict__ CS_live)         // [30][8192][3]
{
  const size_t m = (size_t)m_slot[0];
  const float* gsrc = slots + m*(size_t)NGDN;
  const float* csrc = cslots + m*(size_t)NCS;
  const size_t i0 = (size_t)blockIdx.x * 2048 + (size_t)threadIdx.x * 8;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    const size_t i = i0 + k;
    if (i < NGDN) S_live[i] = gsrc[i];
    else if (i < NGDN + NCS) CS_live[i - NGDN] = csrc[i - NGDN];
  }
}
