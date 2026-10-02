// MM SESSION B (L2): mmsort8 -- the deterministic counting-sort + item-table
// builder for the grouped expert GEMMs. ONE CTA, 1024 threads, NO ATOMICS.
//   in:  eids[np] u16 (pos-major, rank asc -- exactly what rt8e256 wrote)
//   out: eoff[257] i32  exclusive prefix over expert bins (bin e = [eoff[e],
//                     eoff[e+1]); pairs inside a bin keep (pos,rank) asc
//                     (the stable rank-count scatter: rank(i) = #{j<i: same e})
//        plist[np] u16 the sorted pair ids
//        items[] u32    packed (expert<<20 | rowstripe<<16 | m0) descriptors,
//                       expert asc, then rowstripe asc, then token-chunk asc;
//                       per expert ceil(bin/TS) token-chunks x RS row-stripes
//        nitp[1] u32    the item count
// Determinism: every output slot is written from a fixed sequential
// computation (no atomic order dependence) -> det x2 identical.
// TS/RS via -D (16/4 the Session-B default; the histogram law: bins need
// M-headroom to ~256, NOT Poisson-30).
#ifndef TS
#define TS 16
#endif
#ifndef RS
#define RS 4
#endif
extern "C" __global__ void __launch_bounds__(1024) mmsort8(
    const unsigned short* __restrict__ eids,
    int* __restrict__ eoff,
    unsigned short* __restrict__ plist,
    unsigned int* __restrict__ items,
    unsigned int* __restrict__ nitp,
    const int np)
{
  __shared__ unsigned short se[2048];
  __shared__ int cnt[256];
  __shared__ int eo[257];
  __shared__ int ibase[257];
  const int t = threadIdx.x;
  for (int i = t; i < np; i += 1024) se[i] = eids[i];
  __syncthreads();
  if (t < 256) {
    int c = 0;
    for (int i = 0; i < np; ++i) c += (se[i] == t);
    cnt[t] = c;
  }
  __syncthreads();
  if (t == 0) {
    int s = 0;
    for (int e = 0; e < 256; ++e) { eo[e] = s; s += cnt[e]; }
    eo[256] = s;
    for (int e = 0; e <= 256; ++e) eoff[e] = eo[e];
    int b = 0;
    for (int e = 0; e < 256; ++e) {
      ibase[e] = b;
      b += ((cnt[e] + TS - 1) / TS) * RS;
    }
    ibase[256] = b;
    *nitp = (unsigned int)b;
  }
  __syncthreads();
  // stable scatter (rank-count, atomic-free, deterministic)
  for (int i = t; i < np; i += 1024) {
    const int e = se[i];
    int r = 0;
    for (int j = 0; j < i; ++j) r += (se[j] == e);
    plist[eo[e] + r] = (unsigned short)i;
  }
  __syncthreads();
  // item descriptors: thread e writes ceil(cnt/TS)*RS entries
  if (t < 256) {
    const int tc = (cnt[t] + TS - 1) / TS;
    int k = 0;
    for (int it = 0; it < tc; ++it)
      for (int rs = 0; rs < RS; ++rs)
        items[ibase[t] + (k++)] = ((unsigned int)t << 20) | ((unsigned int)rs << 16)
                                | (unsigned int)(it * TS);
  }
}
