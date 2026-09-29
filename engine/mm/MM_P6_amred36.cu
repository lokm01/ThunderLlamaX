// MM P6: amred36 -- reduce PART[PP][7760] packed u64 -> amds[PP] (greedy top-1
// per probe position, tie -> lower vocab row). 1 CTA x 256 thr; PP runtime arg.
#define NCTA 7760
extern "C" __global__ void __launch_bounds__(256) amred36(
    const unsigned long long* __restrict__ part,  // [PP][NCTA]
    int* __restrict__ amds,                       // [PP]
    const int PP)
{
  __shared__ __align__(16) unsigned long long sm[8];
  const int tid = threadIdx.x;
  for (int p = 0; p < PP; ++p) {
    const unsigned long long* row = part + (size_t)p*NCTA;
    unsigned long long best = 0ull;
    for (int i = tid; i < NCTA; i += 256) { const unsigned long long c = row[i]; if (c > best) best = c; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const unsigned long long ov = __shfl_down_sync(0xffffffffu, best, o);
      if (ov > best) best = ov;
    }
    if ((tid & 31) == 0) sm[tid >> 5] = best;
    __syncthreads();
    if (tid == 0) {
      unsigned long long b_ = sm[0];
      #pragma unroll
      for (int w_ = 1; w_ < 8; ++w_) { if (sm[w_] > b_) b_ = sm[w_]; }
      amds[p] = (int)(262143ull - (b_ & 0x3FFFFull));
    }
    __syncthreads();
  }
}
