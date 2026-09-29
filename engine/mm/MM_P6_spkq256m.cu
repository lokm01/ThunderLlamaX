// MM P6: spkq256m -- the long-ctx query core (L > 8192 where the smem score
// buffer LMAX would exceed the 48KB cap). BIT-EXACT vs spkq256/spkq_h_ref by
// construction: the per-position score (per-thread d kdot, warp tree, red[]
// w-asc sum, *0.0625) is computed identically; softmax m = first-then-strict->
// seq max; z = d==0 seq p-asc sum of expf(sco-m); prob = sco/z IEEE div;
// out = per-thread p-asc running acc += prob*vdeq; y = acc*sigmoid(gate).
// Scores live in a GLOBAL scratch row per CTA (row = blockIdx.x, CTXS wide)
// written pass A, read passes B/C -- fp32 store/load roundtrip is the identity.
// CTXS compile-time. THE 8-ARG LAW: aux pointers via ONE VA table,
// ptbl[9] = the scratch base.
#ifndef CTXS
#define CTXS 16384
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(spkq256m_, CTXS)
#define EPS_F 9.999999974752427e-07f
#define SCA_F 0.0625f

extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ qg,        // [T][8192]
    float* __restrict__ y,               // [T][4096]
    const unsigned long long* __restrict__ ptbl)
{
  const float* __restrict__ qw  = (const float*)(size_t)ptbl[1];
  const float* __restrict__ cs  = (const float*)(size_t)ptbl[2];
  const float* __restrict__ sn  = (const float*)(size_t)ptbl[3];
  const signed char* __restrict__ Kq = (const signed char*)(size_t)ptbl[4];
  const float* __restrict__ Ks = (const float*)(size_t)ptbl[5];
  const signed char* __restrict__ Vq = (const signed char*)(size_t)ptbl[6];
  const float* __restrict__ Vs = (const float*)(size_t)ptbl[7];
  const int*    __restrict__ posb = (const int*)(size_t)ptbl[8];
  float*        __restrict__ scr  = (float*)(size_t)ptbl[9];
  const int t = blockIdx.x >> 4;
  const int h = blockIdx.x & 15;
  const int j = h >> 3;
  const int d = threadIdx.x;
  const int pos = posb[0] + t;
  const int L = pos + 1;
  const int warp = d >> 5, lane = d & 31;
  float* myscr = scr + (size_t)blockIdx.x * CTXS;
  __shared__ float wpt[8];
  __shared__ float red[8];
  __shared__ float rstd_s;

  // ---- q norm + rope (VERBATIM spkq256) ----
  const float qraw = qg[t*8192 + h*512 + d];
  float p = qraw*qraw;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
  if (lane == 0) wpt[warp] = p;
  __syncthreads();
  if (d == 0) {
    float s = wpt[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) s += wpt[i];
    rstd_s = 1.0f / sqrtf(s * (1.0f/256.0f) + EPS_F);
  }
  __syncthreads();
  const float rstd = rstd_s;
  __syncthreads();
  float q = (qraw * rstd) * qw[d];
  if (d < 64) {
    const int dp = (d < 32) ? d + 32 : d - 32;
    const float qp = (qg[t*8192 + h*512 + dp] * rstd) * qw[dp];
    const int ci = (d < 32) ? d : d - 32;
    const float c = cs[pos*32 + ci];
    const float s = sn[pos*32 + ci];
    q = (d < 32) ? (q*c + (-qp)*s) : (q*c + qp*s);
  }

  // ---- pass A: scores -> scratch row; d==0 tracks m (first, then strict >) ----
  float mloc = -1e30f;
  for (int pp = 0; pp < L; ++pp) {
    const signed char* krow = Kq + ((size_t)j*CTXS + pp)*256;
    const float kdeq = (float)krow[d] * Ks[((size_t)j*CTXS + pp)*2 + (d >> 7)];
    float s = q * kdeq;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (d == 0) {
      float st = red[0];
      #pragma unroll
      for (int i = 1; i < 8; ++i) st += red[i];
      const float sco = st * SCA_F;
      myscr[pp] = sco;
      if (pp == 0) mloc = sco; else if (sco > mloc) mloc = sco;
    }
    __syncthreads();
  }
  if (d == 0) rstd_s = mloc;
  __syncthreads();
  const float m = rstd_s;
  __syncthreads();

  // ---- pass B: z = seq p-asc sum of expf(sco-m) (d==0, VERBATIM order) ----
  if (d == 0) {
    float z = 0.f;
    for (int pp = 0; pp < L; ++pp) {
      z += expf(myscr[pp] - m);
    }
    rstd_s = z;
  }
  __syncthreads();
  const float z = rstd_s;
  __syncthreads();

  // ---- pass C: out = p-asc running sum prob*vdeq; output gate ----
  float acc = 0.f;
  for (int pp = 0; pp < L; ++pp) {
    const signed char* vrow = Vq + ((size_t)j*CTXS + pp)*256;
    const float vdeq = (float)vrow[d] * Vs[((size_t)j*CTXS + pp)*2 + (d >> 7)];
    // VERBATIM spkq256 order: e = expf(sco-m); prob = e/z (IEEE div); acc += prob*v
    const float e = expf(myscr[pp] - m);
    const float prob = e / z;
    acc += prob * vdeq;
  }
  const float gate = qg[t*8192 + h*512 + 256 + d];
  const float sg = 1.0f/(1.0f + expf(-gate));
  y[(size_t)t*4096 + h*256 + d] = acc * sg;
}
