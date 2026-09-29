// MM P7: spkq256s -- THE SPLIT-KV LONG-CTX ATTENTION (scores+partials).
// Grid (T*16, S): blockIdx.x = t*16+h; blockIdx.y = split s. 256 thr = 8
// WARPS; positions of split s = [s*L/S, (s+1)*L/S) (integer div -- bijective
// cover of [0,L)); WARP w takes begin+w, begin+w+8, ... (i asc).
// LANE l covers dims l*8..l*8+7 (j asc partial, 5-level xor tree) -- NO
// __syncthreads in the hot loop (the spkq256m 2-syncs-per-position cost was
// the O(L)/CTA wall; warp-per-position + online max-rescale fixes it).
// Per warp (deterministic ONLINE softmax, mirrored exactly by the anchor):
//   m = -3.4e38; z = 0; acc[8] = 0
//   for p asc: s_p = tree(q*kdeq)*0.0625; mn = max(m,s_p); r = expf(m-mn);
//              z = z*r + expf(s_p-mn); acc[j] = acc[j]*r + expf(s_p-mn)*vdeq[j]
//              m = mn
// Partial slot ((bidx*S+s)*8+w)*258: [0]=m [1]=z [2+lane*8+j]=acc[j].
// Preamble (q norm + partial RoPE 64) verbatim spkq256 (thread-per-d).
// THE 8-ARG LAW: aux via the ONE ptbl; pbase direct. CTXS compile-time.
// -fmad=false.
#ifndef CTXS
#define CTXS 16384
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(spkq256s_, CTXS)
#define EPS_F 9.999999974752427e-07f
#define SCA_F 0.0625f

extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ qg,        // [T][8192] (q | gate interleaved/head)
    float* __restrict__ pbase,           // [T*16*S*8][258] partials
    const unsigned long long* __restrict__ ptbl,
    const int S)
{
  const float* __restrict__ qw  = (const float*)(size_t)ptbl[1];
  const float* __restrict__ cs  = (const float*)(size_t)ptbl[2];
  const float* __restrict__ sn  = (const float*)(size_t)ptbl[3];
  const signed char* __restrict__ Kq = (const signed char*)(size_t)ptbl[4];
  const float* __restrict__ Ks = (const float*)(size_t)ptbl[5];
  const signed char* __restrict__ Vq = (const signed char*)(size_t)ptbl[6];
  const float* __restrict__ Vs = (const float*)(size_t)ptbl[7];
  const int* __restrict__ posb = (const int*)(size_t)ptbl[8];
  const int t = blockIdx.x >> 4;
  const int h = blockIdx.x & 15;
  const int s = blockIdx.y;
  const int j_ = h >> 3;
  const int d = threadIdx.x;
  const int pos = posb[0] + t;
  const int L = pos + 1;
  const int warp = d >> 5, lane = d & 31;
  __shared__ float wpt[8];
  __shared__ float qsm[256];
  __shared__ float rstd_s;
  __shared__ float red[8];

  // ---- q norm + rope (verbatim spkq256 preamble) ----
  const float qraw = qg[t*8192 + h*512 + d];
  float p = qraw*qraw;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
  if (lane == 0) wpt[warp] = p;
  __syncthreads();
  if (d == 0) {
    float sv = wpt[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) sv += wpt[i];
    rstd_s = 1.0f / sqrtf(sv * (1.0f/256.0f) + EPS_F);
  }
  __syncthreads();
  const float rstd = rstd_s;
  float q = (qraw * rstd) * qw[d];
  if (d < 64) {
    const int dp = (d < 32) ? d + 32 : d - 32;
    const float qp = (qg[t*8192 + h*512 + dp] * rstd) * qw[dp];
    const int ci = (d < 32) ? d : d - 32;
    const float c = cs[pos*32 + ci];
    const float snv = sn[pos*32 + ci];
    q = (d < 32) ? (q*c + (-qp)*snv) : (q*c + qp*snv);
  }
  qsm[d] = q;
  __syncthreads();

  // ---- the split loop (warp-per-position, online max-rescale) ----
  const int begin = (int)((long long)s * L / S);
  const int end   = (int)((long long)(s + 1) * L / S);
  float qq[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) qq[j] = qsm[lane*8 + j];
  const int kb = lane >> 4;   // the 128-block of this lane's 8 dims
  float m = -3.402823466e38f, z = 0.f;
  float acc[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;

  for (int pp = begin + warp; pp < end; pp += 8) {
    const signed char* krow = Kq + ((size_t)j_*CTXS + pp)*256;
    const float ks = Ks[((size_t)j_*CTXS + pp)*2 + kb];
    float a = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) a += qq[j] * ((float)krow[lane*8 + j] * ks);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    const float sc = a * SCA_F;
    const float mn = (sc > m) ? sc : m;
    const float r = expf(m - mn);
    const float e = expf(sc - mn);
    z = z*r + e;
    const signed char* vrow = Vq + ((size_t)j_*CTXS + pp)*256;
    const float vs = Vs[((size_t)j_*CTXS + pp)*2 + kb];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = acc[j]*r + e * ((float)vrow[lane*8 + j] * vs);
    m = mn;
  }

  // ---- write the partial ----
  float* slot = pbase + ((size_t)((blockIdx.x)*S + s)*8 + warp)*258;
  if (lane == 0) { slot[0] = m; slot[1] = z; }
  #pragma unroll
  for (int j = 0; j < 8; ++j) slot[2 + lane*8 + j] = acc[j];
}
