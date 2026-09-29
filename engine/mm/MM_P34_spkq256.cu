// MM P34: spkq256 -- the full-attention query core (head_dim 256, GQA 16:2).
// Grid (T*16): blockIdx.x = t*16 + h; 256 threads = dims d. THE 8-ARG LAW:
// aux pointers via ONE device VA table:
//   ptbl[0]=kw | [1]=qw | [2]=cos | [3]=sin | [4]=Kq i8 | [5]=Ks | [6]=Vq i8
//   | [7]=Vs | [8]=posb int*  (one table per attn layer)
// q = qg[t][h*512 + d] (attn_q PER-HEAD INTERLEAVED: +256 = the gate);
// qhat = (q*rstd)*qw (plain w, +1 folded); partial RoPE 64; scores vs the
// int8 KV of kv-head j = h>>3 up to pos INCLUSIVE (causal by loop bound):
// s[p] = tree256(q*kdeq) * 0.0625, p asc; fp32 softmax (seq max, exp, seq
// sum, IEEE div); out[d] = running sum p asc prob*vdeq; y = out*sigmoid(gate)
// [the OUTPUT GATE epilogue]. LMAX/CTXS compile-time. -fmad=false.
#ifndef LMAX
#define LMAX 1024
#endif
#ifndef CTXS
#define CTXS 1024
#endif
#define EPS_F 9.999999974752427e-07f
#define SCA_F 0.0625f

extern "C" __global__ void __launch_bounds__(256) spkq256(
    const float* __restrict__ qg,        // [T][8192] (q | gate interleaved/head)
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
  const int* __restrict__ posb = (const int*)(size_t)ptbl[8];
  const int t = blockIdx.x >> 4;
  const int h = blockIdx.x & 15;
  const int j = h >> 3;
  const int d = threadIdx.x;
  const int pos = posb[0] + t;
  const int L = pos + 1;
  const int warp = d >> 5, lane = d & 31;
  __shared__ float wpt[8];
  __shared__ float sco[LMAX];
  __shared__ float rstd_s;
  __shared__ float red[8];

  // ---- q norm + rope ----
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
  float q = (qraw * rstd) * qw[d];
  if (d < 64) {
    const int dp = (d < 32) ? d + 32 : d - 32;
    const float qp = (qg[t*8192 + h*512 + dp] * rstd) * qw[dp];
    const int ci = (d < 32) ? d : d - 32;
    const float c = cs[pos*32 + ci];
    const float s = sn[pos*32 + ci];
    q = (d < 32) ? (q*c + (-qp)*s) : (q*c + qp*s);
  }

  // ---- scores: per position p asc ----
  for (int pp = 0; pp < L; ++pp) {
    const signed char* krow = Kq + ((size_t)j*CTXS + pp)*256;
    const float kdeq = (float)krow[d] * Ks[((size_t)j*CTXS + pp)*2 + (d >> 7)];
    float a = q * kdeq;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[warp] = a;
    __syncthreads();
    if (d == 0) {
      float s = red[0];
      #pragma unroll
      for (int i = 1; i < 8; ++i) s += red[i];
      sco[pp] = s * SCA_F;
    }
    __syncthreads();
  }

  // ---- softmax ----
  // P6 FIX (the L>256 LATENT BUG): the old parallel-exp form only exp'd
  // positions < 256 (thread count); L in 257..LMAX mixed RAW scores into
  // z/out. Restructured d==0-sequential exp + divide: BIT-IDENTICAL for
  // L <= 256 (same expf inputs, same p-asc sum, same per-element IEEE div),
  // correct for all L <= LMAX.
  if (d == 0) {
    float m = sco[0];
    for (int pp = 1; pp < L; ++pp) { if (sco[pp] > m) m = sco[pp]; }
    for (int pp = 0; pp < L; ++pp) sco[pp] = expf(sco[pp] - m);
    float z = 0.f;
    for (int pp = 0; pp < L; ++pp) z += sco[pp];
    for (int pp = 0; pp < L; ++pp) sco[pp] = sco[pp] / z;
  }
  __syncthreads();

  // ---- out: running sum p asc + output gate ----
  float acc = 0.f;
  for (int pp = 0; pp < L; ++pp) {
    const signed char* vrow = Vq + ((size_t)j*CTXS + pp)*256;
    const float vdeq = (float)vrow[d] * Vs[((size_t)j*CTXS + pp)*2 + (d >> 7)];
    acc += sco[pp] * vdeq;
  }
  const float gate = qg[t*8192 + h*512 + 256 + d];
  const float sg = 1.0f/(1.0f + expf(-gate));
  y[(size_t)t*4096 + h*256 + d] = acc * sg;
}
