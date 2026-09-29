// MM P34: spka256 -- full-attn KV append (int8 g128 cache).
// Grid (T*2): blockIdx.x = t*2 + j (kv-head j, token t); 256 threads = dims.
// THE 8-ARG LAW: >8 buffer args fault the dext (proven live: a 10-arg
// pointer-dump kernel faults with no derefs at all; every clean kernel this
// rig has ever run has <=8). All aux pointers arrive via ONE device VA table:
//   ptbl[0]=kw [256] | [1]=qw [256] (spkq) | [2]=cos [CTX][32] | [3]=sin
//   [4]=Kq [2][CTX][256] i8 | [5]=Ks | [6]=Vq i8 | [7]=Vs | [8]=posb int*
// k: rmsnorm over 256 ((k*rstd)*kw PLAIN -- +1 folded in the GGUF), partial
// RoPE 64 (cat-dup cos/sin; partner recomputed locally = bitwise identical),
// int8 g128 quant (s = max|.|/127 per 128-block, 0->1; q = rintf(x/s)).
// v: quant only. pos from the DEVICE posb[0] (graph-safe). -fmad=false.
#define EPS_F 9.999999974752427e-07f
#ifndef CTXS
#define CTXS 1024
#endif
#ifdef RENAMED
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(spka256m_, CTXS)
#else
#define KSYM spka256
#endif

__device__ __forceinline__ float blkmax128(float a, float* mxs, int b, int warp, int lane) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  if (lane == 0) mxs[b*4 + (warp & 3)] = a;
  __syncthreads();
  float m = mxs[b*4+0];
  #pragma unroll
  for (int i = 1; i < 4; ++i) m = fmaxf(m, mxs[b*4+i]);
  return m;
}

extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ kq,        // [T][512]
    const float* __restrict__ vq,        // [T][512]
    const unsigned long long* __restrict__ ptbl)
{
  const float* __restrict__ kw  = (const float*)(size_t)ptbl[0];
  const float* __restrict__ cs  = (const float*)(size_t)ptbl[2];
  const float* __restrict__ sn  = (const float*)(size_t)ptbl[3];
  signed char*  __restrict__ Kq = (signed char*)(size_t)ptbl[4];
  float*        __restrict__ Ks = (float*)(size_t)ptbl[5];
  signed char*  __restrict__ Vq = (signed char*)(size_t)ptbl[6];
  float*        __restrict__ Vs = (float*)(size_t)ptbl[7];
  const int*    __restrict__ posb = (const int*)(size_t)ptbl[8];
  const int t = blockIdx.x >> 1;
  const int j = blockIdx.x & 1;
  const int d = threadIdx.x;
  const int pos = posb[0] + t;
  const int warp = d >> 5, lane = d & 31;
  __shared__ float wpt[8];
  __shared__ float mxs[8];
  __shared__ float rstd_s;

  // ---- k norm + rope ----
  const float kraw = kq[t*512 + j*256 + d];
  float p = kraw*kraw;
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
  float k = (kraw * rstd) * kw[d];
  if (d < 64) {
    const int dp = (d < 32) ? d + 32 : d - 32;
    const float kp = (kq[t*512 + j*256 + dp] * rstd) * kw[dp];
    const int ci = (d < 32) ? d : d - 32;
    const float c = cs[pos*32 + ci];
    const float s = sn[pos*32 + ci];
    k = (d < 32) ? (k*c + (-kp)*s) : (k*c + kp*s);
  }
  __syncthreads();   // mxs reuse barrier

  // ---- k int8 quant + store ----
  const int b = d >> 7;
  {
    const float mk = blkmax128(fabsf(k), mxs, b, warp, lane);
    float sb = mk / 127.0f;
    if (sb == 0.f) sb = 1.0f;
    Kq[((size_t)j*CTXS + pos)*256 + d] = (signed char)rintf(k / sb);
    if (d == 0 || d == 128) Ks[((size_t)j*CTXS + pos)*2 + b] = sb;
  }
  __syncthreads();

  // ---- v quant + store ----
  const float v = vq[t*512 + j*256 + d];
  {
    const float mv = blkmax128(fabsf(v), mxs, b, warp, lane);
    float sb = mv / 127.0f;
    if (sb == 0.f) sb = 1.0f;
    Vq[((size_t)j*CTXS + pos)*256 + d] = (signed char)rintf(v / sb);
    if (d == 0 || d == 128) Vs[((size_t)j*CTXS + pos)*2 + b] = sb;
  }
}
