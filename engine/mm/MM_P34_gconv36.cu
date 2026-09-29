// MM P34: gconv36 -- GDN causal depthwise conv1d k=4 over 8192 channels + silu,
// the T-chain decode form. fp32 in/out. One thread per channel (32 CTAs x 256).
// Per channel c: a = w0*p0; a += w1*p1; a += w2*p2; a += w3*cur (sequential,
// -fmad=false; taps = conv weight [8192][4] channel-major); y = silu(a).
// State [8192][3] fp32 (inputs at t-3,t-2,t-1), updated to the chain's tail.
// silu = v * (1/(1+expf(-v))) matching the numpy port (expf ULP class).
#ifndef TMAX
#define TMAX 3
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(gconv36_, TMAX)
extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ w,        // [8192][4]
    const float* __restrict__ xin,      // [T][8192] pre-conv qkv proj outputs
    float* __restrict__ st,             // [8192][3] state (in-place update)
    float* __restrict__ y)              // [T][8192] silu'd conv output
{
  const int c = blockIdx.x * 256 + threadIdx.x;
  const float w0 = w[c*4+0], w1 = w[c*4+1], w2 = w[c*4+2], w3 = w[c*4+3];
  float p0 = st[c*3+0], p1 = st[c*3+1], p2 = st[c*3+2];
  for (int t = 0; t < TMAX; ++t) {
    const float cur = xin[(size_t)t*8192 + c];
    float a = w0 * p0;
    a += w1 * p1;
    a += w2 * p2;
    a += w3 * cur;
    const float sg = 1.0f/(1.0f + expf(-a));
    y[(size_t)t*8192 + c] = a * sg;
    p0 = p1; p1 = p2; p2 = cur;
  }
  if (c < 8192) { st[c*3+0] = p0; st[c*3+1] = p1; st[c*3+2] = p2; }
}
