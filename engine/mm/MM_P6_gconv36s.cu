// MM P6: gconv36s -- gconv36 (math VERBATIM) + per-t conv-state slots.
// After the shift for token t, (p0,p1,p2) = the state feeding token t+1:
// written to cslotsL[t*CSTRIDE + c*3 + {0,1,2}]. CSTRIDE = 30*24576 = 737280.
#ifndef TMAX
#define TMAX 3
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(gconv36s_, TMAX)
#define CSTRIDE 737280
extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ w,        // [8192][4]
    const float* __restrict__ xin,      // [T][8192]
    float* __restrict__ st,             // [8192][3] live (in-place update)
    float* __restrict__ y,              // [T][8192]
    float* __restrict__ cslotsL)        // per-layer slice of CSLOTS [T][30][8192][3]
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
    float* dst = cslotsL + (size_t)t*CSTRIDE + c*3;
    dst[0] = p0; dst[1] = p1; dst[2] = p2;
  }
  st[c*3+0] = p0; st[c*3+1] = p1; st[c*3+2] = p2;
}
