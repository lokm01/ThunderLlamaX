// MM SESSION E (item 3): k2nz36 (split from MM_E_k2sh.cu -- ONE SYMBOL
// PER CUBIN, the loader law)
#define EPS_F 9.999999974752427e-07f
// MM SESSION E (item 3): k2nz36 -- the gated-RMSNorm apply (the cross-CTA
// epilogue of k2s36h). One CTA per (t, head), 128 threads; yss = half0 +
// half1 (FIXED order); same formula/op order as the stock fused epilogue.
extern "C" __global__ void __launch_bounds__(128) k2nz36(
    const float* __restrict__ YQB,       // [T][4096]
    const float* __restrict__ YSSB,      // [T][64]
    const float* __restrict__ wn,        // [128]
    const float* __restrict__ z,         // [T][4096]
    float* __restrict__ y,               // [T][4096]
    const int T)
{
  const int th = blockIdx.x;
  const int t = th >> 5, h = th & 31;
  const int d = threadIdx.x;
  const float yss = YSSB[(size_t)t*64 + h] + YSSB[(size_t)t*64 + 32 + h];
  const float ms = yss * (1.0f/128.0f);
  const float rstd = 1.0f / sqrtf(ms + EPS_F);
  const float qd = YQB[(size_t)t*4096 + h*128 + d];
  const float zg = z[(size_t)t*4096 + h*128 + d];
  const float sg = 1.0f/(1.0f + expf(-zg));
  y[(size_t)t*4096 + h*128 + d] = ((qd*rstd)*wn[d]) * (zg*sg);
}
