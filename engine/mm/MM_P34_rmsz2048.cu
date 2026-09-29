// MM P34: rmsz2048g -- the GGUF-view trunk RMSNorm (PLAIN w): the converter
// folds (1+w) into every norm.weight except linear_attn.norm (ssm_norm), so
// the engine multiplies the STORED weight directly:
//   y = (x*rstd)*w,  rstd = 1/sqrtf(mean(x^2)+eps), eps = fp32(1e-6).
// Identical reduction order to the P2 rmsz2048 (per-thread strided partials
// {i + 256*t} t asc, warp xor trees, sequential warp-asc sum; IEEE sqrt+div).
// Serves: attn_norm + post_attention_norm (all 40 layers) + output_norm.
extern "C" __global__ void __launch_bounds__(256) rmsz2048g(
    const float* __restrict__ x,    // [P][2048]
    const float* __restrict__ w,    // [2048] F32 (GGUF stored, +1 already folded)
    float* __restrict__ y)          // [P][2048]
{
  const int p = blockIdx.x;
  const float* xr = x + (size_t)p*2048;
  const int th = threadIdx.x;
  __shared__ float wp[8];
  float s = 0.f;
  #pragma unroll
  for (int t = 0; t < 8; ++t) s += xr[th + (t << 8)] * xr[th + (t << 8)];
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((th & 31) == 0) wp[th >> 5] = s;
  __syncthreads();
  if (th == 0) {
    float a = wp[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) a += wp[i];
    const float ms = a * (1.0f/2048.0f);
    wp[0] = 1.0f / sqrtf(ms + 9.999999974752427e-07f);
  }
  __syncthreads();
  const float rstd = wp[0];
  #pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int i = th + (t << 8);
    y[(size_t)p*2048 + i] = (xr[i] * rstd) * w[i];
  }
}
