// MM P7: spkc256 -- the split-KV COMBINE + output-gate epilogue.
// Grid (T*16,): blockIdx.x = t*16+h; 256 thr = dims d. Merges NP = 8*S
// partials (slot i = ((t*16+h)*S + i/8)*8 + i%8 -- i asc = split asc, warp
// asc; mirrors the anchor exactly):
//   m* = strict max over i asc (thread-0 smem reduce is NOT used -- each
//        thread reads all m_i; L2-hot, deterministic)
//   w_i = expf(m_i - m*)
//   Z   = Z + w_i*z_i        (i asc)
//   acc = acc + w_i*out_i[d] (i asc, per thread)
//   y[t][h*256+d] = (acc / Z) * sigmoid(gate)   [IEEE div, then the gate]
// Empty partials carry m=-3.4e38, z=0 -> w_i = expf(very neg) = 0.
extern "C" __global__ void __launch_bounds__(256) spkc256(
    const float* __restrict__ qg,        // [T][8192] (gate = +256 per head)
    float* __restrict__ y,               // [T][4096]
    const float* __restrict__ pbase,     // [T*16*S*8][258]
    const int NP)
{
  const int t = blockIdx.x >> 4;
  const int h = blockIdx.x & 15;
  const int d = threadIdx.x;
  const float* base = pbase + (size_t)blockIdx.x * (NP*258);
  float mstar = base[0];
  for (int i = 1; i < NP; ++i) { const float mi = base[(size_t)i*258]; if (mi > mstar) mstar = mi; }
  float Z = 0.f, acc = 0.f;
  for (int i = 0; i < NP; ++i) {
    const float* sl = base + (size_t)i*258;
    const float w = expf(sl[0] - mstar);
    Z = Z + w * sl[1];
    acc = acc + w * sl[2 + d];
  }
  const float gate = qg[(size_t)t*8192 + h*512 + 256 + d];
  const float sg = 1.0f/(1.0f + expf(-gate));
  y[(size_t)t*4096 + h*256 + d] = (acc / Z) * sg;
}
