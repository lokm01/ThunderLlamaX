// MM P1 PRODUCTION v0: rt8e256 -- the MoE router for qwen35moe decode.
// One CTA per position; 1024 threads = 256 experts x 4 parts.
//   logits: per-thread sequential float4 sums -> smem parts -> deterministic
//           4-way reduce (POC-rt8e256poc verbatim, BIT-EXACT vs numpy).
//   shared-expert gate: 2 elems/thread -> smem -> tree reduce (fixed order).
//   epilogue (thread 0, sequential, THE GOLD-ROUTER CONTRACT):
//     m = max(logits);  top-8 by 8 masked-argmax passes (strict >, tie->lower id)
//     gates[r] = expf(l[sel_r]-m) / sum_k expf(l[sel_k]-m)   (softmax->top8->renorm)
//     sg = 1/(1+expf(-dot(wsh,h)))
//   outputs: eids[P*8] u16 rank-major, gates[P*8] f32 renormed, sg[P] f32.
// ids/top-8 are BIT-EXACT vs the numpy ref (logit ordering); gates/sg carry
// expf-vs-np.exp ULP tolerance (checked allclose in the harness).
#define NP 88
extern "C" __global__ void __launch_bounds__(1024) rt8e256(
    const float* __restrict__ wrt,        // [256][2048] F32 router
    const float* __restrict__ wsh,        // [2048] F32 shared-expert gate w
    const float* __restrict__ h,          // [NP][2048]
    unsigned short* __restrict__ eids,    // [NP][8]
    float* __restrict__ gates,            // [NP][8]
    float* __restrict__ sg)               // [NP]
{
  const int p = blockIdx.x;
  __shared__ float lg[256];
  __shared__ float parts[256][4];
  __shared__ float shp[1024];
  const float* hrow = h + (size_t)p*2048;
  const int e = threadIdx.x >> 2, part = threadIdx.x & 3;
  const float* wrow = wrt + (size_t)e*2048 + part*512;
  float s = 0.f;
  for (int i = 0; i < 512; i += 4) {
    const float4 wv = *(const float4*)(wrow + i);
    const float4 hv = *(const float4*)(hrow + part*512 + i);
    s += wv.x*hv.x; s += wv.y*hv.y; s += wv.z*hv.z; s += wv.w*hv.w;
  }
  parts[e][part] = s;
  shp[threadIdx.x] = wsh[threadIdx.x*2]*hrow[threadIdx.x*2]
                   + wsh[threadIdx.x*2+1]*hrow[threadIdx.x*2+1];
  __syncthreads();
  if (part == 0) lg[e] = parts[e][0] + parts[e][1] + parts[e][2] + parts[e][3];
  __syncthreads();
  if (threadIdx.x == 0) {
    // shared-gate tree reduce (fixed order over 1024 partials)
    float a = shp[0];
    for (int i = 1; i < 1024; ++i) a += shp[i];
    sg[p] = 1.f/(1.f + __expf(-a));
    // softmax max = top-1 logit
    float m = lg[0];
    for (int i = 1; i < 256; ++i) if (lg[i] > m) m = lg[i];
    float s8 = 0.f;
    for (int r = 0; r < 8; ++r) {
      int be = 0; float bv = lg[0];
      for (int e2 = 1; e2 < 256; ++e2) {
        if (lg[e2] > bv) { bv = lg[e2]; be = e2; }
      }
      eids[p*8 + r] = (unsigned short)be;
      gates[p*8 + r] = __expf(bv - m);
      s8 += __expf(bv - m);
      lg[be] = -3.402823466e38f;
    }
    for (int r = 0; r < 8; ++r) gates[p*8 + r] = gates[p*8 + r] / s8;
  }
}
