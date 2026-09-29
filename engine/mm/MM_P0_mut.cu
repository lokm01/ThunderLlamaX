// MM_P0 D3d/D4: mm_eidmut — device-side pair-list mutator (host NEVER rewrites).
// eids[i] = (eids[i]*7 + 3) mod NBANK: affine, deterministic, changes every replay.
// NBANK/NPAIR baked (hardcode-sizes law). One CTA of 128 threads.
#define NPAIR 88
#define NBANK 5120

extern "C" __global__ void __launch_bounds__(128) mm_eidmut(
    unsigned short* __restrict__ eids)
{
  const int i = threadIdx.x;
  if (i < NPAIR) eids[i] = (unsigned short)((eids[i]*7 + 3) % NBANK);
}
