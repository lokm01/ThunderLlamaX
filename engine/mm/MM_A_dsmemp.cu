// MM SESSION A (dyn-smem probe): dsmemp -- dynamic-smem request kernel.
// extern __shared__ (zero static smem in the ELF); the host patches the QMD
// shared_memory_size + smem carveout cfg to SZ+0x400 before launch. Writes
// a pattern across all SZ bytes, syncs, reads it back XOR-scrambled into
// out[1024]; the host recomputes the same sums -> PASS/FAULT per size.
// Gates the L10 structural branch (>48KB smem through OUR QMD path; the
// CUDA runtime's cudaFuncSetAttribute does not exist here).
extern "C" __global__ void __launch_bounds__(1024) dsmemp(
    unsigned int* __restrict__ out,
    const int words)
{
  extern __shared__ unsigned int sm[];
  const int tid = threadIdx.x;
  for (int i = tid; i < words; i += 1024) sm[i] = (unsigned int)i * 2654435761u + 7u;
  __syncthreads();
  unsigned int a = 0u;
  for (int i = tid; i < words; i += 1024) a ^= sm[i] + (unsigned int)i * 2246822519u + (unsigned int)tid;
  out[tid] = a;
}
