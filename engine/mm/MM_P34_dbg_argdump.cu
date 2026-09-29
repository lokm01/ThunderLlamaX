
extern "C" __global__ void __launch_bounds__(256) argdump10(
    const float* __restrict__ a0, const float* __restrict__ a1,
    const float* __restrict__ a2, const float* __restrict__ a3,
    const float* __restrict__ a4, signed char* __restrict__ a5,
    float* __restrict__ a6, signed char* __restrict__ a7,
    float* __restrict__ a8, const int* __restrict__ a9)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long* o = (unsigned long long*)a6;   // reuse a6 as the dump target
    o[0] = (unsigned long long)(size_t)a0;
    o[1] = (unsigned long long)(size_t)a1;
    o[2] = (unsigned long long)(size_t)a2;
    o[3] = (unsigned long long)(size_t)a3;
    o[4] = (unsigned long long)(size_t)a4;
    o[5] = (unsigned long long)(size_t)a5;
    o[6] = 0xdeadbeef00ULL + (unsigned long long)(size_t)a7;
    o[7] = (unsigned long long)(size_t)a8;
    o[8] = (unsigned long long)(size_t)a9;
    // no a9 deref (fault-bisect: pointers only)
  }
}
