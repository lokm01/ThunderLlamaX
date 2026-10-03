extern "C" __global__ void sst(float* o){ __shared__ float a[16384]; a[threadIdx.x]=threadIdx.x; o[threadIdx.x]=a[threadIdx.x]; }
