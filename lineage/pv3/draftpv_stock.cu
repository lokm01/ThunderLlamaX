// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(16) r_12_16_16_2_28mtp_sp2B129(float* data0_6144, float* data1_2408376, half* data2_205520896, float* data3_12288, const int data4_) {
  float buf0[2];
  int gidx0 = blockIdx.x; /* 16 */
  int gidx1 = blockIdx.y; /* 12 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu0 = (lidx0+(gidx0<<4));
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < (data4_+1); Ridx0++) {
    half val0 = (*(data2_205520896+(alu0+(Ridx0<<8)+((gidx1/3)*25690112)+102760448)));
    int alu3 = ((gidx1*200698)+Ridx0);
    float val1 = (*(data1_2408376+(alu3+100349)));
    float val2 = (*(data1_2408376+alu3));
    float cast0 = ((float)(val0));
    *(buf0+0) = ((*(buf0+0))+(val2*cast0));
    *(buf0+1) = ((*(buf0+1))+(val1*cast0));
  }
  int alu7 = (alu0+(gidx1<<10));
  float val3 = (*(data3_12288+(alu7+256)));
  float val4 = (*(data3_12288+(alu7+768)));
  int alu8 = (alu0+(gidx1<<9));
  *(data0_6144+alu8) = ((*(buf0+0))*(1/(1.0f+exp2((val3*-1.4426950216293335f)))));
  *(data0_6144+(alu8+256)) = ((*(buf0+1))*(1/(1.0f+exp2((val4*-1.4426950216293335f)))));
}