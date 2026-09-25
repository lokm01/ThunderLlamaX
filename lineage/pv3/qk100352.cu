// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
extern "C" __global__ void __launch_bounds__(48) r_3_25088_4_4_3_4_2_16_4(float* data0_7225344, float* data1_18432, half* data2_205520896, const int data3_) {
  float buf0[8];
  float buf1[8];
  __shared__ __align__(16) float buf2[384];
  int gidx0 = blockIdx.x; /* 25088 */
  int gidx1 = blockIdx.y; /* 3 */
  int lidx0 = threadIdx.x; /* 4 */
  int lidx1 = threadIdx.y; /* 4 */
  int lidx2 = threadIdx.z; /* 3 */
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  *(buf0+4) = 0.0f;
  *(buf0+5) = 0.0f;
  *(buf0+6) = 0.0f;
  *(buf0+7) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    int alu8 = ((lidx0<<2)+(Ridx0<<4));
    int alu9 = (alu8+(gidx0<<10)+(gidx1*25690112)+(((gidx1+lidx1)/3)*25690112));
    half4 val0 = (*((half4*)((data2_205520896+(alu9+256)))));
    half4 val1 = (*((half4*)((data2_205520896+(alu9+512)))));
    half4 val2 = (*((half4*)((data2_205520896+(alu9+768)))));
    half4 val3 = (*((half4*)((data2_205520896+alu9))));
    int alu10 = (alu8+(lidx2<<8)+(gidx1*6144)+(lidx1*1536));
    float4 val4 = (*((float4*)((data1_18432+(alu10+768)))));
    float4 val5 = (*((float4*)((data1_18432+alu10))));
    float cast0 = ((float)(val0.x));
    float cast1 = ((float)(val0.y));
    float cast2 = ((float)(val0.z));
    float cast3 = ((float)(val0.w));
    float cast4 = ((float)(val1.x));
    float cast5 = ((float)(val1.y));
    float cast6 = ((float)(val1.z));
    float cast7 = ((float)(val1.w));
    float cast8 = ((float)(val2.x));
    float cast9 = ((float)(val2.y));
    float cast10 = ((float)(val2.z));
    float cast11 = ((float)(val2.w));
    float cast12 = ((float)(val3.x));
    float cast13 = ((float)(val3.y));
    float cast14 = ((float)(val3.z));
    float cast15 = ((float)(val3.w));
    *(buf0+0) = ((*(buf0+0))+(val5.x*cast12)+(val5.y*cast13)+(val5.z*cast14)+(val5.w*cast15));
    *(buf0+1) = ((*(buf0+1))+(val4.x*cast12)+(val4.y*cast13)+(val4.z*cast14)+(val4.w*cast15));
    *(buf0+2) = ((*(buf0+2))+(val5.x*cast0)+(val5.y*cast1)+(val5.z*cast2)+(val5.w*cast3));
    *(buf0+3) = ((*(buf0+3))+(val4.x*cast0)+(val4.y*cast1)+(val4.z*cast2)+(val4.w*cast3));
    *(buf0+4) = ((*(buf0+4))+(val5.x*cast4)+(val5.y*cast5)+(val5.z*cast6)+(val5.w*cast7));
    *(buf0+5) = ((*(buf0+5))+(val4.x*cast4)+(val4.y*cast5)+(val4.z*cast6)+(val4.w*cast7));
    *(buf0+6) = ((*(buf0+6))+(val5.x*cast8)+(val5.y*cast9)+(val5.z*cast10)+(val5.w*cast11));
    *(buf0+7) = ((*(buf0+7))+(val4.x*cast8)+(val4.y*cast9)+(val4.z*cast10)+(val4.w*cast11));
  }
  int alu20 = (lidx1<<5);
  int alu21 = (lidx2<<7);
  int alu22 = ((lidx0<<3)+alu20+alu21);
  *((float4*)((buf2+(alu22+4)))) = make_float4((*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7)));
  *((float4*)((buf2+alu22))) = make_float4((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3)));
  __syncthreads();
  *(buf1+0) = 0.0f;
  *(buf1+1) = 0.0f;
  *(buf1+2) = 0.0f;
  *(buf1+3) = 0.0f;
  *(buf1+4) = 0.0f;
  *(buf1+5) = 0.0f;
  *(buf1+6) = 0.0f;
  *(buf1+7) = 0.0f;
  for (int Ridx104 = 0; Ridx104 < 4; Ridx104++) {
    int alu34 = (alu20+(Ridx104<<3)+alu21);
    float4 val6 = (*((float4*)((buf2+(alu34+4)))));
    float4 val7 = (*((float4*)((buf2+alu34))));
    *(buf1+0) = ((*(buf1+0))+val7.x);
    *(buf1+1) = ((*(buf1+1))+val7.y);
    *(buf1+2) = ((*(buf1+2))+val7.z);
    *(buf1+3) = ((*(buf1+3))+val7.w);
    *(buf1+4) = ((*(buf1+4))+val6.x);
    *(buf1+5) = ((*(buf1+5))+val6.y);
    *(buf1+6) = ((*(buf1+6))+val6.z);
    *(buf1+7) = ((*(buf1+7))+val6.w);
  }
  int alu44 = (data3_+1);
  int alu45 = (data3_+2);
  int alu46 = (gidx0<<2);
  int alu47 = (alu46+1);
  int alu48 = (alu46+2);
  int alu49 = (alu46+3);
  bool alu50 = (lidx2!=0);
  bool alu51 = (lidx2!=1);
  float alu52 = ((data3_<alu46)?((float)(-INFINITY)):0.0f);
  float alu53 = ((alu44<alu46)?((float)(-INFINITY)):0.0f);
  float alu54 = ((alu45<alu46)?((float)(-INFINITY)):0.0f);
  float alu55 = (alu51?alu54:alu53);
  float alu56 = (alu50?alu55:alu52);
  float alu57 = ((data3_<alu47)?((float)(-INFINITY)):0.0f);
  float alu58 = ((alu44<alu47)?((float)(-INFINITY)):0.0f);
  float alu59 = ((alu45<alu47)?((float)(-INFINITY)):0.0f);
  float alu60 = (alu51?alu59:alu58);
  float alu61 = (alu50?alu60:alu57);
  float alu62 = ((data3_<alu48)?((float)(-INFINITY)):0.0f);
  float alu63 = ((alu44<alu48)?((float)(-INFINITY)):0.0f);
  float alu64 = ((alu45<alu48)?((float)(-INFINITY)):0.0f);
  float alu65 = (alu51?alu64:alu63);
  float alu66 = (alu50?alu65:alu62);
  float alu67 = ((data3_<alu49)?((float)(-INFINITY)):0.0f);
  float alu68 = ((alu44<alu49)?((float)(-INFINITY)):0.0f);
  float alu69 = ((alu45<alu49)?((float)(-INFINITY)):0.0f);
  float alu70 = (alu51?alu69:alu68);
  float alu71 = (alu50?alu70:alu67);
  int alu72 = (alu46+(lidx2*100352)+(gidx1*2408448)+(lidx1*602112));
  bool alu73 = (lidx0==0);
  if (alu73) {
    *((float4*)((data0_7225344+(alu72+301056)))) = make_float4((((*(buf1+1))*0.0625f)+alu56),(((*(buf1+3))*0.0625f)+alu61),(((*(buf1+5))*0.0625f)+alu66),(((*(buf1+7))*0.0625f)+alu71));
  }
  if (alu73) {
    *((float4*)((data0_7225344+alu72))) = make_float4((((*(buf1+0))*0.0625f)+alu56),(((*(buf1+2))*0.0625f)+alu61),(((*(buf1+4))*0.0625f)+alu66),(((*(buf1+6))*0.0625f)+alu71));
  }
}