// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(16) r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4(float* data0_18432, float* data1_7225344, float* data2_72, float* data3_72, half* data4_205520896, float* data5_36864) {
  float buf0[1];
  float buf1[1];
  float buf2[1];
  float buf3[1];
  float buf4[1];
  float buf5[1];
  float buf6[1];
  float buf7[1];
  int gidx1 = blockIdx.y; /* 24 */
  int gidx2 = blockIdx.z; /* 3 */
  int alu0 = (gidx2+(gidx1*3));
  float val0 = (*(data2_72+alu0));
  int gidx0 = blockIdx.x; /* 16 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu1 = (lidx0+(gidx0<<4));
  int alu2 = (gidx1*301056);
  int alu3 = (gidx2*100352);
  int alu4 = ((gidx1/6)*25690112);
  *(buf7+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 12544; Ridx0++) {
    half val1 = (*(data4_205520896+(alu1+(Ridx0<<8)+alu4+102760448)));
    float val2 = (*(data1_7225344+(alu3+Ridx0+alu2)));
    *(buf7+0) = ((*(buf7+0))+(exp2(((val2-val0)*1.4426950216293335f))*((float)(val1))));
  }
  *(buf6+0) = 0.0f;
  for (int Ridx1 = 0; Ridx1 < 3136; Ridx1++) {
    int alu9 = (alu1+(Ridx1<<10)+alu4);
    half val3 = (*(data4_205520896+(alu9+105971712)));
    half val4 = (*(data4_205520896+(alu9+105971968)));
    half val5 = (*(data4_205520896+(alu9+105972224)));
    half val6 = (*(data4_205520896+(alu9+105972480)));
    float4 val7 = (*((float4*)((data1_7225344+(alu3+(Ridx1<<2)+alu2+12544)))));
    *(buf6+0) = ((*(buf6+0))+(exp2(((val7.x-val0)*1.4426950216293335f))*((float)(val3)))+(exp2(((val7.y-val0)*1.4426950216293335f))*((float)(val4)))+(exp2(((val7.z-val0)*1.4426950216293335f))*((float)(val5)))+(exp2(((val7.w-val0)*1.4426950216293335f))*((float)(val6))));
  }
  *(buf5+0) = 0.0f;
  for (int Ridx2 = 0; Ridx2 < 12544; Ridx2++) {
    half val8 = (*(data4_205520896+(alu1+(Ridx2<<8)+alu4+109182976)));
    float val9 = (*(data1_7225344+(alu3+Ridx2+alu2+25088)));
    *(buf5+0) = ((*(buf5+0))+(exp2(((val9-val0)*1.4426950216293335f))*((float)(val8))));
  }
  *(buf4+0) = 0.0f;
  for (int Ridx3 = 0; Ridx3 < 3136; Ridx3++) {
    int alu16 = (alu1+(Ridx3<<10)+alu4);
    half val10 = (*(data4_205520896+(alu16+112394240)));
    half val11 = (*(data4_205520896+(alu16+112394496)));
    half val12 = (*(data4_205520896+(alu16+112394752)));
    half val13 = (*(data4_205520896+(alu16+112395008)));
    float4 val14 = (*((float4*)((data1_7225344+(alu3+(Ridx3<<2)+alu2+37632)))));
    *(buf4+0) = ((*(buf4+0))+(exp2(((val14.x-val0)*1.4426950216293335f))*((float)(val10)))+(exp2(((val14.y-val0)*1.4426950216293335f))*((float)(val11)))+(exp2(((val14.z-val0)*1.4426950216293335f))*((float)(val12)))+(exp2(((val14.w-val0)*1.4426950216293335f))*((float)(val13))));
  }
  *(buf3+0) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 12544; Ridx4++) {
    half val15 = (*(data4_205520896+(alu1+(Ridx4<<8)+alu4+115605504)));
    float val16 = (*(data1_7225344+(alu3+Ridx4+alu2+50176)));
    *(buf3+0) = ((*(buf3+0))+(exp2(((val16-val0)*1.4426950216293335f))*((float)(val15))));
  }
  *(buf2+0) = 0.0f;
  for (int Ridx5 = 0; Ridx5 < 12544; Ridx5++) {
    half val17 = (*(data4_205520896+(alu1+(Ridx5<<8)+alu4+118816768)));
    float val18 = (*(data1_7225344+(alu3+Ridx5+alu2+62720)));
    *(buf2+0) = ((*(buf2+0))+(exp2(((val18-val0)*1.4426950216293335f))*((float)(val17))));
  }
  *(buf1+0) = 0.0f;
  for (int Ridx6 = 0; Ridx6 < 12544; Ridx6++) {
    half val19 = (*(data4_205520896+(alu1+(Ridx6<<8)+alu4+122028032)));
    float val20 = (*(data1_7225344+(alu3+Ridx6+alu2+75264)));
    *(buf1+0) = ((*(buf1+0))+(exp2(((val20-val0)*1.4426950216293335f))*((float)(val19))));
  }
  *(buf0+0) = 0.0f;
  for (int Ridx7 = 0; Ridx7 < 12544; Ridx7++) {
    half val21 = (*(data4_205520896+(alu1+(Ridx7<<8)+alu4+125239296)));
    float val22 = (*(data1_7225344+(alu3+Ridx7+alu2+87808)));
    *(buf0+0) = ((*(buf0+0))+(exp2(((val22-val0)*1.4426950216293335f))*((float)(val21))));
  }
  float val23 = (*(data3_72+alu0));
  float val24 = (*(data5_36864+(alu1+(gidx1<<9)+(gidx2*12288)+256)));
  float alu31 = (1/val23);
  *(data0_18432+(alu1+(gidx1<<8)+(gidx2*6144))) = ((((*(buf7+0))*alu31)+((*(buf6+0))*alu31)+((*(buf5+0))*alu31)+((*(buf4+0))*alu31)+((*(buf3+0))*alu31)+((*(buf2+0))*alu31)+((*(buf1+0))*alu31)+((*(buf0+0))*alu31))*(1/(1.0f+exp2((val24*-1.4426950216293335f)))));
}