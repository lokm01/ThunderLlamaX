// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(32) r_384_32_3_20_8_4_2_4(float* data0_36864, float* data1_15360, float* data2_3, unsigned char* data3_20480, unsigned char* data4_24084480, float* data5_1024) {
  float buf0[3];
  int gidx0 = blockIdx.x; /* 384 */
  int lidx0 = threadIdx.x; /* 32 */
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  for (int Ridx1_0_0 = 0; Ridx1_0_0 < 20; Ridx1_0_0++) {
    int alu3 = ((gidx0*62720)+(lidx0*1960)+(Ridx1_0_0*98));
    unsigned char val0 = (*(data4_24084480+(alu3+1)));
    unsigned char val1 = (*(data4_24084480+alu3));
    for (int Ridx1_0_1_0 = 0; Ridx1_0_1_0 < 8; Ridx1_0_1_0++) {
      int alu4 = (alu3+(Ridx1_0_1_0<<2));
      unsigned char val2 = (*(data4_24084480+(alu4+66)));
      unsigned char val3 = (*(data4_24084480+(alu4+67)));
      unsigned char val4 = (*(data4_24084480+(alu4+68)));
      unsigned char val5 = (*(data4_24084480+(alu4+69)));
      uint alu5 = ((((uint)(val2))<<0u)+(((uint)(val3))<<8u)+(((uint)(val4))<<16u)+(((uint)(val5))<<24u));
      float alu6 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val1))<<((unsigned short)(0u)))+(((unsigned short)(val0))<<((unsigned short)(8u))))))))*(((float)((alu5>>28u)))+0.5f));
      for (int Ridx1_0_1_1 = 0; Ridx1_0_1_1 < 4; Ridx1_0_1_1++) {
        uint alu7 = ((Ridx1_0_1_1!=2)?21u:14u);
        uint alu8 = ((Ridx1_0_1_1!=1)?alu7:7u);
        uint alu9 = ((Ridx1_0_1_1!=0)?alu8:0u);
        int cast0 = ((int)(((alu5>>alu9)&127u)));
        unsigned char cast1 = ((unsigned char)((cast0|((((cast0>>0)^(cast0>>1)^(cast0>>2)^(cast0>>3)^(cast0>>4)^(cast0>>5)^(cast0>>6))&1)<<7))));
        for (int Ridx1_1_0 = 0; Ridx1_1_0 < 2; Ridx1_1_0++) {
          int alu10 = ((Ridx1_0_1_0<<7)+(Ridx1_0_1_1<<5)+(Ridx1_0_0<<10)+(Ridx1_1_0<<4));
          unsigned char val6 = (*(data3_20480+(alu10+1)));
          unsigned char val7 = (*(data3_20480+(alu10+2)));
          unsigned char val8 = (*(data3_20480+(alu10+3)));
          unsigned char val9 = (*(data3_20480+(alu10+4)));
          unsigned char val10 = (*(data3_20480+(alu10+5)));
          unsigned char val11 = (*(data3_20480+(alu10+6)));
          unsigned char val12 = (*(data3_20480+(alu10+7)));
          unsigned char val13 = (*(data3_20480+(alu10+8)));
          unsigned char val14 = (*(data3_20480+(alu10+9)));
          unsigned char val15 = (*(data3_20480+(alu10+10)));
          unsigned char val16 = (*(data3_20480+(alu10+11)));
          unsigned char val17 = (*(data3_20480+(alu10+12)));
          unsigned char val18 = (*(data3_20480+(alu10+13)));
          unsigned char val19 = (*(data3_20480+(alu10+14)));
          unsigned char val20 = (*(data3_20480+(alu10+15)));
          unsigned char val21 = (*(data3_20480+alu10));
          unsigned char val22 = (*(data4_24084480+(alu3+(Ridx1_0_1_0<<3)+(Ridx1_0_1_1<<1)+Ridx1_1_0+2)));
          int alu11 = (Ridx1_1_0<<2);
          int alu12 = ((Ridx1_0_1_0<<5)+(Ridx1_0_1_1<<3)+(Ridx1_0_0<<8)+alu11);
          float4 val23 = (*((float4*)((data1_15360+(alu12+5120)))));
          float4 val24 = (*((float4*)((data1_15360+(alu12+10240)))));
          float4 val25 = (*((float4*)((data1_15360+alu12))));
          float4 val26 = (*((float4*)((data5_1024+(((int)(val22))<<2)))));
          float cast2 = tg_bitcast<float>((uint)(((((uint)(val9))<<0u)+(((uint)(val10))<<8u)+(((uint)(val11))<<16u)+(((uint)(val12))<<24u))));
          float cast3 = tg_bitcast<float>((uint)(((((uint)(val13))<<0u)+(((uint)(val14))<<8u)+(((uint)(val15))<<16u)+(((uint)(val16))<<24u))));
          float cast4 = tg_bitcast<float>((uint)(((((uint)(val17))<<0u)+(((uint)(val18))<<8u)+(((uint)(val19))<<16u)+(((uint)(val20))<<24u))));
          float cast5 = tg_bitcast<float>((uint)(((((uint)(val21))<<0u)+(((uint)(val6))<<8u)+(((uint)(val7))<<16u)+(((uint)(val8))<<24u))));
          int alu13 = (alu11+1);
          int alu14 = (alu11+2);
          int alu15 = (alu11+3);
          unsigned char alu16 = ((alu11!=4)?((unsigned char)(128u)):((unsigned char)(16u)));
          unsigned char alu17 = ((alu11!=3)?alu16:((unsigned char)(8u)));
          unsigned char alu18 = ((alu11!=2)?alu17:((unsigned char)(4u)));
          unsigned char alu19 = ((alu11!=1)?alu18:((unsigned char)(2u)));
          unsigned char alu20 = ((alu11!=0)?alu19:((unsigned char)(1u)));
          float alu21 = ((((cast1/alu20)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu22 = (alu6*val26.x*alu21);
          unsigned char alu23 = ((alu13!=5)?((unsigned char)(128u)):((unsigned char)(32u)));
          unsigned char alu24 = ((alu13!=4)?alu23:((unsigned char)(16u)));
          unsigned char alu25 = ((alu13!=3)?alu24:((unsigned char)(8u)));
          unsigned char alu26 = ((alu13!=2)?alu25:((unsigned char)(4u)));
          unsigned char alu27 = ((alu13!=1)?alu26:((unsigned char)(2u)));
          float alu28 = ((((cast1/alu27)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu29 = (alu6*val26.y*alu28);
          unsigned char alu30 = ((alu14!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu31 = ((alu14!=5)?alu30:((unsigned char)(32u)));
          unsigned char alu32 = ((alu14!=4)?alu31:((unsigned char)(16u)));
          unsigned char alu33 = ((alu14!=3)?alu32:((unsigned char)(8u)));
          unsigned char alu34 = ((alu14!=2)?alu33:((unsigned char)(4u)));
          float alu35 = ((((cast1/alu34)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu36 = (alu6*val26.z*alu35);
          unsigned char alu37 = ((alu15!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu38 = ((alu15!=5)?alu37:((unsigned char)(32u)));
          unsigned char alu39 = ((alu15!=4)?alu38:((unsigned char)(16u)));
          unsigned char alu40 = ((alu15!=3)?alu39:((unsigned char)(8u)));
          float alu41 = ((((cast1/alu40)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu42 = (alu6*val26.w*alu41);
          *(buf0+0) = ((*(buf0+0))+(alu22*val25.x*cast5*0.5f)+(alu29*val25.y*cast2*0.5f)+(alu36*val25.z*cast3*0.5f)+(alu42*val25.w*cast4*0.5f));
          *(buf0+1) = ((*(buf0+1))+(alu22*val23.x*cast5*0.5f)+(alu29*val23.y*cast2*0.5f)+(alu36*val23.z*cast3*0.5f)+(alu42*val23.w*cast4*0.5f));
          *(buf0+2) = ((*(buf0+2))+(alu22*val24.x*cast5*0.5f)+(alu29*val24.y*cast2*0.5f)+(alu36*val24.z*cast3*0.5f)+(alu42*val24.w*cast4*0.5f));
        }
      }
    }
  }
  float val27 = (*(data2_3+2));
  float2 val28 = (*((float2*)((data2_3+0))));
  int alu50 = (lidx0+(gidx0<<5));
  *(data0_36864+alu50) = ((*(buf0+0))*(1/val28.x));
  *(data0_36864+(alu50+12288)) = ((*(buf0+1))*(1/val28.y));
  *(data0_36864+(alu50+24576)) = ((*(buf0+2))*(1/val27));
}