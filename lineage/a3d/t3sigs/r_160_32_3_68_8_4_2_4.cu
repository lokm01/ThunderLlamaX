// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(32) r_160_32_3_68_8_4_2_4(float* data0_15360, float* data1_15360, float* data2_52224, unsigned char* data3_34119680, float* data4_1024) {
  float buf0[3];
  int gidx0 = blockIdx.x; /* 160 */
  int lidx0 = threadIdx.x; /* 32 */
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  for (int Ridx1_0_0 = 0; Ridx1_0_0 < 68; Ridx1_0_0++) {
    int alu3 = ((gidx0*213248)+(lidx0*6664)+(Ridx1_0_0*98));
    unsigned char val0 = (*(data3_34119680+(alu3+1)));
    unsigned char val1 = (*(data3_34119680+alu3));
    for (int Ridx1_0_1_0 = 0; Ridx1_0_1_0 < 8; Ridx1_0_1_0++) {
      int alu4 = (alu3+(Ridx1_0_1_0<<2));
      unsigned char val2 = (*(data3_34119680+(alu4+66)));
      unsigned char val3 = (*(data3_34119680+(alu4+67)));
      unsigned char val4 = (*(data3_34119680+(alu4+68)));
      unsigned char val5 = (*(data3_34119680+(alu4+69)));
      uint alu5 = ((((uint)(val2))<<0u)+(((uint)(val3))<<8u)+(((uint)(val4))<<16u)+(((uint)(val5))<<24u));
      float alu6 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val1))<<((unsigned short)(0u)))+(((unsigned short)(val0))<<((unsigned short)(8u))))))))*(((float)((alu5>>28u)))+0.5f));
      for (int Ridx1_0_1_1 = 0; Ridx1_0_1_1 < 4; Ridx1_0_1_1++) {
        uint alu7 = ((Ridx1_0_1_1!=2)?21u:14u);
        uint alu8 = ((Ridx1_0_1_1!=1)?alu7:7u);
        uint alu9 = ((Ridx1_0_1_1!=0)?alu8:0u);
        int cast0 = ((int)(((alu5>>alu9)&127u)));
        unsigned char cast1 = ((unsigned char)((cast0|((((cast0>>0)^(cast0>>1)^(cast0>>2)^(cast0>>3)^(cast0>>4)^(cast0>>5)^(cast0>>6))&1)<<7))));
        for (int Ridx1_1_0 = 0; Ridx1_1_0 < 2; Ridx1_1_0++) {
          unsigned char val6 = (*(data3_34119680+(alu3+(Ridx1_0_1_0<<3)+(Ridx1_0_1_1<<1)+Ridx1_1_0+2)));
          int alu10 = (Ridx1_1_0<<2);
          int alu11 = ((Ridx1_0_1_0<<5)+(Ridx1_0_1_1<<3)+(Ridx1_0_0<<8)+alu10);
          float4 val7 = (*((float4*)((data2_52224+(alu11+17408)))));
          float4 val8 = (*((float4*)((data2_52224+(alu11+34816)))));
          float4 val9 = (*((float4*)((data2_52224+alu11))));
          float4 val10 = (*((float4*)((data4_1024+(((int)(val6))<<2)))));
          int alu12 = (alu10+1);
          int alu13 = (alu10+2);
          int alu14 = (alu10+3);
          unsigned char alu15 = ((alu10!=4)?((unsigned char)(128u)):((unsigned char)(16u)));
          unsigned char alu16 = ((alu10!=3)?alu15:((unsigned char)(8u)));
          unsigned char alu17 = ((alu10!=2)?alu16:((unsigned char)(4u)));
          unsigned char alu18 = ((alu10!=1)?alu17:((unsigned char)(2u)));
          unsigned char alu19 = ((alu10!=0)?alu18:((unsigned char)(1u)));
          float alu20 = ((((cast1/alu19)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu21 = (alu6*val10.x*alu20);
          unsigned char alu22 = ((alu12!=5)?((unsigned char)(128u)):((unsigned char)(32u)));
          unsigned char alu23 = ((alu12!=4)?alu22:((unsigned char)(16u)));
          unsigned char alu24 = ((alu12!=3)?alu23:((unsigned char)(8u)));
          unsigned char alu25 = ((alu12!=2)?alu24:((unsigned char)(4u)));
          unsigned char alu26 = ((alu12!=1)?alu25:((unsigned char)(2u)));
          float alu27 = ((((cast1/alu26)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu28 = (alu6*val10.y*alu27);
          unsigned char alu29 = ((alu13!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu30 = ((alu13!=5)?alu29:((unsigned char)(32u)));
          unsigned char alu31 = ((alu13!=4)?alu30:((unsigned char)(16u)));
          unsigned char alu32 = ((alu13!=3)?alu31:((unsigned char)(8u)));
          unsigned char alu33 = ((alu13!=2)?alu32:((unsigned char)(4u)));
          float alu34 = ((((cast1/alu33)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu35 = (alu6*val10.z*alu34);
          unsigned char alu36 = ((alu14!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu37 = ((alu14!=5)?alu36:((unsigned char)(32u)));
          unsigned char alu38 = ((alu14!=4)?alu37:((unsigned char)(16u)));
          unsigned char alu39 = ((alu14!=3)?alu38:((unsigned char)(8u)));
          float alu40 = ((((cast1/alu39)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu41 = (alu6*val10.w*alu40);
          *(buf0+0) = ((*(buf0+0))+(alu21*val9.x*0.5f)+(alu28*val9.y*0.5f)+(alu35*val9.z*0.5f)+(alu41*val9.w*0.5f));
          *(buf0+1) = ((*(buf0+1))+(alu21*val7.x*0.5f)+(alu28*val7.y*0.5f)+(alu35*val7.z*0.5f)+(alu41*val7.w*0.5f));
          *(buf0+2) = ((*(buf0+2))+(alu21*val8.x*0.5f)+(alu28*val8.y*0.5f)+(alu35*val8.z*0.5f)+(alu41*val8.w*0.5f));
        }
      }
    }
  }
  int alu49 = (lidx0+(gidx0<<5));
  float val11 = (*(data1_15360+alu49));
  int alu50 = (alu49+5120);
  float val12 = (*(data1_15360+alu50));
  int alu51 = (alu49+10240);
  float val13 = (*(data1_15360+alu51));
  *(data0_15360+alu49) = (val11+(*(buf0+0)));
  *(data0_15360+alu50) = (val12+(*(buf0+1)));
  *(data0_15360+alu51) = (val13+(*(buf0+2)));
}