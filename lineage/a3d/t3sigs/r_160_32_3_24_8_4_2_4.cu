// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
extern "C" __global__ void __launch_bounds__(32) r_160_32_3_24_8_4_2_4(float* data0_15360, float* data1_15360, half* data2_18432, unsigned char* data3_12042240, float* data4_1024) {
  float buf0[3];
  int gidx0 = blockIdx.x; /* 160 */
  int lidx0 = threadIdx.x; /* 32 */
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  for (int Ridx1_0_0 = 0; Ridx1_0_0 < 24; Ridx1_0_0++) {
    int alu3 = ((gidx0*75264)+(lidx0*2352)+(Ridx1_0_0*98));
    unsigned char val0 = (*(data3_12042240+(alu3+1)));
    unsigned char val1 = (*(data3_12042240+alu3));
    for (int Ridx1_0_1_0 = 0; Ridx1_0_1_0 < 8; Ridx1_0_1_0++) {
      int alu4 = (alu3+(Ridx1_0_1_0<<2));
      unsigned char val2 = (*(data3_12042240+(alu4+66)));
      unsigned char val3 = (*(data3_12042240+(alu4+67)));
      unsigned char val4 = (*(data3_12042240+(alu4+68)));
      unsigned char val5 = (*(data3_12042240+(alu4+69)));
      uint alu5 = ((((uint)(val2))<<0u)+(((uint)(val3))<<8u)+(((uint)(val4))<<16u)+(((uint)(val5))<<24u));
      float alu6 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val1))<<((unsigned short)(0u)))+(((unsigned short)(val0))<<((unsigned short)(8u))))))))*(((float)((alu5>>28u)))+0.5f));
      for (int Ridx1_0_1_1 = 0; Ridx1_0_1_1 < 4; Ridx1_0_1_1++) {
        uint alu7 = ((Ridx1_0_1_1!=2)?21u:14u);
        uint alu8 = ((Ridx1_0_1_1!=1)?alu7:7u);
        uint alu9 = ((Ridx1_0_1_1!=0)?alu8:0u);
        int cast0 = ((int)(((alu5>>alu9)&127u)));
        unsigned char cast1 = ((unsigned char)((cast0|((((cast0>>0)^(cast0>>1)^(cast0>>2)^(cast0>>3)^(cast0>>4)^(cast0>>5)^(cast0>>6))&1)<<7))));
        for (int Ridx1_1_0 = 0; Ridx1_1_0 < 2; Ridx1_1_0++) {
          unsigned char val6 = (*(data3_12042240+(alu3+(Ridx1_0_1_0<<3)+(Ridx1_0_1_1<<1)+Ridx1_1_0+2)));
          int alu10 = (Ridx1_1_0<<2);
          int alu11 = ((Ridx1_0_1_0<<5)+(Ridx1_0_1_1<<3)+(Ridx1_0_0<<8)+alu10);
          half4 val7 = (*((half4*)((data2_18432+(alu11+6144)))));
          half4 val8 = (*((half4*)((data2_18432+(alu11+12288)))));
          half4 val9 = (*((half4*)((data2_18432+alu11))));
          float4 val10 = (*((float4*)((data4_1024+(((int)(val6))<<2)))));
          unsigned char alu12 = ((alu10!=4)?((unsigned char)(128u)):((unsigned char)(16u)));
          unsigned char alu13 = ((alu10!=3)?alu12:((unsigned char)(8u)));
          unsigned char alu14 = ((alu10!=2)?alu13:((unsigned char)(4u)));
          unsigned char alu15 = ((alu10!=1)?alu14:((unsigned char)(2u)));
          unsigned char alu16 = ((alu10!=0)?alu15:((unsigned char)(1u)));
          float alu17 = ((((cast1/alu16)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          half cast2 = ((half)((alu6*val10.x*alu17*0.5f)));
          int alu18 = (alu10+1);
          unsigned char alu19 = ((alu18!=5)?((unsigned char)(128u)):((unsigned char)(32u)));
          unsigned char alu20 = ((alu18!=4)?alu19:((unsigned char)(16u)));
          unsigned char alu21 = ((alu18!=3)?alu20:((unsigned char)(8u)));
          unsigned char alu22 = ((alu18!=2)?alu21:((unsigned char)(4u)));
          unsigned char alu23 = ((alu18!=1)?alu22:((unsigned char)(2u)));
          float alu24 = ((((cast1/alu23)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          half cast3 = ((half)((alu6*val10.y*alu24*0.5f)));
          int alu25 = (alu10+2);
          unsigned char alu26 = ((alu25!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu27 = ((alu25!=5)?alu26:((unsigned char)(32u)));
          unsigned char alu28 = ((alu25!=4)?alu27:((unsigned char)(16u)));
          unsigned char alu29 = ((alu25!=3)?alu28:((unsigned char)(8u)));
          unsigned char alu30 = ((alu25!=2)?alu29:((unsigned char)(4u)));
          float alu31 = ((((cast1/alu30)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          half cast4 = ((half)((alu6*val10.z*alu31*0.5f)));
          int alu32 = (alu10+3);
          unsigned char alu33 = ((alu32!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu34 = ((alu32!=5)?alu33:((unsigned char)(32u)));
          unsigned char alu35 = ((alu32!=4)?alu34:((unsigned char)(16u)));
          unsigned char alu36 = ((alu32!=3)?alu35:((unsigned char)(8u)));
          float alu37 = ((((cast1/alu36)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          half cast5 = ((half)((alu6*val10.w*alu37*0.5f)));
          *(buf0+0) = ((*(buf0+0))+((float)((val9.x*cast2)))+((float)((val9.y*cast3)))+((float)((val9.z*cast4)))+((float)((val9.w*cast5))));
          *(buf0+1) = ((*(buf0+1))+((float)((val7.x*cast2)))+((float)((val7.y*cast3)))+((float)((val7.z*cast4)))+((float)((val7.w*cast5))));
          *(buf0+2) = ((*(buf0+2))+((float)((val8.x*cast2)))+((float)((val8.y*cast3)))+((float)((val8.z*cast4)))+((float)((val8.w*cast5))));
        }
      }
    }
  }
  int alu45 = (lidx0+(gidx0<<5));
  float val11 = (*(data1_15360+alu45));
  int alu46 = (alu45+5120);
  float val12 = (*(data1_15360+alu46));
  int alu47 = (alu45+10240);
  float val13 = (*(data1_15360+alu47));
  *(data0_15360+alu45) = (val11+(*(buf0+0)));
  *(data0_15360+alu46) = (val12+(*(buf0+1)));
  *(data0_15360+alu47) = (val13+(*(buf0+2)));
}