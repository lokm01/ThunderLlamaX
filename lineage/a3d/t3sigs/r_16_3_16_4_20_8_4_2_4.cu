// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(48) r_16_3_16_4_20_8_4_2_4(float* data0_3072, float* data1_15360, float* data2_3, unsigned char* data3_20480, unsigned char* data4_2007040, float* data5_1024) {
  float buf0[4];
  int gidx0 = blockIdx.x; /* 16 */
  int lidx0 = threadIdx.x; /* 3 */
  int lidx1 = threadIdx.y; /* 16 */
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  for (int Ridx1_0_0 = 0; Ridx1_0_0 < 20; Ridx1_0_0++) {
    int alu4 = ((gidx0*31360)+(lidx1*1960)+(Ridx1_0_0*98));
    unsigned char val0 = (*(data4_2007040+(alu4+1)));
    unsigned char val1 = (*(data4_2007040+(alu4+501760)));
    unsigned char val2 = (*(data4_2007040+(alu4+501761)));
    unsigned char val3 = (*(data4_2007040+(alu4+1003520)));
    unsigned char val4 = (*(data4_2007040+(alu4+1003521)));
    unsigned char val5 = (*(data4_2007040+(alu4+1505280)));
    unsigned char val6 = (*(data4_2007040+(alu4+1505281)));
    unsigned char val7 = (*(data4_2007040+alu4));
    for (int Ridx1_0_1_0 = 0; Ridx1_0_1_0 < 8; Ridx1_0_1_0++) {
      int alu5 = (alu4+(Ridx1_0_1_0<<2));
      unsigned char val8 = (*(data4_2007040+(alu5+66)));
      unsigned char val9 = (*(data4_2007040+(alu5+67)));
      unsigned char val10 = (*(data4_2007040+(alu5+68)));
      unsigned char val11 = (*(data4_2007040+(alu5+69)));
      unsigned char val12 = (*(data4_2007040+(alu5+501826)));
      unsigned char val13 = (*(data4_2007040+(alu5+501827)));
      unsigned char val14 = (*(data4_2007040+(alu5+501828)));
      unsigned char val15 = (*(data4_2007040+(alu5+501829)));
      unsigned char val16 = (*(data4_2007040+(alu5+1003586)));
      unsigned char val17 = (*(data4_2007040+(alu5+1003587)));
      unsigned char val18 = (*(data4_2007040+(alu5+1003588)));
      unsigned char val19 = (*(data4_2007040+(alu5+1003589)));
      unsigned char val20 = (*(data4_2007040+(alu5+1505346)));
      unsigned char val21 = (*(data4_2007040+(alu5+1505347)));
      unsigned char val22 = (*(data4_2007040+(alu5+1505348)));
      unsigned char val23 = (*(data4_2007040+(alu5+1505349)));
      uint alu6 = ((((uint)(val8))<<0u)+(((uint)(val9))<<8u)+(((uint)(val10))<<16u)+(((uint)(val11))<<24u));
      uint alu7 = ((((uint)(val12))<<0u)+(((uint)(val13))<<8u)+(((uint)(val14))<<16u)+(((uint)(val15))<<24u));
      uint alu8 = ((((uint)(val16))<<0u)+(((uint)(val17))<<8u)+(((uint)(val18))<<16u)+(((uint)(val19))<<24u));
      uint alu9 = ((((uint)(val20))<<0u)+(((uint)(val21))<<8u)+(((uint)(val22))<<16u)+(((uint)(val23))<<24u));
      float alu10 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val1))<<((unsigned short)(0u)))+(((unsigned short)(val2))<<((unsigned short)(8u))))))))*(((float)((alu7>>28u)))+0.5f));
      float alu11 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val3))<<((unsigned short)(0u)))+(((unsigned short)(val4))<<((unsigned short)(8u))))))))*(((float)((alu8>>28u)))+0.5f));
      float alu12 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val5))<<((unsigned short)(0u)))+(((unsigned short)(val6))<<((unsigned short)(8u))))))))*(((float)((alu9>>28u)))+0.5f));
      float alu13 = (((float)(tg_bitcast<half>((unsigned short)(((((unsigned short)(val7))<<((unsigned short)(0u)))+(((unsigned short)(val0))<<((unsigned short)(8u))))))))*(((float)((alu6>>28u)))+0.5f));
      for (int Ridx1_0_1_1 = 0; Ridx1_0_1_1 < 4; Ridx1_0_1_1++) {
        uint alu14 = ((Ridx1_0_1_1!=2)?21u:14u);
        uint alu15 = ((Ridx1_0_1_1!=1)?alu14:7u);
        uint alu16 = ((Ridx1_0_1_1!=0)?alu15:0u);
        int cast0 = ((int)(((alu6>>alu16)&127u)));
        unsigned char cast1 = ((unsigned char)((cast0|((((cast0>>0)^(cast0>>1)^(cast0>>2)^(cast0>>3)^(cast0>>4)^(cast0>>5)^(cast0>>6))&1)<<7))));
        int cast2 = ((int)(((alu7>>alu16)&127u)));
        unsigned char cast3 = ((unsigned char)((cast2|((((cast2>>0)^(cast2>>1)^(cast2>>2)^(cast2>>3)^(cast2>>4)^(cast2>>5)^(cast2>>6))&1)<<7))));
        int cast4 = ((int)(((alu8>>alu16)&127u)));
        unsigned char cast5 = ((unsigned char)((cast4|((((cast4>>0)^(cast4>>1)^(cast4>>2)^(cast4>>3)^(cast4>>4)^(cast4>>5)^(cast4>>6))&1)<<7))));
        int cast6 = ((int)(((alu9>>alu16)&127u)));
        unsigned char cast7 = ((unsigned char)((cast6|((((cast6>>0)^(cast6>>1)^(cast6>>2)^(cast6>>3)^(cast6>>4)^(cast6>>5)^(cast6>>6))&1)<<7))));
        for (int Ridx1_1_0 = 0; Ridx1_1_0 < 2; Ridx1_1_0++) {
          int alu17 = ((Ridx1_0_1_0<<7)+(Ridx1_0_1_1<<5)+(Ridx1_0_0<<10)+(Ridx1_1_0<<4));
          unsigned char val24 = (*(data3_20480+(alu17+1)));
          unsigned char val25 = (*(data3_20480+(alu17+2)));
          unsigned char val26 = (*(data3_20480+(alu17+3)));
          unsigned char val27 = (*(data3_20480+(alu17+4)));
          unsigned char val28 = (*(data3_20480+(alu17+5)));
          unsigned char val29 = (*(data3_20480+(alu17+6)));
          unsigned char val30 = (*(data3_20480+(alu17+7)));
          unsigned char val31 = (*(data3_20480+(alu17+8)));
          unsigned char val32 = (*(data3_20480+(alu17+9)));
          unsigned char val33 = (*(data3_20480+(alu17+10)));
          unsigned char val34 = (*(data3_20480+(alu17+11)));
          unsigned char val35 = (*(data3_20480+(alu17+12)));
          unsigned char val36 = (*(data3_20480+(alu17+13)));
          unsigned char val37 = (*(data3_20480+(alu17+14)));
          unsigned char val38 = (*(data3_20480+(alu17+15)));
          unsigned char val39 = (*(data3_20480+alu17));
          int alu18 = (alu4+(Ridx1_0_1_0<<3)+(Ridx1_0_1_1<<1)+Ridx1_1_0);
          unsigned char val40 = (*(data4_2007040+(alu18+2)));
          unsigned char val41 = (*(data4_2007040+(alu18+501762)));
          unsigned char val42 = (*(data4_2007040+(alu18+1003522)));
          unsigned char val43 = (*(data4_2007040+(alu18+1505282)));
          int alu19 = (Ridx1_1_0<<2);
          float4 val44 = (*((float4*)((data1_15360+((Ridx1_0_1_0<<5)+(Ridx1_0_1_1<<3)+(Ridx1_0_0<<8)+alu19+(lidx0*5120))))));
          float4 val45 = (*((float4*)((data5_1024+(((int)(val40))<<2)))));
          float4 val46 = (*((float4*)((data5_1024+(((int)(val41))<<2)))));
          float4 val47 = (*((float4*)((data5_1024+(((int)(val42))<<2)))));
          float4 val48 = (*((float4*)((data5_1024+(((int)(val43))<<2)))));
          int alu20 = (alu19+1);
          int alu21 = (alu19+2);
          int alu22 = (alu19+3);
          float alu23 = (val44.x*tg_bitcast<float>((uint)(((((uint)(val39))<<0u)+(((uint)(val24))<<8u)+(((uint)(val25))<<16u)+(((uint)(val26))<<24u)))));
          float alu24 = (val44.y*tg_bitcast<float>((uint)(((((uint)(val27))<<0u)+(((uint)(val28))<<8u)+(((uint)(val29))<<16u)+(((uint)(val30))<<24u)))));
          unsigned char alu25 = ((alu19!=4)?((unsigned char)(128u)):((unsigned char)(16u)));
          unsigned char alu26 = ((alu19!=3)?alu25:((unsigned char)(8u)));
          unsigned char alu27 = ((alu19!=2)?alu26:((unsigned char)(4u)));
          unsigned char alu28 = ((alu19!=1)?alu27:((unsigned char)(2u)));
          unsigned char alu29 = ((alu19!=0)?alu28:((unsigned char)(1u)));
          float alu30 = ((((cast1/alu29)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          unsigned char alu31 = ((alu20!=5)?((unsigned char)(128u)):((unsigned char)(32u)));
          unsigned char alu32 = ((alu20!=4)?alu31:((unsigned char)(16u)));
          unsigned char alu33 = ((alu20!=3)?alu32:((unsigned char)(8u)));
          unsigned char alu34 = ((alu20!=2)?alu33:((unsigned char)(4u)));
          unsigned char alu35 = ((alu20!=1)?alu34:((unsigned char)(2u)));
          float alu36 = ((((cast1/alu35)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu37 = (val44.z*tg_bitcast<float>((uint)(((((uint)(val31))<<0u)+(((uint)(val32))<<8u)+(((uint)(val33))<<16u)+(((uint)(val34))<<24u)))));
          unsigned char alu38 = ((alu21!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu39 = ((alu21!=5)?alu38:((unsigned char)(32u)));
          unsigned char alu40 = ((alu21!=4)?alu39:((unsigned char)(16u)));
          unsigned char alu41 = ((alu21!=3)?alu40:((unsigned char)(8u)));
          unsigned char alu42 = ((alu21!=2)?alu41:((unsigned char)(4u)));
          float alu43 = ((((cast1/alu42)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu44 = (val44.w*tg_bitcast<float>((uint)(((((uint)(val35))<<0u)+(((uint)(val36))<<8u)+(((uint)(val37))<<16u)+(((uint)(val38))<<24u)))));
          unsigned char alu45 = ((alu22!=6)?((unsigned char)(128u)):((unsigned char)(64u)));
          unsigned char alu46 = ((alu22!=5)?alu45:((unsigned char)(32u)));
          unsigned char alu47 = ((alu22!=4)?alu46:((unsigned char)(16u)));
          unsigned char alu48 = ((alu22!=3)?alu47:((unsigned char)(8u)));
          float alu49 = ((((cast1/alu48)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu50 = ((((cast3/alu29)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu51 = ((((cast3/alu35)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu52 = ((((cast3/alu42)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu53 = ((((cast3/alu48)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu54 = ((((cast5/alu29)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu55 = ((((cast5/alu35)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu56 = ((((cast5/alu42)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu57 = ((((cast5/alu48)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu58 = ((((cast7/alu29)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu59 = ((((cast7/alu35)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu60 = ((((cast7/alu42)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          float alu61 = ((((cast7/alu48)&((unsigned char)(1u)))!=((unsigned char)(0u)))?-1.0f:1.0f);
          *(buf0+0) = ((*(buf0+0))+(alu13*val45.x*alu30*alu23*0.5f)+(alu13*val45.y*alu36*alu24*0.5f)+(alu13*val45.z*alu43*alu37*0.5f)+(alu13*val45.w*alu49*alu44*0.5f));
          *(buf0+1) = ((*(buf0+1))+(alu10*val46.x*alu50*alu23*0.5f)+(alu10*val46.y*alu51*alu24*0.5f)+(alu10*val46.z*alu52*alu37*0.5f)+(alu10*val46.w*alu53*alu44*0.5f));
          *(buf0+2) = ((*(buf0+2))+(alu11*val47.x*alu54*alu23*0.5f)+(alu11*val47.y*alu55*alu24*0.5f)+(alu11*val47.z*alu56*alu37*0.5f)+(alu11*val47.w*alu57*alu44*0.5f));
          *(buf0+3) = ((*(buf0+3))+(alu12*val48.x*alu58*alu23*0.5f)+(alu12*val48.y*alu59*alu24*0.5f)+(alu12*val48.z*alu60*alu37*0.5f)+(alu12*val48.w*alu61*alu44*0.5f));
        }
      }
    }
  }
  float val49 = (*(data2_3+lidx0));
  int alu70 = (lidx1+(gidx0<<4)+(lidx0<<8));
  float alu71 = (1/val49);
  *(data0_3072+alu70) = ((*(buf0+0))*alu71);
  *(data0_3072+(alu70+768)) = ((*(buf0+1))*alu71);
  *(data0_3072+(alu70+1536)) = ((*(buf0+2))*alu71);
  *(data0_3072+(alu70+2304)) = ((*(buf0+3))*alu71);
}