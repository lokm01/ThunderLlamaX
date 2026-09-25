// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(16) r_24_16_16_3_1024_256_1024_256_256_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_4_4_4(float* data0_18432, float* data1_196608, float* data2_24, float* data3_24, half* data4_16777216, float* data5_196608, float* data6_24, float* data7_24, float* data8_196608, float* data9_24, float* data10_24, float* data11_36864) {
  float buf0[1];
  float buf1[1];
  float buf2[1];
  float buf3[1];
  float buf4[1];
  float buf5[1];
  float buf6[1];
  float buf7[1];
  float buf8[1];
  float buf9[1];
  float buf10[1];
  float buf11[1];
  float buf12[1];
  float buf13[1];
  float buf14[1];
  float buf15[1];
  float buf16[1];
  float buf17[1];
  float buf18[1];
  float buf19[1];
  float buf20[1];
  float buf21[1];
  float buf22[1];
  float buf23[1];
  int gidx1 = blockIdx.y; /* 24 */
  float val0 = (*(data9_24+gidx1));
  int gidx0 = blockIdx.x; /* 16 */
  int lidx0 = threadIdx.x; /* 16 */
  int alu0 = (lidx0+(gidx0<<4));
  int alu1 = (gidx1<<13);
  int alu2 = ((gidx1/6)<<21);
  *(buf7+0) = 0.0f;
  for (int Ridx16 = 0; Ridx16 < 1024; Ridx16++) {
    half val1 = (*(data4_16777216+(alu0+(Ridx16<<8)+alu2+8388608)));
    float val2 = (*(data8_196608+(alu1+Ridx16)));
    *(buf7+0) = ((*(buf7+0))+(exp2(((val2-val0)*1.4426950216293335f))*((float)(val1))));
  }
  *(buf6+0) = 0.0f;
  for (int Ridx17 = 0; Ridx17 < 1024; Ridx17++) {
    half val3 = (*(data4_16777216+(alu0+(Ridx17<<8)+alu2+8650752)));
    float val4 = (*(data8_196608+(alu1+Ridx17+1024)));
    *(buf6+0) = ((*(buf6+0))+(exp2(((val4-val0)*1.4426950216293335f))*((float)(val3))));
  }
  *(buf5+0) = 0.0f;
  for (int Ridx18 = 0; Ridx18 < 1024; Ridx18++) {
    half val5 = (*(data4_16777216+(alu0+(Ridx18<<8)+alu2+8912896)));
    float val6 = (*(data8_196608+(alu1+Ridx18+2048)));
    *(buf5+0) = ((*(buf5+0))+(exp2(((val6-val0)*1.4426950216293335f))*((float)(val5))));
  }
  *(buf4+0) = 0.0f;
  for (int Ridx19 = 0; Ridx19 < 1024; Ridx19++) {
    half val7 = (*(data4_16777216+(alu0+(Ridx19<<8)+alu2+9175040)));
    float val8 = (*(data8_196608+(alu1+Ridx19+3072)));
    *(buf4+0) = ((*(buf4+0))+(exp2(((val8-val0)*1.4426950216293335f))*((float)(val7))));
  }
  *(buf3+0) = 0.0f;
  for (int Ridx20 = 0; Ridx20 < 1024; Ridx20++) {
    half val9 = (*(data4_16777216+(alu0+(Ridx20<<8)+alu2+9437184)));
    float val10 = (*(data8_196608+(alu1+Ridx20+4096)));
    *(buf3+0) = ((*(buf3+0))+(exp2(((val10-val0)*1.4426950216293335f))*((float)(val9))));
  }
  *(buf2+0) = 0.0f;
  for (int Ridx21 = 0; Ridx21 < 1024; Ridx21++) {
    half val11 = (*(data4_16777216+(alu0+(Ridx21<<8)+alu2+9699328)));
    float val12 = (*(data8_196608+(alu1+Ridx21+5120)));
    *(buf2+0) = ((*(buf2+0))+(exp2(((val12-val0)*1.4426950216293335f))*((float)(val11))));
  }
  *(buf1+0) = 0.0f;
  for (int Ridx22 = 0; Ridx22 < 1024; Ridx22++) {
    half val13 = (*(data4_16777216+(alu0+(Ridx22<<8)+alu2+9961472)));
    float val14 = (*(data8_196608+(alu1+Ridx22+6144)));
    *(buf1+0) = ((*(buf1+0))+(exp2(((val14-val0)*1.4426950216293335f))*((float)(val13))));
  }
  *(buf0+0) = 0.0f;
  for (int Ridx23 = 0; Ridx23 < 1024; Ridx23++) {
    half val15 = (*(data4_16777216+(alu0+(Ridx23<<8)+alu2+10223616)));
    float val16 = (*(data8_196608+(alu1+Ridx23+7168)));
    *(buf0+0) = ((*(buf0+0))+(exp2(((val16-val0)*1.4426950216293335f))*((float)(val15))));
  }
  float val17 = (*(data6_24+gidx1));
  *(buf15+0) = 0.0f;
  for (int Ridx8 = 0; Ridx8 < 1024; Ridx8++) {
    half val18 = (*(data4_16777216+(alu0+(Ridx8<<8)+alu2+8388608)));
    float val19 = (*(data5_196608+(alu1+Ridx8)));
    *(buf15+0) = ((*(buf15+0))+(exp2(((val19-val17)*1.4426950216293335f))*((float)(val18))));
  }
  *(buf14+0) = 0.0f;
  for (int Ridx9 = 0; Ridx9 < 1024; Ridx9++) {
    half val20 = (*(data4_16777216+(alu0+(Ridx9<<8)+alu2+8650752)));
    float val21 = (*(data5_196608+(alu1+Ridx9+1024)));
    *(buf14+0) = ((*(buf14+0))+(exp2(((val21-val17)*1.4426950216293335f))*((float)(val20))));
  }
  *(buf13+0) = 0.0f;
  for (int Ridx10 = 0; Ridx10 < 1024; Ridx10++) {
    half val22 = (*(data4_16777216+(alu0+(Ridx10<<8)+alu2+8912896)));
    float val23 = (*(data5_196608+(alu1+Ridx10+2048)));
    *(buf13+0) = ((*(buf13+0))+(exp2(((val23-val17)*1.4426950216293335f))*((float)(val22))));
  }
  *(buf12+0) = 0.0f;
  for (int Ridx11 = 0; Ridx11 < 1024; Ridx11++) {
    half val24 = (*(data4_16777216+(alu0+(Ridx11<<8)+alu2+9175040)));
    float val25 = (*(data5_196608+(alu1+Ridx11+3072)));
    *(buf12+0) = ((*(buf12+0))+(exp2(((val25-val17)*1.4426950216293335f))*((float)(val24))));
  }
  *(buf11+0) = 0.0f;
  for (int Ridx12 = 0; Ridx12 < 1024; Ridx12++) {
    half val26 = (*(data4_16777216+(alu0+(Ridx12<<8)+alu2+9437184)));
    float val27 = (*(data5_196608+(alu1+Ridx12+4096)));
    *(buf11+0) = ((*(buf11+0))+(exp2(((val27-val17)*1.4426950216293335f))*((float)(val26))));
  }
  *(buf10+0) = 0.0f;
  for (int Ridx13 = 0; Ridx13 < 1024; Ridx13++) {
    half val28 = (*(data4_16777216+(alu0+(Ridx13<<8)+alu2+9699328)));
    float val29 = (*(data5_196608+(alu1+Ridx13+5120)));
    *(buf10+0) = ((*(buf10+0))+(exp2(((val29-val17)*1.4426950216293335f))*((float)(val28))));
  }
  *(buf9+0) = 0.0f;
  for (int Ridx14 = 0; Ridx14 < 1024; Ridx14++) {
    half val30 = (*(data4_16777216+(alu0+(Ridx14<<8)+alu2+9961472)));
    float val31 = (*(data5_196608+(alu1+Ridx14+6144)));
    *(buf9+0) = ((*(buf9+0))+(exp2(((val31-val17)*1.4426950216293335f))*((float)(val30))));
  }
  *(buf8+0) = 0.0f;
  for (int Ridx15 = 0; Ridx15 < 1024; Ridx15++) {
    half val32 = (*(data4_16777216+(alu0+(Ridx15<<8)+alu2+10223616)));
    float val33 = (*(data5_196608+(alu1+Ridx15+7168)));
    *(buf8+0) = ((*(buf8+0))+(exp2(((val33-val17)*1.4426950216293335f))*((float)(val32))));
  }
  float val34 = (*(data2_24+gidx1));
  *(buf23+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    half val35 = (*(data4_16777216+(alu0+(Ridx0<<8)+alu2+8388608)));
    float val36 = (*(data1_196608+(alu1+Ridx0)));
    *(buf23+0) = ((*(buf23+0))+(exp2(((val36-val34)*1.4426950216293335f))*((float)(val35))));
  }
  *(buf22+0) = 0.0f;
  for (int Ridx1 = 0; Ridx1 < 256; Ridx1++) {
    int alu55 = (alu0+(Ridx1<<10)+alu2);
    half val37 = (*(data4_16777216+(alu55+8650752)));
    half val38 = (*(data4_16777216+(alu55+8651008)));
    half val39 = (*(data4_16777216+(alu55+8651264)));
    half val40 = (*(data4_16777216+(alu55+8651520)));
    float4 val41 = (*((float4*)((data1_196608+(alu1+(Ridx1<<2)+1024)))));
    *(buf22+0) = ((*(buf22+0))+(exp2(((val41.x-val34)*1.4426950216293335f))*((float)(val37)))+(exp2(((val41.y-val34)*1.4426950216293335f))*((float)(val38)))+(exp2(((val41.z-val34)*1.4426950216293335f))*((float)(val39)))+(exp2(((val41.w-val34)*1.4426950216293335f))*((float)(val40))));
  }
  *(buf21+0) = 0.0f;
  for (int Ridx2 = 0; Ridx2 < 1024; Ridx2++) {
    half val42 = (*(data4_16777216+(alu0+(Ridx2<<8)+alu2+8912896)));
    float val43 = (*(data1_196608+(alu1+Ridx2+2048)));
    *(buf21+0) = ((*(buf21+0))+(exp2(((val43-val34)*1.4426950216293335f))*((float)(val42))));
  }
  *(buf20+0) = 0.0f;
  for (int Ridx3 = 0; Ridx3 < 256; Ridx3++) {
    int alu62 = (alu0+(Ridx3<<10)+alu2);
    half val44 = (*(data4_16777216+(alu62+9175040)));
    half val45 = (*(data4_16777216+(alu62+9175296)));
    half val46 = (*(data4_16777216+(alu62+9175552)));
    half val47 = (*(data4_16777216+(alu62+9175808)));
    float4 val48 = (*((float4*)((data1_196608+(alu1+(Ridx3<<2)+3072)))));
    *(buf20+0) = ((*(buf20+0))+(exp2(((val48.x-val34)*1.4426950216293335f))*((float)(val44)))+(exp2(((val48.y-val34)*1.4426950216293335f))*((float)(val45)))+(exp2(((val48.z-val34)*1.4426950216293335f))*((float)(val46)))+(exp2(((val48.w-val34)*1.4426950216293335f))*((float)(val47))));
  }
  *(buf19+0) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 256; Ridx4++) {
    int alu66 = (alu0+(Ridx4<<10)+alu2);
    half val49 = (*(data4_16777216+(alu66+9437184)));
    half val50 = (*(data4_16777216+(alu66+9437440)));
    half val51 = (*(data4_16777216+(alu66+9437696)));
    half val52 = (*(data4_16777216+(alu66+9437952)));
    float4 val53 = (*((float4*)((data1_196608+(alu1+(Ridx4<<2)+4096)))));
    *(buf19+0) = ((*(buf19+0))+(exp2(((val53.x-val34)*1.4426950216293335f))*((float)(val49)))+(exp2(((val53.y-val34)*1.4426950216293335f))*((float)(val50)))+(exp2(((val53.z-val34)*1.4426950216293335f))*((float)(val51)))+(exp2(((val53.w-val34)*1.4426950216293335f))*((float)(val52))));
  }
  *(buf18+0) = 0.0f;
  for (int Ridx5 = 0; Ridx5 < 1024; Ridx5++) {
    half val54 = (*(data4_16777216+(alu0+(Ridx5<<8)+alu2+9699328)));
    float val55 = (*(data1_196608+(alu1+Ridx5+5120)));
    *(buf18+0) = ((*(buf18+0))+(exp2(((val55-val34)*1.4426950216293335f))*((float)(val54))));
  }
  *(buf17+0) = 0.0f;
  for (int Ridx6 = 0; Ridx6 < 1024; Ridx6++) {
    half val56 = (*(data4_16777216+(alu0+(Ridx6<<8)+alu2+9961472)));
    float val57 = (*(data1_196608+(alu1+Ridx6+6144)));
    *(buf17+0) = ((*(buf17+0))+(exp2(((val57-val34)*1.4426950216293335f))*((float)(val56))));
  }
  *(buf16+0) = 0.0f;
  for (int Ridx7 = 0; Ridx7 < 1024; Ridx7++) {
    half val58 = (*(data4_16777216+(alu0+(Ridx7<<8)+alu2+10223616)));
    float val59 = (*(data1_196608+(alu1+Ridx7+7168)));
    *(buf16+0) = ((*(buf16+0))+(exp2(((val59-val34)*1.4426950216293335f))*((float)(val58))));
  }
  float val60 = (*(data3_24+gidx1));
  float val61 = (*(data7_24+gidx1));
  float val62 = (*(data10_24+gidx1));
  int alu78 = (alu0+(gidx1<<9));
  float val63 = (*(data11_36864+(alu78+256)));
  float val64 = (*(data11_36864+(alu78+12544)));
  float val65 = (*(data11_36864+(alu78+24832)));
  int alu79 = (alu0+(gidx1<<8));
  float alu80 = (1/val60);
  float alu81 = (1/val61);
  float alu82 = (1/val62);
  *(data0_18432+alu79) = ((((*(buf7+0))*alu82)+((*(buf6+0))*alu82)+((*(buf5+0))*alu82)+((*(buf4+0))*alu82)+((*(buf3+0))*alu82)+((*(buf2+0))*alu82)+((*(buf1+0))*alu82)+((*(buf0+0))*alu82))*(1/(1.0f+exp2((val63*-1.4426950216293335f)))));
  *(data0_18432+(alu79+6144)) = ((((*(buf15+0))*alu81)+((*(buf14+0))*alu81)+((*(buf13+0))*alu81)+((*(buf12+0))*alu81)+((*(buf11+0))*alu81)+((*(buf10+0))*alu81)+((*(buf9+0))*alu81)+((*(buf8+0))*alu81))*(1/(1.0f+exp2((val64*-1.4426950216293335f)))));
  *(data0_18432+(alu79+12288)) = ((((*(buf23+0))*alu80)+((*(buf22+0))*alu80)+((*(buf21+0))*alu80)+((*(buf20+0))*alu80)+((*(buf19+0))*alu80)+((*(buf18+0))*alu80)+((*(buf17+0))*alu80)+((*(buf16+0))*alu80))*(1/(1.0f+exp2((val65*-1.4426950216293335f)))));
}