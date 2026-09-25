// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-C pf_scanchunk.cu — the CHUNKED DELTA-RULE SCAN (FLA WY representation)
// replacing the sequential per-token scan (pf_scan16) at super-chunk scale.
//
// MATH CONTRACT (oracle: pf_scan16.cu VERBATIM scalar formulas; the ONLY
// numerics change = the WY reassociation with log2-space chunk cumsum):
//   per token j (recurrence, from pfs16's exact op order):
//     S_j = A_j*S_{j-1} + khat_j * [ beta_j*v_j - beta_j*(A_j*S_{j-1})^T khat_j ]^T
//   per chunk (g_i = cumsum(lambda_i*log2e), lambda = softplus(a+dtb)*ssma):
//     B[i,m] = beta_i*(khat_i.khat_m)*2^(g_i-g_m)   (i>m, strictly lower)
//     T = (I+B)^-1            (blocked fwd substitution, 2x 32x32 tiles @C=64)
//     U = T(beta.V);  Y = Khat S_-1;  d = U - T(beta*2^g . Y)
//     O = M d + Qe S_-1       (M[i,m] = (qhat_i.khat_m)*2^(g_i-g_m), m<=i;
//                              Qe = qhat .* 2^g_i rows)
//     S' = 2^g_end * S_-1 + KhatT (2^(g_end-g) . d)
//   preprocess VERBATIM: 4-tap causal conv (convlive fallback rows <0),
//   silu on q/k/v, qn = (1/max(sqrt(qss),EPS_Q))*ISQ128, kn = 1/max(...),
//   beta = sigmoid(b), z epilogue incl. the pfs16 half-gate expression.
//
// KERNELS (KSEL): 0 = pfca (K_A, intra-chunk factors, parallel over
// (head,chunk)), 1 = pfcb (K_BC, sequential state pass, grid head x 4
// v-quarts, state slice [128][32] fp32 in smem), 2 = pfcz (z epilogue +
// conv_live writeback). Per launch = ONE GDN block (weights/state/scratch
// via per-block pointer offsets from python).
//
// LAWS: flat indexing, no gridDim/blockDim reads, hardcoded sizes,
// sequential loops, full-warp masks, single 16B-aligned smem array, LDS.32
// fragments (no ldmatrix), per-kernel cubins + warp-token names.
// smem: pfca C=64 41088B (EAGER-ONLY; solve phase) / C=32 19072B (in-graph
// legal); pfcb 34336B (C=64) in-graph legal; pfcz 0.
// Build: -DKSEL -DKNAME -DC (32|64) -DNC -DNTHR
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define QDIM 2048
#define CONV_CH 10240
#define NVH 48
#define VDIM 128
#define EPS_Q 1e-6f
#define EPS_N 1e-6f
#define ISQ128 0.08838834764831845f
#define LDK 136                 // fp16 row pad for [C][128] / [128][k] tiles
#define LDTC (C + 8)            // fp16 row pad for [C][C] / [128][C] tiles
#define NW (NTHR / 32)
// weight plane per block (floats): convw 10240 | dtb 48 | ssma 48 | snw 6144
#define WP_CW 0
#define WP_DTB 40960
#define WP_SSMA 41008
#define WP_SNW 41056
#define WP_STRIDE 47200
// scratch per (head, chunk), in halfs:
#define KH_C (C * LDK)          // khat  [C][128]
#define KHT_C (128 * LDTC)      // khatT [128][C]
#define QE_C (C * LDK)          // qe    [C][128]
#define U_C (C * LDK)           // u     [C][128]  (hi half of U)
#define VR_C (C * LDK)          // vraw  [C][128] (hi half of v)
#define M_C (C * LDTC)          // m     [C][C]
#define T_C (C * LDTC)          // t     [C][C]
#define ULO_C (C * LDK)         // u_lo  [C][128]  (lo half of U)
#define VRLO_C (C * LDK)        // vr_lo [C][128] (lo half of v)
#define MLO_C (C * LDTC)        // m_lo  [C][C]   (lo half of M)
#define TLO_C (C * LDTC)        // t_lo  [C][C]   (lo half of T)
#define DZ_C (256 * C)          // pfcb d zone: 4 vq x [32][2C] (d hi|lo then sg.d lo)
// P7E5 HI-LO LAW: mature (snapshot-scale) magnitudes — the incoming state S,
// v, U=T(beta.V), Y=Khat.S, d (d ~ U when the state is young and v mature)
// AND the products T@(beta.V), T@(beta.2^g.Y), M@d (T/M's own 5e-4 fp16
// error multiplies the LARGE v/d: the C0/C3 residual after the S/v/U/Y/d
// splits) — must NOT ride bare fp16 (amplified ~20x through the 48-block
// stack by (v-kd)/(U - T(beta 2^g Y)) cancellation; the P7E4 nonzero-seed
// gate failure). Every such carrier is split hi+lo fp16 (>=21-bit effective)
// with 3-pass MMAs (hi*hi + hi*lo + lo*hi; lo*lo ~2^-42 skipped); smem is
// reused sequentially (no pfcb growth; d halves live in a per-(h,c,vq)
// GLOBAL zone so they survive the Qe-phase s16 rebuilds).
// P7E6 HI-LO LAW EXTENSION: the NORMALIZED tiles (khat, qe, KhatT, qh) do
// NOT stay single fp16 either — the fp16 quantization of kh (5e-4/elt)
// perturbs the WHOLE WY system consistently: SC solves the fp16-kh model
// exactly (1.9e-5 vs a sequential oracle on the SAME fp16 kh) but that model
// deviates from the true fp32-kh recurrence by ~2.3e-3/block at |S|~30 (the
// P7E5 conv-row-2 gate channel: mature row-2 content at token 2 pushes the
// o-cancellation noise across fp16-ULP boundaries in z -> bit-flips -> 20x
// trunk amplification). kh/qh/qe/KhatT now split hi+lo too, appended after
// meta (LO region), phase-1 rebuilt as 3 passes with smem reloads.
#define HC_HALF (4 * C * LDK + 128 * LDTC + 4 * C * LDTC + 2 * C * LDK + DZ_C)
// meta fp32 tail: bg[C] | sg[C] | gend | pad  -> (2C+4) floats at HC_HALF*2 bytes
#define META_OFF ((HC_HALF * 2 + 15) / 16 * 16)   // byte offset, 16B aligned
#define KHLO_C (C * LDK)        // kh_lo  [C][128]
#define QHLO_C (C * LDK)        // qh_lo  [C][128] (pfca phase-1 pass reload)
#define QELO_C (C * LDK)        // qe_lo  [C][128]
#define KHTLO_C (128 * LDTC)    // kht_lo [128][C]
#define LO_OFF (((META_OFF + (2 * C + 4) * 4 + 15) / 16 * 16))
#define HC_BYTES ((LO_OFF + (KHLO_C + QHLO_C + QELO_C + KHTLO_C) * 2 + 15) / 16 * 16)

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }

__device__ __forceinline__ void hmma16816(float &c0, float &c1, float &c2, float &c3,
                                          const unsigned a0, const unsigned a1,
                                          const unsigned a2, const unsigned a3,
                                          const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ============================== KSEL 0: pfca ==============================
#if KSEL == 0
#if C != 64 && C != 32
#error "C must be 32 or 64"
#endif
#if C == 64
#define TF_LD 68
#else
#define TF_LD 36
#endif
// gzone: lam[C] | g2[C] | bet[C] | bge[C] = 4C floats = 4096B at C=64
// (LAW: 4C*4 bytes — a 1664B zone clobbered bet/bge via the Tf overlap)
#define GZ_BYTES 4096
#define SM_BYTES (C == 64 ? 43520 : 21504)
#if C == 64
#define NTOT1 ((C / 16) * (C / 8))   // 32
#define NTPW1 2
#define MMAW1 16
#else
#define NTOT1 ((C / 16) * (C / 8))   // 8
#define NTPW1 1
#define MMAW1 8
#endif
#define NTOT2 ((C / 16) * 8)
#define NTPW2 (C / 32)
#define MMAW2 16

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const float* __restrict__ wplane, const float* __restrict__ convlive,
    const __half* __restrict__ kvbuf, const float* __restrict__ araw,
    const float* __restrict__ braw, __half* __restrict__ scr)
{
  __shared__ __align__(16) unsigned char sm[SM_BYTES];
  const int h = blockIdx.x / NC;
  const int c = blockIdx.x - h * NC;
  const int kh = h % 16;
  const int qc0 = kh * 128, kc0 = QDIM + kh * 128, vc0 = 2 * QDIM + h * 128;
  const int c0 = c * C;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* convw = wplane + WP_CW;
  const float* dtb = wplane + WP_DTB;
  const float* ssma = wplane + WP_SSMA;
  // gz fp32 zone: lam[C] | g2[C] | bet[C] | bge[C] | pad
  float* lamf = (float*)(sm + 0);
  float* g2f = (float*)(sm + 4 * C);
  float* betf = (float*)(sm + 8 * C);
  float* bgef = (float*)(sm + 12 * C);
  __half* kh_ = (__half*)(sm + GZ_BYTES);
  __half* qh_ = (__half*)(sm + GZ_BYTES + C * LDK * 2);
  // global scratch regions for this (h, c)
  __half* hc = scr + ((size_t)h * NC + c) * (HC_BYTES / 2);
  __half* kh_g = hc;
  __half* kht_g = hc + KH_C;
  __half* qe_g = hc + KH_C + KHT_C;
  __half* u_g = hc + KH_C + KHT_C + QE_C;
  __half* vr_g = hc + KH_C + KHT_C + QE_C + U_C;
  __half* m_g = hc + KH_C + KHT_C + QE_C + U_C + VR_C;
  __half* t_g = m_g + M_C;
  __half* u_lo_g = t_g + T_C;        // P7E5 lo halves (layout: ... m|t|u_lo|vr_lo|m_lo|t_lo|dz|meta)
  __half* vr_lo_g = u_lo_g + ULO_C;
  __half* m_lo_g = vr_lo_g + VRLO_C;
  __half* t_lo_g = m_lo_g + MLO_C;
  float* met_g = (float*)((unsigned char*)hc + META_OFF);
  __half* kh_lo_g = (__half*)((unsigned char*)hc + LO_OFF);
  __half* qh_lo_g = kh_lo_g + KHLO_C;
  __half* qe_lo_g = qh_lo_g + QHLO_C;
  __half* kht_lo_g = qe_lo_g + QELO_C;

  // ---- phase 0: preprocess (conv + silu + norms + beta/lambda) VERBATIM ----
  #pragma unroll 1
  for (int pp = 0; pp < C / 16; ++pp) {
    const int i = pp * 16 + warp;
    const float lami = softplus_f(araw[(size_t)(c0 + i) * 48 + h] + dtb[h]) * ssma[h];
    const float bei = sig_f(braw[(size_t)(c0 + i) * 48 + h]);
    if (lane == 0) { lamf[i] = lami; betf[i] = bei; }
    const __half* rt = kvbuf + (size_t)(c0 + i) * CONV_CH;
    float qr[4], kr[4], qss = 0.f, kss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int qq = qc0 + lane * 4 + j, kk = kc0 + lane * 4 + j, vv = vc0 + lane * 4 + j;
      const float q2 = __half2float(rt[qq]), k2 = __half2float(rt[kk]), v2 = __half2float(rt[vv]);
      const int b3 = (i >= 3 || c > 0), b2 = (i >= 2 || c > 0), b1 = (i >= 1 || c > 0);
      const float w0q = b3 ? __half2float(kvbuf[(size_t)(c0 + i - 3) * CONV_CH + qq]) : convlive[(size_t)(i + 0) * CONV_CH + qq];
      const float w1q = b2 ? __half2float(kvbuf[(size_t)(c0 + i - 2) * CONV_CH + qq]) : convlive[(size_t)(i + 1) * CONV_CH + qq];
      const float w2q = b1 ? __half2float(kvbuf[(size_t)(c0 + i - 1) * CONV_CH + qq]) : convlive[(size_t)(i + 2) * CONV_CH + qq];
      const float w0k = b3 ? __half2float(kvbuf[(size_t)(c0 + i - 3) * CONV_CH + kk]) : convlive[(size_t)(i + 0) * CONV_CH + kk];
      const float w1k = b2 ? __half2float(kvbuf[(size_t)(c0 + i - 2) * CONV_CH + kk]) : convlive[(size_t)(i + 1) * CONV_CH + kk];
      const float w2k = b1 ? __half2float(kvbuf[(size_t)(c0 + i - 1) * CONV_CH + kk]) : convlive[(size_t)(i + 2) * CONV_CH + kk];
      const float w0v = b3 ? __half2float(kvbuf[(size_t)(c0 + i - 3) * CONV_CH + vv]) : convlive[(size_t)(i + 0) * CONV_CH + vv];
      const float w1v = b2 ? __half2float(kvbuf[(size_t)(c0 + i - 2) * CONV_CH + vv]) : convlive[(size_t)(i + 1) * CONV_CH + vv];
      const float w2v = b1 ? __half2float(kvbuf[(size_t)(c0 + i - 1) * CONV_CH + vv]) : convlive[(size_t)(i + 2) * CONV_CH + vv];
      float sq = w0q*convw[qq*4+0] + w1q*convw[qq*4+1] + w2q*convw[qq*4+2] + q2*convw[qq*4+3];
      float sk = w0k*convw[kk*4+0] + w1k*convw[kk*4+1] + w2k*convw[kk*4+2] + k2*convw[kk*4+3];
      float sv = w0v*convw[vv*4+0] + w1v*convw[vv*4+1] + w2v*convw[vv*4+2] + v2*convw[vv*4+3];
      sq *= sig_f(sq); sk *= sig_f(sk); sv *= sig_f(sv);
      qr[j] = sq; kr[j] = sk;
      { const __half vh2 = __float2half(sv);          // P7E5: v hi+lo (mature |v| ~40)
        vr_g[(size_t)i * LDK + lane * 4 + j] = vh2;
        vr_lo_g[(size_t)i * LDK + lane * 4 + j] = __float2half(sv - __half2float(vh2)); }
      qss += sq*sq; kss += sk*sk;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { qss += __shfl_xor_sync(FULL, qss, o); kss += __shfl_xor_sync(FULL, kss, o); }
    const float qn = (1.0f / fmaxf(sqrtf(qss), EPS_Q)) * ISQ128;
    const float kn = 1.0f / fmaxf(sqrtf(kss), EPS_Q);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __half khh = __float2half(kr[j] * kn);   // P7E6: kh/qh hi+lo
      const __half qhh = __float2half(qr[j] * qn);
      kh_[(size_t)i * LDK + lane * 4 + j] = khh;
      qh_[(size_t)i * LDK + lane * 4 + j] = qhh;
      kh_lo_g[(size_t)i * LDK + lane * 4 + j] = __float2half(kr[j] * kn - __half2float(khh));
      qh_lo_g[(size_t)i * LDK + lane * 4 + j] = __float2half(qr[j] * qn - __half2float(qhh));
    }
  }
  __syncthreads();
  if (tid == 0) { float acc = 0.f; for (int i2 = 0; i2 < C; ++i2) { acc += lamf[i2] * 1.4426950408889634f; g2f[i2] = acc; } }
  __syncthreads();
  for (int i2 = tid; i2 < C; i2 += NTHR) bgef[i2] = betf[i2] * exp2f(g2f[i2]);
  __syncthreads();
  // ---- P7E6: dumps BEFORE phase-1 (phase-1 passes reload kh_/qh_ smem) ----
  for (int e2 = tid; e2 < C * 128; e2 += NTHR) {
    const int i2 = e2 >> 7, k2 = e2 & 127;
    const __half kvv = kh_[(size_t)i2 * LDK + k2];
    kh_g[(size_t)i2 * LDK + k2] = kvv;
    kht_g[(size_t)k2 * LDTC + i2] = kvv;
    kht_lo_g[(size_t)k2 * LDTC + i2] = kh_lo_g[(size_t)i2 * LDK + k2];
    const float qvf = __half2float(qh_[(size_t)i2 * LDK + k2]) * exp2f(g2f[i2]);
    const __half qvh = __float2half(qvf);
    qe_g[(size_t)i2 * LDK + k2] = qvh;
    qe_lo_g[(size_t)i2 * LDK + k2] = __float2half(qvf - __half2float(qvh));
  }
  __syncthreads();

  // ---- phase 1: KK^T and QK^T on tensor cores ----
  float accB[NTPW1][4], accM[NTPW1][4];
  #pragma unroll
  for (int p = 0; p < NTPW1; ++p)
    #pragma unroll
    for (int j = 0; j < 4; ++j) { accB[p][j] = 0.f; accM[p][j] = 0.f; }
  const int g = lane >> 2, tp = (lane & 3) * 2;
  // P7E6: 3 passes — B(khA x khB): (hi,hi),(lo,hi),(hi,lo); M(qh x khB): (hi,hi),(hi,lo),(lo,hi)
  #pragma unroll 1
  for (int ph = 0; ph < 3; ++ph) {
    // R2b FIX: accB and accM need DIFFERENT B-operands per pass to realize
    // B: (hi,hi),(lo,hi),(hi,lo) and M: (hi,hi),(hi,lo),(lo,hi). The old code
    // shared ONE khB -> accM got (hi,hi)+(lo,lo): both cross terms MISSING
    // (M rode bare-fp16 input quality; the P7E6 z-amplifier residual).
    const __half* khB = (ph == 2) ? kh_lo_g : kh_g;   // accB B-operand
    const __half* khBm = (ph == 1) ? kh_lo_g : kh_g;  // accM B-operand
    if (ph == 1) {
      __syncthreads();
      for (int e2 = tid; e2 < C * 128; e2 += NTHR) {
        const int i2 = e2 >> 7, k2 = e2 & 127;
        kh_[(size_t)i2 * LDK + k2] = kh_lo_g[(size_t)i2 * LDK + k2];
      }
      __syncthreads();
    } else if (ph == 2) {
      __syncthreads();
      for (int e2 = tid; e2 < C * 128; e2 += NTHR) {
        const int i2 = e2 >> 7, k2 = e2 & 127;
        kh_[(size_t)i2 * LDK + k2] = kh_g[(size_t)i2 * LDK + k2];
        qh_[(size_t)i2 * LDK + k2] = qh_lo_g[(size_t)i2 * LDK + k2];
      }
      __syncthreads();
    }
    #pragma unroll 1
    for (int s = 0; s < 8; ++s) {
      const int kb = s * 16;
      #pragma unroll
      for (int p = 0; p < NTPW1; ++p) {
        const int idx = warp + p * MMAW1;
        if (idx < NTOT1) {
          const int mt = idx / (C / 8), nt = idx - mt * (C / 8);
          const unsigned a0 = *(const unsigned*)(kh_ + (size_t)(mt*16+g) * LDK + kb + tp);
          const unsigned a1 = *(const unsigned*)(kh_ + (size_t)(mt*16+g+8) * LDK + kb + tp);
          const unsigned a2 = *(const unsigned*)(kh_ + (size_t)(mt*16+g) * LDK + kb + tp + 8);
          const unsigned a3 = *(const unsigned*)(kh_ + (size_t)(mt*16+g+8) * LDK + kb + tp + 8);
          const unsigned b0 = *(const unsigned*)(khB + (size_t)(nt*8+g) * LDK + kb + tp);
          const unsigned b1 = *(const unsigned*)(khB + (size_t)(nt*8+g) * LDK + kb + tp + 8);
          hmma16816(accB[p][0], accB[p][1], accB[p][2], accB[p][3], a0, a1, a2, a3, b0, b1);
          const unsigned c0q = *(const unsigned*)(qh_ + (size_t)(mt*16+g) * LDK + kb + tp);
          const unsigned c1q = *(const unsigned*)(qh_ + (size_t)(mt*16+g+8) * LDK + kb + tp);
          const unsigned c2q = *(const unsigned*)(qh_ + (size_t)(mt*16+g) * LDK + kb + tp + 8);
          const unsigned c3q = *(const unsigned*)(qh_ + (size_t)(mt*16+g+8) * LDK + kb + tp + 8);
          const unsigned mb0 = *(const unsigned*)(khBm + (size_t)(nt*8+g) * LDK + kb + tp);
          const unsigned mb1 = *(const unsigned*)(khBm + (size_t)(nt*8+g) * LDK + kb + tp + 8);
          // R2b FIX: every pass feeds accM its OWN B frags (ph1 = (qh_hi, kh_lo))
          hmma16816(accM[p][0], accM[p][1], accM[p][2], accM[p][3], c0q, c1q, c2q, c3q, mb0, mb1);
        }
      }
    }
  }
  // dumps done PRE-phase-1 (P7E6); Tf below reuses kh's space
  float* Tf = (float*)(sm + GZ_BYTES);
  __syncthreads();
  // masks + scales; M -> global fp16; B -> smem Tf (fp32) for the solve
  #pragma unroll
  for (int p = 0; p < NTPW1; ++p) {
    const int idx = warp + p * MMAW1;
    if (idx < NTOT1) {
      const int mt = idx / (C / 8), nt = idx - mt * (C / 8);
      const int i0 = mt * 16 + g, m0 = nt * 8 + tp;
      const float ei0 = exp2f(g2f[i0] - g2f[m0]), ei0b = exp2f(g2f[i0] - g2f[m0 + 1]);
      const float ei8 = exp2f(g2f[i0 + 8] - g2f[m0]), ei8b = exp2f(g2f[i0 + 8] - g2f[m0 + 1]);
      const float bi0 = betf[i0], bi8 = betf[i0 + 8];
      Tf[(size_t)i0 * TF_LD + m0] = (i0 > m0) ? accB[p][0] * bi0 * ei0 : 0.f;
      Tf[(size_t)i0 * TF_LD + m0 + 1] = (i0 > m0 + 1) ? accB[p][1] * bi0 * ei0b : 0.f;
      Tf[(size_t)(i0 + 8) * TF_LD + m0] = (i0 + 8 > m0) ? accB[p][2] * bi8 * ei8 : 0.f;
      Tf[(size_t)(i0 + 8) * TF_LD + m0 + 1] = (i0 + 8 > m0 + 1) ? accB[p][3] * bi8 * ei8b : 0.f;
      const float fm00 = (i0 >= m0) ? accM[p][0] * ei0 : 0.f;
      const float fm01 = (i0 >= m0 + 1) ? accM[p][1] * ei0b : 0.f;
      const float fm10 = (i0 + 8 >= m0) ? accM[p][2] * ei8 : 0.f;
      const float fm11 = (i0 + 8 >= m0 + 1) ? accM[p][3] * ei8b : 0.f;
      const __half hm00 = __float2half(fm00), hm01 = __float2half(fm01);
      const __half hm10 = __float2half(fm10), hm11 = __float2half(fm11);
      *(__half2*)(m_g + (size_t)i0 * LDTC + m0) = __halves2half2(hm00, hm01);
      *(__half2*)(m_g + (size_t)(i0 + 8) * LDTC + m0) = __halves2half2(hm10, hm11);
      *(__half2*)(m_lo_g + (size_t)i0 * LDTC + m0) =   // P7E5: M hi+lo (multiplies large d)
          __halves2half2(__float2half(fm00 - __half2float(hm00)), __float2half(fm01 - __half2float(hm01)));
      *(__half2*)(m_lo_g + (size_t)(i0 + 8) * LDTC + m0) =
          __halves2half2(__float2half(fm10 - __half2float(hm10)), __float2half(fm11 - __half2float(hm11)));
    }
  }
  __syncthreads();

  // ---- phase 2: T = (I+B)^-1 (fp32 solve) ----
  float* Xf = (float*)(sm + GZ_BYTES + C * TF_LD * 4);
#if C == 64
  float* Gq = (float*)(sm + GZ_BYTES + 2 * C * TF_LD * 4);
#endif
  if (warp == 0) {
    // tile P: rows/cols [0,32) — lane j owns column j
    #pragma unroll 1
    for (int j = lane; j < 32; j += 32) {
      #pragma unroll 1
      for (int i2 = j; i2 < 32; ++i2) {
        float a2c = (i2 == j) ? 1.f : 0.f;
        #pragma unroll 1
        for (int m = j; m < i2; ++m) a2c -= Tf[(size_t)i2 * TF_LD + m] * Xf[(size_t)m * TF_LD + j];
        Xf[(size_t)i2 * TF_LD + j] = a2c;
      }
    }
  }
#if C == 64
  if (warp == 1) {
    // tile R: rows/cols [32,64)
    #pragma unroll 1
    for (int j = lane; j < 32; j += 32) {
      #pragma unroll 1
      for (int i2 = j; i2 < 32; ++i2) {
        float a2c = (i2 == j) ? 1.f : 0.f;
        #pragma unroll 1
        for (int m = j; m < i2; ++m)
          a2c -= Tf[(size_t)(32 + i2) * TF_LD + 32 + m] * Xf[(size_t)(32 + m) * TF_LD + 32 + j];
        Xf[(size_t)(32 + i2) * TF_LD + 32 + j] = a2c;
      }
    }
  }
  __syncthreads();
  // coupling: G = Q X0 ; Tll = -(X1 G)
  for (int e2 = tid; e2 < 1024; e2 += NTHR) {
    const int m = e2 >> 5, n = e2 & 31;
    float a2c = 0.f;
    #pragma unroll 1
    for (int k2 = 0; k2 < 32; ++k2) a2c += Tf[(size_t)(32 + m) * TF_LD + k2] * Xf[(size_t)k2 * TF_LD + n];
    Gq[(size_t)m * 36 + n] = a2c;
  }
  __syncthreads();
  for (int e2 = tid; e2 < 1024; e2 += NTHR) {
    const int m = e2 >> 5, n = e2 & 31;
    float a2c = 0.f;
    #pragma unroll 1
    for (int k2 = 0; k2 < 32; ++k2) a2c += Xf[(size_t)(32 + m) * TF_LD + 32 + k2] * Gq[(size_t)k2 * 36 + n];
    Xf[(size_t)(32 + m) * TF_LD + n] = -a2c;
  }
#else
  // C == 32: T = X0 directly (Xf separate region)
#endif
  // R2b FIX 2: T is unit-LOWER-triangular; the solve writes only the lower
  // triangle. The OLD code left Xf's upper triangle as UNINITIALIZED SMEM (the
  // t_g dump + pfcb's d-MMA then consumed launch-history-dependent garbage:
  // 0.0 on a fresh SM, 7.6e-6-class fp16-leftovers after a sibling pfca, or
  // WORSE after unrelated kernels in the trunk = the R2 WY-C32 NaN class).
  // Zero the FULL upper triangle (subsumes the old C=64 upper-right block).
  for (int e2 = tid; e2 < C * C; e2 += NTHR) {
    const int i2 = e2 / C, j2 = e2 - i2 * C;
    if (j2 > i2) Xf[(size_t)i2 * TF_LD + j2] = 0.f;
  }
  __syncthreads();
  // dump t global fp16 (P7E5: hi+lo — T multiplies large beta.V / beta.2^g.Y)
  for (int e2 = tid; e2 < C * C; e2 += NTHR) {
    const float tv = Xf[(size_t)(e2 / C) * TF_LD + (e2 % C)];
    const __half th = __float2half(tv);
    t_g[(size_t)(e2 / C) * LDTC + (e2 % C)] = th;
    t_lo_g[(size_t)(e2 / C) * LDTC + (e2 % C)] = __float2half(tv - __half2float(th));
  }
  // meta: bg | sg | gend
  for (int i2 = tid; i2 < C; i2 += NTHR) {
    met_g[i2] = bgef[i2];
    met_g[C + i2] = exp2f(g2f[C - 1] - g2f[i2]);
  }
  if (tid == 0) met_g[2 * C] = g2f[C - 1];

  // ---- phase 3: U = T(beta V), two v-halves ----
  __half* t16 = (__half*)(sm + GZ_BYTES);                  // [C][LDTC]
  __half* t16l = (__half*)(sm + GZ_BYTES + C * LDTC * 2);  // [C][LDTC] P7E5 lo
  __half* bvt = (__half*)(sm + GZ_BYTES + 2 * C * LDTC * 2); // [64][LDTC]
  // R2b FIX: t16l's write region overlaps Xf's head rows (C=32: 512B = rows 0-3;
  // C=64: 1KB = rows 0-3) -- the old single-pass staging raced late Xf readers (the
  // t_g dump + this loop's own reads) against early t16l writers -> nondeterministic
  // garbage T rows 0-3 (the R2 WY-C32 trunk NaN). Two-pass register staging: read ALL
  // of Xf first, barrier, then write. C*C/NTHR = 2 (C=32) / 8 (C=64) exact.
  {
    const int NI = C * C / NTHR;
    float tvs[NI > 0 ? NI : 1];
    #pragma unroll
    for (int it = 0; it < (NI > 0 ? NI : 1); ++it) {
      const int e2 = tid + it * NTHR;
      tvs[it] = Xf[(size_t)(e2 / C) * TF_LD + (e2 % C)];
    }
    __syncthreads();  // ALL Xf reads (here + the t_g dump above) complete before any t16/t16l write
    #pragma unroll
    for (int it = 0; it < (NI > 0 ? NI : 1); ++it) {
      const int e2 = tid + it * NTHR;
      const float tv = tvs[it];
      const __half th = __float2half(tv);
      t16[(size_t)(e2 / C) * LDTC + (e2 % C)] = th;
      t16l[(size_t)(e2 / C) * LDTC + (e2 % C)] = __float2half(tv - __half2float(th));
    }
  }
  __syncthreads();  // Xf fully consumed (bvt's region overlaps Xf's head)
  for (int vh = 0; vh < 2; ++vh) {
    // P7E5 3-pass: (T_hi,hi),(T_lo,hi),(T_hi,lo); lo*lo ~2^-42 skipped
    float accU[NTPW2][4];
    int ufirst = 1;
    for (int upass = 0; upass < 3; ++upass) {
      const __half* ta = (upass == 1) ? t16l : t16;
      const int bsel = (upass == 2) ? 1 : 0;
      if (bsel == 0) {
        for (int e2 = tid; e2 < 64 * C; e2 += NTHR) {
          const int v = e2 / C, m = e2 % C;
          const float vf = __half2float(vr_g[(size_t)m * LDK + vh * 64 + v])
                        + __half2float(vr_lo_g[(size_t)m * LDK + vh * 64 + v]);
          bvt[(size_t)v * LDTC + m] = __float2half(vf * betf[m]);  // plain beta (2^g lives on the Y side)
        }
      } else {
        for (int e2 = tid; e2 < 64 * C; e2 += NTHR) {
          const int v = e2 / C, m = e2 % C;
          const float vf = __half2float(vr_g[(size_t)m * LDK + vh * 64 + v])
                        + __half2float(vr_lo_g[(size_t)m * LDK + vh * 64 + v]);
          const float bf = vf * betf[m];
          bvt[(size_t)v * LDTC + m] = __float2half(bf - __half2float(bvt[(size_t)v * LDTC + m]));
        }
      }
      __syncthreads();
      if (ufirst) {
        #pragma unroll
        for (int p = 0; p < NTPW2; ++p)
          #pragma unroll
          for (int j = 0; j < 4; ++j) accU[p][j] = 0.f;
        ufirst = 0;
      }
      #pragma unroll 1
      for (int s = 0; s < C / 16; ++s) {
        const int kb = s * 16;
        #pragma unroll
        for (int p = 0; p < NTPW2; ++p) {
          const int idx = warp + p * MMAW2;
          if (idx < NTOT2) {
            const int mt = idx / 8, nt = idx - mt * 8;
            const unsigned a0 = *(const unsigned*)(ta + (size_t)(mt*16+g) * LDTC + kb + tp);
            const unsigned a1 = *(const unsigned*)(ta + (size_t)(mt*16+g+8) * LDTC + kb + tp);
            const unsigned a2 = *(const unsigned*)(ta + (size_t)(mt*16+g) * LDTC + kb + tp + 8);
            const unsigned a3 = *(const unsigned*)(ta + (size_t)(mt*16+g+8) * LDTC + kb + tp + 8);
            const unsigned b0 = *(const unsigned*)(bvt + (size_t)(nt*8+g) * LDTC + kb + tp);
            const unsigned b1 = *(const unsigned*)(bvt + (size_t)(nt*8+g) * LDTC + kb + tp + 8);
            hmma16816(accU[p][0], accU[p][1], accU[p][2], accU[p][3], a0, a1, a2, a3, b0, b1);
          }
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int p = 0; p < NTPW2; ++p) {
      const int idx = warp + p * MMAW2;
      if (idx < NTOT2) {
        const int mt = idx / 8, nt = idx - mt * 8;
        const int i0 = mt * 16 + g, n0 = nt * 8 + tp;
        const __half u0 = __float2half(accU[p][0]), u1 = __float2half(accU[p][1]);
        const __half u2 = __float2half(accU[p][2]), u3 = __float2half(accU[p][3]);
        *(__half2*)(u_g + (size_t)i0 * LDK + vh * 64 + n0) = __halves2half2(u0, u1);
        *(__half2*)(u_g + (size_t)(i0 + 8) * LDK + vh * 64 + n0) = __halves2half2(u2, u3);
        // P7E5: U hi+lo dump (mature |U| ~ beta*v)
        *(__half2*)(u_lo_g + (size_t)i0 * LDK + vh * 64 + n0) =
            __halves2half2(__float2half(accU[p][0] - __half2float(u0)), __float2half(accU[p][1] - __half2float(u1)));
        *(__half2*)(u_lo_g + (size_t)(i0 + 8) * LDK + vh * 64 + n0) =
            __halves2half2(__float2half(accU[p][2] - __half2float(u2)), __float2half(accU[p][3] - __half2float(u3)));
      }
    }
    __syncthreads();
  }
}

// ============================== KSEL 1: pfcb ==============================
#elif KSEL == 1
#if C == 64
#define MP 2
#else
#define MP 1
#endif
#define SP 4
extern "C" __global__ void __launch_bounds__(256) KNAME(
    const __half* __restrict__ scr, float* __restrict__ recp, float* __restrict__ oout)
{
  __shared__ __align__(16) unsigned char sm[128 * 40 * 4 + 32 * LDK * 2 + 32 * LDTC * 2 + (2 * C + 8) * 4];
  const int h = blockIdx.x >> 2;
  const int vq = blockIdx.x & 3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float* sst = (float*)(sm + 0);                 // [128][40]
  __half* s16 = (__half*)(sm + 128 * 40 * 4);    // [32][LDK]
  __half* sy = (__half*)(sm + 128 * 40 * 4 + 32 * LDK * 2);  // [32][LDTC]
  float* gzs = (float*)(sm + 128 * 40 * 4 + 32 * LDK * 2 + 32 * LDTC * 2); // bg|sg|gend
  const __half* hc0 = scr + (size_t)h * NC * (HC_BYTES / 2);
  float* rec = recp + (size_t)h * 16384;

  // load state slice (transposed: rec[h][v][k] -> sst[k][vloc])
  for (int e2 = tid; e2 < 4096; e2 += 256) {
    const int k2 = e2 & 127, v = e2 >> 7;
    sst[(size_t)k2 * 40 + v] = rec[(size_t)(vq * 32 + v) * 128 + k2];
  }
  __syncthreads();
  for (int e2 = tid; e2 < 4096; e2 += 256) {
    const int k2 = e2 & 127, v = e2 >> 7;
    s16[(size_t)v * LDK + k2] = (__half)sst[(size_t)k2 * 40 + v];
  }
  __syncthreads();

  for (int c = 0; c < NC; ++c) {
    const __half* hc = hc0 + (size_t)c * (HC_BYTES / 2);
    const __half* kh_g = hc;
    const __half* qe_g = hc + KH_C + KHT_C;
    const __half* u_g = qe_g + QE_C;
    const __half* m_g = u_g + U_C + VR_C;   // +VR_C: the vr region sits between u and m
    const __half* t_g = m_g + M_C;
    const __half* u_lo_g = t_g + T_C;       // P7E5 lo halves (layout: ... m|t|u_lo|vr_lo|m_lo|t_lo|dz|meta)
    const __half* vr_lo_g = u_lo_g + ULO_C;
    const __half* m_lo_g = vr_lo_g + VRLO_C;
    const __half* t_lo_g = m_lo_g + MLO_C;
    const float* met_g = (const float*)((const unsigned char*)hc + META_OFF);
    const __half* kh_lo_g = (const __half*)((const unsigned char*)hc + LO_OFF);
    const __half* qe_lo_g = kh_lo_g + KHLO_C + QHLO_C;
    const __half* kht_lo_g = qe_lo_g + QELO_C;
    for (int i2 = tid; i2 < C; i2 += 256) { gzs[i2] = met_g[i2]; gzs[C + i2] = met_g[C + i2]; }
    if (tid == 0) gzs[2 * C] = met_g[2 * C];
    __syncthreads();
    const int g = lane >> 2, tp = (lane & 3) * 2;

    // ---- Y = Khat @ S16 (P7E5: state as hi+lo — s16 rebuilt lo, MMA twice) ----
    float ya[MP][4];
    #pragma unroll
    for (int p = 0; p < MP; ++p)
      #pragma unroll
      for (int j = 0; j < 4; ++j) ya[p][j] = 0.f;
    #pragma unroll 1
    for (int sph = 0; sph < 2; ++sph) {   // P7E6: s16 hi/lo x kh hi/lo (skip lo,lo)
      if (sph == 1) {
        __syncthreads();   // hi-pass reads done; rebuild s16 = lo from the fp32 state
        for (int e2 = tid; e2 < 4096; e2 += 256) {
          const int k2 = e2 & 127, v = e2 >> 7;
          const __half sh2 = (__half)sst[(size_t)k2 * 40 + v];
          s16[(size_t)v * LDK + k2] = __float2half(sst[(size_t)k2 * 40 + v] - __half2float(sh2));
        }
        __syncthreads();
      }
      #pragma unroll 1
      for (int khp = 0; khp < 2; ++khp) {
        if (sph == 1 && khp == 1) continue;
        const __half* kha = khp ? kh_lo_g : kh_g;
        #pragma unroll 1
        for (int s = 0; s < 8; ++s) {
          const int kb = s * 16;
          #pragma unroll
          for (int p = 0; p < MP; ++p) {
            const int idx = warp + p * 8;
            const int mt = idx >> 2, nt = idx & 3;
            const unsigned a0 = *(const unsigned*)(kha + (size_t)(mt*16+g) * LDK + kb + tp);
            const unsigned a1 = *(const unsigned*)(kha + (size_t)(mt*16+g+8) * LDK + kb + tp);
            const unsigned a2 = *(const unsigned*)(kha + (size_t)(mt*16+g) * LDK + kb + tp + 8);
            const unsigned a3 = *(const unsigned*)(kha + (size_t)(mt*16+g+8) * LDK + kb + tp + 8);
            const unsigned b0 = *(const unsigned*)(s16 + (size_t)(nt*8+g) * LDK + kb + tp);
            const unsigned b1 = *(const unsigned*)(s16 + (size_t)(nt*8+g) * LDK + kb + tp + 8);
            hmma16816(ya[p][0], ya[p][1], ya[p][2], ya[p][3], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
    // write sy[v][i] = bg_i*Y[i][v] as hi+lo (P7E5: scaled in fp32 from the
    // fp32 acc — NO raw-fp16 roundtrip; lo stashed in the dead s16 region)
    __syncthreads();
    #pragma unroll
    for (int p = 0; p < MP; ++p) {
      const int idx = warp + p * 8;
      const int mt = idx >> 2, nt = idx & 3;
      const int i0 = mt * 16 + g, v0 = nt * 8 + tp;
      const float bf0 = ya[p][0] * gzs[i0], bf1 = ya[p][1] * gzs[i0];
      const float bf2 = ya[p][2] * gzs[i0 + 8], bf3 = ya[p][3] * gzs[i0 + 8];
      const __half h0 = __float2half(bf0), h1 = __float2half(bf1);
      const __half h2 = __float2half(bf2), h3 = __float2half(bf3);
      sy[(size_t)v0 * LDTC + i0] = h0;
      sy[(size_t)(v0 + 1) * LDTC + i0] = h1;
      sy[(size_t)v0 * LDTC + i0 + 8] = h2;
      sy[(size_t)(v0 + 1) * LDTC + i0 + 8] = h3;
      s16[(size_t)v0 * LDK + i0] = __float2half(bf0 - __half2float(h0));
      s16[(size_t)(v0 + 1) * LDK + i0] = __float2half(bf1 - __half2float(h1));
      s16[(size_t)v0 * LDK + i0 + 8] = __float2half(bf2 - __half2float(h2));
      s16[(size_t)(v0 + 1) * LDK + i0 + 8] = __float2half(bf3 - __half2float(h3));
    }
    __syncthreads();

    // ---- d = U - T @ sy (P7E5: sy holds beta.2^g.Y as hi+lo — T-MMA twice) ----
    float da[MP][4];
    #pragma unroll
    for (int p = 0; p < MP; ++p)
      #pragma unroll
      for (int j = 0; j < 4; ++j) da[p][j] = 0.f;
    #pragma unroll 1
    for (int dpass = 0; dpass < 3; ++dpass) {   // (T,Y) = (hi,hi),(lo,hi),(hi,lo)
      const __half* ta = (dpass == 1) ? t_lo_g : t_g;
      if (dpass == 2) {
        __syncthreads();   // hi-pass reads done; swap the lo stash into sy
        for (int e2 = tid; e2 < 32 * C; e2 += 256) {
          const int v = e2 / C, m = e2 % C;
          sy[(size_t)v * LDTC + m] = s16[(size_t)v * LDK + m];
        }
        __syncthreads();
      }
      #pragma unroll 1
      for (int s = 0; s < C / 16; ++s) {
        const int kb = s * 16;
        #pragma unroll
        for (int p = 0; p < MP; ++p) {
          const int idx = warp + p * 8;
          const int mt = idx >> 2, nt = idx & 3;
          const unsigned a0 = *(const unsigned*)(ta + (size_t)(mt*16+g) * LDTC + kb + tp);
          const unsigned a1 = *(const unsigned*)(ta + (size_t)(mt*16+g+8) * LDTC + kb + tp);
          const unsigned a2 = *(const unsigned*)(ta + (size_t)(mt*16+g) * LDTC + kb + tp + 8);
          const unsigned a3 = *(const unsigned*)(ta + (size_t)(mt*16+g+8) * LDTC + kb + tp + 8);
          const unsigned b0 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp);
          const unsigned b1 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp + 8);
          hmma16816(da[p][0], da[p][1], da[p][2], da[p][3], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    __syncthreads();  // all T@sy reads done before d overwrites sy
    // d = U(hi+lo) - acc; write d as hi+lo (P7E5 — see HI-LO LAW). sy = d_hi
    // (M@d pass-0 B operand); the per-(h,c,vq) dz GLOBAL zone keeps d_hi|d_lo
    // for the M@d lo pass and the sg.d build (it must survive the Qe s16 rebuilds).
    __half* dz = (__half*)hc + KH_C + KHT_C + QE_C + U_C + VR_C + M_C + T_C + ULO_C + VRLO_C + MLO_C + TLO_C + vq * (64 * C);
    #pragma unroll
    for (int p = 0; p < MP; ++p) {
      const int idx = warp + p * 8;
      const int mt = idx >> 2, nt = idx & 3;
      const int i0 = mt * 16 + g, v0 = nt * 8 + tp;
      da[p][0] = (__half2float(u_g[(size_t)i0 * LDK + vq * 32 + v0]) + __half2float(u_lo_g[(size_t)i0 * LDK + vq * 32 + v0])) - da[p][0];
      da[p][1] = (__half2float(u_g[(size_t)i0 * LDK + vq * 32 + v0 + 1]) + __half2float(u_lo_g[(size_t)i0 * LDK + vq * 32 + v0 + 1])) - da[p][1];
      da[p][2] = (__half2float(u_g[(size_t)(i0 + 8) * LDK + vq * 32 + v0]) + __half2float(u_lo_g[(size_t)(i0 + 8) * LDK + vq * 32 + v0])) - da[p][2];
      da[p][3] = (__half2float(u_g[(size_t)(i0 + 8) * LDK + vq * 32 + v0 + 1]) + __half2float(u_lo_g[(size_t)(i0 + 8) * LDK + vq * 32 + v0 + 1])) - da[p][3];
      const __half dh0 = __float2half(da[p][0]), dh1 = __float2half(da[p][1]);
      const __half dh2 = __float2half(da[p][2]), dh3 = __float2half(da[p][3]);
      sy[(size_t)v0 * LDTC + i0] = dh0;
      sy[(size_t)(v0 + 1) * LDTC + i0] = dh1;
      sy[(size_t)v0 * LDTC + i0 + 8] = dh2;
      sy[(size_t)(v0 + 1) * LDTC + i0 + 8] = dh3;
      dz[(size_t)v0 * (2 * C) + i0] = dh0;
      dz[(size_t)(v0 + 1) * (2 * C) + i0] = dh1;
      dz[(size_t)v0 * (2 * C) + i0 + 8] = dh2;
      dz[(size_t)(v0 + 1) * (2 * C) + i0 + 8] = dh3;
      dz[(size_t)v0 * (2 * C) + C + i0] = __float2half(da[p][0] - __half2float(dh0));
      dz[(size_t)(v0 + 1) * (2 * C) + C + i0] = __float2half(da[p][1] - __half2float(dh1));
      dz[(size_t)v0 * (2 * C) + C + i0 + 8] = __float2half(da[p][2] - __half2float(dh2));
      dz[(size_t)(v0 + 1) * (2 * C) + C + i0 + 8] = __float2half(da[p][3] - __half2float(dh3));
    }
    __syncthreads();

    // ---- O = M @ d(hi+lo) + Qe @ S16(hi+lo) ----
    float oa[MP][4];
    #pragma unroll
    for (int p = 0; p < MP; ++p)
      #pragma unroll
      for (int j = 0; j < 4; ++j) oa[p][j] = 0.f;
    #pragma unroll 1
    for (int mpass = 0; mpass < 3; ++mpass) {   // (M,d) = (hi,hi),(lo,hi),(hi,lo)
      const __half* ma = (mpass == 1) ? m_lo_g : m_g;
      if (mpass == 2) {
        __syncthreads();
        for (int e2 = tid; e2 < 32 * C; e2 += 256) {
          const int v = e2 / C, m = e2 % C;
          sy[(size_t)v * LDTC + m] = dz[(size_t)v * (2 * C) + C + m];
        }
        __syncthreads();
      }
      #pragma unroll 1
      for (int s = 0; s < C / 16; ++s) {
        const int kb = s * 16;
        #pragma unroll
        for (int p = 0; p < MP; ++p) {
          const int idx = warp + p * 8;
          const int mt = idx >> 2, nt = idx & 3;
          const unsigned a0 = *(const unsigned*)(ma + (size_t)(mt*16+g) * LDTC + kb + tp);
          const unsigned a1 = *(const unsigned*)(ma + (size_t)(mt*16+g+8) * LDTC + kb + tp);
          const unsigned a2 = *(const unsigned*)(ma + (size_t)(mt*16+g) * LDTC + kb + tp + 8);
          const unsigned a3 = *(const unsigned*)(ma + (size_t)(mt*16+g+8) * LDTC + kb + tp + 8);
          const unsigned b0 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp);
          const unsigned b1 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp + 8);
          hmma16816(oa[p][0], oa[p][1], oa[p][2], oa[p][3], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    __syncthreads();
    #pragma unroll 1
    for (int sph = 0; sph < 2; ++sph) {   // P7E6: s16 hi/lo x qe hi/lo (skip lo,lo)
      for (int e2 = tid; e2 < 4096; e2 += 256) {
        const int k2 = e2 & 127, v = e2 >> 7;
        const float sv2 = sst[(size_t)k2 * 40 + v];
        const __half sh2 = (__half)sv2;
        s16[(size_t)v * LDK + k2] = sph ? __float2half(sv2 - __half2float(sh2)) : sh2;
      }
      __syncthreads();
      #pragma unroll 1
      for (int qhp = 0; qhp < 2; ++qhp) {
        if (sph == 1 && qhp == 1) continue;
        const __half* qea = qhp ? qe_lo_g : qe_g;
        #pragma unroll 1
        for (int s = 0; s < 8; ++s) {        // Qe @ S  (K = 128)
          const int kb = s * 16;
          #pragma unroll
          for (int p = 0; p < MP; ++p) {
            const int idx = warp + p * 8;
            const int mt = idx >> 2, nt = idx & 3;
            const unsigned d0 = *(const unsigned*)(qea + (size_t)(mt*16+g) * LDK + kb + tp);
            const unsigned d1 = *(const unsigned*)(qea + (size_t)(mt*16+g+8) * LDK + kb + tp);
            const unsigned d2 = *(const unsigned*)(qea + (size_t)(mt*16+g) * LDK + kb + tp + 8);
            const unsigned d3 = *(const unsigned*)(qea + (size_t)(mt*16+g+8) * LDK + kb + tp + 8);
            const unsigned e0 = *(const unsigned*)(s16 + (size_t)(nt*8+g) * LDK + kb + tp);
            const unsigned e1 = *(const unsigned*)(s16 + (size_t)(nt*8+g) * LDK + kb + tp + 8);
            hmma16816(oa[p][0], oa[p][1], oa[p][2], oa[p][3], d0, d1, d2, d3, e0, e1);
          }
        }
        __syncthreads();
      }
    }
    #pragma unroll
    for (int p = 0; p < MP; ++p) {
      const int idx = warp + p * 8;
      const int mt = idx >> 2, nt = idx & 3;
      const int i0 = mt * 16 + g, v0 = nt * 8 + tp;
      const size_t ob = (size_t)(c * C + i0) * 6144 + h * 128 + vq * 32 + v0;
      oout[ob] = oa[p][0];
      oout[ob + 1] = oa[p][1];
      oout[(size_t)(c * C + i0 + 8) * 6144 + h * 128 + vq * 32 + v0] = oa[p][2];
      oout[(size_t)(c * C + i0 + 8) * 6144 + h * 128 + vq * 32 + v0 + 1] = oa[p][3];
    }
    // ---- S' = 2^g_end * S + KhatT @ (sg.d hi+lo)  (P7E5) ----
    // build sg.d from the dz d halves (fp32 math): sy = hi, dz[C+m] = lo
    for (int e2 = tid; e2 < 32 * C; e2 += 256) {
      const int v = e2 / C, m = e2 % C;
      const float bf = (__half2float(dz[(size_t)v * (2 * C) + m]) + __half2float(dz[(size_t)v * (2 * C) + C + m])) * gzs[C + m];
      const __half bh = __float2half(bf);
      sy[(size_t)v * LDTC + m] = bh;
      dz[(size_t)v * (2 * C) + C + m] = __float2half(bf - __half2float(bh));
    }
    __syncthreads();
    const __half* kht_g = hc + KH_C;
    const float gd = exp2f(gzs[2 * C]);
    float sa[SP][4];
    #pragma unroll
    for (int p = 0; p < SP; ++p)
      #pragma unroll
      for (int j = 0; j < 4; ++j) sa[p][j] = 0.f;
    #pragma unroll 1
    for (int spass = 0; spass < 3; ++spass) {   // P7E6: (hi,hi),(hi,lo),(lo,hi)
      const __half* kta = (spass == 2) ? kht_lo_g : kht_g;
      if (spass == 1 || spass == 2) {
        __syncthreads();
        for (int e2 = tid; e2 < 32 * C; e2 += 256) {
          const int v = e2 / C, m = e2 % C;
          if (spass == 1) {   // swap in the lo halves
            sy[(size_t)v * LDTC + m] = dz[(size_t)v * (2 * C) + C + m];
          } else {            // restore the hi halves (recompute from dz, fp32)
            const float bf = (__half2float(dz[(size_t)v * (2 * C) + m]) + __half2float(dz[(size_t)v * (2 * C) + C + m])) * gzs[C + m];
            const __half bh = __float2half(bf);
            sy[(size_t)v * LDTC + m] = bh;
            dz[(size_t)v * (2 * C) + C + m] = __float2half(bf - __half2float(bh));
          }
        }
        __syncthreads();
      }
      #pragma unroll 1
      for (int p = 0; p < SP; ++p) {
        const int idx = warp + p * 8;
        const int mt = idx >> 2, nt = idx & 3;
        #pragma unroll 1
        for (int s = 0; s < C / 16; ++s) {
          const int kb = s * 16;
          const unsigned a0 = *(const unsigned*)(kta + (size_t)(mt*16+g) * LDTC + kb + tp);
          const unsigned a1 = *(const unsigned*)(kta + (size_t)(mt*16+g+8) * LDTC + kb + tp);
          const unsigned a2 = *(const unsigned*)(kta + (size_t)(mt*16+g) * LDTC + kb + tp + 8);
          const unsigned a3 = *(const unsigned*)(kta + (size_t)(mt*16+g+8) * LDTC + kb + tp + 8);
          const unsigned b0 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp);
          const unsigned b1 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp + 8);
          hmma16816(sa[p][0], sa[p][1], sa[p][2], sa[p][3], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    #pragma unroll
    for (int p = 0; p < SP; ++p) {
      const int idx = warp + p * 8;
      const int mt = idx >> 2, nt = idx & 3;
      const int k0 = mt * 16 + g, v0 = nt * 8 + tp;
      sst[(size_t)k0 * 40 + v0] = sst[(size_t)k0 * 40 + v0] * gd + sa[p][0];
      sst[(size_t)k0 * 40 + v0 + 1] = sst[(size_t)k0 * 40 + v0 + 1] * gd + sa[p][1];
      sst[(size_t)(k0 + 8) * 40 + v0] = sst[(size_t)(k0 + 8) * 40 + v0] * gd + sa[p][2];
      sst[(size_t)(k0 + 8) * 40 + v0 + 1] = sst[(size_t)(k0 + 8) * 40 + v0 + 1] * gd + sa[p][3];
    }
    __syncthreads();
    if (c + 1 < NC) {
      for (int e2 = tid; e2 < 4096; e2 += 256) {
        const int k2 = e2 & 127, v = e2 >> 7;
        s16[(size_t)v * LDK + k2] = (__half)sst[(size_t)k2 * 40 + v];
      }
      __syncthreads();
    }
  }
  // store state back
  for (int e2 = tid; e2 < 4096; e2 += 256) {
    const int k2 = e2 & 127, v = e2 >> 7;
    rec[(size_t)(vq * 32 + v) * 128 + k2] = sst[(size_t)k2 * 40 + v];
  }
}

// ============================== KSEL 3: pfcbdbg (dump d) ==============================
#elif KSEL == 3
#if C == 64
#define MP 2
#else
#define MP 1
#endif
extern "C" __global__ void __launch_bounds__(256) KNAME(
    const __half* __restrict__ scr, float* __restrict__ recp, float* __restrict__ oout)
{
  __shared__ __align__(16) unsigned char sm[128 * 40 * 4 + 32 * LDK * 2 + 32 * LDTC * 2 + (2 * C + 8) * 4];
  const int h = blockIdx.x >> 2;
  const int vq = blockIdx.x & 3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float* sst = (float*)(sm + 0);
  __half* s16 = (__half*)(sm + 128 * 40 * 4);
  __half* sy = (__half*)(sm + 128 * 40 * 4 + 32 * LDK * 2);
  float* gzs = (float*)(sm + 128 * 40 * 4 + 32 * LDK * 2 + 32 * LDTC * 2);
  const __half* hc = scr + (size_t)h * NC * (HC_BYTES / 2);
  const __half* kh_g = hc;
  const __half* u_g = hc + KH_C + KHT_C + QE_C;
  const __half* t_g = hc + KH_C + KHT_C + QE_C + U_C + VR_C + M_C;
  const float* met_g = (const float*)((const unsigned char*)hc + META_OFF);
  for (int e2 = tid; e2 < 4096; e2 += 256) {
    const int k2 = e2 & 127, v = e2 >> 7;
    sst[(size_t)k2 * 40 + v] = recp[(size_t)h * 16384 + (size_t)(vq * 32 + v) * 128 + k2];
  }
  __syncthreads();
  for (int e2 = tid; e2 < 4096; e2 += 256) {
    const int k2 = e2 & 127, v = e2 >> 7;
    s16[(size_t)v * LDK + k2] = (__half)sst[(size_t)k2 * 40 + v];
  }
  __syncthreads();
  for (int i2 = tid; i2 < C; i2 += 256) { gzs[i2] = met_g[i2]; gzs[C + i2] = met_g[C + i2]; }
  if (tid == 0) gzs[2 * C] = met_g[2 * C];
  __syncthreads();
  const int g = lane >> 2, tp = (lane & 3) * 2;
  float ya[MP][4];
  #pragma unroll
  for (int p = 0; p < MP; ++p)
    #pragma unroll
    for (int j = 0; j < 4; ++j) ya[p][j] = 0.f;
  for (int s = 0; s < 8; ++s) {
    const int kb = s * 16;
    #pragma unroll
    for (int p = 0; p < MP; ++p) {
      const int idx = warp + p * 8;
      const int mt = idx >> 2, nt = idx & 3;
      const unsigned a0 = *(const unsigned*)(kh_g + (size_t)(mt*16+g) * LDK + kb + tp);
      const unsigned a1 = *(const unsigned*)(kh_g + (size_t)(mt*16+g+8) * LDK + kb + tp);
      const unsigned a2 = *(const unsigned*)(kh_g + (size_t)(mt*16+g) * LDK + kb + tp + 8);
      const unsigned a3 = *(const unsigned*)(kh_g + (size_t)(mt*16+g+8) * LDK + kb + tp + 8);
      const unsigned b0 = *(const unsigned*)(s16 + (size_t)(nt*8+g) * LDK + kb + tp);
      const unsigned b1 = *(const unsigned*)(s16 + (size_t)(nt*8+g) * LDK + kb + tp + 8);
      hmma16816(ya[p][0], ya[p][1], ya[p][2], ya[p][3], a0, a1, a2, a3, b0, b1);
    }
  }
  #pragma unroll
  for (int p = 0; p < MP; ++p) {
    const int idx = warp + p * 8;
    const int mt = idx >> 2, nt = idx & 3;
    const int i0 = mt * 16 + g, v0 = nt * 8 + tp;
    sy[(size_t)v0 * LDTC + i0] = __float2half(ya[p][0]);
    sy[(size_t)(v0 + 1) * LDTC + i0] = __float2half(ya[p][1]);
    sy[(size_t)v0 * LDTC + i0 + 8] = __float2half(ya[p][2]);
    sy[(size_t)(v0 + 1) * LDTC + i0 + 8] = __float2half(ya[p][3]);
  }
  __syncthreads();
  for (int e2 = tid; e2 < 32 * C; e2 += 256) {
    const int v = e2 / C, m = e2 % C;
    sy[(size_t)v * LDTC + m] = __float2half(__half2float(sy[(size_t)v * LDTC + m]) * gzs[m]);
  }
  __syncthreads();
  float da[MP][4];
  #pragma unroll
  for (int p = 0; p < MP; ++p)
    #pragma unroll
    for (int j = 0; j < 4; ++j) da[p][j] = 0.f;
  for (int s = 0; s < C / 16; ++s) {
    const int kb = s * 16;
    #pragma unroll
    for (int p = 0; p < MP; ++p) {
      const int idx = warp + p * 8;
      const int mt = idx >> 2, nt = idx & 3;
      const unsigned a0 = *(const unsigned*)(t_g + (size_t)(mt*16+g) * LDTC + kb + tp);
      const unsigned a1 = *(const unsigned*)(t_g + (size_t)(mt*16+g+8) * LDTC + kb + tp);
      const unsigned a2 = *(const unsigned*)(t_g + (size_t)(mt*16+g) * LDTC + kb + tp + 8);
      const unsigned a3 = *(const unsigned*)(t_g + (size_t)(mt*16+g+8) * LDTC + kb + tp + 8);
      const unsigned b0 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp);
      const unsigned b1 = *(const unsigned*)(sy + (size_t)(nt*8+g) * LDTC + kb + tp + 8);
      hmma16816(da[p][0], da[p][1], da[p][2], da[p][3], a0, a1, a2, a3, b0, b1);
    }
  }
  __syncthreads();
  // DUMP: raw ya (Y) into slot h=46, d (post-fixup) into slot h=47 (h==0 ONLY)
  if (h != 0) return;
  #pragma unroll
  for (int p = 0; p < MP; ++p) {
    const int idx = warp + p * 8;
    const int mt = idx >> 2, nt = idx & 3;
    const int i0 = mt * 16 + g, v0 = nt * 8 + tp;
    oout[(size_t)i0 * 6144 + 46 * 128 + vq * 32 + v0] = ya[p][0];
    oout[(size_t)i0 * 6144 + 46 * 128 + vq * 32 + v0 + 1] = ya[p][1];
    oout[(size_t)(i0 + 8) * 6144 + 46 * 128 + vq * 32 + v0] = ya[p][2];
    oout[(size_t)(i0 + 8) * 6144 + 46 * 128 + vq * 32 + v0 + 1] = ya[p][3];
    const float d0 = __half2float(u_g[(size_t)i0 * LDK + vq * 32 + v0]) - da[p][0];
    const float d1 = __half2float(u_g[(size_t)i0 * LDK + vq * 32 + v0 + 1]) - da[p][1];
    const float d2 = __half2float(u_g[(size_t)(i0 + 8) * LDK + vq * 32 + v0]) - da[p][2];
    const float d3 = __half2float(u_g[(size_t)(i0 + 8) * LDK + vq * 32 + v0 + 1]) - da[p][3];
    oout[(size_t)i0 * 6144 + 47 * 128 + vq * 32 + v0] = d0;
    oout[(size_t)i0 * 6144 + 47 * 128 + vq * 32 + v0 + 1] = d1;
    oout[(size_t)(i0 + 8) * 6144 + 47 * 128 + vq * 32 + v0] = d2;
    oout[(size_t)(i0 + 8) * 6144 + 47 * 128 + vq * 32 + v0 + 1] = d3;
  }
}
// ============================== KSEL 2: pfcz ==============================
#elif KSEL == 2
extern "C" __global__ void __launch_bounds__(256) KNAME(
    const float* __restrict__ oout, const __half* __restrict__ gate,
    const float* __restrict__ snw, __half* __restrict__ zout,
    const __half* __restrict__ kvbuf, float* __restrict__ convlive)
{
  // grid = NH*NC*(C/8); CTA = (h, c, 8-row tile); one row per WARP (v2: the
  // per-thread serial 128-loop was latency-bound — 18 GB/s class)
  const int r8 = blockIdx.x % (C / 8);
  const int hc_ = blockIdx.x / (C / 8);
  const int h = hc_ / NC;
  const int c = hc_ - h * NC;
  const int c0 = c * C;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int kh = h % 16;
  const int qc0 = kh * 128, kc0 = QDIM + kh * 128, vc0 = 2 * QDIM + h * 128;
  const int i = c0 + r8 * 8 + warp;
  const float* orow = oout + (size_t)i * 6144 + h * 128;
  float zz = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) { const float cc = orow[lane * 4 + j]; zz += cc * cc; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) zz += __shfl_xor_sync(FULL, zz, o);
  const float rz = rsqrtf(zz / 128 + EPS_N);
  const __half* grow = gate + (size_t)i * 6144 + h * 128;
  __half* zrow = zout + (size_t)i * 6144 + h * 128;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int v = lane * 4 + j;
    const __half gg = grow[v];
    zrow[v] = __float2half((orow[v] * rz * snw[v]) * __half2float(
        __hmul(gg, hrcp((__half)1.0f + hexp2(__hmul(gg, __float2half(-1.4423828125f)))))));
  }
  // conv_live writeback (last chunk only): rows T-3..T of kvbuf (fp32)
  if (c == NC - 1) {
    for (int i2 = tid; i2 < 3 * 384; i2 += 256) {
      const int row = i2 / 384, off = i2 - row * 384;
      const int sec = off >> 7, lo = off & 127;
      const int ch = (sec == 0) ? qc0 + lo : (sec == 1) ? kc0 + lo : vc0 + lo;
      convlive[(size_t)row * CONV_CH + ch] = __half2float(kvbuf[(size_t)(c0 + C - 3 + row) * CONV_CH + ch]);
    }
  }
}
#endif
