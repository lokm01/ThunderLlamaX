// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W0/E4: dext data-path bandwidth probes (standalone harness).
// HARD RULES honored: flat indexing ONLY, no gridDim reads (dext reports 0) —
// strides/grid sizes arrive as kernel args. Bounds guards everywhere.
// 256 threads/CTA, sm_86 (RTX 3090, 82 SMs).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define FULL 0xffffffffu

// K_STREAM: pure-read bandwidth over an fp16 buffer, 16B vector loads,
// per-thread fp32 accumulation, 1 float written per thread (anti-DCE).
extern "C" __global__ void __launch_bounds__(256) k_stream(
    const __half* __restrict__ w, float* __restrict__ out,
    const int nvec, const int nthreads)
{
    const int tid = (blockIdx.x << 8) + threadIdx.x;  // blockDim.x reads 0 on this dext (cbuf0[0x0]=0) — hardcode 256
    if (tid >= nthreads) return;
    float acc = 0.f;
    for (int i = tid; i < nvec; i += nthreads) {
        const float4 p = *(const float4*)(w + ((long)i << 3));
        const __half2* h = (const __half2*)&p;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 a = __half22float2(h[j]);
            acc += a.x + a.y;
        }
    }
    out[tid] = acc;
}

// K_GEMV_FP16: y[N] = W[N,K] @ x[K]. Warp-per-row, coalesced 16B loads
// (each lane takes consecutive 8-half chunks), fp32 accumulate, warp shfl reduce.
extern "C" __global__ void __launch_bounds__(256) k_gemv16(
    const __half* __restrict__ w, const __half* __restrict__ x, float* __restrict__ y,
    const int N, const int K, const int nwarps)
{
    const int tid = (blockIdx.x << 8) + threadIdx.x;  // blockDim.x reads 0 on this dext (cbuf0[0x0]=0) — hardcode 256
    const int lane = tid & 31, warp = tid >> 5;
    const int nch = K >> 3;                       // 8-half chunks per row
    for (int row = warp; row < N; row += nwarps) {
        const __half* wr = w + (long)row * K;
        float acc = 0.f;
        for (int c = lane; c < nch; c += 32) {
            const float4 pw = *(const float4*)(wr + ((long)c << 3));
            const float4 px = *(const float4*)(x + ((long)c << 3));
            const __half2* hw = (const __half2*)&pw;
            const __half2* hx = (const __half2*)&px;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 a = __half22float2(hw[j]);
                const float2 b = __half22float2(hx[j]);
                acc += a.x * b.x + a.y * b.y;
            }
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(FULL, acc, off);
        if (lane == 0) y[row] = acc;
    }
}

// Optimized mock: register-only nibble extraction (no byte-view of uint4 —
// that pattern risks local-mem spills) + 4 independent accumulators for ILP.
extern "C" __global__ void __launch_bounds__(256) k_gemviq2(
    const unsigned char* __restrict__ wq, const __half* __restrict__ sc,
    const __half* __restrict__ x, float* __restrict__ y,
    const int N, const int K, const int nwarps)
{
    const int tid = (blockIdx.x << 8) + threadIdx.x;  // blockDim.x reads 0 on this dext (cbuf0[0x0]=0) — hardcode 256
    const int lane = tid & 31, warp = tid >> 5;
    const int nch = K >> 5;
    for (int row = warp; row < N; row += nwarps) {
        const unsigned char* wr = wq + (long)row * (K >> 1);
        const __half* sr = sc + (long)row * (K >> 7);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int c = lane; c < nch; c += 32) {
            const uint4 p = *(const uint4*)(wr + ((long)c << 4));
            const float s = __half2float(sr[c >> 2]);
            const __half2* xr = (const __half2*)(x + ((long)c << 5));
            float n0 = 0.f, n1 = 0.f, n2 = 0.f, n3 = 0.f;
            const unsigned int wv0 = p.x, wv1 = p.y, wv2 = p.z, wv3 = p.w;
            #define UNPACK8(WV, U) do { \
                const float2 f0 = __half22float2(xr[4*(U)+0]); \
                const float2 f1 = __half22float2(xr[4*(U)+1]); \
                const float2 f2 = __half22float2(xr[4*(U)+2]); \
                const float2 f3 = __half22float2(xr[4*(U)+3]); \
                n0 += ((int)((WV      ) & 15u) - 8) * f0.x; \
                n1 += ((int)((WV >>  4) & 15u) - 8) * f0.y; \
                n2 += ((int)((WV >>  8) & 15u) - 8) * f1.x; \
                n3 += ((int)((WV >> 12) & 15u) - 8) * f1.y; \
                n0 += ((int)((WV >> 16) & 15u) - 8) * f2.x; \
                n1 += ((int)((WV >> 20) & 15u) - 8) * f2.y; \
                n2 += ((int)((WV >> 24) & 15u) - 8) * f3.x; \
                n3 += ((int)((WV >> 28)       ) - 8) * f3.y; \
            } while (0)
            UNPACK8(wv0, 0); UNPACK8(wv1, 1); UNPACK8(wv2, 2); UNPACK8(wv3, 3);
            #undef UNPACK8
            a0 += n0 * s; a1 += n1 * s; a2 += n2 * s; a3 += n3 * s;
        }
        float acc = ((a0 + a1) + (a2 + a3));
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(FULL, acc, off);
        if (lane == 0) y[row] = acc;
    }
}

// K_GEMV_IQ4_MOCK: int4-g128 dequant-GEMV stand-in. wq = 2 nibbles/byte
// [N, K/2], sc = fp16 scales [N, K/128]. Lane processes 32-element chunks
// (one uint4 = 16B), all 32 elements share one scale group. Nibble in [-8,7].
extern "C" __global__ void __launch_bounds__(256) k_gemviq(
    const unsigned char* __restrict__ wq, const __half* __restrict__ sc,
    const __half* __restrict__ x, float* __restrict__ y,
    const int N, const int K, const int nwarps)
{
    const int tid = (blockIdx.x << 8) + threadIdx.x;  // blockDim.x reads 0 on this dext (cbuf0[0x0]=0) — hardcode 256
    const int lane = tid & 31, warp = tid >> 5;
    const int nch = K >> 5;                       // 32-element chunks per row
    for (int row = warp; row < N; row += nwarps) {
        const unsigned char* wr = wq + (long)row * (K >> 1);
        const __half* sr = sc + (long)row * (K >> 7);
        float acc = 0.f;
        for (int c = lane; c < nch; c += 32) {
            const uint4 p = *(const uint4*)(wr + ((long)c << 4));
            const float s = __half2float(sr[c >> 2]);
            const __half* xr = x + ((long)c << 5);
            const unsigned char* pb = (const unsigned char*)&p;
            float part = 0.f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                part += (float)((pb[b] & 15) - 8) * __half2float(xr[2*b])
                      + (float)((pb[b] >> 4) - 8) * __half2float(xr[2*b+1]);
            }
            acc += part * s;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(FULL, acc, off);
        if (lane == 0) y[row] = acc;
    }
}
