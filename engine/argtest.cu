// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
extern "C" __global__ void __launch_bounds__(256) k6(float* a, float* b, float* c, float* d, float* e, float* f)
{ const int tid = (blockIdx.x << 8) + threadIdx.x; if (tid == 0) a[0] = f[0] + 1.f; }
extern "C" __global__ void __launch_bounds__(256) k8(float* a, float* b, float* c, float* d, float* e, float* f, float* g, float* h)
{ const int tid = (blockIdx.x << 8) + threadIdx.x; if (tid == 0) a[0] = h[0] + 1.f; }
extern "C" __global__ void __launch_bounds__(256) k10(float* a, float* b, float* c, float* d, float* e, float* f, float* g, float* h, float* i, float* j)
{ const int tid = (blockIdx.x << 8) + threadIdx.x; if (tid == 0) a[0] = j[0] + 1.f; }
extern "C" __global__ void __launch_bounds__(256) k12(float* a, float* b, float* c, float* d, float* e, float* f, float* g, float* h, float* i, float* j, float* k, float* l)
{ const int tid = (blockIdx.x << 8) + threadIdx.x; if (tid == 0) a[0] = l[0] + 1.f; }
extern "C" __global__ void __launch_bounds__(256) k14(float* a, float* b, float* c, float* d, float* e, float* f, float* g, float* h, float* i, float* j, float* k, float* l, float* m, float* n)
{ const int tid = (blockIdx.x << 8) + threadIdx.x; if (tid == 0) a[0] = n[0] + 1.f; }
