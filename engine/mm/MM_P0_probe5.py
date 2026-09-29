#!/usr/bin/env python3
"""MM_P0 D3 fault bisect #5 — VA-pointer-table legality (with size-legal banks).
S1: 12-arg VA-table kernel + 8x400MB banks + real gridf + 88-pair launch
S2: 20x373MB (7.3GB, 5120 experts) + scattered 88-pair launch + TIMING
(one boot; staged; timing grabbed at the last clean stage as a bonus)."""
import os, sys, subprocess, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, GATE_B = 1458176, 450560

KSRC = r"""
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define KB 8
#define ROWB (110*KB)
#define NROW 512
extern "C" __global__ void __launch_bounds__(1024) kbv(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs, const float* __restrict__ gridf,
    float* __restrict__ ys)
{
  const int p = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[p]];
  const float* x = xs + (size_t)p*2048;
  float* y = ys + (size_t)p*512;
  for (int r0 = 0; r0 < NROW; r0 += 32) {
    const unsigned char* rowp = base + (size_t)(r0+warp)*ROWB;
    float a = 0.f;
    #pragma unroll
    for (int b = 0; b < KB; ++b) {
      const unsigned char* blk = rowp + b*110;
      const float d = __half2float(*((const __half*)blk));
      const int g0i = lane*2, g1i = lane*2 + 1;
      const int sraw = lane >> 2;
      const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
      const unsigned int sg = blk[74 + lane];
      const int koff = (b << 8) + (lane << 3);
      const float x0 = x[koff+0], x1 = x[koff+1], x2 = x[koff+2], x3 = x[koff+3];
      const float x4 = x[koff+4], x5 = x[koff+5], x6 = x[koff+6], x7 = x[koff+7];
      const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
      const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
      const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
      const float* gr0 = gridf + (size_t)qb0*4;
      const float* gr1 = gridf + (size_t)qb1*4;
      float w0 = d*sc*gr0[0]; if (sg & 0x01u) w0 = -w0;
      float w1 = d*sc*gr0[1]; if (sg & 0x02u) w1 = -w1;
      float w2 = d*sc*gr0[2]; if (sg & 0x04u) w2 = -w2;
      float w3 = d*sc*gr0[3]; if (sg & 0x08u) w3 = -w3;
      float w4 = d*sc*gr1[0]; if (sg & 0x10u) w4 = -w4;
      float w5 = d*sc*gr1[1]; if (sg & 0x20u) w5 = -w5;
      float w6 = d*sc*gr1[2]; if (sg & 0x40u) w6 = -w6;
      float w7 = d*sc*gr1[3]; if (sg & 0x80u) w7 = -w7;
      a += w0*x0; a += w1*x1; a += w2*x2; a += w3*x3;
      a += w4*x4; a += w5*x5; a += w6*x6; a += w7*x7;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(FULL, a, o);
    if (lane == 0) y[r0+warp] = a;
  }
}
"""

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    open(f"{BASE}/MM_P0_kbv.cu", "w").write(KSRC)
    r = subprocess.run(f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={BASE}/MM_P0_kbv.cubin {BASE}/MM_P0_kbv.cu",
                       shell=True, capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-800:]); sys.exit(1)
    lib = open(f"{BASE}/MM_P0_kbv.cubin", "rb").read()
    kbv = NVProgram(dev, TinyELF(lib=lib, name="kbv", target=dev.renderer.target, signature=tuple()))
    dev.synchronize(); print("[kbv loaded]", flush=True)
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    gridf = up(v.view(np.int8).astype(np.float32).reshape(512, 4).copy())
    xs = up(np.random.default_rng(2).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32))
    ys = dev.allocator.alloc(88*512*4, BufferSpec())
    eids8 = up(np.arange(8, dtype=np.uint16))
    dev.synchronize(); print("[bufs up]", flush=True)

    rng = np.random.default_rng(7)
    def mkbanks(n, per_bytes):
        out = []
        for s in range(n):
            b = dev.allocator.alloc(per_bytes, BufferSpec())
            for off in range(0, per_bytes, 64<<20):
                nb = min(64<<20, per_bytes-off)
                a = rng.integers(0, 256, nb, dtype=np.uint8); keep.append(a)
                dev.allocator._copyin(b.offset(offset=off, size=nb), memoryview(a.data).cast("B"))
            out.append(b)
        return out

    # S1: 8 x 400MB, 8 experts, VA table
    E1 = 274   # experts per 400MB shard
    banks = mkbanks(8, E1*SLAB)
    pt = np.array([banks[e//E1].va_addr + (e % E1)*SLAB for e in range(E1*8)], dtype=np.uint64)
    ptbl = up(pt)
    dev.synchronize()
    kbv(ptbl, eids8, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[S1] VA-table + 8x400MB: CLEAN", flush=True)

    # S2: 20 x 373MB = 7.3GB, 5120 experts, 88-pair scattered + timing
    E2 = 256
    banks2 = mkbanks(20, E2*SLAB)
    pt2 = np.array([banks2[e//E2].va_addr + (e % E2)*SLAB for e in range(E2*20)], dtype=np.uint64)
    ptbl2 = up(pt2)
    scat = np.concatenate([np.random.default_rng(100+i).choice(E2*20, 8, replace=False) for i in range(11)]).astype(np.uint16)
    eb = up(scat)
    dev.synchronize()
    print("[S2] 7.3GB bank up; launching scattered 88", flush=True)
    kbv(ptbl2, eb, xs, gridf, ys, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
    print("[S2] scattered 88: CLEAN", flush=True)
    byts = 88*GATE_B
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        kbv(ptbl2, eb, xs, gridf, ys, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
        ts.append((time.perf_counter()-t0)*1e3)
    tmin = min(ts)
    print(f"[S2-timing] scattered-88: min {tmin:.3f} ms -> {byts/tmin/1e3:.1f} GB/s (distinct={len(set(scat.tolist()))})", flush=True)
    grp = up(np.tile(np.arange(8, dtype=np.uint16), 11))
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        kbv(ptbl2, grp, xs, gridf, ys, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
        ts.append((time.perf_counter()-t0)*1e3)
    print(f"[S2-timing] grouped-8: min {min(ts):.3f} ms -> {byts/min(ts)/1e3:.1f} GB/s", flush=True)
    big = up((rng.integers(0, E2*20, 88)).astype(np.uint16))
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        kbv(ptbl2, big, xs, gridf, ys, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
        ts.append((time.perf_counter()-t0)*1e3)
    print(f"[S2-timing] bankscatter-88 (full 7.3GB): min {min(ts):.3f} ms -> {byts/min(ts)/1e3:.1f} GB/s", flush=True)

if __name__ == "__main__":
    main()
