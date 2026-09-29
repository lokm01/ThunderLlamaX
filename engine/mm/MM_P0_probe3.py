#!/usr/bin/env python3
"""MM_P0 D3 fault bisect #3 — kernel-source bisection over the 12-arg signature.
K1: 12-arg signature, minimal body (touch all args, no dequant).
K2: + shard division/select addressing (read 1 byte per bank via selected base).
K3: + the full IQ3_S dequant row loop (the real kernel).
One boot; stops at first fault."""
import os, sys, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, SHARDN = 1458176, 850

COMMON = r"""
#include <cuda_fp16.h>
#define SHARDN 850
#define SLABB 1458176
"""

K1 = COMMON + r"""
extern "C" __global__ void __launch_bounds__(1024) kb1(
    const unsigned char* __restrict__ b0, const unsigned char* __restrict__ b1,
    const unsigned char* __restrict__ b2, const unsigned char* __restrict__ b3,
    const unsigned char* __restrict__ b4, const unsigned char* __restrict__ b5,
    const unsigned char* __restrict__ b6, const unsigned char* __restrict__ b7,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs, const float* __restrict__ gridf, float* __restrict__ ys)
{
  const int p = blockIdx.x;
  float a = (float)(b0[0]+b1[0]+b2[0]+b3[0]+b4[0]+b5[0]+b6[0]+b7[0]);
  a += (float)eids[p] + xs[p*2048] + gridf[0];
  ys[p*512] = a;
}
"""

K2 = COMMON + r"""
extern "C" __global__ void __launch_bounds__(1024) kb2(
    const unsigned char* __restrict__ b0, const unsigned char* __restrict__ b1,
    const unsigned char* __restrict__ b2, const unsigned char* __restrict__ b3,
    const unsigned char* __restrict__ b4, const unsigned char* __restrict__ b5,
    const unsigned char* __restrict__ b6, const unsigned char* __restrict__ b7,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs, const float* __restrict__ gridf, float* __restrict__ ys)
{
  const int p = blockIdx.x;
  const unsigned int e = eids[p];
  const size_t off = (size_t)(e % SHARDN) * SLABB;
  const unsigned int s = e / SHARDN;
  const unsigned char* base = b0;
  if (s == 1) base = b1; else if (s == 2) base = b2; else if (s == 3) base = b3;
  else if (s == 4) base = b4; else if (s == 5) base = b5; else if (s == 6) base = b6;
  else if (s == 7) base = b7;
  float a = (float)(base + off)[0];
  a += xs[p*2048] + gridf[0];
  ys[p*512] = a;
}
"""

K3 = open(f"{BASE}/MM_P0_gx8e256nw32.cu").read().replace("gx8e256nw32", "kb3")

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    progs = {}
    for nm, src in (("kb1", K1), ("kb2", K2), ("kb3", K3)):
        open(f"{BASE}/MM_P0_{nm}.cu", "w").write(src)
        r = subprocess.run(f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={BASE}/MM_P0_{nm}.cubin {BASE}/MM_P0_{nm}.cu",
                           shell=True, capture_output=True, text=True, env=env)
        if r.returncode: print(r.stderr[-800:]); sys.exit(1)
        lib = open(f"{BASE}/MM_P0_{nm}.cubin", "rb").read()
        progs[nm] = NVProgram(dev, TinyELF(lib=lib, name=nm, target=dev.renderer.target, signature=tuple()))
        dev.synchronize()
        print(f"[built+loaded] {nm}", flush=True)
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    banks = [up(np.random.default_rng(100+s).integers(0, 256, 46<<20, dtype=np.uint8)) for s in range(8)]
    eids = up(np.arange(8, dtype=np.uint16))
    xs = up(np.random.default_rng(2).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32))
    gridf = up(np.zeros((512, 4), dtype=np.float32))
    ys = dev.allocator.alloc(88*512*4, BufferSpec())
    dev.synchronize()
    print("[bufs up]", flush=True)
    for nm in ("kb1", "kb2", "kb3"):
        progs[nm](*banks, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        mv = memoryview(bytearray(8*512*4)).cast("B")
        dev.allocator._copyout(mv, ys)
        y = np.frombuffer(mv, dtype=np.float32)
        print(f"[{nm}] CLEAN finite={np.isfinite(y).all()} y0={y[0]:.4f}", flush=True)
    print("[ALL KERNEL VARIANTS CLEAN]")

if __name__ == "__main__":
    main()
