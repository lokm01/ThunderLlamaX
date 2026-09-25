# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
dev = Device["NV"]
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
# pv with FEW args first: strip to out + V + gates (3 bufs + dummy int)
src = open("~/tinygrad-metal/splitkv/splitkv_pv.cu").read()
# rebuild a 3-arg version inline
small = '''#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pv_small(
    float* __restrict__ data0, const __half* __restrict__ data4,
    float* __restrict__ data11, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  if (r == 0 && g == 0 && d == 0) data0[0] = 777.0f;
  const __half* V_g = data4 + ((size_t)(g / 6) << 21) + 8388608;
  float acc = 0.0f;
  for (int p = 0; p < 64; ++p) acc += __half2float(V_g[((size_t)p << 8) + d]);
  const float gate = data11[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  data0[d + ((size_t)g << 8) + (size_t)r * 6144] = acc * (1.0f/(1.0f+exp2f(-gate*1.4426950216293334f)));
}
'''
open("~/tinygrad-metal/splitkv/pv_small.cu","w").write(small)
import subprocess
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin")+":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://<colima-socket>")
subprocess.run(["nvcc","-arch=sm_86","-cubin","--output-file=~/tinygrad-metal/splitkv/pv_small.cubin","~/tinygrad-metal/splitkv/pv_small.cu"],check=True,env=env, capture_output=True)
bV = up(np.ones(16777216, np.float16))
bG = up(np.full(36864, 10.0, np.float32))
outT = Tensor.zeros(18432).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_small.cubin","rb").read(), name="pv_small",
      target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
p(ob, bV, bG, global_size=(3,24,2), local_size=(128,1,1), vals=(0,), wait=True)
o = outT.numpy()
print(f"[pv_small] o[0]={o[0]:.4f} (sentinel 777 or 64*sigmoid) max={o.max():.4f}", flush=True)
