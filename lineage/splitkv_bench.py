# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Split-KV standalone benchmark: L=32768 vs tensor attention timing."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
import subprocess
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://<colima-socket>")
for n in ("splitkv_k1", "splitkv_k2"):
    subprocess.run(["nvcc", "-arch=sm_86", "-cubin", f"--output-file=~/tinygrad-metal/splitkv/{n}.cubin",
                    f"~/tinygrad-metal/splitkv/{n}.cu"], check=True, env=env, capture_output=True)
dev = Device["NV"]
_pi = ("v", 0, dtypes.int32, ())
k1 = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/splitkv_k1.cubin","rb").read(),
      name="splitkv_k1", target=dev.renderer.target, signature=(_pi,)*4))
k2 = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/splitkv_k2.cubin","rb").read(),
      name="splitkv_k2", target=dev.renderer.target, signature=(_pi,)))

Hkv, Rep, T, KD, VD = 8, 2, 3, 128, 128
rng = np.random.default_rng(5)
for L, POS in ((100352, 97810),):
    q = rng.normal(0,.3,(48,KD)).astype(np.float16)
    kc = rng.normal(0,.3,(L,Hkv,KD)).astype(np.float16)
    vc = rng.normal(0,.3,(L,Hkv,VD)).astype(np.float16)
    def up(a): t = Tensor(a).contiguous().realize(); return t.uop.buf_uop.buffer._bufs["NV"]
    q_b, k_b, v_b = up(q), up(kc), up(vc)
    kv_bytes = L * Hkv * (KD+VD) * 2
    for S in (6, 8, 10, 12):
        slots = S*4
        oacc = Tensor.zeros(slots,16,T,VD).contiguous().realize()
        lse = Tensor.full((slots,16,T), float("-inf")).contiguous().realize()
        out = Tensor.zeros(16,T,VD,dtype=dtypes.float16).contiguous().realize()
        ob, lb, xb = (t.uop.buf_uop.buffer._bufs["NV"] for t in (oacc,lse,out))
        n_chunk = (POS+T+S-1)//S
        for _ in range(3):
            k1(q_b,k_b,v_b,ob,lb, global_size=(Hkv*S,1,1), local_size=(128,1,1), vals=(POS,T,S,n_chunk), wait=True)
            k2(ob,lb,xb, global_size=(48,1,1), local_size=(128,1,1), vals=(slots,), wait=True)
        # realistic: 16 layer-pairs back-to-back, ONE sync (in-graph pattern)
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            for _l in range(16):
                k1(q_b,k_b,v_b,ob,lb, global_size=(Hkv*S,1,1), local_size=(128,1,1), vals=(POS,T,S,n_chunk))
                k2(ob,lb,xb, global_size=(48,1,1), local_size=(128,1,1), vals=(slots,))
            Device["NV"].synchronize()
        dt = (time.perf_counter()-t0)/N/16*1e3
        # 16 layers per probe: x16 / but report per-layer too
        print(f"L={L} S={S}: {dt:.3f} ms/layer-pair -> x16 = {dt*16:.1f} ms/probe | {kv_bytes/dt/1e9:.0f} GB/s", flush=True)
print("DONE")
