# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
dev = Device["NV"]
D = "~/tinygrad-metal/pv3"
CD = "~/cdump100k"
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk_cu(path, name):
    os.system(f"cp {path} /tmp/bs.cu")
    return None
def bench_raw(src_path, name, grid, thr, shapes, n=3, inner=32):
    import subprocess
    r = subprocess.run(["~/.local/bin/nvcc", "-arch=sm_86", "-cubin", "-o", "~/tinygrad-metal/pv3/bs.cubin", src_path],
                       capture_output=True, env={**os.environ, "PATH": "~/.local/bin:/opt/homebrew/bin:/usr/bin:/bin", "DOCKER_HOST": "unix://<colima-socket>"})
    if r.returncode: print(name, "COMPILE FAIL"); return None
    p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/pv3/bs.cubin","rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
    args = []
    for (numel, kind) in shapes:
        if kind == "u": args.append(up(np.random.randint(0, 255, numel).astype(np.uint8)))
        else: args.append(up((np.random.standard_normal(numel)*0.3).astype(np.float32)))
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=(0,))
        p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
K = [
 ("r_3_16_64_128_2", (64,16,3), (128,1,1), [(6144,"f"),(40960,"f"),(163840,"u"),(16,"f"),(786432,"f")]),
 ("r_3_16_64_2",     (16,3,1),  (64,1,1),  [(48,"f"),(61440,"f"),(163840,"u")]),
 ("r_3_16_8_16_4_4_2_4",  (8,16,3), (16,4,1), [(6144,"f"),(112640,"f"),(163840,"u"),(786432,"f"),(384,"f"),(128,"f")]),
 ("r_3_16_16_16_2_4_2_4", (16,16,3),(16,2,1), [(6144,"f"),(112640,"f"),(163840,"u"),(786432,"f"),(384,"f"),(128,"f")]),
 ("E_2048_32_4_3",   (2048,1,1),(32,1,1), [(786432,"f"),(786432,"f")]),
]
tot = 0
for name, g, t, shapes in K:
    ms = bench_raw(f"{CD}/c_{name}.cu", name, g, t, shapes)
    if ms is not None:
        print(f"{name}: {ms*1000:.1f}us  x48blocks = {ms*48:.2f}ms/probe-if-once")
