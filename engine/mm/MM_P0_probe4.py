#!/usr/bin/env python3
"""MM_P0 D3 fault bisect #4 — the hand-NVProgram ARG SIZE ladder.
kb3 (= the real gx kernel) launched over 8 distinct banks of growing size:
100MB, 200MB, 400MB, 800MB, 1.16GB, 2GB, 3.66GB (8 banks each stage).
The first faulting stage brackets the dext's per-arg (or total) cap.
Also: real iq3s gridf (not zeros) from the start."""
import os, sys, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, SHARDN = 1458176, 850

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    src = open(f"{BASE}/MM_P0_gx8e256nw32.cu").read().replace("gx8e256nw32", "kb3")
    open(f"{BASE}/MM_P0_kb3.cu", "w").write(src)
    r = subprocess.run(f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={BASE}/MM_P0_kb3.cubin {BASE}/MM_P0_kb3.cu",
                       shell=True, capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-800:]); sys.exit(1)
    lib = open(f"{BASE}/MM_P0_kb3.cubin", "rb").read()
    kb3 = NVProgram(dev, TinyELF(lib=lib, name="kb3", target=dev.renderer.target, signature=tuple()))
    dev.synchronize(); print("[kb3 loaded]", flush=True)
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    gridf = up(v.view(np.int8).astype(np.float32).reshape(512, 4).copy())
    eids = up(np.arange(8, dtype=np.uint16))
    xs = up(np.random.default_rng(2).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32))
    ys = dev.allocator.alloc(88*512*4, BufferSpec())
    dev.synchronize(); print("[bufs up, REAL gridf]", flush=True)

    rng = np.random.default_rng(7)
    for mb in (100, 200, 400, 800, 1188, 2048, 3660):
        nb = (mb << 20) // 8
        banks = []
        for s in range(8):
            b = dev.allocator.alloc(nb, BufferSpec())
            a = rng.integers(0, 256, nb, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(b, memoryview(a.data).cast("B"))
            banks.append(b)
        dev.synchronize()
        kb3(*banks, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        # free the banks (LRU keeps them; force release to avoid VRAM creep)
        for b in banks: dev.allocator.free(b, b.size, BufferSpec())
        print(f"[{mb}MB x8 banks] CLEAN", flush=True)
    print("[SIZE LADDER ALL CLEAN — size is NOT the trigger]")

if __name__ == "__main__":
    main()
