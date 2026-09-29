#!/usr/bin/env python3
"""MM_P0 D3 bisect: does the gx cubin fault on a CLEAN machine, and is
-fmad=false the cause? Runs each cubin variant once, small buffers."""
import os, sys, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"

def build(name, extra):
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    r = subprocess.run(f"nvcc -arch=sm_86 -cubin {extra} --output-file={BASE}/{name}.cubin {BASE}/MM_P0_gxsrc.cu -DKNAME={name}",
                       shell=True, capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-800:]); sys.exit(1)
    print(f"[built] {name}")

def main():
    variant = sys.argv[1]   # "fmad" or "nofmad"
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    build(variant, "-fmad=false" if variant == "fmad" else "")
    lib = open(f"{BASE}/{variant}.cubin", "rb").read()
    P = NVProgram(dev, TinyELF(lib=lib, name=variant, target=dev.renderer.target, signature=tuple()))
    print("[loaded]")
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    NB = 32
    bank = up(np.random.default_rng(1).integers(0, 256, 1458176*NB, dtype=np.uint8))
    ptbl = up(np.arange(NB, dtype=np.uint64)*np.uint64(1458176))
    eids = up(np.arange(8, dtype=np.uint16))
    xs = up(np.random.default_rng(2).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32))
    gridf = up(np.zeros((512, 4), dtype=np.float32))
    ys = dev.allocator.alloc(88*512*4, BufferSpec())
    dev.synchronize()
    print("[buffers up] launching...")
    P(bank, ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    mv = memoryview(bytearray(8*512*4)).cast("B")
    dev.allocator._copyout(mv, ys)
    y = np.frombuffer(mv, dtype=np.float32)
    print(f"[{variant}] LAUNCHED CLEAN; ys finite={np.isfinite(y).all()} sample={y[:4]}")

if __name__ == "__main__":
    main()
