#!/usr/bin/env python3
"""MM_P0 D3 bisect #6 — THE NAME TEST. Same source, same buffers, two cubins:
symbol 'gx8e256nw32' vs symbol 'kbv'. N1 launches the gx-named one; if clean,
N2 launches kbv-named. A fault at N1 = the name (or its TinyELF name string)."""
import os, sys, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, GATE_B, SHARD, NSHARD = 1458176, 450560, 256, 20

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    src = open(f"{BASE}/MM_P0_gx8e256nw32.cu").read()
    progs = {}
    for sym, fn in (("gx8e256nw32", "gxname"), ("kbv", "kbv2")):
        open(f"{BASE}/MM_P0_{fn}.cu", "w").write(src.replace("gx8e256nw32", sym))
        r = subprocess.run(f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={BASE}/MM_P0_{fn}.cubin {BASE}/MM_P0_{fn}.cu",
                           shell=True, capture_output=True, text=True, env=env)
        if r.returncode: print(r.stderr[-800:]); sys.exit(1)
        lib = open(f"{BASE}/MM_P0_{fn}.cubin", "rb").read()
        progs[sym] = NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target, signature=tuple()))
        dev.synchronize()
        print(f"[loaded] {sym}", flush=True)
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
    rng = np.random.default_rng(7)
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    CH = 64 << 20
    for bank in banks:
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD-off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl = up(np.array([banks[e//SHARD].va_addr + (e % SHARD)*SLAB for e in range(SHARD*NSHARD)], dtype=np.uint64))
    dev.synchronize(); print("[bufs up: 20x373MB + ptbl]", flush=True)
    progs["gx8e256nw32"](ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[N1] gx launch with 2 progs loaded: CLEAN", flush=True)
    # stage-load the other four, launching gx after each
    def load(nm):
        lib = open(f"{BASE}/{nm}.cubin", "rb").read()
        p = NVProgram(dev, TinyELF(lib=lib, name=nm, target=dev.renderer.target, signature=tuple()))
        dev.synchronize(); print(f"[loaded] {nm}", flush=True); return p
    for nm in ("MM_P0_mut", "MM_P0_rt8e256poc", "MM_P0_hrot", "MM_P0_k2s36"):
        load(nm)
        progs["gx8e256nw32"](ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        print(f"[N2] gx launch after loading {nm}: CLEAN", flush=True)
    print("[ALL CLEAN — program set is not it either]")

if __name__ == "__main__":
    main()
