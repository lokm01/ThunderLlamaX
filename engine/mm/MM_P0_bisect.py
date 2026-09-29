#!/usr/bin/env python3
"""MM_P0 D3 fault bisect: one variant per boot (faulted dext state is sticky).
V1: 8GB bank alloc, NO fill, launch.        V2: + 127-chunk fill.
V3: + structured overlay + real gridf.      V4: harness-exact (defaults).
Usage: MM_P0_bisect.py V1|V2|V3|V4"""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB = 1458176
NBANK = int(os.environ.get("MM_NBANK", "5600"))

def main():
    V = sys.argv[1]
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    lib = open(f"{BASE}/MM_P0_gx8e256nw32.cubin", "rb").read()
    gx = NVProgram(dev, TinyELF(lib=lib, name="MM_P0_gx8e256nw32", target=dev.renderer.target, signature=tuple()))
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    bank = dev.allocator.alloc(SLAB*NBANK, BufferSpec()) if V != "V0" else up(
        np.random.default_rng(1).integers(0, 256, SLAB*32, dtype=np.uint8))
    if V in ("V2", "V3", "V4"):
        rng = np.random.default_rng(7)
        CH = 64 << 20
        for off in range(0, SLAB*NBANK, CH):
            n = min(CH, SLAB*NBANK - off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl = up(np.arange(NBANK, dtype=np.uint64)*np.uint64(SLAB))
    if V in ("V3", "V4"):
        def iq3s_grid_f32():
            from tinygrad.runtime.autogen.ggml_common import iq3s_grid
            v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
            return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()
        gridf = up(iq3s_grid_f32())
        for e in range(8):
            r2 = np.random.default_rng(1000+e)
            rows = r2.integers(0, 256, 450560, dtype=np.uint8).reshape(512, 880).copy()
            d16 = r2.integers(0x3800, 0x4000, 512).astype(np.uint16) | (r2.integers(0, 2, 512).astype(np.uint16) << 15)
            rows[:, 0:2] = np.ascontiguousarray(d16).view(np.uint8).reshape(512, 2)
            keep.append(rows)
            dev.allocator._copyin(bank.offset(offset=e*SLAB, size=450560), memoryview(rows.tobytes()))
    else:
        gridf = up(np.zeros((512, 4), dtype=np.float32))
    eids = up(np.arange(8, dtype=np.uint16))
    xs = up(np.random.default_rng(2).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32))
    ys = dev.allocator.alloc(88*512*4, BufferSpec())
    dev.synchronize()
    print(f"[{V}] buffers up; launching gx...", flush=True)
    gx(bank, ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    mv = memoryview(bytearray(8*512*4)).cast("B")
    dev.allocator._copyout(mv, ys)
    y = np.frombuffer(mv, dtype=np.float32)
    print(f"[{V}] CLEAN LAUNCH; finite={np.isfinite(y).all()} sample={y[:3]}", flush=True)

if __name__ == "__main__":
    main()
