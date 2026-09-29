#!/usr/bin/env python3
"""MM_P0 D3 fault bisect #2 — staged, one boot, stops at first fault:
T1: load ALL 5 progs, launch gx over 8x46MB banks (12-arg, shard-select)
T2: alloc 8x1.16GiB banks, fill, relaunch
T3: 88-pair scattered launch
T4: tiny router+mut launches"""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, SHARD, NSHARD = 1458176, 850, 8

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    def prog(n):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    gx = prog("MM_P0_gx8e256nw32"); dev.synchronize()
    print("[T0] gx loaded+sync", flush=True)
    mut = prog("MM_P0_mut"); dev.synchronize()
    print("[T0] mut loaded+sync", flush=True)
    rt = prog("MM_P0_rt8e256poc"); dev.synchronize()
    print("[T0] rt loaded+sync", flush=True)
    hrot = prog("MM_P0_hrot"); dev.synchronize()
    print("[T0] hrot loaded+sync", flush=True)
    k2s = prog("MM_P0_k2s36_t11") if os.path.exists(f"{BASE}/MM_P0_k2s36_t11.cubin") else prog("MM_P0_k2s36"); dev.synchronize()
    print("[T0] k2s loaded+sync", flush=True)
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def iq3s():
        from tinygrad.runtime.autogen.ggml_common import iq3s_grid
        v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
        return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()
    gridf = up(iq3s()); dev.synchronize(); print("[T0] gridf up", flush=True)
    xs = up(np.random.default_rng(2).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32)); dev.synchronize(); print("[T0] xs up", flush=True)
    eids = up(np.arange(8, dtype=np.uint16)); dev.synchronize(); print("[T0] eids up", flush=True)
    ys = dev.allocator.alloc(88*512*4, BufferSpec()); dev.synchronize()
    print("[T0] small buffers up", flush=True)

    # T1a: ONE 46MB bank, passed as ALL 8 shard args (12-arg launch, no new allocs)
    small0 = up(np.random.default_rng(100).integers(0, 256, 46<<20, dtype=np.uint8))
    dev.synchronize(); print("[T1a-pre] one 46MB bank up", flush=True)
    gx(small0, small0, small0, small0, small0, small0, small0, small0, eids, xs, gridf, ys,
       global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[T1a] 12-arg gx over one bank x8: CLEAN", flush=True)
    # T1b: 8 DISTINCT 46MB banks
    small = [up(np.random.default_rng(100+s).integers(0, 256, 46<<20, dtype=np.uint8)) for s in range(8)]
    dev.synchronize(); print("[T1b-pre] 8x46MB banks up", flush=True)
    gx(*small, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[T1b] 8 distinct banks: CLEAN", flush=True)

    # T2: 8x1.16GiB banks, chunk-filled
    rng = np.random.default_rng(7)
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    CH = 64 << 20
    for bk in banks:
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD-off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bk.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    dev.synchronize()
    gx(*banks, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[T2] 8x1.16GiB banks: CLEAN", flush=True)

    # T3: 88-pair scattered
    scat = np.concatenate([np.random.default_rng(100+i).choice(256, 8, replace=False) for i in range(11)]).astype(np.uint16)
    eb = up(scat)
    gx(*banks, eb, xs, gridf, ys, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
    print("[T3] 88-pair scattered: CLEAN", flush=True)

    # T4: mut + rt launches
    mut(eb, global_size=(1,1,1), local_size=(128,1,1), wait=True)
    print("[T4a] mm_eidmut: CLEAN", flush=True)
    W = up(np.random.default_rng(5).uniform(-0.05, 0.05, (256, 2048)).astype(np.float32))
    h = up(np.random.default_rng(6).uniform(-0.5, 0.5, (88, 2048)).astype(np.float32))
    e2 = up(np.zeros(704, dtype=np.uint16))
    rt(W, h, e2, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
    print("[T4b] rt8e256poc: CLEAN", flush=True)
    hrot(h, global_size=(1,1,1), local_size=(128,1,1), wait=True)
    print("[T4c] mm_hrot: CLEAN — ALL STAGES PASS", flush=True)

if __name__ == "__main__":
    main()
