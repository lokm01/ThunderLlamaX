#!/usr/bin/env python3
"""MM_P0 D3 bisect #7 — morph probe6 (clean) into the harness (fault) stage by
stage, one boot, stop at first fault:
M1: build ALL 5 cubins first (build_all verbatim), load gx only, launch
M2: + load all 5 back-to-back (no launches between), launch
M3: + harness bank fill (structured rows embedded in banks[0] chunk 0) + ptbl
M4: + harness gx_ref numpy precompute (y_expect) before launch
M5: + launch twice (det-x2 loop shape)"""
import os, sys, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, GATE_B, SHARD, NSHARD, NPAIR = 1458176, 450560, 256, 20, 88

def sh(cmd, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

def build_all():
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    for src, extra in (("MM_P0_gx8e256nw32", "-fmad=false"), ("MM_P0_mut", ""),
                       ("MM_P0_rt8e256poc", "-fmad=false"), ("MM_P0_hrot", ""),
                       ("MM_P0_k2s36", "")):
        cu, cb = f"{BASE}/{src}.cu", f"{BASE}/{src}.cubin"
        r = sh(f"nvcc -arch=sm_86 -cubin {extra} --output-file={cb} {cu}", env=env)
        if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
        print(f"[built] {src}.cubin", flush=True)
    for src in ("MM_P0_gx8e256nw32", "MM_P0_rt8e256poc", "MM_P0_k2s36", "MM_P0_mut", "MM_P0_hrot"):
        r = sh(f"cuobjdump -res-usage {BASE}/{src}.cubin")
        lines = [l.strip() for l in r.stdout.splitlines() if "STACK" in l.upper() or "SPILL" in l.upper()]
        print(f"[audit] {src}: " + (" | ".join(lines) if lines else "(clean)"), flush=True)

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

def gx_ref(rows, x, gridf):
    lane = np.arange(32, dtype=np.int64); koff = lane << 3
    partial = np.zeros((512, 32), dtype=np.float32)
    for b in range(8):
        blk = rows[:, b*110:(b+1)*110]
        d = np.ascontiguousarray(blk[:, 0:2]).view(np.float16).astype(np.float32)[:, 0]
        sraw = lane >> 2
        nib = (np.take(blk, 106 + (sraw >> 1), axis=1) >> ((sraw & 1) << 2)) & 0xF
        sc = 1.0 + 2.0*nib.astype(np.float32)
        sg = np.take(blk, 74 + lane, axis=1)
        qlo = np.take(blk, 2 + 2*lane, axis=1).astype(np.uint16)
        qhi = np.take(blk, 3 + 2*lane, axis=1).astype(np.uint16)
        q16 = qlo | (qhi << 8)
        g0i, g1i = lane*2, lane*2 + 1
        bit0 = (np.take(blk, 66 + (g0i >> 3), axis=1) >> (g0i & 7)) & 1
        bit1 = (np.take(blk, 66 + (g1i >> 3), axis=1) >> (g1i & 7)) & 1
        qb0 = (q16 & 0xFF).astype(np.int64) + (bit0.astype(np.int64) << 8)
        qb1 = (q16 >> 8).astype(np.int64) + (bit1.astype(np.int64) << 8)
        t1 = d[:, None] * sc
        w = np.empty((512, 32, 8), dtype=np.float32)
        w[:, :, 0:4] = t1[:, :, None] * gridf[qb0]
        w[:, :, 4:8] = t1[:, :, None] * gridf[qb1]
        for j in range(8):
            wj = w[:, :, j].copy()
            m = (sg & (1 << j)) != 0
            wj[m] = -wj[m]
            partial = partial + (wj * x[koff + j][None, :]).astype(np.float32)
    p = partial
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return np.ascontiguousarray(p[:, 0])

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    build_all()
    def prog(n):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    gx = prog("MM_P0_gx8e256nw32")
    dev.synchronize(); print("[M1] gx loaded (single)", flush=True)
    keep = []
    def up(arr):
        a = np.ascontiguousarray(arr); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    gridf_np = iq3s_grid_f32(); gridf = up(gridf_np)
    xs_np = (np.random.default_rng(11).uniform(-0.5, 0.5, (NPAIR, 2048))).astype(np.float32)
    xs = up(xs_np)
    eids = up(np.arange(8, dtype=np.uint16))
    ys = dev.allocator.alloc(NPAIR*512*4, BufferSpec())
    rng = np.random.default_rng(7)
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    ref_rows = {}
    CH = 64 << 20
    for si, bank in enumerate(banks):
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD - off)
            a = rng.integers(0, 256, n, dtype=np.uint8)
            if si == 0 and off == 0:
                a = a.reshape(-1)
                for e in range(8):
                    r2 = np.random.default_rng(1000+e)
                    rows = r2.integers(0, 256, GATE_B, dtype=np.uint8).reshape(512, 880).copy()
                    dexp = r2.integers(0x3800, 0x4000, 512).astype(np.uint16)
                    dsign = (r2.integers(0, 2, 512).astype(np.uint16) << 15)
                    d16 = dexp | dsign
                    rows[:, 0:2] = np.ascontiguousarray(d16).view(np.uint8).reshape(512, 2)
                    ref_rows[e] = rows
                    a[e*SLAB:e*SLAB+GATE_B] = rows.reshape(-1)
                a = a.reshape(n)
            keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl = up(np.array([banks[e//SHARD].va_addr + (e % SHARD)*SLAB for e in range(SHARD*NSHARD)], dtype=np.uint64))
    dev.synchronize(); print("[M3] harness bank fill + ptbl done", flush=True)
    gx(ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[M1-launch] CLEAN (build5 + load1 + harness buffers)", flush=True)
    # M2: load the rest back-to-back, launch
    mut = prog("MM_P0_mut"); rt = prog("MM_P0_rt8e256poc"); hrot = prog("MM_P0_hrot"); k2s = prog("MM_P0_k2s36")
    dev.synchronize(); print("[M2] all 5 loaded back-to-back", flush=True)
    gx(ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    print("[M2-launch] CLEAN", flush=True)
    # M4: gx_ref precompute then launch
    y_exp = np.stack([gx_ref(ref_rows[e], xs_np[e], gridf_np) for e in range(8)])
    gx(ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    mv = memoryview(bytearray(8*512*4)).cast("B")
    dev.allocator._copyout(mv, ys)
    y0 = np.frombuffer(mv, dtype=np.float32).reshape(8, 512)
    print(f"[M4] CLEAN; vs-ref bit-exact: {np.array_equal(y0, y_exp)}", flush=True)

if __name__ == "__main__":
    main()
