#!/usr/bin/env python3
"""MM P34 dbg — the spka256 fault bisect: (a) dump the pointer args the kernel
actually receives (>8-arg envelope suspicion), (b) phase-gated early-return
variants. One GPU process; progress in ~/mm_p34_dbg.txt."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
PROG = os.path.expanduser("~/mm_p34_dbg.txt")

def rec(tag, s):
    with open(PROG, "a") as f: f.write(f"{tag} {s}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[DBG] {tag} {s}", flush=True)

def done(tag):
    return os.path.exists(PROG) and any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

DBG_CU = r'''
extern "C" __global__ void __launch_bounds__(256) argdump10(
    const float* __restrict__ a0, const float* __restrict__ a1,
    const float* __restrict__ a2, const float* __restrict__ a3,
    const float* __restrict__ a4, signed char* __restrict__ a5,
    float* __restrict__ a6, signed char* __restrict__ a7,
    float* __restrict__ a8, const int* __restrict__ a9)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long* o = (unsigned long long*)a6;   // reuse a6 as the dump target
    o[0] = (unsigned long long)(size_t)a0;
    o[1] = (unsigned long long)(size_t)a1;
    o[2] = (unsigned long long)(size_t)a2;
    o[3] = (unsigned long long)(size_t)a3;
    o[4] = (unsigned long long)(size_t)a4;
    o[5] = (unsigned long long)(size_t)a5;
    o[6] = 0xdeadbeef00ULL + (unsigned long long)(size_t)a7;
    o[7] = (unsigned long long)(size_t)a8;
    o[8] = (unsigned long long)(size_t)a9;
    // no a9 deref (fault-bisect: pointers only)
  }
}
'''

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    import subprocess
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    src = f"{BASE}/MM_P34_dbg_argdump.cu"
    cb = f"{BASE}/MM_P34_dbg_argdump.cubin"
    open(src, "w").write(DBG_CU)
    r = subprocess.run(f"nvcc -arch=sm_86 -cubin --output-file={cb} {src}", shell=True, capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
    lib = open(cb, "rb").read()
    P = NVProgram(dev, TinyELF(lib=lib, name="argdump10", target=dev.renderer.target, signature=tuple()))
    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def dn(b, n, dtype=np.uint64):
        mv = memoryview(bytearray(int(n)*8)).cast("B")
        dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dtype).copy()
    bufs = [up(np.zeros(1024, dtype=np.float32)) for _ in range(6)] + \
           [up(np.zeros((2, 1024, 256), dtype=np.int8)), up(np.zeros(64, dtype=np.float32)),
            up(np.zeros((2, 1024, 256), dtype=np.int8)), up(np.array([7], dtype=np.int32))]
    vas = [b.va_addr for b in bufs]
    P(*bufs, global_size=(1,1,1), local_size=(256,1,1), wait=True)
    got = dn(bufs[6], 10)   # a6 was reused as the dump target
    ok = True
    for i in range(10):
        expect = vas[i]
        g = int(got[i]) if i != 6 else int(got[i]) - 0xdeadbeef00
        match = (g == expect)
        if i == 9: pass   # pointer only (deref variant faults)
        if not match: ok = False
        rec(f"A{i}", f"expect {expect:#x} got {g:#x} {'OK' if match else 'MISMATCH'}")
    rec("ADUMP", "ALL ARGS OK" if ok else "ARG MISMATCH FOUND")
    # then the real spka256 launch (minimal)
    if not done("SPKA"):
        cbk = f"{BASE}/MM_P34_spka256.cubin"
        S = NVProgram(dev, TinyELF(lib=open(cbk, "rb").read(), name="spka256", target=dev.renderer.target, signature=tuple()))
        cos = np.ascontiguousarray(np.cos(np.arange(1024*32, dtype=np.float32).reshape(1024, 32)))
        POS = up(np.array([7], dtype=np.int32))
        r2 = S(up(np.zeros(512, dtype=np.float32)), up(np.zeros(512, dtype=np.float32)),
               up(np.ones(256, dtype=np.float32)), up(cos), up(cos),
               bufs[6], bufs[7], bufs[6], bufs[7], POS,
               global_size=(2,1,1), local_size=(256,1,1), wait=True)
        rec("SPKA", f"CLEAN ({r2})")

if __name__ == "__main__":
    main()
