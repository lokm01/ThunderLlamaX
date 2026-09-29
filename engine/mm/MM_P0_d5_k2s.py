#!/usr/bin/env python3
"""MM P0 D5 — k2s GDN scan bench at qwen35moe dims [32 heads][128 v][128 k] fp32.

Per-T cubins (the production k2s1/k2s3/... convention; -DTMAX=T), 30 sequential
layer-launches each (grid 32 CTAs x 256 thr), T = 1..11; per-layer us; the
state-traffic GB/s; comparison vs the production Qwen3.8 class ([48][128][128],
48 blocks — measured per the R-banks) and the analytic HMMA-ize price if the
t-chain is latency-bound.
"""
import os, sys, time, subprocess
import numpy as np

os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
NVH, TMAX = 32, 11

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram

    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    progs = {}
    for T in range(1, TMAX+1):
        cb = f"{BASE}/MM_P0_k2s36_t{T}.cubin"
        r = subprocess.run(f"nvcc -arch=sm_86 -cubin -DTMAX={T} --output-file={cb} {BASE}/MM_P0_k2s36.cu",
                           shell=True, capture_output=True, text=True, env=env)
        if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
        lib = open(cb, "rb").read()
        progs[T] = NVProgram(dev, TinyELF(lib=lib, name=f"mm_k2s36_t{T}", target=dev.renderer.target, signature=tuple()))
    print("[built+loaded] 11 per-T cubins", flush=True)
    r = subprocess.run(f"cuobjdump -res-usage {BASE}/MM_P0_k2s36_t11.cubin", shell=True, capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if "STACK" in l.upper() or "SPILL" in l.upper(): print("[audit t11]", l.strip())

    keep = []
    def up(arr):
        a = np.ascontiguousarray(arr); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    rng = np.random.default_rng(3)
    qkv = up((rng.standard_normal(TMAX*NVH*384)*0.1).astype(np.float16))
    abdt = up((rng.standard_normal(TMAX*NVH*3)*0.1).astype(np.float32))
    ssm_a = up((rng.uniform(0.5, 1.5, NVH)).astype(np.float32))
    dtb = up((rng.uniform(-0.1, 0.1, NVH)).astype(np.float32))
    rec_in = up((rng.standard_normal(NVH*16384)*0.05).astype(np.float32))
    rec_out = dev.allocator.alloc(TMAX*NVH*16384*4, BufferSpec())
    core = dev.allocator.alloc(TMAX*NVH*128*4, BufferSpec())
    dev.synchronize()

    print("\n== D5 mm_k2s36 @ [32][128][128] fp32 (2 MiB/layer state), 30 layers ==")
    print(f"  {'T':>3s} {'us/layer':>9s} {'ms/30L':>8s} {'stateGB/s':>9s} {'us/layer/T':>10s}")
    rows = []
    for T in sorted(progs):
        P = progs[T]
        P(qkv, abdt, ssm_a, dtb, rec_in, rec_out, core, global_size=(NVH,1,1), local_size=(256,1,1), wait=True)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(30):
                P(qkv, abdt, ssm_a, dtb, rec_in, rec_out, core, global_size=(NVH,1,1), local_size=(256,1,1), wait=False)
            dev.synchronize()
            ts.append((time.perf_counter()-t0)/30*1e6)
        us = min(ts)
        st_b = NVH*16384*4*(T+1)      # T slab writes + reads of prior slab each step (+1 in)
        rows.append((T, us))
        print(f"  {T:3d} {us:9.1f} {us*30/1e3:8.3f} {st_b/(us*1e-6)/1e9:9.1f} {us/T:10.1f}")
    # sanity: outputs finite (nothing elided / no NaN)
    mv = memoryview(bytearray(TMAX*NVH*128*4)).cast("B")
    dev.allocator._copyout(mv, core)
    c = np.frombuffer(mv, dtype=np.float32)
    print(f"  core finite: {np.isfinite(c).all()} (|max| {np.abs(c).max():.3f})")
    # today's production reference (Qwen3.8 k2s @T=11): ~from the R5 phase split
    # probe 62.6ms K=7 across 48 blocks incl. attn+FFN; the k2s pool there was
    # measured ~1.1-1.5ms/layer @T=11 (48-head 3MiB state). Report ratio:
    t11 = dict(rows)[11]
    print(f"\n  t11 per-layer {t11:.1f} us -> 30-layer probe adds {t11*30/1e3:.2f} ms per cycle")
    print(f"  per-t increment (t2-t1): {dict(rows)[2]-dict(rows)[1]:.1f} us (the sequential-chain slope)")

if __name__ == "__main__":
    main()
