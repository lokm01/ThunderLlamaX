#!/usr/bin/env python3
"""MM SESSION A -- dyn-smem probe (FAULT-ISOLATED STANDALONE; runs LAST).

No weights, no Rig7 -- a minimal GPU process: load dsmemp, patch its QMD
(shared_memory_size + 100KB carveout cfg), launch at 64KB then 96KB dynamic
smem, verify the readback against the host reference. A fault here can only
kill this process (the session results are already fsynced on disk).
Also tries a static 64KB nvcc compile (expected refused) if the cubin is
absent. Output: engine0/mm/mm_dsmem.json
"""
import os, sys, json, subprocess

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")
OUT = os.path.join(BASE, "engine0", "mm", "mm_dsmem.json")
NVCC_ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
                DOCKER_HOST="unix://~/.colima/default/docker.sock")

def fsync_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    dfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try: os.fsync(dfd)
    finally: os.close(dfd)

def main():
    import numpy as np
    from engine0 import dev
    from tinygrad.device import TinyELF, BufferSpec
    from tinygrad.runtime.ops_nv import NVProgram
    import tinygrad.runtime.ops_nv as ops_nv

    res = {"note": "probe patches QMD shared_memory_size (no cudaFuncSetAttribute on this path)"}
    # static 64KB compile attempt (documented; expected refused)
    try:
        src = os.path.join(BASE, "engine0", "mm", "MM_A_dsmem_static_try.cu")
        with open(src, "w") as f:
            f.write("extern \"C\" __global__ void sst(float* o){ __shared__ float a[16384]; "
                    "a[threadIdx.x]=threadIdx.x; o[threadIdx.x]=a[threadIdx.x]; }\n")
        r = subprocess.run(f"nvcc -arch=sm_86 -cubin -o /tmp/sst.cubin {src}",
                           shell=True, capture_output=True, text=True, env=NVCC_ENV)
        res["static_64k_nvcc"] = "COMPILED (unexpected)" if r.returncode == 0 else \
            "REFUSED: " + (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?")
    except Exception as e:
        res["static_64k_nvcc"] = f"probe error {e}"
    print(f"[ds] static 64KB nvcc: {res['static_64k_nvcc']}", flush=True)

    lib = open(os.path.join(BASE, "engine0", "mm", "MM_A_dsmemp.cubin"), "rb").read()
    INT_SIG = (None, 4, __import__("tinygrad").dtype.dtypes.int32, ())
    prg = NVProgram(dev, TinyELF(lib=lib, name="dsmemp",
                                  target=dev.renderer.target, signature=(INT_SIG,)))
    print(f"[ds] dsmemp loaded (static shmem_usage={prg.shmem_usage} B)", flush=True)

    def alloc(n):
        return dev.allocator.alloc(n, BufferSpec())
    def dn(b, n, dt=np.float32):
        mv = memoryview(bytearray(int(n) * np.dtype(dt).itemsize)).cast("B")
        dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dt).copy()

    for kb in (64, 96):
        words = kb * 1024 // 4
        try:
            ob = alloc(1024 * 4)
            dev.allocator._copyin(ob, memoryview(np.zeros(1024, dtype=np.uint32).tobytes()))
            sz = kb * 1024 + 1024
            prg.qmd.write(shared_memory_size=sz,
                          min_sm_config_shared_mem_size=100 * 1024 // 4096 + 1,
                          target_sm_config_shared_mem_size=100 * 1024 // 4096 + 1)
            prg(ob, global_size=(1, 1, 1), local_size=(1024, 1, 1),
                vals=(words,), wait=True)
            # host reference (uint32 wraparound domain; xor is order-independent)
            i32 = np.arange(words, dtype=np.uint64)
            sm = ((i32 * np.uint64(2654435761)) + np.uint64(7)) & np.uint64(0xFFFFFFFF)
            k2 = (i32 * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
            got = dn(ob, 1024, np.uint32)
            bad = 0
            for tid in range(1024):
                idx = np.arange(tid, words, 1024)
                ref = (sm[idx] + k2[idx] + np.uint64(tid)) & np.uint64(0xFFFFFFFF)
                if np.bitwise_xor.reduce(ref.astype(np.uint32)) != got[tid]:
                    bad += 1
            res[f"{kb}kb"] = {"verdict": "PASS" if bad == 0 else "WRONG-DATA",
                              "detail": f"{bad}/1024 threads mismatch"}
        except Exception as e:
            res[f"{kb}kb"] = {"verdict": "FAULT", "detail": str(e)[:300]}
        print(f"[ds] {kb}KB dynamic smem: {res[f'{kb}kb']}", flush=True)
        fsync_json(OUT, res)
    fsync_json(OUT, res)
    print(f"[ds] done -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
