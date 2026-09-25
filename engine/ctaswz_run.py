# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time, heapq
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVProgram

dev = Device["NV"]
NSM = dev.num_gpcs * dev.num_tpc_per_gpc * dev.num_sm_per_tpc
GRID = int(os.getenv("CTASWZ_GRID_MULT", "4")) * NSM
SPIN = int(os.getenv("CTASWZ_SPIN", "3000000"))
print(f"[ctaswz] NSM={NSM} warps/SM={dev.max_warps_per_sm} grid={GRID} spin={SPIN}cyc", flush=True)
lib = open("~/tinygrad-metal/engine0/ctaswz.cubin","rb").read()
buf = dev.allocator.alloc(GRID*4*8, BufferSpec())
from collections import Counter
for kname in os.getenv("CTASWZ_KERNELS","ctaswz,ctaswz8k,ctaswz32k").split(","):
  kname = kname.strip()
  prg = NVProgram(dev, TinyELF(lib=lib, name=kname, target=dev.renderer.target, signature=tuple()))
  slots = prg.qmd.read("free_cta_slots_empty_sm")
  minc = prg.qmd.read("min_sm_config_shared_mem_size")
  print(f"[ctaswz] {kname}: regs={prg.regs_usage} smem={prg.shmem_usage} maxthr={prg.max_threads} "
        f"qmd.free_cta_slots_empty_sm={slots} qmd.min_sm_cfg={minc}", flush=True)
  dev.allocator._copyin(buf, memoryview(b"\x00"*(GRID*4*8)).cast("B")); dev.synchronize()
  t0 = time.time()
  prg(buf, global_size=(GRID,1,1), local_size=(128,1,1), wait=True)
  dt = (time.time()-t0)*1e3
  mv = memoryview(bytearray(GRID*4*8)); dev.allocator._copyout(mv, buf)
  a = np.frombuffer(mv, dtype=np.uint64).reshape(GRID,4)
  smids, ts, te = a[:,0], a[:,1], a[:,2]
  n_unfilled=int(((te<=0)|(ts<=0)).sum()); print(f"  unfilled={n_unfilled}/{GRID}", flush=True)
  if n_unfilled: print("  first rows:", a[:4].tolist(), a[-2:].tolist(), flush=True)
  ok = (te>ts)&(ts>0); smids,ts,te = smids[ok],ts[ok],te[ok]
  maxres, per_sm = 0, {}
  for s in np.unique(smids):
    iv = sorted(zip(ts[smids==s].tolist(), te[smids==s].tolist()))
    h, best = [], 0
    for x,y in iv:
      while h and h[0] <= x: heapq.heappop(h)
      heapq.heappush(h, y); best = max(best, len(h))
    per_sm[s] = best; maxres = max(maxres, best)
  waves = maxres*1.0 and (GRID/maxres + NSM-1) and None
  print(f"[ctaswz] {kname}: wall={dt:.1f}ms MAX_CO_RESIDENT_CTAS_PER_SM={maxres} "
        f"perSM_hist={dict(sorted(Counter(per_sm.values()).items()))} distinct_sms={len(per_sm)}", flush=True)
print("[ctaswz done]", flush=True)
