# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c: bench-only companion to r7d_test.py (corr already proven BIT-IDENTICAL).
Dedicated correctly-sized output buffers per kernel (the reuse-OOB lesson)."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import dev, iq3_grid_f32
from engine0 import Bufs
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)

def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

def main():
  P = Bufs()
  rng = np.random.default_rng(0)
  P.up("pk_fg", np.load(f"{BASE}/packed/fg0.npy"))
  P.up("pk_fu", np.load(f"{BASE}/packed/fu0.npy"))
  P.up("pk_fd", np.load(f"{BASE}/packed/fd0.npy"))
  P.up("r7_fg", np.load(f"{BASE}/packed7/fg0.npy"))
  P.up("r7_fu", np.load(f"{BASE}/packed7/fu0.npy"))
  P.up("r7_fd", np.load(f"{BASE}/packed7/fd0.npy"))
  P.up("gridf", iq3_grid_f32())
  d = P.d
  P.up("x1", (rng.standard_normal(5120) * 0.10).astype(np.float16))
  P.up("x3", (rng.standard_normal(3 * 5120) * 0.10).astype(np.float16))
  P.up("x8", (rng.standard_normal(8 * 5120) * 0.10).astype(np.float16))
  P.up("gact1", (rng.standard_normal(17408) * 0.10).astype(np.float16))
  P.up("gact3", (rng.standard_normal(3 * 17408) * 0.10).astype(np.float16))
  P.up("gact8", (rng.standard_normal(8 * 17408) * 0.10).astype(np.float16))
  P.up("hh1", (rng.standard_normal(5120) * 0.05).astype(np.float32))
  P.up("hh3", (rng.standard_normal(3 * 5120) * 0.05).astype(np.float32))
  P.up("hh8", (rng.standard_normal(8 * 5120) * 0.05).astype(np.float32))
  progs = {n: prog(n) for n in ("ffn8", "ffn8r7", "down8", "down8r7",
                                "ffn8v_3", "ffn8v3r7", "down8nw32_3", "down8nw32v3r7",
                                "ffn8v8", "ffn8v8r7", "down8nw32_8", "down8nw32v8r7")}
  HF, FL = np.float16, np.float32
  pairs = [
    ("ffn8", "ffn8r7", "fg", lambda o: (d["pk_fg"], d["pk_fu"], d["gridf"], d["x1"], o),
     lambda o: (d["r7_fg"], d["r7_fu"], d["gridf"], d["x1"], o), 17408, HF, (2176, 1, 1), LS),
    ("down8", "down8r7", "fd", lambda o: (d["pk_fd"], d["gridf"], d["gact1"], d["hh1"], o),
     lambda o: (d["r7_fd"], d["gridf"], d["gact1"], d["hh1"], o), 5120, FL, (640, 1, 1), LS),
    ("ffn8v_3", "ffn8v3r7", "fg", lambda o: (d["pk_fg"], d["pk_fu"], d["gridf"], d["x3"], o),
     lambda o: (d["r7_fg"], d["r7_fu"], d["gridf"], d["x3"], o), 3 * 17408, HF, (2176, 1, 1), LS),
    ("down8nw32_3", "down8nw32v3r7", "fd", lambda o: (d["pk_fd"], d["gridf"], d["gact3"], d["hh3"], o),
     lambda o: (d["r7_fd"], d["gridf"], d["gact3"], d["hh3"], o), 3 * 5120, FL, (160, 1, 1), (1024, 1, 1)),
    ("ffn8v8", "ffn8v8r7", "fg", lambda o: (d["pk_fg"], d["pk_fu"], d["gridf"], d["x8"], o),
     lambda o: (d["r7_fg"], d["r7_fu"], d["gridf"], d["x8"], o), 8 * 17408, HF, (2176, 1, 1), LS),
    ("down8nw32_8", "down8nw32v8r7", "fd", lambda o: (d["pk_fd"], d["gridf"], d["gact8"], d["hh8"], o),
     lambda o: (d["r7_fd"], d["gridf"], d["gact8"], d["hh8"], o), 8 * 5120, FL, (160, 1, 1), (1024, 1, 1)),
  ]
  dev.synchronize(); P._keep.clear()
  fb = os.path.getsize(f"{BASE}/packed/fg0.npy") + os.path.getsize(f"{BASE}/packed/fu0.npy")
  db = os.path.getsize(f"{BASE}/packed/fd0.npy")
  for oname, rname, cls, oargs, rargs, n, dt, g, ls in pairs:
    nb = n * (2 if dt == HF else 4)
    P.poison(f"bo_{oname}", nb, dt, 7.7)
    P.poison(f"br_{oname}", nb, dt, 7.7)
    def mkr(pn, af):
      ob = P.d[f"br_{oname}"] if pn == rname else P.d[f"bo_{oname}"]
      def f():
        progs[pn](*af(ob), global_size=g, local_size=ls, wait=True)
      return f
    fo = mkr(oname, oargs); fr = mkr(rname, rargs)
    res = {}
    for tag, f in ((oname, fo), (rname, fr)):
      for _ in range(3): f()
      ts = []
      for _ in range(10):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
      res[tag] = min(ts)
      wgb = (fb if cls == "fg" else db) / 1e9
      print(f"[bench] {tag}: min {min(ts):.3f} ms med {sorted(ts)[5]:.3f}  {wgb/res[tag]*1e3:.1f} GB/s", flush=True)
    print(f"[delta] {oname} -> {rname}: {res[rname]/res[oname]:.3f}x", flush=True)

if __name__ == "__main__":
  main()
