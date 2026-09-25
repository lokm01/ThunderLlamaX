# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a RUNG-1A gate: old (pre7a_bak) vs new (uint4-merged) r7 cubins on REAL
packed7 weights, block 0. Poison-first, warm-up pair, det x2, nz==0 required,
then synced min-of-10 bench. Usage: ~/tg311/bin/python -u r7a_test.py"""
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

def prog(path, n):
  lib = open(path, "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

def main():
  P = Bufs()
  rng = np.random.default_rng(0)
  for tag in ("fg", "fu", "fd"):
    P.up(f"r7_{tag}", np.load(f"{BASE}/packed7/{tag}0.npy"))
  P.up("gridf", iq3_grid_f32())
  d = P.d
  KS = ["ffn8r7", "down8r7", "ffn8v3r7", "down8nw32v3r7", "ffn8v8r7", "down8nw32v8r7"]
  progs = {("old", n): prog(f"{BASE}/pre7a_bak/{n}.cubin", n) for n in KS}
  progs.update({("new", n): prog(f"{BASE}/{n}.cubin", n) for n in KS})
  P.up("x1", (rng.standard_normal(5120) * 0.10).astype(np.float16))
  P.up("x3", (rng.standard_normal(3 * 5120) * 0.10).astype(np.float16))
  P.up("x8", (rng.standard_normal(8 * 5120) * 0.10).astype(np.float16))
  P.up("gact1", (rng.standard_normal(17408) * 0.10).astype(np.float16))
  P.up("gact3", (rng.standard_normal(3 * 17408) * 0.10).astype(np.float16))
  P.up("gact8", (rng.standard_normal(8 * 17408) * 0.10).astype(np.float16))
  P.up("hh1", (rng.standard_normal(5120) * 0.05).astype(np.float32))
  P.up("hh3", (rng.standard_normal(3 * 5120) * 0.05).astype(np.float32))
  P.up("hh8", (rng.standard_normal(8 * 5120) * 0.05).astype(np.float32))
  dev.synchronize(); P._keep.clear()

  # launch recipes: (args-builder, nout, dtype, grid, ls)
  def ffn_args(tag, o): return (d[f"r7_fg"], d[f"r7_fu"], d["gridf"], d[tag], o)
  def down_args(tag, o): return (d[f"r7_fd"], d["gridf"], d[f"gact{tag}"], d[f"hh{tag}"], o)
  REC = {
    "ffn8r7": (lambda o: ffn_args("x1", o), 17408, np.float16, (2176, 1, 1), LS),
    "down8r7": (lambda o: down_args("1", o), 5120, np.float32, (640, 1, 1), LS),
    "ffn8v3r7": (lambda o: ffn_args("x3", o), 3 * 17408, np.float16, (2176, 1, 1), LS),
    "down8nw32v3r7": (lambda o: down_args("3", o), 3 * 5120, np.float32, (160, 1, 1), (1024, 1, 1)),
    "ffn8v8r7": (lambda o: ffn_args("x8", o), 8 * 17408, np.float16, (2176, 1, 1), LS),
    "down8nw32v8r7": (lambda o: down_args("8", o), 8 * 5120, np.float32, (160, 1, 1), (1024, 1, 1)),
  }
  # warm-up pair (the bare-world first-pair law)
  P.poison("wo0", 17408 * 2, np.float16, 7.7); P.poison("wn0", 17408 * 2, np.float16, 7.7)
  progs[("old", "ffn8r7")](*ffn_args("x1", d["wo0"]), global_size=(2176,1,1), local_size=LS, wait=True)
  progs[("new", "ffn8r7")](*ffn_args("x1", d["wn0"]), global_size=(2176,1,1), local_size=LS, wait=True)
  dev.synchronize(); print("[warm] ffn8r7 pair done", flush=True)

  def cmp16(na, nb, n, tag):
    a = P.down(na, (n,), np.float16); b = P.down(nb, (n,), np.float16)
    nz = int((a.view(np.uint16) != b.view(np.uint16)).sum())
    print(f"[corr] {tag}: nz={nz}/{n}", flush=True)
    P._keep.clear()
    return nz

  for name, (mk, n, dt, g, ls) in REC.items():
    for it in range(2):
      for side in ("o", "n"):
        P.poison(f"{side}_out", n * (2 if dt == np.float16 else 4), dt, 7.7)
        progs[("old" if side == "o" else "new", name)](*mk(d[f"{side}_out"]), global_size=g, local_size=ls, wait=True)
      nz = cmp16("o_out", "n_out", n, f"{name} det{it}")
      assert nz == 0, (name, it, nz)
  print("[corr] ALL 6 KERNELS OLD==NEW BIT-IDENTICAL det-x2", flush=True)

  # bench: synced min-of-10
  times = {}
  for name, (mk, n, dt, g, ls) in REC.items():
    ts = {"old": [], "new": []}
    for side, key in (("o", "old"), ("n", "new")):
      P.poison(f"b_{side}", n * (2 if dt == np.float16 else 4), dt, 7.7)
      f = lambda: progs[(key, name)](*mk(d[f"b_{side}"]), global_size=g, local_size=ls, wait=True)
      for _ in range(3): f()
      for _ in range(10):
        t0 = time.perf_counter(); f(); ts[key].append((time.perf_counter() - t0) * 1e3)
    o, nw = min(ts["old"]), min(ts["new"])
    times[name] = (o, nw)
    print(f"[bench] {name}: old {o:.3f} ms  new {nw:.3f} ms  x{o/nw:.3f}", flush=True)
    P._keep.clear()
  fb = os.path.getsize(f"{BASE}/packed7/fg0.npy") * 2
  db = os.path.getsize(f"{BASE}/packed7/fd0.npy")
  for name, (o, nw) in times.items():
    nb = fb if "ffn" in name else db
    print(f"[bw] {name}: old {nb/1e9/o*1e3:.1f} -> new {nb/1e9/nw*1e3:.1f} GB/s (weight bytes)", flush=True)
  print("[r7a_test done]", flush=True)

if __name__ == "__main__":
  main()
