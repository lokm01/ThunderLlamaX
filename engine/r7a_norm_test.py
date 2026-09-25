# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a norms rung gate: OLD (serial t-loop cubins, old grids) vs NEW (per-row
CTA cubins, new grids) on deterministic inputs — nz==0 det-x2 + bench.
Run: ~/tg311/bin/python -u r7a_norm_test.py"""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import dev
def iq3s_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3s_grid
  return np.array([(w >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.float32)
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
  rng = np.random.default_rng(7)
  d = P.d
  P.up("grid512", iq3s_grid_f32())
  # inputs
  P.up("x8", (rng.standard_normal(8 * 5120) * 0.3).astype(np.float32))
  P.up("x3", (rng.standard_normal(3 * 5120) * 0.3).astype(np.float32))
  P.up("ao8", (rng.standard_normal(8 * 5120) * 0.2).astype(np.float16))
  P.up("ao3", (rng.standard_normal(3 * 5120) * 0.2).astype(np.float16))
  P.up("nw", (rng.standard_normal(5120) * 0.1 + 1.0).astype(np.float32))
  P.up("nw2", (rng.standard_normal(5120) * 0.1 + 1.0).astype(np.float32))
  P.up("alpha", (rng.standard_normal(48 * 5120) * 0.05).astype(np.float32))
  P.up("beta", (rng.standard_normal(48 * 5120) * 0.05).astype(np.float32))
  P.up("emb8", rng.integers(0, 256, 4096 * 2200, dtype=np.uint8).tobytes() if False else rng.integers(0, 256, 4096 * 2200, dtype=np.uint8))
  for t in range(8):
    P.up(f"tok{t}", np.array([int(rng.integers(0, 4096))], dtype=np.int32))
  for t in range(3):
    P.up(f"tok3_{t}", np.array([int(rng.integers(0, 4096))], dtype=np.int32))
  dev.synchronize(); P._keep.clear()

  KS = ["h_embed8", "k0n8", "k0ab8", "hh8", "h_embed3", "k0n3", "k0ab3", "hh3"]
  progs = {("old", n): prog(f"{BASE}/pre7a_norm_bak/{n}.cubin", n) for n in KS}
  progs.update({("new", n): prog(f"{BASE}/{n}.cubin", n) for n in KS})

  OLDG = {"h_embed8": 1, "k0n8": 1, "k0ab8": 13, "hh8": 1, "h_embed3": 1, "k0n3": 1, "k0ab3": 13, "hh3": 1}
  NEWG = {"h_embed8": 8, "k0n8": 8, "k0ab8": 104, "hh8": 8, "h_embed3": 3, "k0n3": 3, "k0ab3": 39, "hh3": 3}

  def mkargs(name):
    M = 8 if name.endswith("8") else 3
    xx = d["x8"] if M == 8 else d["x3"]
    ao = d["ao8"] if M == 8 else d["ao3"]
    if name.startswith("h_embed"):
      srcs = [d[f"tok{t}"] for t in range(M)] if M == 8 else [d[f"tok3_{t}"] for t in range(M)]
      return (d["emb8"], d["grid512"], *srcs, d["out_f32"]), M * 5120 * 4, np.float32
    if name == "k0n8" or name == "k0n3":
      return (xx, d["nw"], d["out_f16"]), M * 5120 * 2, np.float16
    if name.startswith("k0ab"):
      return (xx, d["nw"], d["alpha"], d["beta"], d["out_f16"], d["out2_f32"], d["out3_f32"]), None, None
    if name.startswith("hh"):
      return (xx, ao, d["nw2"], d["out_f32"], d["out_f16"]), None, None
    raise AssertionError(name)

  def run(name, side):
    M = 8 if name.endswith("8") else 3
    a, nb, dt = mkargs(name)
    g = OLDG[name] if side == "old" else NEWG[name]
    progs[(side, name)](*a[:-1], P.d[f"{side}_out"], global_size=(g, 1, 1), local_size=LS, wait=True) if False else None
    # handled below

  # explicit runners (k0ab/hh write two/three outputs — give each side its own set)
  def runner(name, side):
    M = 8 if name.endswith("8") else 3
    xx = d["x8"] if M == 8 else d["x3"]
    ao = d["ao8"] if M == 8 else d["ao3"]
    g = OLDG[name] if side == "old" else NEWG[name]
    p = progs[(side, name)]
    if name.startswith("h_embed"):
      srcs = [d[f"tok{t}"] for t in range(8)] if M == 8 else [d[f"tok3_{t}"] for t in range(3)]
      return lambda: p(d["emb8"], d["grid512"], *srcs, d[f"{side}_x"], global_size=(g,1,1), local_size=LS, wait=True), M*5120, np.float32
    if name in ("k0n8", "k0n3"):
      return lambda: p(xx, d["nw"], d[f"{side}_xh"], global_size=(g,1,1), local_size=LS, wait=True), M*5120, np.float16
    if name.startswith("k0ab"):
      def f():
        p(xx, d["nw"], d["alpha"], d["beta"], d[f"{side}_xh"], d[f"{side}_a"], d[f"{side}_b"], global_size=(g,1,1), local_size=LS, wait=True)
      return f, None, None
    if name.startswith("hh"):
      def f():
        p(xx, ao, d["nw2"], d[f"{side}_hh"], d[f"{side}_xh"], global_size=(g,1,1), local_size=LS, wait=True)
      return f, None, None
    raise AssertionError(name)

  # warm-up pair (bare-world law)
  P.up("old_x", np.zeros(8*5120, np.float32)); P.up("new_x", np.zeros(8*5120, np.float32))
  f0, _, _ = runner("h_embed8", "old"); f1, _, _ = runner("h_embed8", "new")
  f0(); f1(); dev.synchronize(); print("[warm] h_embed8 pair done", flush=True)

  for name in KS:
    M = 8 if name.endswith("8") else 3
    for it in range(2):
      # fresh outputs per side
      P.poison("old_xh", M*5120*2, np.float16, 7.7); P.poison("new_xh", M*5120*2, np.float16, 7.7)
      P.up("old_a", np.full(M*48, -3.3, np.float32)); P.up("new_a", np.full(M*48, -3.3, np.float32))
      P.up("old_b", np.full(M*48, -3.3, np.float32)); P.up("new_b", np.full(M*48, -3.3, np.float32))
      P.poison("old_hh", M*5120*4, np.float32, 7.7e31); P.poison("new_hh", M*5120*4, np.float32, 7.7e31)
      P.poison("old_x", M*5120*4, np.float32, 7.7); P.poison("new_x", M*5120*4, np.float32, 7.7)
      fo, nb, dt = runner(name, "old"); fn, _, _ = runner(name, "new")
      fo(); fn()
      tot_nz = 0
      if name.startswith("h_embed"):
        a = P.down("old_x", (M*5120,), np.float32); b = P.down("new_x", (M*5120,), np.float32)
        tot_nz += int((a.view(np.uint32) != b.view(np.uint32)).sum())
      elif name in ("k0n8", "k0n3"):
        a = P.down("old_xh", (M*5120,), np.float16); b = P.down("new_xh", (M*5120,), np.float16)
        tot_nz += int((a.view(np.uint16) != b.view(np.uint16)).sum())
      elif name.startswith("k0ab"):
        for tag in ("xh", "a", "b"):
          dt2 = np.float16 if tag == "xh" else np.float32
          a = P.down(f"old_{tag}", (M*5120 if tag=="xh" else M*48,), dt2); b = P.down(f"new_{tag}", (M*5120 if tag=="xh" else M*48,), dt2)
          tot_nz += int((a.view(np.uint16 if tag=="xh" else np.uint32) != b.view(np.uint16 if tag=="xh" else np.uint32)).sum())
      else:
        for tag in ("hh", "xh"):
          dt2 = np.float32 if tag == "hh" else np.float16
          a = P.down(f"old_{tag}", (M*5120,), dt2); b = P.down(f"new_{tag}", (M*5120,), dt2)
          tot_nz += int((a.view(np.uint32 if tag=="hh" else np.uint16) != b.view(np.uint32 if tag=="hh" else np.uint16)).sum())
      print(f"[corr] {name} det{it}: nz={tot_nz}", flush=True)
      assert tot_nz == 0, (name, it, tot_nz)
      P._keep.clear()
  print("[corr] ALL 8 NORM KERNELS OLD==NEW BIT-IDENTICAL det-x2", flush=True)

  # bench
  for name in KS:
    ts = {"old": [], "new": []}
    for side in ("old", "new"):
      f, _, _ = runner(name, side)
      for _ in range(3): f()
      for _ in range(10):
        t0 = time.perf_counter(); f(); ts[side].append((time.perf_counter()-t0)*1e3)
    print(f"[bench] {name}: old {min(ts['old']):.4f} ms  new {min(ts['new']):.4f} ms  x{min(ts['old'])/min(ts['new']):.2f}", flush=True)
  print("[r7a_norm_test done]", flush=True)

if __name__ == "__main__":
  main()
