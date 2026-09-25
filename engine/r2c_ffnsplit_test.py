# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c split-plane FFN gate: (a) CORR — fused pfg3_ffn_r7_m64_nw4k128 vs
[fgp m64-nw8 + fup m64-nw8 + pfk_smul64] on REAL packed7 weights (fg0/fu0),
poison-first, det x2 — must be BIT-IDENTICAL (plain epilogue writes (half)acc;
smul applies the identical fused sequence); (b) bench — synced min-of-10.
M=64 x rows.
Run: ~/tg311/bin/python -u r2c_ffnsplit_test.py"""
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
  rng = np.random.default_rng(1)
  P.up("r7_fg", np.load(f"{BASE}/packed7/fg0.npy"))
  P.up("r7_fu", np.load(f"{BASE}/packed7/fu0.npy"))
  P.up("gridf", iq3_grid_f32())
  d = P.d
  M = 64
  P.up("hhx", (rng.standard_normal(M * 5120) * 0.10).astype(np.float16))
  for nm, n in (("gact_f", M * 17408), ("ag", M * 17408), ("au", M * 17408), ("gact_s", M * 17408)):
    P.poison(nm, n * 2, np.float16, 7.7)
  progs = {n: prog(n) for n in ("pfg3_ffn_r7_m64_nw4k128", "pfg3_fgp_r7_m64_nw8k128",
                                "pfg3_fup_r7_m64_nw8k128", "pfk_smul64")}
  dev.synchronize(); P._keep.clear()
  # warm-up pair (the bare-world first-pair law)
  progs["pfg3_ffn_r7_m64_nw4k128"](d["r7_fg"], d["r7_fu"], d["gridf"], d["hhx"], d["gact_f"], global_size=(544, 1, 1), local_size=(128, 1, 1), wait=True)
  progs["pfg3_fgp_r7_m64_nw8k128"](d["r7_fg"], d["gridf"], d["hhx"], d["ag"], global_size=(272, 1, 1), local_size=LS, wait=True)
  dev.synchronize()
  print("[warm] pair done", flush=True)

  def fused():
    progs["pfg3_ffn_r7_m64_nw4k128"](d["r7_fg"], d["r7_fu"], d["gridf"], d["hhx"], d["gact_f"], global_size=(544, 1, 1), local_size=(128, 1, 1), wait=True)
  def split():
    progs["pfg3_fgp_r7_m64_nw8k128"](d["r7_fg"], d["gridf"], d["hhx"], d["ag"], global_size=(272, 1, 1), local_size=LS, wait=True)
    progs["pfg3_fup_r7_m64_nw8k128"](d["r7_fu"], d["gridf"], d["hhx"], d["au"], global_size=(272, 1, 1), local_size=LS, wait=True)
    progs["pfk_smul64"](d["ag"], d["au"], d["gact_s"], global_size=(M * 17408 // 2048, 1, 1), local_size=LS, wait=True)

  for it in range(2):
    for f, nm in ((fused, "gact_f"), (split, "gact_s")):
      P.poison(nm, M * 17408 * 2, np.float16, 7.7) if nm == "gact_f" else P.poison(nm, M * 17408 * 2, np.float16, 7.7)
      f()
    a = P.down("gact_f", (M * 17408,), np.float16); b = P.down("gact_s", (M * 17408,), np.float16)
    nz = int((a.view(np.uint16) != b.view(np.uint16)).sum())
    print(f"[corr] det{it}: nz={nz}/{M*17408} maxabsdiff={float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max()):.3e}", flush=True)
    assert nz == 0, (it, nz)
    P._keep.clear()

  # bench: synced min-of-10
  def bench(tag, f):
    for _ in range(3): f()
    ts = []
    for _ in range(10):
      t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    print(f"[bench] {tag}: min {min(ts):.3f} ms med {sorted(ts)[5]:.3f}", flush=True)
    return min(ts)
  tf = bench("fused_m64_nw4", fused)
  ts_ = bench("split_m64_nw8+smul", split)
  wb = 2 * os.path.getsize(f"{BASE}/packed7/fg0.npy")
  print(f"[bench] ratio split/fused = {ts_/tf:.3f}x  ({tf:.3f} -> {ts_:.3f} ms/64r/block)", flush=True)
  print(f"[bw] fused {wb/1e9/tf*1e3:.1f} GB/s | split {wb/1e9/ts_*1e3:.1f} GB/s (r7 bytes)", flush=True)

if __name__ == "__main__":
  main()
