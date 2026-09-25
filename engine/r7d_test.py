# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c: A/B bit-exact gates + bench for the decode-side r7 GEMVs (r7d.cu) vs the
LIVE originals (ffn8/down8 = trunk T=1; ffn8v_3/down8nw32_3 = K2 probe GEMVV;
ffn8v8/down8nw32_8 = the K=7 deep probe) on REAL weights (packed/ vs packed7/,
block 0). Poison-first, warm-up pair before any corr (the P17 bare-world law),
det x2, then synced min-of-10 timing.

Run: ~/tg311/bin/python -u r7d_test.py   (GPU; ~2 min)
"""
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
  W = {}
  for tag in ("fg", "fu", "fd"):
    W[("pk", tag)] = P.up(f"pk_{tag}", np.load(f"{BASE}/packed/{tag}0.npy"))
    W[("r7", tag)] = P.up(f"r7_{tag}", np.load(f"{BASE}/packed7/{tag}0.npy"))
  P.up("gridf", iq3_grid_f32())
  d = P.d
  progs = {n: prog(n) for n in ("ffn8", "ffn8r7", "down8", "down8r7",
                                "ffn8v_3", "ffn8v3r7", "down8nw32_3", "down8nw32v3r7",
                                "ffn8v8", "ffn8v8r7", "down8nw32_8", "down8nw32v8r7")}
  # x planes (fp16), deterministic
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

  # warm-up pair (the bare-world first-pair law)
  for nm in ("og1", "or1"):
    P.poison(nm, 17408 * 2, np.float16, 7.7)
  progs["ffn8"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x1"], d["og1"], global_size=(2176, 1, 1), local_size=LS, wait=True)
  progs["ffn8r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x1"], d["or1"], global_size=(2176, 1, 1), local_size=LS, wait=True)
  dev.synchronize()
  print("[warm] ffn8 pair done", flush=True)

  def cmp16(na, nb, n, tag):
    a = P.down(na, (n,), np.float16); b = P.down(nb, (n,), np.float16)
    nz = int((a.view(np.uint16) != b.view(np.uint16)).sum())
    print(f"[corr] {tag}: nz={nz}/{n} maxabsdiff={float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max()):.3e}", flush=True)
    P._keep.clear()
    return nz

  def run_pair(tag, f_orig, f_r7, nout, dt, det=2):
    for it in range(det):
      for side, f in (("o", f_orig), ("r", f_r7)):
        P.poison(f"{side}_out", nout * (2 if dt == np.float16 else 4), dt, 7.7)
        f()
      nz = cmp16("o_out", "r_out", nout, f"{tag} det{it}")
      assert nz == 0, (tag, it, nz)

  # ---- ffn8 / ffn8r7 (M=1) ----
  run_pair("ffn8-r7",
    lambda: progs["ffn8"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x1"], d["o_out"], global_size=(2176, 1, 1), local_size=LS, wait=True),
    lambda: progs["ffn8r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x1"], d["r_out"], global_size=(2176, 1, 1), local_size=LS, wait=True),
    17408, np.float16)
  # ---- down8 / down8r7 (M=1) ----
  run_pair("down8-r7",
    lambda: progs["down8"](d["pk_fd"], d["gridf"], d["gact1"], d["hh1"], d["o_out"], global_size=(640, 1, 1), local_size=LS, wait=True),
    lambda: progs["down8r7"](d["r7_fd"], d["gridf"], d["gact1"], d["hh1"], d["r_out"], global_size=(640, 1, 1), local_size=LS, wait=True),
    5120, np.float32)
  # ---- ffn8v_3 / ffn8v3r7 (M=3) ----
  run_pair("ffn8v3-r7",
    lambda: progs["ffn8v_3"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x3"], d["o_out"], global_size=(2176, 1, 1), local_size=LS, wait=True),
    lambda: progs["ffn8v3r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x3"], d["r_out"], global_size=(2176, 1, 1), local_size=LS, wait=True),
    3 * 17408, np.float16)
  # ---- down8nw32_3 / down8nw32v3r7 (M=3) ----
  run_pair("down8nw32v3-r7",
    lambda: progs["down8nw32_3"](d["pk_fd"], d["gridf"], d["gact3"], d["hh3"], d["o_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True),
    lambda: progs["down8nw32v3r7"](d["r7_fd"], d["gridf"], d["gact3"], d["hh3"], d["r_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True),
    3 * 5120, np.float32)
  # ---- ffn8v8 / ffn8v8r7 (M=8) ----
  run_pair("ffn8v8-r7",
    lambda: progs["ffn8v8"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x8"], d["o_out"], global_size=(2176, 1, 1), local_size=LS, wait=True),
    lambda: progs["ffn8v8r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x8"], d["r_out"], global_size=(2176, 1, 1), local_size=LS, wait=True),
    8 * 17408, np.float16)
  # ---- down8nw32_8 / down8nw32v8r7 (M=8) ----
  run_pair("down8nw32v8-r7",
    lambda: progs["down8nw32_8"](d["pk_fd"], d["gridf"], d["gact8"], d["hh8"], d["o_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True),
    lambda: progs["down8nw32v8r7"](d["r7_fd"], d["gridf"], d["gact8"], d["hh8"], d["r_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True),
    8 * 5120, np.float32)
  print("[corr] ALL BIT-IDENTICAL", flush=True)

  # ---- bench: synced min-of-10 (the race-inflate law: synced only) ----
  def bench(tag, f, reps=10, nb=None, dt=None, n=None):
    if nb is not None:
      P.poison("bo", nb, dt, 7.7); P.poison("br", nb, dt, 7.7)
    for _ in range(3): f()
    ts = []
    for _ in range(reps):
      t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    print(f"[bench] {tag}: min {min(ts):.3f} ms  med {sorted(ts)[len(ts)//2]:.3f}", flush=True)
    return min(ts)

  fb = os.path.getsize(f"{BASE}/packed/fg0.npy") + os.path.getsize(f"{BASE}/packed/fu0.npy")
  db = os.path.getsize(f"{BASE}/packed/fd0.npy")
  def mk(tag, progname, mkargs, nb, dt):
    P.poison(f"b_{tag}", nb, dt, 7.7)
    return lambda: progs[progname](*mkargs(P.d[f"b_{tag}"]), wait=True)
  HF, FL = np.float16, np.float32
  pairs = [
    ("ffn8", "ffn8r7", lambda o: (d["pk_fg"], d["pk_fu"], d["gridf"], d["x1"], o), lambda o: (d["r7_fg"], d["r7_fu"], d["gridf"], d["x1"], o), 17408, HF, (2176,1,1), LS),
    ("down8", "down8r7", lambda o: (d["pk_fd"], d["gridf"], d["gact1"], d["hh1"], o), lambda o: (d["r7_fd"], d["gridf"], d["gact1"], d["hh1"], o), 5120, FL, (640,1,1), LS),
    ("ffn8v_3", "ffn8v3r7", lambda o: (d["pk_fg"], d["pk_fu"], d["gridf"], d["x3"], o), lambda o: (d["r7_fg"], d["r7_fu"], d["gridf"], d["x3"], o), 3*17408, HF, (2176,1,1), LS),
    ("down8nw32_3", "down8nw32v3r7", lambda o: (d["pk_fd"], d["gridf"], d["gact3"], d["hh3"], o), lambda o: (d["r7_fd"], d["gridf"], d["gact3"], d["hh3"], o), 3*5120, FL, (160,1,1), (1024,1,1)),
    ("ffn8v8", "ffn8v8r7", lambda o: (d["pk_fg"], d["pk_fu"], d["gridf"], d["x8"], o), lambda o: (d["r7_fg"], d["r7_fu"], d["gridf"], d["x8"], o), 8*17408, HF, (2176,1,1), LS),
    ("down8nw32_8", "down8nw32v8r7", lambda o: (d["pk_fd"], d["gridf"], d["gact8"], d["hh8"], o), lambda o: (d["r7_fd"], d["gridf"], d["gact8"], d["hh8"], o), 8*5120, FL, (160,1,1), (1024,1,1)),
  ]
  times = {}
  for oname, rname, oargs, rargs, n, dt, g, ls in pairs:
    nb = n * (2 if dt == HF else 4)
    P.poison(f"bo_{oname}", nb, dt, 7.7); P.poison(f"br_{oname}", nb, dt, 7.7)
    def mkn(progname, argsf, g=g, ls=ls):
      def f():
        progs[progname](*argsf(P.d["bo_" + oname if progname == oname else "br_" + oname]), global_size=g, local_size=ls, wait=True)
      return f
    fo = mkn(oname, oargs); fr = mkn(rname, rargs)
    times[oname] = bench(oname, fo); times[rname] = bench(rname, fr)
  fb = os.path.getsize(f"{BASE}/packed/fg0.npy") + os.path.getsize(f"{BASE}/packed/fu0.npy")
  db = os.path.getsize(f"{BASE}/packed/fd0.npy")
  for nm in times:
    nb = fb if "ffn" in nm else db
    print(f"[bw] {nm}: {nb/1e9/times[nm]*1e3:.1f} GB/s (weight bytes)", flush=True)
  if False:
    t_ffn = bench("ffn8", lambda: progs["ffn8"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x1"], d["o_out"], global_size=(2176, 1, 1), local_size=LS, wait=True))
  t_ffn7 = bench("ffn8r7", lambda: progs["ffn8r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x1"], d["r_out"], global_size=(2176, 1, 1), local_size=LS, wait=True))
  t_d = bench("down8", lambda: progs["down8"](d["pk_fd"], d["gridf"], d["gact1"], d["hh1"], d["o_out"], global_size=(640, 1, 1), local_size=LS, wait=True))
  t_d7 = bench("down8r7", lambda: progs["down8r7"](d["r7_fd"], d["gridf"], d["gact1"], d["hh1"], d["r_out"], global_size=(640, 1, 1), local_size=LS, wait=True))
  t_f3 = bench("ffn8v_3", lambda: progs["ffn8v_3"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x3"], d["o_out"], global_size=(2176, 1, 1), local_size=LS, wait=True))
  t_f37 = bench("ffn8v3r7", lambda: progs["ffn8v3r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x3"], d["r_out"], global_size=(2176, 1, 1), local_size=LS, wait=True))
  t_d3 = bench("down8nw32_3", lambda: progs["down8nw32_3"](d["pk_fd"], d["gridf"], d["gact3"], d["hh3"], d["o_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True))
  t_d37 = bench("down8nw32v3r7", lambda: progs["down8nw32v3r7"](d["r7_fd"], d["gridf"], d["gact3"], d["hh3"], d["r_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True))
  t_f8 = bench("ffn8v8", lambda: progs["ffn8v8"](d["pk_fg"], d["pk_fu"], d["gridf"], d["x8"], d["o_out"], global_size=(2176, 1, 1), local_size=LS, wait=True))
  t_f87 = bench("ffn8v8r7", lambda: progs["ffn8v8r7"](d["r7_fg"], d["r7_fu"], d["gridf"], d["x8"], d["r_out"], global_size=(2176, 1, 1), local_size=LS, wait=True))
  t_d8 = bench("down8nw32_8", lambda: progs["down8nw32_8"](d["pk_fd"], d["gridf"], d["gact8"], d["hh8"], d["o_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True))
  t_d87 = bench("down8nw32v8r7", lambda: progs["down8nw32v8r7"](d["r7_fd"], d["gridf"], d["gact8"], d["hh8"], d["r_out"], global_size=(160, 1, 1), local_size=(1024, 1, 1), wait=True))
  for nm, t, nb in (("ffn8", t_ffn, fb), ("ffn8r7", t_ffn7, fb), ("down8", t_d, db), ("down8r7", t_d7, db),
                    ("ffn8v_3", t_f3, fb), ("ffn8v3r7", t_f37, fb), ("down8nw32_3", t_d3, db), ("down8nw32v3r7", t_d37, db),
                    ("ffn8v8", t_f8, fb), ("ffn8v8r7", t_f87, fb), ("down8nw32_8", t_d8, db), ("down8nw32v8r7", t_d87, db)):
    print(f"[bw] {nm}: {nb/1e9/t*1e3:.1f} GB/s (weight bytes)", flush=True)

if __name__ == "__main__":
  main()
