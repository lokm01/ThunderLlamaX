# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7 E1 FINAL runner: one kernel per cubin (THE R7 LAW: multi-kernel cubins
mis-execute on this fork/dext — wrong-code 'Out Of Range Register' faults;
single-kernel cubins clean; the repo's -DKNAME one-kernel-per-cubin build
convention is LOAD-BEARING). Synced min-of-N timing, per-arm GB/s +
warp-instr/cyc/SM + clock64 cycles (arm a0 + clocks embedded in every arm)."""
import os, sys, time, subprocess, re, collections
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
# (kernel, guard, ITER letter, iters, ldg-per-body, stg-per-body)
ARMS = [
  ("e1a_ldg_nw8",  "E1A",  "A", 96, 8, 0),
  ("e1a0_ldg_nw8", "E1A0", "A", 96, 8, 0),
  ("e1b_fma_nw8",  "E1B",  "B", 2130, 0, 0),
  ("e1c_hmma_nw8", "E1C",  "C", 3200, 0, 0),
  ("e1d_mix_nw8",  "E1D",  "D", 384, 2, 0),
  ("e1e_copy_nw8", "E1E",  "E", 96, 4, 4),
  ("e1f_dldg_nw8", "E1F",  "F", 96, 16, 0),
  ("e1g_empty_nw8", "E1G",  "F", 96, 0, 0),
]
GRID = 82
SLICE_U4 = 393216
BYTES = GRID * SLICE_U4 * 16   # ~516MB

def sh(cmd):
  for t in range(4):
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode == 0 or "failed to connect" not in (r.stderr or ""):
      return r
    time.sleep(2)
  return r

def build_all():
  ok = True
  for kn, gd, L, it, _, _ in ARMS:
    cub = f"{BASE}/r7_e1d_{kn}.cubin"
    r = sh(["nvcc", "-arch=sm_86", "-cubin", f"-D{gd}=1", f"-DITER_{L}={it}",
            "--output-file=" + cub, f"{BASE}/r7_e1d.cu"])
    if r.returncode or not os.path.exists(cub):
      print(f"[build] {kn} FAIL\n{r.stderr[-800:]}"); ok = False; continue
    reg = [l for l in (r.stderr or "").splitlines() if "Used" in l]
    print(f"[build] {kn} OK {reg[-1].strip() if reg else ''}")
  return ok

def sass_census(kn):
  cub = f"{BASE}/r7_e1d_{kn}.cubin"
  r = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-sass", cub])
  if r.returncode:
    return None, None
  ops = re.findall(r"\*/\s+@?!?\S*\s+([A-Z][A-Z0-9.]+)", r.stdout)
  ops = [o for o in ops if o not in ("BRA", "EXIT", "NOP")]
  c = collections.Counter(ops)
  nspill = len(re.findall(r"\b(STL|LDL)\b", r.stdout))
  return c, nspill

def main():
  if not build_all():
    sys.exit(1)
  P = Bufs()
  P.alloc("src", BYTES)
  P.alloc("dst", BYTES)
  P.up("out", np.zeros(GRID * 16, dtype=np.uint32))
  dev.synchronize()
  # DEVICE-side fill (host DMA of 0.5GB = the DART danger class; banked law).
  # Single-kernel cubin (the R7 multi-kernel-cubin law).
  r = sh(["nvcc", "-arch=sm_86", "-cubin", "--output-file=" + BASE + "/r7_e1d_fill.cubin", BASE + "/r7_e1d_fill.cu"])
  assert r.returncode == 0 and os.path.exists(BASE + "/r7_e1d_fill.cubin"), r.stderr[-500:]
  flib = open(BASE + "/r7_e1d_fill.cubin", "rb").read()
  fill = NVProgram(dev, TinyELF(lib=flib, name="e1_fill", target=dev.renderer.target, signature=tuple()))
  NU4 = BYTES // 16
  G = (NU4 + 8191) // 8192
  fill(P.d["src"], global_size=(G, 1, 1), local_size=(256, 1, 1), wait=True)
  fill(P.d["dst"], global_size=(G, 1, 1), local_size=(256, 1, 1), wait=True)
  print(f"[e1] buffers ready: src/dst {BYTES>>20}MB device-filled (grid {G})", flush=True)

  prgs = {}
  census = {}
  for kn, *_ in ARMS:
    lib = open(f"{BASE}/r7_e1d_{kn}.cubin", "rb").read()
    prgs[kn] = NVProgram(dev, TinyELF(lib=lib, name=kn, target=dev.renderer.target, signature=tuple()))
    census[kn], nsp = sass_census(kn)
    st = sum(census[kn].values()) if census[kn] else 0
    l128 = sum(v for k, v in census[kn].items() if k.startswith("LDG.E.128")) if census[kn] else 0
    hmma = sum(v for k, v in census[kn].items() if k.startswith("HMMA")) if census[kn] else 0
    print(f"[e1] {kn}: regs={prgs[kn].regs_usage} static={st} LDG128={l128} HMMA={hmma} spills={nsp}", flush=True)
  dev.synchronize()

  # warm the machine: 2 dummy passes of arm E
  for _ in range(2):
    prgs["e1e_copy_nw8"](P.d["src"], P.d["dst"], P.d["out"], global_size=(GRID,1,1), local_size=(256,1,1), wait=True)

  print("\n== E1 RESULTS (synced min-of-8 + 2 warm; 82 CTAs x 256 thr = 1 CTA/SM) ==", flush=True)
  res = {}
  for kn, gd, L, it, ldgb, stgb in ARMS:
    best = 1e9
    for i in range(10):
      t0 = time.perf_counter()
      if kn == "e1e_copy_nw8":
        prgs[kn](P.d["src"], P.d["dst"], P.d["out"], global_size=(GRID,1,1), local_size=(256,1,1), wait=True)
      else:
        prgs[kn](P.d["src"], P.d["out"], global_size=(GRID,1,1), local_size=(256,1,1), wait=True)
      best = min(best, time.perf_counter() - t0)
    st = sum(census[kn].values())
    bytes_moved = (ldgb + stgb) * 16 * 256 * GRID * it
    gbs = bytes_moved / best / 1e9
    # clock windows from arm a0's out buffer (cycle calibration)
    res[kn] = (best, st, gbs, it, ldgb, stgb)
    print(f"[e1] {kn:14s} wall {best*1e3:8.3f}ms static {st:4d} GB/s {gbs:7.1f}", flush=True)
    if kn == "e1a0_ldg_nw8":
      mv = memoryview(bytearray(GRID*16*4)); dev.allocator._copyout(mv, P.d["out"])
      a = np.frombuffer(mv, dtype=np.uint32)[:GRID*4].reshape(GRID, 4)
      ok = (a[:,2] > a[:,1]) & (a[:,1] > 0)
      cyc = float(np.median((a[ok,2].astype(np.int64) - a[ok,1].astype(np.int64))))
      print(f"[e1] clock64 median cycles/CTA (arm a0): {cyc:.0f} -> SM clock ~= {cyc/best/1e9:.2f} GHz (incl ~0.1ms launch)", flush=True)

  print("\n== VERDICT ARITHMETIC ==", flush=True)
  if all(k in res for k in ("e1a_ldg_nw8", "e1b_fma_nw8", "e1c_hmma_nw8", "e1d_mix_nw8", "e1f_dldg_nw8")):
    # cycles per static instruction (loop-dominated => static ~ dynamic/ratio)
    r = {k: res[k][0] / res[k][1] for k in res}
    print("ms per static instr: " + " ".join(f"{k.split('_')[0]}={r[k]*1e3:.2e}" for k in r))
    A, B, C, D, F = r["e1a_ldg_nw8"], r["e1b_fma_nw8"], r["e1c_hmma_nw8"], r["e1d_mix_nw8"], r["e1f_dldg_nw8"]
    print(f"instr-rate ratios (1.0 = same issue rate): A/B {A/B:.3f}  A/C {A/C:.3f}  D/B {D/B:.3f}  D/C {D/C:.3f}  F/A {F/A:.3f}")
    print(f"load rates: A {res['e1a_ldg_nw8'][2]:.1f} GB/s | E copy {res['e1e_copy_nw8'][2]:.1f} GB/s | D {res['e1d_mix_nw8'][2]:.1f} GB/s")
  print("[e1] DONE")

if __name__ == "__main__":
  main()
