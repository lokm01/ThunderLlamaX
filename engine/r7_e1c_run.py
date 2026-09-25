# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7 E1 v2 runner: bisect-friendly (E1_ARMS env selects arms), SASS audit,
synced min-of-N timing, per-arm GB/s + warp-instr/cyc/SM."""
import os, sys, time, subprocess, re, collections
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
ITERS = {"e1a_ldg_nw8": 192, "e1a0_ldg_nw8": 192, "e1b_fma_nw8": 213, "e1c_hmma_nw8": 320,
         "e1d_mix_nw8": 256, "e1e_copy_nw8": 192, "e1f_dldg_nw8": 96}
ARMS = [a.strip() for a in os.getenv("E1_ARMS", ",".join(ITERS)).split(",") if a.strip()]
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

def build():
  defs = " ".join("-DITER_%s=%d" % (L, v) for L, v in
                  (("A", 192), ("B", 213), ("C", 320), ("D", 256), ("E", 192), ("F", 96)))
  r = sh(["nvcc", "-arch=sm_86", "-cubin"] + defs.split() + ["-Xptxas", "-v",
          "--output-file=" + BASE + "/r7_e1c.cubin", BASE + "/r7_e1c.cu"])
  ok = r.returncode == 0 and os.path.exists(BASE + "/r7_e1c.cubin")
  print("[e1] build:", "OK" if ok else "FAIL")
  for ln in (r.stderr or "").splitlines():
    if "registers" in ln and "Used" in ln:
      print("   ", ln.strip())
  if not ok:
    print(r.stderr[-2000:]); sys.exit(1)

def sass_audit():
  r = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-sass", BASE + "/r7_e1c.cubin"])
  text = r.stdout
  counts = {}
  for kn in ITERS:
    m = re.search(r"Function : " + kn + r"\b(.*?)(?:\tFunction :|\Z)", text, re.S)
    if not m:
      continue
    body = m.group(1)
    nspill = len(re.findall(r"\b(STL|LDL)\b", body))
    ops = re.findall(r"\*/\s+@?!?\S*\s+([A-Z][A-Z0-9.]+)", body)
    ops = [o for o in ops if o not in ("BRA", "EXIT", "NOP")]
    counts[kn] = collections.Counter(ops)
    tot = sum(counts[kn].values())
    l128 = sum(v for k, v in counts[kn].items() if k.startswith("LDG.E.128"))
    print(f"[sass] {kn}: static {tot:4d} LDG128 x{l128:3d} spills {nspill} top {counts[kn].most_common(4)}")
  return counts

def main():
  build()
  counts = sass_audit()
  dev = Device["NV"]
  src = dev.allocator.alloc(BYTES, BufferSpec())
  dst = dev.allocator.alloc(BYTES, BufferSpec())
  outb = dev.allocator.alloc(GRID * 16 * 4, BufferSpec(cpu_access=True, nolru=True))
  lib = open(BASE + "/r7_e1c.cubin", "rb").read()
  prgs = {}
  for kn in ARMS:
    prgs[kn] = NVProgram(dev, TinyELF(lib=lib, name=kn, target=dev.renderer.target, signature=tuple()))
  dev.synchronize()

  def run_arm(kn, n=8, warm=2):
    best = 1e9
    for i in range(n + warm):
      t0 = time.perf_counter()
      if kn == "e1e_copy_nw8":
        prgs[kn](src, dst, outb, global_size=(GRID, 1, 1), local_size=(256, 1, 1), wait=True)
      else:
        prgs[kn](src, outb, global_size=(GRID, 1, 1), local_size=(256, 1, 1), wait=True)
      dt = time.perf_counter() - t0
      if i >= warm:
        best = min(best, dt)
    return best

  print("\n== E1v2 RESULTS (synced min-of-8; 82 CTAs x 256 thr = 1 CTA/SM; " + " ".join(ARMS) + ") ==")
  res = {}
  for kn in ARMS:
    wall = run_arm(kn)
    st = sum(counts[kn].values()) if kn in counts else 0
    ldg_dyn = {"e1a_ldg_nw8": 8 * 192, "e1a0_ldg_nw8": 8 * 192, "e1d_mix_nw8": 2 * 256,
               "e1f_dldg_nw8": 16 * 96, "e1e_copy_nw8": 4 * 192}.get(kn, 0)
    stg_dyn = 4 * 192 if kn == "e1e_copy_nw8" else 0
    bytes_moved = (ldg_dyn + stg_dyn) * 16 * 256 * GRID
    gbs = bytes_moved / wall / 1e9
    # clock cycle read for the a0 arm (has clock64 windows); others: infer
    # cycles from wall x ~1.7GHz? NO — use wall for rate ratios; report instr/ms
    winstr_per_sm = st * 8  # 8 warps x static (loop-dominated)
    res[kn] = (wall, st, gbs, winstr_per_sm)
    print(f"[e1] {kn:14s} wall {wall*1e3:8.3f}ms  static {st:4d}  w-instr/ms/SM {winstr_per_sm/(wall*1e3)/1e3:8.3f}k  GB/s {gbs:7.1f}", flush=True)

  if all(k in res for k in ("e1a_ldg_nw8", "e1b_fma_nw8", "e1c_hmma_nw8", "e1d_mix_nw8")):
    print("\n== VERDICT ARITHMETIC (cycles per static instr ~ inverse issue rate) ==")
    for k in ARMS:
      if k in res:
        print(f"  {k}: {res[k][0]/res[k][1]*1e3:.4f} ms/static-instr")
    r = {k: res[k][0] / res[k][1] for k in ARMS}
    print(f"  A/B {r['e1a_ldg_nw8']/r['e1b_fma_nw8']:.3f}  A/C {r['e1a_ldg_nw8']/r['e1c_hmma_nw8']:.3f}  "
          f"D/B {r['e1d_mix_nw8']/r['e1b_fma_nw8']:.3f}  D/C {r['e1d_mix_nw8']/r['e1c_hmma_nw8']:.3f}")
  print("[e1] DONE")

if __name__ == "__main__":
  main()
