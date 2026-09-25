# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7b rung-2a runner: mbarrier / named-barrier smoke on the dext.
Arms: sync (production shape), nbar (bar.sync pipeline), mbarr (mbarrier
pipeline), empty (launch floor). Correctness: u64 mod-2^64 checksums per CTA
== host expected, identical across arms. One kernel per cubin (THE R7 LAW)."""
import os, sys, time, subprocess
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_QMD_BARRIERS", "16")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
ARMS = [("r7b_mb_sync", "MB_SYNC"), ("r7b_mb_nbar", "MB_NBAR"),
        ("r7b_mb_mbarr", "MB_MBARR"), ("r7b_mb_empty", "MB_EMPTY")]
GRID, NSTAGE, SLOTS = 82, 384, 512
NSLOT_TOT = GRID * NSTAGE * SLOTS          # uint4 slots
BYTES = NSLOT_TOT * 16                     # ~258MB

def sh(cmd):
  for t in range(4):
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode == 0 or "failed to connect" not in (r.stderr or ""):
      return r
    time.sleep(2)
  return r

def build():
  ok = True
  for kn, gd in ARMS:
    cub = f"{BASE}/{kn}.cubin"
    r = sh(["nvcc", "-arch=sm_86", "-cubin", f"-D{gd}=1", "--output-file=" + cub, f"{BASE}/r7b_mb.cu"])
    if r.returncode or not os.path.exists(cub):
      print(f"[build] {kn} FAIL\n{r.stderr[-600:]}"); ok = False; continue
    m = [l for l in (r.stderr or "").splitlines() if "Used" in l]
    print(f"[build] {kn} OK " + (m[-1].strip() if m else ""), flush=True)
  r = sh(["nvcc", "-arch=sm_86", "-cubin", "--output-file=" + BASE + "/r7b_fill.cubin", BASE + "/r7b_fill.cu"])
  assert r.returncode == 0 and os.path.exists(BASE + "/r7b_fill.cubin"), r.stderr[-500:]
  print("[build] fill OK", flush=True)
  return ok

def main():
  assert build()
  P = Bufs()
  P.alloc("src", BYTES)
  P.up("out", np.zeros(GRID, dtype=np.uint32))
  P.up("out64", np.zeros(GRID, dtype=np.uint64))
  dev.synchronize()
  fill = NVProgram(dev, TinyELF(lib=open(BASE + "/r7b_fill.cubin", "rb").read(), name="r7b_fill",
                                target=dev.renderer.target, signature=tuple()))
  G = (NSLOT_TOT + 255) // 256
  fill(P.d["src"], global_size=(G, 1, 1), local_size=(256, 1, 1), wait=True)
  print(f"[mb] src {BYTES>>20}MB filled device-side (grid {G})", flush=True)
  prgs = {}
  for kn, _ in ARMS:
    prgs[kn] = NVProgram(dev, TinyELF(lib=open(f"{BASE}/{kn}.cubin", "rb").read(), name="r7b_mb",
                                      target=dev.renderer.target, signature=tuple()))
    print(f"[mb] {kn}: regs={prgs[kn].regs_usage}", flush=True)
  dev.synchronize()

  def run(kn):
    P.win_up("out", 0, np.zeros(GRID, dtype=np.uint32))
    P.win_up("out64", 0, np.zeros(GRID, dtype=np.uint64))
    prgs[kn](P.d["src"], P.d["out"], P.d["out64"], global_size=(GRID, 1, 1), local_size=(256, 1, 1), wait=True)

  # host expected: cta c: sum over stages/slots of (i + 3i+1 + 5i+2 + 7i+3) mod 2^64
  i0 = np.arange(GRID, dtype=np.uint64)[:, None, None] * np.uint64(NSTAGE * SLOTS) \
       + np.arange(NSTAGE, dtype=np.uint64)[None, :, None] * np.uint64(SLOTS) \
       + np.arange(SLOTS, dtype=np.uint64)[None, None, :]
  exp = ((i0 % np.uint64(2**32)) * np.uint64(1 + 3 + 5 + 7) + np.uint64(1 + 2 + 3)).sum(axis=(1, 2))

  res = {}
  for kn, _ in ARMS:
    run(kn)
    fl = np.frombuffer(bytes(P.down("out", (GRID,), np.uint32)), dtype=np.uint32)
    cs = np.frombuffer(bytes(P.down("out64", (GRID,), np.uint64)), dtype=np.uint64)
    okc = bool((cs == exp).all()) and int((fl & 0xC0000000).sum()) == 0 and int((fl & 0xFFFFFFF).sum()) == GRID
    res[kn] = okc
    print(f"[mb] {kn}: checksum {'OK' if okc else 'FAIL'} flags={fl[:4].tolist()} cs[0]={cs[0]} exp[0]={exp[0]}", flush=True)
  best = {}
  for kn, _ in ARMS:
    b = 1e9
    for _ in range(10):
      t0 = time.perf_counter()
      prgs[kn](P.d["src"], P.d["out"], P.d["out64"], global_size=(GRID, 1, 1), local_size=(256, 1, 1), wait=True)
      b = min(b, time.perf_counter() - t0)
    best[kn] = b
    gbs = BYTES / b / 1e9
    print(f"[mb] {kn:14s} wall {b*1e3:8.3f}ms  GB/s {gbs:7.1f}", flush=True)
  print("\n== VERDICT ==")
  print(f"mbarr functional: {res['r7b_mb_mbarr']}  nbar functional: {res['r7b_mb_nbar']}")
  s, n, m = best["r7b_mb_sync"], best["r7b_mb_nbar"], best["r7b_mb_mbarr"]
  print(f"pipeline speedup vs sync-shape: nbar x{s/n:.3f}  mbarr x{s/m:.3f}")
  print("[mb] DONE")

if __name__ == "__main__":
  main()
