# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-C builder: pf_scanchunk cubins (pfca/pfcb/pfcz at C=64/NC=8 and C=32/NC=1).
Same docker nvcc + symbol-check pattern as build_p7b.py. Warp-token names.
smem: pfca c64 41088B (EAGER-ONLY, solve phase) / c32 19072B; pfcb 34336B
(c64) / 32288B (c32); pfcz 0.
"""
import subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True)
    time.sleep(2)
  return r

def build(env, name, src, extra, kname):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={kname}"] + extra + \
        ["-Xptxas", "-v", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src}"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if kname not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(kname)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

TARGETS = [
  # super-chunk tier (512 tokens = 8 x 64)
  ("pfca_c64_nc8_nw16", 0, ["-DC=64", "-DNC=8", "-DNTHR=512"]),
  ("pfcb_c64_nc8_nw8",  1, ["-DC=64", "-DNC=8", "-DNTHR=256"]),
  ("pfcz_c64_nc8_nw8",  2, ["-DC=64", "-DNC=8", "-DNTHR=256"]),
  # single-chunk gate tier (C=64 NC=1)
  ("pfca_c64_nc1_nw16", 0, ["-DC=64", "-DNC=1", "-DNTHR=512"]),
  ("pfcb_c64_nc1_nw8",  1, ["-DC=64", "-DNC=1", "-DNTHR=256"]),
  ("pfcz_c64_nc1_nw8",  2, ["-DC=64", "-DNC=1", "-DNTHR=256"]),
  # P7e super-chunk tier (256 tokens = 4 x 64)
  ("pfca_c64_nc4_nw16", 0, ["-DC=64", "-DNC=4", "-DNTHR=512"]),
  ("pfcb_c64_nc4_nw8",  1, ["-DC=64", "-DNC=4", "-DNTHR=256"]),
  ("pfcz_c64_nc4_nw8",  2, ["-DC=64", "-DNC=4", "-DNTHR=256"]),
  # fwd32 drop-in tier (one 32-token chunk per pass)
  ("pfcbdbg_c64_nc1_nw8", 3, ["-DC=64", "-DNC=1", "-DNTHR=256"]),
  ("pfca_c32_nc1_nw16", 0, ["-DC=32", "-DNC=1", "-DNTHR=512"]),
  ("pfcb_c32_nc1_nw8",  1, ["-DC=32", "-DNC=1", "-DNTHR=256"]),
  ("pfcz_c32_nc1_nw8",  2, ["-DC=32", "-DNC=1", "-DNTHR=256"]),
  # R2: the M64-trunk tier (one 64-row chunk = 2 C=32 sub-chunks)
  ("pfca_c32_nc2_nw16", 0, ["-DC=32", "-DNC=2", "-DNTHR=512"]),
  ("pfcb_c32_nc2_nw8",  1, ["-DC=32", "-DNC=2", "-DNTHR=256"]),
  ("pfcz_c32_nc2_nw8",  2, ["-DC=32", "-DNC=2", "-DNTHR=256"]),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  ok = True
  for name, ksel, extra in TARGETS:
    if only and not any(o in name for o in only): continue
    ok &= build(env, name, "pf_scanchunk.cu", extra + [f"-DKSEL={ksel}"], name)
  sys.exit(0 if ok else 1)

if __name__ == "__main__":
  main()
