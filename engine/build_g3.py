# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Build SKV-G3 cubins (spk_g3.cu; per-kernel cubin law). -D bakes KNAME/CTXK/
ROWS/S/CH/TILE/LB. Prints ptxas register/smem usage per kernel."""
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

def build(env, name, ctxk, rows, S, tile=32, lb=0):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", f"-DS={S}", f"-DCH={ctxk//S}", f"-DTILE={tile}",
         f"-DLB={lb}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/spk_g3.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  if os.path.exists(f"{BASE}/{name}.cu"): os.remove(f"{BASE}/{name}.cu")
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l]
  print(f"[build] {name} OK (ROWS={rows} S={S} LB={lb}) {' | '.join(regs[-2:])}", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  C2, C100 = 2304, 100352
  jobs = [
    # 2k py-validation (S=32 == trunk default)
    ("spk_g3a3_2k",  C2,  3, 32, 32, 0),
    ("spk_g3a1_2k",  C2,  1, 32, 32, 0),
    # 100k sweep (ROWS=3): S variants + LB variants at S=128
    ("spk_g3s64a3_100k",  C100, 3, 64,  32, 0),
    ("spk_g3a3_100k",     C100, 3, 128, 32, 0),
    ("spk_g3s256a3_100k", C100, 3, 256, 32, 0),
    ("spk_g3l2a3_100k",   C100, 3, 128, 32, 2),
    ("spk_g3l3a3_100k",   C100, 3, 128, 32, 3),
    ("spk_g3l3s64a3_100k",  C100, 3, 64, 32, 3),
    ("spk_g3l3s256a3_100k", C100, 3, 256, 32, 3),
    # 100k integration pair (trunk T=1)
    ("spk_g3a1_100k", C100, 1, 128, 32, 0),
    ("spk_g3l3a1_100k", C100, 1, 128, 32, 3),
  ]
  for j in jobs: build(env, *j)
  print("[build_g3 done]", flush=True)

if __name__ == "__main__":
  main()
