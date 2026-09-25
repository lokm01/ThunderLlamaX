# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Build SKV-G4 cubins (spk_g4.cu; fat-CTA NW parameter). Per-kernel cubin law."""
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

def build(env, name, ctxk, rows, S, nw=8, tile=32):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", f"-DS={S}", f"-DCH={ctxk//S}", f"-DTILE={tile}",
         f"-DNW={nw}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/spk_g4.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  if os.path.exists(f"{BASE}/{name}.cu"): os.remove(f"{BASE}/{name}.cu")
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l]
  print(f"[build] {name} OK (ROWS={rows} S={S} NW={nw}) {' | '.join(regs[-2:])}", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  C2, C100 = 2304, 100352
  jobs = [
    # 2k py-validation (S=32; NW=16 validates the fat-CTA path numerically)
    ("spk_g4nw16a3_2k",  C2,  3, 32, 16),
    ("spk_g4nw16a1_2k",  C2,  1, 32, 16),
    # 100k sweep (ROWS=3)
    ("spk_g4nw8s256a3_100k",  C100, 3, 256, 8),   # control vs G3-LB3
    ("spk_g4nw16s128a3_100k", C100, 3, 128, 16),
    ("spk_g4nw16s256a3_100k", C100, 3, 256, 16),
    ("spk_g4nw16s512a3_100k", C100, 3, 512, 16),
    ("spk_g4nw24s256a3_100k", C100, 3, 256, 24),  # 768-thr even fatter
  ]
  for j in jobs: build(env, *j)
  print("[build_g4 done]", flush=True)

if __name__ == "__main__":
  main()
