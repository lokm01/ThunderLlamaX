# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2G: build HMMA K1 cubins (spk_g4hm.cu) for both ROWS builds. 100k only
(S=256, NW=32). Per-kernel cubin law + cuobjdump symbol check in-container."""
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

def build_hm(env, name, ctxk, rows, S, nw=32, tile=32):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", f"-DS={S}", f"-DCH={ctxk//S}", f"-DTILE={tile}",
         f"-DNW={nw}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/spk_g4hm.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l or "spill" in l]
  print(f"[build] {name} OK (ROWS={rows}) {' | '.join(regs[-3:])}", flush=True)
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if name not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(name)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); sys.exit(1)
  print(f"[build] {name} symbol OK", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  C100 = 100352
  build_hm(env, "spk_g4nw32hm3_100k", C100, 3, 256, 32)
  build_hm(env, "spk_g4nw32hm1_100k", C100, 1, 256, 32)
  print("[build_hm done]", flush=True)

if __name__ == "__main__":
  main()
