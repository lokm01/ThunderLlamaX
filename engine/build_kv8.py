# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2E KV8: build int8-KV cubins (spk_g4q.cu K1 + spk_preq.cu KPRE).
100k only (S=256, NW=32 — the locked canonical config). Per-kernel cubin law."""
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

def build_g4q(env, name, ctxk, rows, S, nw=32, tile=32):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", f"-DS={S}", f"-DCH={ctxk//S}", f"-DTILE={tile}",
         f"-DNW={nw}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/spk_g4q.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l]
  print(f"[build] {name} OK (ROWS={rows} S={S} NW={nw}) {' | '.join(regs[-2:])}", flush=True)

def build_preq(env, name, ctxk, rows):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/spk_preq.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l]
  print(f"[build] {name} OK (ROWS={rows}) {' | '.join(regs[-2:])}", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix:/~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  C100 = 100352
  build_preq(env, "spk_pre1q_100k", C100, 1)
  build_preq(env, "spk_pre3q_100k", C100, 3)
  build_g4q(env, "spk_g4nw32qa1_100k", C100, 1, 256, 32)
  build_g4q(env, "spk_g4nw32qa3_100k", C100, 3, 256, 32)
  print("[build_kv8 done]", flush=True)

if __name__ == "__main__":
  main()
