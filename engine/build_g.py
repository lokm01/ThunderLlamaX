# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Build SKV-G cubins (spk_g.cu; per-kernel cubin law). -D bakes KNAME/CTXK/
ROWS/S/CH/TILE/VARIANT/PF/D2. Prints ptxas register/smem usage per kernel."""
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

def build(env, name, ctxk, rows, S, tile=32, variant=1, pf=0, d2=0):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", f"-DS={S}", f"-DCH={ctxk//S}", f"-DTILE={tile}",
         f"-DVARIANT={variant}", f"-DPF={pf}", f"-DD2={d2}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/spk_g.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  # cubin kept as the ONLY artifact; the .cu is the single source of truth
  if os.path.exists(f"{BASE}/{name}.cu"): os.remove(f"{BASE}/{name}.cu")
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l]
  print(f"[build] {name} OK (ROWS={rows} S={S} TILE={tile} V={variant} PF={pf} D2={d2}) {' | '.join(regs[-2:])}", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  C2, C100 = 2304, 100352
  jobs = [
    # 2k py-validation + 2k integration (S=32 == trunk default)
    ("spk_g1a3_2k",  C2,  3, 32, 32, 0, 0, 0),
    ("spk_g2a3_2k",  C2,  3, 32, 32, 1, 0, 0),
    ("spk_g1a1_2k",  C2,  1, 32, 32, 0, 0, 0),
    ("spk_g2a1_2k",  C2,  1, 32, 32, 1, 0, 0),
    # 100k sweep (ROWS=3) — spec order: G1-S64, G1-S128, G2-S32, G2-S64, G2-S128,
    # G2-S128-PF, G2-S128-D2, G2-S256, T64-substitute G2-S64-T16
    ("spk_g1s64a3_100k",  C100, 3, 64,  32, 0, 0, 0),
    ("spk_g1s128a3_100k", C100, 3, 128, 32, 0, 0, 0),
    ("spk_g2s32a3_100k",  C100, 3, 32,  32, 1, 0, 0),
    ("spk_g2s64a3_100k",  C100, 3, 64,  32, 1, 0, 0),
    ("spk_g2a3_100k",     C100, 3, 128, 32, 1, 0, 0),
    ("spk_g2pfa3_100k",   C100, 3, 128, 32, 1, 1, 0),
    ("spk_g2d2a3_100k",   C100, 3, 128, 16, 1, 0, 1),
    ("spk_g2s256a3_100k", C100, 3, 256, 32, 1, 0, 0),
    ("spk_g2t16s64a3_100k", C100, 3, 64, 16, 1, 0, 0),
    # 100k integration pair (S=128 == W3 production SKV_S)
    ("spk_g2a1_100k", C100, 1, 128, 32, 1, 0, 0),
    ("spk_g1a1_100k", C100, 1, 128, 32, 0, 0, 0),
  ]
  for j in jobs: build(env, *j)
  print("[build_g done]", flush=True)

if __name__ == "__main__":
  main()
