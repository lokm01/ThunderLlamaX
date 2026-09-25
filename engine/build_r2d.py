# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2d builder: DBUF ring-depth-4 cubins (pf_gemm3/pf_gemm3m -DRING4=1) for the
4 non-FFN m64 GEMM classes (ffn excluded: units x2 spill-blocked per R2c).
Same docker nvcc + symbol-check pattern as build_p7b.py. Names keep the
nw8k128 warp-tokens; r7q4 = ring-depth-4. Rebuild-neutral: existing targets
don't pass -DRING4 (preprocessor default 0).
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
    print(f"[build] {name} FAIL\n{r.stderr[-2500:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if kname not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(kname)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

HM = ["-DHMMA=1"]
R7Q4 = ["-DQCLASS=1", "-DREPACK=1", "-DRING4=1"]
TARGETS = [
  ("pfg3_iq3d_r7q4_m64_nw8k128", "pf_gemm3.cu",
   R7Q4 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=1"] + HM, None),
  ("pfg3_iq3o_r7q4_m64_nw8k128", "pf_gemm3.cu",
   R7Q4 + ["-DKDIM=6144", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=0"] + HM, None),
  ("pfg3m_gdnqg_r7q4_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DRING4=1"] + HM +
   ["-DQCA=2", "-DNDA=10240", "-DGRID_A=160", "-DRPA=0", "-DQCB=1", "-DNDB=6144", "-DGRID_B=96", "-DRPB=1"], None),
  ("pfg3m_attnqkvi3_r7q4_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DRING4=1"] + HM +
   ["-DQCA=1", "-DNDA=12288", "-DGRID_A=192", "-DRPA=1", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"], None),
  ("pfg3m_attnqkvq6_r7q4_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DRING4=1"] + HM +
   ["-DQCA=3", "-DNDA=12288", "-DGRID_A=192", "-DRPA=0", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"], None),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, src, extra, kname in TARGETS:
    if only and not any(o in name for o in only): continue
    if build(env, name, src, extra, kname or name): n_ok += 1
    else: n_fail += 1
  print(f"[build_r2d done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
