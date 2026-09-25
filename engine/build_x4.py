# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import subprocess, os, sys, time
BASE = "~/tinygrad-metal/engine0"
ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
R7 = ["-DQCLASS=1", "-DREPACK=1"]
HM = ["-DHMMA=1"]
T = [
  ("pfg3_ffn_r7_m64_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=64", "-DFFN=1", "-DRES=0"] + HM),
  ("pfg3_iq3d_r7_m64_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=1"] + HM),
  ("pfg3_iq3o_r7_m64_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=6144", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=0"] + HM),
  ("pfg3m_gdnqg_r7q4_m64_nw8k128", "pf_gemm3m.cu", ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DRING4=1"] + HM +
   ["-DQCA=2", "-DNDA=10240", "-DGRID_A=160", "-DRPA=0", "-DQCB=1", "-DNDB=6144", "-DGRID_B=96", "-DRPB=1"]),
  ("pfg3m_attnqkvi3_r7_m64_nw8k128", "pf_gemm3m.cu", ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"] + HM +
   ["-DQCA=1", "-DNDA=12288", "-DGRID_A=192", "-DRPA=1", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1", "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"]),
  ("pfg3m_attnqkvq6_r7_m64_nw8k128", "pf_gemm3m.cu", ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"] + HM +
   ["-DQCA=3", "-DNDA=12288", "-DGRID_A=192", "-DRPA=0", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1", "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"]),
]
ok = True
for nm, src, extra in T:
  bak = f"{BASE}/{nm}.cubin.prex4.bak"
  if not os.path.exists(bak):
    subprocess.run(["cp", f"{BASE}/{nm}.cubin", bak])
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={nm}"] + extra + ["-Xptxas", "-v",
        f"--output-file={BASE}/{nm}.cubin", f"{BASE}/{src}"]
  r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "spill" in l.lower()]
  if r.returncode:
    print(f"[{nm}] FAIL\n{r.stderr[-1200:]}"); ok = False
  else:
    print(f"[{nm}] OK {' | '.join(regs[-2:])}", flush=True)
sys.exit(0 if ok else 1)
