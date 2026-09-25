# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c M=128 builder: the m128 GEMM generation (MTILE=128, NTILE=32, nw4 =
43520B smem — legal to 48KB static per the P17 w64 precedent), the split-plane
plain FFN GEMMs + the silu-mul epilogue, the WY-C32 NC=4 scan tier, and the
w128h wide attention (ROWS=128 HRP=1, RMAX=128 = the w64's 48KB-exact smem).
Same docker nvcc + symbol-check pattern as build_p7b.py."""
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

HM = ["-DHMMA=1"]
R7 = ["-DQCLASS=1", "-DREPACK=1"]
C100 = ["-DCTXK=100352", "-DKV8=1", "-DQH=1"]

TARGETS = [
  # ---- single-class m128 (MTILE=128 NTILE=32 nw4; smem 43520B) ----
  ("pfg3_iq3d_r7_m128_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128", "-DFFN=0", "-DRES=1"] + HM, None),
  ("pfg3_iq3o_r7_m128_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=6144", "-DNDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128", "-DFFN=0", "-DRES=0"] + HM, None),
  # split-plane FFN: plain (half)acc outputs, silu-mul applied by pfk_smul128
  ("pfg3_fgp_r7_m128_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128", "-DFFN=0", "-DRES=0"] + HM, None),
  ("pfg3_fup_r7_m128_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128", "-DFFN=0", "-DRES=0"] + HM, None),
  ("pfk_smul128", "pfk_smul.cu", [], None),
  # ---- merged twins m128 (NTILE=32 nw4) ----
  ("pfg3m_gdnqg_r7_m128_nw4k128", "pf_gemm3m.cu",
   ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128"] + HM +
   ["-DQCA=2", "-DNDA=10240", "-DGRID_A=320", "-DRPA=0", "-DQCB=1", "-DNDB=6144", "-DGRID_B=192", "-DRPB=1"], None),
  ("pfg3m_attnqkvi3_r7_m128_nw4k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128"] + HM +
   ["-DQCA=1", "-DNDA=12288", "-DGRID_A=384", "-DRPA=1", "-DQCB=1", "-DNDB=1024", "-DGRID_B=32", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=32", "-DRPC=0"], None),
  ("pfg3m_attnqkvq6_r7_m128_nw4k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=128"] + HM +
   ["-DQCA=3", "-DNDA=12288", "-DGRID_A=384", "-DRPA=0", "-DQCB=1", "-DNDB=1024", "-DGRID_B=32", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=32", "-DRPC=0"], None),
  # ---- WY-C32 NC=4 scan tier (one triple per 128-row chunk) ----
  ("pfca_c32_nc4_nw16", "pf_scanchunk.cu", ["-DC=32", "-DNC=4", "-DNTHR=512", "-DKSEL=0"], None),
  ("pfcb_c32_nc4_nw8", "pf_scanchunk.cu", ["-DC=32", "-DNC=4", "-DNTHR=256", "-DKSEL=1"], None),
  ("pfcz_c32_nc4_nw8", "pf_scanchunk.cu", ["-DC=32", "-DNC=4", "-DNTHR=256", "-DKSEL=2"], None),
  # ---- w128h wide attention (ROWS=128 HRP=1 NW=16; RMAX=128 = 48KB exact) ----
  ("pfaw_w128h_s13_100k", "pf_attnw.cu", C100 + ["-DS=13", "-DROWS=128", "-DHRP=1", "-DNW=16", "-DMINB=1"], "pfaw128h"),
  ("pfcw128h_s13", "pf_attnw.cu", C100 + ["-DS=13", "-DROWS=128", "-DHRP=1", "-DPFC_T32=1"], "pfcw128h"),
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
  print(f"[build_r2c128 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
