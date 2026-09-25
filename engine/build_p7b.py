# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-B builder: pf_gemm3 (single-class) + pf_gemm3m (merged twin) cubins.
Same docker nvcc + symbol-check pattern as build_p6.py. Per-kernel cubin law
+ warp-token names (nw4/6/8/16 = 128/192/256/512 threads) + r7/m32/m64 tokens.
smem notes: MTILE*XS_LD + (FFN?2:1)*NTILE*WS_LD halfs; <=36864B = in-graph
law; 43520B = eager-only (P6 precedent).
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
R7 = ["-DQCLASS=1", "-DREPACK=1"]
CL = ["-DQCLASS=1", "-DREPACK=0"]
TARGETS = [
  # ---- single-class gemm3, MT=32 (the pf_fwd32 drop-in tier) ----
  ("pfg3_ffn_r7_m32_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32", "-DFFN=1", "-DRES=0"] + HM, None),
  ("pfg3_iq3d_r7_m32_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32", "-DFFN=0", "-DRES=1"] + HM, None),
  ("pfg3_iq3o_r7_m32_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=6144", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32", "-DFFN=0", "-DRES=0"] + HM, None),
  # ---- single-class gemm3, MT=64 (the W-resident tier) ----
  ("pfg3_ffn_r7_m64_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=64", "-DFFN=1", "-DRES=0"] + HM, None),
  ("pfg3_ffn_r7_m64_nw6k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=192", "-DNTILE=48", "-DKCH=128", "-DMTILE=64", "-DFFN=1", "-DRES=0"] + HM, None),
  ("pfg3_iq3d_r7_m64_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=1"] + HM, None),
  ("pfg3_iq3o_r7_m64_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=6144", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=0"] + HM, None),
  # ---- classic controls (REPACK=0): template sanity m32 + M64 amortization-only ----
  ("pfg3_ffn_cl_m32_nw8k128", "pf_gemm3.cu", CL + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32", "-DFFN=1", "-DRES=0"] + HM, None),
  ("pfg3_ffn_cl_m64_nw6k128", "pf_gemm3.cu", CL + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=192", "-DNTILE=48", "-DKCH=128", "-DMTILE=64", "-DFFN=1", "-DRES=0"] + HM, None),
  ("pfg3_iq3d_cl_m64_nw8k128", "pf_gemm3.cu", CL + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=1"] + HM, None),
  # ---- merged twins: gdnqg (q5 classic + gate r7) ----
  ("pfg3m_gdnqg_r7_m32_nw16k128", "pf_gemm3m.cu",
   ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=512", "-DNTILE=128", "-DKCH=128", "-DMTILE=32"] + HM +
   ["-DQCA=2", "-DNDA=10240", "-DGRID_A=80", "-DRPA=0", "-DQCB=1", "-DNDB=6144", "-DGRID_B=48", "-DRPB=1"], None),
  ("pfg3m_gdnqg_r7_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"] + HM +
   ["-DQCA=2", "-DNDA=10240", "-DGRID_A=160", "-DRPA=0", "-DQCB=1", "-DNDB=6144", "-DGRID_B=96", "-DRPB=1"], None),
  # ---- merged twins: attnqkv i3 (q r7 + k r7 + v q4k classic) ----
  ("pfg3m_attnqkvi3_r7_m32_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32"] + HM +
   ["-DQCA=1", "-DNDA=12288", "-DGRID_A=192", "-DRPA=1", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"], None),
  ("pfg3m_attnqkvi3_r7_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"] + HM +
   ["-DQCA=1", "-DNDA=12288", "-DGRID_A=192", "-DRPA=1", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"], None),
  # ---- merged twins: attnqkv q6 (q6 classic + k r7 + v q4k classic) ----
  ("pfg3m_attnqkvq6_r7_m32_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32"] + HM +
   ["-DQCA=3", "-DNDA=12288", "-DGRID_A=192", "-DRPA=0", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16", "-DRPB=1",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16", "-DRPC=0"], None),
  ("pfg3m_attnqkvq6_r7_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"] + HM +
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
  print(f"[build_p7b done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
