# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P6 builder: M32 pGEMM cubins (32 x-rows per CTA — the load-stream
amortizer; see pf_gemm.cu P6 section). Same docker nvcc + symbol-check
pattern as build_pf.py. Per-kernel cubin law + warp-token names + m32 token.
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

# (cubin_name, src, extra, kname) — None kname => KNAME == cubin name
G16 = ["-DHMMA=1", "-DM32=1"]
TARGETS = [
  # trunk single-class M32 (KDIM/NDIM mirror the shipped M=16 builds)
  ("pfg_ffn_m32_hm_nw8k128",    "pf_gemm.cu", ["-DQCLASS=1", "-DKDIM=5120",  "-DNDIM=17408", "-DNTHR=256", "-DNTILE=64",  "-DKCH=128", "-DFFN=1", "-DRES=0"] + G16, None),
  ("pfg_iq3d_m32_res_hm_nw8k128", "pf_gemm.cu", ["-DQCLASS=1", "-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64",  "-DKCH=128", "-DFFN=0", "-DRES=1"] + G16, None),
  ("pfg_iq3s_m32_hm_nw8k128",   "pf_gemm.cu", ["-DQCLASS=5", "-DKDIM=6144",  "-DNDIM=5120",  "-DNTHR=256", "-DNTILE=64",  "-DKCH=128", "-DFFN=0", "-DRES=0"] + G16, None),
  ("pfg_iq3o_m32_hm_nw8k128",   "pf_gemm.cu", ["-DQCLASS=1", "-DKDIM=6144",  "-DNDIM=5120",  "-DNTHR=256", "-DNTILE=64",  "-DKCH=128", "-DFFN=0", "-DRES=0"] + G16, None),
  ("pfg_q8o_m32_hm_nw8k64",     "pf_gemm.cu", ["-DQCLASS=6", "-DKDIM=6144",  "-DNDIM=5120",  "-DNTHR=256", "-DNTILE=64",  "-DKCH=64",  "-DFFN=0", "-DRES=0"] + G16, None),
  # merged multi-segment M32 (GDN qkv+gate nw16; attn q+k+v nw8 — QCA per variant)
  ("pfg2_gdnqg_m32_hm_nw16k128", "pf_gemm2.cu",
   ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=512", "-DNTILE=128", "-DKCH=128", "-DM32=1",
    "-DQCA=2", "-DNDA=10240", "-DGRID_A=80", "-DQCB=1", "-DNDB=6144", "-DGRID_B=48"], None),
  ("pfg2_attnqkvq6_m32_hm_nw8k128", "pf_gemm2.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DM32=1",
    "-DQCA=3", "-DNDA=12288", "-DGRID_A=192", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16"], None),
  ("pfg2_attnqkvi3_m32_hm_nw8k128", "pf_gemm2.cu",
   ["-DSEGN=3", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DM32=1",
    "-DQCA=1", "-DNDA=12288", "-DGRID_A=192", "-DQCB=1", "-DNDB=1024", "-DGRID_B=16",
    "-DQCC=4", "-DNDC=1024", "-DGRID_C=16"], None),
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
  print(f"[build_p6 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
