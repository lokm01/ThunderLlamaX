# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P2 builder: prefill M=16 machinery cubins (pKPRE-M / pATTN-M / pSCAN-M /
norms / draft Q4_0 pGEMM classes / RES residual-add GEMM mode). Reuses the P1
builder's docker nvcc + symbol-check pattern. Per-kernel cubin law + name-
encoded launch config tokens (nw32 -> 1024 threads).
NOTE: the symbol check matches the KERNEL symbol (kname), which for the pf_*
family is the hardcoded extern-C name (pfk_pre16, pfs16, ...) while the CUBIN
file carries the variant suffix (_2k/_100k/s32/...)."""
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

# (cubin_name, src, extra, kname)  — kname None => same as cubin name (pf_gemm KNAME macro)
TARGETS = [
  # pKPRE-M (KV8 int8-KV canonical): 2k + 100k slab strides
  ("pfk_pre16_2k",    "pf_kpre16.cu", ["-DCTXK=2304"], "pfk_pre16"),
  ("pfk_pre16_100k",  "pf_kpre16.cu", ["-DCTXK=100352"], "pfk_pre16"),
  # pATTN-M K1 (1024thr; nw32 token) + K2 combine. S=32 matches the canonical
  # spk_g4nw32qh1_100k (CH=3136) for differential validation; 2k S=8 (CH=288).
  ("pfa16nw32_s32_100k", "pf_attn.cu", ["-DCTXK=100352", "-DS=32", "-DKSEL=1"], "pfa16"),
  ("pfa16nw32_s8_2k",    "pf_attn.cu", ["-DCTXK=2304", "-DS=8", "-DKSEL=1"], "pfa16"),
  ("pfc16_s32",          "pf_attn.cu", ["-DS=32", "-DKSEL=2"], "pfc16"),
  ("pfc16_s8",           "pf_attn.cu", ["-DS=8", "-DKSEL=2"], "pfc16"),
  # pSCAN-M
  ("pfs16", "pf_scan16.cu", [], "pfs16"),
  # norms / embed / draft-cat family (one kernel per cubin)
  ("pfk_emb16",    "pf_norms16.cu", ["-DKSEL=1"], "pfk_emb16"),
  ("pfk_n16",      "pf_norms16.cu", ["-DKSEL=2"], "pfk_n16"),
  ("pfk_ab16",     "pf_norms16.cu", ["-DKSEL=3"], "pfk_ab16"),
  ("pfk_hh16",     "pf_norms16.cu", ["-DKSEL=4"], "pfk_hh16"),
  ("pfd_dnorm16",  "pf_norms16.cu", ["-DKSEL=5"], "pfd_dnorm16"),
  # draft Q4_0-repacked pGEMM classes (dfgu-verified two-region layout)
  ("pfg_q4dq_hm_nw8k128",  "pf_gemm.cu", ["-DQCLASS=7", "-DKDIM=5120",  "-DNDIM=12288", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=0"], None),
  ("pfg_q4dk_hm_nw8k128",  "pf_gemm.cu", ["-DQCLASS=7", "-DKDIM=5120",  "-DNDIM=1024",  "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=0"], None),
  ("pfg_q4dv_hm_nw8k128",  "pf_gemm.cu", ["-DQCLASS=7", "-DKDIM=5120",  "-DNDIM=1024",  "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=0"], None),
  ("pfg_q4do_hm_nw8k128",  "pf_gemm.cu", ["-DQCLASS=7", "-DKDIM=5120",  "-DNDIM=5120",  "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=0"], None),
  ("pfg_q4dd_res_hm_nw8k128", "pf_gemm.cu", ["-DQCLASS=7", "-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=1"], None),
  ("pfg_q4eh_res_hm_nw8k128", "pf_gemm.cu", ["-DQCLASS=7", "-DKDIM=10240", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=1"], None),
  # trunk down-projection RES mode (fd: IQ3_XXS 17408->5120)
  ("pfg_iq3d_res_hm_nw8k128", "pf_gemm.cu", ["-DQCLASS=1", "-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DHMMA=1", "-DFFN=0", "-DRES=1"], None),
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
  print(f"[build_pf2 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
