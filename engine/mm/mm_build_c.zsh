#!/bin/zsh
# MM SESSION C -- build + audit the Session-C cubins (nvcc via the colima
# container; NO GPU needed). Root-level kernels (the production location --
# Rig7 loads from BASE). Audit = cuobjdump -res-usage (regs, smem, spill --
# the zero-spill law). THE NAME LAW: program name == symbol.
# Usage: zsh mm_build_c.zsh          (idempotent; MM_C_REBUILD=1 forces)
#       zsh mm_build_c.zsh pgmq8m32  (single kernel)
set -u
BASE=~/tinygrad-metal
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin
export DOCKER_HOST=unix://~/.colima/default/docker.sock

only=${1:-}
build() {  # build <out.cubin> <src.cu> [extra flags...]  (ABSOLUTE paths --
           # the nvcc shim execs inside the container: relative paths die)
  local out=$BASE/$1; shift
  local src=$BASE/$1; shift
  if [[ -n $only ]] && [[ ${out:t} != *${only}* ]]; then return; fi
  if [[ ! -f $out ]] || [[ ${MM_C_REBUILD:-0} == 1 ]]; then
    echo "[build] ${out:t}"
    nvcc -arch=sm_86 -cubin -fmad=false --output-file=$out $src "$@" || { echo "BUILD FAIL $out"; exit 1; }
  fi
  echo "[audit] ${out:t}"
  cuobjdump -res-usage $out | grep -E "Function|REG|STACK|SHARED|spill" | sed "s|^|  ${out:t}: |"
}

# L1b: the trunk out/o mma M-GEMM (the 65% residual item)
build MM_C_pgmq8m32.cubin MM_C_pgmq8m32.cu
# routed-dn act-restaging fold (bit-exact route)
build MM_C_gxm_dnf.cubin   MM_C_gxm_dnf.cu
echo "[build-c] done"
