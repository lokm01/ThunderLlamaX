#!/bin/zsh
# MM SESSION D -- build + audit the Session-D cubins (nvcc via the colima
# container; NO GPU needed). Root-level kernels (the production location --
# Rig7 loads from BASE). Audit = cuobjdump -res-usage (regs, smem, spill --
# the zero-spill law). THE NAME LAW: program name == symbol.
# Usage: zsh mm_build_d.zsh          (idempotent; MM_D_REBUILD=1 forces)
#       zsh mm_build_d.zsh spkqw4    (single kernel)
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
  if [[ ! -f $out ]] || [[ ${MM_D_REBUILD:-0} == 1 ]]; then
    echo "[build] ${out:t}"
    nvcc -arch=sm_86 -cubin -fmad=false --output-file=$out $src "$@" || { echo "BUILD FAIL $out"; exit 1; }
  fi
  echo "[audit] ${out:t}"
  cuobjdump -res-usage $out | grep -E "Function|REG|STACK|SHARED|spill" | sed "s|^|  ${out:t}: |"
}

# L4: the row-grouped wide PF attention (the 96k lever) -- RW variants
build MM_D_spkqw4_98304.cubin MM_D_spkqw.cu -DCTXS=98304 -DRW=4
build MM_D_spkqw8_98304.cubin MM_D_spkqw.cu -DCTXS=98304 -DRW=8
echo "[build-d] done"
