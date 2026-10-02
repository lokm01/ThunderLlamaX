#!/bin/zsh
# MM SESSION B -- build + audit the Session-B production cubins (nvcc via the
# colima container; NO GPU needed). Root-level kernels (the production
# location -- Rig7 loads from BASE). Audit = cuobjdump -res-usage (regs,
# smem, spill -- the zero-spill law). THE NAME LAW: program name == symbol.
# Usage: zsh mm_build_b.zsh          (idempotent; MM_B_REBUILD=1 forces)
#       zsh mm_build_b.zsh gxm_up    (single kernel)
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
  if [[ ! -f $out ]] || [[ ${MM_B_REBUILD:-0} == 1 ]]; then
    echo "[build] ${out:t}"
    nvcc -arch=sm_86 -cubin -fmad=false --output-file=$out $src "$@" || { echo "BUILD FAIL $out"; exit 1 }
  fi
  echo "[audit] ${out:t}"
  cuobjdump -res-usage $out | grep -E "Function|REG|STACK|SHARED|spill" | sed "s|^|  ${out:t}: |"
}

# L2: the grouped expert path (TS=16 RS=4 GN=1024 the Session-B defaults)
build MM_B_mmsort8.cubin MM_B_mmsort8.cu
build MM_B_gxm_up.cubin  MM_B_gxm_up.cu
build MM_B_gxm_up4.cubin MM_B_gxm_up4.cu
build MM_B_gxm_dn.cubin  MM_B_gxm_dn.cu
build MM_B_gxm_dn6.cubin MM_B_gxm_dn6.cu
# L1: the seat-loop trunk ports
build MM_B_gvs32.cubin   MM_B_gvs32.cu
build MM_B_gvs32r.cubin  MM_B_gvs32r.cu
build MM_B_gvsab.cubin   MM_B_gvsab.cu
# L3: the shared-expert M-batch pair
build MM_B_shgu32.cubin  MM_B_shgu32.cu
build MM_B_shdn32.cubin  MM_B_shdn32.cu
# L5: the PF64 tail scan pair (existing P34 sources, TMAX=64)
build MM_P34_gconv36_64.cubin MM_P34_gconv36.cu -DTMAX=64
build MM_P34_k2s36_64.cubin   MM_P34_k2s36.cu   -DTMAX=64
echo "[build-b] done"
