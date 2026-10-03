#!/bin/zsh
# MM SESSION E -- build + audit the Session-E cubins (nvcc via the colima
# container; NO GPU needed). Audit = cuobjdump -res-usage (zero-spill law).
# THE NAME LAW: program name == symbol.  Usage: zsh mm_build_e.zsh [kernel]
set -u
BASE=~/tinygrad-metal
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin
export DOCKER_HOST=unix://~/.colima/default/docker.sock

only=${1:-}
build() {
  local out=$BASE/$1; shift
  local src=$BASE/$1; shift
  if [[ -n $only ]] && [[ ${out:t} != *${only}* ]]; then return; fi
  if [[ ! -f $out ]] || [[ ${MM_E_REBUILD:-0} == 1 ]]; then
    echo "[build] ${out:t}"
    nvcc -arch=sm_86 -cubin -fmad=false --output-file=$out $src "$@" || { echo "BUILD FAIL $out"; exit 1; }
  fi
  echo "[audit] ${out:t}"
  cuobjdump -res-usage $out | grep -E "Function|REG|STACK|SHARED|spill" | sed "s|^|  ${out:t}: |"
}

# E1: the gathered-row-list mma routed gate+up
build MM_E_gxu_gm.cubin MM_E_gxu_gm.cu
# E2: the M-seat batched router (BIT-EXACT gold-router contract)
build MM_E_rt8e_m2.cubin MM_E_rt8e_m.cu -DSEATS=2
build MM_E_rt8e_m4.cubin MM_E_rt8e_m.cu -DSEATS=4
if [[ -z $only ]]; then :; fi
echo "[build-e] done"
# E1b: the gathered dn mma
build MM_E_gxd_gm.cubin MM_E_gxd_gm.cu
# E2: the shared-expert mma pair (ONE SYMBOL PER CUBIN -- the loader law;
# the 2-symbol cubin mispicked program metadata -> OOR faults)
build MM_E_shgm512.cubin MM_E_shgm512.cu
build MM_E_sdm2048.cubin MM_E_sdm2048.cu
# E3: the split-row scan + the norm-apply epilogue
build MM_E_k2s36h_256.cubin MM_E_k2s36h.cu -DTMAX=256
build MM_E_k2s36h_64.cubin MM_E_k2s36h.cu -DTMAX=64
build MM_E_k2nz36.cubin MM_E_k2nz36.cu
