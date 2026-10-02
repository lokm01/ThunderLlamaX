#!/bin/zsh
# MM SESSION A -- build + audit the bench cubins (nvcc via the colima
# container; NO GPU needed -- run while the daemon still serves).
# Everything lands in engine0/mm/. Audit = cuobjdump -res-usage (regs,
# smem, spill). THE NAME LAW: every program name == its cubin symbol.
set -u
BASE=~/tinygrad-metal
MM=$BASE/engine0/mm
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin
export DOCKER_HOST=unix://~/.colima/default/docker.sock

cd $MM
build() {  # build <out.cubin> <src.cu> [extra flags...]  (ABSOLUTE paths --
           # the nvcc shim execs inside the container: relative paths die)
  local out=$MM/$1; shift
  local src=$MM/$1; shift
  if [[ ! -f $out ]] || [[ ${MM_A_REBUILD:-0} == 1 ]]; then
    echo "[build] ${out:t}"
    nvcc -arch=sm_86 -cubin -fmad=false --output-file=$out $src "$@" || { echo "BUILD FAIL $out"; exit 1 }
  fi
}

build MM_A_gv8k2048ps.cubin MM_A_gv8k2048ps.cu
build MM_A_gv8k4096rs.cubin MM_A_gv8k4096rs.cu
for swbc in "8 8" "16 8" "32 8" "64 4"; do
  set -- $=swbc
  build MM_A_gvsl_${1}_${2}.cubin MM_A_gvsl.cu -DSW=${1} -DBC=${2}
done
build MM_A_cp4k.cubin MM_A_cp4k.cu
build MM_A_dsmemp.cubin MM_A_dsmemp.cu
build MM_A_bwread.cubin MM_A_bwread.cu
build MM_A_bwread_ldg.cubin MM_A_bwread.cu -DLDG=1

echo "[audit] res-usage:"
for f in MM_A_*.cubin; do
  echo "--- $f"
  cuobjdump -res-usage $f 2>/dev/null | sed -n '/Function :\|REG:/p;/STACK:/p;/SHARED:/p' | head -8
done

echo "[manifest] sha256:"
shasum -a 256 MM_A_*.cubin MM_A_*.cu | tee $MM/mm_a_build_manifest.txt
echo "[build] done"
