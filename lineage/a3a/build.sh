# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/sh
# A3a build: compile the hand CUDA kernels to sm_86 cubins with the nvcc docker shim.
# NOTE: the shim execs inside the cuda-nvcc-persistent container, which only mounts
# $HOME and /var/folders -> ALWAYS pass ABSOLUTE paths under $HOME.
set -e
D=$(cd "$(dirname "$0")" && pwd)          # .../tinygrad-metal/a3a
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
nvcc -arch=sm_86 -cubin -o "$D/iq3.cubin" "$D/iq3_gemv.cu"
nvcc -arch=sm_86 -cubin -o "$D/smoke.cubin" "$D/smoke.cu"
ls -la "$D"/*.cubin
