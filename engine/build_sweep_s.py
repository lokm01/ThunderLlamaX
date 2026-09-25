# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
import build_pf2
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
build_pf2.sh(["docker", "ps"], env)
for S in (64, 128, 256):
  build_pf2.build(env, f"pfa16nw32_s{S}_100k", "pf_attn.cu", ["-DCTXK=100352", f"-DS={S}", "-DKSEL=1"], "pfa16")
  build_pf2.build(env, f"pfc16_s{S}", "pf_attn.cu", [f"-DS={S}", "-DKSEL=2"], "pfc16")
