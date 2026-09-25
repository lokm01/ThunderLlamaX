# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Split gdn_block.cu into per-kernel .cu files and compile each to its own cubin
(multi-kernel cubins mis-load on this dext: identical code faults from the 10-kernel
cubin while per-family cubins pass)."""
import re, subprocess, os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{BASE}/gdn_block.cu").read()
hdr = src[:src.find("extern \"C\" __global__")]
ks = re.findall(r"extern \"C\" __global__ void __launch_bounds__\(\d+\) (\w+)\(", src)
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
bodies = re.split(r"(?=extern \"C\" __global__ void)", src[src.find("extern \"C\" __global__"):])
bodies = [b for b in bodies if b.strip()]
assert len(bodies) == len(ks), (len(bodies), len(ks))
for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    r = subprocess.run(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{name}.cu"],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"[build] {name} FAIL\n{r.stderr[-800:]}"); sys.exit(1)
    print(f"[build] {name} OK", flush=True)
print("[build all done]", flush=True)
