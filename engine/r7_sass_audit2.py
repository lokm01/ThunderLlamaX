# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7 DECIDER 2 (fixed): SASS load-width audit — simple opcode grep histogram."""
import subprocess, os, re, collections, json, time

BASE = "~/tinygrad-metal/engine0"
ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")

DECODE = [
  ("ffn8v3r7","gemv_ffn",64), ("down8nw32v3r7","gemv_down",64), ("q5g8v_3","gemv_qkv",48),
  ("op38nw32_3","gemv_out_i3",34), ("k3aonw32_3","gemv_out_q6",14), ("ao8nw32_3","gemv_o",16),
  ("aq3k8v_3","gemm_attnqkv",12), ("aq6k8_3","gemm_attnqkv_q6",4), ("head8v_3","gemv_head",1),
  ("ffn8v8r7","gemv_ffn8",64), ("down8nw32v8r7","gemv_down8",64), ("q5g8v8","gemv_qkv8",48),
  ("op38nw32_8","gemv_out8_i3",34), ("k3aonw32_8","gemv_out8_q6",14), ("ao8nw32_8","gemv_o8",16),
  ("aq3k8v8","gemm_attnqkv8",12), ("aq6k8v8","gemm_attnqkv8_q6",4), ("head8v8","gemv_head8",1),
]
PREFILL = [
  ("pfg3_ffn_r7_m64_nw4k128","pf_ffn",128), ("pfg3_iq3d_r7_m64_nw8k128","pf_fd",64),
  ("pfg3_iq3o_r7_m64_nw8k128","pf_out",24), ("pfg3m_gdnqg_r7q4_m64_nw8k128","pf_qg",48),
  ("pfg3m_attnqkvi3_r7_m64_nw8k128","pf_qkv",16), ("pfg3m_attnqkvq6_r7_m64_nw8k128","pf_qkv6",16),
  ("pfg3_ffn_r7_m32_nw8k128","pf_ffn32",2), ("pfg3_iq3d_r7_m32_nw8k128","pf_fd32",2),
  ("pfg3_iq3o_r7_m32_nw8k128","pf_out32",2),
]
OP_RE = re.compile(r"\b(LDG(?:STS)?(?:\.[A-Z0-9]+)+|LDS(?:\.[A-Z0-9]+)+|LDSM(?:\.[A-Z0-9]+)+|STG(?:\.[A-Z0-9]+)+|HMMA(?:\.[A-Z0-9]+)+)\b")

def width_b(op):
  m = re.search(r"\.(256|128|96|64)$", op)
  if m:
    return int(m.group(1)) // 8
  if op.endswith(".U16") or op.endswith(".S16"):
    return 2
  if op.endswith(".U8") or op.endswith(".S8"):
    return 1
  if op.endswith(".U64") or op.endswith(".S64"):
    return 8
  return 4

def audit(cubin):
  r = None
  for t in range(4):
    r = subprocess.run(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-sass", f"{BASE}/{cubin}.cubin"],
                       capture_output=True, text=True, env=ENV)
    if r.returncode == 0 or "failed to connect" not in (r.stderr or ""):
      break
    time.sleep(2)
  if r.returncode:
    return None
  counts = collections.Counter()
  for ln in r.stdout.splitlines():
    for op in OP_RE.findall(ln):
      counts[op] += 1
  return counts

def main():
  rows = []
  for cubin, cls, nl in DECODE + PREFILL:
    counts = audit(cubin)
    if counts is None:
      print(f"[sass] {cubin}: FAIL", flush=True); continue
    ldg = {op: n for op, n in counts.items() if op.startswith("LDG")}
    hmma = sum(n for op, n in counts.items() if op.startswith("HMMA"))
    lds = sum(n for op, n in counts.items() if op.startswith("LDS") and not op.startswith("LDGSTS"))
    tot = sum(ldg.values()); wsum = sum(width_b(op) * n for op, n in ldg.items())
    mean_w = wsum / max(tot, 1)
    detail = " ".join(f"{op} x{n}" for op, n in sorted(ldg.items(), key=lambda x: -x[1]))
    print(f"[sass] {cubin:36s} {cls:15s} n/cyc={nl!s:>4} HMMA={hmma:3d} LDS={lds:3d} meanLDGw={mean_w:5.2f}B | {detail}", flush=True)
    rows.append({"cubin": cubin, "cls": cls, "n_per_cyc": nl, "hmma": hmma, "lds": lds, "ldg": dict(ldg)})
  with open(os.path.expanduser("~/r7_sass_audit.json"), "w") as f:
    json.dump(rows, f, indent=1)
  print("[sass] json -> ~/r7_sass_audit.json", flush=True)

if __name__ == "__main__":
  main()
