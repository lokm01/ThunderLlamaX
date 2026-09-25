# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-C: long generate from the canonical snapshot park (pos 97810).
Discriminates pos-boundary (~1000) vs cycle-count (~950) trigger."""
import json, socket, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro import C, rpc

def log(*a):
  line = " ".join(str(x) for x in a)
  print(line, flush=True)
  try:
    os.makedirs("~/logs", exist_ok=True)
    with open("~/logs/longgen_snap.log", "a") as f: f.write(line + "\n")
  except Exception: pass

c = C()
r = rpc(c, 1, "prefill", {"snapshot": "~/snap100k"})
log("park", r)
c.close()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
c = C()
c.send({"id": 2, "method": "generate", "params": {"max_cycles": n, "stop_token_ids": []}})
t0 = time.time(); toks = 0; k = 0
while True:
    r = c.recv()
    if r.get("id") != 2: continue
    if r.get("event") == "cycle":
        k = r["cycle"]; toks += len(r["tokens"])
        if k % 100 == 0: log(f"cycle {k} toks {toks} {time.time()-t0:.0f}s")
    elif r.get("event") in ("done", "cancelled"):
        log("END", r.get("event"), "cycles", k, "toks", toks, f"{time.time()-t0:.0f}s"); break
