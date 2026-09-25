# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-C minimal fault reproducer: prefill FRESH + one long generate.
Pure engine-direct (no API, no cancel, no disconnect). Faulted the dext at
~868 / ~971 cycles (48-49.5s) on the 2MB-ring boots. Usage: longgen.py [cycles]"""
import json, socket, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro import C, rpc, IDS

def log(*a):
  line = " ".join(str(x) for x in a)
  print(line, flush=True)
  try:
    os.makedirs("~/logs", exist_ok=True)
    with open("~/logs/longgen.log", "a") as f: f.write(line + "\n")
  except Exception: pass

c = C()
r = rpc(c, 1, "prefill", {"mode": "FRESH", "ids": IDS, "conversation_id": "longgen"})
log("prefill", r)
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
