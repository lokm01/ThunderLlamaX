# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Final persistence check: relaunch + snapshot_load(g4snap) + generate 5 ==
the A3 continuation recorded in gate4b ([6545, 9956, 6545, 9956, 9956...])."""
import os, sys, json, socket, time
import numpy as np
SOCK = "/tmp/llm-engine.sock"
class C:
  def __init__(s):
    s.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.s.connect(SOCK); s.buf = b""
  def send(s, o): s.s.sendall((json.dumps(o) + "\n").encode())
  def recv(s, t=300):
    s.s.settimeout(t)
    while b"\n" not in s.buf:
      ch = s.s.recv(65536)
      if not ch: raise EOFError
      s.buf += ch
    l, s.buf = s.buf.split(b"\n", 1); return json.loads(l)
  def rpc(s, rid, m, p=None):
    s.send({"id": rid, "method": m, "params": p or {}})
    while True:
      r = s.recv()
      if r.get("id") == rid and "event" not in r: return r
for _ in range(240):
  try:
    c = C(); r = c.rpc(1, "status", timeout=5)
    if r.get("ok") and r["result"]["ready"]: break
  except Exception: pass
  time.sleep(10)
else: sys.exit("never up")
r = c.rpc(2, "snapshot_load", {"path": "/tmp/g4snap"}, timeout=600)
print("[persist] load:", r.get("result", r))
c.send({"id": 3, "method": "generate", "params": {"max_cycles": 5}})
toks = []
while True:
  r = c.recv()
  if r.get("id") != 3: continue
  if r.get("event") == "cycle": toks += r["tokens"]
  elif r.get("event") in ("done", "cancelled"): break
print(f"[persist] regenerated: {toks}")
A3 = [6545, 9956, 6545, 9956, 6545]
ok = toks[:5] == A3
print(f"[persist] == gate4b A3 continuation: {ok} -> {'PASS' if ok else 'FAIL'}")
c.rpc(4, "shutdown", timeout=60)
print("[persist] shutdown sent", flush=True)
