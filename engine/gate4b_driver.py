# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""GATE 4b (compact socket driver): snapshot save/load round-trip (delta-window),
follow-up determinism via the socket, shutdown, relaunch persistence."""
import os, sys, json, socket, time
import numpy as np

SOCK = "/tmp/llm-engine.sock"
SNAP = "~/snap100k"
G4SNAP = "/tmp/g4snap"
delta = np.load(f"{SNAP}/gate3_delta.npy").tolist()

class Client:
  def __init__(self):
    self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.s.connect(SOCK); self.buf = b""
  def send(self, obj): self.s.sendall((json.dumps(obj) + "\n").encode())
  def recv(self, timeout=300):
    self.s.settimeout(timeout)
    while b"\n" not in self.buf:
      c = self.s.recv(65536)
      if not c: raise EOFError
      self.buf += c
    line, self.buf = self.buf.split(b"\n", 1)
    return json.loads(line)
  def rpc(self, rid, method, params=None, timeout=300):
    self.send({"id": rid, "method": method, "params": params or {}})
    while True:
      r = self.recv(timeout)
      if r.get("id") == rid and "event" not in r: return r

def gen(c, rid, max_cycles, cancel_after=None):
  toks = []; n = 0; cancelled = False
  c.send({"id": rid, "method": "generate", "params": {"max_cycles": max_cycles}})
  while True:
    r = c.recv()
    if r.get("id") != rid: continue
    ev = r.get("event")
    if ev == "cycle":
      toks += r["tokens"]; n += 1
      if cancel_after is not None and n >= cancel_after:
        c.send({"id": 99000 + rid, "method": "cancel", "params": {}})
    elif ev in ("done", "cancelled"): return toks, n, (ev == "cancelled")

FAILED = [False]
def step(name, ok, extra=""):
  print(f"[gate4b] {'PASS' if ok else 'FAIL'}: {name} {extra}", flush=True)
  if not ok: FAILED[0] = True

def connect_wait():
  for _ in range(240):
    try:
      c = Client(); r = c.rpc(8001, "status", timeout=5)
      if r.get("ok") and r["result"]["ready"]: return c
    except Exception: pass
    time.sleep(10)
  sys.exit("engine never came up")

c = connect_wait()
r = c.rpc(8002, "prefill", {"snapshot": SNAP}, timeout=300)
step("park at snapshot", r.get("ok") and r["result"]["pos"] == 97810)

A, _, _ = gen(c, 8003, 5)
r = c.rpc(8004, "snapshot_save", {"path": G4SNAP}, timeout=600)
step("snapshot_save (delta window)", r.get("ok") and r["result"]["delta_rows"][0] == 97810,
     str(r.get("result", ""))[:140])
A2, _, _ = gen(c, 8005, 5)
step("post-save advance", len(A2) >= 5)
r = c.rpc(8006, "snapshot_load", {"path": G4SNAP}, timeout=600)
step("snapshot_load", r.get("ok") and r["result"]["pos"] == 97810 + len(A), str(r.get("result", ""))[:140])
A3, _, _ = gen(c, 8007, 5)
step("regenerate after load == pre-save generate", A3[:5] == A[:5], f"{A[:5]} vs {A3[:5]}")

# follow-up determinism through the socket (same request twice)
outs = []
for t in range(2):
  c.rpc(8010 + 2*t, "prefill", {"snapshot": SNAP}, timeout=300)
  rr = c.rpc(8011 + 2*t, "prefill", {"mode": "FOLLOW_UP", "ids": delta}, timeout=600)
  B, _, _ = gen(c, 8012 + 2*t, 20)
  outs.append(B)
nmin = min(len(outs[0]), len(outs[1]))
step("FOLLOW-UP deterministic x2", outs[0][:nmin] == outs[1][:nmin],
     f"len {len(outs[0])}/{len(outs[1])}, first12 {outs[0][:12]}")

c.rpc(8020, "shutdown", timeout=60)
time.sleep(5)
alive = os.popen("pgrep -f 'test_w100k.py' | wc -l").read().strip()
step("shutdown exits", alive == "0", f"pgrep={alive}")

print("[gate4b] relaunching engine (persistence)...", flush=True)
os.system("cd ~/tinygrad-metal/engine0 && nohup env PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH "
          "DOCKER_HOST=unix://~/.colima/default/docker.sock DEV=NV SKV=1 SKV_K=g4nw32 SKV_S=256 "
          "SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 M1A_SERVE=1 ~/tg311/bin/python -u test_w100k.py "
          ">> /tmp/serve2.log 2>&1 &")
c2 = connect_wait()
r = c2.rpc(8030, "snapshot_load", {"path": G4SNAP}, timeout=600)
step("relaunch + snapshot persists (load)", r.get("ok"), str(r.get("result", ""))[:140])
A4, _, _ = gen(c2, 8031, 5)
step("relaunched regenerate == A", A4[:5] == A[:5], f"{A4[:5]} vs {A[:5]}")
c2.rpc(8032, "shutdown", timeout=60)
print(f"=== GATE 4(b) {'PASS' if not FAILED[0] else 'FAIL'} ===", flush=True)
