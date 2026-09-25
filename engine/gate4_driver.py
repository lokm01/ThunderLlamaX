# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-A GATE 4 driver: scripted JSON-RPC session over /tmp/llm-engine.sock.
Sequence: status -> prefill(snapshot, FRESH) -> generate 60 (exact vs engine T=1 ref)
-> FOLLOW-UP prefill(200 delta) -> generate 40 (exact vs gate3 resident output)
-> cancel mid-generate -> clean FRESH re-request -> snapshot_save/load round-trip
-> shutdown -> relaunch engine -> snapshot persists (reload + regenerate exact).
Run AFTER gate23.py (uses gate3_delta.npy + gate3_out_resident.npy)."""
import os, sys, json, socket, time
import numpy as np

SOCK = "/tmp/llm-engine.sock"
SNAP = "~/snap100k"
G4SNAP = "/tmp/g4snap"
ref = np.load(f"{SNAP}/engine_t1_ref_kv8_qh.npy").tolist()
delta = np.load(f"{SNAP}/gate3_delta.npy").tolist()
t2exp = np.load(f"{SNAP}/gate3_out_resident.npy").tolist()

class Client:
  def __init__(self):
    self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.s.connect(SOCK); self.buf = b""
  def send(self, obj):
    self.s.sendall((json.dumps(obj) + "\n").encode())
  def recv(self, timeout=120):
    self.s.settimeout(timeout)
    while b"\n" not in self.buf:
      c = self.s.recv(65536)
      if not c: raise EOFError
      self.buf += c
    line, self.buf = self.buf.split(b"\n", 1)
    return json.loads(line)
  def rpc(self, rid, method, params=None, timeout=120):
    self.send({"id": rid, "method": method, "params": params or {}})
    while True:
      r = self.recv(timeout)
      if r.get("id") == rid and "event" not in r:
        return r

def gen_collect(c, rid, max_cycles, stop=None, cancel_after=None):
  """issue generate; collect cycle events; optionally cancel after N cycles."""
  toks = []; ncyc = 0; cancelled = False
  c.send({"id": rid, "method": "generate", "params": {"max_cycles": max_cycles, "stop_token_ids": stop or []}})
  while True:
    r = c.recv(timeout=300)
    if r.get("id") != rid: continue
    ev = r.get("event")
    if ev == "cycle":
      toks += r["tokens"]; ncyc += 1
      if cancel_after is not None and ncyc >= cancel_after:
        c.send({"id": 99000 + rid, "method": "cancel", "params": {}})
    elif ev in ("done", "cancelled"):
      return toks, ncyc, (ev == "cancelled")
    elif "ok" in r:
      return r, ncyc, False

def step(name, ok, extra=""):
  print(f"[gate4] {'PASS' if ok else 'FAIL'}: {name} {extra}", flush=True)
  if not ok: globals()["FAILED"] = True
globals()["FAILED"] = False

for attempt in range(240):   # wait for daemon boot (~10 min)
  try:
    c = Client(); c.rpc(1, "status", timeout=5); break
  except Exception:
    time.sleep(10)
else:
  sys.exit("engine never came up")

r = c.rpc(2, "status")
step("status ready", r.get("ok") and r["result"]["ready"], str(r.get("result", {}))[:120])

r = c.rpc(3, "prefill", {"snapshot": SNAP}, timeout=300)
step("prefill snapshot FRESH at pos", r.get("ok") and r["result"]["pos"] == len(np.load(f"{SNAP}/ids.npy")), str(r.get("result")))

t0g = time.time()
toks, ncyc, _ = gen_collect(c, 4, 60)
dtg = time.time() - t0g
step("generate 60 exact", toks[:60] == ref, f"{sum(1 for k in range(60) if toks[k] == ref[k])}/60, "
     f"{len(toks)} toks/{ncyc} cyc = {len(toks)/dtg:.1f} tok/s (expect ~40)")

r = c.rpc(5, "prefill", {"mode": "FOLLOW_UP", "ids": delta}, timeout=600)
step("FOLLOW-UP prefill", r.get("ok") and r["result"]["fed"] == 201, str(r.get("result")))
toks2, ncyc2, _ = gen_collect(c, 6, 40)
step("turn-2 exact vs gate3 resident", toks2[:40] == t2exp, f"{sum(1 for k in range(40) if toks2[k] == t2exp[k])}/40")

toks3, ncyc3, canc = gen_collect(c, 7, 600, cancel_after=5)
step("cancel mid-generate", canc and ncyc3 <= 8, f"cancelled={canc} cycles={ncyc3}")

r = c.rpc(8, "prefill", {"snapshot": SNAP}, timeout=300)
toks4, _, _ = gen_collect(c, 9, 10)
step("cancel-then-FRESH-request clean", r.get("ok") and len(toks4) >= 10 and toks4[:10] == ref[:10])

r = c.rpc(10, "snapshot_save", {"path": G4SNAP}, timeout=600)
step("snapshot_save", r.get("ok"), str(r.get("result", ""))[:120])
toks5, _, _ = gen_collect(c, 11, 5)
r = c.rpc(12, "snapshot_load", {"path": G4SNAP}, timeout=600)
step("snapshot_load (same process)", r.get("ok") and r["result"]["pos"] > int(np.load(f"{SNAP}/ids.npy").shape[0]) + 60, str(r.get("result", ""))[:120])
toks5b, _, _ = gen_collect(c, 13, 5)
step("post-reload regenerate exact", toks5b[:5] == toks5[:5], f"{toks5[:5]} vs {toks5b[:5]}")

r = c.rpc(14, "shutdown", timeout=60)
time.sleep(5)
alive = os.popen("pgrep -f 'python.*serve.py' | wc -l").read().strip()
step("shutdown exits", alive == "0", f"pgrep={alive}")
print("[gate4] relaunching engine for persistence check...", flush=True)
os.system("cd ~/tinygrad-metal/engine0 && nohup env PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH "
          "DOCKER_HOST=unix://~/.colima/default/docker.sock DEV=NV SKV=1 SKV_K=g4nw32 SKV_S=256 "
          "SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 M1A_SERVE=1 ~/tg311/bin/python -u test_w100k.py "
          "> /tmp/serve2.log 2>&1 &")
for attempt in range(240):
  try:
    c2 = Client(); rr = c2.rpc(21, "status", timeout=5)
    if rr.get("ok") and rr["result"]["ready"]: break
  except Exception: pass
  time.sleep(10)
else:
  sys.exit("relaunch never became ready")
r = c2.rpc(22, "snapshot_load", {"path": G4SNAP}, timeout=600)
step("relaunch + snapshot persists (load)", r.get("ok"), str(r.get("result", ""))[:120])
toks6, _, _ = gen_collect(c2, 23, 5)
step("relaunched regenerate exact", toks6[:5] == toks5[:5], f"{toks6[:5]} vs {toks5[:5]}")
c2.rpc(24, "shutdown", timeout=60)
print(f"=== GATE 4 {'PASS' if not FAILED else 'FAIL'} ===", flush=True)
