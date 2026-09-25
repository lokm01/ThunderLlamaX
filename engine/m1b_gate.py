# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-B engine gates: the two M1-A open issues + FRESH lifecycle + idle RPC.

Gate A (issue a): the EXACT M1-A kill sequence + extension, run x2:
  park -> gen -> save -> load -> gen -> FOLLOW_UP -> gen -> FOLLOW_UP -> gen
  (M1-A repro died during the 2nd follow_up's trunk-prefill; the daemon now logs
  every stage to /tmp/m1a_serve.log — if it dies again, the last marker wins).
Gate ALPHA (HOST-PROCESS BOOT LAW): tok/cyc >= 2.2 over 20 cycles (alpha>=0.85
class ~2.78; collapsed draft = 1.0).
Gate FRESH: short-prompt FRESH prefill -> gen x2, deterministic + exact.
Gate IDLE (issue b): 300 s idle -> status RPC latency < 1 s (keepalive active).
"""
import os, sys, json, socket, time
import numpy as np

SOCK = "/tmp/llm-engine.sock"
SNAP = "~/snap100k"
DELTA = np.load(f"{SNAP}/gate3_delta.npy").tolist()

class Client:
  def __init__(self, timeout=300.0):
    self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.s.settimeout(timeout); self.s.connect(SOCK); self.buf = b""
  def send(self, obj): self.s.sendall((json.dumps(obj) + "\n").encode())
  def recv(self):
    while b"\n" not in self.buf:
      c = self.s.recv(65536)
      if not c: raise EOFError("engine closed")
      self.buf += c
    line, self.buf = self.buf.split(b"\n", 1)
    return json.loads(line)
  def rpc(self, rid, method, params=None):
    self.send({"id": rid, "method": method, "params": params or {}})
    while True:
      r = self.recv()
      if r.get("id") == rid and "event" not in r: return r
  def gen(self, rid, max_cycles, stop_token_ids=None, events=False):
    toks = []; n = 0
    t0 = time.perf_counter()
    self.send({"id": rid, "method": "generate",
               "params": {"max_cycles": max_cycles, "stop_token_ids": stop_token_ids or []}})
    while True:
      r = self.recv()
      if r.get("id") != rid: continue
      ev = r.get("event")
      if ev == "cycle":
        toks += r["tokens"]; n += 1
      elif ev in ("done", "cancelled"):
        if events: print(f"    [gen] terminal={ev} usage={r.get('usage')}")
        return toks, n, (time.perf_counter() - t0), (ev == "cancelled")

FAILED = [False]
def step(name, ok, extra=""):
  print(f"[m1b] {'PASS' if ok else 'FAIL'}: {name} {extra}", flush=True)
  if not ok: FAILED[0] = True

def connect_wait(timeout_s=1500):
  t0 = time.time()
  while time.time() - t0 < timeout_s:
    try:
      c = Client(10.0); r = c.rpc(9000, "status")
      if r.get("ok") and r["result"]["ready"]: return c
    except Exception: pass
    time.sleep(10)
  sys.exit("engine never came up")

print("== waiting for engine ==", flush=True)
c = connect_wait()
st = c.rpc(9001, "status")["result"]
print(f"[m1b] engine up: pos={st['pos']} ctxk={st['ctxk']} keepalive={st.get('keepalive_s')}s", flush=True)

# ---------- Gate ALPHA (BOOT LAW) ----------
c.rpc(9100, "prefill", {"snapshot": SNAP})
A, n, dt, _ = c.gen(9101, 20)
tpc = len(A) / n
step(f"alpha/boot-law tok/cyc {tpc:.2f} >= 2.2", tpc >= 2.2, f"({len(A)} toks / {n} cyc in {dt:.1f}s)")

# ---------- Gate A x2: the M1-A kill sequence + extension ----------
for it in (1, 2):
  print(f"== Gate A iteration {it} ==", flush=True)
  r = c.rpc(9200, "prefill", {"snapshot": SNAP})
  step(f"it{it} park", r.get("ok") and r["result"]["pos"] == 97810)
  A1, _, _, _ = c.gen(9201, 5)
  r = c.rpc(9202, "snapshot_save", {"path": f"/tmp/m1b_g{it}"})
  step(f"it{it} save", r.get("ok") and r["result"]["pos"] > 97810, str(r.get("result", ""))[:100])
  A2, _, _, _ = c.gen(9203, 5)
  step(f"it{it} post-save gen", len(A2) >= 5)
  r = c.rpc(9204, "snapshot_load", {"path": f"/tmp/m1b_g{it}"})
  step(f"it{it} load", r.get("ok") and r["result"]["pos"] == 97810 + len(A1))
  A3, _, _, _ = c.gen(9205, 5)
  step(f"it{it} load-resume deterministic (A3==A2)", A3[:5] == A2[:5], f"{A2[:6]} vs {A3[:6]}")
  r = c.rpc(9206, "prefill", {"mode": "FOLLOW_UP", "ids": DELTA})
  step(f"it{it} FOLLOW_UP #1 (post-load)", r.get("ok") and r["result"]["fed"] == len(DELTA) + 1,
       str(r.get("result", ""))[:100])
  B1, _, _, _ = c.gen(9207, 10)
  step(f"it{it} gen after FU#1", len(B1) >= 10)
  r = c.rpc(9208, "prefill", {"mode": "FOLLOW_UP", "ids": DELTA})
  step(f"it{it} FOLLOW_UP #2 (post-load) — the M1-A killer", r.get("ok"),
       str(r.get("result", ""))[:100])
  B2, _, _, _ = c.gen(9209, 10)
  step(f"it{it} gen after FU#2", len(B2) >= 10)
  # one more save/load on top for belt+braces
  r = c.rpc(9210, "snapshot_save", {"path": f"/tmp/m1b_g{it}b"})
  step(f"it{it} save #2", r.get("ok"))
  r = c.rpc(9211, "snapshot_load", {"path": f"/tmp/m1b_g{it}b"})
  step(f"it{it} load #2", r.get("ok"))
  B3, _, _, _ = c.gen(9212, 5)
  step(f"it{it} gen after 2nd load", len(B3) >= 5)

  # back-to-back FOLLOW-UPs with NO generate between (the strictest reading of
  # the original M1-A repro: "load → gen → FOLLOW-UP ×2")
  r = c.rpc(9213, "prefill", {"mode": "FOLLOW_UP", "ids": DELTA})
  step(f"it{it} back-to-back FU (no gen between) #1", r.get("ok"))
  r = c.rpc(9214, "prefill", {"mode": "FOLLOW_UP", "ids": DELTA})
  step(f"it{it} back-to-back FU #2 — strict repro form", r.get("ok"), str(r.get("result", ""))[:100])
  BB, _, _, _ = c.gen(9215, 5)
  step(f"it{it} gen after back-to-back FUs", len(BB) >= 5)

# ---------- Gate FRESH (short prompt, handler exactness) ----------
prompt = [int(t) for t in np.load(f"{SNAP}/ids.npy")[:64]]  # arbitrary 64-token prompt
outs = []
for rep in (1, 2):
  r = c.rpc(9300, "prefill", {"mode": "FRESH", "ids": prompt})
  step(f"fresh rep{rep}", r.get("ok") and r["result"]["fed"] == 64 and r["result"]["pos"] == 64,
       str(r.get("result", ""))[:100])
  F, _, _, _ = c.gen(9301, 5)
  outs.append(F[:8])
step("FRESH deterministic x2", outs[0] == outs[1], f"{outs[0][:8]}")

# ---------- stop-token gate (used by the API) ----------
c.rpc(9400, "prefill", {"snapshot": SNAP})
# at the park, cycle 1 emits [6545, ...] deterministically -> stop on 6545
S, n, _, _ = c.gen(9401, 100, stop_token_ids=[6545], events=True)
step("stop_token_ids stops generation", 6545 in S and n < 100, f"stopped after {n} cycles, {len(S)} toks, tail {S[-4:]}")
im_end = 248046  # <|im_end|> — VERIFIED via gate G (tokenizer eot_id; NOT Qwen2's 151645)
S2, n2, _, _ = c.gen(9402, 30, stop_token_ids=[im_end])
step("im_end stop id accepted (no fault)", n2 <= 30)

# ---------- Gate IDLE (issue b) ----------
if os.getenv("SKIP_IDLE", "0") != "1":
  print("== idling 300 s (keepalive should hold the link) ==", flush=True)
  time.sleep(300)
  t0 = time.perf_counter()
  r = c.rpc(9500, "status")
  d1 = time.perf_counter() - t0
  step("idle 5min -> status < 1s", r.get("ok") and d1 < 1.0, f"{d1*1000:.0f} ms")
  t0 = time.perf_counter()
  r = c.rpc(9501, "snapshot_load", {"path": "/tmp/m1b_g1"})
  d2 = time.perf_counter() - t0
  step("idle 5min -> snapshot_load ok", r.get("ok") and d2 < 60, f"{d2:.1f} s")

print(f"=== M1-B ENGINE GATES {'PASS' if not FAILED[0] else 'FAIL'} ===", flush=True)
sys.exit(1 if FAILED[0] else 0)
