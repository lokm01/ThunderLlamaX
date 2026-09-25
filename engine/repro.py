# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-C fault bisect — MINIMAL engine-direct reproducers (no API server needed).
Talks raw JSON-lines to /tmp/llm-engine.sock exactly like api_server.py does.
Run: python3 repro.py <mode> [iters]
Modes:
  fresh-loop    control: prefill FRESH + generate 30, repeated (NO cancel anywhere)
  cancel-long   generate max_cycles=1e5, cancel after T secs, wait cancelled; repeat (nothing else)
  cancel-next   cancel-long + IMMEDIATELY (new conn) prefill FRESH + generate 30 (the <=70ms window)
  cancel-close  cancel + close socket WITHOUT reading the cancelled event (the exact API bail pattern), then idle gap, then status
  gen-stop      generate with stop_token_ids (done_stop path, no cancel), then FRESH prefill + generate (the gate-(c) pattern)
All output is teed to ~/logs/repro_<mode>.log (reboot-survivor)."""
import os, sys, json, time, socket

SOCK = "/tmp/llm-engine.sock"
LOGD = "~/logs"
IDS = [9707, 11, 1917, 29916, 525, 2311, 311, 2324, 13, 29889]  # arbitrary valid ids

def log(*a):
  line = " ".join(str(x) for x in a)
  print(line, flush=True)
  try:
    os.makedirs(LOGD, exist_ok=True)
    with open(f"{LOGD}/repro_{MODE}.log", "a") as f: f.write(line + "\n")
  except Exception: pass

class C:
  def __init__(self, timeout=600.0):
    self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); self.s.settimeout(timeout)
    self.s.connect(SOCK); self.buf = b""
  def send(self, obj): self.s.sendall((json.dumps(obj) + "\n").encode())
  def recv(self):
    while b"\n" not in self.buf:
      ch = self.s.recv(65536)
      if not ch: raise EOFError("engine closed")
      self.buf += ch
    line, self.buf = self.buf.split(b"\n", 1)
    return json.loads(line)
  def close(self):
    try: self.s.close()
    except Exception: pass

def rpc(c, mid, method, params=None):
  c.send({"id": mid, "method": method, "params": params or {}})
  while True:
    r = c.recv()
    if r.get("id") == mid and "event" not in r:
      if not r.get("ok"): raise RuntimeError(f"{method}: {r.get('error')}")
      return r["result"]

def prefill_fresh(mid=1):
  c = C()
  r = rpc(c, mid, "prefill", {"mode": "FRESH", "ids": IDS, "conversation_id": f"repro_{MODE}"})
  c.close(); return r

def generate(ncycles=30, mid=2, stop_ids=None):
  """Returns (tokens, end_event)."""
  c = C(); toks = []
  c.send({"id": mid, "method": "generate", "params": {"max_cycles": ncycles, "stop_token_ids": stop_ids or []}})
  ev = None
  while True:
    r = c.recv()
    if r.get("id") != mid: continue
    if r.get("event") == "cycle": toks += r["tokens"]
    elif r.get("event") in ("done", "cancelled"): ev = r; break
  c.close(); return toks, ev

def cancel_after(secs, mid=3, wait_event=True, close_after_cancel=False):
  """Start a long generate; cancel after secs. wait_event=False mirrors the API
  bail() pattern: send cancel, never read the cancelled event, close socket."""
  c = C(); toks = []
  c.send({"id": mid, "method": "generate", "params": {"max_cycles": 100000, "stop_token_ids": []}})
  t0 = time.time(); cancelled = False; ev = None
  c.s.settimeout(0.05)
  while time.time() - t0 < secs + 30:
    try:
      r = c.recv()
      if r.get("id") == mid and r.get("event") == "cycle": toks += r["tokens"]
      elif r.get("id") == mid and r.get("event") in ("done", "cancelled"): ev = r; break
    except socket.timeout:
      pass
    if not cancelled and time.time() - t0 >= secs:
      c.send({"id": 99, "method": "cancel", "params": {}})
      cancelled = True
      if close_after_cancel:
        c.close()
        return toks, {"closed_without_wait": True}
      c.s.settimeout(30)
      # fall through: keep reading until cancelled event
  if ev is None and not close_after_cancel:
    try:
      while True:
        r = c.recv()
        if r.get("id") == mid and r.get("event") in ("done", "cancelled"): ev = r; break
    except Exception: pass
  if not close_after_cancel: c.close()
  return toks, ev

def healthy():
  try:
    c = C(5.0); r = rpc(c, 0, "status"); c.close(); return r
  except Exception as e:
    return {"err": str(e)}

def run(mode, iters):
  for i in range(1, iters + 1):
    t0 = time.time()
    if mode == "fresh-loop":
      p = prefill_fresh(); tks, ev = generate(30)
      log(f"[{mode}] it{i} prefill pos={p['pos']} gen {len(tks)}toks ev={'done' if ev and ev.get('event')=='done' else ev} {time.time()-t0:.1f}s")
    elif mode == "cancel-long":
      tks, ev = cancel_after(2.0)
      log(f"[{mode}] it{i} cancelled at {len(tks)}toks ev={ev.get('event') if ev else ev} {time.time()-t0:.1f}s")
    elif mode == "cancel-next":
      tks, ev = cancel_after(1.5)
      log(f"[{mode}] it{i} cancel at {len(tks)}toks -> next request immediately")
      p = prefill_fresh(); tks2, ev2 = generate(30)
      log(f"[{mode}] it{i} next-ok prefill pos={p['pos']} gen {len(tks2)}toks {time.time()-t0:.1f}s")
    elif mode == "cancel-close":
      tks, ev = cancel_after(1.5, wait_event=False, close_after_cancel=True)
      time.sleep(3.0)
      st = healthy()
      log(f"[{mode}] it{i} closed-after-cancel; status: {str(st)[:120]} {time.time()-t0:.1f}s")
    elif mode == "gen-stop":
      tks, ev = generate(30, stop_ids=[248046])
      log(f"[{mode}] it{i} gen1 {len(tks)}toks stop={ev.get('stop')}")
      p = prefill_fresh(); tks2, ev2 = generate(20)
      log(f"[{mode}] it{i} next prefill pos={p['pos']} gen {len(tks2)}toks {time.time()-t0:.1f}s")
    else:
      raise SystemExit(f"unknown mode {mode}")
  log(f"[{mode}] ALL {iters} ITERATIONS CLEAN")

if __name__ == "__main__":
  MODE = sys.argv[1] if len(sys.argv) > 1 else "cancel-next"
  iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
  log(f"=== repro {MODE} x{iters} start {time.strftime('%H:%M:%S')} ===")
  run(MODE, iters)
