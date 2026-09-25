# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-C 15-minute soak: alternating chat streams + client-side cancels +
queued requests + idle gaps against the API. The engine keepalive runs
throughout (idle gaps > 10s). Detects: machine reboots (ssh not needed — we're
local), engine restarts (engine uptime resets), device faults (health 503 /
engine_down), response errors. Run: python3 soak.py [minutes]"""
import os, sys, json, time, http.client, threading, random

HOST, PORT = "127.0.0.1", 8080
LOGD = "~/logs"
MINS = float(sys.argv[1]) if len(sys.argv) > 1 else 15
random.seed(1234)

def log(*a):
  line = " ".join(str(x) for x in a)
  print(line, flush=True)
  try:
    os.makedirs(LOGD, exist_ok=True)
    with open(f"{LOGD}/soak.log", "a") as f: f.write(line + "\n")
  except Exception: pass

def req(method, path, body=None, timeout=300):
  conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
  conn.request(method, path, body=json.dumps(body) if body is not None else None,
               headers={"Content-Type": "application/json"})
  r = conn.getresponse(); data = r.read(); conn.close()
  try: j = json.loads(data)
  except Exception: j = {}
  return r.status, j

def engine_uptime():
  try:
    s, j = req("GET", "/health", timeout=10)
    if s == 200: return j["engine"].get("uptime_s")
    return f"http{s}"
  except Exception as e: return f"err:{str(e)[:40]}"

class Fault(Exception): pass

def stream_and_abort(kill_after, max_tokens=600):
  conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
  body = {"messages": [{"role": "user", "content": "Tell me a very long story about the sea."}],
          "max_tokens": max_tokens, "stream": True}
  conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
               headers={"Content-Type": "application/json"})
  r = conn.getresponse()
  if r.status != 200: conn.close(); raise Fault(f"stream http {r.status}")
  n = 0; t0 = time.time()
  while time.time() - t0 < kill_after:
    ch = r.read1(65536)
    if not ch: break
    n += len(ch)
  conn.close()   # abrupt abort
  return n

def main():
  t_end = time.time() + MINS * 60
  up0 = engine_uptime(); log(f"=== soak start {MINS}m; engine uptime {up0}")
  it = 0; faults = 0
  while time.time() < t_end:
    it += 1
    try:
      # 1) plain short non-stream
      s, j = req("POST", "/v1/chat/completions",
                 {"messages": [{"role": "user", "content": f"Say the number {it}."}], "max_tokens": 12})
      if s != 200: faults += 1; log(f"[{it}] non-stream http {s}: {str(j)[:120]}")
      # 2) stream + abrupt abort (cancel path)
      try:
        n = stream_and_abort(random.uniform(2.0, 5.0))
        log(f"[{it}] stream aborted after {n}B")
      except Fault as e:
        faults += 1; log(f"[{it}] STREAM FAULT: {e}")
      # 3) queued concurrency: 3 short requests at once
      res = []
      def one(i):
        try:
          s, j = req("POST", "/v1/chat/completions",
                     {"messages": [{"role": "user", "content": f"Name a color {i}."}], "max_tokens": 8})
          res.append(s)
        except Exception as e: res.append(str(e)[:40])
      ths = [threading.Thread(target=one, args=(i,)) for i in range(3)]
      [t.start() for t in ths]; [t.join() for t in ths]
      if not all(x == 200 for x in res):
        faults += 1; log(f"[{it}] queue results {res}")
      # 4) idle gap (keepalive exercised)
      time.sleep(random.uniform(8.0, 20.0))
      # 5) engine health + restart check
      up = engine_uptime()
      if not isinstance(up, (int, float)):
        faults += 1; log(f"[{it}] ENGINE UNHEALTHY: {up}")
      elif up < 30:
        faults += 1; log(f"[{it}] ENGINE RESTARTED during soak (uptime {up})")
    except Exception as e:
      faults += 1; log(f"[{it}] EXCEPTION {repr(e)[:200]}")
  up_end = engine_uptime()
  log(f"=== soak end: {it} rounds, faults={faults}, engine uptime {up_end}")
  return 1 if faults else 0

if __name__ == "__main__":
  sys.exit(main())
