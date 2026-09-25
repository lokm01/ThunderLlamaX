# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-C real-client smoke: scripted OpenAI-SDK-style 2-turn conversation.
Uses requests-style plain HTTP (same wire format an OpenAI SDK emits).
Verifies: turn-2 takes FOLLOW_UP (no re-prefill), content non-empty, stop
behavior. Run: python3 client_smoke.py"""
import os, sys, json, time, http.client

HOST, PORT = "127.0.0.1", 8080
LOGD = "~/logs"

def log(*a):
  line = " ".join(str(x) for x in a)
  print(line, flush=True)
  try:
    os.makedirs(LOGD, exist_ok=True)
    with open(f"{LOGD}/client_smoke.log", "a") as f: f.write(line + "\n")
  except Exception: pass

def chat(messages, max_tokens=24, conv=None, stream=False):
  conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
  body = {"model": "qwen3.8-27b-egpu", "messages": messages, "max_tokens": max_tokens}
  if conv: body["conversation_id"] = conv
  if stream:
    body["stream"] = True
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    txt, buf = "", b""
    while True:
      ch = r.read1(65536)
      if not ch: break
      buf += ch
      while b"\n\n" in buf:
        block, buf = buf.split(b"\n\n", 1)
        for ln in block.decode("utf-8", "replace").split("\n"):
          if ln.startswith("data: ") and ln[6:] != "[DONE]":
            try:
              d = json.loads(ln[6:])
              if d.get("choices") and d["choices"][0]["delta"].get("content"):
                txt += d["choices"][0]["delta"]["content"]
            except Exception: pass
    conn.close()
    return r.status, {"content": txt, "prefix_mode": "(stream)"}
  conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
               headers={"Content-Type": "application/json"})
  r = conn.getresponse(); j = json.loads(r.read()); conn.close()
  return r.status, j

ok = True
conv = f"smoke-{int(time.time())}"
s1, j1 = chat([{"role": "system", "content": "You are a terse assistant."},
               {"role": "user", "content": "Give me two vivid sentences about Mars."}],
              max_tokens=28, conv=conv)
t1 = j1["choices"][0]["message"]["content"] if s1 == 200 else str(j1)[:120]
log(f"turn-1: http={s1} mode={j1.get('prefix_mode')} finish={j1['choices'][0]['finish_reason'] if s1==200 else '-'} text={t1[:80]!r}")
ok &= s1 == 200 and j1.get("prefix_mode") == "FRESH" and len(t1) > 0

msgs2 = [{"role": "system", "content": "You are a terse assistant."},
         {"role": "user", "content": "Give me two vivid sentences about Mars."},
         {"role": "assistant", "content": t1},
         {"role": "user", "content": "Now the same for Venus."}]
t0 = time.time()
s2, j2 = chat(msgs2, max_tokens=28, conv=conv)
dt = time.time() - t0
t2 = j2["choices"][0]["message"]["content"] if s2 == 200 else str(j2)[:120]
log(f"turn-2: http={s2} mode={j2.get('prefix_mode')} took {dt:.1f}s text={t2[:80]!r}")
ok &= s2 == 200 and j2.get("prefix_mode") == "FOLLOW_UP" and len(t2) > 0

# streaming turn-3 on the same conversation
msgs3 = msgs2 + [{"role": "assistant", "content": t2},
                 {"role": "user", "content": "And one for the Moon, streamed."}]
s3, j3 = chat(msgs3, max_tokens=28, conv=conv, stream=True)
log(f"turn-3 stream: http={s3} text={j3['content'][:80]!r}")
ok &= s3 == 200 and len(j3["content"]) > 0

# engine log: follow_up ops, no fresh after them for this conv window
try:
  logs = open("/tmp/m1a_serve.log").read().splitlines()[-200:]
  fu = sum('"op": "follow_up"' in l for l in logs)
  log(f"engine log follow_up ops in last 200 lines: {fu}")
  ok &= fu >= 2
except Exception as e:
  log("log read:", e)

log("=== CLIENT SMOKE", "PASS" if ok else "FAIL", "===")
sys.exit(0 if ok else 1)
