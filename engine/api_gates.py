# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""M1-B API gates (a)-(g) against a running api_server on 127.0.0.1:8080.
Uses stdlib http.client only. Engine reference for gate (a) talks to the
engine socket directly. Run: python3 api_gates.py"""
import os, sys, json, time, http.client, socket, subprocess, threading

HOST, PORT = "127.0.0.1", 8080
ENGINE_SNAP = "~/snap100k"
FAILED = [False]

def step(name, ok, extra=""):
  print(f"[api-gate] {'PASS' if ok else 'FAIL'}: {name} {extra}", flush=True)
  if not ok: FAILED[0] = True

def req(method, path, body=None, timeout=300, headers=None, raw=False):
  conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
  h = {"Content-Type": "application/json"}
  if headers: h.update(headers)
  conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
  r = conn.getresponse()
  data = r.read()
  conn.close()
  if raw: return r, data
  try: j = json.loads(data)
  except Exception: j = {"_raw": data[:200].decode("utf-8", "replace")}
  return r, j

def sse_collect(body, kill_after=None, want_raw=False):
  """POST stream; collect parsed SSE events. kill_after: (secs) -> abort connection."""
  conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
  conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
               headers={"Content-Type": "application/json"})
  r = conn.getresponse()
  if r.status != 200:
    return r.status, None, r.read()[:300]
  events, buf, t0 = [], b"", time.time()
  aborted = False
  try:
    while True:
      if kill_after is not None and time.time() - t0 > kill_after:
        conn.close(); aborted = True; break
      ch = r.read1(65536)
      if not ch: break
      buf += ch
      while b"\n\n" in buf:
        block, buf = buf.split(b"\n\n", 1)
        lines = block.decode("utf-8", "replace").split("\n")
        for ln in lines:
          if ln.startswith("data: "):
            payload = ln[6:]
            if payload == "[DONE]": events.append(("[DONE]", None))
            else:
              try: events.append(("data", json.loads(payload)))
              except Exception: events.append(("badjson", payload))
          elif ln.startswith(":"):
            events.append(("comment", ln))
  except Exception as e:
    if not aborted: raise
  if not aborted:
    conn.close()
  return 200, events, aborted

def eng_rpc(method, params=None, timeout=600):
  s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(timeout)
  s.connect("/tmp/llm-engine.sock"); buf = b""
  s.sendall((json.dumps({"id": 7, "method": method, "params": params or {}}) + "\n").encode())
  while True:
    while b"\n" not in buf:
      ch = s.recv(65536)
      if not ch: raise EOFError
      buf += ch
    line, buf = buf.split(b"\n", 1)
    r = json.loads(line)
    if r.get("id") == 7 and "event" not in r:
      s.close(); return r

def eng_gen(max_cycles, stop_ids=None):
  """Direct engine generate on current resident state; returns token ids."""
  s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(600)
  s.connect("/tmp/llm-engine.sock"); buf = b""
  s.sendall((json.dumps({"id": 8, "method": "generate", "params":
              {"max_cycles": max_cycles, "stop_token_ids": stop_ids or []}}) + "\n").encode())
  toks = []
  while True:
    while b"\n" not in buf:
      ch = s.recv(65536)
      if not ch: raise EOFError
      buf += ch
    line, buf = buf.split(b"\n", 1)
    r = json.loads(line)
    if r.get("id") != 8: continue
    if r.get("event") == "cycle": toks += r["tokens"]
    elif r.get("event") in ("done", "cancelled"): break
  s.close(); return toks

def serve_log_tail(n=400):
  try:
    return open("/tmp/m1a_serve.log").read().splitlines()[-n:]
  except Exception: return []

# ---------- health + models ----------
r, j = req("GET", "/health", timeout=10)
step("health 200 ready", r.status == 200 and j.get("status") == "ok", str(j)[:160])
r, j = req("GET", "/v1/models", timeout=10)
step("v1/models", r.status == 200 and j["data"][0]["id"] == "qwen3.8-27b-egpu")

# ---------- (a) non-stream chat: well-formed + greedy-exact vs engine ----------
msg_a = [{"role": "user", "content": "Give me three words about the sky."}]
body_a = {"model": "qwen3.8-27b-egpu", "messages": msg_a, "max_tokens": 16, "seed": 42}
r, j = req("POST", "/v1/chat/completions", body_a)
ok_form = (r.status == 200 and j.get("object") == "chat.completion"
           and j["choices"][0]["message"]["role"] == "assistant"
           and isinstance(j["choices"][0]["message"]["content"], str)
           and j["model"] == body_a["model"]
           and j["usage"]["prompt_tokens"] > 0 and j["usage"]["completion_tokens"] > 0
           and j["choices"][0]["finish_reason"] in ("stop", "length")
           and "seed" in j.get("ignored_params", []))
step("(a) non-stream well-formed OpenAI JSON", ok_form, str(j)[:200])
_msga = j["choices"][0]["message"]
text_a = (_msga.get("reasoning_content") or "") + (_msga.get("content") or "")

# engine reference: FRESH the same rendered ids, generate, detok
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_server import TOK, TEMPLATE   # vendored tokenizer (cache-pickle fast)
from api_server import _template_render
ids_ref = TOK.encode(_template_render(msg_a, True, {"reasoning_effort": "medium"}))
# R6 batch protocol: prefill+generate share ONE conn (conn->slot binding;
# the legacy daemon tolerated a cross-conn generate against resident state)
_s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); _s.settimeout(600)
_s.connect("/tmp/llm-engine.sock")
_s.sendall((json.dumps({"id": 7, "method": "prefill", "params": {"mode": "FRESH", "ids": ids_ref, "conversation_id": "__gate_a_ref__"}}) + chr(10)).encode())
_buf = b""
while True:
    while b"\n" not in _buf: _buf += _s.recv(65536)
    _line, _buf = _buf.split(b"\n", 1)
    rr = json.loads(_line)
    if rr.get("id") == 7 and "event" not in rr: break
assert rr.get("ok"), rr
_s.sendall((json.dumps({"id": 8, "method": "generate", "params": {"max_cycles": 32, "stop_token_ids": [TOK.eot_id, TOK.eos_id]}}) + chr(10)).encode())
toks_ref = []
while True:
    while b"\n" not in _buf: _buf += _s.recv(65536)
    _line, _buf = _buf.split(b"\n", 1)
    r = json.loads(_line)
    if r.get("id") != 8: continue
    if r.get("event") == "cycle": toks_ref += r["tokens"]
    elif r.get("event") in ("done", "cancelled"): break
_s.close()
n_api = j["usage"]["completion_tokens"]
# the API stops at exactly max_tokens TOKENS; compare the same token window
text_ref = TOK.decode(toks_ref[:n_api])
step("(a) greedy-exact vs engine reference (same token window)",
     text_a == text_ref and len(text_a) > 0,
     f"ntok={n_api} api={text_a!r} vs ref={text_ref!r}")

# ---------- (b) stream: valid SSE, deltas == non-stream, [DONE], usage ----------
body_b = dict(body_a); body_b["stream"] = True; body_b["stream_options"] = {"include_usage": True}
st, events, _ = sse_collect(body_b)
datas = [e[1] for e in events if e[0] == "data"]
done_ok = any(e[0] == "[DONE]" for e in events)
usage_ok = any(d.get("usage") for d in datas if isinstance(d, dict))
content = "".join(d["choices"][0]["delta"].get("content", "")
                  for d in datas if isinstance(d, dict) and d.get("choices")
                  and d["choices"][0]["delta"].get("content"))
reasoning = "".join(d["choices"][0]["delta"].get("reasoning_content", "")
                  for d in datas if isinstance(d, dict) and d.get("choices")
                  and d["choices"][0]["delta"].get("reasoning_content"))
finish = [d["choices"][0].get("finish_reason") for d in datas if isinstance(d, dict)
          and d.get("choices") and d["choices"][0].get("finish_reason")]
step("(b) SSE sequence + [DONE] + usage chunk",
     st == 200 and done_ok and usage_ok and len(finish) == 1 and finish[0] in ("stop", "length"),
     f"events={len(events)} finish={finish}")
# identical prompt + deterministic FRESH -> stream text == engine ref window
# V-07: the model thinks first — the stream comparison covers the FULL
# client-visible continuation (reasoning deltas + content deltas), matching
# the non-stream message (reasoning_content + content).
step("(b) concatenated stream deltas == non-stream text", (reasoning + content) == text_a,
     f"len {len(reasoning+content)} vs {len(text_a)}; stream={(reasoning+content)!r}"[:160])

# ---------- (c) FOLLOW-UP economics via the API ----------
# M1-C: deterministic split. A natural im_end stop truncates the emit batch
# client-side -> engine fed <=2 tokens past what the client can render
# (STOP-BATCH OVER-COMMIT LAW) -> next turn MUST fall back to FRESH. A
# length-capped turn keeps client-visible == engine-fed exactly -> next turn
# MUST take FOLLOW_UP. Both directions asserted.
conv = "gate-c-conv"
t1 = {"model": "x", "messages": [{"role": "user", "content": "Count from one to five."}],
      "max_tokens": 96, "conversation_id": conv}
r1, j1 = req("POST", "/v1/chat/completions", t1)
step("(c) turn-1 accepted + natural stop (im_end) fires",
     r1.status == 200 and j1.get("prefix_mode") == "FRESH" and j1["choices"][0]["finish_reason"] == "stop",
     f"prefix_mode={j1.get('prefix_mode')} finish={j1['choices'][0]['finish_reason'] if r1.status==200 else '-'}")
t2 = dict(t1); t2["messages"] = t1["messages"] + [
  {"role": "assistant", "content": j1["choices"][0]["message"]["content"]},
  {"role": "user", "content": "Now count from six to ten."}]
r2, j2 = req("POST", "/v1/chat/completions", t2)
step("(c) after mid-batch im_end stop -> deterministic FRESH fallback (over-commit law)",
     r2.status == 200 and j2.get("prefix_mode") == "FRESH",
     f"prefix_mode={j2.get('prefix_mode')}")
# length-capped conversation -> exact mirror -> FOLLOW_UP guaranteed
conv2 = "gate-c2-conv"
u1 = {"model": "x", "messages": [{"role": "user", "content": "Write a vivid paragraph about a lighthouse in a storm."}],
      "max_tokens": 24, "conversation_id": conv2}
r3, j3 = req("POST", "/v1/chat/completions", u1)
step("(c2) length-capped turn-1 (finish=length)",
     r3.status == 200 and j3["choices"][0]["finish_reason"] == "length" and j3.get("prefix_mode") == "FRESH",
     f"finish={j3['choices'][0]['finish_reason'] if r3.status==200 else '-'}")
u2 = dict(u1); u2["messages"] = u1["messages"] + [
  {"role": "assistant", "content": j3["choices"][0]["message"]["content"]},
  {"role": "user", "content": "Now write one about a calm morning harbor."}]
t0 = time.time()
r4, j4 = req("POST", "/v1/chat/completions", u2)
dt2 = time.time() - t0
step("(c2) turn-2 prefix reuse -> FOLLOW_UP (exact mirror)",
     r4.status == 200 and j4.get("prefix_mode") == "FOLLOW_UP",
     f"prefix_mode={j4.get('prefix_mode')} turn-2 took {dt2:.1f}s")
logs = "\n".join(serve_log_tail(120))
step("(c2) engine log shows follow_up (delta prefill, not full re-prefill)",
     '"op": "follow_up"' in logs and '"op": "prefill_fresh"' not in logs.split("follow_up")[-1][:2000],
     "follow_up in engine log")

# ---------- (d) cancel on disconnect ----------
body_d = {"messages": [{"role": "user", "content": "Write a long story about a robot."}],
          "max_tokens": 4000, "stream": True}
t0 = time.time()
st_d, ev_d, aborted_d = sse_collect(body_d, kill_after=4.0)
step("(d) stream aborts client-side", st_d == 200 and aborted_d)
time.sleep(1.5)
t0 = time.time()
r_ok, j_ok = req("POST", "/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8}, timeout=60)
dt = time.time() - t0
step("(d) engine cancelled + next request served fast", r_ok.status == 200 and dt < 45,
     f"next request {dt:.1f}s")
step("(d) engine log shows cancelled", '"stage": "cancelled"' in "\n".join(serve_log_tail(60)))

# ---------- (e) queue: 2 concurrent serialized; 6th -> 429 ----------
results = []
def one(i):
  try:
    r, j = req("POST", "/v1/chat/completions",
               {"messages": [{"role": "user", "content": f"Say the number {i}."}],
                "max_tokens": 12}, timeout=300)
    results.append((i, r.status, time.time()))
  except Exception as e:
    results.append((i, -1, time.time()))
t0 = time.time()
ths = [threading.Thread(target=one, args=(i,)) for i in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
oks = [x for x in results if x[1] == 200]
step("(e) 2 concurrent both complete (serialized)", len(oks) == 2, f"{results} in {time.time()-t0:.1f}s")

results = []
ths = [threading.Thread(target=one, args=(i,)) for i in range(6)]
[t.start() for t in ths]; [t.join() for t in ths]
n200 = sum(1 for x in results if x[1] == 200)
n429 = sum(1 for x in results if x[1] == 429)
try:
    _hb = req("GET", "/health", timeout=10)[1]
    _bb = int((_hb.get("engine") or {}).get("batch_b") or 1)
except Exception:
    _bb = 1
if _bb >= 2:
    step("(e) 6 concurrent on batch permits: all served via queue, none dropped", n200 == 6 and n429 == 0,
         f"200s={n200} 429s={n429} batch_b={_bb}")
else:
    step("(e) 6 concurrent -> 5 served + >=1 429 with Retry-After", n200 == 5 and n429 >= 1,
         f"200s={n200} 429s={n429}")

# ---------- (f) 400-field matrix ----------
for name, body in [
    ("tools", {"messages": msg_a, "tools": [{"type": "function", "function": {"name": "f"}}]}),
    ("logprobs", {"messages": msg_a, "logprobs": True}),
    ("n=2", {"messages": msg_a, "n": 2}),
    ("temperature=0.7", {"messages": msg_a, "temperature": 0.7}),
    ("top_p=0.5", {"messages": msg_a, "top_p": 0.5}),
    ("multimodal", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}]}),
    ("response_format", {"messages": msg_a, "response_format": {"type": "json_object"}}),
]:
  r, j = req("POST", "/v1/chat/completions", body, timeout=15)
  ok = r.status == 400 and isinstance(j.get("error", {}).get("message"), str) and len(j["error"]["message"]) > 10
  step(f"(f) 400 matrix: {name}", ok, f"status={r.status}")
r, j = req("POST", "/v1/chat/completions", {"messages": msg_a, "temperature": 0, "top_p": 1.0, "max_tokens": 4}, timeout=120)
step("(f) temperature=0/top_p=1 accepted", r.status == 200)

print(f"=== M1-B API GATES {'PASS' if not FAILED[0] else 'FAIL'} ===", flush=True)
sys.exit(1 if FAILED[0] else 0)
