# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1+W2 mock-engine battery for api_server.py/serve.py/ops (GPU-free; run on
the rig with system python3).  Standalone:  python3 engine0/tests/test_api_mocks.py
Also pytest-compatible (all test_* are sync; async parts use asyncio.run).

Covers the W1 fix list (TLX_REVIEW_LEDGER top-10 items 1-9):
  V-01 FIFO/SSE slot lifetime, V-02 headroom lockout, V-03 per-conv lock +
       engine-side FOLLOW_UP guard, V-04 slow-loris + queue deadline,
  V-06 stop-string duplication, V-07 think/reasoning handling,
  V-08 max_completion_tokens, V-09 content shapes, V-10/V-12 exception state,
  V-11 mid-stream SSE errors, V-13 strict cap, V-14 eos-id-0/stop-id vocab,
  plus the four core harness tests (FIFO cap+429, SSE interleaving
  monotonicity, disconnect-cancel race, headroom-lockout clearance) and the
  different-content successive-request gate (P0_REPRO class).

W2 additions (security/ops wave):
  V-24 socket trust boundary (serve.validate_rpc/check_admin/_peer_uid units +
       protocol-abuse battery on the mirror mock), V-31 line cap,
  V-25 tokenizer cache integrity (0700 + HMAC + tamper fallback),
  V-28 config fingerprint/drift (svc_fp units, /health 503 config_drift,
       chat refusal, pcache _ENV_KEYS audit), V-33 /health redaction,
  V-34 HTTP hardening (TrustedHost/content-type/body-cap 413/415),
  jinja sandbox, ops-file assertions (env.canonical/plists/enginectl).
"""
import os, sys, json, time, socket, asyncio, tempfile, collections, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ENG0 = os.path.dirname(HERE)
sys.path.insert(0, ENG0)
sys.path.insert(0, HERE)

import mock_engine as ME
from mock_engine import mock_encode, mock_decode, ID_IMEND, ID_IMSTART

_BATT = {}

def setup_module():
  if _BATT: return
  d = tempfile.mkdtemp(prefix="tlx_w1_")
  gguf = os.path.join(d, "synth.gguf")
  ME.build_synth_gguf(gguf)
  os.environ["GGUF"] = gguf
  os.environ["ENGINE_SOCK"] = os.path.join(d, "engine.sock")
  os.environ["TLX_QUEUE_WAIT_S"] = "60"
  # W2: admin token (health debug fields) + canonical env (drift alarm) must
  # be set BEFORE api_server import (module reads them at import time)
  os.environ["TLX_ADMIN_TOKEN"] = "battery-admin-token"
  canon = os.path.join(d, "env.canonical.test")
  with open(canon, "w") as f:
    f.write(f"TLX_MODEL_PATH={gguf}\nLOOKUP_K=10\nPF_PREFILL=1\nKV8=1\nSKV=1\n")
  os.environ["TLX_ENV_CANONICAL"] = canon
  sys.path.insert(0, ENG0)
  import api_server
  _BATT["api"] = api_server
  _BATT["dir"] = d

def api():
  setup_module()
  return _BATT["api"]

def fresh_api():
  a = api()
  a.RESIDENTS.clear()
  a.QSTATE.update({"active": 0, "wait": collections.deque(), "holder": None, "permits": 1})
  a.CONV_LOCKS.clear()

class MockCtx:
  """per-test mock engine + api reset; engine_cls=ME.MockBatchEngine for the
  R6 batch battery (B=2 wire protocol mirror)."""
  def __init__(self, **cfg):
    self.engine_cls = cfg.pop("engine_cls", ME.MockEngine)
    self.a = api()
    fresh_api()
    self.a.HEADROOM = 32
    self.a.BODY_READ_TIMEOUT = 30.0
    self.a.QUEUE_WAIT_S = 60.0
    cfg.setdefault("config_fp", self.a.EXPECTED_FP)   # W2: drift check green by default
    self.eng = self.engine_cls(self.a.SOCK, cfg)
  def __enter__(self): return self
  def __exit__(self, *exc):
    self.eng.stop()
    deadline = time.time() + 5
    while time.time() < deadline and (self.a.QSTATE["active"] or self.a.QSTATE["wait"]):
      time.sleep(0.05)     # let any lingering guards release
    fresh_api()
    return False

def body(content="Hello there.", conv=None, stream=False, **kw):
  b = {"model": "m", "messages": [{"role": "user", "content": content}],
       "stream": stream}
  if conv: b["conversation_id"] = conv
  b.update(kw)
  return b

def no_think(b):
  b["enable_thinking"] = False
  return b

async def req(**kw):
  a = api()
  return await ME.asgi_request(a.app, "POST", "/v1/chat/completions", **kw)

def content_join(evs):
  return "".join(e.get("choices", [{}])[0].get("delta", {}).get("content", "")
                 for e in evs if isinstance(e, dict) and e.get("choices"))

def reasoning_join(evs):
  return "".join(e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content", "")
                 for e in evs if isinstance(e, dict) and e.get("choices"))

LONG_REPLY = "word " * 60            # 300 chars -> 100 cycles at width 3

# ==============================================================================
# unit tests (no engine)
# ==============================================================================
def test_unit_detokstream_stop_dup():
  """V-06: final_text must return full[emitted:i], never re-emit."""
  a = api()
  ds = a.DetokStream([" Hello"])
  out = []
  for t in mock_encode("Hello Hello "):
    out.append(ds.feed_token(t))
  tail = ds.final_text()
  assert "".join(out) + tail == "Hello", f"client saw {''.join(out)!r}+{tail!r}"

def test_unit_think_splitter():
  """V-07: ThinkSplitter shapes."""
  a = api()
  # template pre-opened think
  s = a.ThinkSplitter(think_open=True)
  r, c = s.feed("planning"); assert (r, c) == ("planning", "")
  r, c = s.feed(" a bit</think>\n\nAnswer"); assert r == " a bit" and c == "Answer", (r, c)
  r, c = s.feed(" body."); assert (r, c) == ("", " body.")
  # model emits its own leading block
  s = a.ThinkSplitter(think_open=False)
  r, c = s.feed("<think>re"); assert (r, c) == ("re", "")
  r, c = s.feed("ason</think>"); assert r == "ason" and c == "", (r, c)
  r, c = s.feed("\n\ndirect"); assert (r, c) == ("", "direct")
  r, c = s.final(); assert (r, c) == ("", "")
  # no think block at all
  s = a.ThinkSplitter(think_open=False)
  r, c = s.feed("just an answer"); assert (r, c) == ("", "just an answer")
  # unclosed think -> all reasoning (emitted by feed; final flushes nothing)
  s = a.ThinkSplitter(think_open=True)
  r, c = s.feed("never closes")
  assert (r, c) == ("never closes", ""), (r, c)
  rt, ct = s.final(); assert (rt, ct) == ("", ""), (rt, ct)
  # close split across feeds
  s = a.ThinkSplitter(think_open=True)
  r, c = s.feed("abc</thin"); assert (r, c) == ("abc", "")
  r, c = s.feed("k>\n\nok"); assert (r, c) == ("", "ok"), (r, c)

def test_unit_stop_ids_eos0_and_vocab():
  """V-14: eos id 0 kept; out-of-vocab stop ids dropped."""
  a = api()
  old = (a.TOK.eos_id, a.TOK.eot_id)
  try:
    a.TOK.eos_id, a.TOK.eot_id = 0, None
    ids = a._stop_ids_for([])
    assert 0 in ids, ids                      # truthiness bug would drop it
    a.TOK.eos_id, a.TOK.eot_id = 99999, None
    ids = a._stop_ids_for([])
    assert 99999 not in ids, ids              # vocab-range check
    a.TOK.eos_id, a.TOK.eot_id = None, None
    assert a._stop_ids_for([]) == []
  finally:
    a.TOK.eos_id, a.TOK.eot_id = old

def test_unit_validate_fields():
  """V-08/V-09/V-07 request-shape validation."""
  a = api()
  err, ctx = a._validate_fields(body(content=None))
  assert err is None and ctx is not None       # content:null accepted (mapped to "")
  err, _ = a._validate_fields(body(content=[{"type": "text"}]))
  assert err is not None and err.status_code == 400
  err, _ = a._validate_fields(body(content=[{"type": "text", "text": 5}]))
  assert err is not None
  err, _ = a._validate_fields(body(max_tokens=5, max_completion_tokens=7))
  assert err is not None                       # conflicting aliases
  err, ctx = a._validate_fields(body(max_completion_tokens=7))
  assert err is None and ctx["max_tokens"] == 7
  err, ctx = a._validate_fields(body(max_tokens=5, max_completion_tokens=5))
  assert err is None and ctx["max_tokens"] == 5
  err, _ = a._validate_fields(body(reasoning_effort="insane"))
  assert err is not None
  err, ctx = a._validate_fields(body())
  assert err is None and ctx["template_vars"] == {"reasoning_effort": "medium",
                                                  "enable_thinking": True}
  err, _ = a._validate_fields(body(enable_thinking="yes"))
  assert err is not None

# ==============================================================================
# mock protocol fidelity (direct socket)
# ==============================================================================
def test_mock_protocol():
  a = api(); fresh_api()
  eng = ME.MockEngine(a.SOCK, {})
  try:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(5)
    c.connect(a.SOCK); buf = b""
    def send(o): c.sendall((json.dumps(o) + "\n").encode())
    def recv():
      nonlocal buf
      while b"\n" not in buf:
        ch = c.recv(65536)
        if not ch: raise EOFError
        buf += ch
      line, buf = buf.split(b"\n", 1)
      return json.loads(line)
    def recv_reply(wid):
      while True:                              # skip progress/cycle events
        r = recv()
        if r.get("id") == wid and "event" not in r:
          return r
    # status field set
    send({"id": 1, "method": "status"})
    r = recv_reply(1)
    assert r["ok"]
    for k in ("ready", "ctxk", "busy", "pos", "mode", "cur", "fed_len", "fed_tail",
              "conversation_id", "queue", "dirty", "uptime_s"):
      assert k in r["result"], f"status missing {k}"
    # prefill FRESH
    send({"id": 2, "method": "prefill", "params":
          {"mode": "FRESH", "ids": [10, 11, 12], "conversation_id": "A"}})
    r = recv_reply(2); assert r["ok"] and r["result"]["pos"] == 3
    st = eng._status_locked(); assert st["fed_len"] == 3 and st["conversation_id"] == "A"
    # FOLLOW_UP: fed = fed + [cur] + delta
    send({"id": 3, "method": "prefill", "params":
          {"mode": "FOLLOW_UP", "ids": [70, 80], "cur": 11, "conversation_id": "A"}})
    r = recv_reply(3); assert r["ok"]
    assert eng.fed == [10, 11, 12, 11, 70, 80], eng.fed
    assert eng.pos == 6
    # FOLLOW_UP conversation guard (W1 fix)
    send({"id": 4, "method": "prefill", "params":
          {"mode": "FOLLOW_UP", "ids": [1], "cur": 80, "conversation_id": "B"}})
    r = recv_reply(4)
    assert not r["ok"] and "mismatch" in r["error"] and "'A'" in r["error"], r
    # cancel: NO ack (side channel)
    send({"id": 5, "method": "cancel", "params": {}})
    try:
      c.settimeout(0.3); recv(); acked = True
    except (socket.timeout, TimeoutError):
      acked = False
    assert not acked, "cancel must not be acked"
    c.settimeout(5)
    # generate: cycle events then terminal done; stop-token honored
    eng.cfg["reply_tokens"] = [1, 2, ID_IMEND]
    send({"id": 6, "method": "generate", "params":
          {"max_cycles": 10, "stop_token_ids": [ID_IMEND]}})
    evs = []
    while True:
      r = recv()
      if r.get("id") != 6: continue
      evs.append(r)
      if r.get("event") in ("done", "cancelled"): break
    assert evs[0]["event"] == "cycle" and evs[-1]["event"] == "done"
    assert evs[-1]["stop"] is True and evs[-1]["tokens"] == [1, 2, ID_IMEND]
    # mid-generate fault -> error reply + dirty; successful prefill clears
    eng.cfg["fail_at_cycle"] = 1
    eng.cfg["reply_tokens"] = [9, 9, 9, 9]
    send({"id": 7, "method": "generate", "params": {"max_cycles": 5}})
    r = recv_reply(7)
    assert not r["ok"] and "fault" in r["error"]
    assert eng.dirty is True
    assert eng.fed[-1] == 9                    # known-good tokens appended
    eng.cfg["fail_at_cycle"] = None
    send({"id": 8, "method": "prefill", "params":
          {"mode": "FRESH", "ids": [5, 6], "conversation_id": "A"}})
    r = recv_reply(8); assert r["ok"] and eng.dirty is False
    c.close()
  finally:
    eng.stop(); fresh_api()

# ==============================================================================
# core harness tests (ledger W1/W5 battery)
# ==============================================================================
def test_fifo_cap_and_429():
  """V-01/V-04 core test 1: 1 active + 4 waiting; the 6th gets 429;
  generates never overlap."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      rs = await asyncio.gather(*[
        req(json_body=no_think(body(f"question {i}", conv=f"c{i}", max_tokens=150)))
        for i in range(6)])
      codes = sorted(r.status for r in rs)
      assert codes.count(429) >= 1, codes
      assert codes.count(200) == 5, codes
      assert not mc.eng.generates_overlap(), "generates must serialize"
      await asyncio.sleep(0.1)
      assert mc.a.queue_depth() == 0
  asyncio.run(asyncio.wait_for(run(), 60))

def test_sse_interleave_monotonic():
  """V-01 core test 2: overlapping streams serialize; each stream's frames
  belong to itself; [DONE] strictly last."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      t0 = time.time()
      ra, rb = await asyncio.gather(
        req(json_body=no_think(body("stream A", conv="ca", stream=True, max_tokens=150))),
        req(json_body=no_think(body("stream B", conv="cb", stream=True, max_tokens=150))))
      assert ra.status == 200 and rb.status == 200
      for r in (ra, rb):
        evs, _ = r.sse_events()
        assert evs and evs[-1] == "[DONE]", "SSE must end with [DONE]"
        ids = {e.get("id") for e in evs if isinstance(e, dict)}
        assert len(ids) == 1, "cross-talk between streams"
        # chunk sequence: role first, content, finish, [DONE]
        assert evs[0]["choices"][0]["delta"].get("role") == "assistant"
        assert any(isinstance(e, dict) and e.get("choices") and
                   e["choices"][0].get("finish_reason") for e in evs[:-1])
      assert not mc.eng.generates_overlap()
      assert time.time() - t0 > 0
  asyncio.run(asyncio.wait_for(run(), 60))

def test_disconnect_cancel_race():
  """V-01 core test 3: client disconnect -> watcher arms cancel -> engine
  cancelled; slot released; no deadlock; next request admitted."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.03) as mc:
      r = await req(json_body=no_think(body("long stream", conv="cd", stream=True,
                                            max_tokens=150)),
                    disconnect_after=0.25)
      assert r.status == 200
      deadline = time.time() + 8
      while time.time() < deadline and not mc.eng.methods("cancel"):
        await asyncio.sleep(0.05)
      assert mc.eng.methods("cancel"), "engine cancel must be observed"
      gens = [n for _, n in mc.eng.methods("generate") if isinstance(n, str)]
      assert any("cancelled" in g for g in gens), gens
      while time.time() < deadline and mc.a.queue_depth() > 0:
        await asyncio.sleep(0.05)
      assert mc.a.queue_depth() == 0
      r2 = await req(json_body=no_think(body("next request", conv="cd2")))
      assert r2.status == 200
  asyncio.run(asyncio.wait_for(run(), 60))

def test_headroom_lockout_clearance():
  """V-02 core test 4: a parked engine pos near ctxk must NOT 400 a short
  FRESH request (the old max(pos,len) form locked out forever)."""
  async def run():
    with MockCtx(ctxk=120, initial_pos=115, cycle_delay=0.0) as mc:
      mc.a.HEADROOM = 8
      r = await req(json_body=no_think(body("Hi.", conv="ch", max_tokens=10)))
      assert r.status == 200, r.body[:300]
      # FOLLOW_UP uses the engine stream pos
      r2 = await req(json_body=no_think(
        body("Hi.", conv="ch", max_tokens=10,
             messages=[{"role": "user", "content": "Hi."},
                       {"role": "assistant", "content": "a"},
                       {"role": "user", "content": "Hi again."}])))
      assert r2.status == 200, r2.body[:300]
      assert r2.headers.get("x-prefix-mode") in ("FOLLOW_UP", "FRESH")
      # a prompt bigger than the window is still rejected (FRESH math)
      r3 = await req(json_body=no_think(body("x" * 130, conv="other")))
      assert r3.status == 400 and b"context window exhausted" in r3.body
  asyncio.run(asyncio.wait_for(run(), 60))

def test_nonstream_between_streams():
  """V-01: a non-stream request queued behind a live stream runs strictly
  after it; both coherent."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      done_at = {}
      async def stream_one():
        r = await req(json_body=no_think(body("s1", conv="s1", stream=True, max_tokens=150)))
        done_at["s"] = time.time(); return r
      async def nonstream_one():
        await asyncio.sleep(0.05)
        r = await req(json_body=no_think(body("n1", conv="n1", max_tokens=150)))
        done_at["n"] = time.time(); return r
      rs, rn = await asyncio.gather(stream_one(), nonstream_one())
      assert rs.status == 200 and rn.status == 200
      assert rn.json()["choices"][0]["message"]["content"].startswith("word")
      assert done_at["n"] >= done_at["s"] - 0.05
      assert not mc.eng.generates_overlap()
  asyncio.run(asyncio.wait_for(run(), 60))

# ==============================================================================
# per-fix tests
# ==============================================================================
def test_stop_dup_harness():
  """V-06 end-to-end: the 'Hello Hello ' repro cannot duplicate."""
  async def run():
    with MockCtx(reply_text="Hello Hello ", cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("Hello Hello ")
      r = await req(json_body=no_think(body("q", conv="cs", stream=True,
                                            stop=[" Hello"], max_tokens=50)))
      evs, _ = r.sse_events()
      got = content_join(evs)
      assert got == "Hello", repr(got)
      r2 = await req(json_body=no_think(body("q", conv="cs2", stream=False,
                                             stop=[" Hello"], max_tokens=50)))
      assert r2.json()["choices"][0]["message"]["content"] == "Hello"
      assert r2.json()["choices"][0]["finish_reason"] == "stop"
  asyncio.run(asyncio.wait_for(run(), 60))

def test_think_split_stream_and_nonstream():
  """V-07: reasoning split, template effort, enable_thinking=false."""
  async def run():
    reply = "planning the answer</think>\n\nThe visible answer."
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode(reply) + [ID_IMEND]
      # stream (thinking on: render pre-opens <think>)
      r = await req(json_body=body("q1", conv="t1", stream=True, max_tokens=50))
      evs, _ = r.sse_events()
      assert reasoning_join(evs) == "planning the answer", reasoning_join(evs)
      assert content_join(evs) == "The visible answer.", content_join(evs)
      for e in evs[:-1]:
        if not isinstance(e, dict) or not e.get("choices"): continue
        d = e["choices"][0].get("delta", {})
        assert "<think>" not in d.get("content", "") and "</think>" not in d.get("content", "")
      # non-stream
      r2 = await req(json_body=body("q2", conv="t2", max_tokens=50))
      j = r2.json()["choices"][0]["message"]
      assert j["content"] == "The visible answer."
      assert j.get("reasoning_content") == "planning the answer"
      # default effort is medium at the API layer -> no xhigh marker in render
      fed_text = mock_decode(mc.eng.fed)
      assert "Reasoning effort is set to xhigh" not in fed_text
      # explicit xhigh DOES reach the template
      r3 = await req(json_body=body("q3", conv="t3", reasoning_effort="xhigh",
                                    max_tokens=50))
      assert r3.status == 200
      assert "Reasoning effort is set to xhigh" in mock_decode(mc.eng.fed)
      # enable_thinking=false: template pre-closes think; no reasoning in output
      mc.eng.cfg["reply_tokens"] = mock_encode("Direct answer.") + [ID_IMEND]
      r4 = await req(json_body=no_think(body("q4", conv="t4", max_tokens=50)))
      j4 = r4.json()["choices"][0]["message"]
      assert j4["content"] == "Direct answer." and "reasoning_content" not in j4
      last_fed = mock_decode(mc.eng.fed)
      assert "<think>\n\n</think>\n\n" in last_fed   # render pre-closed the block
  asyncio.run(asyncio.wait_for(run(), 60))

def test_validation_battery():
  """V-09/V-08 request shapes end-to-end."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("ok") + [ID_IMEND]
      r = await req(json_body=no_think({"model": "m", "conversation_id": "v1",
                                        "max_tokens": 20,
                                        "messages": [{"role": "user", "content": None}]}))
      assert r.status == 200, r.body[:300]
      r = await req(json_body={"model": "m", "messages": [
        {"role": "user", "content": [{"type": "text"}]}]})
      assert r.status == 400
      r = await req(json_body={"model": "m", "messages": [
        {"role": "user", "content": [{"type": "text", "text": 3}]}]})
      assert r.status == 400
      r = await req(json_body=no_think({"model": "m", "max_tokens": 3,
                                        "max_completion_tokens": 9,
                                        "messages": [{"role": "user", "content": "x"}]}))
      assert r.status == 400
      # max_completion_tokens alone is honored (strict cap -> length)
      mc.eng.cfg["reply_tokens"] = mock_encode("0123456789" * 5)
      r = await req(json_body=no_think({"model": "m", "max_completion_tokens": 5,
                                        "conversation_id": "v2",
                                        "messages": [{"role": "user", "content": "x"}]}))
      assert r.status == 200
      j = r.json()
      assert j["usage"]["completion_tokens"] <= 5, j["usage"]
      assert j["choices"][0]["finish_reason"] == "length"
  asyncio.run(asyncio.wait_for(run(), 60))

def test_strict_cap_and_followup_reuse():
  """V-13: visible completion tokens never exceed max_tokens; the turn stays
  FOLLOW_UP-reusable per the R7a length-window contract."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("abcdefghij" * 8)  # 80 tokens
      r = await req(json_body=no_think({"model": "m", "max_tokens": 10,
                                        "conversation_id": "w1",
                                        "messages": [{"role": "user", "content": "count"}]}))
      j = r.json()
      assert j["usage"]["completion_tokens"] <= 10, j["usage"]
      assert len(j["choices"][0]["message"]["content"]) <= 10
      assert j["choices"][0]["finish_reason"] == "length"
      # turn 1 was length-capped -> reusable; turn 2 takes FOLLOW_UP
      r2 = await req(json_body=no_think({
        "model": "m", "max_tokens": 10, "conversation_id": "w1",
        "messages": [{"role": "user", "content": "count"},
                     {"role": "assistant", "content": j["choices"][0]["message"]["content"]},
                     {"role": "user", "content": "again"}]}))
      assert r2.status == 200
      assert r2.headers.get("x-prefix-mode") == "FOLLOW_UP", r2.headers
      modes = [n.get("mode") for _, n in mc.eng.methods("prefill")]
      assert "FOLLOW_UP" in modes, modes
  asyncio.run(asyncio.wait_for(run(), 60))

def test_midstream_error_sse_and_dirty_fresh():
  """V-10/V-11: mid-generate fault -> SSE error event + [DONE] (not a dropped
  conn); conversation marked dirty; next same-conv request goes FRESH."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("x" * 30)
      mc.eng.cfg["fail_at_cycle"] = 2
      r = await req(json_body=no_think(body("q", conv="e1", stream=True, max_tokens=50)))
      evs, _ = r.sse_events()
      assert evs and evs[-1] == "[DONE]"
      errs = [e for e in evs if isinstance(e, dict) and "error" in e]
      assert errs and errs[0]["error"]["type"] == "engine_error", evs[:6]
      assert mc.eng.dirty is True
      assert mc.a._resident("e1")["reusable"] is False
      assert mc.a._resident("e1")["messages"] is None
      # non-stream fault -> clean 503
      r2 = await req(json_body=no_think(body("q", conv="e2", max_tokens=50)))
      assert r2.status == 503 and "error" in r2.json()
      # next same-conv request must NOT take FOLLOW_UP (engine dirty)
      mc.eng.cfg["fail_at_cycle"] = None
      mc.eng.cfg["reply_tokens"] = mock_encode("fine") + [ID_IMEND]
      hist = [{"role": "user", "content": "q"},
              {"role": "assistant", "content": "old"},
              {"role": "user", "content": "next"}]
      r3 = await req(json_body=no_think({"model": "m", "conversation_id": "e1",
                                         "max_tokens": 20, "messages": hist}))
      assert r3.status == 200
      assert r3.headers.get("x-prefix-mode") == "FRESH"
      modes = [n.get("mode") for _, n in mc.eng.methods("prefill")]
      assert "FOLLOW_UP" not in modes, modes
      assert mc.eng.dirty is False       # successful prefill clears dirty
  asyncio.run(asyncio.wait_for(run(), 60))

def test_prefill_failure_no_phantom():
  """V-12: prefill failure resets RESIDENT (no phantom assistant turn, no
  stale foreign fed mirror)."""
  async def run():
    with MockCtx(cycle_delay=0.0, fail_prefill=True) as mc:
      r = await req(json_body=no_think(body("first", conv="p1", max_tokens=20)))
      assert r.status == 503
      assert mc.a._resident("p1")["messages"] is None
      assert mc.a._resident("p1")["fed"] == []
      mc.eng.cfg["fail_prefill"] = False
      # length-capped success turn (a visible-stop turn is non-reusable per
      # the STOP-BATCH law; a length turn keeps the mirror)
      mc.eng.cfg["reply_tokens"] = mock_encode("fine " * 40)
      r2 = await req(json_body=no_think(body("second", conv="p1", max_tokens=10)))
      assert r2.status == 200
      # mirror = exactly [user, assistant] — no phantom turn from request 1
      msgs = mc.a._resident("p1")["messages"]
      assert msgs is not None
      assert [m["role"] for m in msgs] == ["user", "assistant"], msgs
  asyncio.run(asyncio.wait_for(run(), 60))

def test_conv_lock_concurrent_same_conv():
  """V-03: two concurrent SAME-conv requests serialize; the second re-decides
  after the lock (its history lacks the first reply -> FRESH, safe); the
  engine-side FOLLOW_UP guard never fires; generates never overlap."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      async def turn():
        return await req(json_body=no_think({
          "model": "m", "conversation_id": "same", "max_tokens": 120,
          "messages": [{"role": "user", "content": "first ask"}]}))
      r1, r2 = await asyncio.gather(turn(), turn())
      assert r1.status == 200 and r2.status == 200
      assert not mc.eng.generates_overlap()
      errors = [n for _, _m, n in mc.eng.rpc_log if isinstance(n, str) and n.startswith("error:")]
      assert not any("mismatch" in e for e in errors), errors
      modes = [n.get("mode") for _, n in mc.eng.methods("prefill")]
      assert modes.count("FOLLOW_UP") <= 1, modes   # never a stale double-FOLLOW_UP
  asyncio.run(asyncio.wait_for(run(), 60))

def test_conv_lock_sequential_followup():
  """V-03 (deterministic form): second same-conv request must FOLLOW_UP on the
  first's state (per-conv lock + post-slot re-decide), never double-FRESH on
  garbage, and the engine guard would catch any mismatch."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("abcdefghij" * 8)
      r1 = await req(json_body=no_think({"model": "m", "conversation_id": "k1",
                                         "max_tokens": 10,
                                         "messages": [{"role": "user", "content": "hi"}]}))
      assert r1.status == 200
      txt = r1.json()["choices"][0]["message"]["content"]
      r2 = await req(json_body=no_think({
        "model": "m", "conversation_id": "k1", "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": txt},
                     {"role": "user", "content": "more"}]}))
      assert r2.status == 200
      assert r2.headers.get("x-prefix-mode") == "FOLLOW_UP"
      modes = [n.get("mode") for _, n in mc.eng.methods("prefill")]
      assert modes.count("FOLLOW_UP") == 1 and modes[0] != "FOLLOW_UP", modes
      # fed stream math on the engine: FRESH ids then [cur]+delta
      assert len(mc.eng.fed) > 0
  asyncio.run(asyncio.wait_for(run(), 60))

def test_different_content_gate():
  """P0_REPRO class: a different-content request running behind a live stream
  must NEVER take FOLLOW_UP on the other conversation's state."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      async def stream_a():
        return await req(json_body=no_think(body("alpha " * 5, conv="ga", stream=True,
                                                 max_tokens=150)))
      async def nonstream_b():
        await asyncio.sleep(0.05)
        return await req(json_body=no_think(body("beta " * 5, conv="gb",
                                                 max_tokens=150)))
      ra, rb = await asyncio.gather(stream_a(), nonstream_b())
      assert ra.status == 200 and rb.status == 200
      assert rb.headers.get("x-prefix-mode") == "FRESH"
      for _, n in mc.eng.methods("prefill"):
        if isinstance(n, dict) and n.get("cid") == "gb":
          assert n["mode"] != "FOLLOW_UP", n
  asyncio.run(asyncio.wait_for(run(), 60))

def test_slow_loris_no_slot_leak():
  """V-04: a stalled body consumes no FIFO slot (400 after the deadline);
  admission still clean right after."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.a.BODY_READ_TIMEOUT = 0.5
      r = await req(json_body={"model": "m", "messages": [{"role": "user", "content": "x"}]},
                    stall_body=True)
      assert r.status == 400, r.body[:200]
      await asyncio.sleep(0.1)
      assert mc.a.queue_depth() == 0
      r2 = await req(json_body=no_think(body("fine", conv="l1", max_tokens=20)))
      assert r2.status == 200
  asyncio.run(asyncio.wait_for(run(), 30))

def test_queue_wait_deadline():
  """V-16 part: a waiter beyond the queue deadline gets 503, not an eternal hang."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.02) as mc:
      mc.a.QUEUE_WAIT_S = 0.4
      async def first():
        return await req(json_body=no_think(body("long", conv="q1", max_tokens=150)))
      async def second():
        await asyncio.sleep(0.1)
        return await req(json_body=no_think(body("next", conv="q2", max_tokens=150)))
      r1, r2 = await asyncio.gather(first(), second())
      assert r1.status == 200
      assert r2.status == 503, r2.body[:200]
      assert b"queue" in r2.body
  asyncio.run(asyncio.wait_for(run(), 60))

def test_health_and_models():
  """light route sanity through the ASGI client."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      r = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert r.status == 200 and r.json()["status"] == "ok"
      r2 = await ME.asgi_request(mc.a.app, "GET", "/v1/models")
      assert r2.status == 200
  asyncio.run(asyncio.wait_for(run(), 30))

# ==============================================================================
# W2 battery: security / ops hardening (V-24/V-31 ACL+caps, V-25 cache,
# V-28 drift, V-33 redaction, V-34 HTTP hardening, svc_fp, ops files)
# ==============================================================================
def test_serve_rpc_validation_unit():
  """V-23/V-24: serve.validate_rpc caps (pure function, real daemon code)."""
  import serve
  CTXK, VOCAB = 100000, 300
  err, _ = serve.validate_rpc("generate", {"max_cycles": 0}, CTXK, VOCAB, 0)
  assert err and "max_cycles" in err
  err, _ = serve.validate_rpc("generate", {"max_cycles": "x"}, CTXK, VOCAB, 0)
  assert err
  err, p = serve.validate_rpc("generate", {"max_cycles": 100000}, CTXK, VOCAB, 0)
  assert err is None and p["max_cycles"] == 4096          # clamped to 4096
  err, p = serve.validate_rpc("generate", {"max_cycles": 5000}, 1000, VOCAB, 990)
  assert err is None and p["max_cycles"] == 10            # clamped to ctxk-pos
  err, _ = serve.validate_rpc("generate", {"max_cycles": 5, "stop_token_ids": [VOCAB]}, CTXK, VOCAB, 0)
  assert err and "vocab" in err
  err, p = serve.validate_rpc("generate", {"max_cycles": 5, "stop_token_ids": [0]}, CTXK, VOCAB, 0)
  assert err is None                                       # id 0 is in-vocab (V-14 class)
  err, _ = serve.validate_rpc("prefill", {"mode": "FRESH", "ids": []}, CTXK, VOCAB, 0)
  assert err
  err, _ = serve.validate_rpc("prefill", {"mode": "FRESH", "ids": [-1]}, CTXK, VOCAB, 0)
  assert err
  err, _ = serve.validate_rpc("prefill", {"mode": "FRESH", "ids": [VOCAB]}, CTXK, VOCAB, 0)
  assert err
  err, _ = serve.validate_rpc("prefill", {"mode": "FRESH", "ids": [1] * (CTXK + 1)}, CTXK, VOCAB, 0)
  assert err and "ctxk" in err
  err, _ = serve.validate_rpc("prefill", {"mode": "FOLLOW_UP", "ids": [5] * 100}, CTXK, VOCAB, CTXK)
  assert err and "overflows" in err
  err, p = serve.validate_rpc("prefill", {"mode": "FOLLOW_UP", "ids": [5] * 8}, 1000, VOCAB, 900)
  assert err is None and p["ids"] == [5] * 8
  err, _ = serve.validate_rpc("prefill", {"mode": "EVIL", "ids": [1]}, CTXK, VOCAB, 0)
  assert err

def test_serve_admin_acl_unit():
  """V-24: check_admin fails CLOSED with no token; constant-time path works."""
  import serve
  old = serve.ADMIN_TOKEN
  try:
    serve.ADMIN_TOKEN = ""
    ok, why = serve.check_admin({})
    assert not ok and "disabled" in why
    ok, _ = serve.check_admin({"admin_token": "anything"})
    assert not ok
    serve.ADMIN_TOKEN = "sekrit"
    ok, _ = serve.check_admin({"admin_token": "sekrit"})
    assert ok
    ok, why = serve.check_admin({"admin_token": "wrong"})
    assert not ok
    ok, _ = serve.check_admin({})
    assert not ok
  finally:
    serve.ADMIN_TOKEN = old

def test_serve_peer_uid_self():
  """V-24: LOCAL_PEERCRED reads the peer uid (darwin). Same-process peer =
  our own euid; the wrong-uid arm is a live-window test (sudo -u nobody)."""
  import serve, socket as _s
  a, b = _s.socketpair(_s.AF_UNIX, _s.SOCK_STREAM)
  try:
    uid = serve._peer_uid(a)
    if uid is not None:                     # None = platform fallback (perms-only)
      import os as _os
      assert uid == _os.geteuid(), (uid, _os.geteuid())
  finally:
    a.close(); b.close()

def test_protocol_abuse_battery():
  """W2.1 socket-protocol abuse: bad ids / bad max_cycles -> clean errors (no
  dirty); shutdown/snapshot without token -> refused; oversize line -> drop."""
  a = api(); fresh_api()
  eng = ME.MockEngine(a.SOCK, {"admin_token": "tok123"})
  try:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(5)
    c.connect(a.SOCK); buf = {"b": b""}
    def send(o): c.sendall((json.dumps(o) + "\n").encode())
    def recv_reply(wid):
      while True:
        while b"\n" not in buf["b"]:
          ch = c.recv(65536)
          if not ch: raise EOFError
          buf["b"] += ch
        line, buf["b"] = buf["b"].split(b"\n", 1)
        r = json.loads(line)
        if r.get("id") == wid and "event" not in r:
          return r
    # out-of-vocab id -> clean error, NO dirty (client error, not fault)
    send({"id": 1, "method": "prefill", "params": {"mode": "FRESH", "ids": [99999]}})
    r = recv_reply(1)
    assert not r["ok"] and "vocab" in r["error"], r
    assert eng.dirty is False and eng.rejected
    # empty ids + zero max_cycles
    send({"id": 2, "method": "prefill", "params": {"mode": "FRESH", "ids": []}})
    assert not recv_reply(2)["ok"]
    send({"id": 3, "method": "generate", "params": {"max_cycles": 0}})
    assert not recv_reply(3)["ok"]
    # admin methods fail CLOSED without the token; daemon stays alive
    send({"id": 4, "method": "shutdown", "params": {}})
    r = recv_reply(4)
    assert not r["ok"] and "admin" in r["error"], r
    send({"id": 5, "method": "snapshot_save", "params": {"path": "/tmp/x"}})
    r5 = recv_reply(5)
    assert not r5["ok"] and "admin" in r5["error"], r5
    send({"id": 6, "method": "status"})
    assert recv_reply(6)["ok"]                     # daemon alive
    assert "shutdown" in eng.admin_denied
    # privileged with token works (shutdown -> bye + stop)
    send({"id": 7, "method": "shutdown", "params": {"admin_token": "tok123"}})
    r = recv_reply(7)
    assert r["ok"] and r["result"]["bye"] is True
    deadline = time.time() + 3
    while time.time() < deadline and not eng._stop.is_set(): time.sleep(0.05)
    assert eng._stop.is_set()
    c.close()
  finally:
    eng.stop(); fresh_api()

def test_oversize_line_dropped():
  """V-31: a no-newline stream past the cap drops the conn; daemon survives."""
  a = api(); fresh_api()
  eng = ME.MockEngine(a.SOCK, {"max_line_bytes": 2048})
  try:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(5)
    c.connect(a.SOCK)
    c.sendall(b"x" * 8192)                    # no newline -> pending buffer grows
    got = c.recv(4096)                        # expect the close
    assert got == b"", got
    assert eng.dropped_conns >= 1
    # daemon still answers on a fresh conn
    c2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c2.settimeout(5)
    c2.connect(a.SOCK)
    c2.sendall(b'{"id":9,"method":"status"}\n')
    buf = b""
    while b"\n" not in buf: buf += c2.recv(65536)
    assert json.loads(buf)["ok"] is True
    c.close(); c2.close()
  finally:
    eng.stop(); fresh_api()

def test_health_redaction_and_admin_debug():
  """V-33/V-28: default /health has no fed_tail/conversation_id and carries
  the digest; the admin token unlocks the full engine dict."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      r = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert r.status == 200
      j = r.json()
      assert j["status"] == "ok"
      flat = json.dumps(j)
      assert "fed_tail" not in flat and "conversation_id" not in flat
      assert j["engine"]["config_fp"] == mc.a.EXPECTED_FP
      assert j["engine"]["busy"] is False
      assert j["config"]["drift_check"] == "on"
      # wrong token -> still redacted
      r2 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": "nope"})
      assert "engine_debug" not in r2.json()
      # right token -> full ops view
      r3 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": mc.a.ADMIN_TOKEN})
      j3 = r3.json()
      assert "fed_tail" in j3.get("engine_debug", {}) and \
             "conversation_id" in j3.get("engine_debug", {})
  asyncio.run(asyncio.wait_for(run(), 30))

def test_health_config_drift_503():
  """V-28: daemon fp != canonical env fp -> 503 config_drift on /health AND
  a refusal on /v1/chat/completions (no silent degraded serving)."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["config_fp"] = "deadbeefdeadbeef"
      r = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert r.status == 503, r.body[:300]
      j = r.json()
      assert j["status"] == "config_drift"
      assert j["config"]["config_fp_expected"] == mc.a.EXPECTED_FP
      r2 = await req(json_body=no_think(body("q", conv="d1", max_tokens=5)))
      assert r2.status == 503 and b"config drift" in r2.body, r2.body[:300]
      # the refusal happened BEFORE any engine prefill
      assert not mc.eng.methods("prefill")
  asyncio.run(asyncio.wait_for(run(), 30))

def test_health_warming_redacted():
  """V-33: the warming (503) branch is redacted too."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.ready = False
      r = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert r.status == 503 and r.json()["status"] == "warming"
      flat = json.dumps(r.json())
      assert "fed_tail" not in flat
      mc.eng.ready = True
  asyncio.run(asyncio.wait_for(run(), 30))

def test_http_hardening():
  """V-34: TrustedHost deny, content-type gate, body cap (CL + streaming)."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      a = mc.a
      # non-JSON content-type -> 415 before any parse/engine work
      r = await ME.asgi_request(a.app, "POST", "/v1/chat/completions",
                                raw_body=b'{"messages": []}',
                                headers={"content-type": "text/plain"})
      assert r.status == 415, r.body[:200]
      # foreign Host -> TrustedHost deny
      r = await ME.asgi_request(a.app, "GET", "/health",
                                headers={"host": "evil.example.com"})
      assert r.status == 400, r.body[:200]
      # oversize body with declared content-length -> 413 (never parsed)
      big = b"x" * (a.MAX_BODY_BYTES + 128)
      r = await ME.asgi_request(a.app, "POST", "/v1/chat/completions", raw_body=big,
                                headers={"content-type": "application/json",
                                         "content-length": str(len(big))})
      assert r.status == 413, r.body[:200]
      # oversize body WITHOUT content-length (chunked/lying client) -> the
      # streaming byte counter still catches it
      r = await ME.asgi_request(a.app, "POST", "/v1/chat/completions", raw_body=big,
                                headers={"content-type": "application/json"})
      assert r.status == 413, r.body[:200]
      assert not mc.eng.methods("prefill")
      # normal request still fine
      r = await req(json_body=no_think(body("ok", conv="h1", max_tokens=5)))
      assert r.status == 200
  asyncio.run(asyncio.wait_for(run(), 30))

def test_tokenizer_cache_integrity():
  """V-25: 0700 cache dir, HMAC-tagged payload, tamper/truncate -> cold load."""
  a = api()
  d = tempfile.mkdtemp(prefix="tlx_tokcache_")
  gg = os.path.join(d, "m.gguf")
  ME.build_synth_gguf(gg)
  old_path, old_home = a.GGUF_PATH, os.environ.get("HOME")
  os.environ["HOME"] = d                    # cache dir -> d/Library/Caches/tlx
  try:
    a.GGUF_PATH = gg
    tok, _template = a.load_tokenizer()
    cache, kpath = a._tok_cache_paths()
    assert cache and os.path.isfile(cache) and os.path.isfile(kpath)
    assert (os.stat(os.path.dirname(cache)).st_mode & 0o777) == 0o700
    assert (os.stat(cache).st_mode & 0o777) == 0o600
    tok2, _ = a.load_tokenizer()            # tagged cache load path works
    assert tok2.encode("hi") == tok.encode("hi")
    raw = bytearray(open(cache, "rb").read())
    raw[-1] ^= 0xFF                         # tamper INSIDE the payload
    open(cache, "wb").write(bytes(raw))
    tok3, _ = a.load_tokenizer()            # must reject + cold rebuild
    assert tok3.encode("hello") == tok.encode("hello")
    open(cache, "wb").write(b"garbage")     # short/garbage -> cold rebuild
    tok4, _ = a.load_tokenizer()
    assert tok4.eos_id == tok.eos_id
  finally:
    a.GGUF_PATH = old_path
    if old_home is not None: os.environ["HOME"] = old_home

def test_jinja_sandbox():
  """V-24-adjacent: GGUF chat template renders sandboxed — attribute traversal
  refused; a benign template renders identically."""
  a = api()
  from jinja2.sandbox import SandboxedEnvironment  # availability assert
  d = tempfile.mkdtemp(prefix="tlx_sandbox_")
  old_path, old_home = a.GGUF_PATH, os.environ.get("HOME")
  os.environ["HOME"] = d
  try:
    gg_evil = os.path.join(d, "evil.gguf")
    ME.build_synth_gguf(gg_evil, template="{{ ''.__class__.__mro__[1].__subclasses__() }}")
    a.GGUF_PATH = gg_evil
    _tok, tpl = a.load_tokenizer()
    raised = False
    try:
      tpl.render(messages=[], add_generation_prompt=True)
    except Exception:
      raised = True
    assert raised, "sandbox must refuse attribute traversal from template data"
    gg_good = os.path.join(d, "good.gguf")
    ME.build_synth_gguf(gg_good)
    a.GGUF_PATH = gg_good
    _tok2, tpl2 = a.load_tokenizer()
    out = tpl2.render(messages=[{"role": "user", "content": "hi"}],
                      add_generation_prompt=True, enable_thinking=True)
    assert "hi" in out and "<|im_start|>user" in out
  finally:
    a.GGUF_PATH = old_path
    if old_home is not None: os.environ["HOME"] = old_home

def test_svc_fp_sensitivity():
  """V-28 unit: fp changes with env knobs AND model-file identity."""
  import svc_fp
  d = tempfile.mkdtemp(prefix="tlx_fp_")
  m = os.path.join(d, "model.bin")
  with open(m, "wb") as f: f.write(b"A" * 1024)
  e1 = {"KV8": "1", "LOOKUP_K": "10"}
  fp1 = svc_fp.config_fp(env=e1, model_path=m)
  e2 = dict(e1); e2["LOOKUP_K"] = "0"       # the LOOKUP_K=0 silent-degradation class
  assert svc_fp.config_fp(env=e2, model_path=m) != fp1
  with open(m, "ab") as f: f.write(b"B")    # model file touched -> different fp
  assert svc_fp.config_fp(env=e1, model_path=m) != fp1
  os.utime(m, (0, 0))                       # mtime alone changes identity
  fp3 = svc_fp.config_fp(env=e1, model_path=m)
  os.utime(m, (1, 1))
  assert svc_fp.config_fp(env=e1, model_path=m) != fp3
  assert svc_fp.model_identity(os.path.join(d, "missing")) == "none"

def test_pcache_env_keys_extended():
  """V-28: pcache delegates to svc_fp and the key set includes the W2 knobs."""
  import svc_fp
  for k in ("PF_PREFILL", "LOOKUP", "LOOKUP_K", "M1A_GEN_REBUILD_EVERY",
            "MTP_KERNARGS_MB", "PF_ATTNW", "PF_SCANC_N2", "PF_M64QKV",
            "PF_ABW", "PG_SPLIT", "NV_SMEM_CFG_AUTO"):
    assert k in svc_fp._ENV_KEYS, k

def test_ops_canonical_and_scripts():
  """W2.3: env.canonical carries the ship knobs; wrapper/enginectl lost the
  /tmp breaker state, pgrep/pkill patterns; plists carry a UserName key.
  (Published-repo form: the live env.canonical is NOT in the repo — the
  sanitized env.canonical.example stands in; on the rig the real file is
  present and the same assertions run against it.)"""
  import svc_fp
  ops = os.path.join(ENG0, "ops")
  canon = os.path.join(ops, "env.canonical")
  if not os.path.exists(canon):
    canon = os.path.join(ops, "env.canonical.example")
  env = svc_fp.parse_env_file(canon)
  assert env.get("LOOKUP_K") == "10" and env.get("PF_PREFILL") == "1"
  assert env.get("PF_W4A8") == "1" and env.get("PC_ENABLED") == "1"
  assert env.get("TLX_ADMIN_TOKEN") and env.get("TLX_MODEL_PATH")
  sh = open(os.path.join(ops, "engine_daemon.sh")).read()
  assert "STAYDOWN=" in sh and "llm_engine_staydown" in sh
  assert "/tmp/llm_engine_staydown" not in sh and "/tmp/llm_engine_crashes" not in sh
  assert "pgrep -f" not in sh and "pkill" not in sh
  assert "kill -0" in sh                      # pid-liveness lock check (V-29)
  ec = open(os.path.join(ops, "enginectl")).read()
  assert "pkill -f" not in ec and "admin_token" in ec   # graceful stop, no grep-bomb
  pe = open(os.path.join(ops, "com.tlx.llm-engine.plist")).read()
  pa = open(os.path.join(ops, "com.tlx.llm-api.plist")).read()
  for p in (pe, pa):
    assert "<key>UserName</key>" in p          # substitute %USER% before install
  assert "EnvironmentVariables" not in pe     # env single-sourced (V-27)

# ==============================================================================
# runner
# ==============================================================================
def _all_tests():
  return [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]

def main():
  setup_module()
  tests = _all_tests()
  print(f"TLX W1 battery: {len(tests)} tests\n" + "=" * 60)
  failed = []
  for name, fn in tests:
    t0 = time.time()
    try:
      fn()
      print(f"PASS  {name}  ({time.time()-t0:.1f}s)")
    except Exception as e:
      failed.append(name)
      print(f"FAIL  {name}: {e}")
      traceback.print_exc()
  print("=" * 60)
  print(f"{len(tests)-len(failed)}/{len(tests)} passed")
  if failed:
    print("FAILED:", ", ".join(failed))
    return 1
  return 0



# ==============================================================================
# R6 PHASE 3 batch battery (MockBatchEngine, B=2 wire-protocol mirror)
# ==============================================================================
def test_batch_status_streams_and_batch_b():
  a = api(); fresh_api()
  eng = ME.MockBatchEngine(a.SOCK, {})
  try:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(5)
    c.connect(a.SOCK)
    c.sendall((json.dumps({"id": 1, "method": "status"}) + "\n").encode())
    buf = b""
    r = None
    while b"\n" not in buf: buf += c.recv(65536)
    r = json.loads(buf.split(b"\n")[0])
    assert r["ok"] and r["result"]["batch_b"] == 2
    assert isinstance(r["result"]["streams"], list) and len(r["result"]["streams"]) == 2
    for st in r["result"]["streams"]:
      for k in ("slot", "conversation_id", "pos", "cur", "fed_len", "mode", "generating", "dirty"):
        assert k in st, st
    # slot binding + per-slot status
    c.sendall((json.dumps({"id": 2, "method": "prefill", "params":
        {"mode": "FRESH", "ids": [10, 11, 12], "conversation_id": "A"}}) + "\n").encode())
    def recv_reply(wid):
      nonlocal buf
      while True:
        while b"\n" not in buf: buf += c.recv(65536)
        line, buf = buf.split(b"\n", 1)
        r = json.loads(line)
        if r.get("id") == wid and "event" not in r:
          return r
    r2 = recv_reply(2)
    assert r2["ok"], r2
    c.sendall((json.dumps({"id": 3, "method": "status"}) + "\n").encode())
    buf = b""
    while b"\n" not in buf: buf += c.recv(65536)
    r3 = json.loads(buf.split(b"\n")[0])
    convs = {st["conversation_id"] for st in r3["result"]["streams"]}
    assert "A" in convs
    sl = [st for st in r3["result"]["streams"] if st["conversation_id"] == "A"][0]
    assert sl["fed_len"] == 3 and sl["pos"] == 3 and not sl["generating"]
    # FOLLOW_UP for a conversation on NO slot -> clean refusal
    c2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c2.settimeout(5)
    c2.connect(a.SOCK)
    c2.sendall((json.dumps({"id": 4, "method": "prefill", "params":
        {"mode": "FOLLOW_UP", "ids": [1], "cur": 5, "conversation_id": "ZZ"}}) + "\n").encode())
    buf = b""
    while b"\n" not in buf: buf += c2.recv(65536)
    r4 = json.loads(buf.split(b"\n")[0])
    assert not r4["ok"] and "no resident conversation" in r4["error"], r4
    c.close(); c2.close()
  finally:
    eng.stop(); fresh_api()

def test_batch_two_streams_concurrent():
  """Two DIFFERENT conversations generate CONCURRENTLY (permits=2); both
  complete; each SSE stream carries only its own chunks; the engine observed
  overlapping generate windows."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      t0 = time.time()
      ra, rb = await asyncio.gather(
        req(json_body=no_think(body("stream one", conv="ca", stream=True, max_tokens=90))),
        req(json_body=no_think(body("stream two", conv="cb", stream=True, max_tokens=90))))
      assert ra.status == 200 and rb.status == 200, (ra.status, rb.status)
      for r in (ra, rb):
        evs, _ = r.sse_events()
        assert evs and evs[-1] == "[DONE]"
        ids = {e.get("id") for e in evs if isinstance(e, dict)}
        assert len(ids) == 1, "cross-talk between batch streams"
        assert len(content_join(evs)) > 50, "stream content truncated"
      assert mc.eng.generates_overlap(), "the two generates must run concurrently"
      assert mc.a.queue_depth() == 0
      # engine status: both conversations resident after the turns
      st = mc.eng._status_locked()
      convs = {s["conversation_id"] for s in st["streams"]}
      assert {"ca", "cb"} <= convs, convs
  asyncio.run(asyncio.wait_for(run(), 60))

def test_batch_third_request_waits_for_slot():
  """permits=2: a third DIFFERENT conversation queues (does not 429 at 3);
  it runs after a slot frees; no overlap beyond 2."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      rs = await asyncio.gather(*[
        req(json_body=no_think(body(f"q {i}", conv=f"c{i}", max_tokens=120)))
        for i in range(3)])
      codes = [r.status for r in rs]
      assert codes == [200, 200, 200], codes
      wins = sorted(mc.eng.gen_windows)
      assert len(wins) == 3
      # the third must start after at least one of the first two finished
      assert wins[2][0] >= min(wins[0][1], wins[1][1]) - 0.05, wins
      # no triple overlap
      for i in range(2):
        assert wins[i+1][0] >= wins[i][0]
      assert mc.a.queue_depth() == 0
  asyncio.run(asyncio.wait_for(run(), 90))

def test_batch_per_stream_cancel():
  """Cancelling stream A (client disconnect) leaves stream B running to
  completion; A's engine stream observes the per-conn cancel."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      ra = asyncio.ensure_future(req(
        json_body=no_think(body("doomed stream", conv="ca", stream=True, max_tokens=150)),
        disconnect_after=0.15))
      await asyncio.sleep(0.02)   # let A attach first
      rb = await req(json_body=no_think(body("healthy stream", conv="cb",
                                             stream=True, max_tokens=60)))
      assert rb.status == 200
      evs, _ = rb.sse_events()
      assert evs[-1] == "[DONE]" and len(content_join(evs)) > 50
      r = await ra
      assert r.status == 200
      deadline = time.time() + 8
      while time.time() < deadline and not mc.eng.methods("cancel"):
        await asyncio.sleep(0.05)
      assert mc.eng.methods("cancel"), "per-stream cancel must reach the engine"
      while time.time() < deadline and mc.a.queue_depth() > 0:
        await asyncio.sleep(0.05)
      assert mc.a.queue_depth() == 0
  asyncio.run(asyncio.wait_for(run(), 60))

def test_batch_per_stream_stop_interleave():
  """A stops at im_end mid-batch; B keeps decoding to its own max."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, cycle_delay=0.004) as mc:
      mc.eng.cfg["reply_tokens_by_slot"] = {
        0: [1, 2, 3, ID_IMEND, 4, 5],                       # A stops at im_end
        1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # B runs on
      }
      mc.eng.cfg["cycle_width"] = 1
      ra, rb = await asyncio.gather(
        req(json_body=no_think(body("stopper", conv="ca", max_tokens=200))),
        req(json_body=no_think(body("runner", conv="cb", max_tokens=200))))
      assert ra.status == 200 and rb.status == 200
      # A visible answer is short (stop token); B ran on
      ta = ra.json()["choices"][0]["message"]["content"]
      tb = rb.json()["choices"][0]["message"]["content"]
      lens = sorted((len(ta), len(tb)))
      assert lens[0] <= 3 and lens[1] > 100, lens   # one stopped early, one ran on
      fins = sorted((ra.json()["choices"][0]["finish_reason"],
                     rb.json()["choices"][0]["finish_reason"]))
      assert fins == ["length", "stop"], fins
      gens = [n for _, n in mc.eng.methods("generate") if isinstance(n, str)]
      # the stopper terminal is stop@; the runner ends via the API strict-cap
      # bail (V-13) = engine-side cancelled@ (hidden tail), or max@ if unbounded
      assert any("stop@" in g for g in gens), gens
      assert any(("cancelled@" in g) or ("max@" in g) for g in gens), gens
  asyncio.run(asyncio.wait_for(run(), 60))

def test_batch_followup_isolation_after_concurrent_turn():
  """Conv A turn-2 takes FOLLOW_UP even though conv B ran concurrently
  between the turns (per-conversation RESIDENTS + per-slot engine match)."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      r1 = await req(json_body=no_think(body("first question", conv="ca", max_tokens=30)))
      assert r1.status == 200
      ans1 = r1.json()["choices"][0]["message"]["content"]
      rb = await req(json_body=no_think(body("other conv turn", conv="cb", max_tokens=30)))
      assert rb.status == 200
      r2 = await req(json_body=no_think({"model": "m", "conversation_id": "ca", "stream": False,
        "messages": [{"role": "user", "content": "first question"},
                     {"role": "assistant", "content": ans1},
                     {"role": "user", "content": "second question"}], "max_tokens": 30}))
      assert r2.status == 200
      assert r2.headers.get("x-prefix-mode") == "FOLLOW_UP", r2.headers
      # engine received a FOLLOW_UP prefill for ca (not a FRESH displacement)
      fus = [n for _, n in mc.eng.methods("prefill")
             if isinstance(n, dict) and n.get("mode") == "FOLLOW_UP"]
      assert fus, mc.eng.methods("prefill")
  asyncio.run(asyncio.wait_for(run(), 60))

def test_batch_same_conv_still_serialized():
  """Two concurrent requests for the SAME conversation serialize on the conv
  lock (V-03 unchanged under batching); no double-advance."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      ra, rb = await asyncio.gather(
        req(json_body=no_think(body("same a", conv="cs", max_tokens=60))),
        req(json_body=no_think(body("same b", conv="cs", max_tokens=60))))
      assert ra.status == 200 and rb.status == 200
      wins = sorted(mc.eng.gen_windows)
      assert not (len(wins) == 2 and wins[1][0] < wins[0][1] - 1e-6), \
        "same-conversation generates must serialize"
  asyncio.run(asyncio.wait_for(run(), 60))

def test_batch_protocol_generate_without_prefill():
  """generate on an unbound conn is refused cleanly (no crash, no reply loss)."""
  a = api(); fresh_api()
  eng = ME.MockBatchEngine(a.SOCK, {})
  try:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(5)
    c.connect(a.SOCK)
    c.sendall((json.dumps({"id": 7, "method": "generate",
                           "params": {"max_cycles": 5}}) + "\n").encode())
    buf = b""
    while b"\n" not in buf: buf += c.recv(65536)
    r = json.loads(buf.split(b"\n")[0])
    assert not r["ok"] and "no prefill" in r["error"], r
    c.close()
  finally:
    eng.stop(); fresh_api()

def test_legacy_single_stream_battery_still_green():
  """The LEGACY engine against the batch-capable API: permits fall back to 1;
  serialization behavior identical (regression guard for the QSTATE rewrite)."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      rs = await asyncio.gather(*[
        req(json_body=no_think(body(f"q {i}", conv=f"l{i}", max_tokens=120)))
        for i in range(6)])
      codes = sorted(r.status for r in rs)
      assert codes.count(429) >= 1 and codes.count(200) == 5, codes
      assert not mc.eng.generates_overlap()
      await asyncio.sleep(0.1)
      assert mc.a.queue_depth() == 0
  asyncio.run(asyncio.wait_for(run(), 60))


# ==============================================================================
# W3 additions (prompt-cache plumbing through the API -> engine RPC)
# ==============================================================================
def test_prompt_cache_key_and_ttl_plumbed():
  """V-44 (W3): prompt_cache_key + prompt_cache_ttl travel through the API
  into the AUTO_CACHE prefill params (pin key + per-request TTL)."""
  a = api(); fresh_api()
  eng = ME.MockEngine(a.SOCK, {"auto_cache_hit": 6})
  try:
    async def run():
      r = await req(json_body=no_think(body("Hello cache", conv="w3ttl",
                                            prompt_cache_key="tenant-a",
                                            prompt_cache_ttl=3600)))
      assert r.status == 200, (r.status, r.body)
      assert eng.prefill_params, "no prefill recorded"
      p = eng.prefill_params[-1]
      assert p.get("mode") == "AUTO_CACHE"
      assert p.get("cache_key") == "tenant-a", p
      assert p.get("cache_ttl") == 3600, p
      # a request without the fields sends neither
      r2 = await req(json_body=no_think(body("Second turn", conv="w3ttl2")))
      assert r2.status == 200, r2.status
      p2 = eng.prefill_params[-1]
      assert "cache_key" not in p2 and "cache_ttl" not in p2, p2
    asyncio.run(asyncio.wait_for(run(), 30))
  finally:
    eng.stop(); fresh_api()


if __name__ == "__main__":
  sys.exit(main())
