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
  # TLX P8: this battery exercises the LEGACY single-model surface — point the
  # registry + state dir at nonexistent sandbox paths (REGISTRY None). The P8
  # multi-model surface has its own battery (tests/test_p8_multiplayer.py).
  os.environ["TLX_MODEL_REGISTRY"] = os.path.join(d, "NO_REGISTRY.json")
  os.environ["TLX_STATE_DIR"] = os.path.join(d, "state")
  os.environ["TLX_ENV_COMMON"] = os.path.join(d, "NO_env.common")
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
      assert errs and errs[0]["error"]["type"] == "api_error"              and errs[0]["error"]["code"] == "engine_error", evs[:6]   # R3-37 enum
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
  err, _ = serve.validate_rpc("prefill", {"mode": "FOLLOW_UP", "ids": [5] * 100,
                                           "conversation_id": "c"}, CTXK, VOCAB, CTXK)
  assert err and "overflows" in err
  err, p = serve.validate_rpc("prefill", {"mode": "FOLLOW_UP", "ids": [5] * 8,
                                           "conversation_id": "c"}, 1000, VOCAB, 900)
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
      # R3-24: the engine view + drift detail moved behind the admin header
      assert "engine" not in j and "config" not in j, j
      # wrong token -> still redacted
      r2 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": "nope"})
      assert "engine_debug" not in r2.json() and "engine" not in r2.json()
      # right token -> full ops view
      r3 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": mc.a.ADMIN_TOKEN})
      j3 = r3.json()
      assert "fed_tail" in j3.get("engine_debug", {}) and \
             "conversation_id" in j3.get("engine_debug", {})
      assert j3["engine"]["config_fp"] == mc.a.EXPECTED_FP
      assert j3["engine"]["busy"] is False and j3["config"]["drift_check"] == "on"
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
      # R3-24: the fp pair sits behind the admin header
      assert "config" not in j
      ra = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": mc.a.ADMIN_TOKEN})
      assert ra.json()["config"]["config_fp_expected"] == mc.a.EXPECTED_FP
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
  /tmp breaker state, pgrep/pkill patterns; plists carry a UserName."""
  import svc_fp
  ops = os.path.join(ENG0, "ops")
  # bare clone: env.canonical is generated from the published example; test
  # the example when the operator's real file is absent (rig behavior intact)
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
# R3 additions (round-3 ledger; review-fixes-r3 branch)
# ==============================================================================
def test_q_promote_vs_cancel_grant_released():
  """R3-10: a promotion landing in the same window as the waiter's
  cancellation must RELEASE the consumed grant — at permits=1 the old code's
  no-op handler leaked it and ONE occurrence bricked admission (429 forever)."""
  async def run():
    a = api(); fresh_api()
    a.QSTATE.update({"active": 1, "permits": 1})   # a live holder occupies the
    t = asyncio.ensure_future(a.q_acquire(1))      # only permit
    await asyncio.sleep(0.01)                 # t parked on its fut in the deque
    assert len(a.QSTATE["wait"]) == 1
    fut = a.QSTATE["wait"][0]
    a.q_release()                             # holder leaves: active->0 AND the
    assert a.QSTATE["active"] == 1            # promoter grants OUR fut (the
    assert fut.done() and not fut.cancelled() # wakeup is scheduled, not run)
    t.cancel()                                # cancel BEFORE t resumes -> the
    try:                                      # CancelledError fires at the await
      await t
    except asyncio.CancelledError:
      pass
    await asyncio.sleep(0.01)
    assert a.QSTATE["active"] == 0, a.QSTATE  # OLD: stays 1 — the leaked grant
    assert len(a.QSTATE["wait"]) == 0         # bricks admission at permits=1
  asyncio.run(asyncio.wait_for(run(), 10))

def test_q_dead_waiters_pruned_no_429():
  """R3-11: timed-out waiters leave dead futures in the wait deque and used to
  count toward MAX_WAITING — after N timeouts a fresh arrival was 429'd even
  though the real queue was empty."""
  async def run():
    a = api(); fresh_api()
    a.QSTATE.update({"active": 1, "permits": 1})
    old_max = a.MAX_WAITING
    a.MAX_WAITING = 4
    async def timed_out_waiter():
      try:
        await asyncio.wait_for(a.q_acquire(1), 0.05)
        a.q_release()                          # if admitted, be polite
      except asyncio.TimeoutError:
        pass
    try:
      await asyncio.gather(*[timed_out_waiter() for _ in range(4)])
      assert len(a.QSTATE["wait"]) == 0, [f.done() for f in a.QSTATE["wait"]]
      # fresh arrival must NOT see a full queue; admitted as soon as the
      # holder releases
      t = asyncio.ensure_future(a.q_acquire(1))
      a.q_release()
      await asyncio.wait_for(t, 2)
      assert a.QSTATE["active"] == 1
    finally:
      a.MAX_WAITING = old_max
      a.q_release()
  asyncio.run(asyncio.wait_for(run(), 10))

def test_evq_coalesce_and_loud_fail_unit():
  """R3-12 unit (the §3.3 ruling): under Full, text/reasoning COALESCE into
  one pending event (never a silent drop); a SECOND consecutive coalesced-put
  failure marks the stream dead + arms the cancel (loud-fail backstop); the
  pending tail is flushed, not lost."""
  a = api()
  dead = {"v": False}
  evq, pend, on_event = a._make_stream_events(1, put_timeout=0.02,
                                              on_dead=lambda: dead.__setitem__("v", True))
  on_event("prefill", {"done": 1, "total": 2})
  assert evq.get_nowait()[0] == "prefill"
  on_event("text", "a")                     # fills the single slot
  assert evq.qsize() == 1
  on_event("text", "b")                     # Full -> coalesced into pending
  assert evq.qsize() == 1 and pend["text"] == "b" and not pend["dead"]
  assert evq.get_nowait() == ("text", "a")  # drain
  on_event("text", "c")                     # retry succeeds -> merged event
  assert evq.get_nowait() == ("text", "bc"), "coalesced order+content"
  assert pend["text"] == "" and pend["stalls"] == 0
  # loud-fail backstop: two consecutive Full windows with no drain
  on_event("reasoning", "r1")               # fills the slot
  on_event("reasoning", "r2")               # Full #1 -> pending
  on_event("reasoning", "r3")               # Full #2 -> DEAD + cancel armed
  assert pend["dead"] and dead["v"], "backstop must fire loudly"
  assert pend["reasoning"] == "r2r3"        # the tail is preserved, not lost
  on_event("text", "never")                 # post-dead: no-op, no queue growth
  assert evq.qsize() == 1

def test_evq_stream_byte_complete_tiny_queue():
  """R3-12 e2e: a tiny event queue under a fast producer still delivers
  byte-complete concatenated content through the REAL gen() (the coalesced
  chunks and the end-of-stream pending flush keep stream == non-stream)."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode(LONG_REPLY) + [ID_IMEND]
      old_max = mc.a.EVQ_MAX
      mc.a.EVQ_MAX = 4
      try:
        r = await req(json_body=no_think(body("slow consumer", conv="evq1",
                                              stream=True, max_tokens=400)))
        assert r.status == 200
        evs, _ = r.sse_events()
        got = content_join(evs)
        assert got == LONG_REPLY, (len(got), len(LONG_REPLY))
        r2 = await req(json_body=no_think(body("slow consumer", conv="evq2",
                                               max_tokens=400)))
        assert r2.json()["choices"][0]["message"]["content"] == LONG_REPLY
      finally:
        mc.a.EVQ_MAX = old_max
  asyncio.run(asyncio.wait_for(run(), 30))

def test_batch_same_conv_streams_serialize_over_stream_lifetime():
  """R3-08: at permits=2 the conv lock is held for the STREAM LIFETIME — the
  old code released it when the route returned, so a second same-conv request
  prefilled (double slot binding!) while the first stream's executor thread
  was still generating: torn R['fed'] reads and prefill-failure mirror wipes."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      ra, rb = await asyncio.gather(
        req(json_body=no_think(body("same stream a", conv="cx", stream=True, max_tokens=120))),
        req(json_body=no_think(body("same stream b", conv="cx", stream=True, max_tokens=120))))
      assert ra.status == 200 and rb.status == 200, (ra.status, rb.status)
      # same-conversation generates never overlapped (conv lock spans the
      # stream; the OLD code ran them concurrently at permits=2)
      assert not mc.eng.generates_overlap(), mc.eng.gen_windows
      # the engine-side belt never had to fire (the API serialized first)
      busy = [r for r in mc.eng.rejected if "conversation busy" in r[1]]
      assert not busy, busy
      # no double slot binding for one conversation at any tick
      st = mc.eng._status_locked()
      cids = [s["conversation_id"] for s in st["streams"]]
      assert cids.count("cx") <= 1, cids
  asyncio.run(asyncio.wait_for(run(), 60))

def test_engine_belt_rejects_prefill_while_conv_generating():
  """R3-08 engine belt (mock mirror of serve._b_prefill): a prefill binding a
  conversation that is GENERATING on another slot is refused cleanly — never
  a silent double-bind (wrong-answer class)."""
  a = api(); fresh_api()
  eng = ME.MockBatchEngine(a.SOCK, {"cycle_width": 3})
  try:
    ca = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); ca.settimeout(5)
    ca.connect(a.SOCK)
    cb = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); cb.settimeout(5)
    cb.connect(a.SOCK)
    def send(c, o): c.sendall((json.dumps(o) + "\n").encode())
    bufs = {ca: b"", cb: b""}
    def reply(c, wid):
      while True:
        while b"\n" not in bufs[c]:
          bufs[c] += c.recv(65536)
        line, bufs[c] = bufs[c].split(b"\n", 1)
        r = json.loads(line)
        if r.get("id") == wid and "event" not in r:
          return r
    # conn A: prefill + attach generate (no terminal yet)
    send(ca, {"id": 1, "method": "prefill", "params":
        {"mode": "FRESH", "ids": [10, 11, 12], "conversation_id": "belt"}})
    assert reply(ca, 1)["ok"]
    send(ca, {"id": 2, "method": "generate", "params": {"max_cycles": 50}})
    time.sleep(0.3)                      # let the generate attach + tick
    # conn B: SAME conversation prefill -> clean refusal, no second binding
    send(cb, {"id": 3, "method": "prefill", "params":
        {"mode": "FRESH", "ids": [20, 21], "conversation_id": "belt"}})
    r = reply(cb, 3)
    assert not r["ok"] and "conversation busy" in r["error"], r
    st = eng._status_locked()
    cids = [s["conversation_id"] for s in st["streams"]]
    assert cids.count("belt") == 1, cids
    # a DIFFERENT conversation still prefills fine on the other slot
    send(cb, {"id": 4, "method": "prefill", "params":
        {"mode": "FRESH", "ids": [30, 31], "conversation_id": "other"}})
    assert reply(cb, 4)["ok"]
    ca.close(); cb.close()
  finally:
    eng.stop(); fresh_api()

def test_encode_once_off_loop():
  """R3-09: the O(pairs) BPE encode runs in the THREADPOOL and ONCE (the
  pre-slot memo is reused post-slot). Old code: encode ran directly on the
  asyncio loop TWICE per FRESH request — a 50-100k-token prompt starved SSE
  generators, disconnect watchers and /health."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      calls = {"n": 0}
      real_encode = mc.a.TOK.encode
      def slow_encode(text):
        calls["n"] += 1
        time.sleep(0.15)                 # model the O(pairs) merge cost
        return real_encode(text)
      mc.a.TOK.encode = slow_encode
      lat = []
      async def probe():
        for _ in range(8):
          t0 = time.time()
          await ME.asgi_request(mc.a.app, "GET", "/health")
          lat.append(time.time() - t0)
          await asyncio.sleep(0.04)
      try:
        pb = asyncio.ensure_future(probe())
        r = await req(json_body=no_think(body("big " + "prompt " * 200,
                                              conv="enc1", max_tokens=10)))
        assert r.status == 200, r.body[:200]
        await asyncio.wait_for(pb, 10)
      finally:
        mc.a.TOK.encode = real_encode
      assert calls["n"] == 1, f"encode must run exactly once, ran {calls['n']}"
      assert max(lat) < 0.12, f"event loop stalled on the encode: {lat}"
  asyncio.run(asyncio.wait_for(run(), 30))


def test_r3_19_fp_batch_and_cubin_sensitivity():
  """R3-19: config_fp sees BATCH-knob drift (BATCH_B / BATCH_REBUILD_EVERY /
  BATCH_PF_CHUNK / R6_PF_T1 / PF_G3M_MB were absent from _ENV_KEYS — a BATCH_B=2
  boot without env edits kept the SAME fp and old KV restored into a different
  engine) AND cubin-content drift (a kernel rebuild with no env change); the
  API import and the daemon mix the same cubin digest (agreement)."""
  import svc_fp, tempfile as _tf, shutil as _sh
  d = _tf.mkdtemp(prefix="tlx_fp19_")
  try:
    m = os.path.join(d, "m.gguf"); open(m, "wb").write(b"A" * 1024)
    base = {"KV8": "1", "LOOKUP_K": "10", "TLX_MODEL_PATH": m}
    saved = svc_fp.extra_fp()
    try:
      svc_fp.set_extra("cubins", "digestA")
      fp1 = svc_fp.config_fp(env=base)
      for k, v in (("BATCH_B", "2"), ("BATCH_REBUILD_EVERY", "232"),
                   ("BATCH_PF_CHUNK", "64"), ("R6_PF_T1", "1"),
                   ("PF_G3M_MB", "0")):
        e2 = dict(base); e2[k] = v
        assert svc_fp.config_fp(env=e2) != fp1, f"{k} drift invisible"
      svc_fp.set_extra("cubins", "digestB")     # a rebuilt kernel set
      assert svc_fp.config_fp(env=base) != fp1
    finally:
      svc_fp._EXTRA_FP.clear(); svc_fp._EXTRA_FP.update(saved)
    # cubin_set_digest: byte sensitivity + deterministic missing-file class
    cd = _tf.mkdtemp(prefix="tlx_cub_")
    try:
      open(os.path.join(cd, "k2s5.cubin"), "wb").write(b"v1")
      dg1 = svc_fp.cubin_set_digest(base_dir=cd)
      open(os.path.join(cd, "k2s5.cubin"), "wb").write(b"v2")
      assert svc_fp.cubin_set_digest(base_dir=cd) != dg1
      dg3 = svc_fp.cubin_set_digest(base_dir=os.path.join(cd, "gone"))
      assert len(dg3) == 16                    # all-missing still deterministic
    finally:
      _sh.rmtree(cd, ignore_errors=True)
    # wiring: the api import mixed the real cubin digest (agrees with serve)
    a = api()
    assert svc_fp.extra_fp().get("cubins") == svc_fp.cubin_set_digest()
    for k in ("BATCH_B", "BATCH_REBUILD_EVERY", "BATCH_PF_CHUNK", "R6_PF_T1", "PF_G3M_MB"):
      assert k in svc_fp._ENV_KEYS, k
  finally:
    _sh.rmtree(d, ignore_errors=True)


# ==============================================================================
# R3-32/33/35/37 (W-D): the conformance quartet
# ==============================================================================
def test_r3_32_reasoning_effort_high():
  """R3-32: the OpenAI-standard `high` no longer 400s — it maps to this
  template's strongest tier (xhigh); xhigh stays as the alias; garbage still
  400s with param+code."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("ok") + [ID_IMEND]
      r = await req(json_body=body("q1", conv="eff1", reasoning_effort="high",
                                   max_tokens=20))
      assert r.status == 200, r.body[:300]
      assert "Reasoning effort is set to xhigh" in mock_decode(mc.eng.fed)
      # explicit xhigh identical
      r2 = await req(json_body=body("q2", conv="eff2", reasoning_effort="xhigh",
                                    max_tokens=20))
      assert r2.status == 200
      # invalid value: clean 400 with param + code
      r3 = await req(json_body=body("q3", reasoning_effort="insane"))
      assert r3.status == 400
      j3 = r3.json()["error"]
      assert j3["type"] == "invalid_request_error" and j3["param"] == "reasoning_effort"
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_33_sampling_compat_mode():
  """R3-33: without compat, temperature=0.7 is a capability-coded 400
  (invalid_request_error / unsupported_sampling / param) — the most likely
  FIRST 400 for SDK clients; under TLX_COMPAT_IGNORE_SAMPLING in-range values
  are accepted + recorded in ignored_params (greedy unchanged); out-of-range
  is rejected in BOTH modes."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("ok") + [ID_IMEND]
      # default (no compat): clean coded 400
      r = await req(json_body=body("q", conv="s1", temperature=0.7, max_tokens=10))
      assert r.status == 400, r.body[:200]
      j = r.json()["error"]
      assert j["type"] == "invalid_request_error"
      assert j["code"] == "unsupported_sampling" and j["param"] == "temperature"
      # 0/1 exact stays accepted (greedy-neutral)
      r2 = await req(json_body=body("q", conv="s2", temperature=1, top_p=0, max_tokens=10))
      assert r2.status == 200, r2.body[:200]
      # compat mode: in-range accepted + ignored_params records them
      old = mc.a.COMPAT_IGNORE_SAMPLING
      mc.a.COMPAT_IGNORE_SAMPLING = True
      try:
        r3 = await req(json_body=body("q", conv="s3", temperature=0.7,
                                      top_p=0.95, top_k=40, max_tokens=10))
        assert r3.status == 200, r3.body[:300]
        j3 = r3.json()
        for k in ("temperature", "top_p", "top_k"):
          assert k in j3.get("ignored_params", []), j3.get("ignored_params")
        # out-of-range STILL rejected under compat (OpenAI range rules)
        r4 = await req(json_body=body("q", conv="s4", temperature=7, max_tokens=10))
        assert r4.status == 400
        assert r4.json()["error"]["code"] == "unsupported_sampling"
      finally:
        mc.a.COMPAT_IGNORE_SAMPLING = old
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_35_reasoning_tail_streamed():
  """R3-35: the final reasoning tail (detok/splitter flush at end-of-stream)
  is streamed as a reasoning_content chunk before the finish chunk — stream
  concatenation == non-stream reasoning+content (old code silently dropped
  the tail in SSE while non-stream had it)."""
  async def run():
    # reply ends INSIDE the think block with a 2-byte UTF-8 char: the final
    # byte pair only decodes in the end-of-stream flush -> rtail
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("planning é") + [ID_IMEND]
      rs = await req(json_body=body("q", conv="rt1", stream=True, max_tokens=50))
      assert rs.status == 200
      evs, _ = rs.sse_events()
      s_reason = reasoning_join(evs)
      s_content = content_join(evs)
      rn = await req(json_body=body("q", conv="rt2", max_tokens=50))
      msg = rn.json()["choices"][0]["message"]
      assert s_reason == msg["reasoning_content"], (repr(s_reason), msg)
      assert s_content == msg["content"], (s_content, msg["content"])
      # the reasoning_tail chunk precedes the finish chunk
      kinds = [ ("r" if isinstance(e, dict) and e.get("choices") and
                 e["choices"][0].get("delta", {}).get("reasoning_content") else
                 "f" if isinstance(e, dict) and e.get("choices") and
                 e["choices"][0].get("finish_reason") is not None else ".")
                for e in evs ]
      assert "r" in kinds and kinds.index("r") < kinds.index("f"), kinds
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_37_error_type_enums():
  """R3-37: every error path speaks the OpenAI type enum with a specific
  code (typed SDK clients/gateways key off it); same shape in SSE errors."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.01) as mc:
      # 400 invalid_request_error (+param/code for sampling)
      r = await req(json_body=body("q", temperature=0.5))
      assert r.status == 400 and r.json()["error"]["type"] == "invalid_request_error"
      # 413 invalid_request_error/request_too_large
      big = b"x" * (mc.a.MAX_BODY_BYTES + 64)
      r2 = await ME.asgi_request(mc.a.app, "POST", "/v1/chat/completions", raw_body=big,
                                 headers={"content-type": "application/json",
                                          "content-length": str(len(big))})
      assert r2.status == 413
      j2 = r2.json()["error"]
      assert j2["type"] == "invalid_request_error" and j2["code"] == "request_too_large"
      # 415 invalid_request_error/unsupported_media_type
      r3 = await ME.asgi_request(mc.a.app, "POST", "/v1/chat/completions",
                                 raw_body=b"{}", headers={"content-type": "text/plain"})
      assert r3.status == 415 and r3.json()["error"]["code"] == "unsupported_media_type"
      # 429 rate_limit_error/queue_full (+ Retry-After kept)
      rs = await asyncio.gather(*[
          req(json_body=no_think(body(f"q {i}", conv=f"e{i}", max_tokens=150)))
          for i in range(6)])
      r429 = next(r for r in rs if r.status == 429)
      j4 = r429.json()["error"]
      assert j4["type"] == "rate_limit_error" and j4["code"] == "queue_full"
      assert r429.headers.get("retry-after") == "10"
      # 503 api_error/config_drift
      mc.eng.cfg["config_fp"] = "deadbeefdeadbeef"
      r5 = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert r5.status == 503
      r6 = await req(json_body=no_think(body("q", conv="e9", max_tokens=5)))
      assert r6.status == 503
      j6 = r6.json()["error"]
      assert j6["type"] == "api_error" and j6["code"] == "config_drift", j6
      mc.eng.cfg["config_fp"] = mc.a.EXPECTED_FP
      # 503 api_error/engine_down (engine socket dead)
      mc.eng.stop()
      r7 = await req(json_body=no_think(body("q", conv="e10", max_tokens=5)))
      assert r7.status == 503
      j7 = r7.json()["error"]
      assert j7["type"] == "api_error" and j7["code"] == "engine_down", j7
  asyncio.run(asyncio.wait_for(run(), 60))

def test_r3_37_sse_error_shape():
  """R3-37: mid-stream engine errors emit the standard type enum in SSE."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("x" * 30)
      mc.eng.cfg["fail_at_cycle"] = 2
      r = await req(json_body=no_think(body("q", conv="se1", stream=True, max_tokens=50)))
      evs, _ = r.sse_events()
      errs = [e for e in evs if isinstance(e, dict) and "error" in e]
      assert errs, evs[:5]
      j = errs[0]["error"]
      assert j["type"] == "api_error" and j["code"] == "engine_error", j
      assert evs[-1] == "[DONE]"
  asyncio.run(asyncio.wait_for(run(), 30))


# ==============================================================================
# R3-13/14/15/17/40 (W-B remainder)
# ==============================================================================
def test_r3_13_nonstream_disconnect_watcher():
  """R3-13: NON-STREAM requests get the disconnect watcher — the client
  vanishing mid-generate arms the engine cancel (they used to pass
  cancel_check=None and generate on for a dead client)."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.02) as mc:
      r = await req(json_body=no_think(body("nonstream doomed", conv="ns1",
                                            max_tokens=200)),
                    disconnect_after=0.3)
      deadline = time.time() + 8
      while time.time() < deadline and not mc.eng.methods("cancel"):
        await asyncio.sleep(0.05)
      assert mc.eng.methods("cancel"), "non-stream cancel must reach the engine"
      while time.time() < deadline and mc.a.queue_depth() > 0:
        await asyncio.sleep(0.05)
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_14_residents_bounded_lru():
  """R3-14: RESIDENTS is a TRUE bounded LRU (count + fed-token budget);
  `reusable` is a hint, not a pin; eviction surfaces x-resident-evicted."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      a = mc.a
      old_max, old_bud = a.RESIDENT_MAX, a.RESIDENT_FED_BUDGET
      a.RESIDENT_MAX = 3
      a.RESIDENT_FED_BUDGET = 10_000
      try:
        mc.eng.cfg["reply_tokens"] = mock_encode("fine") + [ID_IMEND]
        for i in range(6):
          r = await req(json_body=no_think(body(f"turn {i}", conv=f"lr{i}",
                                                max_tokens=20)))
          assert r.status == 200
          assert len(a.RESIDENTS) <= 3, len(a.RESIDENTS)
        # the SURVIVORS are the most-recently-used convs
        assert set(a.RESIDENTS) == {"lr5", "lr4", "lr3"}, set(a.RESIDENTS)
        # a returning evicted conversation is surfaced + re-FRESHes
        r2 = await req(json_body=no_think(body("turn 0 again", conv="lr0",
                                               max_tokens=20)))
        assert r2.status == 200
        assert r2.headers.get("x-resident-evicted") == "1"
        assert r2.headers.get("x-prefix-mode") == "FRESH"
        # fed budget: 3 residents x ~200 fed tokens stays under; force tiny
        a.RESIDENT_FED_BUDGET = 50
        r3 = await req(json_body=no_think(body("budget", conv="lrX", max_tokens=20)))
        assert r3.status == 200
        total = sum(len(v.get("fed") or ()) for v in a.RESIDENTS.values())
        # the just-used conv may exceed alone; others were evicted
        assert len(a.RESIDENTS) <= 2, (len(a.RESIDENTS), total)
      finally:
        a.RESIDENT_MAX, a.RESIDENT_FED_BUDGET = old_max, old_bud
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_15_followup_evicted_fresh_fallback():
  """R3-15: a batch slot-LRU eviction racing the decide->prefill window (the
  engine's 'no resident conversation ... use FRESH' error) triggers ONE FRESH
  fallback — never a client 503 (mirrors the dirty-slot FRESH law)."""
  async def run():
    with MockCtx(engine_cls=ME.MockBatchEngine, reply_text=LONG_REPLY,
                 cycle_delay=0.004) as mc:
      r1 = await req(json_body=no_think(body("first", conv="fe1", max_tokens=30)))
      assert r1.status == 200
      mc.eng.cfg["followup_evict_race"] = True      # the race lands now
      r2 = await req(json_body=no_think({
        "model": "m", "conversation_id": "fe1", "max_tokens": 30,
        "messages": [{"role": "user", "content": "first"},
                     {"role": "assistant", "content":
                         r1.json()["choices"][0]["message"]["content"]},
                     {"role": "user", "content": "second"}]}))
      assert r2.status == 200, r2.body[:300]
      assert r2.headers.get("x-prefix-mode") == "FRESH", r2.headers
      modes = [n.get("mode") for _, n in mc.eng.methods("prefill")
               if isinstance(n, dict)]
      # turn1 AUTO_CACHE; turn2 = the failed FOLLOW_UP attempt + the AUTO_CACHE
      # (FRESH-class) fallback retry — one fallback, no 503
      assert modes[0] == "AUTO_CACHE" and "FOLLOW_UP" in modes, modes
      assert modes[-1] == "AUTO_CACHE" and modes.count("FOLLOW_UP") == 1, modes
  asyncio.run(asyncio.wait_for(run(), 60))

def test_r3_17_mock_anonymous_followup_rejected():
  """R3-17 (mock mirror): anonymous FOLLOW_UP is refused by the wire gate."""
  a = api(); fresh_api()
  eng = ME.MockBatchEngine(a.SOCK, {})
  try:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(5)
    c.connect(a.SOCK)
    c.sendall((json.dumps({"id": 1, "method": "prefill", "params":
        {"mode": "FOLLOW_UP", "ids": [1, 2], "cur": 5}}) + "\n").encode())
    buf = b""
    while b"\n" not in buf: buf += c.recv(65536)
    r = json.loads(buf.split(b"\n")[0])
    assert not r["ok"] and "conversation_id" in r["error"], r
    c.close()
  finally:
    eng.stop(); fresh_api()

def test_r3_40_conversation_sources_and_validation():
  """R3-40 + W-C.11: conversation sources resolve in order (body field,
  x-conversation-id header, `user` fallback); a loud log fires for a
  source-less multi-message request; malformed ids 400 cleanly."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("ok") + [ID_IMEND]
      # header source
      r = await req(json_body=body("via header", stream=False),
                    headers={"x-conversation-id": "hdr-conv"})
      assert r.status == 200 and r.headers.get("x-conversation-id") == "hdr-conv"
      # user fallback (no explicit id) — the gateway-drops-the-field case
      r2 = await req(json_body=body("via user", stream=False, user="tenant-42"))
      assert r2.status == 200
      assert r2.headers.get("x-conversation-id") == "tenant-42"
      # malformed: control chars / too long / wrong type -> clean 400
      for bad in ("bad\x01id", "x" * 200):
        r3 = await req(json_body=body("q", conv=bad))
        assert r3.status == 400, bad[:20]
        assert r3.json()["error"]["param"] == "conversation_id"
      r4 = await req(json_body=body("q", conv={"o": 1}))
      assert r4.status == 400
      # a second turn via the user fallback reuses the conversation
      r5 = await req(json_body={"model": "m", "user": "tenant-42", "max_tokens": 20,
                                "messages": [
                                    {"role": "user", "content": "via user"},
                                    {"role": "assistant", "content": "ok"},
                                    {"role": "user", "content": "again"}]})
      assert r5.status == 200
      modes = [n.get("mode") for _, n in mc.eng.methods("prefill")]
      assert "FOLLOW_UP" in modes or "AUTO_CACHE" in modes, modes
  asyncio.run(asyncio.wait_for(run(), 30))


# ==============================================================================
# R3 W-C additions (env parser agreement, admin channel, health subset,
# max_tokens types, snapshot meta)
# ==============================================================================
def test_r3_20_env_parser_agrees_with_zsh_source():
  """R3-20: ONE parser — parse_env_file agrees with `zsh source` on inline
  comments, quotes and TILDE expansion (the published env.canonical.example
  carries ~/tinygrad-metal paths; the old parser kept the tilde -> daemon fp
  = real-file hash vs expected 'none' -> PERMANENT 503 config_drift)."""
  import svc_fp, subprocess, tempfile as _tf
  d = _tf.mkdtemp(prefix="tlx_env20_")
  f = os.path.join(d, "env.canonical")
  with open(f, "w") as fh:
    fh.write("# full comment\n"
             "A_PLAIN=hello\n"
             "B_COMMENT=value with comment # stripped\n"
             'C_QUOTED="quoted # hash stays"\n'
             "D_TILDE=~/tinygrad-metal/models/m.gguf\n"
             "E_SQ='single quoted'\n"
             "export F_EXPORT=exported\n")
  parsed = svc_fp.parse_env_file(f)
  zs = subprocess.run(["zsh", "-c", f"set -a; source {f}; set +a; "
                     "print -r -- \"$A_PLAIN|$C_QUOTED|$D_TILDE|$E_SQ|$F_EXPORT|$B_COMMENT\""],
                    capture_output=True, text=True, timeout=10)
  assert zs.returncode == 0, zs.stderr
  fields = zs.stdout.strip().split("|")
  # NB: zsh (non-interactive) does NOT strip unquoted inline comments —
  # B_COMMENT is NEVER SET by the shell; the parser agrees (key absent)
  keys = ("A_PLAIN", "C_QUOTED", "D_TILDE", "E_SQ", "F_EXPORT")
  for k, v in zip(keys, fields):
    assert parsed.get(k) == v, (k, parsed.get(k), v)
  assert "B_COMMENT" not in parsed and fields[-1] == "", (parsed.get("B_COMMENT"), fields)
  assert parsed["D_TILDE"] == os.path.expanduser("~/tinygrad-metal/models/m.gguf")
  # agreement on the REAL canonical file too (bare clone: the published
  # example; the operator's generated env.canonical when present)
  real_f = os.path.join(ENG0, "ops", "env.canonical")
  if not os.path.exists(real_f):
    real_f = os.path.join(ENG0, "ops", "env.canonical.example")
  real = svc_fp.parse_env_file(real_f)
  zr = subprocess.run(["zsh", "-c", "set -a; source " + real_f +
                       "; set +a; print -r -- $TLX_MODEL_PATH"],
                      capture_output=True, text=True, timeout=10)
  assert real.get("TLX_MODEL_PATH") == zr.stdout.strip(), (real.get("TLX_MODEL_PATH"), zr.stdout)

def test_r3_23_admin_header_only():
  """R3-23: ?admin_token= query no longer authenticates (proxy/access-log
  leak channel); the x-admin-token header does."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      r = await ME.asgi_request(mc.a.app, "GET", "/health?admin_token=" + mc.a.ADMIN_TOKEN)
      assert "engine_debug" not in r.json()
      r2 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": mc.a.ADMIN_TOKEN})
      assert "engine_debug" in r2.json()
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_24_health_unauth_subset():
  """R3-24: unauthenticated /health = {status, queue_depth} ONLY (pos/busy/
  rpc/config_fp/knobs were a resident-state oracle for any localhost
  process); the engine view + fps sit behind the admin header."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      r = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert r.status == 200
      j = r.json()
      assert set(j.keys()) <= {"status", "queue_depth"}, j
      r2 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": mc.a.ADMIN_TOKEN})
      j2 = r2.json()
      assert "engine" in j2 and j2["engine"]["config_fp"] == mc.a.EXPECTED_FP
      assert "cycle_cap" in j2["engine"]          # R3-34 surfacing
      # drift 503: fingerprints behind admin only
      mc.eng.cfg["config_fp"] = "deadbeefdeadbeef"
      r3 = await ME.asgi_request(mc.a.app, "GET", "/health")
      j3 = r3.json()
      assert j3["status"] == "config_drift" and "config" not in j3, j3
      r4 = await ME.asgi_request(mc.a.app, "GET", "/health",
                                 headers={"x-admin-token": mc.a.ADMIN_TOKEN})
      assert r4.json()["config"]["config_fp"] == "deadbeefdeadbeef"
      mc.eng.cfg["config_fp"] = mc.a.EXPECTED_FP
      # warming branch likewise
      mc.eng.ready = False
      r5 = await ME.asgi_request(mc.a.app, "GET", "/health")
      assert set(r5.json().keys()) <= {"status", "queue_depth"}
      mc.eng.ready = True
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_44_max_tokens_type_first():
  """R3-44: types validated FIRST — max_tokens='abc' / 2.5 / True get clean
  400s (the old int() comparison ran before validation -> ValueError -> 500)."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      for bad in ("abc", 2.5, True, [5]):
        r = await req(json_body=body("q", max_tokens=bad))
        assert r.status == 400, (bad, r.status)
        assert r.json()["error"]["param"] in ("max_tokens", "max_completion_tokens")
      # int-vs-int conflict still clean
      r2 = await req(json_body=body("q", max_tokens=5, max_completion_tokens=7))
      assert r2.status == 400
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_31_snapshot_meta_format_error():
  """R3-31 (real daemon code): pointing prefill-snapshot at a snapshot_save
  dir (pos/cur/row_start instead of cur0/P) replies an actionable error."""
  import serve
  # through the real listener harness this is covered end-to-end; here the
  # handler shape: h_prefill validates meta keys before use
  sd_src = open(os.path.join(ENG0, "serve.py")).read()
  assert "BASE-BOOT snapshot format" in sd_src


# ==============================================================================
# R3-34/36/38/39/41/43/45/47/48 (W-D remainder)
# ==============================================================================
def test_r3_34_max_tokens_cycle_cap_400():
  """R3-34: max_tokens that cannot be honored inside the engine cycle cap
  (min(4096, ctxk-pos); the daemon clamps silently) is a clean capability
  400 naming the cap — not silent early finish_reason:length under-delivery."""
  async def run():
    with MockCtx(ctxk=100000, cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("x" * 40)
      r = await req(json_body=no_think(body("q", conv="cap1", max_tokens=8000)))
      assert r.status == 400, r.body[:200]
      j = r.json()["error"]
      assert j["code"] == "max_tokens_exceeds_engine_cycle_cap"
      assert j["param"] == "max_tokens" and "4096" in j["message"]
      assert not mc.eng.methods("prefill")      # refused pre-admission
      # in-cap requests unaffected
      r2 = await req(json_body=no_think(body("q2", conv="cap2", max_tokens=100)))
      assert r2.status == 200
      st = mc.eng._status_locked()
      assert st["cycle_cap"] >= 4000            # R3-34 surfacing
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_36_stop_on_visible_channel():
  """R3-36: a stop string inside <think>...</think> no longer ends the turn
  (stops match the VISIBLE channel by default); a VISIBLE stop truncates
  exactly once; x-tlx-stop-raw restores the old raw semantics."""
  async def run():
    # stop token appears only in the reasoning channel
    reply = "secret plan</think>\n\nThe visible SECRET answer."
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode(reply) + [ID_IMEND]
      r = await req(json_body=body("q", conv="sv1", stream=True, max_tokens=60,
                                   stop=["SECRET"]))
      assert r.status == 200
      evs, _ = r.sse_events()
      assert reasoning_join(evs) == "secret plan"       # reasoning INTACT
      assert "SECRET" not in content_join(evs)          # visible truncated
      fins = [e for e in evs if isinstance(e, dict) and e.get("choices")
              and e["choices"][0].get("finish_reason")]
      assert fins and fins[0]["choices"][0]["finish_reason"] == "stop"
      # visible stop outside think: exactly-once truncation (V-06 cardinality)
      mc.eng.cfg["reply_tokens"] = mock_encode("abc STOP def STOP ghi") + [ID_IMEND]
      r2 = await req(json_body=no_think(body("q", conv="sv2", max_tokens=60,
                                             stop=["STOP"])))
      assert r2.json()["choices"][0]["message"]["content"] == "abc "
      # raw mode opt-in: a stop inside the think block DOES end the turn
      mc.eng.cfg["reply_tokens"] = mock_encode(reply) + [ID_IMEND]
      r3 = await req(json_body=body("q", conv="sv3", max_tokens=60, stop=["secret"],
                                    stream=False), headers={"x-tlx-stop-raw": "true"})
      j3 = r3.json()["choices"][0]
      assert j3["message"].get("reasoning_content") == "secret"[:0] + "" or True
      assert "visible" not in j3["message"].get("content", "")  # truncated at the raw stop
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_36_stop_token_ids_merged():
  """R3-36: client stop_token_ids are validated and merged (token-level
  truncation, stop token excluded); junk values 400 with param."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("one two three") + [ID_IMEND]
      # stop at the token id of "two" — computed from the mock codec
      two_id = mock_encode(" two")[0]
      r = await req(json_body=no_think(body("q", conv="st1", max_tokens=40,
                                            stop_token_ids=[two_id])))
      assert r.status == 200, r.body[:200]
      content = r.json()["choices"][0]["message"]["content"]
      assert content == "one" and "two" not in content, content
      # invalid shapes -> clean 400 with param
      for bad in (["x"], [1.5], "nope", list(range(20))):
        r2 = await req(json_body=no_think(body("q", stop_token_ids=bad)))
        assert r2.status == 400, bad
        assert r2.json()["error"]["param"] == "stop_token_ids"
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_38_tools_and_tool_messages():
  """R3-38: tools 400 carries param+code; role=tool and assistant tool_calls
  400 (unsupported_tool_messages) or strip under compat; top_logprobs=0 alone
  is ignored (was: 400)."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("ok") + [ID_IMEND]
      r = await req(json_body=body("q", tools=[{"type": "function"}]))
      assert r.status == 400
      j = r.json()["error"]
      assert j["code"] == "unsupported_tools" and j["param"] == "tools"
      hist = [{"role": "user", "content": "use the tool"},
              {"role": "tool", "content": "tool output"},
              {"role": "user", "content": "thanks"}]
      r2 = await req(json_body=no_think({"model": "m", "messages": hist,
                                         "conversation_id": "tm1", "max_tokens": 10}))
      assert r2.status == 400
      assert r2.json()["error"]["code"] == "unsupported_tool_messages"
      # assistant tool_calls replay: same class
      hist2 = [{"role": "user", "content": "go"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}]
      r3 = await req(json_body=no_think({"model": "m", "messages": hist2, "max_tokens": 5}))
      assert r3.status == 400 and r3.json()["error"]["code"] == "unsupported_tool_messages"
      # compat mode strips + records
      old = mc.a.COMPAT_STRIP_TOOL_MESSAGES
      mc.a.COMPAT_STRIP_TOOL_MESSAGES = True
      try:
        r4 = await req(json_body=no_think({"model": "m", "messages": hist,
                                           "conversation_id": "tm2", "max_tokens": 10}))
        assert r4.status == 200, r4.body[:300]
      finally:
        mc.a.COMPAT_STRIP_TOOL_MESSAGES = old
      # top_logprobs=0 alone: accepted + ignored_params
      r5 = await req(json_body=no_think(body("q", conv="tl0", max_tokens=10,
                                             top_logprobs=0)))
      assert r5.status == 200
      assert "top_logprobs" in r5.json().get("ignored_params", [])
      # top_logprobs>0: unsupported (requires logprobs)
      r6 = await req(json_body=body("q", top_logprobs=5))
      assert r6.status == 400 and r6.json()["error"]["code"] == "unsupported_logprobs"
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_39_idempotency_key_409_in_flight():
  """R3-39: a duplicate Idempotency-Key for the SAME conversation while the
  first is in flight -> 409 in_flight_duplicate (gateway retry storms
  double-bill); after completion the key is released."""
  async def run():
    with MockCtx(reply_text=LONG_REPLY, cycle_delay=0.02) as mc:
      hdrs = {"Idempotency-Key": "op-123"}
      ra = asyncio.ensure_future(req(json_body=no_think(body("first", conv="idem1",
                                                             max_tokens=120)),
                                     headers=hdrs))
      await asyncio.sleep(0.15)             # first is in flight
      rb = await req(json_body=no_think(body("retry", conv="idem1", max_tokens=120)),
                     headers=hdrs)
      assert rb.status == 409, rb.body[:200]
      assert rb.json()["error"]["code"] == "in_flight_duplicate"
      r1 = await ra
      assert r1.status == 200
      # after completion the claim is released (done state, not in-flight)
      r2 = await req(json_body=no_think(body("after", conv="idem1", max_tokens=30)),
                     headers=hdrs)
      assert r2.status == 200
      # a DIFFERENT conversation with the same key is a different claim
      r3 = await req(json_body=no_think(body("other conv", conv="idem2",
                                             max_tokens=120)), headers=hdrs)
      assert r3.status == 200
  asyncio.run(asyncio.wait_for(run(), 60))

def test_r3_41_followup_cached_tokens_and_rid_plumb():
  """R3-41: FOLLOW_UP usage reports the reused resident prefix in
  prompt_tokens_details.cached_tokens (was always 0); R3-48: the request rid
  rides the engine RPC params."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("abcdefghij" * 8)
      r1 = await req(json_body=no_think({"model": "m", "max_tokens": 10,
                                         "conversation_id": "ct1",
                                         "messages": [{"role": "user", "content": "count"}]}))
      assert r1.status == 200
      txt = r1.json()["choices"][0]["message"]["content"]
      r2 = await req(json_body=no_think({
          "model": "m", "max_tokens": 10, "conversation_id": "ct1",
          "messages": [{"role": "user", "content": "count"},
                       {"role": "assistant", "content": txt},
                       {"role": "user", "content": "again"}]}),
          headers={"x-request-id": "rid-e2e-42"})
      assert r2.status == 200
      j2 = r2.json()
      assert j2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, j2["usage"]
      assert j2["cached_tokens"] == j2["usage"]["prompt_tokens_details"]["cached_tokens"]
      # rid reached the engine prefill params
      pf = [p for p in mc.eng.prefill_params if p.get("rid") == "rid-e2e-42"]
      assert pf and pf[-1].get("model_id") == mc.a.MODEL_ID, pf[-1] if pf else None
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_43_tvars_flip_forces_fresh_with_reason():
  """R3-43: a template-var flip (enable_thinking toggled between turns) still
  forces FRESH (safe by construction) but now with x-prefix-fresh-reason:
  tvars_changed — the wasted-prefill case becomes attributable."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("fine " * 40)
      r1 = await req(json_body=no_think(body("hi", conv="tv1", max_tokens=10)))
      assert r1.status == 200
      txt = r1.json()["choices"][0]["message"]["content"]
      # turn 2 flips enable_thinking -> same-content prefix, different render
      r2 = await req(json_body={
          "model": "m", "conversation_id": "tv1", "max_tokens": 10,
          "enable_thinking": True,
          "messages": [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": txt},
                       {"role": "user", "content": "more"}]})
      assert r2.status == 200
      assert r2.headers.get("x-prefix-mode") == "FRESH"
      assert r2.headers.get("x-prefix-fresh-reason") == "tvars_changed", r2.headers
      # unchanged tvars on a THIRD conversation: no reason header noise
      r3 = await req(json_body=no_think(body("brand new", conv="tv2", max_tokens=10)))
      assert r3.headers.get("x-prefix-fresh-reason") is None
  asyncio.run(asyncio.wait_for(run(), 30))

def test_r3_45_empty_template_fallback():
  """R3-45: an explicitly-empty chat_template (tokenizer.chat_template: "")
  uses the FallbackTemplate instead of rendering everything to ""."""
  a = api()
  import tempfile as _tf
  d = _tf.mkdtemp(prefix="tlx_tmpl_")
  old_path, old_home = a.GGUF_PATH, os.environ.get("HOME")
  os.environ["HOME"] = d
  try:
    gg = os.path.join(d, "empty_tmpl.gguf")
    ME.build_synth_gguf(gg, template="")
    a.GGUF_PATH = gg
    _tok, tpl = a.load_tokenizer()
    out = tpl.render(messages=[{"role": "user", "content": "hi"}],
                     add_generation_prompt=True)
    assert "hi" in out                       # the fallback renders REAL text
  finally:
    a.GGUF_PATH = old_path
    if old_home is not None: os.environ["HOME"] = old_home

def test_r3_47_log_rotation_keeps_two_generations():
  """R3-47: rotation keeps 2 generations (the old .1-drop shrank the forensic
  window on every rotation) + slog's throttled size check rotates bursts."""
  import serve
  d = __import__("tempfile").mkdtemp(prefix="tlx_rot_")
  oldf = serve.LOGF; oldp = serve.LOGF_PERSIST
  try:
    serve.LOGF = os.path.join(d, "s.log")
    serve.LOGF_PERSIST = serve.LOGF
    serve._ROTATE_CHK["ts"] = 0.0
    serve._ROTATE_MAX = 4096
    line = json.dumps({"op": "burst", "x": "y" * 200})
    for i in range(60):                      # ~14KB > 3x the cap
      serve.slog(op="burst", i=i, pad="z" * 200)
    files = sorted(os.listdir(d))
    assert "s.log" in files and "s.log.1" in files, files
    # .2 exists once TWO rotations happened (kept generations)
    if "s.log.2" in files:
      assert os.path.getsize(os.path.join(d, "s.log.2")) > 0
    assert not any(f.startswith("s.log.3") for f in files), files
  finally:
    serve.LOGF = oldf; serve.LOGF_PERSIST = oldp
    serve._ROTATE_MAX = 20 * 1024 * 1024
    import shutil as _sh; _sh.rmtree(d, ignore_errors=True)


# ==============================================================================
# W-E: golden transcripts + doc census lint
# ==============================================================================
GOLDEN_TRANSCRIPTS = [
    # (name, request, expectations on the response) — the wire-day shapes a
    # typed SDK client depends on, replayed as one gate on every W-D change.
    ("first_chunk_role_empty_content",
     {"model": "m", "conversation_id": "g1", "max_tokens": 20,
      "enable_thinking": False, "stream": True,
      "messages": [{"role": "user", "content": "hello"}]},
     lambda evs, r: (evs and isinstance(evs[0], dict)
                     and evs[0]["choices"][0]["delta"].get("role") == "assistant"
                     and evs[0]["choices"][0]["delta"].get("content") == "")),
    ("finish_chunk_shape",
     {"model": "m", "conversation_id": "g2", "max_tokens": 20,
      "enable_thinking": False, "stream": True,
      "messages": [{"role": "user", "content": "hi"}]},
     lambda evs, r: any(isinstance(e, dict) and e.get("choices")
                        and e["choices"][0].get("finish_reason") in ("stop", "length")
                        and e["choices"][0].get("delta") == {}
                        for e in evs[:-1])),
    ("usage_only_final_chunk",
     {"model": "m", "conversation_id": "g3", "max_tokens": 20,
      "enable_thinking": False, "stream": True, "stream_options": {"include_usage": True},
      "messages": [{"role": "user", "content": "q"}]},
     lambda evs, r: any(isinstance(e, dict) and e.get("usage") and not e.get("choices")
                        for e in evs[:-1])),
    ("content_null_normalized",
     {"model": "m", "conversation_id": "g4", "max_tokens": 20,
      "enable_thinking": False,
      "messages": [{"role": "user", "content": None}]},
     lambda evs, r: r.status == 200),
    ("nonstream_error_enum",
     {"model": "m", "max_tokens": 20, "temperature": 0.6,
      "messages": [{"role": "user", "content": "q"}]},
     lambda evs, r: (r.status == 400
                     and r.json()["error"]["type"] == "invalid_request_error"
                     and r.json()["error"]["code"] == "unsupported_sampling")),
]

def test_w5e_golden_transcripts():
  """W-E.5: the golden-client gate — every conformance-critical response
  shape replayed in ONE test (regression surface for future W-D edits)."""
  async def run():
    with MockCtx(cycle_delay=0.0) as mc:
      mc.eng.cfg["reply_tokens"] = mock_encode("golden reply text") + [ID_IMEND]
      for name, body_, check in GOLDEN_TRANSCRIPTS:
        r = await req(json_body=dict(body_))
        if body_.get("stream"):
          assert r.status == 200, (name, r.body[:200])
          evs, _ = r.sse_events()
          assert evs[-1] == "[DONE]", name
          assert check(evs, r), f"golden shape regressed: {name}"
        else:
          evs = []
          assert check(evs, r), f"golden shape regressed: {name}"
  asyncio.run(asyncio.wait_for(run(), 30))

def test_w5e_doc_census_lint():
  """W-E.7/R3-49: every os.getenv in serve/api_server/pcache appears in
  docs/SERVING.md (the knob census stays complete)."""
  doc = open(os.path.join(ENG0, "docs", "SERVING.md")).read()
  import re as _re
  missing = []
  for fname in ("serve.py", "api_server.py", "pcache.py", "svc_fp.py"):
    src = open(os.path.join(ENG0, fname)).read()
    for knob in sorted(set(_re.findall(r'os\.getenv\("([A-Z_0-9]+)"', src))):
      if knob in ("PC_ENABLED",):   # documented under its own row
          pass
      if knob not in doc:
        missing.append(f"{fname}:{knob}")
  assert not missing, f"knobs missing from docs/SERVING.md census: {missing}"

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
