# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/usr/bin/env python3
"""M1-B API façade — OpenAI-compatible chat completions on top of the engine
daemon (/tmp/llm-engine.sock). ZERO GPU/tinygrad imports: the tokenizer is
vendored (SimpleTokenizer from the fork's tinygrad/llm/cli.py, byte-identical
math) plus a minimal GGUF-KV parser; this process is restartable at any time.

Run: python3 api_server.py   (system python3 + fastapi/uvicorn/jinja2 --user)
Env: PORT (default 8080), ENGINE_SOCK, GGUF (model path), BIND (127.0.0.1).

W1 review-fix wave (TLX_REVIEW_LEDGER, branch review-fixes-w1):
  V-01  streaming requests hold the FIFO slot for the whole SSE lifetime
        (release moved into the stream generator's finally + a guard task).
  V-02  headroom is mode-aware (FRESH computes from the new prompt only) —
        kills the permanent-400 lockout after a long conversation.
  V-03  per-conversation asyncio lock around the request + engine-side
        FOLLOW_UP conversation guard (serve.py).
  V-04  FIFO slot acquired only AFTER body parse+validation; body-read
        deadline; queue-waiter deadline (V-16 part; bounded event queue).
  V-06  DetokStream.final_text no longer re-emits already-streamed text.
  V-07  reasoning_effort/enable_thinking forwarded as template vars (default
        effort "medium", NOT the template's xhigh); a leading think-block is
        split out of content into reasoning_content (stream + non-stream);
        reasoning tokens do not count against max_tokens (visible-answer
        guarantee); the strict length cap applies to VISIBLE content tokens.
  V-08  max_completion_tokens accepted as an alias (mutual-exclusion check).
  V-09  content:null mapped to ""; textless {"type":"text"} parts -> 400.
  V-10  generate-exception path marks the conversation dirty/non-reusable,
        resets RESIDENT; engine error replies during generate raise promptly.
  V-11  mid-stream engine errors become an SSE error event + [DONE].
  V-12  RESIDENT.reusable set only after prefill+state confirmed; prefill
        failure resets RESIDENT (no phantom assistant turn).
  V-13  client-visible completion tokens never exceed max_tokens (strict);
        the engine-fed overshoot tail stays hidden and forces FRESH reuse
        semantics next turn (mirror==client contract).
  V-14  eos/eot id 0 kept (identity, not truthiness); stop ids vocab-checked.
  V-21  RESIDENT["fed"] extended in place (no O(n^2) rebuild).

Field matrix / protocol / ops: see M1B_SERVING.md.

W2 review-fix wave (TLX_REVIEW_LEDGER):
  V-25 tokenizer cache: moved out of /tmp into ~/Library/Caches/tlx (0700)
        with an HMAC integrity tag; tampered/short cache -> cold rebuild.
  V-34 HTTP hardening: TrustedHost (localhost allowlist, env-extensible),
        request body cap (content-length AND streaming-byte counter) before
        parse, application/json content-type required, CORS stays default-deny
        (no CORS middleware = browsers cannot read cross-origin responses).
  V-24-adjacent jinja sandbox: GGUF chat template renders in
        jinja2.sandbox.SandboxedEnvironment (attribute/underscore access denied).
  V-28/V-33 /health: config fingerprint + knobs surfaced; fed_tail /
        conversation_id redacted by default (ops detail behind the admin
        token); 503 config_drift when the daemon fp != the canonical env fp
        (same check on /v1/chat/completions — silent slow-path degradation
        becomes a loud refusal)."""
from __future__ import annotations
import os, sys, json, time, socket, struct, uuid, asyncio, itertools, re
import hmac, hashlib
import itertools as _it, unicodedata
import queue as _pyqueue

GGUF_PATH = os.getenv("GGUF", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
SOCK = os.getenv("ENGINE_SOCK", "/tmp/llm-engine.sock")
PORT = int(os.getenv("PORT", "8080"))
BIND = os.getenv("BIND", "127.0.0.1")
MODEL_ID = "qwen3.8-27b-egpu"
MAX_WAITING = 4            # FIFO queue cap beyond the 1 active request
HEADROOM = 32              # ctx safety margin for stop/over-commit
# W1 knobs (V-04/V-16): deadlines for the pre-slot phases and the waiter.
BODY_READ_TIMEOUT = float(os.getenv("TLX_BODY_TIMEOUT_S", "30"))
QUEUE_WAIT_S = float(os.getenv("TLX_QUEUE_WAIT_S", "300"))
STREAM_GUARD_GRACE_S = float(os.getenv("TLX_GUARD_GRACE_S", "30"))
EVQ_MAX = int(os.getenv("TLX_EVQ_MAX", "8192"))

# ---- TLX W2 knobs -------------------------------------------------------------
ADMIN_TOKEN = os.getenv("TLX_ADMIN_TOKEN", "")   # mirrors the daemon's; gates
                                                 # /health debug fields (V-33)
MAX_BODY_BYTES = int(float(os.getenv("TLX_MAX_BODY_MB", "10")) * 1e6)   # V-34
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "::1"] + [
    h.strip() for h in os.getenv("TLX_ALLOWED_HOSTS", "").split(",") if h.strip()]
OPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")
ENV_CANONICAL = os.getenv("TLX_ENV_CANONICAL", os.path.join(OPS_DIR, "env.canonical"))
ENGINE_STAYDOWN_PATHS = (
    "~/tinygrad-metal/logs/llm_engine_staydown",   # W2 persistent (V-26)
    "/tmp/llm_engine_staydown")                              # legacy path

import svc_fp   # the ONE fp implementation (shared with pcache.py/serve.py)

def _expected_config():
    """Expected daemon fingerprint derived from ops/env.canonical (V-27/V-28):
    the single-sourced engine env. None when the canonical file is absent ->
    drift check disabled (manual/dev boots), loudly noted in /health."""
    env = svc_fp.parse_env_file(ENV_CANONICAL)
    if not env:
        return None, None
    model = env.get("TLX_MODEL_PATH") or GGUF_PATH
    return svc_fp.config_fp(env=env, model_path=model), env

EXPECTED_FP, EXPECTED_ENV = _expected_config()

# ================= GGUF KV parser (metadata only; no tensors) =================
def parse_gguf_kv(path):
  f = open(path, "rb")
  assert f.read(4) == b"GGUF", "not a gguf"
  struct.unpack("<I", f.read(4))
  struct.unpack("<Q", f.read(8))          # n_tensors
  n_kv, = struct.unpack("<Q", f.read(8))
  def rd_str():
    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
  def rd_val(t):
    if t == 0: return struct.unpack("<B", f.read(1))[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return bool(f.read(1)[0])
    if t == 8: return rd_str()
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    if t == 9:
      et, = struct.unpack("<I", f.read(4)); cnt, = struct.unpack("<Q", f.read(8))
      return [rd_val(et) for _ in range(cnt)]
    raise ValueError(f"gguf val type {t}")
  kv = {}
  for _ in range(n_kv):
    k = rd_str(); t, = struct.unpack("<I", f.read(4)); kv[k] = rd_val(t)
  f.close()
  return kv

# ================= Vendored SimpleTokenizer (fork cli.py, verbatim math) ======
class SimpleTokenizer:
  def __init__(self, normal_tokens: dict, special_tokens: dict, preset: str = "llama3",
               bos_id=None, eos_id: int = 0, eot_id=None):
    preset = {"qwen35": "qwen2", "qwen35moe": "qwen2"}.get(preset, preset)
    if preset not in ("llama3", "llama-v3", "llama-bpe", "qwen2", "olmo", "kimi-k2", "tekken", "glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i, b in enumerate(b for b in range(256) if b not in bs)}
    def ucat_range(pre: str) -> str:
      cps = enumerate(cp for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
      runs = [list(g) for _, g in _it.groupby(cps, lambda e: e[1] - e[0])]
      return "".join(re.escape(chr(g[0][1])) + (f"-{re.escape(chr(g[-1][1]))}" if len(g) > 1 else "") for g in runs)
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")
    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv: dict):
    normal, special = [], []
    for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]):
      (normal if kv["tokenizer.ggml.token_type"][idx] == 1 else special).append((tok, idx))
    special_dict = dict(special)
    return SimpleTokenizer(dict(normal), special_dict, kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id', special_dict.get('<|im_end|>')))

  def _encode_word(self, word: bytes):
    if (early_token := self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk: str):
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text: str):
    tokens, pos = [], 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])
  def decode(self, ids): return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def is_end(self, token_id: int): return token_id in (self.eos_id, self.eot_id)

class FallbackTemplate:  # vendored (fork cli.py) — only used if GGUF lacks a template
  def __init__(self, tok): self.tok = tok
  def role(self, role):
    if self.tok.preset == 'olmo': return "<|" + role + "|>\n"
    if self.tok.preset == 'kimi-k2': return "<|im_" + role + "|>" + role + "<|im_middle|>"
    if self.tok.preset == 'qwen2': return "<|im_start|>" + role + "\n"
    if self.tok.preset == 'glm4': return "<|" + role + "|>"
    if self.tok.preset == 'tekken':
      if role == 'user': return "[INST]"
      if role == 'assistant': return ""
      raise ValueError(f"Unsupported role '{role}' for preset '{self.tok.preset}'")
    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
  def end_turn(self):
    if self.tok.preset == 'olmo': return "\n"
    if self.tok.preset == 'kimi-k2': return self.tok.decode([self.tok.eos_id])
    if self.tok.preset == 'qwen2': return self.tok.decode([self.tok.eos_id]) + "\n"
    if self.tok.preset == 'glm4': return ""
    if self.tok.preset == 'tekken': return "[/INST]"
    return self.tok.decode([self.tok.eos_id])
  def render(self, messages, tools=None, add_generation_prompt=True, preserve_thinking=False):
    out = self.tok.decode([] if self.tok.bos_id is None else [self.tok.bos_id]) + ("<sop>" if self.tok.preset == 'glm4' else "")
    for msg in messages:
      out += self.role(msg["role"])
      content = msg.get("content")
      if isinstance(content, str): out += content
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": out += c["text"]
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      elif content is not None: raise RuntimeError(f"unknown content type: {type(content)}")
      out += self.end_turn()
    return out + self.role("assistant") if add_generation_prompt else out

# jinja template callables MUST be module-level: the (tok, template) pair is
# pickled into the tokenizer cache, and lambdas/closures fail to pickle (the
# old /tmp cache write silently never worked for jinja templates).
def _jinja_tojson(obj, **kwargs):
  return json.dumps(obj, **kwargs)

def _jinja_raise_exception(msg):
  raise ValueError(str(msg))

def _tok_cache_paths():
  """V-25: cache under a 0700 owner dir (never /tmp — world-writable + wiped);
  name keyed on gguf size+mtime as before."""
  d = os.path.expanduser("~/Library/Caches/tlx")
  try:
    os.makedirs(d, mode=0o700, exist_ok=True)
    if hasattr(os, "chmod"):
      try: os.chmod(d, 0o700)      # makedirs mode is masked by umask; enforce
      except Exception: pass
  except Exception:
    return None, None
  st = os.stat(GGUF_PATH)
  key = hashlib.sha256(f"{GGUF_PATH}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:24]
  return os.path.join(d, f"tok_{key}.pkl"), os.path.join(d, "tok_hmac.key")

def _hmac_key(kpath):
  """Load/create the per-user HMAC key (0600). Integrity (not secrecy) is the
  goal: a tampered/swapped cache must FAIL the tag and fall back to cold load."""
  try:
    with open(kpath, "rb") as f: return f.read()
  except Exception: pass
  k = os.urandom(32)
  with open(kpath, "wb") as f: f.write(k)
  try: os.chmod(kpath, 0o600)
  except Exception: pass
  return k

def _build_template(ct, tok):
  """Template from the raw GGUF string. NOTE: jinja from_string templates are
  NOT picklable (compiled code objects), so the CACHE stores the raw string
  and the template is rebuilt on every load (cheap; the expensive part of a
  cold load is the GGUF KV parse + tokenizer construction)."""
  if ct is None:
    return FallbackTemplate(tok)
  import jinja2
  from jinja2.sandbox import SandboxedEnvironment
  env = SandboxedEnvironment()
  env.filters['tojson'] = _jinja_tojson
  env.globals['raise_exception'] = _jinja_raise_exception
  return env.from_string(ct)

def load_tokenizer():
  import pickle
  cache, kpath = _tok_cache_paths()
  def _cache_load():
    if not cache or not os.path.exists(cache): return None
    try:
      key = _hmac_key(kpath)
      raw = open(cache, "rb").read()
      tag, payload = raw[:32], raw[32:]
      if not hmac.compare_digest(hmac.new(key, payload, hashlib.sha256).digest(), tag):
        print("[api] tokenizer cache FAILED integrity tag — cold rebuild", flush=True)
        return None
      return pickle.loads(payload)     # (tok, template_string)
    except Exception:
      return None
  hit = _cache_load()
  if hit is not None:
    tok, ct = hit
    return tok, _build_template(ct, tok)
  t0 = time.time()
  kv = parse_gguf_kv(GGUF_PATH)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  ct = kv.get('tokenizer.chat_template')
  if ct is None:
    print("[api] WARNING: GGUF has no chat_template; using fallback", flush=True)
  obj = (tok, ct)
  if cache:
    try:      # atomic write: tag = HMAC(key, payload); tmp+rename
      key = _hmac_key(kpath)
      payload = pickle.dumps(obj, protocol=4)
      tag = hmac.new(key, payload, hashlib.sha256).digest()
      tmp = cache + ".tmp"
      with open(tmp, "wb") as f:
        f.write(tag + payload); f.flush(); os.fsync(f.fileno())
      os.replace(tmp, cache)
      try: os.chmod(cache, 0o600)
      except Exception: pass
    except Exception as e:
      print(f"[api] tokenizer cache write failed (non-fatal): {e}", flush=True)
  print(f"[api] tokenizer loaded ({len(tok._normal_tokens)} normal, {len(tok._special_tokens)} special, "
        f"template={'jinja' if ct else 'fallback'}, eos={tok.eos_id} eot={tok.eot_id}, {time.time()-t0:.1f}s)", flush=True)
  return tok, _build_template(ct, tok)

TOK, TEMPLATE = load_tokenizer()

# ================= engine client (blocking sockets; run in executor) ==========
class EngineError(Exception): pass

class EngConn:
  def __init__(self, timeout=600.0):
    self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.s.settimeout(timeout)
    try: self.s.connect(SOCK)
    except Exception as e: raise EngineError(f"engine socket: {e}")
    self.buf = b""
  def send(self, obj): self.s.sendall((json.dumps(obj) + "\n").encode())
  def recv(self):
    while b"\n" not in self.buf:
      ch = self.s.recv(65536)
      if not ch: raise EngineError("engine closed connection")
      self.buf += ch
    line, self.buf = self.buf.split(b"\n", 1)
    return json.loads(line)
  def close(self):
    try: self.s.close()
    except Exception: pass

def eng_status(timeout=2.0):
  c = EngConn(timeout)
  try:
    c.send({"id": 1, "method": "status"})
    while True:
      r = c.recv()
      if r.get("id") == 1 and "event" not in r:
        if not r.get("ok"): raise EngineError(r.get("error", "engine error"))
        return r["result"]
  finally: c.close()

# ================= FIFO queue (1 active + 4 waiting) ==========================
# NOTE: no asyncio primitives at import time (py3.9 binds them to a dead loop —
# the "Future attached to a different loop" 500s). Futures are created per
# request inside the running loop; the deque is plain state.
# R6 P3: the admission semaphore generalizes to PERMITS (= engine batch_b, read
# per-request from engine status; 1 = the legacy single active request). The
# V-01 release-after-consumption ordering is preserved exactly.
class QueueFull(Exception): pass
import collections
QSTATE = {"active": 0, "wait": collections.deque(), "holder": None, "permits": 1}

def queue_depth():
  return QSTATE["active"] + len(QSTATE["wait"])

def q_promote():
  while QSTATE["wait"] and QSTATE["active"] < QSTATE["permits"]:
    fut = QSTATE["wait"].popleft()
    if not fut.done():
      QSTATE["active"] += 1; fut.set_result(True); return

async def q_acquire(permits=1):
  loop = asyncio.get_running_loop()
  QSTATE["permits"] = max(1, int(permits))
  if QSTATE["active"] < QSTATE["permits"] and not QSTATE["wait"]:
    QSTATE["active"] += 1; QSTATE["holder"] = loop
    return
  if len(QSTATE["wait"]) >= MAX_WAITING:
    raise QueueFull()
  fut = loop.create_future()
  QSTATE["wait"].append(fut)
  try:
    await fut                       # holder flag set by promoter
    QSTATE["holder"] = loop
  except asyncio.CancelledError:
    if not fut.done(): fut.cancel()  # dead fut skipped by promoter
    raise

def q_release():
  QSTATE["active"] = max(0, QSTATE["active"] - 1); QSTATE["holder"] = None
  q_promote()

# ---- V-03: per-conversation locks (created lazily INSIDE the running loop) ---
CONV_LOCKS = {}
def _conv_lock(cid):
  key = cid or ""
  lk = CONV_LOCKS.get(key)
  if lk is None:
    lk = CONV_LOCKS.setdefault(key, asyncio.Lock())
  if len(CONV_LOCKS) > 256:   # registry hygiene: drop uncontended locks
    for k in [k for k, v in CONV_LOCKS.items() if not v.locked() and k != key]:
      CONV_LOCKS.pop(k, None)
  return lk

# ============ resident-conversation memory (API side; R6: per-conversation) ===
# One mirror per conversation_id — two conversations run CONCURRENTLY on the
# batch engine; same-conversation requests stay serialized by the conv lock.
# The global RESIDENT dict remains as the conv_id=None entry (anonymous
# single-shot requests), so legacy behavior is unchanged.
RESIDENTS = {}
def _resident(cid):
  key = cid or ""
  r = RESIDENTS.get(key)
  if r is None:
    r = RESIDENTS.setdefault(key, {"conversation_id": cid, "fed": [], "messages": None, "reusable": False})
  if len(RESIDENTS) > 64:   # registry hygiene
    for k in [k for k, v in RESIDENTS.items() if not v.get("reusable") and k != key]:
      RESIDENTS.pop(k, None)
  return r
def _resident_reset(cid):
  R = _resident(cid)
  R.update({"conversation_id": None if not cid else cid, "fed": [], "messages": None, "reusable": False})

def _engine_stream_for(eng_st, conv_id):
  """The engine-side stream record for THIS conversation (batch daemon exposes
  per-slot streams; a legacy daemon falls back to the top-level fields)."""
  if conv_id is None: return None
  streams = eng_st.get("streams")
  if isinstance(streams, list):
    for st in streams:
      if st.get("conversation_id") == conv_id:
        return st
    return None
  if eng_st.get("conversation_id") == conv_id:
    return {"fed_len": eng_st.get("fed_len"), "dirty": eng_st.get("dirty"), "pos": eng_st.get("pos")}
  return None

def log(*a): print("[api]", *a, flush=True)

# ================= incremental detok: UTF-8 holdback + stop holdback ==========
class DetokStream:
  """Byte-exact streaming detokenization. Design (SERVING_PLAN): accumulate
  _tok2bytes per token, decode the longest valid UTF-8 prefix (hold back a
  partial multibyte char), and hold back max(stop_len)-1 CHARS so a stop
  sequence can never be emitted to the client before it is detected."""
  def __init__(self, stop_strings):
    self.pending = bytearray()
    self.full = ""          # all decoded text
    self.emitted = 0        # chars of self.full already emitted
    self.stops = [s for s in stop_strings if s] or []
    self.maxstop = max((len(s) for s in self.stops), default=0)
    self.stop_found = None  # (stop_string, index)
  def _utf8_prefix(self):
    """Decode the LONGEST valid UTF-8 prefix (hold back only a partial
    multibyte char).  W1 fix: the vendored form tried the SHORTEST prefix
    first (k=3..0), permanently lagging ASCII output by up to 3 bytes and
    hiding stream tails from downstream stages (masked in production only
    by final_text's flush).  Longest-first keeps concatenated output
    byte-identical while making per-token emission immediate — required for
    the strict max_tokens cap (V-13) and the think splitter (V-07)."""
    for k in range(0, min(3, len(self.pending)) + 1):
      try:
        s = self.pending[:len(self.pending)-k].decode("utf-8")
      except UnicodeDecodeError:
        continue
      del self.pending[:len(self.pending)-k]
      return s
    return ""
  def feed_token(self, tid) -> str:
    self.pending += TOK._tok2bytes.get(int(tid), b"")
    self.full += self._utf8_prefix()
    return self._safe_emit()
  def _search_stop(self):
    if self.stop_found is not None or not self.stops: return
    for s in self.stops:
      i = self.full.find(s, max(0, self.emitted - self.maxstop))
      if i >= 0: self.stop_found = (s, i); return
  def _safe_emit(self) -> str:
    self._search_stop()
    if self.stop_found is not None:
      _, i = self.stop_found                     # emit up to the stop, then freeze
      out = self.full[self.emitted:i] if self.emitted < i else ""
      self.emitted = max(self.emitted, i)
      return out
    safe_end = max(self.emitted, len(self.full) - (self.maxstop - 1) if self.maxstop else len(self.full))
    out = self.full[self.emitted:safe_end]; self.emitted = safe_end
    return out
  def final_text(self) -> str:
    """End of stream: flush UTF-8 remainder (replace if dangling) + apply stop.
    V-06 FIX: a stopped stream returns only the NOT-YET-EMITTED prefix
    (full[emitted:i]) — never re-emits text the client already received."""
    if self.stop_found is None:
      self.full += self.pending.decode("utf-8", "replace"); self.pending = bytearray()
      self._search_stop()
    if self.stop_found is not None:
      _, i = self.stop_found
      out = self.full[self.emitted:i]
      self.emitted = max(self.emitted, i)
      return out
    out = self.full[self.emitted:]; self.emitted = len(self.full)
    return out

# ============ V-07: think/reasoning splitter ==================================
class ThinkSplitter:
  """Splits a completion stream into (reasoning, content).

  Handles the real Qwen3.8 template shapes:
   - render pre-opens an unclosed <think> (thinking on) -> think_open=True;
   - render pre-closes the block (enable_thinking=false) -> all content;
   - template-less models emitting their own LEADING <think> ... </think>
     block (detected in-stream).
  Guarantees: reasoning text never reaches content; the closing tag and the
  single blank-line separator after it are never emitted on either channel.
  NOTE: client stop-STRINGS are matched by DetokStream below this stage, i.e.
  against the raw stream (reasoning included) — documented behavior."""
  OPEN, CLOSE, SEP = "<think>", "</think>", "\n\n"

  def __init__(self, think_open=False):
    self.buf = ""
    self.in_think = bool(think_open)
    self.detected = bool(think_open)   # leading-block detection done
    self.sep_pending = False           # consume one SEP right after a close

  def feed(self, text):
    """Returns (reasoning_piece, content_piece) — exactly-once, ordered."""
    r_out, c_out = [], []
    self.buf += text
    while True:
      if not self.detected:
        if len(self.buf) < len(self.OPEN):
          if self.buf == "" or self.OPEN.startswith(self.buf):
            break                                    # hold partial marker
          self.detected = True; continue             # not a think block
        if self.buf.startswith(self.OPEN):
          self.detected = True; self.in_think = True
          self.buf = self.buf[len(self.OPEN):]; continue
        self.detected = True; continue               # content from the start
      elif self.in_think:
        i = self.buf.find(self.CLOSE)
        if i < 0:
          keep = 0                                   # hold a partial CLOSE suffix
          for k in range(min(len(self.CLOSE) - 1, len(self.buf)), 0, -1):
            if self.CLOSE.startswith(self.buf[-k:]): keep = k; break
          emit = len(self.buf) - keep
          if emit > 0:
            r_out.append(self.buf[:emit]); self.buf = self.buf[emit:]
          break
        r_out.append(self.buf[:i])
        self.buf = self.buf[i + len(self.CLOSE):]
        self.in_think = False; self.detected = True; self.sep_pending = True
        continue
      else:
        if self.sep_pending:
          if self.buf.startswith(self.SEP):
            self.buf = self.buf[len(self.SEP):]; self.sep_pending = False; continue
          if len(self.buf) < len(self.SEP) and self.SEP.startswith(self.buf):
            break                                    # hold partial separator
          self.sep_pending = False                   # no separator present
          continue
        if self.buf:
          c_out.append(self.buf); self.buf = ""
        break
    return "".join(r_out), "".join(c_out)

  def final(self):
    """Flush at end of stream. An unclosed think block stays reasoning."""
    r, c = "", self.buf
    if self.in_think:
      r, c = self.buf, ""
    elif not self.detected:
      r, c = "", self.buf      # short undecidable tail -> visible content
    if self.sep_pending:
      if c.startswith(self.SEP): c = c[len(self.SEP):]
      elif self.SEP.startswith(c): c = ""
    self.buf = ""
    return r, c

def _think_open(rtext):
  """True when the render leaves the generation INSIDE an open think block."""
  return rtext.rfind("<think>") > rtext.rfind("</think>")

# ================= FastAPI app =================================================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="qwen3.8-27b eGPU (M1-B)", version="1.1")

# ---- V-34: request hardening (order: TrustedHost outermost, then body cap) ----
# CORS posture is DEFAULT-DENY: no CORS middleware is installed, so browsers
# get no cross-origin read grants; the localhost TrustedHost allowlist also
# kills DNS-rebind reads of /health + SSE.
class _BodyTooLarge(Exception): pass

class BodyCapMiddleware:
  """Rejects oversized bodies BEFORE parsing: content-length pre-check plus a
  streaming byte counter (chunked/lying-length clients). The O(n^2) BPE encode
  of a multi-MB body can never start."""
  def __init__(self, app_, max_bytes):
    self.app = app_
    self.max_bytes = max_bytes
  async def __call__(self, scope, receive, send):
    if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
      return await self.app(scope, receive, send)
    cl = None
    for k, v in scope.get("headers", []):
      if k == b"content-length":
        try: cl = int(v)
        except Exception: cl = None
    if cl is not None and cl > self.max_bytes:
      return await self._reject(send)
    started = {"v": False}
    async def _send(msg):
      if msg["type"] == "http.response.start": started["v"] = True
      await send(msg)
    counted = {"n": 0}
    async def capped_receive():
      msg = await receive()
      if msg.get("type") == "http.request":
        counted["n"] += len(msg.get("body") or b"")
        if counted["n"] > self.max_bytes:
          raise _BodyTooLarge()
      return msg
    try:
      await self.app(scope, capped_receive, _send)
    except _BodyTooLarge:
      if not started["v"]:
        await self._reject(send)
  async def _reject(self, send):
    body = json.dumps({"error": {"message": f"request body exceeds {self.max_bytes} bytes",
                                 "type": "request_too_large"}}).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})

app.add_middleware(BodyCapMiddleware, max_bytes=MAX_BODY_BYTES)
from starlette.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

def oai_error(status, message, err_type="invalid_request_error", param=None, code=None):
  return JSONResponse(status_code=status, content={
    "error": {"message": message, "type": err_type, "param": param, "code": code}})

def _safe_engine_fields(st):
  """V-33: the unauthenticated /health subset — no fed_tail, no
  conversation_id, no cur (conversation content/identity stays behind the
  admin token)."""
  return {k: st.get(k) for k in ("ready", "busy", "rpc", "pos", "ctxk", "mode",
                                 "queue", "dirty", "uptime_s", "config_fp",
                                 "lookup_k", "pf_prefill", "cycles_since_rebuild", "batch_b")}

def _admin_ok(request):
  if not ADMIN_TOKEN: return False
  tok = request.headers.get("x-admin-token") or request.query_params.get("admin_token") or ""
  return hmac.compare_digest(str(tok), ADMIN_TOKEN)

def _drift_state(st):
  """None when the daemon config matches ops/env.canonical (or the check is
  not applicable); (daemon_fp, expected_fp) on drift. None cases: canonical
  file absent (manual/dev boot — check disabled, loudly reported) or the
  daemon predates W2 (no config_fp in status)."""
  if EXPECTED_FP is None: return None
  fp = st.get("config_fp")
  if fp is None or fp == EXPECTED_FP: return None
  return (fp, EXPECTED_FP)

@app.get("/health")
async def health(request: Request):
  full = _admin_ok(request)   # ops detail (fed_tail/conversation_id) only behind the token
  try:
    st = await run_in_threadpool(eng_status, 2.0)
  except Exception as e:
    degraded = any(os.path.exists(p) for p in ENGINE_STAYDOWN_PATHS)
    return JSONResponse(status_code=503, headers={"Retry-After": "5"}, content={
      "status": "engine_degraded" if degraded else "engine_down", "detail": str(e)[:200],
      "queue_depth": queue_depth()})
  drift = _drift_state(st)
  if drift is not None:
    return JSONResponse(status_code=503, headers={"Retry-After": "30"}, content={
      "status": "config_drift",
      "detail": f"daemon config_fp {drift[0]} != canonical {drift[1]} — the engine "
                f"booted without ops/env.canonical (slow-path/LOOKUP-off class). "
                f"Restart the engine through the wrapper.",
      "config": {"config_fp": drift[0], "config_fp_expected": drift[1]},
      "queue_depth": queue_depth()})
  if not st.get("ready"):
    return JSONResponse(status_code=503, headers={"Retry-After": "5"}, content={
      "status": "warming", "engine": _safe_engine_fields(st), "queue_depth": queue_depth()})
  out = {"status": "ok", "engine": _safe_engine_fields(st), "queue_depth": queue_depth(),
         "config": {"config_fp": st.get("config_fp"),
                    "config_fp_expected": EXPECTED_FP,
                    "drift_check": "on" if EXPECTED_FP is not None else "DISABLED (no env.canonical)"}}
  if full:
    out["engine_debug"] = st   # full daemon status incl. fed_tail/conversation_id
  return out

@app.get("/v1/models")
async def models():
  return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "egpu-rig"}]}

def _validate_fields(body):
  """Returns (error_response|None, ctx|None). Shared by stream+non-stream."""
  msgs = body.get("messages")
  if not isinstance(msgs, list) or not msgs:
    return oai_error(400, "'messages' must be a non-empty list"), None
  for m in msgs:
    if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "developer"):
      return oai_error(400, f"invalid message (roles: system/user/assistant): {str(m)[:120]}"), None
    c = m.get("content")
    if isinstance(c, list):
      for part in c:
        # V-09: a text part MUST carry a string 'text' ({"type":"text"} alone -> 400)
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
          return oai_error(400, "multimodal content is not supported (text-only in M1); parts must be "
                                '{"type":"text","text":"..."} with a string text field'), None
    elif not isinstance(c, str) and c is not None:
      return oai_error(400, f"unsupported content type {type(c).__name__} (text only)"), None
  for k in ("temperature", "top_p", "top_k"):
    v = body.get(k)
    if v is None: continue
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) in (0.0, 1.0)):
      return oai_error(400, f"{k}={v}: sampling is not yet supported (M2). Only omitted/0/1 accepted "
                            "(greedy decode, bit-exact)."), None
  if body.get("tools") is not None or body.get("tool_choice") is not None:
    return oai_error(400, "tools/function-calling are not supported in M1 (no tool runtimes: "
                          "Cline/Cursor Agent/Claude Code will not work against this endpoint)"), None
  if body.get("logprobs") or body.get("top_logprobs") is not None:
    return oai_error(400, "logprobs are not supported in M1"), None
  if body.get("n") not in (None, 1):
    return oai_error(400, "n>1 is not supported in M1 (single completion)"), None
  if body.get("response_format") is not None:
    return oai_error(400, "response_format is not supported in M1 (plain text only)"), None
  stop = body.get("stop")
  if stop is not None:
    if isinstance(stop, str): stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) and s for s in stop) or len(stop) > 4:
      return oai_error(400, "'stop' must be a string or a list of up to 4 non-empty strings"), None
  # V-08: max_completion_tokens alias (reject when both set and disagreeing)
  mt, mct = body.get("max_tokens"), body.get("max_completion_tokens")
  if mt is not None and mct is not None and int(mt) != int(mct):
    return oai_error(400, "'max_tokens' and 'max_completion_tokens' are both set and differ; "
                          "send at most one of them"), None
  max_tokens = mt if mt is not None else mct
  if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
    return oai_error(400, "'max_tokens'/'max_completion_tokens' must be a positive integer"), None
  # V-07: reasoning options forwarded to the template (default effort medium,
  # NOT the template's own xhigh default)
  eff = body.get("reasoning_effort")
  if eff is not None and eff not in ("xhigh", "medium", "low"):
    return oai_error(400, "'reasoning_effort' must be one of xhigh/medium/low "
                          "(default medium at this endpoint)"), None
  eth = body.get("enable_thinking")
  if eth is not None and not isinstance(eth, bool):
    return oai_error(400, "'enable_thinking' must be a boolean"), None
  ctx = {
    "stop": stop, "max_tokens": max_tokens, "stream": bool(body.get("stream", False)),
    "include_usage": bool(((body.get("stream_options") or {}).get("include_usage"))
                          if isinstance(body.get("stream_options"), dict) else False),
    "conversation_id": body.get("conversation_id"),
    "prompt_cache_key": None, "prompt_cache_ttl": None,
    "template_vars": {"reasoning_effort": eff or "medium",
                      "enable_thinking": (eth if eth is not None else True)},
    "ignored": [k for k in ("seed", "presence_penalty", "frequency_penalty", "penalty_scores",
                            "user", "parallel_tool_calls") if k in body],
  }
  pck = body.get("prompt_cache_key")
  if pck is not None and (not isinstance(pck, str) or not pck or len(pck) > 128):
    return oai_error(400, "'prompt_cache_key' must be a non-empty string (<=128 chars)"), None
  pcttl = body.get("prompt_cache_ttl")
  if pcttl is not None and (not isinstance(pcttl, int) or isinstance(pcttl, bool) or pcttl < 60):
    return oai_error(400, "'prompt_cache_ttl' must be an integer >= 60 seconds"), None
  ctx["prompt_cache_key"] = pck
  ctx["prompt_cache_ttl"] = pcttl
  return None, ctx

def _normalize_content(m):
  c = m.get("content")
  if c is None: return ""                    # V-09: content:null -> ""
  if isinstance(c, str): return c
  return "".join(p["text"] for p in c if p.get("type") == "text")

def _template_render(msgs, add_generation_prompt, tvars=None):
  """Single render entry (V-07): identical vars for request + history renders."""
  norm = [{"role": m["role"], "content": _normalize_content(m)} for m in msgs]
  try:
    return TEMPLATE.render(messages=norm, add_generation_prompt=add_generation_prompt,
                           **(tvars or {}))
  except TypeError:
    # FallbackTemplate (no jinja) does not take the options
    return TEMPLATE.render(messages=norm, add_generation_prompt=add_generation_prompt)

def _render_text(msgs, tvars=None):
  return _template_render(msgs, True, tvars)

def _render_and_encode(msgs):
  return TOK.encode(_render_text(msgs))

def _decide_prefix(rtext, engine_st, conv_id, tvars=None, phase="decide"):
  """(ids2, mode, cur, delta). M1-C RE-ENCODE LAW: the model's own token splits
  are NOT BPE-canonical (e.g. header "\\n" + " need" re-encodes to one merged
  token), so re-encoding a full conversation render can never reproduce the
  ids the engine was fed. Correct construction: TEXT-prefix check against the
  recorded render, then ids2 = recorded_fed + encode(render tail) — the tail
  starts at a special-token boundary, so its standalone encode is exact.
  FOLLOW_UP additionally requires the engine fed == mirror (no stop-batch
  over-commit, no unread cancel-latency cycle) and no client-side stop
  truncation (STOP-BATCH OVER-COMMIT LAW -> deterministic FRESH fallback).
  W1: a dirty engine (faulted last session, V-10) also forces FRESH.
  R6 P3: the mirror is PER-CONVERSATION (RESIDENTS); the engine side is matched
  via the per-slot `streams` list (a slot eviction / second-conversation
  displacement forces FRESH naturally)."""
  R = _resident(conv_id)
  est = _engine_stream_for(engine_st, conv_id)
  fed = R["fed"]
  msgs = R.get("messages")
  rend = ""
  if msgs:
    try: rend = _template_render(msgs, False, tvars)   # history only
    except Exception: rend = ""
  _pok = bool(rend) and rtext.startswith(rend)
  _div = ""
  if rend and not _pok:
    i = next((k for k in range(min(len(rend), len(rtext))) if rend[k] != rtext[k]), min(len(rend), len(rtext)))
    _div = f" div@{i} rend={rend[max(0,i-30):i+15]!r} rtext={rtext[max(0,i-30):i+15]!r}"
  log(f"prefix decision [{phase}] conv={conv_id}: reusable={R.get('reusable')} "
      f"eng_stream={'y' if est else 'n'} dirty={(est or {}).get('dirty')} "
      f"fed={len(fed)}/{(est or {}).get('fed_len')} "
      f"rend={len(rend)} rtext={len(rtext)} prefix_ok={_pok}{_div}")
  if (conv_id is not None and R["conversation_id"] == conv_id
      and est is not None
      and not est.get("dirty")
      and R.get("reusable") and fed and rend
      and rtext.startswith(rend) and len(rtext) > len(rend)
      and est.get("fed_len") == len(fed)):
    tail_ids = TOK.encode(rtext[len(rend):])
    ids2 = fed + tail_ids
    return ids2, "FOLLOW_UP", ids2[len(fed)], ids2[len(fed)+1:]
  ids2 = TOK.encode(rtext)
  return ids2, "FRESH", None, ids2

def _stop_ids_for(stops):
  ids = {t for t in (TOK.eot_id, TOK.eos_id) if t is not None}   # V-14: id 0 kept
  specials = getattr(TOK, "_special_tokens", {})
  for s in stops or []:
    if s in specials: ids.add(specials[s])
  nvocab = len(TOK._tok2bytes)
  out = sorted(t for t in ids if isinstance(t, int) and 0 <= t < nvocab)
  if len(out) != len(ids):
    log(f"stop ids dropped (outside vocab range 0..{nvocab-1}): {sorted(ids)}")
  return out

def _sse(obj): return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

def _run_engine_sync(mode, cur, delta, ids2, rtext, messages, conv_id, max_tokens, stop_token_ids,
                     stops, on_event=None, cancel_check=None, cache_key=None, think_open=False,
                     cache_ttl=None):
  """Blocking engine session for one request (runs in executor thread). Owns the
  DetokStream/ThinkSplitter so stop-STRING matches also cancel the engine
  promptly.  on_event(kind, payload) fires from this thread; kinds: "prefill",
  "text" (client-visible content), "reasoning". Mutates the PER-CONVERSATION
  resident mirror R (fed mirror, render-text mirror, reusable flag). Returns
  {tokens, visible, text_tail, reasoning, finish, cancelled, mode, ...}."""
  result = {"tokens": [], "visible": 0, "text_tail": "", "reasoning": "",
            "finish": "stop", "cancelled": False, "mode": mode,
            "prefix_mode": mode, "cached_tokens": 0}
  ds = DetokStream(stops)
  split = ThinkSplitter(think_open=think_open)
  mirror_parts = []           # client-visible reply content (render mirror)
  reasoning_parts = []
  R = _resident(conv_id)                 # R6 P3: per-conversation mirror
  R["reusable"] = False                  # V-12: set True only after prefill+state confirmed
  c = EngConn(600.0)
  def _feed_token(t):
    piece = ds.feed_token(t)
    if not piece: return None
    rseg, cseg = split.feed(piece)
    if rseg:
      reasoning_parts.append(rseg)
      if on_event:
        try: on_event("reasoning", rseg)
        except Exception: pass
    if cseg:
      mirror_parts.append(cseg)
      result["visible"] += 1
      if on_event:
        try: on_event("text", cseg)
        except Exception: pass
    return cseg
  try:
    if mode == "FRESH":
      # R1 AUTO_CACHE: the daemon walks the hash-chain trie; mode comes back
      # FRESH (+ingest) or CACHE_HIT (restore + M64 tail). cached_tokens =
      # restored prefix length (OpenAI prompt_tokens_details semantics).
      _p = {"mode": "AUTO_CACHE", "ids": ids2, "conversation_id": conv_id}
      if cache_key: _p["cache_key"] = cache_key
      if cache_ttl: _p["cache_ttl"] = int(cache_ttl)   # W3 V-44: honored through the RPC
      c.send({"id": 10, "method": "prefill", "params": _p})
    else:
      c.send({"id": 10, "method": "prefill",
              "params": {"mode": "FOLLOW_UP", "ids": delta, "cur": cur, "conversation_id": conv_id}})
    try:
      while True:
        r = c.recv()
        if r.get("id") != 10 or "event" in r:
          if r.get("event") == "prefill_progress" and on_event:
            on_event("prefill", r)
          continue
        if not r.get("ok"): raise EngineError(r.get("error", "prefill failed"))
        _pr = r.get("result") or {}
        if _pr.get("mode"): result["prefix_mode"] = _pr["mode"]
        if _pr.get("cached_tokens"): result["cached_tokens"] = int(_pr["cached_tokens"])
        break
    except Exception:
      # V-12: prefill failure must not leave a phantom/foreign resident
      # conversation behind — reset the mirror entirely.
      R.update({"conversation_id": conv_id, "fed": [], "messages": None, "reusable": False})
      raise
    R["conversation_id"] = conv_id
    R["fed"] = list(ids2)            # engine fed == ids2 exactly (cur override)
    R["messages"] = list(messages)   # M1-C: message-history mirror (rendered on
                                            # demand; raw reply text is NEVER a render
                                            # prefix — the template re-inserts an empty
                                            # <think> block for historical assistant msgs)
    R["reusable"] = True             # V-12: prefill confirmed + state mirrored —
                                            # the turn is reusable unless a bail marks it
    # M1-C BOUNDED-CYCLE LAW: never ask the engine for unbounded generation.
    # Each cycle emits >=1 token, so max_tokens + 4 cycles always suffices for
    # any client-visible cap; the old 100000 let a missed cancel (disconnect not
    # seen) turn into a ~1.5h runaway generate that faulted the dext.
    c.send({"id": 11, "method": "generate",
            "params": {"max_cycles": int(max_tokens) + 4, "stop_token_ids": stop_token_ids}})
    stopset = set(stop_token_ids)
    def terminal_drain():
      """M1-C: after a client-side bail, read the engine's terminal event and
      mirror any tokens the client never saw into R["fed"] (engine
      truth).  W1/V-13: those tokens are NEVER client-visible (strict cap) —
      the hidden tail keeps the R7a exact-prefix contract (the next turn
      extends the ENGINE-fed stream, which contains the stop/overshoot), so
      the turn stays reusable for FOLLOW_UP."""
      try:
        c.s.settimeout(10)
        while True:
          r = c.recv()
          if r.get("id") == 11 and r.get("event") in ("done", "cancelled"):
            etoks = r.get("tokens") or []
            seen = len(R["fed"]) - len(ids2)
            if len(etoks) > seen:
              R["fed"].extend(etoks[seen:])
            return
          if r.get("id") == 11 and not r.get("ok", True):
            # fault-during-drain: the daemon replied an error, not a terminal
            raise EngineError(r.get("error", "generate failed"))
      except EngineError:
        raise
      except Exception:
        return
    def bail(finish):
      result["finish"] = finish
      c.send({"id": 99, "method": "cancel", "params": {}})   # engine self-stops too
      terminal_drain()
      return result
    try:
      while True:
        r = c.recv()
        ev = r.get("event")
        if r.get("id") == 11 and not r.get("ok", True):
          # V-10: the daemon replies {"ok":false} when generate faults — the
          # old loop ignored it and hung until the 600s socket timeout.
          raise EngineError(r.get("error", "generate failed"))
        if ev == "cycle":
          toks = r["tokens"]
          R["fed"].extend(toks)               # V-21: extend, no rebuild
          hit = next((k for k, t in enumerate(toks) if t in stopset), None)
          emit = toks if hit is None else toks[:hit]
          overflow = None
          for j, t in enumerate(emit):
            # V-13/V-07: the cap counts VISIBLE content tokens; reasoning
            # (inside the think block) does not burn the client budget.
            if not split.in_think and result["visible"] >= max_tokens:
              overflow = emit[j:]
              break
            result["tokens"].append(t)
            _feed_token(t)
          if overflow is not None:
            R["fed"].extend(overflow)   # engine-fed truth, never visible
            return bail("length")              # R7a contract: hidden tail keeps reuse
          if hit is not None:
            # STOP-BATCH OVER-COMMIT LAW: tokens past the stop were fed but the
            # client will never render them -> next turn must fall back FRESH.
            # R7a length-window refinement: a stop BEYOND the visible cap is
            # invisible to the client -> the length path (strict cap; the
            # overshoot tail mirrors into fed and the turn stays reusable).
            if result["visible"] < max_tokens:
              R["reusable"] = False
              return bail("stop")
            return bail("length")
          if ds.stop_found is not None:
            R["reusable"] = False
            return bail("stop")                # stop-STRING matched (same class)
          if not split.in_think and result["visible"] >= max_tokens:
            return bail("length")              # strict cap; hidden drain tail reuses
          if cancel_check and cancel_check():
            result["cancelled"] = True
            R["reusable"] = False       # client abandoned mid-stream
            return bail("stop")
        elif ev == "done":
          # done without the stop flag = max_cycles exhausted (budget cap)
          result["finish"] = "length" if not r.get("stop") else "stop"
          etoks = r.get("tokens") or []
          seen = len(R["fed"]) - len(ids2)
          if len(etoks) > seen:                # defensive: mirror any unseen tail
            R["fed"].extend(etoks[seen:])
          return result
        elif ev == "cancelled":
          result["cancelled"] = True
          R["reusable"] = False
          return result
    except EngineError:
      # V-10: faulted session — known-good tokens are already mirrored in fed;
      # the conversation is dirty (engine side reports it in status) and the
      # resident mirror must never be reused.
      R["reusable"] = False
      R["messages"] = None
      raise
    except Exception as e:
      R["reusable"] = False
      R["messages"] = None
      raise EngineError(f"generate failed after {len(result['tokens'])} tokens: {e!r}")
  finally:
    # V-07/V-06: flush the detok remainder THROUGH the splitter (the UTF-8
    # and stop holdbacks can still hold chars when the stream ends).
    rtail, ctail = "", ""
    try:
      flush = ds.final_text()
      if flush:
        rtail, ctail = split.feed(flush)
      r2, c2 = split.final()
      rtail += r2; ctail += c2
    except Exception:
      pass
    result["reasoning"] = "".join(reasoning_parts) + rtail
    result["text_tail"] = ctail
    try:
      result["reasoning_tokens"] = len(TOK.encode(result["reasoning"])) if result["reasoning"] else 0
    except Exception:
      result["reasoning_tokens"] = 0
    if R.get("reusable"):
      R["messages"] = list(messages) + [
        {"role": "assistant", "content": "".join(mirror_parts) + ctail}]
    else:
      R["messages"] = None
    c.close()

async def _chat(request: Request):
  # ---- V-34: content-type gate (a text/plain simple-POST must never mutate) --
  ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
  if ctype != "application/json":
    return oai_error(415, "content-type must be application/json")
  # ---- V-04: parse + validate BEFORE the FIFO slot is acquired ---------------
  # (a slow-loris client trickling its body must never pin an engine slot)
  try:
    body = await asyncio.wait_for(request.json(), BODY_READ_TIMEOUT)
  except asyncio.TimeoutError:
    return oai_error(400, f"request body not received within {BODY_READ_TIMEOUT:.0f}s")
  except asyncio.CancelledError:
    raise
  except _BodyTooLarge:
    return oai_error(413, f"request body exceeds {MAX_BODY_BYTES} bytes",
                     err_type="request_too_large")
  except Exception:
    return oai_error(400, "invalid JSON body")
  conv_id = body.get("conversation_id") or request.headers.get("x-conversation-id")
  err, ctx = _validate_fields(body)
  if err: return err

  try:
    eng_st = await run_in_threadpool(eng_status, 5.0)
  except Exception as e:
    return oai_error(503, f"engine unavailable: {str(e)[:150]}", err_type="engine_error")
  # ---- V-28: config-drift refusal. A daemon booted outside env.canonical
  # (e.g. launchd's stale plist env -> LOOKUP_K=0 + T=1 slow FRESH path) must
  # fail LOUDLY, not serve degraded while /health says ok.
  drift = _drift_state(eng_st)
  if drift is not None:
    return oai_error(503, f"engine config drift: daemon config_fp {drift[0]} != canonical "
                          f"{drift[1]} — restart the engine through ops/engine_daemon.sh "
                          f"(sources ops/env.canonical)", err_type="config_drift")

  tvars = ctx["template_vars"]
  rtext = await run_in_threadpool(_render_text, body["messages"], tvars)
  _decide_prefix(rtext, eng_st, conv_id, tvars, phase="pre-slot")   # validation pass only

  # ---- V-01/V-04: admission AFTER validation; waiter deadline (V-16) --------
  # R6 P3: admission PERMITS = the engine's batch width (batch_b; 1 on a
  # legacy/single-stream daemon) — two DIFFERENT conversations run
  # concurrently; same-conversation requests still serialize on the conv lock.
  permits = max(1, int(eng_st.get("batch_b") or 1))
  try:
    await asyncio.wait_for(q_acquire(permits), QUEUE_WAIT_S)
  except asyncio.TimeoutError:
    resp = oai_error(503, f"engine queue wait exceeded {QUEUE_WAIT_S:.0f}s", err_type="engine_busy")
    resp.headers["Retry-After"] = "10"
    return resp
  except QueueFull:
    raise
  request.state.slot_held = True

  # ---- V-03: per-conversation lock; decide INSIDE the lock (TOCTOU guard) ----
  lk = _conv_lock(conv_id)
  await lk.acquire()
  try:
    try:
      eng_st = await run_in_threadpool(eng_status, 5.0)
    except Exception as e:
      return oai_error(503, f"engine unavailable: {str(e)[:150]}", err_type="engine_error")
    ids2, mode, cur, delta = _decide_prefix(rtext, eng_st, conv_id, tvars, phase="post-slot")
    # V-02: mode-aware headroom. A FRESH/AUTO_CACHE prefill RESETS the engine
    # pos to the new prompt — the stale parked pos must not clamp it (the old
    # max(pos, len) form 400-locked every FRESH request after a 100k turn).
    # R6 P3: FOLLOW_UP headroom reads THIS conversation's stream pos (the
    # top-level pos is ambiguous with 2 resident streams).
    est = _engine_stream_for(eng_st, conv_id)
    fu_pos = (est or {}).get("pos", eng_st.get("pos", 0))
    base_pos = max(fu_pos, len(ids2)) if mode == "FOLLOW_UP" else len(ids2)
    headroom = eng_st.get("ctxk", 100352) - base_pos - HEADROOM
    max_tokens = ctx["max_tokens"] or 512
    if headroom < 1:
      return oai_error(400, f"context window exhausted: prompt {len(ids2)} tokens, mode {mode}, "
                            f"engine pos {eng_st.get('pos')}, ctx {eng_st.get('ctxk')}")
    if max_tokens > headroom:
      log(f"clamping max_tokens {max_tokens} -> {headroom}")
      max_tokens = headroom

    model_echo = body.get("model", MODEL_ID)
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    stops = ctx["stop"] or []
    stop_token_ids = _stop_ids_for(stops)
    think_open = _think_open(rtext)
    log(f"request {rid}: mode={mode} prompt={len(ids2)} fed={eng_st.get('fed_len')} "
        f"delta={len(delta)} max_tokens={max_tokens} stops={stops} stop_ids={stop_token_ids} "
        f"conv={conv_id} think_open={think_open}")

    if not ctx["stream"]:
      pieces = []
      def on_event(kind, p):
        if kind == "text": pieces.append(p)
      res = await run_in_threadpool(_run_engine_sync, mode, cur, delta, ids2, rtext, body["messages"],
                                    conv_id, max_tokens, stop_token_ids, stops, on_event, None,
                                    ctx.get("prompt_cache_key"), think_open, ctx.get("prompt_cache_ttl"))
      text = "".join(pieces) + res["text_tail"]
      msg = {"role": "assistant", "content": text}
      if res.get("reasoning"): msg["reasoning_content"] = res["reasoning"]
      out = {"id": rid, "object": "chat.completion", "created": created, "model": model_echo,
             "choices": [{"index": 0, "message": msg, "finish_reason": res["finish"]}],
             "usage": {"prompt_tokens": len(ids2), "completion_tokens": len(res["tokens"]),
                       "total_tokens": len(ids2) + len(res["tokens"]),
                       "prompt_tokens_details": {"cached_tokens": res.get("cached_tokens", 0)},
                       "completion_tokens_details": {"reasoning_tokens": res.get("reasoning_tokens", 0)}},
             "prefix_mode": res["prefix_mode"], "cached_tokens": res.get("cached_tokens", 0),
             "conversation_id": conv_id}
      if ctx["ignored"]: out["ignored_params"] = ctx["ignored"]
      hdrs = {"x-prefix-mode": str(res["prefix_mode"]),
              "x-cached-tokens": str(res.get("cached_tokens", 0))}
      if conv_id: hdrs["x-conversation-id"] = conv_id
      return JSONResponse(content=out, headers=hdrs)

    # ---- stream ----
    # V-16: bounded event queue; the engine thread back-pressures on full.
    evq = _pyqueue.Queue(maxsize=EVQ_MAX)
    cancel_flag = {"v": False}
    def on_event(kind, p):
      try:
        evq.put((kind, p), timeout=5.0)
      except _pyqueue.Full:
        if kind != "text":
          return                       # drop prefill progress comments only
        log(f"{rid}: event queue full — text event dropped (slow consumer)")
    loop = asyncio.get_running_loop()

    async def _watch_disconnect():
      """M1-C: RELIABLE disconnect detection. request.is_disconnected() (the
      anyio-cancelled receive trick) proved UNRELIABLE on this stack: a client
      that closed mid-stream was not seen for 44+ s -> runaway engine generate
      -> dext fault. A task that OWNS receive() gets the http.disconnect message
      deterministically."""
      try:
        while True:
          msg = await request.receive()
          if msg.get("type") == "http.disconnect":
            cancel_flag["v"] = True
            log(f"{rid}: client disconnected -> engine cancel armed")
            return
      except Exception:
        return

    fut = asyncio.ensure_future(run_in_threadpool(
        _run_engine_sync, mode, cur, delta, ids2, rtext, body["messages"], conv_id,
        max_tokens, stop_token_ids, stops, on_event, lambda: cancel_flag["v"],
        ctx.get("prompt_cache_key"), think_open, ctx.get("prompt_cache_ttl")))
    watcher = asyncio.ensure_future(_watch_disconnect())
    gen_done = asyncio.Event()
    released = {"v": False}
    gen_completed = {"v": False}
    def release_slot():
      """V-01: exactly-once FIFO release, owned by the stream (the route's
      finally runs BEFORE the generator starts — it must not release here)."""
      if released["v"]: return
      released["v"] = True
      cancel_flag["v"] = True          # arm engine cancel on any release path
      for t in (watcher,):
        try: t.cancel()
        except Exception: pass
      q_release()
    async def _slot_guard():
      """Backstop: if the SSE generator is never driven to completion (client
      vanished before first byte, ASGI mishap), the slot is still released
      once the bounded engine session ends (+ grace)."""
      try:
        await asyncio.shield(fut)
      except asyncio.CancelledError:
        return
      except Exception as e:
        log(f"engine session failed (guard {rid}): {e!r}")
      try:
        await asyncio.wait_for(gen_done.wait(), STREAM_GUARD_GRACE_S)
      except Exception:
        pass
      release_slot()   # no-op when gen()'s finally already released
    asyncio.ensure_future(_slot_guard())

    async def gen():
      def chunk(delta_obj, finish=None):
        return _sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": model_echo, "choices": [{"index": 0, "delta": delta_obj,
                                                       "finish_reason": finish}]})
      try:
        yield chunk({"role": "assistant", "content": ""})
        while True:
          while True:
            try: kind, p = evq.get_nowait()
            except _pyqueue.Empty: break
            if kind == "prefill":
              d, t = p.get("done", 0), max(p.get("total", 1), 1)
              yield f": prefill {int(100*d/t)}% ({p.get('stage','prefill')})\n\n"
            elif kind == "text":
              yield chunk({"content": p})
            elif kind == "reasoning":
              yield chunk({"reasoning_content": p})
          if fut.done():
            # M1-C: drain any text events still queued behind fut's completion
            # (the 20ms poll can lag the threadpool thread's last on_event calls;
            # dropping them truncated stream tails nondeterministically).
            while True:
              try: kind2, p2 = evq.get_nowait()
              except _pyqueue.Empty: break
              if kind2 == "text": yield chunk({"content": p2})
              elif kind2 == "reasoning": yield chunk({"reasoning_content": p2})
            break
          if cancel_flag["v"]:       # set by the watcher (or early close below)
            yield ": client disconnected — cancelling engine\n\n"
            break
          await asyncio.sleep(0.02)
        try:
          res = await fut
        except EngineError as e:
          # V-11: mid-stream engine death = explicit SSE error event + [DONE],
          # never a silently truncated connection.
          yield _sse({"error": {"message": f"engine error: {str(e)[:300]}",
                                "type": "engine_error", "param": None, "code": None}})
          yield "data: [DONE]\n\n"
          log(f"stream error {rid}: {str(e)[:200]}")
          return
        except asyncio.CancelledError:
          raise
        except Exception as e:
          yield _sse({"error": {"message": f"internal error: {repr(e)[:300]}", "type": "server_error"}})
          yield "data: [DONE]\n\n"
          return
        if res["text_tail"]: yield chunk({"content": res["text_tail"]})
        yield chunk({}, finish=res["finish"])
        if ctx["include_usage"]:
          yield _sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                      "model": model_echo, "choices": [],
                      "usage": {"prompt_tokens": len(ids2), "completion_tokens": len(res["tokens"]),
                                "total_tokens": len(ids2) + len(res["tokens"]),
                                "prompt_tokens_details": {"cached_tokens": res.get("cached_tokens", 0)},
                                "completion_tokens_details": {"reasoning_tokens": res.get("reasoning_tokens", 0)}}})
        yield "data: [DONE]\n\n"
        gen_completed["v"] = True
        log(f"done {rid}: finish={res['finish']} ntok={len(res['tokens'])} "
            f"visible={res['visible']} cancelled={res['cancelled']}")
      finally:
        # M1-C: GeneratorExit/early close (client gone, server dropping the
        # response) MUST arm the engine cancel — run_chat polls cancel_check()
        # every cycle and bails into its cancel-send. Harmless on normal exit
        # (run_chat already returned).  V-01: the FIFO slot is released HERE
        # after the terminal drain on normal completion; on an early close the
        # guard task releases it once the (now cancelling) engine session ends.
        log(f"stream closed {rid}: completed={gen_completed['v']} "
            f"cancelled={cancel_flag['v']}")
        cancel_flag["v"] = True
        try: watcher.cancel()
        except Exception: pass
        gen_done.set()
        if gen_completed["v"]:
          release_slot()
    request.state.stream_release = release_slot
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "x-conversation-id": conv_id or "",
                                      "X-Accel-Buffering": "no"})

  finally:
    lk.release()

@app.post("/v1/chat/completions")
async def chat_completions_route(request: Request):
  try:
    return await _chat(request)
  except QueueFull:
    resp = oai_error(429, f"engine queue full ({QSTATE['permits']} active + {MAX_WAITING} waiting); retry shortly", err_type="engine_busy")
    resp.headers["Retry-After"] = "10"
    return resp
  except EngineError as e:
    return oai_error(503, f"engine error: {str(e)[:200]}", err_type="engine_error")
  except Exception as e:
    import traceback; traceback.print_exc()
    return oai_error(500, f"internal error: {str(e)[:200]}", err_type="server_error")
  finally:
    # V-01: for STREAMING responses the slot belongs to the SSE generator (the
    # route's finally runs before the generator starts); release_slot() is
    # exactly-once and shared between gen()'s finally and the guard task.
    if getattr(request.state, "slot_held", False) and not getattr(request.state, "stream_release", None):
      q_release()

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host=BIND, port=PORT, log_level="info", access_log=False)
