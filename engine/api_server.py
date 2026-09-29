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
# L7.1 admission-hardening knobs (the exposed R3-10-class leak): bounded conv-
# lock acquire; the admission watchdog heals a leaked permit when the ENGINE is
# demonstrably idle while QSTATE says a holder is active with waiters parked.
CONV_LOCK_WAIT_S = float(os.getenv("TLX_CONV_LOCK_WAIT_S", "120"))
ADMIT_WATCHDOG_S = float(os.getenv("TLX_ADMIT_WATCHDOG_S", "10"))
ADMIT_HEAL_AFTER_S = float(os.getenv("TLX_ADMIT_HEAL_AFTER_S", "90"))
EVQ_MAX = int(os.getenv("TLX_EVQ_MAX", "8192"))
# R3-33: sampling compat mode — accept in-range OpenAI sampling params
# (recorded in ignored_params; decode stays greedy). Default OFF: a clean,
# capability-coded 400 instead.
COMPAT_IGNORE_SAMPLING = os.getenv("TLX_COMPAT_IGNORE_SAMPLING", "0") == "1"
# R3-38: strip role=tool / assistant tool_calls messages instead of rejecting
# (silently corrupting replayed tool-use histories was the worse failure).
COMPAT_STRIP_TOOL_MESSAGES = os.getenv("TLX_COMPAT_STRIP_TOOL_MESSAGES", "0") == "1"
# R3-39: Idempotency-Key/x-request-id - short-TTL registry; a duplicate of an
# IN-FLIGHT (key, conversation) is a 409 (gateway retry storms re-execute the
# same logical request against the same pin: double-billing + wasted GPU).
IDEMPOTENCY_TTL_S = float(os.getenv("TLX_IDEMPOTENCY_TTL_S", "120"))
_IDEMP = {}
def _idemp_claim(key, conv):
  now = time.time()
  for k in [k for k, v in _IDEMP.items() if now - v["ts"] > IDEMPOTENCY_TTL_S]:
    _IDEMP.pop(k, None)
  prev = _IDEMP.get((key, conv))
  if prev is not None and prev["state"] == "in_flight":
    return False
  _IDEMP[(key, conv)] = {"state": "in_flight", "ts": now}
  return True
def _idemp_done(key, conv):
  if key:
    _IDEMP[(key, conv)] = {"state": "done", "ts": time.time()}

# ---- TLX W2 knobs -------------------------------------------------------------
ADMIN_TOKEN = os.getenv("TLX_ADMIN_TOKEN", "")   # mirrors the daemon's; gates
                                                 # /health debug fields (V-33)
MAX_BODY_BYTES = int(float(os.getenv("TLX_MAX_BODY_MB", "10")) * 1e6)   # V-34
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "::1"] + [
    h.strip() for h in os.getenv("TLX_ALLOWED_HOSTS", "").split(",") if h.strip()]
OPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")
ENV_CANONICAL = os.getenv("TLX_ENV_CANONICAL", os.path.join(OPS_DIR, "env.canonical"))
ENV_COMMON_PATH = os.getenv("TLX_ENV_COMMON", os.path.join(OPS_DIR, "env.common"))
ENGINE_STAYDOWN_PATHS = (
    "~/tinygrad-metal/logs/llm_engine_staydown",   # W2 persistent (V-26)
    "/tmp/llm_engine_staydown")                              # legacy path

# ---- TLX P8 MULTI-MODEL: registry + durable swap state (FILE-DERIVED) --------
# ops/model_registry.json + ops/state/{current_model,next_model,swap_in_progress}
# are the source of truth — a relaunched API reconstructs swap state from disk,
# never memory (D4). Auto-swap on a non-resident request is DELIBERATELY not
# implemented (thrash); the 409 carries the enginectl switch hint.
# TLX_MODEL_REGISTRY/TLX_STATE_DIR overrides exist for the GPU-free batteries.
STATE_DIR = os.getenv("TLX_STATE_DIR", os.path.join(OPS_DIR, "state"))

def _load_registry():
  _rp = os.getenv("TLX_MODEL_REGISTRY", os.path.join(OPS_DIR, "model_registry.json"))
  try:
    with open(_rp) as f:
      return json.load(f)
  except Exception as e:
    print(f"[api] NOTE: model registry unavailable ({e!r}) — single-model mode", flush=True)
    return None

REGISTRY = _load_registry()

def _state_read(name):
  try:
    with open(os.path.join(STATE_DIR, name)) as f:
      return f.readline().strip()
  except OSError:
    return ""

def resident_model():
  """File-derived resident model id (ops/state/current_model > registry default
  > the legacy single-model id)."""
  if REGISTRY is None:
    return MODEL_ID
  mid = _state_read("current_model") or REGISTRY.get("default_model")
  return mid if mid in REGISTRY.get("models", {}) else MODEL_ID

def swap_state():
  """(in_progress, next_model) from the durable intent files."""
  nxt = _state_read("next_model")
  sw = bool(nxt) and os.path.isfile(os.path.join(STATE_DIR, "swap_in_progress"))
  return sw, (nxt or None)

def model_info(mid):
  return (REGISTRY or {}).get("models", {}).get(mid)

def model_bootable(mid):
  m = model_info(mid)
  if m is None: return False
  envf = os.path.join(OPS_DIR, "env.canonical.d", os.path.basename(m.get("env_file", "")))
  mp = m.get("model_path", "")
  host = m.get("engine_host", "test_w100k.py")
  engdir = os.path.dirname(OPS_DIR)   # the real engine dir (engine0)
  return (os.path.isfile(envf) and (not mp or os.path.isfile(mp))
          and (os.path.isfile(os.path.join(engdir, host)) or os.path.isfile(host)))

def model_caps(mid):
  """Per-model caps (B.5): (ctxk, max_output_tokens)."""
  m = model_info(mid) or {}
  return int(m.get("ctxk", 100352)), int(m.get("max_output_tokens", 32768))

import svc_fp   # the ONE fp implementation (shared with pcache.py/serve.py)
# R3-19: fingerprint the SAME cubin set the daemon loads (bytes only; both
# sides hash identically or every boot would look like config drift).
svc_fp.set_extra("cubins", svc_fp.cubin_set_digest())

def _expected_config():
    """Expected daemon fingerprint derived from the P8 env split
    (ops/env.common + ops/env.canonical.d/<current_model>.env — the same union
    the wrapper sources), falling back to the monolithic ops/env.canonical
    (pre-P8 installs). None when no env source exists -> drift check disabled
    (manual/dev boots), loudly noted in /health.
    Re-derived PER CALL (the /health check + swap transitions): a swap changes
    the expected fp without an API restart — the file-derived swap state (D4).
    TLX_MODEL_ID is NOT in svc_fp._ENV_KEYS: the dense union produces the SAME
    config_fp as the pre-P8 monolithic file (pcache/drift continuity)."""
    mid = resident_model()
    m = model_info(mid) or {}
    env = {}
    # TLX P8 FIX (found live at the MoE bring-up): the registry carries
    # env_file RELATIVE WITH ITS SUBDIR ("env.canonical.d/<model>.env") —
    # basename() alone dropped the subdir, the isfile() check failed, and
    # every derivation silently fell back to the MONOLITHIC env.canonical
    # (the dense numerics) -> permanent config_drift for any split-env model.
    _ef = str(m.get("env_file") or "")
    envf = _ef if os.path.isabs(_ef) else os.path.join(OPS_DIR, *_ef.split("/"))
    if os.path.isfile(ENV_COMMON_PATH) and os.path.isfile(envf):
        env.update(svc_fp.parse_env_file(ENV_COMMON_PATH))
        env.update(svc_fp.parse_env_file(envf))
    elif os.path.isfile(ENV_CANONICAL):
        env = svc_fp.parse_env_file(ENV_CANONICAL)
    else:
        return None, None
    if not env:
        return None, None
    model = env.get("TLX_MODEL_PATH") or m.get("model_path") or GGUF_PATH
    # TLX P8 (MoE bridge): the extras must match the RESIDENT daemon's — the
    # dense daemon mixes the cubin digest, the MoE daemon the mm_pack sha.
    # Derived per model here (clear first: the import-time cubins extra must
    # not leak into a MoE derivation or vice versa).
    svc_fp.clear_extras()
    if (m.get("arch") or "") == "moe-gdn":
        svc_fp.set_mm_pack_extra(env)
    else:
        svc_fp.set_extra("cubins", svc_fp.cubin_set_digest())
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

def _tok_cache_paths(path=None):
  """V-25: cache under a 0700 owner dir (never /tmp — world-writable + wiped);
  name keyed on gguf path+size+mtime (P8: per-model by construction)."""
  d = os.path.expanduser("~/Library/Caches/tlx")
  try:
    os.makedirs(d, mode=0o700, exist_ok=True)
    if hasattr(os, "chmod"):
      try: os.chmod(d, 0o700)      # makedirs mode is masked by umask; enforce
      except Exception: pass
  except Exception:
    return None, None
  p = path or GGUF_PATH
  st = os.stat(p)
  key = hashlib.sha256(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:24]
  return os.path.join(d, f"tok_{key}.pkl"), os.path.join(d, "tok_hmac.key")

# R3-25: the tokenizer-cache HMAC key lives OUTSIDE the cache dir when
# configured (env.canonical TLX_TOK_HMAC_KEY) — a cache-dir-only writer via a
# perms regression used to be able to forge the tag over its own pickle.
TOK_HMAC_KEY_ENV = os.getenv("TLX_TOK_HMAC_KEY", "")

def _hmac_key(kpath):
  """Load/create the per-user HMAC key (0600). Integrity (not secrecy) is the
  goal: a tampered/swapped cache must FAIL the tag and fall back to cold load.
  R3-25: the env key (env.canonical) takes precedence; the cache-dir key is
  the fallback for unconfigured boots (noted loudly)."""
  if TOK_HMAC_KEY_ENV:
    return TOK_HMAC_KEY_ENV.encode()
  try:
    with open(kpath, "rb") as f: return f.read()
  except Exception: pass
  k = os.urandom(32)
  with open(kpath, "wb") as f: f.write(k)
  try: os.chmod(kpath, 0o600)
  except Exception: pass
  print("[api] NOTE: TLX_TOK_HMAC_KEY unset — tokenizer-cache HMAC key lives "
        "in the cache dir (R3-25: set it in ops/env.canonical)", flush=True)
  return k

def _build_template(ct, tok):
  """Template from the raw GGUF string. NOTE: jinja from_string templates are
  NOT picklable (compiled code objects), so the CACHE stores the raw string
  and the template is rebuilt on every load (cheap; the expensive part of a
  cold load is the GGUF KV parse + tokenizer construction)."""
  if not ct:     # R3-45: "" is an EXPLICITLY-EMPTY template - it used to pass
    return FallbackTemplate(tok)   # and silently render every prompt to 
  import jinja2
  from jinja2.sandbox import SandboxedEnvironment
  env = SandboxedEnvironment()
  env.filters['tojson'] = _jinja_tojson
  env.globals['raise_exception'] = _jinja_raise_exception
  return env.from_string(ct)

def load_tokenizer(path=None):
  import pickle
  p = path or GGUF_PATH
  cache, kpath = _tok_cache_paths(p)
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
  kv = parse_gguf_kv(p)
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

# ================= TLX P8 MULTI-MODEL: per-model lazy tokenizer ================
# B.5: dense and MoE share the Qwen2 BPE family but carry their own GGUF KV
# (chat template + special tokens); each model's tokenizer loads (and
# disk-caches) on first use. Binds the module TOK/TEMPLATE globals — the API
# serves exactly ONE resident model at a time (a swap drains + reboots), so the
# bind follows the resident model.
_TOK_LAZY = {}
def tokenizer_for(mid):
  global TOK, TEMPLATE
  if mid not in _TOK_LAZY:
    path = (model_info(mid) or {}).get("model_path") or GGUF_PATH
    t0 = time.time()
    _TOK_LAZY[mid] = load_tokenizer(path)
    print(f"[api] P8 tokenizer for '{mid}' loaded in {time.time()-t0:.1f}s", flush=True)
  TOK, TEMPLATE = _TOK_LAZY[mid]
  return TOK, TEMPLATE

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
QSTATE = {"active": 0, "wait": collections.deque(), "holder": None, "permits": 1,
          "holder_rid": None, "holder_since": None}

def queue_depth():
  return QSTATE["active"] + len(QSTATE["wait"])

def q_promote():
  while QSTATE["wait"] and QSTATE["active"] < QSTATE["permits"]:
    fut = QSTATE["wait"].popleft()
    if fut.done():
      continue                          # R3-11: dead waiter (timed out/cancelled)
    QSTATE["active"] += 1; fut.set_result(True); return

def _q_prune_wait():
  """R3-11: drop done futures from the wait deque — timed-out waiters used to
  sit there forever and count toward MAX_WAITING (429s that outlived the
  actual queue; the only reaper was q_promote on release)."""
  w = QSTATE["wait"]
  while w and w[0].done():
    w.popleft()
  dead = [f for f in w if f.done()]
  for f in dead:
    w.remove(f)

async def q_acquire(permits=1):
  loop = asyncio.get_running_loop()
  QSTATE["permits"] = max(1, int(permits))
  _q_prune_wait()
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
    # R3-10 promote-vs-cancel: the promotion can land in the same event-loop
    # window as this task's cancellation (wait_for deadline) — CPython then
    # raises CancelledError AT the await even though the fut resolved. The
    # grant was consumed; returning it here is the only path (the old
    # `if not fut.done()` no-op leaked it — at permits=1 ONE occurrence
    # bricked admission with permanent 429s until API restart).
    if fut.done() and not fut.cancelled():
      q_release()                   # give the consumed grant back
    else:
      fut.cancel()                  # dead fut: pruned here + skipped by promoter
      _q_prune_wait()
    raise

def q_release():
  QSTATE["active"] = max(0, QSTATE["active"] - 1); QSTATE["holder"] = None
  QSTATE["holder_rid"] = None; QSTATE["holder_since"] = None   # L7.1 diagnostics
  q_promote()

# ---- L7.1: the admission watchdog -------------------------------------------
# Post-L7 live finding: the engine now SURVIVES the dead-consumer mix, which
# exposed an API-side permit leak (the R3-10 class, new variant): QSTATE says
# a holder is active with waiters parked while the ENGINE sits idle — one
# occurrence bricked admission until API restart. The watchdog heals it:
# active>0 + waiters + engine demonstrably NOT busy for ADMIT_HEAL_AFTER_S
# -> loud log (holder rid + duration) + q_release. A genuinely-running holder
# always shows engine busy/rpc (prefills included), so the discriminator
# cannot fire on legitimate work.
_admit_stale_since = {"t": None}
async def _admission_watchdog():
  import time as _t
  while True:
    await asyncio.sleep(ADMIT_WATCHDOG_S)
    try:
      if QSTATE["active"] <= 0 or not QSTATE["wait"]:
        _admit_stale_since["t"] = None; continue
      st = {}
      try:
        st = await asyncio.wait_for(run_in_threadpool(eng_status, 5.0), 8.0)
      except Exception:
        continue                      # engine unreachable: NEVER heal blind
      eng_busy = bool(st.get("busy")) or bool(st.get("rpc"))
      if eng_busy:
        _admit_stale_since["t"] = None; continue
      now = _t.time()
      if _admit_stale_since["t"] is None:
        _admit_stale_since["t"] = now; continue
      if now - _admit_stale_since["t"] >= ADMIT_HEAL_AFTER_S:
        log(f"ADMISSION HEAL (L7.1): QSTATE active={QSTATE['active']} "
            f"waiters={len(QSTATE['wait'])} holder_rid={QSTATE.get('holder_rid')} "
            f"engine idle for {now - _admit_stale_since['t']:.0f}s — releasing a "
            f"leaked permit (the R3-10 class)")
        _admit_stale_since["t"] = None
        q_release()
    except Exception as e:
      log(f"admission_watchdog_error: {e!r}")

def start_admission_watchdog():
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return
  asyncio.ensure_future(_admission_watchdog())

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
# R3-14: a TRUE bounded LRU (count + fed-token byte budget). The old prune
# only evicted NON-reusable entries while every successful turn sets
# reusable=True — UUID-per-chat UIs accumulated fed-mirrors (~2.8MB per
# 100k-token conversation) forever. `reusable` is a hint, not a pin; eviction
# costs only a FRESH prefill and is surfaced via x-resident-evicted.
RESIDENTS = {}
RESIDENT_MAX = int(os.getenv("TLX_RESIDENT_MAX", "64"))
RESIDENT_FED_BUDGET = int(os.getenv("TLX_RESIDENT_FED_MAX_TOKENS", str(2_000_000)))
_RECENTLY_EVICTED = collections.OrderedDict()   # cid -> ts (bounded)
def _note_evicted(cid, n_fed):
  if cid:
    _RECENTLY_EVICTED[cid] = time.time()
    while len(_RECENTLY_EVICTED) > 64:
      _RECENTLY_EVICTED.popitem(last=False)
def _evict_residents(keep_key):
  def _cands():
    return [v for k, v in RESIDENTS.items()
            if k != keep_key and not (CONV_LOCKS.get(k) and CONV_LOCKS[k].locked())]
  while len(RESIDENTS) > RESIDENT_MAX:
    c = _cands()
    if not c: break
    v = min(c, key=lambda r: r.get("last_use", 0))
    RESIDENTS.pop(v["conversation_id"] or "", None)
    _note_evicted(v["conversation_id"], len(v.get("fed") or ()))
  total = sum(len(v.get("fed") or ()) for v in RESIDENTS.values())
  while total > RESIDENT_FED_BUDGET:
    c = _cands()
    if not c: break
    v = min(c, key=lambda r: r.get("last_use", 0))
    n = len(v.get("fed") or ())
    RESIDENTS.pop(v["conversation_id"] or "", None)
    _note_evicted(v["conversation_id"], n)
    total -= n
def _resident(cid):
  key = cid or ""
  r = RESIDENTS.get(key)
  if r is None:
    r = RESIDENTS.setdefault(key, {"conversation_id": cid, "fed": [], "messages": None, "reusable": False})
  r["last_use"] = time.time()
  _evict_residents(key)
  return r
def _resident_reset(cid):
  R = _resident(cid)
  R.update({"conversation_id": None if not cid else cid, "fed": [], "messages": None, "reusable": False})

def _is_eviction_class(errmsg):
  """R3-15: the engine-side 'slot evicted between decide and prefill' error
  class (batch slot-LRU raced our decide->prefill window)."""
  m = str(errmsg)
  return ("no resident conversation" in m) or ("mismatch" in m and "use FRESH" in m)

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

class _VisibleStopGate:
  """R3-36: client stop-STRINGS matched on the VISIBLE content channel
  (post-think-split) by default - a stop string inside <think>...</think> used
  to end the request though it never appears in visible content. Same holdback
  law as DetokStream (never emit the last maxstop-1 chars until the window
  clears); opt back into RAW-stream stops via x-tlx-stop-raw: true."""
  def __init__(self, stops):
    self.full = ""; self.emitted = 0
    self.stops = [x for x in stops if x] or []
    self.maxstop = max((len(x) for x in self.stops), default=0)
    self.stop_found = None            # (stop_string, index)
  def _search(self):
    if self.stop_found is not None or not self.stops: return
    for x in self.stops:
      i = self.full.find(x, max(0, self.emitted - self.maxstop))
      if i >= 0: self.stop_found = (x, i); return
  def feed(self, piece):
    self.full += piece
    return self._safe()
  def _safe(self):
    self._search()
    if self.stop_found is not None:
      _, i = self.stop_found
      out = self.full[self.emitted:i] if self.emitted < i else ""
      self.emitted = max(self.emitted, i)
      return out
    safe_end = max(self.emitted, len(self.full) - (self.maxstop - 1) if self.maxstop else len(self.full))
    out = self.full[self.emitted:safe_end]; self.emitted = safe_end
    return out
  def final(self):
    if self.stop_found is None:
      self._search()
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

@app.on_event("startup")
async def _l7_startup():
  start_admission_watchdog()      # L7.1: heal the leaked-permit class (never blind)

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
                                 "type": "invalid_request_error", "param": None,
                                 "code": "request_too_large"}}).encode()
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
  """R3-24: the ENGINE view is behind the admin header — the old unauth
  subset (pos + busy + rpc + config_fp + lookup_k + pf_prefill) was a
  resident-state oracle for ANY localhost process or a proxying gateway."""
  return {k: st.get(k) for k in ("ready", "busy", "rpc", "pos", "ctxk", "mode",
                                 "queue", "dirty", "uptime_s", "config_fp",
                                 "lookup_k", "pf_prefill", "cycles_since_rebuild",
                                 "batch_b", "cycle_cap", "pc", "model_id", "ka")}  # L2b; ka = TLX P10 pool counters

def _admin_ok(request):
  """R3-23: HEADER ONLY (x-admin-token) — the ?admin_token= query param sat in
  proxy/access logs; query strings are the classic credential-leak channel."""
  if not ADMIN_TOKEN: return False
  tok = request.headers.get("x-admin-token") or ""
  return hmac.compare_digest(str(tok), ADMIN_TOKEN)

def _drift_state(st):
  """None when the daemon config matches the P8 env union (env.common +
  env.canonical.d/<current_model>.env — re-derived PER CALL so a swap changes
  the expected fp without an API restart; falls back to the monolithic
  ops/env.canonical); (daemon_fp, expected_fp) on drift. None cases: no env
  source (manual/dev boot — check disabled, loudly reported) or the daemon
  predates W2 (no config_fp in status)."""
  efp, _ = _expected_config()
  if efp is None: return None
  fp = st.get("config_fp")
  if fp is None or fp == efp: return None
  return (fp, efp)

@app.get("/health")
async def health(request: Request):
  full = _admin_ok(request)   # ops detail (fed_tail/conversation_id) only behind the token
  # TLX P8: a swap in progress is a 503 with ETA (the durable intent files are
  # the source of truth — reconstructed from disk after any API relaunch).
  _sw, _nxt = swap_state()
  if _sw:
    out = {"status": "swap_in_progress", "queue_depth": 0,
           "detail": f"model swap to '{_nxt}' armed (durable intent); the engine "
                     f"reboots into it (~10-13 min boot). Retry after."}
    if full:
      out["swap"] = {"next_model": _nxt, "resident_model": resident_model()}
    return JSONResponse(status_code=503, headers={"Retry-After": "600"}, content=out)
  try:
    st = await run_in_threadpool(eng_status, 2.0)
  except Exception as e:
    degraded = any(os.path.exists(p) for p in ENGINE_STAYDOWN_PATHS)
    return JSONResponse(status_code=503, headers={"Retry-After": "5"}, content={
      "status": "engine_degraded" if degraded else "engine_down", "detail": str(e)[:200],
      "queue_depth": queue_depth()})
  drift = _drift_state(st)
  if drift is not None:
    # R3-24: the fingerprint pair + runbook detail sit behind the admin header
    out = {"status": "config_drift", "queue_depth": queue_depth()}
    if full:
      out["detail"] = (f"daemon config_fp {drift[0]} != canonical {drift[1]} — the engine "
                       f"booted without ops/env.canonical (slow-path/LOOKUP-off class). "
                       f"Restart the engine through the wrapper.")
      out["config"] = {"config_fp": drift[0], "config_fp_expected": drift[1]}
    return JSONResponse(status_code=503, headers={"Retry-After": "30"}, content=out)
  if not st.get("ready"):
    out = {"status": "warming", "queue_depth": queue_depth()}
    if full:
      out["engine"] = _safe_engine_fields(st)
    return JSONResponse(status_code=503, headers={"Retry-After": "5"}, content=out)
  # R3-24: unauthenticated /health = {status, queue_depth} ONLY (pos/knobs/
  # fp = resident-state oracle); the engine view needs the admin header.
  out = {"status": "ok", "queue_depth": queue_depth()}
  if full:
    _efp, _ = _expected_config()
    out["model"] = {"resident": resident_model(), "registry": bool(REGISTRY)}
    out["engine"] = _safe_engine_fields(st)
    out["config"] = {"config_fp": st.get("config_fp"),
                     "config_fp_expected": _efp,
                     "drift_check": "on" if _efp is not None else "DISABLED (no env source)"}
    out["engine_debug"] = st   # full daemon status incl. fed_tail/conversation_id
  return out

@app.get("/v1/models")
async def models():
  # TLX P8: the registry-driven model list. `status`: resident (the engine
  # booted it), loadable (validated: env + weights + host present — an
  # `enginectl switch` away), unavailable (a validation failure). Legacy
  # single-model shape when no registry.
  if REGISTRY is None:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "egpu-rig"}]}
  _res = resident_model()
  _sw, _nxt = swap_state()
  data = []
  for mid, m in sorted(REGISTRY.get("models", {}).items()):
    if mid == _res:
      status = "swapping-out" if (_sw and _nxt and _nxt != mid) else "resident"
    elif _sw and mid == _nxt:
      status = "swapping-in"
    else:
      status = "loadable" if model_bootable(mid) else "unavailable"
    data.append({"id": mid, "object": "model", "created": 0, "owned_by": "egpu-rig",
                 "status": status, "display_name": m.get("display_name", ""),
                 "context_window": m.get("ctxk")})
  return {"object": "list", "data": data}

def _validate_fields(body):
  """Returns (error_response|None, ctx|None). Shared by stream+non-stream."""
  msgs = body.get("messages")
  if not isinstance(msgs, list) or not msgs:
    return oai_error(400, "'messages' must be a non-empty list"), None
  for m in msgs:
    if not isinstance(m, dict):
      return oai_error(400, f"invalid message (roles: system/user/assistant): {str(m)[:120]}"), None
    # R3-38: tool-message shapes are a DISTINCT refusal class — checked BEFORE
    # the generic role check so the client gets param+code, not a vague 400
    if m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("tool_calls")):
      if COMPAT_STRIP_TOOL_MESSAGES:
        continue                    # strip indices collected below
      return oai_error(400, "tool messages and assistant tool_calls are not supported at "
                            "this endpoint (replaying tool-use histories would silently "
                            "corrupt them; TLX_COMPAT_STRIP_TOOL_MESSAGES=1 strips them "
                            "and records the strip in ignored_params)",
                       param="messages", code="unsupported_tool_messages"), None
    if m.get("role") not in ("system", "user", "assistant", "developer"):
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
  sampling_ignored = []
  for k in ("temperature", "top_p", "top_k"):
    v = body.get(k)
    if v is None: continue
    num = isinstance(v, (int, float)) and not isinstance(v, bool)
    in_range = num and (
        (0.0 <= float(v) <= 2.0) if k == "temperature" else
        (0.0 < float(v) <= 1.0) if k == "top_p" else
        (float(v) >= 1 and float(v) == int(float(v))))
    if num and float(v) in (0.0, 1.0):
      continue                 # greedy-neutral: accepted as before
    # R3-33: capability-coded refusal (or in-range acceptance under the
    # TLX_COMPAT_IGNORE_SAMPLING compat mode — recorded in ignored_params,
    # decode stays greedy/bit-exact)
    if COMPAT_IGNORE_SAMPLING and in_range:
      sampling_ignored.append(k)
      continue
    want = "0 <= temperature <= 2" if k == "temperature" else \
           "0 < top_p <= 1" if k == "top_p" else "top_k an integer >= 1"
    return oai_error(400, f"{k}={v}: sampling is not supported at this endpoint "
                          f"(greedy decode, bit-exact). {want} is accepted under the "
                          f"TLX_COMPAT_IGNORE_SAMPLING compat mode (values recorded in "
                          f"ignored_params; the decode stays greedy).",
                     param=k, code="unsupported_sampling"), None
  strip_tool_idx = []
  if COMPAT_STRIP_TOOL_MESSAGES:
    strip_tool_idx = [i for i, m in enumerate(msgs)
                      if m.get("role") == "tool"
                      or (m.get("role") == "assistant" and m.get("tool_calls"))]
  if body.get("tools") is not None or body.get("tool_choice") is not None:
    return oai_error(400, "tools/function-calling are not supported at this endpoint (no "
                          "tool runtimes: Cline/Cursor Agent/Claude Code will not work "
                          "against this endpoint)",
                     param="tools", code="unsupported_tools"), None
  # R3-38: top_logprobs=0 alone (logprobs falsy) is IGNORED, not a 400
  if body.get("logprobs"):
    return oai_error(400, "logprobs are not supported at this endpoint",
                     param="logprobs", code="unsupported_logprobs"), None
  _tl = body.get("top_logprobs")
  extra_ignored = []
  if _tl is not None:
    if not isinstance(_tl, int) or isinstance(_tl, bool) or _tl < 0:
      return oai_error(400, "'top_logprobs' must be a non-negative integer",
                       param="top_logprobs"), None
    if _tl > 0:
      return oai_error(400, "logprobs are not supported at this endpoint "
                            "(top_logprobs > 0 requires them)",
                       param="top_logprobs", code="unsupported_logprobs"), None
    extra_ignored.append("top_logprobs")
  # R3-36: client stop_token_ids - validated and MERGED (token-level stops)
  _stid = body.get("stop_token_ids")
  if _stid is not None:
    if (not isinstance(_stid, list) or len(_stid) > 16
        or not all(isinstance(t, int) and not isinstance(t, bool) for t in _stid)):
      return oai_error(400, "'stop_token_ids' must be a list of up to 16 integers",
                       param="stop_token_ids", code="invalid_stop_token_ids"), None
  if body.get("n") not in (None, 1):
    return oai_error(400, "n>1 is not supported in M1 (single completion)"), None
  if body.get("response_format") is not None:
    return oai_error(400, "response_format is not supported in M1 (plain text only)"), None
  stop = body.get("stop")
  if stop is not None:
    if isinstance(stop, str): stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) and s for s in stop) or len(stop) > 4:
      return oai_error(400, "'stop' must be a string or a list of up to 4 non-empty strings"), None
  # V-08: max_completion_tokens alias (reject when both set and disagreeing).
  # R3-44: TYPES FIRST (the old int(mt) != int(mct) comparison ran before
  # type validation — a non-numeric string raised ValueError -> 500).
  mt, mct = body.get("max_tokens"), body.get("max_completion_tokens")
  for _n, _v in (("max_tokens", mt), ("max_completion_tokens", mct)):
    if _v is not None and (not isinstance(_v, int) or isinstance(_v, bool)):
      return oai_error(400, f"'{_n}' must be an integer", param=_n,
                       code="invalid_max_tokens"), None
  if mt is not None and mct is not None and mt != mct:
    return oai_error(400, "'max_tokens' and 'max_completion_tokens' are both set and differ; "
                          "send at most one of them"), None
  max_tokens = mt if mt is not None else mct
  if max_tokens is not None and max_tokens < 1:
    return oai_error(400, "'max_tokens'/'max_completion_tokens' must be a positive integer",
                     param="max_tokens", code="invalid_max_tokens"), None
  # V-07: reasoning options forwarded to the template (default effort medium,
  # NOT the template's own xhigh default). R3-32: the OpenAI-standard `high`
  # maps to this template's strongest tier (xhigh); xhigh stays as the alias.
  eff = body.get("reasoning_effort")
  if eff is not None and eff not in ("xhigh", "high", "medium", "low"):
    return oai_error(400, "'reasoning_effort' must be one of low/medium/high "
                          "(default medium at this endpoint; xhigh accepted as "
                          "this template's strongest-tier alias)",
                     param="reasoning_effort", code="invalid_reasoning_effort"), None
  eff = "xhigh" if eff == "high" else eff
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
                            "parallel_tool_calls") if k in body] + sampling_ignored + extra_ignored,
    "strip_tool_idx": strip_tool_idx,
    "stop_token_ids": _stid,
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

def _encode_with_memo(rtext, memo=None):
  """R3-09: encode ONCE per request. The pre-slot pass (threadpool) fills the
  per-request memo; the post-slot decision reuses it. The vendored encoder is
  O(pairs) per merge — a 50-100k-token prompt encoded on the asyncio loop
  (twice, pre- and post-slot, in the old code) stalls every SSE generator,
  the disconnect watcher and /health for seconds."""
  if memo is not None and memo.get("rtext") == rtext and memo.get("ids") is not None:
    return memo["ids"]
  ids = TOK.encode(rtext)
  if memo is not None:
    memo["rtext"] = rtext; memo["ids"] = ids
  return ids

def _decide_prefix(rtext, engine_st, conv_id, tvars=None, phase="decide", enc_memo=None):
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
  _fresh_reason = None
  _had_mirror = bool(fed or R.get("messages"))   # R3-43: first turns are not news
  if R.get("tvars") and tvars is not None and R["tvars"] != tvars:
    # R3-43: a template-var flip (e.g. enable_thinking toggled between turns)
    # silently forced FRESH with NO diagnostic — a wasted 100k prefill the
    # operator couldn't attribute. Surface the reason.
    _fresh_reason = "tvars_changed"
  if (_fresh_reason is None
      and conv_id is not None and R["conversation_id"] == conv_id
      and R.get("model_id", MODEL_ID) == MODEL_ID          # R3-42
      and est is not None
      and not est.get("dirty")
      and R.get("reusable") and fed and rend
      and rtext.startswith(rend) and len(rtext) > len(rend)
      and est.get("fed_len") == len(fed)):
    tail_ids = TOK.encode(rtext[len(rend):])
    ids2 = fed + tail_ids
    return ids2, "FOLLOW_UP", ids2[len(fed)], ids2[len(fed)+1:], _fresh_reason
  ids2 = _encode_with_memo(rtext, enc_memo)
  _reason = _fresh_reason
  if _reason is None and _had_mirror and conv_id and not R.get("reusable"):
    _reason = "no_mirror"
  elif _reason is None and _had_mirror and rend:
    _reason = "prefix_mismatch"
  return ids2, "FRESH", None, ids2, _reason

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

def _make_stream_events(maxsize, put_timeout=5.0, on_dead=None):
  """R3-12 (ledger §3.3 ruling: coalesce-first, loud-fail backstop — never a
  silent drop): the bounded SSE event queue shared between the engine thread
  and the ASGI generator. A slow-but-ALIVE consumer (TCP backpressure) must
  not lose its generation: on Full, pending text/reasoning deltas COALESCE
  into one event per kind (the maxsize-event queue becomes effectively
  byte-bounded). Only when a COALESCED put still times out twice (the
  consumer is gone) do we mark dead + call on_dead (the route arms the
  engine cancel); the generator then emits an SSE error event and ends.
  Prefill-progress comments stay droppable (cosmetic)."""
  evq = _pyqueue.Queue(maxsize=maxsize)
  pend = {"text": "", "reasoning": "", "stalls": 0, "dead": False}
  def on_event(kind, p):
    if pend["dead"]:
      return
    if kind not in ("text", "reasoning"):
      try:
        evq.put((kind, p), timeout=put_timeout)
      except _pyqueue.Full:
        return                       # cosmetic progress only
      return
    pend[kind] += p
    try:
      evq.put((kind, pend[kind]), timeout=put_timeout)
      pend[kind] = ""
      pend["stalls"] = 0
    except _pyqueue.Full:
      pend["stalls"] += 1           # coalesced: everything stays pending
      if pend["stalls"] >= 2:       # a full 2x put_timeout window with no
        pend["dead"] = True         # drain = dead consumer
        log("event queue full after coalescing — arming cancel (dead consumer)")
        if on_dead is not None:
          try: on_dead()
          except Exception: pass
  return evq, pend, on_event

def _run_engine_sync(mode, cur, delta, ids2, rtext, messages, conv_id, max_tokens, stop_token_ids,
                     stops, on_event=None, cancel_check=None, cache_key=None, think_open=False,
                     cache_ttl=None, stop_raw=False, rid=None, tvars=None):
  """Blocking engine session for one request (runs in executor thread). Owns the
  DetokStream/ThinkSplitter so stop-STRING matches also cancel the engine
  promptly.  on_event(kind, payload) fires from this thread; kinds: "prefill",
  "text" (client-visible content), "reasoning". Mutates the PER-CONVERSATION
  resident mirror R (fed mirror, render-text mirror, reusable flag). Returns
  {tokens, visible, text_tail, reasoning, finish, cancelled, mode, ...}."""
  result = {"tokens": [], "visible": 0, "text_tail": "", "reasoning": "",
            "finish": "stop", "cancelled": False, "mode": mode,
            "prefix_mode": mode, "cached_tokens": 0}
  # R3-36: stop strings match the VISIBLE channel by default (x-tlx-stop-raw:
  # true restores the old raw-stream semantics)
  ds = DetokStream(stops if stop_raw else [])
  vgate = _VisibleStopGate([] if stop_raw else stops)
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
      cseg = vgate.feed(cseg)            # R3-36: visible-channel stop gate
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
      _p = {"mode": "AUTO_CACHE", "ids": ids2, "conversation_id": conv_id,
            "model_id": MODEL_ID}
      if rid: _p["rid"] = rid                  # R3-48: end-to-end correlation
      if cache_key: _p["cache_key"] = cache_key
      if cache_ttl: _p["cache_ttl"] = int(cache_ttl)   # W3 V-44: honored through the RPC
      c.send({"id": 10, "method": "prefill", "params": _p})
    else:
      c.send({"id": 10, "method": "prefill",
              "params": {"mode": "FOLLOW_UP", "ids": delta, "cur": cur,
                         "conversation_id": conv_id, "model_id": MODEL_ID,
                         **({"rid": rid} if rid else {})}})   # R3-42/R3-48
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
    if mode == "FOLLOW_UP" and not result.get("cached_tokens"):
      # R3-41: the reused resident prefix IS prompt-cache hit accounting
      result["cached_tokens"] = max(0, len(ids2) - len(delta))
    R["conversation_id"] = conv_id
    R["model_id"] = MODEL_ID               # R3-42: (model_id, cid) keying
    if tvars is not None:
      R["tvars"] = dict(tvars)             # R3-43: tvars stored WITH the mirror
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
            "params": {"max_cycles": int(max_tokens) + 4, "stop_token_ids": stop_token_ids,
                       **({"rid": rid} if rid else {})}})      # R3-48
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
    # P9 EVAL FIX (first-token loss): in every prefill mode the ENGINE's
    # first RESPONSE token is the prefill result's `cur` (predicted at the
    # boundary hidden, held in cur_slot -- deliberately NOT part of the fed
    # stream nor the cycle emits; serve.py: "generate appends emits,
    # FOLLOW_UP appends [cur]+delta"). The cycle events start at the SECOND
    # response token, so `cur` must be fed through the visible path ONCE,
    # before the event loop, or every completion loses its first token
    # (one-token answers came back EMPTY; found by the P9 needle eval +
    # eval/probe_engine.py).
    _cur0 = _pr.get("cur")
    if _cur0 is not None:
      _cur0 = int(_cur0)
      result["tokens"].append(_cur0)
      _feed_token(_cur0)
      if _cur0 in stopset:
        R["reusable"] = False
        return bail("stop")
      if not split.in_think and result["visible"] >= max_tokens:
        return bail("length")
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
          if (ds.stop_found if stop_raw else vgate.stop_found) is not None:
            R["reusable"] = False
            return bail("stop")                # stop-STRING matched (same class;
                                               # R3-36: on the VISIBLE channel)
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
      if not stop_raw and ctail:              # R3-36: gate the visible tail too
        ctail = vgate.feed(ctail) + vgate.final()
    except Exception:
      pass
    result["reasoning"] = "".join(reasoning_parts) + rtail
    result["reasoning_tail"] = rtail     # R3-35: the stream must emit this too
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
    return oai_error(415, "content-type must be application/json",
                     code="unsupported_media_type")
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
                     code="request_too_large")
  except Exception:
    return oai_error(400, "invalid JSON body")
  # R3-40: conversation sources in order — body field, forwarded header, then
  # the `user` field as the LAST fallback (gateways commonly drop unknown body
  # fields; `user` survives OpenAI-compatible/OpenAI passthroughs).
  _u = body.get("user")
  conv_id = (body.get("conversation_id") or request.headers.get("x-conversation-id")
             or (_u if isinstance(_u, str) and _u else None))
  # W-C.11 (qwen-12/grok-7 convergence): the conversation pin is a cache KEY —
  # type/length/control-char validated like any other client-supplied key
  if conv_id is not None and (not isinstance(conv_id, str) or not conv_id
                              or len(conv_id) > 128
                              or any(ord(c) < 32 for c in conv_id)):
    return oai_error(400, "'conversation_id' must be a non-empty string (<=128 chars, "
                          "printable)", param="conversation_id",
                     code="invalid_conversation_id")
  err, ctx = _validate_fields(body)
  if err: return err
  # ---- TLX P8 B.5: the model field is LOAD-BEARING --------------------------------
  # 404 unknown id; 409 known-but-not-resident (NO auto-swap — an engine swap
  # is a ~15 min reboot, not a per-request decision); default (absent field) =
  # the resident model. The response always echoes the SERVED model.
  _res = resident_model()
  _sw, _nxt = swap_state()
  if _sw:
    resp = oai_error(503, f"model swap to '{_nxt}' in progress (the engine reboots into it; "
                          f"~10-13 min)", err_type="engine_busy", code="swap_in_progress")
    resp.headers["Retry-After"] = "600"
    return resp
  _req_model = body.get("model")
  if REGISTRY is not None and isinstance(_req_model, str) and _req_model:
    # (legacy no-registry mode: the field is accepted verbatim, pre-P8 shape)
    if _req_model not in REGISTRY.get("models", {}) and _req_model != MODEL_ID:
      return oai_error(404, f"unknown model '{_req_model}' (see GET /v1/models)",
                       param="model", code="model_not_found")
    if _req_model not in (_res, MODEL_ID):
      resp = oai_error(409, f"model '{_req_model}' is not resident (current: '{_res}'); "
                            f"an operator swap is required: enginectl switch {_req_model}",
                       param="model", code="model_not_resident")
      resp.headers["Retry-After"] = "60"
      return resp
  try:
    tokenizer_for(_res)   # bind TOK/TEMPLATE to the RESIDENT model (lazy per model)
  except Exception as e:
    return oai_error(503, f"tokenizer load failed for resident model '{_res}': {str(e)[:150]}",
                     err_type="api_error", code="tokenizer_unavailable")
  _cap_ctxk, _cap_out = model_caps(_res)
  if ctx["max_tokens"] is not None and ctx["max_tokens"] > _cap_out:
    return oai_error(400, f"'max_tokens' exceeds the per-model cap ({_cap_out} for '{_res}')",
                     param="max_tokens", code="max_tokens_exceeds_model_cap")

  try:
    eng_st = await run_in_threadpool(eng_status, 5.0)
  except Exception as e:
    return oai_error(503, f"engine unavailable: {str(e)[:150]}",
                     err_type="api_error", code="engine_down")
  # ---- V-28: config-drift refusal. A daemon booted outside env.canonical
  # (e.g. launchd's stale plist env -> LOOKUP_K=0 + T=1 slow FRESH path) must
  # fail LOUDLY, not serve degraded while /health says ok.
  drift = _drift_state(eng_st)
  if drift is not None:
    return oai_error(503, f"engine config drift: daemon config_fp {drift[0]} != canonical "
                          f"{drift[1]} — restart the engine through ops/engine_daemon.sh "
                          f"(sources ops/env.canonical)", err_type="api_error", code="config_drift")

  tvars = ctx["template_vars"]
  rtext = await run_in_threadpool(_render_text, body["messages"], tvars)
  # R3-09: the FULL validation decision (history render + BPE encode — the
  # expensive, loop-stalling part) runs in the THREADPOOL and fills the
  # per-request encode memo; the post-slot pass re-runs only the cheap
  # decision (prefix checks + a fresh eng_status) against the memo.
  enc_memo = {}
  await run_in_threadpool(_decide_prefix, rtext, eng_st, conv_id, tvars, "pre-slot", enc_memo)
  # R3-39: idempotency claim AFTER validation, BEFORE admission
  idem_key = request.headers.get("idempotency-key") or request.headers.get("x-request-id")
  if idem_key and not _idemp_claim(idem_key, conv_id):
    resp = oai_error(409, "an in-flight request holds this Idempotency-Key for the same "
                          "conversation; retry after it completes (gateway retry storms "
                          "double-bill GPU time)", err_type="invalid_request_error",
                     code="in_flight_duplicate")
    return resp

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
  # L7.1 (the exposed R3-10-class leak): a leaked conv lock (any stream whose
  # release_slot never ran) used to block lk.acquire() FOREVER while holding
  # the admission permit — anonymous requests share the ""-keyed lock, so ONE
  # leak bricked the whole API (observed live: QSTATE active=1, engine idle,
  # permanent 429s). Bounded acquire converts any future leak into a 503.
  lk = _conv_lock(conv_id)
  conv_held = {"v": False}
  conv_released = {"v": False}
  def release_conv():
    if conv_released["v"]: return
    conv_released["v"] = True
    if conv_held["v"]:
      conv_held["v"] = False
      try: lk.release()
      except Exception: pass
  try:
    await asyncio.wait_for(lk.acquire(), CONV_LOCK_WAIT_S)
    conv_held["v"] = True
  except asyncio.TimeoutError:
    q_release()                     # the permit must not leak with us
    resp = oai_error(503, f"conversation lock busy >{CONV_LOCK_WAIT_S:.0f}s "
                          "(a prior stream on this conversation failed to release; "
                          "engine is not blocked)", err_type="engine_busy")
    resp.headers["Retry-After"] = "10"
    log("conv_lock_timeout (L7.1): admission released; suspected leaked lock "
        f"holder_rid={QSTATE.get('holder_rid')}")
    return resp
  try:
    try:
      eng_st = await run_in_threadpool(eng_status, 5.0)
    except Exception as e:
      return oai_error(503, f"engine unavailable: {str(e)[:150]}",
                     err_type="api_error", code="engine_down")
    ids2, mode, cur, delta, fresh_reason = await run_in_threadpool(
        _decide_prefix, rtext, eng_st, conv_id, tvars, "post-slot", enc_memo)
    # V-02: mode-aware headroom. A FRESH/AUTO_CACHE prefill RESETS the engine
    # pos to the new prompt — the stale parked pos must not clamp it (the old
    # max(pos, len) form 400-locked every FRESH request after a 100k turn).
    # R6 P3: FOLLOW_UP headroom reads THIS conversation's stream pos (the
    # top-level pos is ambiguous with 2 resident streams).
    est = _engine_stream_for(eng_st, conv_id)
    fu_pos = (est or {}).get("pos", eng_st.get("pos", 0))
    base_pos = max(fu_pos, len(ids2)) if mode == "FOLLOW_UP" else len(ids2)
    # TLX P8 B.5: the per-model cap = min(registry ctxk, live engine ctxk)
    _eff_ctxk = min(int(eng_st.get("ctxk", 100352)), model_caps(resident_model())[0])
    headroom = _eff_ctxk - base_pos - HEADROOM
    max_tokens = ctx["max_tokens"] or 512
    if headroom < 1:
      return oai_error(400, f"context window exhausted: prompt {len(ids2)} tokens, mode {mode}, "
                            f"engine pos {eng_st.get('pos')}, ctx {eng_st.get('ctxk')}",
                       code="context_window_exhausted")
    if max_tokens > headroom:
      log(f"clamping max_tokens {max_tokens} -> {headroom}")
      max_tokens = headroom
    # R3-34 (honest-first): the engine clamps max_cycles to min(4096, ctxk-pos);
    # a request that CANNOT be honored within that budget is a clean 400 here
    # (silent early finish_reason:length under-delivery is the alternative).
    cycle_cap = max(1, min(4096, int(eng_st.get("ctxk", 100352)) - int(base_pos)))
    if max_tokens + 4 > cycle_cap:
      return oai_error(400, f"max_tokens {max_tokens} exceeds the engine cycle cap "
                            f"({cycle_cap} at context position {base_pos}; the daemon "
                            f"clamps generation to min(4096, ctxk-pos) cycles and would "
                            f"end the completion early with finish_reason length). Ask for "
                            f"fewer tokens; the live cap is in /health (admin) cycle_cap.",
                       param="max_tokens", code="max_tokens_exceeds_engine_cycle_cap")

    # TLX P8 B.5: the echo is the SERVED model (the resident id), never the
    # request string verbatim — a stale client model name must not round-trip.
    # (Legacy no-registry mode keeps the pre-P8 verbatim echo.)
    model_echo = resident_model() if REGISTRY is not None else body.get("model", MODEL_ID)
    rid = request.headers.get("x-request-id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    stops = ctx["stop"] or []
    stop_raw = (request.headers.get("x-tlx-stop-raw") or "").lower() in ("1", "true")
    # R3-36: client stop_token_ids validated + merged (vocab-checked)
    _client_stops = [t for t in (ctx.get("stop_token_ids") or [])
                     if isinstance(t, int) and 0 <= t < len(TOK._tok2bytes)]
    stop_token_ids = sorted(set(_stop_ids_for(stops)) | set(_client_stops))
    # R3-38: compat-stripped tool messages never reach the render
    _render_msgs = ([m for i, m in enumerate(body["messages"])
                     if i not in set(ctx.get("strip_tool_idx") or ())]
                    if ctx.get("strip_tool_idx") else body["messages"])
    think_open = _think_open(rtext)
    # R3-40: LOUD log when a multi-message request has no conversation source
    # (behind a gateway that drops the field, every turn silently thrashes
    # anonymous-resident/FRESH instead of FOLLOW_UP reuse).
    if conv_id is None and len(body["messages"]) > 1:
      log(f"request {rid}: WARNING multi-message request without a conversation "
          f"source (conversation_id body field / x-conversation-id header / user "
          f"fallback all absent) — prefix reuse is dead for this client")
    QSTATE["holder_rid"] = rid; QSTATE["holder_since"] = time.time()  # L7.1 diagnostics
    log(f"request {rid}: mode={mode} prompt={len(ids2)} fed={eng_st.get('fed_len')} "
        f"delta={len(delta)} max_tokens={max_tokens} stops={stops} stop_ids={stop_token_ids} "
        f"conv={conv_id} think_open={think_open}")

    # ---- R3-13/R3-15: shared disconnect watcher + evicted->FRESH fallback ----
    cancel_flag = {"v": False}
    async def _watch_disconnect():
      """M1-C: RELIABLE disconnect detection (request.is_disconnected proved
      UNRELIABLE on this stack). A task that OWNS receive() gets the
      http.disconnect message deterministically. R3-13: installed for
      NON-STREAM requests too — they used to pass cancel_check=None and
      generate on for a vanished client."""
      try:
        while True:
          msg = await request.receive()
          if msg.get("type") == "http.disconnect":
            cancel_flag["v"] = True
            log(f"{rid}: client disconnected -> engine cancel armed")
            return
      except Exception:
        return
    def _session_with_fallback():
      """R3-15: a batch slot-LRU eviction between our decide and the prefill
      RPC surfaces as the engine's mismatch/'use FRESH' error — retry ONCE as
      FRESH (mirrors the dirty-slot FRESH law) instead of a client 503."""
      try:
        return _run_engine_sync(mode, cur, delta, ids2, rtext, _render_msgs,
                                conv_id, max_tokens, stop_token_ids, stops, on_event,
                                lambda: cancel_flag["v"], ctx.get("prompt_cache_key"),
                                think_open, ctx.get("prompt_cache_ttl"),
                                stop_raw=stop_raw, rid=rid, tvars=tvars)
      except EngineError as e:
        if mode != "FOLLOW_UP" or not _is_eviction_class(e):
          raise
        R = _resident(conv_id); R["reusable"] = False
        st2 = eng_status(5.0)
        ids2b, mode2, cur2, delta2, _fr2 = _decide_prefix(rtext, st2, conv_id, tvars,
                                                          "evicted-fallback", enc_memo)
        log(f"{rid}: FOLLOW_UP slot evicted between decide and prefill — "
            f"one FRESH fallback (mode {mode2})")
        return _run_engine_sync(mode2, cur2, delta2, ids2b, rtext, _render_msgs,
                                conv_id, max_tokens, stop_token_ids, stops, on_event,
                                lambda: cancel_flag["v"], ctx.get("prompt_cache_key"),
                                think_open, ctx.get("prompt_cache_ttl"),
                                stop_raw=stop_raw, rid=rid, tvars=tvars)

    if not ctx["stream"]:
      pieces = []
      def on_event(kind, p):
        if kind == "text": pieces.append(p)
      watcher = asyncio.ensure_future(_watch_disconnect())   # R3-13
      try:
        res = await run_in_threadpool(_session_with_fallback)
      finally:
        cancel_flag["v"] = True
        try: watcher.cancel()
        except Exception: pass
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
      if conv_id and _RECENTLY_EVICTED.pop(conv_id, None) is not None:
        hdrs["x-resident-evicted"] = "1"       # R3-14: eviction is observable
      if fresh_reason:
        hdrs["x-prefix-fresh-reason"] = fresh_reason   # R3-43
      _idemp_done(idem_key, conv_id)           # R3-39
      return JSONResponse(content=out, headers=hdrs)

    # ---- stream ----
    # V-16: bounded event queue; R3-12: coalesce-first + loud-fail backstop
    # (the engine thread NEVER silently drops client-visible text).
    evq, pend, on_event = _make_stream_events(
        EVQ_MAX, on_dead=lambda: cancel_flag.__setitem__("v", True))
    loop = asyncio.get_running_loop()

    fut = asyncio.ensure_future(run_in_threadpool(_session_with_fallback))
    watcher = asyncio.ensure_future(_watch_disconnect())
    gen_done = asyncio.Event()
    released = {"v": False}
    gen_completed = {"v": False}
    def release_slot():
      """V-01: exactly-once FIFO release, owned by the stream (the route's
      finally runs BEFORE the generator starts — it must not release here).
      R3-08: the per-conversation lock is released HERE too — the stream
      lifetime IS the conversation-critical-section lifetime (the old code
      released it at route-return, letting a second same-conv request run
      concurrently with the first's executor-thread engine session at
      permits=2: torn R["fed"] reads, mirror wipes on prefill failure)."""
      if released["v"]: return
      released["v"] = True
      cancel_flag["v"] = True          # arm engine cancel on any release path
      for t in (watcher,):
        try: t.cancel()
        except Exception: pass
      q_release()
      release_conv()
      _idemp_done(idem_key, conv_id)           # R3-39
    async def _slot_guard():
      """Backstop: if the SSE generator is never driven to completion (client
      vanished before first byte, ASGI mishap), the slot is still released
      once the bounded engine session ends (+ grace).
      L7.1: a CANCELLED guard used to `return` WITHOUT releasing — one leaked
      permit bricked admission (engine idle, QSTATE active=1 forever). The
      exactly-once flag makes the belt free."""
      try:
        await asyncio.shield(fut)
      except asyncio.CancelledError:
        release_slot()   # L7.1 belt: cancelled guard must not leak the slot
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
          if pend["dead"]:
            # R3-12 loud-fail backstop: the coalesced puts timed out twice —
            # the consumer is gone. Explicit SSE error + end; NEVER a silent
            # hole (the engine cancel was armed by on_dead).
            yield _sse({"error": {"message": "stream backpressure: client not consuming; "
                                            "generation cancelled",
                                  "type": "api_error", "param": None,
                                  "code": "stream_backpressure"}})
            yield "data: [DONE]\n\n"
            log(f"stream error {rid}: backpressure (queue full after coalescing)")
            gen_completed["v"] = True
            return
          if fut.done():
            # M1-C: drain any text events still queued behind fut's completion
            # (the 20ms poll can lag the threadpool thread's last on_event calls;
            # dropping them truncated stream tails nondeterministically).
            while True:
              try: kind2, p2 = evq.get_nowait()
              except _pyqueue.Empty: break
              if kind2 == "text": yield chunk({"content": p2})
              elif kind2 == "reasoning": yield chunk({"reasoning_content": p2})
            # R3-12: flush a coalesced-but-unqueued tail (a final Full left
            # text in pending with no further engine event to retry the put)
            if pend["reasoning"]:
              yield chunk({"reasoning_content": pend["reasoning"]}); pend["reasoning"] = ""
            if pend["text"]:
              yield chunk({"content": pend["text"]}); pend["text"] = ""
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
                                "type": "api_error", "param": None, "code": "engine_error"}})
          yield "data: [DONE]\n\n"
          log(f"stream error {rid}: {str(e)[:200]}")
          return
        except asyncio.CancelledError:
          raise
        except Exception as e:
          yield _sse({"error": {"message": f"internal error: {repr(e)[:300]}",
                                "type": "server_error", "param": None, "code": "internal_error"}})
          yield "data: [DONE]\n\n"
          return
        if res.get("reasoning_tail"):   # R3-35: before the finish chunk
          yield chunk({"reasoning_content": res["reasoning_tail"]})
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
    sseh = {"Cache-Control": "no-cache", "x-conversation-id": conv_id or "",
            "X-Accel-Buffering": "no"}
    if conv_id and _RECENTLY_EVICTED.pop(conv_id, None) is not None:
      sseh["x-resident-evicted"] = "1"          # R3-14
    if fresh_reason:
      sseh["x-prefix-fresh-reason"] = fresh_reason     # R3-43
    return StreamingResponse(gen(), media_type="text/event-stream", headers=sseh)

  finally:
    # R3-08: streaming responses hand BOTH the slot and the conv lock to the
    # stream lifecycle (release_slot); non-stream + early returns release here.
    if not getattr(request.state, "stream_release", None):
      release_conv()

@app.post("/v1/chat/completions")
async def chat_completions_route(request: Request):
  try:
    return await _chat(request)
  except QueueFull:
    resp = oai_error(429, f"engine queue full ({QSTATE['permits']} active + {MAX_WAITING} waiting); retry shortly",
                     err_type="rate_limit_error", code="queue_full")
    resp.headers["Retry-After"] = "10"
    return resp
  except EngineError as e:
    return oai_error(503, f"engine error: {str(e)[:200]}",
                     err_type="api_error", code="engine_error")
  except Exception as e:
    import traceback; traceback.print_exc()
    return oai_error(500, f"internal error: {str(e)[:200]}",
                     err_type="server_error", code="internal_error")
  finally:
    # V-01: for STREAMING responses the slot belongs to the SSE generator (the
    # route's finally runs before the generator starts); release_slot() is
    # exactly-once and shared between gen()'s finally and the guard task.
    if getattr(request.state, "slot_held", False) and not getattr(request.state, "stream_release", None):
      q_release()

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host=BIND, port=PORT, log_level="info", access_log=False)
