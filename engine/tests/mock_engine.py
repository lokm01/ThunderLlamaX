# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1.0 mock engine daemon + harness utilities (GPU-free, pure stdlib, py3.9+).

Implements the REAL serve.py wire protocol so the API server can be tested
end-to-end without the engine: newline-delimited JSON over a unix socket.

Protocol fidelity (mirrors engine0/serve.py WITH the W1+W2 fixes):
  - methods: status / prefill / generate / cancel / snapshot_save /
    snapshot_load / shutdown.  Unknown -> {"ok": false, "error": ...}.
  - status: full field set the API consumes for _decide_prefix:
    ready, ctxk, busy, rpc, pos, mode, cur, fed_len, fed_tail (last 16),
    conversation_id, queue, dirty, uptime_s + the W2 telemetry: config_fp,
    lookup_k, pf_prefill, cycles_since_rebuild.
  - cancel is a SIDE-CHANNEL FLAG: no ack frame is ever written (serve.py:488).
  - prefill FRESH/AUTO_CACHE: fed=ids, pos=len(ids); AUTO_CACHE may return
    mode CACHE_HIT + cached_tokens.  FOLLOW_UP: fed = fed + [cur] + ids,
    pos advances by 1 + len(ids) (serve.py:239).
  - FOLLOW_UP conversation guard (W1 fix): mismatched conversation_id is
    refused with a clean error echoing the resident id.
  - generate: per-cycle {"event": "cycle", "cycle", "pos", "tokens"} records;
    terminal "done" ALWAYS sent (stop or max cycles), "cancelled" carries the
    tokens so far; a faulting generate replies {"ok": false, "error": ...}
    (exactly what serve.py handle() does) and sets dirty.
  - prefill_progress events during prefill.
  - W2 (V-23/V-24 mirror): input caps — max_cycles clamped to [1, 4096],
    ids must be non-empty in-vocab ints within ctxk, stop ids in vocab;
    clean {"ok": false} replies that NEVER mark dirty.  Admin methods
    (shutdown/snapshot_*) fail CLOSED unless cfg admin_token matches
    params.admin_token.  Oversize lines (no newline past cfg max_line_bytes)
    drop the connection (counter: dropped_conns).

Also provides: build_synth_gguf() - a tiny synthetic GGUF (256 single-byte
normal tokens + 4 Qwen specials) so api_server's vendored tokenizer/template
load path runs for real, and ASGIClient - a minimal ASGI test client
(no httpx needed).
"""
import os, sys, json, time, socket, struct, threading, tempfile, asyncio, queue

# ============================ synthetic GGUF ==================================
# Byte<->token-id mapping EXACTLY as SimpleTokenizer builds it from the vocab:
# normal tokens are the 256 bytes mapped through the GPT-2 byte encoder.
_BS = [*range(33, 127), *range(161, 173), *range(174, 256)]
_BYTE_DECODER = {chr(b): b for b in _BS} | \
    {chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in _BS)}
# token STRING for byte b (inverse of _BYTE_DECODER)
_BYTE_TOKEN = {b: s for s, b in _BYTE_DECODER.items()}

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
ID_IMSTART, ID_IMEND, ID_THINK_OPEN, ID_THINK_CLOSE = 256, 257, 258, 259
NORM_IDS = 256                       # ids 0..255 = bytes; 256..259 = specials
NVOCAB = NORM_IDS + len(SPECIALS)

# Synthetic chat template: faithful subset of the real Qwen3.8 template's
# decision points (xhigh default effort marker, open vs pre-closed <think>).
CHAT_TEMPLATE = (
    "{%- set ri = '' %}\n"
    "{%- if enable_thinking is undefined or enable_thinking is true %}\n"
    "{%- set resolved = reasoning_effort|default('xhigh') %}\n"
    "{%- if resolved == 'xhigh' %}{%- set ri = 'Reasoning effort is set to xhigh. Please think carefully through the task.' %}{%- endif %}\n"
    "{%- endif %}\n"
    "{%- if ri %}{{ '<|im_start|>system\n' + ri + '<|im_end|>\n' }}{%- endif %}\n"
    "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\n' + (m.content if m.content is string else '') + '<|im_end|>\n' }}{%- endfor %}\n"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}"
    "{%- else %}{{ '<think>\n' }}{%- endif %}{%- endif %}"
)

def _w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b))); f.write(b)

def _w_val(f, t, v):
    if t == 4: f.write(struct.pack("<I", v))
    elif t == 7: f.write(struct.pack("<B", 1 if v else 0))
    elif t == 8: _w_str(f, v)
    elif t == 9:
        et = v[0]
        f.write(struct.pack("<I", et)); f.write(struct.pack("<Q", len(v[1])))
        for item in v[1]: _w_val(f, et, item)
    else: raise ValueError(t)

def build_synth_gguf(path, eos_id=ID_IMEND, eot_id=ID_IMEND, template=CHAT_TEMPLATE):
    """Write a minimal GGUF whose KV section feeds api_server.load_tokenizer()."""
    tokens = [_BYTE_TOKEN[b] for b in range(256)] + SPECIALS
    types = [1] * 256 + [2] * len(SPECIALS)
    kv = [
        ("general.architecture", 8, "qwen3"),
        ("tokenizer.ggml.tokens", 9, (8, tokens)),
        ("tokenizer.ggml.token_type", 9, (4, types)),
        ("tokenizer.ggml.pre", 8, "qwen2"),
        ("tokenizer.ggml.add_bos_token", 7, False),
        ("tokenizer.ggml.eos_token_id", 4, int(eos_id)),
        ("tokenizer.ggml.eot_token_id", 4, int(eot_id)),
        ("tokenizer.chat_template", 8, template),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF"); f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))                  # n_tensors
        f.write(struct.pack("<Q", len(kv)))
        for name, t, v in kv:
            _w_str(f, name); f.write(struct.pack("<I", t)); _w_val(f, t, v)
    return path

# ---------------------------- mock-side codec --------------------------------
_SPECIAL_IDS = {s: NORM_IDS + i for i, s in enumerate(SPECIALS)}
_IDS_SPECIAL = {v: k for k, v in _SPECIAL_IDS.items()}

def mock_encode(text):
    """Token ids for text (longest-special-first, then single bytes).
    Normal vocab is one token per byte, so a char c (ord<256) encodes to ord(c)."""
    ids, i = [], 0
    while i < len(text):
        for s in sorted(SPECIALS, key=len, reverse=True):
            if text.startswith(s, i):
                ids.append(_SPECIAL_IDS[s]); i += len(s); break
        else:
            o = ord(text[i])
            ids.append(o if o < 256 else ord("?"))
            i += 1
    return ids

def mock_decode(ids):
    out = []
    for t in ids:
        t = int(t)
        if t in _IDS_SPECIAL: out.append(_IDS_SPECIAL[t])
        elif 0 <= t < 256: out.append(chr(t))
    return "".join(out)

# ============================ mock engine daemon ==============================
class MockEngine:
  """Threaded unix-socket daemon speaking the serve.py protocol (fixed form)."""

  def __init__(self, sock_path, cfg=None):
    self.sock_path = sock_path
    self.cfg = {
        "ctxk": 100352, "cycle_delay": 0.01, "cycle_width": 3,
        "reply_text": "The mock engine answer. ", "reply_tokens": None,
        "fail_prefill": False, "fail_at_cycle": None, "fail_prefill_times": 1,
        "wedge_generate": False, "auto_cache_hit": 0, "stale_fed_len": 0,
        "initial_pos": 0,
        # W2 fidelity knobs
        "admin_token": "",            # "" = admin methods fail CLOSED (serve default)
        "max_line_bytes": 4 * 1024 * 1024,   # V-31 mirror
        "config_fp": None,            # set by tests (api.EXPECTED_FP by default)
        "lookup_k": "10", "pf_prefill": "1",
    }
    if cfg: self.cfg.update(cfg)
    self.lock = threading.Lock()
    self.fed = []
    self.conversation_id = None
    self.pos = int(self.cfg["initial_pos"])
    self.cur = 0
    self.mode = None
    self.busy = False
    self.rpc = None
    self.cancel_flag = False
    self.dirty = False
    self.ready = True
    self.cycles_since_rebuild = 0
    self.dropped_conns = 0           # V-31 line-cap drops (assert target)
    self.admin_denied = []           # (method) ACL refusals (assert target)
    self.rejected = []               # (method, error) validation refusals
    self.t0 = time.time()
    self.rpc_log = []          # (t, method, note) timeline for assertions
    self.prefill_params = []    # full prefill params (W3: cache_key/ttl e2e)
    self.gen_windows = []      # (t_start, t_end) per completed generate
    self.gen_active = 0
    self._fail_prefill_used = 0
    self._srv = None
    self._stop = threading.Event()
    self.q = queue.Queue()     # REAL daemon shape: RPCs serialized on one
                               # worker; cancel/status are side-channels the
                               # per-conn readers honor even mid-generate
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()
    self._thread = threading.Thread(target=self._listen, daemon=True)
    self._thread.start()
    # wait until listening
    deadline = time.time() + 5
    while self._srv is None and time.time() < deadline: time.sleep(0.01)
    if self._srv is None: raise RuntimeError("mock engine failed to start")

  def _worker(self):
    while not self._stop.is_set():
      try:
        conn, req = self.q.get(timeout=0.2)
      except queue.Empty:
        continue
      self._handle(conn, req)

  # ---------------- lifecycle ----------------
  def _listen(self):
    try:
      if os.path.exists(self.sock_path): os.unlink(self.sock_path)
      srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      srv.bind(self.sock_path); srv.listen(16); srv.settimeout(0.2)
      self._srv = srv
      while not self._stop.is_set():
        try:
          conn, _ = srv.accept()
        except socket.timeout:
          continue
        except OSError:
          break
        threading.Thread(target=self._conn_thread, args=(conn,), daemon=True).start()
    finally:
      try: srv.close()
      except Exception: pass

  def stop(self):
    self._stop.set()
    try:
      if self._srv: self._srv.close()
    except Exception: pass
    try:
      if os.path.exists(self.sock_path): os.unlink(self.sock_path)
    except Exception: pass

  def _conn_thread(self, conn):
    buf = b""
    conn.settimeout(None)
    try:
      while not self._stop.is_set():
        chunk = conn.recv(65536)
        if not chunk: break
        buf += chunk
        # V-31 mirror: a no-newline stream past the cap drops the conn
        if len(buf) > int(self.cfg["max_line_bytes"]):
          with self.lock:
            self.dropped_conns += 1
          break
        while b"\n" in buf:
          line, buf = buf.split(b"\n", 1)
          if not line.strip(): continue
          if len(line) > int(self.cfg["max_line_bytes"]):
            with self.lock:
              self.dropped_conns += 1
            return
          try: req = json.loads(line)
          except Exception: continue
          m = req.get("method")
          if m == "cancel":
            # REAL PROTOCOL: side-channel flag, NO ack frame (serve.py:488) —
            # honored by the reader even while a generate runs on this conn
            with self.lock:
              self.cancel_flag = True
              self.rpc_log.append((time.time(), "cancel", None))
            continue
          if m == "status" and self.ready:
            # lock-free inline status (listener-thread answered, serve.py:492)
            try:
              conn.sendall((json.dumps({"id": req.get("id"), "ok": True,
                  "result": self._status_locked()}) + "\n").encode())
            except Exception: pass
            continue
          self.q.put((conn, req))
    except Exception:
      pass
    finally:
      try: conn.close()
      except Exception: pass

  def _send(self, conn, obj):
    try: conn.sendall((json.dumps(obj) + "\n").encode())
    except Exception: pass

  def _status_locked(self):
    fed_len = len(self.fed) - int(self.cfg["stale_fed_len"])
    return {"ready": self.ready, "ctxk": self.cfg["ctxk"], "busy": self.busy,
            "rpc": self.rpc, "pos": self.pos, "mode": self.mode, "cur": self.cur,
            "fed_len": max(0, fed_len), "fed_tail": self.fed[-16:],
            "conversation_id": self.conversation_id, "queue": 0,
            "dirty": self.dirty, "uptime_s": round(time.time() - self.t0, 1),
            "config_fp": self.cfg.get("config_fp"),
            "lookup_k": str(self.cfg.get("lookup_k")),
            "pf_prefill": str(self.cfg.get("pf_prefill")),
            "cycles_since_rebuild": self.cycles_since_rebuild}

  # ---------------- W2 mirrors of the serve.py RPC gates -----------------------
  def _check_admin(self, p):
    """serve.check_admin: fail CLOSED when no token configured."""
    tok = str((p or {}).get("admin_token") or "")
    want = str(self.cfg.get("admin_token") or "")
    import hmac as _hmac
    return bool(want) and _hmac.compare_digest(tok, want)

  def _validate_rpc(self, m, p):
    """serve.validate_rpc mirror — returns (err, params')."""
    p = p or {}
    if m == "generate":
      mc = p.get("max_cycles", 60)
      if not isinstance(mc, int) or isinstance(mc, bool) or mc < 1:
        return f"max_cycles must be an integer >= 1 (got {mc!r})", None
      cap = max(1, min(4096, self.cfg["ctxk"] - max(0, self.pos)))
      if mc > cap: mc = cap
      for t in p.get("stop_token_ids", []) or []:
        if not isinstance(t, int) or isinstance(t, bool) or not (0 <= t < NVOCAB):
          return f"stop_token_id {t!r} outside vocab range 0..{NVOCAB-1}", None
      p = dict(p); p["max_cycles"] = mc
      return None, p
    if m == "prefill":
      if "snapshot" in p: return None, p
      mode = p.get("mode", "FRESH")
      if mode not in ("FRESH", "FOLLOW_UP", "AUTO_CACHE", "PF_BATCH"):
        return f"prefill mode {mode!r} not recognized", None
      ids = p.get("ids")
      if not isinstance(ids, list) or not ids:
        return "prefill ids must be a non-empty list of ints", None
      for t in ids:
        if not isinstance(t, int) or isinstance(t, bool) or not (0 <= t < NVOCAB):
          return f"prefill id {t!r} outside vocab range 0..{NVOCAB-1}", None
      if mode == "FOLLOW_UP":
        if max(0, self.pos) + 1 + len(ids) > self.cfg["ctxk"]:
          return (f"FOLLOW_UP delta overflows context: pos {self.pos} + 1 + "
                  f"{len(ids)} > ctxk {self.cfg['ctxk']}"), None
      elif len(ids) > self.cfg["ctxk"]:
        return f"prefill ids length {len(ids)} exceeds ctxk {self.cfg['ctxk']}", None
      return None, p
    return None, p

  # ---------------- handlers (serialize GPU-class RPCs per conn) ----------------
  def _handle(self, conn, req):
    rid = req.get("id"); m = req.get("method"); p = req.get("params") or {}
    # V-24 mirror: privileged methods fail CLOSED without the token; refused
    # requests never touch state (no dirty marking)
    if m in ("shutdown", "snapshot_save", "snapshot_load") or \
       (m == "prefill" and "snapshot" in p):
      if not self._check_admin(p):
        with self.lock:
          self.admin_denied.append(m)
          self.rpc_log.append((time.time(), m, "admin_denied"))
        self._send(conn, {"id": rid, "ok": False, "error": "admin required"})
        return
    if m in ("generate", "prefill"):
      err, p2 = self._validate_rpc(m, p)
      if err is not None:
        with self.lock:
          self.rejected.append((m, err))
          self.rpc_log.append((time.time(), m, f"rejected:{err}"))
        self._send(conn, {"id": rid, "ok": False, "error": f"invalid params: {err}"})
        return
      p = p2
    gpu_rpc = m in ("generate", "prefill", "snapshot_save", "snapshot_load")
    if gpu_rpc:
      with self.lock: self.busy = True; self.rpc = m
    try:
      if m == "status":
        r = self._status_locked()
      elif m == "prefill":
        r = self._h_prefill(conn, rid, p)
      elif m == "generate":
        r = self._h_generate(conn, rid, p)
      elif m == "snapshot_save":
        r = {"path": p.get("path"), "pos": self.pos, "cur": self.cur,
             "delta_rows": [0, 0]}
      elif m == "snapshot_load":
        self.mode = "snapshot_load"
        with self.lock: self.dirty = True   # epoch bump: API must FRESH (V-35 class)
        r = {"pos": self.pos, "cur": self.cur, "delta_rows": [0, 0]}
      elif m == "shutdown":
        self._send(conn, {"id": rid, "ok": True, "result": {"bye": True}})
        self.rpc_log.append((time.time(), "shutdown", None))
        self._stop.set()
        return
      else:
        raise ValueError(f"unknown method {m}")
      self._send(conn, {"id": rid, "ok": True, "result": r})
    except Exception as e:
      # serve.py handle(): clean error reply; engine-mutating failures mark dirty
      with self.lock:
        if m in ("prefill", "generate", "snapshot_load"): self.dirty = True
        self.rpc_log.append((time.time(), m, f"error:{e}"))
      self._send(conn, {"id": rid, "ok": False, "error": str(e)})
    finally:
      if gpu_rpc:
        with self.lock: self.busy = False; self.rpc = None

  def _h_prefill(self, conn, rid, p):
    mode = p.get("mode", "FRESH")
    ids = [int(t) for t in p.get("ids", [])]
    if not ids: raise ValueError("prefill: empty ids")
    with self.lock:
      self.prefill_params.append(dict(p))
      self.rpc_log.append((time.time(), "prefill", {"mode": mode, "n": len(ids),
                                                    "cid": p.get("conversation_id")}))
      if self.cfg["fail_prefill"] and self._fail_prefill_used < self.cfg["fail_prefill_times"]:
        self._fail_prefill_used += 1
        raise RuntimeError("mock prefill failure (scripted)")
      if mode == "FOLLOW_UP":
        # W1 fix (engine side): refuse a FOLLOW_UP for a foreign conversation
        req_cid = p.get("conversation_id")
        if req_cid != self.conversation_id:
          raise ValueError(f"FOLLOW_UP conversation_id mismatch: engine resident="
                           f"{self.conversation_id!r} request={req_cid!r} "
                           f"fed_len={len(self.fed)}")
        cur = int(p["cur"]) if p.get("cur") is not None else self.cur
        newcur = ids[-1] if ids else cur
        self._prog(conn, rid, 0, 2, "prefill_t1")
        time.sleep(self.cfg["cycle_delay"])
        self._prog(conn, rid, 2, 2, "prefill_t1")
        self.fed = self.fed + [cur] + ids           # serve.py:239
        self.pos = self.pos + 1 + len(ids)
        self.cur = newcur; self.mode = "FOLLOW_UP"
        self.conversation_id = p.get("conversation_id")
        self.dirty = False
        return {"pos": self.pos, "cur": newcur, "fed": len(ids)}
      # FRESH / AUTO_CACHE
      hit = int(self.cfg["auto_cache_hit"])
      if mode == "AUTO_CACHE" and 0 < hit < len(ids):
        self._prog(conn, rid, 0, 2, "restore")
        time.sleep(self.cfg["cycle_delay"])
        self._prog(conn, rid, 1, 2, "prefill_t1")
        self.fed = list(ids[:hit])
        self.pos = hit; self.cur = ids[hit - 1]
        # tail re-prefill
        tail = ids[hit:]
        self.fed = list(ids)
        self.pos = len(ids); self.cur = ids[-1]
        self.mode = "CACHE_HIT"; self.conversation_id = p.get("conversation_id")
        self.dirty = False
        self._prog(conn, rid, 2, 2, "prefill_t1")
        return {"pos": self.pos, "cur": self.cur, "fed": len(ids),
                "mode": "CACHE_HIT", "cached_tokens": hit}
      self._prog(conn, rid, 0, 2, "prefill_t1")
      time.sleep(self.cfg["cycle_delay"])
      self._prog(conn, rid, 2, 2, "prefill_t1")
      self.fed = list(ids)
      self.pos = len(ids); self.cur = ids[-1]
      self.mode = "FRESH"; self.conversation_id = p.get("conversation_id")
      self.dirty = False
      return {"pos": self.pos, "cur": self.cur, "fed": len(ids),
              "mode": "FRESH", "cached_tokens": 0}

  def _prog(self, conn, rid, done, total, stage):
    self._send(conn, {"id": rid, "event": "prefill_progress", "stage": stage,
                      "done": done, "total": total})

  def _reply_token_stream(self):
    if self.cfg["reply_tokens"] is not None:
      return list(self.cfg["reply_tokens"])
    return mock_encode(self.cfg["reply_text"])

  def _h_generate(self, conn, rid, p):
    with self.lock:
      self.rpc_log.append((time.time(), "generate", {"max_cycles": p.get("max_cycles")}))
    if self.cfg["wedge_generate"]:
      time.sleep(600)      # never answers (wedge fault)
      return {"tokens": [], "cycles": 0}
    mc = int(p.get("max_cycles", 60))
    stops = set(int(t) for t in p.get("stop_token_ids", []))
    stream = self._reply_token_stream()
    width = max(1, int(self.cfg["cycle_width"]))
    batches = [stream[i:i + width] for i in range(0, len(stream), width)] or [[ID_IMEND]]
    t_start = time.time()
    with self.lock:
      self.busy = True; self.gen_active += 1
      self.cancel_flag = False     # serve.py:306: cleared at generate entry
      pos_base = self.pos
    all_toks = []
    try:
      for k in range(1, mc + 1):
        with self.lock:
          cancelled = self.cancel_flag
          self.cycles_since_rebuild += 1   # serve mirror: global across generates
        if cancelled:
          self._send(conn, {"id": rid, "event": "cancelled", "tokens": all_toks,
                            "cycles": k - 1})
          with self.lock:
            self.fed = self.fed + all_toks
            self.rpc_log.append((time.time(), "generate", "cancelled"))
          return {"cancelled": True, "tokens": all_toks, "cycles": k - 1}
        batch = batches[(k - 1) % len(batches)]
        all_toks += batch
        self._send(conn, {"id": rid, "event": "cycle", "cycle": k,
                          "pos": pos_base + len(all_toks), "tokens": batch})
        with self.lock:
          self.rpc_log.append((time.time(), "cycle", {"k": k, "n": len(batch)}))
        if self.cfg["fail_at_cycle"] == k:
          # serve.py handle() on a faulting sess.step(): error reply + dirty.
          # (The W1-fixed daemon also appends known-good tokens to fed first.)
          with self.lock:
            self.fed = self.fed + all_toks
            self.dirty = True
          raise RuntimeError(f"mock mid-generate fault at cycle {k}")
        time.sleep(self.cfg["cycle_delay"])
        if stops & set(batch):
          with self.lock:
            self.fed = self.fed + all_toks
          self._send(conn, {"id": rid, "event": "done", "tokens": all_toks,
                            "cycles": k, "pos": pos_base + len(all_toks), "stop": True,
                            "usage": {"tokens": len(all_toks), "cycles": k}})
          self.rpc_log.append((time.time(), "generate", f"done_stop@{k}"))
          return {"tokens": all_toks, "cycles": k, "stop": True}
      with self.lock:
        self.fed = self.fed + all_toks
      self._send(conn, {"id": rid, "event": "done", "tokens": all_toks, "cycles": mc,
                        "pos": pos_base + len(all_toks),
                        "usage": {"tokens": len(all_toks), "cycles": mc}})
      self.rpc_log.append((time.time(), "generate", f"done_max@{mc}"))
      return {"tokens": all_toks, "cycles": mc}
    finally:
      with self.lock:
        self.busy = False; self.gen_active -= 1
        self.pos = pos_base + len(all_toks)
        self.gen_windows.append((t_start, time.time()))

  # ---------------- assertions helpers ----------------
  def methods(self, name, note_substr=None):
    out = []
    for t, m, note in self.rpc_log:
      if m == name and (note_substr is None or note_substr in str(note)):
        out.append((t, note))
    return out

  def generates_overlap(self):
    w = sorted(self.gen_windows)
    for i in range(1, len(w)):
      if w[i][0] < w[i - 1][1] - 1e-6: return True
    return False

# ============================ minimal ASGI client =============================
class ASGIResponse:
  def __init__(self):
    self.status = None; self.raw_headers = []; self.chunks = []; self.done = False
  @property
  def headers(self):
    return {k.decode().lower(): v.decode() for k, v in self.raw_headers}
  @property
  def body(self): return b"".join(self.chunks)
  def json(self): return json.loads(self.body)
  def sse_events(self):
    """Parsed data: frames [(obj|"[DONE]")] plus comment lines."""
    evs, comments = [], []
    for part in self.chunks:
      text = part.decode("utf-8", "replace")
      for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame: continue
        if frame.startswith(":"):
          comments.append(frame); continue
        if frame.startswith("data: "):
          payload = frame[6:]
          evs.append("[DONE]" if payload.strip() == "[DONE]" else json.loads(payload))
    return evs, comments

async def asgi_request(app, method, path, json_body=None, raw_body=None,
                       headers=None, disconnect_after=None, stall_body=False,
                       drop_early=None, spec_version="2.4"):
  """Drive the ASGI app directly. Returns ASGIResponse.

  spec_version 2.4 = Starlette's SIMPLE streaming path (no built-in
  disconnect listener) — exercises the API's OWN watcher machinery, the
  production M1-C assumption.  Pass "2.3" to also run Starlette's
  listen_for_disconnect task-group path.
  disconnect_after: seconds after body consumed -> enqueue http.disconnect.
  stall_body: send only the first byte (more_body=True) and never the rest.
  drop_early: make send() raise once after N body chunks (write failure).
  """
  scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
           "http_version": "1.1", "method": method, "scheme": "http",
           "path": path, "raw_path": path.encode(), "query_string": b"",
           "root_path": "", "headers": [], "client": ("127.0.0.1", 42000),
           "server": ("127.0.0.1", 80)}
  # W2: TrustedHostMiddleware requires a Host header on every request; a
  # caller-provided host REPLACES the default (starlette reads the FIRST one)
  for k, v in (headers or {}).items():
    scope["headers"].append((k.lower().encode(), v.encode()))
  if not any(k == b"host" for k, _ in scope["headers"]):
    scope["headers"].append((b"host", b"127.0.0.1"))
  recv_q = asyncio.Queue()
  if json_body is not None:
    raw = json.dumps(json_body).encode()
    scope["headers"].append((b"content-type", b"application/json"))
    scope["headers"].append((b"content-length", str(len(raw)).encode()))
  else:
    raw = raw_body if raw_body is not None else b""
  if stall_body:
    recv_q.put_nowait({"type": "http.request", "body": raw[:1], "more_body": True})
  elif raw:
    recv_q.put_nowait({"type": "http.request", "body": raw, "more_body": False})
  else:
    recv_q.put_nowait({"type": "http.request", "body": b"", "more_body": False})

  async def receive():
    return await recv_q.get()

  if disconnect_after is not None:
    async def _disc():
      await asyncio.sleep(disconnect_after)
      recv_q.put_nowait({"type": "http.disconnect"})
    asyncio.get_running_loop().create_task(_disc())

  resp = ASGIResponse()
  sent_bodies = {"n": 0}

  async def send(msg):
    if msg["type"] == "http.response.start":
      resp.status = msg["status"]; resp.raw_headers = msg.get("headers", [])
    elif msg["type"] == "http.response.body":
      sent_bodies["n"] += 1
      if drop_early is not None and sent_bodies["n"] > drop_early:
        raise RuntimeError("simulated client write failure (disconnect)")
      if msg.get("body"): resp.chunks.append(msg["body"])
      if not msg.get("more_body"): resp.done = True

  await app(scope, receive, send)
  return resp


# ======================= R6 PHASE 3: the batch mock ============================
class MockBatchEngine(MockEngine):
  """Wire-protocol mirror of serve.py _batch_main (BATCH_B=2):
    - status carries batch_b=2 + per-slot `streams` (conversation_id, pos,
      cur, fed_len, mode, generating, dirty); legacy top-level fields mirror
      slot 0 (old-API compat).
    - prefill binds conn->slot (FOLLOW_UP needs the convo resident; FRESH
      picks convo-pinned -> pristine -> LRU idle) and REPLIES when done.
    - generate ATTACHES (no reply until the terminal); the INVERTED worker
      loop ticks every active stream one cycle per iteration (true
      interleave), honoring per-stream cancel and stop/max terminals with
      done/cancelled events + the deferred result frame.
    - cancel is a per-conn side-channel (no ack); conn EOF unbinds.
    - snapshot ops rejected while slot 1 in use or any stream generating.
  """
  def __init__(self, sock_path, cfg=None):
    cfg = dict(cfg or {})
    cfg.setdefault("cycle_delay", 0.005)
    # NB: the parent __init__ STARTS the worker thread — slot state must exist
    # before super().__init__ runs (the overridden _worker reads self.slots).
    self.slots = [self._new_slot(0), self._new_slot(1)]
    self.by_conn = {}
    super().__init__(sock_path, cfg)

  def _new_slot(self, s):
    return {"s": s, "fed": [], "conversation_id": None, "pos": 0, "cur": 0,
            "mode": None, "dirty": False, "used": False, "gen": None}

  def _active(self):
    return any(sl["gen"] is not None for sl in self.slots)

  # ---- the inverted scheduler loop ----
  def _worker(self):
    while not self._stop.is_set():
      active = self._active()
      try:
        conn, req = self.q.get(timeout=0.002 if active else 0.2)
        self._handle(conn, req)
        continue
      except queue.Empty:
        pass
      if self._active():
        self._tick()

  # ---- per-conn reader: per-stream cancel routing + unbind at EOF ----
  def _conn_thread(self, conn):
    buf = b""
    conn.settimeout(None)
    try:
      while not self._stop.is_set():
        chunk = conn.recv(65536)
        if not chunk: break
        buf += chunk
        while b"\n" in buf:
          line, buf = buf.split(b"\n", 1)
          if not line.strip(): continue
          try: req = json.loads(line)
          except Exception: continue
          m = req.get("method")
          if m == "cancel":
            with self.lock:
              s = self.by_conn.get(conn)
              if s is not None and self.slots[s]["gen"] is not None:
                self.slots[s]["gen"]["cancel"] = True
              self.cancel_flag = True       # prefill-phase cancels (simplify)
              self.rpc_log.append((time.time(), "cancel", {"slot": s}))
            continue
          if m == "status" and self.ready:
            try:
              conn.sendall((json.dumps({"id": req.get("id"), "ok": True,
                  "result": self._status_locked()}) + "\n").encode())
            except Exception: pass
            continue
          self.q.put((conn, req))
    except Exception:
      pass
    finally:
      with self.lock:
        s = self.by_conn.pop(conn, None)
        if s is not None:
          sl = self.slots[s]
          if sl["gen"] is not None: sl["gen"]["cancel"] = True   # dead client
      try: conn.close()
      except Exception: pass

  def _status_locked(self):
    s0 = self.slots[0]
    return {"ready": self.ready, "ctxk": self.cfg["ctxk"],
            "busy": self._active(), "rpc": "generate" if self._active() else None,
            "pos": s0["pos"], "mode": s0["mode"], "cur": s0["cur"],
            "fed_len": len(s0["fed"]) - int(self.cfg["stale_fed_len"]),
            "fed_tail": s0["fed"][-16:], "conversation_id": s0["conversation_id"],
            "queue": 0, "dirty": s0["dirty"], "uptime_s": round(time.time() - self.t0, 1),
            "config_fp": self.cfg.get("config_fp"),
            "lookup_k": str(self.cfg.get("lookup_k")),
            "pf_prefill": str(self.cfg.get("pf_prefill")),
            "cycles_since_rebuild": self.cycles_since_rebuild,
            "batch_b": 2,
            "streams": [{"slot": sl["s"], "conversation_id": sl["conversation_id"],
                         "pos": sl["pos"], "cur": sl["cur"], "fed_len": len(sl["fed"]),
                         "mode": sl["mode"], "generating": sl["gen"] is not None,
                         "dirty": sl["dirty"], "used": sl["used"]} for sl in self.slots]}

  # ---- batch handlers ----
  def _handle(self, conn, req):
    rid = req.get("id"); m = req.get("method"); p = req.get("params") or {}
    if m in ("shutdown", "snapshot_save", "snapshot_load") or (m == "prefill" and "snapshot" in p):
      if not self._check_admin(p):
        with self.lock:
          self.admin_denied.append(m)
          self.rpc_log.append((time.time(), m, "admin_denied"))
        self._send(conn, {"id": rid, "ok": False, "error": "admin required"})
        return
    if m in ("generate", "prefill"):
      err, p2 = self._validate_rpc(m, p)
      if err is not None:
        with self.lock:
          self.rejected.append((m, err))
          self.rpc_log.append((time.time(), m, f"rejected:{err}"))
        self._send(conn, {"id": rid, "ok": False, "error": f"invalid params: {err}"})
        return
      p = p2
    try:
      if m == "status":
        self._send(conn, {"id": rid, "ok": True, "result": self._status_locked()})
      elif m == "prefill":
        if "snapshot" in p:
          if self.slots[1]["used"] or self._active():
            self._send(conn, {"id": rid, "ok": False, "error": "snapshot prefill rejected: batch slots in use"})
            self.rpc_log.append((time.time(), "prefill_snapshot", "rejected_batch"))
            return
          self._send(conn, {"id": rid, "ok": True, "result": {"pos": self.slots[0]["pos"], "cur": self.slots[0]["cur"], "fed": 0}})
          return
        self._b_prefill(conn, rid, p)
      elif m == "generate":
        self._b_generate(conn, rid, p)
      elif m == "shutdown":
        self._send(conn, {"id": rid, "ok": True, "result": {"bye": True}})
        self.rpc_log.append((time.time(), "shutdown", None))
        self._stop.set()
        return
      else:
        raise ValueError(f"unknown method {m}")
    except Exception as e:
      with self.lock:
        self.rpc_log.append((time.time(), m, f"error:{e}"))
      self._send(conn, {"id": rid, "ok": False, "error": str(e)})

  def _pick_slot(self, p):
    cid = p.get("conversation_id"); mode = p.get("mode", "FRESH")
    if mode == "FOLLOW_UP":
      for sl in self.slots:
        if sl["conversation_id"] == cid and sl["gen"] is None:
          return sl
      raise ValueError(f"FOLLOW_UP conversation_id mismatch: no resident conversation "
                       f"{cid!r} on any slot (evicted? dirty? use FRESH)")
    for sl in self.slots:
      if cid is not None and sl["conversation_id"] == cid and sl["gen"] is None:
        return sl
    for sl in self.slots:
      if not sl["used"] and sl["gen"] is None:
        return sl
    idle = [sl for sl in self.slots if sl["gen"] is None]
    if idle:
      return idle[0] if len(idle) == 1 else min(idle, key=lambda x: len(x["fed"]))
    raise ValueError("engine busy: all batch slots are generating")

  def _b_prefill(self, conn, rid, p):
    mode = p.get("mode", "FRESH")
    ids = [int(t) for t in p.get("ids", [])]
    if not ids: raise ValueError("prefill: empty ids")
    sl = self._pick_slot(p)
    with self.lock:
      self.prefill_params.append(dict(p))
      self.rpc_log.append((time.time(), "prefill", {"mode": mode, "n": len(ids),
                                                    "cid": p.get("conversation_id"), "slot": sl["s"]}))
      if self.cfg["fail_prefill"] and self._fail_prefill_used < self.cfg["fail_prefill_times"]:
        self._fail_prefill_used += 1
        raise RuntimeError("mock prefill failure (scripted)")
      sl["conversation_id"] = p.get("conversation_id")
      sl["used"] = True; sl["dirty"] = False; sl["gen"] = None
      self.by_conn[conn] = sl["s"]
      if mode == "FOLLOW_UP":
        cur = int(p["cur"]) if p.get("cur") is not None else sl["cur"]
        self._prog(conn, rid, 0, 2, "prefill_t1")
        time.sleep(self.cfg["cycle_delay"])
        self._prog(conn, rid, 2, 2, "prefill_t1")
        sl["fed"] = sl["fed"] + [cur] + ids
        sl["pos"] = sl["pos"] + 1 + len(ids)
        sl["cur"] = ids[-1] if ids else cur
        sl["mode"] = "FOLLOW_UP"
        r = {"pos": sl["pos"], "cur": sl["cur"], "fed": len(ids)}
      else:
        hit = int(self.cfg["auto_cache_hit"])
        if mode == "AUTO_CACHE" and 0 < hit < len(ids):
          self._prog(conn, rid, 0, 2, "restore")
          time.sleep(self.cfg["cycle_delay"])
          sl["fed"] = list(ids); sl["pos"] = len(ids); sl["cur"] = ids[-1]
          sl["mode"] = "CACHE_HIT"
          self._prog(conn, rid, 2, 2, "prefill_t1")
          r = {"pos": sl["pos"], "cur": sl["cur"], "fed": len(ids),
               "mode": "CACHE_HIT", "cached_tokens": hit}
        else:
          self._prog(conn, rid, 0, 2, "prefill_t1")
          time.sleep(self.cfg["cycle_delay"])
          self._prog(conn, rid, 2, 2, "prefill_t1")
          sl["fed"] = list(ids); sl["pos"] = len(ids); sl["cur"] = ids[-1]
          sl["mode"] = "FRESH"
          r = {"pos": sl["pos"], "cur": sl["cur"], "fed": len(ids),
               "mode": "FRESH", "cached_tokens": 0}
    self._send(conn, {"id": rid, "ok": True, "result": r})

  def _b_generate(self, conn, rid, p):
    s = self.by_conn.get(conn)
    if s is None:
      self._send(conn, {"id": rid, "ok": False, "error": "no prefill on this connection (send prefill first)"})
      return
    sl = self.slots[s]
    if sl["gen"] is not None:
      self._send(conn, {"id": rid, "ok": False, "error": "slot already generating"})
      return
    if sl["dirty"]:
      self._send(conn, {"id": rid, "ok": False, "error": "slot state dirty; re-prefill (FRESH) first"})
      return
    mc = int(p.get("max_cycles", 60))
    with self.lock:
      sl["gen"] = {"conn": conn, "rid": rid, "stops": set(int(t) for t in p.get("stop_token_ids", [])),
                   "mc": mc, "k": 0, "all_toks": [], "cancel": False,
                   "t0": time.time(), "pos_base": sl["pos"]}
      self.rpc_log.append((time.time(), "generate", {"attach": sl["conversation_id"], "slot": sl["s"]}))
      self.gen_active += 1
    # NO reply: the terminal sends done/cancelled + the result frame

  def _reply_batches(self, sl):
    by_slot = self.cfg.get("reply_tokens_by_slot")
    stream = list(by_slot[sl["s"]]) if by_slot else self._reply_token_stream()
    width = max(1, int(self.cfg["cycle_width"]))
    return [stream[i:i + width] for i in range(0, len(stream), width)] or [[ID_IMEND]]

  def _finish(self, sl, kind):
    g = sl["gen"]; sl["gen"] = None
    conn = g["conn"]
    sl["pos"] = g["pos_base"] + len(g["all_toks"])
    with self.lock:
      self.gen_active -= 1
      self.gen_windows.append((g["t0"], time.time()))
      self.rpc_log.append((time.time(), "generate", f"{kind}@{g['k']}"))
    if kind == "cancelled":
      self._send(conn, {"id": g["rid"], "event": "cancelled", "tokens": g["all_toks"], "cycles": g["k"]})
      self._send(conn, {"id": g["rid"], "ok": True, "result": {"cancelled": True, "tokens": g["all_toks"], "cycles": g["k"]}})
      return
    self._send(conn, {"id": g["rid"], "event": "done", "tokens": g["all_toks"], "cycles": g["k"],
                      "pos": sl["pos"], "stop": kind == "stop",
                      "usage": {"tokens": len(g["all_toks"]), "cycles": g["k"]}})
    res = {"tokens": g["all_toks"], "cycles": g["k"]}
    if kind == "stop": res["stop"] = True
    self._send(conn, {"id": g["rid"], "ok": True, "result": res})

  def _tick(self):
    for sl in self.slots:
      g = sl["gen"]
      if g is None: continue
      if g["cancel"]:
        self._finish(sl, "cancelled"); continue
      batch = self._reply_batches(sl)[(g["k"]) % len(self._reply_batches(sl))]
      g["k"] += 1; g["all_toks"] += batch
      sl["fed"] = sl["fed"] + batch
      self._send(g["conn"], {"id": g["rid"], "event": "cycle", "cycle": g["k"],
                             "pos": g["pos_base"] + len(g["all_toks"]), "tokens": batch})
      with self.lock:
        self.rpc_log.append((time.time(), "cycle", {"slot": sl["s"], "k": g["k"]}))
        self.cycles_since_rebuild += 1
      if self.cfg.get("fail_at_cycle") == g["k"]:
        with self.lock:
          sl["dirty"] = True
        self._send(g["conn"], {"id": g["rid"], "ok": False, "error": f"mock mid-generate fault at cycle {g['k']}"})
        sl["gen"] = None
        with self.lock:
          self.gen_active -= 1
          self.gen_windows.append((g["t0"], time.time()))
        continue
      time.sleep(self.cfg["cycle_delay"])
      if g["stops"] & set(batch):
        self._finish(sl, "stop"); continue
      if g["k"] >= g["mc"]:
        self._finish(sl, "max")
