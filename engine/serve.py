"""M1-A/M1-B serving daemon LIBRARY (v2). No engine imports at module load — the
host process must be `python -u test_w100k.py` with M1A_SERVE=1 (HOST-PROCESS
BOOT LAW, see M1A_SERVING.md). Attach point: after build_graphs.

v2 (M1-B) changes vs v1:
- STRUCTURED LOGGING: slog() JSON-lines to stdout AND /tmp/m1a_serve.log with
  stage markers + dev.timeline_value around every GPU-touching phase (built to
  debug the two M1-A open issues: 2nd-FOLLOW-UP-after-load native kill, and
  first-RPC-after-idle hang). Heartbeat thread -> log file only.
- KEEPALIVE: main loop Q.get(timeout=KEEPALIVE_S) -> tiny eager dposadd probe
  + sync (the proven health-probe pattern, NOT a graph — lone-graph law kept).
  Prevents the GPU/TB idle state that wedged the first big submit after idle.
- PER-CONNECTION listener threads (v1 served one connection to EOF before the
  next — status/health during a generate on another conn starved).
- status: fed_len / fed_tail / cur / conversation_id / qsize / vram telemetry.
- Conversation tracking: ST.fed = the full fed token stream (park/FRESH set,
  generate appends emits, FOLLOW_UP appends [cur]+delta) — the API's
  longest-prefix reuse check reads it.
- FOLLOW_UP accepts optional "cur" override (win_up cur_slot first) so the fed
  stream can equal the client-rendered ids exactly.
- prefill/generate emit "prefill_progress" events for SSE comments.
- generate done event carries usage.
Protocol: see M1A_SERVING.md / M1B_SERVING.md.

TLX W2 (security/ops wave, TLX_REVIEW_LEDGER):
  V-24 socket trust boundary: 0600 socket, peer-uid allowlist (LOCAL_PEERCRED),
        admin-token ACL on shutdown/snapshot_*, input caps BEFORE any
        seed_hist/win_up (ids vocab+length, max_cycles, stop ids).
  V-30 sendall stopgap: per-conn send lock + send timeout; a stalled consumer
        can no longer wedge the GPU thread (full outbound queues = R6).
  V-31 line cap: >TLX_MAX_LINE_MB without a newline drops the conn.
  V-36/V-37 _clean_exit: the ONE exit path (SOCK unlink + pcache flush +
        staydown-on-request) shared by shutdown/keepalive/device-fault/watchdog.
  V-17 busy/rpc exposed for ANY GPU-owning RPC (status + /health).
  V-28 config fingerprint: status carries config_fp + knobs; the API alarms on
        drift (the LOOKUP_K=0-class silent degradation becomes a 503).
  watchdog: wall-clock step deadline (TLX_STEP_TIMEOUT_S) -> clean-exit(1) per
        the device-fault law (a wedged GPU heals via supervisor reboot, never
        by hanging silently). The gcycle/ops_nv halves are W4.
R6 PHASE 3 (batch serving; active when BATCH_B>=2 AND the host passes r6h):
  - Stream model: one BStream per engine slot; conn->slot bound at prefill
    (route()); generate/cancel attach to the bound stream (zero protocol
    change — the API already runs prefill->generate->cancel on one EngConn).
  - The INVERTED main loop: drain Q (non-blocking while streams generate) ->
    one batch step -> keepalive only at zero. A prefill is a BARRIER op that
    runs CHUNK-INTERLEAVED (BATCH_PF_CHUNK tokens/chunk; decode steps for the
    other streams + Q drain + per-stream cancel polls between chunks) — the
    fairness fix AND the ~950-budget reset in one mechanism.
  - GLOBAL rebuild counter (scheduler-level): fence-all rebuild (canonical
    graphs + the whole R6 graph matrix) at BATCH_REBUILD_EVERY GLOBAL cycles;
    any barrier op resets the counter. Per-request k%256 is WRONG under
    batching (B streams hit it simultaneously -> first rebuild at 256*B,
    inside the 850-1025 fault window at B>=3).
  - Depth-1 submit/wait per step at ANY composition (one ~452-kernel graph set
    in flight — the ~904 in-flight budget holds at any B; the re-patch race
    class is void).
  - Slot-0 solo decode runs the CANONICAL DecodeSession (full deep-K speed);
    slot-1 solo runs the R6 k2 graph; both-active runs the per-cycle
    composition matrix (per-stream deep flags from the previous emit — the
    READOUT-ORDER law; greedy output is proposal-source-invariant = Tier-1).
  - snapshot_save/load rejected while >1 slot in use or any stream generating.
  - BATCH_B=1 (or no r6h): the legacy single-stream loop runs VERBATIM.

NOTE: numpy is imported lazily inside run_daemon so this module stays importable
by the GPU-free mock battery (system python3 has no numpy)."""
import os, sys, time, json, socket, struct, hmac, queue, threading

SOCK = "/tmp/llm-engine.sock"
LOGF = "/tmp/m1a_serve.log"
# M1-C: REBOOT-SURVIVOR LOG LAW — /tmp is wiped at reboot (and the GPU-EXIT/
# REBOOT LAW turns every device fault into a reboot), so the structured log
# MUST also land on a persistent path or all fault forensics are destroyed.
LOGF_PERSIST = "~/tinygrad-metal/logs/m1a_serve.log"
KEEPALIVE_S = float(os.getenv("M1A_KEEPALIVE_S", "10"))
CYCLE_SLOG = bool(int(os.getenv("M1A_CYCLE_SLOG", "0")))
# M1-C ~50s-law knobs: quiesce the decode loop every N cycles for M ms
GEN_PAUSE_EVERY = int(os.getenv("M1A_GEN_PAUSE_EVERY", "0"))
GEN_PAUSE_MS = float(os.getenv("M1A_GEN_PAUSE_MS", "150"))
# M1-C ~950-cycle law: rebuild graphs every N cycles (0 = off)
GEN_REBUILD_EVERY = int(os.getenv("M1A_GEN_REBUILD_EVERY", "256"))

# ---- R6 PHASE 3 batch knobs (legacy path untouched when BATCH_B<2) ----
BATCH_B = int(os.getenv("BATCH_B", "1"))
# the ~950-cycle budget is GLOBAL-CONTINUOUS: fence-all rebuild at N GLOBAL
# cycles (~232 for margin under the 850-1025 fault window; a per-request
# k%256 counter is wrong under batching)
BATCH_REBUILD_EVERY = int(os.getenv("BATCH_REBUILD_EVERY", "232"))
# prefill chunk size (tokens) — decode steps for other streams interleave here
BATCH_PF_CHUNK = int(os.getenv("BATCH_PF_CHUNK", "64"))
# L7 FIX 5: the GLOBAL graph-submit budget. The R6 GRAPH-CLASS PREFILL BUDGET
# law: prefill chunk-graph replays (ALL prefills under PF_PG=1) SPEND the
# ~950-cycle dext budget without being counted — ST.cycles_since_rebuild
# counted decode cycles only, so an abort train could pre-spend the budget and
# the next generate rode into the 850-1025 fault envelope blind.
# gcycle.ParityGraph.submit increments dev.global_cycle_ctr for EVERY graph
# class; the rebuild decision and _gpu_rpc_entry() consult it. ~928 submits =
# the 232 spec-cycle margin (4 submits/cycle) under the fault envelope.
# 0 disables (legacy behavior).
GLOBAL_REBUILD_EVERY = int(os.getenv("TLX_GLOBAL_CYCLE_REBUILD_EVERY", "928"))

# ---- TLX W2 knobs (all env-gated; kill-switch = unset -> safe default) ----
# V-24: privileged methods (shutdown / snapshot_save / snapshot_load) need
# params.admin_token == TLX_ADMIN_TOKEN. UNSET = admin methods REFUSED
# (fail-closed). ops/env.canonical carries the token for enginectl/api.
ADMIN_TOKEN = os.getenv("TLX_ADMIN_TOKEN", "")
# V-24: peer-uid allowlist for the engine socket (uid 0 + our own euid always
# allowed; TLX_SOCKET_UIDS adds more, comma-separated).
ALLOWED_PEER_UIDS = {0, os.geteuid()} | {
    int(x) for x in os.getenv("TLX_SOCKET_UIDS", "").split(",")
    if x.strip() and x.strip().lstrip("-").isdigit()}
# V-31: max pending line bytes before the conn is dropped (4 MiB default).
MAX_LINE_BYTES = int(float(os.getenv("TLX_MAX_LINE_MB", "4")) * 1e6)
# V-30: socket op timeout (send stall protection; recv tolerates timeouts).
SEND_TIMEOUT_S = float(os.getenv("TLX_SEND_TIMEOUT_S", "10"))
# R3-13/R3-51: admission caps for the RPC queue, the batch pending deque and
# the per-conn thread count (all unbounded before — the abandoned-request
# work class: each pending entry pins ~3MB of ids; a conn flood pins threads).
MAX_Q_DEPTH = int(os.getenv("TLX_ENGINE_Q_MAX", "32"))
MAX_PENDING = int(os.getenv("TLX_PENDING_MAX", "8"))
MAX_CONNS = int(os.getenv("TLX_CONN_MAX", "64"))

class _PrefillCancelled(Exception):
  """R3-13: the legacy prefill path honors the armed ST.cancel flag at its
  prog callbacks (the flag was armed by send-stall/disconnect but only
  h_generate ever read it — an abandoned 100k prefill ran to completion).
  L7: ONE module-level class for BOTH the legacy and the batch path (the old
  batch-local shadow class is deleted); raised ONLY by cancel_checkpoint /
  _b_cancel_checkpoint (drain-then-raise) — never out of send() or a raw
  `if cancel: raise` site."""
# watchdog (W2 serving half): max wall-clock between generate-cycle/prefill-
# chunk progress beats while an RPC owns the GPU -> clean-exit(1).
STEP_TIMEOUT_S = float(os.getenv("TLX_STEP_TIMEOUT_S", "120"))
# W4.3 knobs: TLX_GLOBAL_REBUILD drives GEN_REBUILD_EVERY from the GLOBAL
# cycle counter (default; 0 = legacy per-request k%N). TLX_CYC_RESET resets
# the device cyc_slot at rebuild (default OFF: changes the cycle-event `cycle`
# field numbering; W5 decision item).
TLX_GLOBAL_REBUILD = os.getenv("TLX_GLOBAL_REBUILD", "1") == "1"
TLX_CYC_RESET = os.getenv("TLX_CYC_RESET", "0") == "1"
# V-26: staydown marker now PERSISTS (reboot-survivor) under the engine logs.
LOGS_DIR = os.getenv("TLX_LOGS_DIR", "~/tinygrad-metal/logs")
STAYDOWN = os.path.join(LOGS_DIR, "llm_engine_staydown")
# R3-42: the model identity for stream residency/status (pre-MoE: env default)
_MODEL_ID = os.getenv("TLX_MODEL_ID", "qwen3.8-27b-egpu")
_PRIV_METHODS = ("shutdown", "snapshot_save", "snapshot_load")
# R3-03: shutdown is GPU-owning too (its dev.synchronize must run watched —
# a wedged GPU used to hang the graceful stop forever with no attribution).
_GPU_RPC = ("generate", "prefill", "snapshot_save", "snapshot_load", "shutdown")

# ---- TLX W4.3: emit-sequence monotonicity (gemini's choke-point design) ----
# The per-stream emit record must obey the accept law EVERY cycle:
#   pos_new == prev_pos + m + 1  and  len(tokens) == m + 1, m in [0, 63].
# A violation means graph/state mispairing (the R4/R7 class) or a wedged
# accept kernel — the consumer must go DIRTY + error, never stream silently.
def _emit_seq_violation(prev_pos, r):
  try:
    m = int(r["m"]); pos_new = int(r["pos_new"]); toks = r["tokens"]
  except Exception as e:
    return f"emit shape error: {e!r}"
  if not (0 <= m <= 63): return f"m={m} out of [0,63]"
  if len(toks) != m + 1: return f"tokens={len(toks)} != m+1={m+1}"
  if pos_new != prev_pos + m + 1:
    return f"pos_new={pos_new} != prev_pos({prev_pos}) + m + 1 = {prev_pos + m + 1}"
  return None

class State: pass
ST = State(); ST.ready = False; ST.busy = False; ST.cancel = False
ST.t0 = time.time(); ST.mode = None
ST.fed = []            # full fed token stream of the resident conversation
ST.convo_id = None     # API-assigned conversation pin
ST.dirty = False       # W1 V-10: a faulted session left state the API can't
                       # reuse -> status reports it; FOLLOW_UP refuses until a
                       # prefill re-establishes consistent state
ST.last_rpc = time.time()
# W2 telemetry (V-17/V-28 + watchdog beat)
ST.rpc = None; ST.config_fp = None; ST.vocab = None
ST.cycles_since_rebuild = 0; ST.step_beat = time.time()

def _ka_slog_fields():
  """TLX P10: ka-pool counters for slog lines (lazy import — serve.py keeps
  'no engine imports at module load')."""
  try:
    from ka_pool import ka_stats
    ks = ka_stats()
    return dict(ka_fresh=ks["fresh"], ka_reused=ks["reused"], ka_pooled=ks["pooled"],
                ka_live=ks["live"], mapfd=ks["mapfd_total"])
  except Exception:
    return {}
ST.last_keepalive_ok = time.time()   # R3-02: idle-arm the watchdog on this
ST.rebuild_fails = 0                 # R3-06: consecutive failed rebuilds
ST.boot_beats = 0                    # R3-04b: boot-path heartbeats
# L7 abort-safety state:
ST.cancel_armed_by = None            # (source, rpc/slot) attribution for the armed cancel
ST._pf_stream = None                 # (batch) the BStream owning the active barrier prefill

# ---- R3 (W-E.1): clean-exit testability hook ---------------------------------
# _clean_exit must terminate the process in production; the real-listener
# harness (engine0/tests/serve_harness.py) overrides EXIT_FN to record the
# exit and unwind the daemon thread instead of os._exit. NEVER set in prod.
EXIT_FN = None
def _exit_now(code, reason):
  if EXIT_FN is not None:
    EXIT_FN(code, reason)    # harness hook (may raise to unwind — intended)
  os._exit(code)

# ---------------- V-24 pure helpers (unit-testable, no GPU imports) ---------
PEERCRED_LENIENT = os.getenv("TLX_PEERCRED_LENIENT", "0") == "1"

def _peer_uid(conn):
  """Peer euid via macOS LOCAL_PEERCRED (struct xucred: version, uid, gid...).
  R3-21: a FULL xucred-sized read (the old 8-byte getsockopt can truncate or
  fail on some stacks -> None). Callers now treat None as REJECT (fail-closed)
  unless TLX_PEERCRED_LENIENT=1 — the old fail-open accepted the conn on the
  0600 perms alone."""
  sol = getattr(socket, "SOL_LOCAL", 0)
  opt = getattr(socket, "LOCAL_PEERCRED", 1)
  try:
    raw = conn.getsockopt(sol, opt, 128)     # struct xucred is smaller; pad
    if len(raw) >= 8:
      _ver, uid = struct.unpack("=II", raw[:8])
      return int(uid)
  except Exception:
    pass
  return None

def check_admin(params):
  """(ok, error) gate for privileged methods. Constant-time compare."""
  if not ADMIN_TOKEN:
    return False, "admin methods disabled (TLX_ADMIN_TOKEN unset in the daemon env)"
  tok = str((params or {}).get("admin_token") or "")
  if hmac.compare_digest(tok, ADMIN_TOKEN):
    return True, None
  return False, "admin_token missing or invalid"

def validate_rpc(method, params, ctxk, vocab, pos=0):
  """V-23/V-24 input caps, PURE (no engine state): returns (err, params).
  err None -> params sanitized (max_cycles clamped into [1, min(4096,
  ctxk-pos)]). Runs BEFORE any seed_hist/win_up so bad ids can never reach
  the embedding/GPU (the R4 probe-poison OOB->reboot class).
  R3-46: a None pos (pre-sync caller) coerces to 0 instead of TypeError.
  R3-17: FOLLOW_UP requires a conversation_id — anonymous (None) would match
  ANY None-convo slot including the parked slot 0 (the silent-misroute class)."""
  pos = int(pos or 0)
  p = params or {}
  if method == "generate":
    mc = p.get("max_cycles", 60)
    if not isinstance(mc, int) or isinstance(mc, bool) or mc < 1:
      return f"max_cycles must be an integer >= 1 (got {mc!r})", None
    cap = max(1, min(4096, ctxk - max(0, int(pos))))
    if mc > cap: mc = cap
    stops = p.get("stop_token_ids", [])
    if not isinstance(stops, list):
      return "stop_token_ids must be a list of ints", None
    for t in stops:
      if not isinstance(t, int) or isinstance(t, bool) or not (0 <= t < vocab):
        return f"stop_token_id {t!r} outside vocab range 0..{vocab-1}", None
    p = dict(p); p["max_cycles"] = mc
    return None, p
  if method == "prefill":
    if "snapshot" in p:                      # snapshot-restore prefill (admin-gated separately)
      return None, p
    mode = p.get("mode", "FRESH")
    if mode not in ("FRESH", "FOLLOW_UP", "AUTO_CACHE", "PF_BATCH"):
      return f"prefill mode {mode!r} not recognized", None
    ids = p.get("ids")
    if not isinstance(ids, list) or not ids:
      return "prefill ids must be a non-empty list of ints", None
    for t in ids:
      if not isinstance(t, int) or isinstance(t, bool) or not (0 <= t < vocab):
        return f"prefill id {t!r} outside vocab range 0..{vocab-1}", None
    if mode == "FOLLOW_UP":
      if p.get("conversation_id") is None:
        return ("FOLLOW_UP requires a conversation_id (anonymous requests must use "
                "FRESH/AUTO_CACHE — None would bind ANY None-convo slot, including "
                "the parked slot 0)"), None
      if max(0, int(pos)) + 1 + len(ids) > ctxk:
        return (f"FOLLOW_UP delta overflows context: pos {pos} + 1 + {len(ids)} "
                f"> ctxk {ctxk}"), None
    elif len(ids) > ctxk:
      return f"prefill ids length {len(ids)} exceeds ctxk {ctxk}", None
    return None, p
  return None, p

def _log_rotate(paths, max_bytes, keep=2):
  """V-38b + R3-47: size-cap rotate keeping `keep` generations (.1 .. .N).
  The old single-generation drop lost .1 on EVERY rotation — under CYCLE_SLOG
  bursts the retained forensic window shrank to ~2x20MB."""
  for p in paths:
    try:
      if os.path.exists(p) and os.path.getsize(p) > max_bytes:
        for i in range(keep - 1, 0, -1):
          if os.path.exists(f"{p}.{i}"):
            os.replace(f"{p}.{i}", f"{p}.{i+1}")
        os.replace(p, p + ".1")
    except Exception:
      pass

_ROTATE_CHK = {"ts": 0.0, "n": 0}
_ROTATE_MAX = 20 * 1024 * 1024
_ROTATE_EVERY_N = 32          # a burst must rotate too, not just the 10min hb

# ---------------- V-19/V-30 per-conn send lock + stall protection ----------
# R3-16: locks keyed by id(conn) — a CLOSED conn reads fileno() == -1 and an
# accept() can hand the freed fd to a NEW conn: the old fileno keying made
# the dead conn and the new conn SHARE a lock (the cross-client-write class)
# and left stale entries behind. Each conn carries a DEAD flag written under
# its send lock at teardown; send() re-checks it INSIDE the lock so a frame
# can never start on a socket the closer already closed.
_SEND_LOCKS = {}
_SEND_LOCKS_G = threading.Lock()
def _send_lock(conn):
  key = id(conn)
  with _SEND_LOCKS_G:
    ent = _SEND_LOCKS.get(key)
    if ent is None:
      ent = _SEND_LOCKS[key] = {"lock": threading.Lock(), "dead": False}
  return ent

def _send_unlock(conn):
  with _SEND_LOCKS_G:
    _SEND_LOCKS.pop(id(conn), None)

def _mark_dead(conn):
  """R3-16: the ONE teardown close path — dead flag + close UNDER the send
  lock (a daemon thread blocked in sendall with SEND_TIMEOUT_S and the
  closer synchronize here; no later send can begin on the closed socket)."""
  ent = _send_lock(conn)
  with ent["lock"]:
    ent["dead"] = True
    try: conn.close()
    except Exception: pass
  _send_unlock(conn)

def send(conn, obj, abort_on_fail=False, cancel_fn=None):
  """V-19: serialized frames per conn (inline status vs cycle events can race).
  V-30 stopgap: the socket carries SEND_TIMEOUT_S so a consumer that stopped
  reading raises instead of wedging the GPU thread; abort_on_fail (mid-RPC
  event streams) arms the engine cancel so the stalled session ends early.
  R6: cancel_fn overrides the legacy global ST.cancel arm (per-stream cancel).
  R3-16: dead conns (teardown ran) fail fast under the lock — no frame can
  land on a reused fd."""
  try:
    ent = _send_lock(conn)
    with ent["lock"]:
      if ent["dead"]:
        raise OSError("conn dead (closed at teardown)")
      conn.sendall((json.dumps(obj) + "\n").encode())
  except Exception as e:
    if abort_on_fail:
      if cancel_fn is not None: cancel_fn()
      elif ST.busy:
        ST.cancel = True
        ST.cancel_armed_by = ("send_stalled", ST.rpc)   # L7: attribution
    slog(op="send_stalled", error=repr(e), cancel_armed=True, rpc=ST.rpc)


def log(*a): print("[serve]", *a, flush=True)

_LOGL = threading.Lock()
def slog(**kw):
  """Structured log line: stdout + append to LOGF + persistent mirror
  (reboot-survivor: /tmp dies with the machine on every fault-induced reboot)."""
  rec = {"ts": round(time.time(), 3), "uptime": round(time.time() - ST.t0, 1), **kw}
  line = json.dumps(rec, default=str)
  with _LOGL:
    _ROTATE_CHK["n"] += 1
    now2 = time.time()
    if (now2 - _ROTATE_CHK["ts"] > 30
        or _ROTATE_CHK["n"] >= _ROTATE_EVERY_N):   # R3-47: time OR burst arm
      _ROTATE_CHK["ts"] = now2; _ROTATE_CHK["n"] = 0
      try:
        _log_rotate((LOGF, LOGF_PERSIST), _ROTATE_MAX, keep=2)
      except Exception:
        pass
    for p in (LOGF, LOGF_PERSIST):
      try:
        with open(p, "a") as f:
          f.write(line + "\n"); f.flush()
          if p == LOGF_PERSIST:
            os.fsync(f.fileno())   # L7 (the L5 finding): page-cache lines are lost on the hard reset
      except Exception: pass
    print("[serve:slog]", line, flush=True)

def run_daemon(E, G, sess, decode_n, ref, ids, P0, CUR0, SNAP, CTXK, r6h=None):
  import numpy as np        # lazy: keeps this module API-side importable (W2)
  from mtp import RBLK, CBLK   # cached modules from the host process
  import mtp as _mtp
  from engine0 import dev
  import svc_fp
  try:
    from trunk import VOCAB
  except Exception:
    VOCAB = 248320            # Qwen3.8 vocab fallback (validation-only use)
  ST.vocab = int(VOCAB)
  # R3-19: mix the ACTUALLY-LOADED cubin set into config_fp (a kernel rebuild
  # with no env change must invalidate the pcache + drift-alarm, not silently
  # restore old KV into a different engine).
  svc_fp.set_extra("cubins", svc_fp.cubin_set_digest())
  ST.config_fp = svc_fp.config_fp()      # V-28: env + model-file identity
  # TLX A.1 (TLX_T1_MODE): expose the trunk engine to the DecodeSession (the
  # T=1 mode's parity graphs). B=1 only for v1 — the batch path drives its own
  # per-stream engines (Gs1) and its _process_emit selector is separate; a
  # None gcycle simply disables the mode (the tracker never fires).
  if getattr(_mtp, "TLX_T1_MODE", 0) and BATCH_B < 2 and G is not None:
    E.gcycle = G
    slog(op="t1_mode", enabled=True, trig=_mtp.TLX_T1_TRIG)
  # TLX P8 B.2 (item #0): fail-fast that the engine's LOADED model file is the
  # one the env fingerprinted (TLX_MODEL_PATH). engine0.py reads TLX_MODEL_PATH
  # at import; a stale default-path boot under a model env (or a host that
  # loaded a different GGUF) must die HERE, not serve the wrong weights with a
  # healthy /health. Realpath compare (symlink-safe); warn-only when the env is
  # unset (the pre-P8 boot shape). Exit 18 = config-class crash (the wrapper's
  # breaker bounds the retry loop; the wrapper's registry check catches the
  # common classes BEFORE the python ever boots).
  try:
    import engine0 as _e0
    _want = os.getenv("TLX_MODEL_PATH")
    _loaded = getattr(_e0, "GGUF", None)
    if _want:
      if _loaded is None or os.path.realpath(_loaded) != os.path.realpath(_want):
        print(f"[serve] FATAL model_mismatch: engine loaded {_loaded!r} but TLX_MODEL_PATH={_want!r}", flush=True)
        raise SystemExit(18)
      slog(op="model_identity", path=_loaded, model_id=_MODEL_ID)
    else:
      slog(op="model_identity", path=_loaded, model_id=_MODEL_ID, note="TLX_MODEL_PATH unset — pre-P8 default boot")
  except SystemExit:
    raise
  except Exception as _e:
    slog(op="model_identity_check_failed", error=repr(_e))
  ST.rpc = None
  ST.step_beat = time.time()
  # L7 (kimi A.1): route native-fault tracebacks to the PERSISTENT log — the
  # death signature showed empty crashlogs + lost slog tails (page cache died
  # with the box); faulthandler writes synchronously from the signal handler.
  try:
    import faulthandler
    ST._fh_file = open(LOGF_PERSIST + ".faults", "a", buffering=1)
    faulthandler.enable(file=ST._fh_file, all_threads=True)
  except Exception:
    pass

  def _beat():
    """Watchdog heartbeat: called at every generate cycle and every prefill/
    snapshot progress step while an RPC owns the GPU. R3-02: a beat is also
    keepalive evidence (the GPU demonstrably progresses) — it refreshes
    last_keepalive_ok so a healthily-SERVING daemon can never trip the
    idle-arm, while a silent wedge (no RPCs, keepalive failing) can."""
    ST.step_beat = time.time()
    ST.last_keepalive_ok = ST.step_beat

  def _clean_exit(code, reason, flush=True, staydown=False):
    """V-36/V-37: the ONE exit path — SOCK unlink, bounded pcache flush
    (queued nodes + manifest), optional persistent staydown marker (the RPC
    shutdown intent must survive launchd KeepAlive + reboot), then exit."""
    try: os.unlink(SOCK)
    except Exception: pass
    # LIVE FIX (R3 window, L3): the staydown marker writes BEFORE the flush
    # and is FSYNCED — the GPU-EXIT law reboots the box during a graceful
    # shutdown (observed live: the reboot landed while PC.flush() ran; the
    # marker never wrote and RunAtLoad relaunched the engine the operator
    # had just stopped). The socket unlink stays FIRST (unblocks enginectl).
    if staydown:
      try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(STAYDOWN, "a") as f:
          f.write(f"{time.time()} {reason}\n")
          f.flush(); os.fsync(f.fileno())
        # L5: the marker's dirent must survive the GPU-EXIT hard reset
        dfd = os.open(LOGS_DIR, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
      except Exception: pass
    if PC is not None and flush:
      try:
        PC.flush()                 # W3: bounded drain + manifest save (V-36/V-40)
      except Exception as e:
        slog(op="clean_exit_flush_error", error=repr(e))
    slog(op="clean_exit", code=code, reason=reason, staydown=staydown)
    _exit_now(code, reason)

  KV8_FLAG = bool(int(os.getenv("KV8", "1")))
  # R5d: LOOKUP history seeding — the R3/R4-documented serve-flip procedure.
  # tok_hist[0:len(fed)) must hold the fed stream BEFORE the first generate
  # cycle: an unseeded (-1) prefix self-matches in the n-gram scan and the
  # lookup overwrites drings with -1 -> embedding OOB fault (the R4
  # probe-poison law). Emitted tokens are appended device-side by accept.
  LK_ACTIVE = bool(_mtp.LOOKUP or _mtp.LOOKUP_K)
  def seed_hist(fed):
    if LK_ACTIVE and len(fed):
      E.P.win_up("tok_hist", 0, np.asarray(fed, dtype=np.int32))
      dev.synchronize()
  # ---- R1: the durable prompt cache (env PC_ENABLED=0 to disable) ----
  PC = None
  if os.getenv("PC_ENABLED", "1") == "1" and int(os.getenv("KV8", "1")) == 1:
    try:
      import pcache
      PC = pcache.PromptCache()
      slog(op="pc_boot", stage="loaded", entries=len(PC.man["entries"]),
           bytes_gb=round(PC.total_bytes() / 1e9, 2), quota_gb=round(pcache.QUOTA_BYTES / 1e9, 1))
    except Exception as e:
      slog(op="pc_boot_error", error=repr(e))
  ST.pc_last_end = 0        # deepest cached boundary of the resident conversation
  ST.pc_last_hkey = None

  def tl(): return getattr(dev, "timeline_value", -1)

  def _t1_park(where):
    """TLX A.1: exit a live T=1 episode at a quiescent point so the SPEC world
    (rec4/conv4 slot 4, cur_slot) is current for the next consumer. Every
    h_generate return path, the rebuild fence, and the prefill/snapshot
    handlers park before touching spec-side state. No-op when not in episode
    (TLX_T1_MODE=0 or never entered)."""
    if not getattr(sess, "t1mode", 0):
      return
    try:
      _tok = int(E.P.down_at("tok_slot", 0, 1, np.int32)[0])
      sess._t1_exit(0, _tok)
      slog(op="t1_park", where=where, cycles=sess.nt1cycles, tl=tl())
    except Exception as e:
      # best-effort: a faulted device reboots the box anyway (GPU-EXIT law) —
      # but never leave the episode flag set with the state unresolved.
      sess.t1mode = 0
      slog(op="t1_park_failed", where=where, error=repr(e), tl=tl())

  # ---------------- L7 ABORT-SAFETY PROTOCOL ----------------------------------
  # Converged diagnosis (4-agent audit): sticky ST.cancel poisoned every later
  # client's prefill; the abort path raised with eager fill_draft kernels still
  # in flight (no fence); the barrier dropped on a busy device; prefill graph
  # submits spent the dext budget invisibly. The five fixes below make cancel
  # a QUIESCENT, ATTRIBUTED, FENCED protocol:
  #   FIX 1  entry-clear the cancel flag at EVERY GPU RPC (kills the train)
  #   FIX 2  cancel_checkpoint() = the ONLY legal raise site (drain-then-raise)
  #   FIX 3  fill_draft checks AFTER its every-16 sync (mtp.py)
  #   FIX 4  abort handler: dev.synchronize() + slot invalidation BEFORE the
  #          barrier/Q hands the GPU to the next RPC
  #   FIX 5  dev.global_cycle_ctr counts ALL graph classes; entry rebuild
  def _arm_cancel(src):
    ST.cancel = True
    ST.cancel_armed_by = (src, ST.rpc)

  def _gpu_rpc_entry():
    """L7 FIX 1+5 (legacy dispatcher): every GPU-owning RPC starts with a
    CLEAN cancel slate — ST.cancel was a sticky global only h_generate ever
    cleared, so one send-stall/disconnect armed every later client's prefill
    for abort (the poison train). Plus the global graph-submit budget check:
    spend the dext budget down (fence-class rebuild) BEFORE new work when
    prefill chunk replays pushed it over."""
    ST.cancel = False
    ST.cancel_armed_by = None
    if GLOBAL_REBUILD_EVERY and getattr(dev, "global_cycle_ctr", 0) >= GLOBAL_REBUILD_EVERY:
      try:
        dev.synchronize()               # rebuild ONLY at a quiescent point
        E.build_graphs()
        dev.global_cycle_ctr = 0
        ST.cycles_since_rebuild = 0     # fresh graph set = budget reset (both proxies)
        slog(op="l7_entry_rebuild", gctr_at=GLOBAL_REBUILD_EVERY, tl=tl(), **_ka_slog_fields())
      except Exception as e:
        ST.rebuild_fails += 1
        slog(op="l7_entry_rebuild_failed", error=repr(e), fails=ST.rebuild_fails)
        if ST.rebuild_fails >= 3:
          _clean_exit(1, "rebuild_budget_exhausted", flush=True)

  def cancel_checkpoint(stage):
    """L7 FIX 2: the ONLY legal raise site for prefill cancellation. Never
    raises with GPU work outstanding: dev.synchronize() FIRST (eager
    fill_draft launches carry NO timeline values — only a full ring drain
    retires them), then the abort slog, then the raise."""
    if not ST.cancel:
      return
    _beat()
    dev.synchronize()
    slog(op="abort", stage=stage, tl=tl(), armed_by=ST.cancel_armed_by,
         cycles_since_rebuild=ST.cycles_since_rebuild,
         gctr=getattr(dev, "global_cycle_ctr", -1))
    raise _PrefillCancelled()

  def _engine_cancel_check(stage):
    """The mtp.CANCEL_CHECK install target — works for BOTH paths: the legacy
    global flag or the batch per-stream flag of the active barrier prefill."""
    st = ST._pf_stream
    if not (ST.cancel or (st is not None and st.cancel_req)):
      return
    _beat()
    dev.synchronize()
    slog(op="abort", stage=stage, slot=(st.s if st is not None else None),
         tl=tl(), armed_by=ST.cancel_armed_by,
         cycles_since_rebuild=ST.cycles_since_rebuild,
         gctr=getattr(dev, "global_cycle_ctr", -1))
    raise _PrefillCancelled()
  _mtp.CANCEL_CHECK = _engine_cancel_check

  def _mk_prog(conn, rid):
    """L7 FIX 2: prefill progress callbacks are BEAT+SEND only — a send
    failure ARMS the cancel (with attribution); the quiescent checkpoint that
    FOLLOWS honors it. prog call sites are post-wait (prefill_t1 per-token,
    pf chunk loops post PG_WAIT) so the checkpoint's drain is ~free."""
    def prog(done, total, stage="prefill_t1"):
      _beat()
      send(conn, {"id": rid, "event": "prefill_progress", "stage": stage,
                  "done": done, "total": total}, abort_on_fail=True,
           cancel_fn=lambda: _arm_cancel("send_stalled"))
      cancel_checkpoint(stage)
    return prog


  def _rebuild_capacity_asserts():
    """W4.3 (V-57): at the quiescent rebuild point, the device-side hist
    indices must have headroom for the next window — accept writes
    m_hist[cyc_slot] (1<<20) and tok_hist[pos + t] (CTXK+256 rows), and
    nothing else ties them to the rebuild cadence."""
    try:
      _rm = int(getattr(_mtp, "RM", 3))
      pos = int(E.P.down_at("pos_slot", 0, 1)[0])
      assert pos + _rm + 2 <= CTXK + 256, \
        f"tok_hist headroom: pos {pos} + RM {_rm} > CTXK+256 (accept OOB write class)"
      cyc = int(E.P.down_at("cyc_slot", 0, 1)[0])
      assert cyc < (1 << 20) - 4096, \
        f"cyc_slot {cyc} near m_hist 1<<20 cap (accept/l_hist OOB class; TLX_CYC_RESET=1 resets)"
      if TLX_CYC_RESET:
        E.P.win_up("cyc_slot", 0, np.zeros(1, dtype=np.int32)); dev.synchronize()
        slog(op="cyc_slot_reset", was=cyc)
    except Exception as e:
      slog(op="rebuild_capacity_assert_failed", error=repr(e))
      raise

  def keepalive():
    """Tiny eager probe (NOT a graph): 4B copyin + 1-block kernel + sync.
    Keeps the dext/TB link from entering the idle state that wedged the first
    big submit after idle (M1-A issue 2). Lone-graph law: no graphs touched.
    R3-02: a successful probe refreshes last_keepalive_ok (the idle-arm
    watchdog input)."""
    E.P.win_up("fillpos", 0, np.array([123], dtype=np.int32))
    E.pr["dposadd"](E.P.d["fillpos"], E.P.d["dpos1"], global_size=(1,1,1), local_size=(256,1,1))
    dev.synchronize()
    hp = int(E.P.down_at("dpos1", 0, 1)[0]); assert hp == 124, hp
    ST.last_keepalive_ok = time.time()

  def h_status(p):
    return {"ready": ST.ready, "ctxk": CTXK, "busy": ST.busy,
            "rpc": ST.rpc,                       # V-17: which GPU RPC owns busy
            "pos": ST.pos_cache, "mode": ST.mode,
            "cur": ST.cur_cache,
            "fed_len": len(ST.fed), "fed_tail": ST.fed[-64:],
            "conversation_id": ST.convo_id, "queue": Q.qsize() if Q else 0,
            "dirty": ST.dirty,
            "keepalive_s": KEEPALIVE_S,
            "keep_len": len(E.P._keep),
            "uptime_s": round(time.time() - ST.t0, 1),
            "config_fp": ST.config_fp,           # V-28: env + model identity
            "lookup_k": os.getenv("LOOKUP_K"),
            "pf_prefill": os.getenv("PF_PREFILL"),
            "model_id": _MODEL_ID,               # R3-42
            "cycles_since_rebuild": ST.cycles_since_rebuild,
            "cycle_cap": max(1, min(4096, CTXK - max(0, int(ST.pos_cache)))),  # R3-34
            "pc": _pc_status()}                  # R3-28: cache health

  def _pc_status():
    """R3-28: pcache health through status -> /health debug (pc_drop /
    writer errors / quota — lets ops tell cache-degraded from engine-degraded)."""
    if PC is None:
      return None
    try:
      return {"entries": len(PC.man["entries"]), "total_bytes": PC.total_bytes(),
              "quota_bytes": pcache.QUOTA_BYTES, "stats": dict(PC.stats)}
    except Exception as e:
      return {"error": repr(e)}

  def _sync_caches():
    """Refresh host-side pos/cur caches for the lock-free inline status."""
    ST.pos_cache = int(E.P.down_at("pos_slot", 0, 1)[0])
    ST.cur_cache = int(E.P.down_at("cur_slot", 0, 1)[0])

  # ---------------- R1 prompt-cache handlers (daemon thread only) ----------------
  def _pc_node_done(hk, B):
    ST.pc_last_end = B; ST.pc_last_hkey = hk
    PC.add_protect(hk)          # W3: protect mutations under PC.lock (V-41)

  def _pc_ingest_cb(toks):
    """Fires at every 64-chunk quiescent boundary of a FRESH prefill; captures
    a node at each PC_STRIDE boundary (+ the final 64-aligned boundary). The
    GPU download (~200MB) happens here on the daemon thread (legal: chunk
    quiescence); the disk write is async on the pcache writer."""
    def cb(pos_after, final_b):
      want = (pos_after % pcache.STRIDE == 0 and pos_after >= pcache.STRIDE) or \
             (pos_after == final_b and pos_after >= pcache.HASH_BLK)
      if not want: return
      A = ST.pc_last_end
      if pos_after - A < pcache.HASH_BLK: return
      t0 = time.perf_counter()
      node = pcache.capture_node(E, A, pos_after, "midprefill", fed_prefix=toks, parent=ST.pc_last_hkey)
      PC.write_node(node)
      _pc_node_done(node["hkey"], pos_after)
      slog(op="pc_ingest", pos=pos_after, win=[A, pos_after],
           secs=round(time.perf_counter() - t0, 2), wq=PC.wq.qsize())
    return cb

  def _pc_prefill(conn, rid, p, toks):
    """AUTO_CACHE: deepest-chain restore + M64 tail. Result dict on hit
    (mode CACHE_HIT, cached_tokens=B); None -> caller falls to FRESH+ingest.
    W3: lookup protects the chain inside PC.lock (TOCTOU); restore validates
    the whole chain on disk BEFORE any win_up — NodeCorrupt quarantines the
    node and returns None (clean FRESH fallback, engine untouched); pins
    carry the per-request TTL (V-44)."""
    cache_key = p.get("cache_key") or p.get("prompt_cache_key")
    ttl = p.get("cache_ttl") or p.get("prompt_cache_ttl")
    min_cov = max(pcache.MIN_HIT, int(0.5 * len(toks)))
    t0 = time.perf_counter()
    B, chain = PC.lookup(toks, min_hit=min_cov)
    if not chain:
      slog(op="pc_lookup", stage="miss", tl=tl(), n=len(toks), secs=round(time.perf_counter() - t0, 2))
      return None
    slog(op="pc_lookup", stage="hit", tl=tl(), B=B, nodes=len(chain), cached_gb=round(sum(int(e["bytes"]) for _, e in chain) / 1e9, 2))
    PC.touch(chain)
    if cache_key: PC.pin(chain, str(cache_key)[:128], ttl=ttl)
    PC.set_protect(hk for hk, _ in chain)
    t0 = time.perf_counter()
    try:
      B2, cur = pcache.restore_chain(E, chain, beat=_beat); assert B2 == B
    except pcache.NodeCorrupt as nc:
      slog(op="pc_corrupt", hkey=(nc.hkey or "")[:16], why=nc.why, tl=tl())
      PC.quarantine(nc.hkey)
      return None       # -> FRESH + ingest (validation precedes any upload)
    ST.fed = list(toks[:B]); ST.mode = "CACHE_HIT"; ST.convo_id = p.get("conversation_id")
    seed_hist(ST.fed)
    _pc_node_done(chain[-1][0], B)
    _sync_caches()
    slog(op="pc_restore", stage="done", tl=tl(), B=B, secs=round(time.perf_counter() - t0, 1))
    prog = _mk_prog(conn, rid)          # L7: beat+send; checkpoint raises
    if len(toks) > B:
      t0 = time.perf_counter()
      E.P.win_up("cur_slot", 0, np.array([int(toks[B])], dtype=np.int32))
      newcur, posn, nd = E.follow_up(G, toks[B + 1:],
                                     log=lambda s, **kw: slog(op="follow_up", stage=s, tl=tl(), **kw), prog=prog)
      ST.fed = list(toks)
      seed_hist(ST.fed)
      slog(op="pc_tail", stage="done", tl=tl(), n=nd, secs=round(time.perf_counter() - t0, 1))
    else:
      newcur, posn = cur, B
    ST.pos_cache = posn; ST.cur_cache = newcur
    slog(op="prefill_cache_hit", stage="done", tl=tl(), pos=posn, cur=newcur, cached_tokens=B)
    return {"pos": posn, "cur": newcur, "fed": len(ST.fed), "mode": "CACHE_HIT", "cached_tokens": B}

  def _pc_turn_end(pos):
    """Turn-end ingest (quiescent, spec slot-4 world). Only at 64-aligned pos
    (the hash-chain boundary law). Windows continue the resident chain."""
    if PC is None: return
    if pos % pcache.HASH_BLK != 0 or pos - ST.pc_last_end < pcache.HASH_BLK: return
    if pos > len(ST.fed): return
    try:
      t0 = time.perf_counter()
      node = pcache.capture_node(E, ST.pc_last_end, pos, "turnend", fed_prefix=ST.fed, parent=ST.pc_last_hkey)
      PC.write_node(node)
      _pc_node_done(node["hkey"], pos)
      slog(op="pc_turn_end_ingest", pos=pos, win_secs=round(time.perf_counter() - t0, 2))
    except Exception as e:
      slog(op="pc_turn_end_error", error=repr(e))

  def _pc_boot_ingest():
    hk = pcache.hkey_prefix(ids[:P0])
    with PC.lock:
      have = hk in PC.man["entries"]
    if have:
      ST.pc_last_end = P0; ST.pc_last_hkey = hk; PC.set_protect({hk})
      slog(op="pc_boot", stage="skip_present", hkey=hk[:16])
      return
    t0 = time.perf_counter()
    node = pcache.boot_node(E, SNAP, P0, CUR0, ids,
                            progress=lambda j, n: (_beat(), slog(op="pc_boot", stage="quant", j=j, n=n)))
    PC.write_node(node)
    _pc_node_done(node["hkey"], P0)
    slog(op="pc_boot", stage="done", secs=round(time.perf_counter() - t0, 1),
         hkey=node["hkey"][:16], bytes_gb=round(PC.total_bytes() / 1e9, 2))

  def h_prefill(conn, rid, p):
    _t1_park("prefill_entry")   # TLX A.1 belt: spec world current (generate-end parks normally)
    if "snapshot" in p:
      _beat()
      slog(op="prefill_snapshot", stage="begin", tl=tl(), path=p["snapshot"])
      m = json.load(open(p["snapshot"] + "/meta.json"))
      # R3-31: actionable format error (a snapshot_save dir pointed here used
      # to die on a raw KeyError('cur0'))
      for _k in ("cur0", "P"):
        if _k not in m:
          raise ValueError(f"snapshot meta missing {_k!r}: this endpoint expects the "
                           f"BASE-BOOT snapshot format (cur0, P); a snapshot_save dir "
                           f"carries (pos, cur, row_start, row_end, ...) and belongs "
                           f"to snapshot_load")
      E.reset_snapshot(p["snapshot"], int(m["cur0"]), int(m["P"]))
      ST.mode = "snapshot"; ST.fed = list(ids); ST.convo_id = p.get("conversation_id")
      seed_hist(ids)
      _sync_caches()
      r = {"pos": ST.pos_cache, "cur": ST.cur_cache, "fed": 0}
      slog(op="prefill_snapshot", stage="done", tl=tl(), **r)
      return r
    mode = p.get("mode", "FRESH"); toks = [int(t) for t in p["ids"]]
    if mode == "FOLLOW_UP":
      # W1 V-03 (engine side): FOLLOW_UP appends to the RESIDENT conversation.
      # Refuse a foreign conversation id outright (the API's per-conversation
      # lock is the primary guard; this is defense-in-depth against any
      # TOCTOU window). Echo the resident id in the error for diagnosis.
      req_cid = p.get("conversation_id")
      if req_cid != ST.convo_id:
        raise ValueError(f"FOLLOW_UP conversation_id mismatch: engine resident="
                         f"{ST.convo_id!r} request={req_cid!r} fed_len={len(ST.fed)}")
      if ST.dirty:
        # L7 FIX 4 (dirty-gate hardening): a cancelled/faulted prefill left
        # half-written state — FOLLOW_UP must NEVER append to it (the API
        # already forces FRESH via status.dirty; this closes the engine side).
        raise ValueError("engine state dirty (cancel/fault); FRESH required")
    ST.convo_id = p.get("conversation_id")
    if PC is not None and mode in ("FRESH", "AUTO_CACHE"):
      r = _pc_prefill(conn, rid, p, toks)
      if r is not None:
        return r
    if mode == "FOLLOW_UP":
      cur_override = p.get("cur")
      if cur_override is not None:
        E.P.win_up("cur_slot", 0, np.array([int(cur_override)], dtype=np.int32))
      cur = int(E.P.down_at("cur_slot", 0, 1)[0])
      _beat()
      slog(op="follow_up", stage="begin", tl=tl(), n=len(toks), cur=cur, pos0=int(E.P.down_at("pos_slot", 0, 1)[0]))
      prog = _mk_prog(conn, rid)          # L7: beat+send; checkpoint raises
      _prefix_len = len(ST.fed)               # R3-41: the reused resident prefix
      newcur, posn, nd = E.follow_up(G, toks, log=lambda s, **kw: slog(op="follow_up", stage=s, tl=tl(), **kw), prog=prog)
      ST.fed = ST.fed + [cur] + toks
      seed_hist(ST.fed)
      ST.mode = "FOLLOW_UP"
      ST.pos_cache = posn; ST.cur_cache = newcur
      r = {"pos": posn, "cur": newcur, "fed": nd, "cached_tokens": _prefix_len}
      slog(op="follow_up", stage="done", tl=tl(), **r)
      return r
    slog(op="prefill_fresh", stage="begin", tl=tl(), n=len(toks))
    _beat()
    prog = _mk_prog(conn, rid)          # L7: beat+send; checkpoint raises
    if PC is not None and os.getenv("PF_PREFILL") == "1":
      # R1: single source of truth (gates drive the same pcache.fresh_prefill)
      ST.pc_last_end = 0; ST.pc_last_hkey = None; PC.clear_protect()
      newcur, posn, _f64 = pcache.fresh_prefill(
          E, G, toks, prog=prog,
          log=lambda s, **kw: slog(op="prefill_batch", stage=s, tl=tl(), **kw),
          ingest=_pc_ingest_cb(toks) if len(toks) >= 64 else None,
          log_ingest=lambda s, **kw: slog(op="pc_ingest", stage=s, **kw))
      ST.fed = list(toks); ST.mode = "FRESH"
      seed_hist(toks)
      ST.pos_cache = posn; ST.cur_cache = newcur
      r = {"pos": posn, "cur": newcur, "fed": len(toks), "mode": "FRESH", "cached_tokens": 0}
      slog(op="prefill_fresh", stage="done", tl=tl(), **r)
      return r
    E.reset_fresh(toks[0])
    E.stload_trunk()   # M1-B FIX #1: seed trunk GDN from spec slot 4 (= ZEROS after
                       # reset_fresh). Without this the trunk prefill ran from
                       # whatever GDN state the previous conversation left in
                       # rec{i}/conv{i} -> nondeterministic FRESH outputs.
    # M1-B FIX #2: zero the OTHER conv parity too (stload_trunk writes conv{i}_0
    # only; after an odd-length FOLLOW_UP the live state sat in conv{i}_1 and
    # leaked into subsequent FRESH prefills: FRESH-after-FU gave cur=13 vs
    # cur=0 elsewhere, deterministically per context).
    for _i in E.gdn_idx:
      E._mfill(f"conv{_i}_1", 0, CBLK)
    dev.synchronize()
    # P3: batched M=16 prefill — mode PF_BATCH explicit, or FRESH with PF_PREFILL=1
    use_pf = (mode == "PF_BATCH") or (mode == "FRESH" and os.getenv("PF_PREFILL") == "1")  # PF_T1 forces the T=1 path
    # P4 Stage 2: batched fill_draft runs interleaved in prefill_batch (PF_DFILL)
    skip_fd = use_pf and len(toks) >= 16 and os.getenv("PF_DFILL", "1") == "1"
    if not skip_fd:
      E.fill_draft(toks, start_pos=0, seed_hd=None, prog=lambda d, t: prog(d, t, "fill_draft"))
    if use_pf and len(toks) >= 16:
      import pf_prefill
      slog(op="prefill_batch", stage="begin", tl=tl(), n=len(toks))
      pf_prefill.prefill_batch(E, G, toks,
                               prog=lambda d, t: prog(d, t, "prefill_batch"),
                               log=lambda s, **kw: slog(op="prefill_batch", stage=s, tl=tl(), **kw))
    else:
      E.prefill_t1(G, toks, log=lambda k, n: slog(op="prefill_t1", stage="tok", k=k, n=n, tl=tl()), prog=prog)
    E.stseed_spec(len(toks) & 1)
    newcur = int(E.P.down_at("tok_slot", 0, 1)[0])
    E.P.win_up("cur_slot", 0, np.array([newcur], dtype=np.int32))
    E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
    E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
    E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
    dev.synchronize()
    ST.fed = list(toks); ST.mode = "FRESH"
    seed_hist(toks)
    if PC is not None: ST.pc_last_end = 0; ST.pc_last_hkey = None; PC.clear_protect()
    ST.pos_cache = int(E.P.down_at("pos_slot", 0, 1)[0]); ST.cur_cache = newcur
    r = {"pos": ST.pos_cache, "cur": newcur, "fed": len(toks)}
    slog(op="prefill_fresh", stage="done", tl=tl(), **r)
    return r

  def h_generate(conn, rid, p):
    mc = int(p.get("max_cycles", 60)); stops = set(int(t) for t in p.get("stop_token_ids", []))
    ST.cancel = False                 # L7: also done at _gpu_rpc_entry (belt)
    ST.cancel_armed_by = None
    t0 = time.perf_counter()
    sess.begin()
    slog(op="generate", stage="begin", tl=tl(), max_cycles=mc)
    all_toks = []
    k = 0; r = None
    _prev_pos = [int(ST.pos_cache)]
    try:
      for k in range(1, mc + 1):
        _beat()
        if ST.cancel:
          send(conn, {"id": rid, "event": "cancelled", "tokens": all_toks, "cycles": k - 1}, abort_on_fail=True)
          ST.fed = ST.fed + all_toks
          _t1_park("generate_cancelled")   # TLX A.1: spec world current before any later RPC
          slog(op="generate", stage="cancelled", cycles=k-1, ntoks=len(all_toks), tl=tl())
          return {"cancelled": True, "tokens": all_toks, "cycles": k - 1}
        r = sess.step()
        # W4.3 emit-sequence monotonicity: violation = dirty + error event + raise
        _viol = _emit_seq_violation(_prev_pos[0], r)
        _prev_pos[0] = int(r["pos_new"])
        if _viol is not None:
          ST.dirty = True
          slog(op="emit_seq_violation", violation=_viol, k=k, tl=tl())
          send(conn, {"id": rid, "event": "error", "error": f"emit_seq_violation: {_viol}"}, abort_on_fail=True)
          raise RuntimeError(f"emit_seq_violation: {_viol}")
        ST.pos_cache = r["pos_new"]
        all_toks += r["tokens"]
        ST.cycles_since_rebuild += 1       # V-51 precursor: GLOBAL across generates
        _beat()
        if CYCLE_SLOG: slog(op="cycle", k=k, tl=tl(), pos=r["pos_new"], m=r["m"])
        # M1-C ~950-CYCLE LAW: a single continuous run of back-to-back spec
        # cycles faults the dext at 850-1025 cycles (6/6 observed, pos- and
        # thermal-independent, idle gaps do not reset it, ring-size-independent).
        # Interleaved prefill-class work RESETS the budget (33 repro sessions,
        # ~990 cycles total: zero faults). Workaround: rebuild the four graphs
        # at a quiescent point every N cycles (fresh queue objects + kernargs —
        # the same class of work a prefill does). Verified: 3000-cycle gens clean.
        # W4.3 (V-51): the ~950-cycle budget is GLOBAL-CONTINUOUS across
        # generates — the legacy per-request k%N trigger NEVER fires for short
        # generates (60-tok requests accumulate unbounded). Drive from the
        # global counter (identical behavior within one long generate; safe at
        # B=1; required for R6). TLX_GLOBAL_REBUILD=0 restores the legacy k%N.
        _rebuild_now = (ST.cycles_since_rebuild >= GEN_REBUILD_EVERY) if TLX_GLOBAL_REBUILD \
                       else (k % GEN_REBUILD_EVERY == 0)
        if GEN_REBUILD_EVERY and _rebuild_now:
          _rebuild_capacity_asserts()
          try:
            _t1_park("rebuild_fence")     # TLX A.1: spec world current before begin()
            E.build_graphs(); sess.begin()
            # TLX A.1: the T=1 parity graphs are a cycled graph class too —
            # refresh their queues at the same fence (the ~950-cycle dext law
            # counts EVERY submit class; dev.global_cycle_ctr already does).
            if getattr(E, "gcycle", None) is not None:
              G.build()
            ST.cycles_since_rebuild = 0
            dev.global_cycle_ctr = 0      # L7 FIX 5: fence-class reset (both proxies)
            ST.rebuild_fails = 0            # R3-06: reset ONLY on success
            slog(op="gen_rebuild", k=k, global_cycles=True, tl=tl(), **_ka_slog_fields())
          except Exception as e:
            # TLX W5 (finding #9, the KERNARGS-SLAB LEAK law): a failed rebuild
            # stays NON-FATAL for THIS generate (old graphs valid, sess holds
            # them). R3-06 escalation gap: the W5 posture reset the COUNTER on
            # failure, so a persistent failure (the demonstrated ka-slab
            # exhaustion class) left the ~950-cycle dext budget unenforced and
            # marched the box into the 850-1025 window -> device fault ->
            # MACHINE REBOOT. The counter now resets ONLY on success; 3 failed
            # fences in a row (or >=800 cycles while failing) exits cleanly,
            # inside the safety margin, with attribution.
            ST.rebuild_fails += 1
            slog(op="gen_rebuild_failed", error=repr(e), global_cycles=True,
                 fails=ST.rebuild_fails, tl=tl())
            if ST.rebuild_fails >= 3 or ST.cycles_since_rebuild >= 800:
              slog(op="rebuild_budget_exhausted", fails=ST.rebuild_fails,
                   cycles_since_rebuild=ST.cycles_since_rebuild, k=k, tl=tl())
              _clean_exit(1, "rebuild_budget_exhausted", flush=True)
          _beat()
        # (M1-C pause knobs — kept for reference; proven NOT to prevent the
        # ~950-cycle fault: idle gaps don't reset the dext-side budget.)
        if GEN_PAUSE_EVERY and k % GEN_PAUSE_EVERY == 0:
          dev.synchronize(); time.sleep(GEN_PAUSE_MS / 1000.0)
        send(conn, {"id": rid, "event": "cycle", "cycle": r["cycle"], "pos": r["pos_new"], "tokens": r["tokens"]}, abort_on_fail=True)
        if stops & set(r["tokens"]):
          ST.fed = ST.fed + all_toks
          _t1_park("generate_stop")        # TLX A.1
          _pc_turn_end(r["pos_new"])   # R1: quiescent turn end -> chain continues
          send(conn, {"id": rid, "event": "done", "tokens": all_toks, "cycles": k, "pos": r["pos_new"], "stop": True,
                      "usage": {"tokens": len(all_toks), "cycles": k, "secs": round(time.perf_counter()-t0, 2)}}, abort_on_fail=True)
          slog(op="generate", stage="done_stop", cycles=k, ntoks=len(all_toks), stop=True)
          return {"tokens": all_toks, "cycles": k, "stop": True}
      ST.fed = ST.fed + all_toks
      _t1_park("generate_max")          # TLX A.1
      _pc_turn_end(r["pos_new"])   # R1: quiescent turn end -> chain continues
      send(conn, {"id": rid, "event": "done", "tokens": all_toks, "cycles": mc, "pos": r["pos_new"],
                  "usage": {"tokens": len(all_toks), "cycles": mc, "secs": round(time.perf_counter()-t0, 2)}}, abort_on_fail=True)
      slog(op="generate", stage="done_max", cycles=mc, ntoks=len(all_toks))
      return {"tokens": all_toks, "cycles": mc}
    except Exception:
      # W1 V-10: tokens from COMPLETED cycles must still reach the fed mirror
      # (known-good tokens), and the conversation is dirty — the API forces
      # FRESH for the next request instead of riding a desynced prefix.
      ST.fed = ST.fed + all_toks
      ST.dirty = True
      slog(op="generate", stage="faulted", cycles_completed=k-1, ntoks=len(all_toks), tl=tl())
      raise

  def h_snapshot_save(p):
    _t1_park("snapshot_entry")   # TLX A.1 belt: cur_slot must be the committed stream
    path = p["path"]; os.makedirs(path, exist_ok=True)
    pos = int(E.P.down_at("pos_slot", 0, 1)[0]); cur = int(E.P.down_at("cur_slot", 0, 1)[0])
    r0, r1 = P0, min(pos + 16, CTXK)
    assert r1 > r0, f"nothing to save past the base prefix (pos {pos} <= P0 {P0})"
    t0 = time.perf_counter()
    slog(op="snapshot_save", stage="begin", tl=tl(), pos=pos, rows=[r0, r1])
    for i in E.attn_idx:
      _beat()
      np.save(f"{path}/kvb_{i}.npy", E.P.down_at(f"kv{i}", r0*2*4*256, (r1-r0)*2*4*256, np.uint8))
      if KV8_FLAG: np.save(f"{path}/sc_{i}.npy", E.P.down_at(f"sc{i}", r0*2*4*8, (r1-r0)*2*4*8, np.float16))
    # R1 GAP-1: persist the draft KV too — without it a cross-restart restore
    # re-pays fill_draft (~255s @100k) or runs with a dead draft (alpha -> 0).
    if KV8_FLAG:
      np.save(f"{path}/kvd.npy", E.P.down_at("kv_d", r0*2*4*256, (r1-r0)*2*4*256, np.uint8))
      np.save(f"{path}/scd.npy", E.P.down_at("sc_d", r0*2*4*8, (r1-r0)*2*4*8, np.float16))
    slog(op="snapshot_save", stage="kv_done", tl=tl())
    for j, i in enumerate(E.gdn_idx):
      _beat()
      np.save(f"{path}/rc4_{i}.npy", E.P.down_at("rec4", (j*5+4)*RBLK*4, RBLK, np.float32))
      np.save(f"{path}/cv4_{i}.npy", E.P.down_at("conv4", (j*5+4)*CBLK*4, CBLK, np.float32))
    np.save(f"{path}/h_seed.npy", E.P.down_at("h_seed", 0, 5120, np.float32))
    np.save(f"{path}/dhd_seed.npy", E.P.down_at("dhd_seed", 0, 5120, np.float32))
    json.dump({"pos": pos, "cur": cur, "ctxk": CTXK, "row_start": r0, "row_end": r1,
               "kv8": bool(KV8_FLAG), "base": SNAP},
              open(f"{path}/meta.json", "w"))
    r = {"path": path, "pos": pos, "cur": cur, "delta_rows": [r0, r1],
         "secs": round(time.perf_counter()-t0, 1)}
    slog(op="snapshot_save", stage="done", tl=tl(), secs=r["secs"])
    return r

  def h_snapshot_load(p):
    path = p["path"]; m = json.load(open(path + "/meta.json"))
    assert int(m["ctxk"]) == CTXK and bool(m["kv8"]) == bool(KV8_FLAG) and m.get("base") == SNAP
    pos, cur = int(m["pos"]), int(m["cur"])
    r0, r1 = int(m["row_start"]), int(m["row_end"])
    slog(op="snapshot_load", stage="begin", tl=tl(), pos=pos, rows=[r0, r1])
    E.reset_snapshot(SNAP, CUR0, P0)      # poison+seed GDN slot 4 + prompt KV + slots
    slog(op="snapshot_load", stage="reset_done", tl=tl())
    for i in E.attn_idx:
      _beat()
      E.P.win_up(f"kv{i}", r0*2*4*256, np.load(f"{path}/kvb_{i}.npy"))
      E.P.win_up(f"sc{i}", r0*2*4*8, np.load(f"{path}/sc_{i}.npy"))
      E._flush()
    slog(op="snapshot_load", stage="kv_up_done", tl=tl())
    for j, i in enumerate(E.gdn_idx):
      _beat()
      E.P.win_up("rec4", (j*5+4)*RBLK*4, np.load(f"{path}/rc4_{i}.npy", mmap_mode="r"))
      E.P.win_up("conv4", (j*5+4)*CBLK*4, np.load(f"{path}/cv4_{i}.npy", mmap_mode="r"))
      if j % 16 == 0: dev.synchronize()
    E.P.win_up("h_seed", 0, np.load(f"{path}/h_seed.npy"))
    E.P.win_up("dhd_seed", 0, np.load(f"{path}/dhd_seed.npy"))
    if KV8_FLAG and os.path.exists(f"{path}/kvd.npy"):   # R1 GAP-1 restore
      E.P.win_up("kv_d", r0*2*4*256, np.load(f"{path}/kvd.npy", mmap_mode="r"))
      E.P.win_up("sc_d", r0*2*4*8, np.load(f"{path}/scd.npy", mmap_mode="r"))
    E.P.win_up("cur_slot", 0, np.array([cur], dtype=np.int32))
    E.P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32))
    E.P.win_up("tok_slot", 0, np.array([cur], dtype=np.int32))
    E.P.win_up("m_slot", 0, np.zeros(1, dtype=np.int32))
    E.P.win_up("cyc_slot", 0, np.zeros(1, dtype=np.int32))
    E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
    E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
    E._mfill("m_hist", 0, 1024); E._mfill("tok_hist", -1, CTXK + 256)
    dev.synchronize(); E.P._keep.clear()
    ST.mode = "snapshot_load"
    seed_hist(ids)   # R5d: best-effort (base ids) — decode appends device-side
    # fed-stream reconstruction: base prompt + tokens emitted since row_start.
    # meta does not carry the emitted stream; caller resumes deterministic state
    # via cur/pos. fed_tail is best-effort (base ids); API resyncs via cur/pos.
    ST.fed = list(ids)
    slog(op="snapshot_load", stage="done", tl=tl(), pos=pos, cur=cur)
    return {"pos": pos, "cur": cur, "delta_rows": [r0, r1]}

  def handle(conn, req):
    rid = req.get("id"); m = req.get("method"); p = req.get("params") or {}
    idle_before = time.time() - ST.last_rpc; ST.last_rpc = time.time()
    slog(op="rpc", method=m, tl=tl(), idle_before=round(idle_before, 1), busy=ST.busy,
         rid=str((p or {}).get("rid") or "")[:40])   # R3-48: end-to-end correlation
    # V-24: privileged methods need the ops token (fail-closed when unset).
    # prefill-with-snapshot reads an arbitrary path into the GPU -> privileged.
    if m in _PRIV_METHODS or (m == "prefill" and "snapshot" in p):
      ok, why = check_admin(p)
      if not ok:
        send(conn, {"id": rid, "ok": False, "error": f"admin required: {why}"})
        slog(op="admin_denied", method=m)
        return None
    # V-23/V-24: input caps BEFORE any seed_hist/win_up — a rejected request
    # never touches engine state (client error, not dirty).
    if m in ("generate", "prefill"):
      err, p2 = validate_rpc(m, p, CTXK, ST.vocab or (1 << 30), ST.pos_cache)
      if err is not None:
        send(conn, {"id": rid, "ok": False, "error": f"invalid params: {err}"})
        slog(op="rpc_rejected", method=m, error=err)
        return None
      p = p2
    # V-17: busy + rpc name for ANY GPU-owning RPC (a FRESH 100k prefill runs
    # minutes while the old code reported busy=False).
    gpu_rpc = m in _GPU_RPC
    if gpu_rpc:
      _gpu_rpc_entry()                 # L7 FIX 1+5: clean cancel slate + budget spend-down
      ST.busy = True; ST.rpc = m; _beat()
      ST.active_conn = conn            # R3-21: cancel scoping
    try:
      if m == "generate":
        r = h_generate(conn, rid, p)
        ST.last_rpc = time.time()
        return r
      if m == "status": r = h_status(p)
      elif m == "prefill":
        r = h_prefill(conn, rid, p)
        ST.dirty = False   # W1 V-10: a completed prefill re-establishes state
      elif m == "snapshot_save": r = h_snapshot_save(p)
      elif m == "snapshot_load": r = h_snapshot_load(p)
      elif m == "cancel": return {"cancelled": True}   # flag already set by listener
      elif m == "shutdown":
        send(conn, {"id": rid, "ok": True, "result": {"bye": True}})
        dev.synchronize()
        # V-36: shutdown is an OPERATOR intent — flush queued pcache nodes,
        # unlink the socket, and leave the persistent staydown marker so the
        # launchd KeepAlive restart honors the stop instead of rebooting a
        # config the operator just chose to end. enginectl clear-breaker re-enables.
        slog(op="shutdown", stage="clean_exit")
        _clean_exit(0, "shutdown_rpc", flush=True, staydown=True)
      else: raise ValueError(f"unknown method {m}")
      send(conn, {"id": rid, "ok": True, "result": r})
      ST.last_rpc = time.time()
      return r
    except _PrefillCancelled:
      # L7 FIX 4 (legacy): drain-then-release — the Q must not hand the GPU to
      # the next RPC while the aborted prefill's kernels may still be in
      # flight (cancel_checkpoint drained; this sync is the belt), the
      # resident conversation is INVALIDATED (dirty + unbound + unfed), and
      # the client gets a clean cancelled reply.
      _beat()
      dev.synchronize()
      ST.dirty = True
      ST.convo_id = None; ST.fed = []; ST.mode = None
      ST.pos_cache = 0; ST.cur_cache = 0
      try: send(conn, {"id": rid, "ok": False, "error": "cancelled"})
      except Exception: pass
      slog(op="prefill", stage="cancelled", tl=tl(),
           armed_by=ST.cancel_armed_by, gctr=getattr(dev, "global_cycle_ctr", -1))
      return None
    except Exception as e:
      import traceback; traceback.print_exc()
      try: dev.synchronize()          # L7: a mid-GPU fault must not leak in-flight work either
      except Exception: pass
      slog(op="rpc_error", method=m, error=repr(e), tb=traceback.format_exc()[-2000:])
      try: send(conn, {"id": rid, "ok": False, "error": str(e)})
      except Exception: pass
      # W1 V-10: a faulted engine-mutating RPC leaves untrustworthy state —
      # surface it via status so the API refuses FOLLOW_UP until a fresh prefill.
      if m in ("prefill", "generate", "snapshot_load"): ST.dirty = True
      # DEVICE-FAULT EXIT LAW: once the dext is in err_state every later op
      # fails; a faulted daemon is a zombie. Exit NOW — the supervisor
      # (launchd KeepAlive + circuit breaker) reboots the conversation cleanly.
      # V-37: through _clean_exit (SOCK unlink + pcache flush).
      if "Device fault" in repr(e) or "device hang" in repr(e).lower():
        slog(op="device_fault", stage="exiting")
        _clean_exit(1, "device_fault", flush=True)
      return None
    finally:
      if gpu_rpc:
        ST.busy = False; ST.rpc = None
        if ST.active_conn is conn:
          ST.active_conn = None        # R3-21

  # ===================== R6 PHASE 3: the batch scheduler =====================
  def _batch_main(Q, listener, heartbeat, watchdog):
    import collections
    import r6_serve
    from gcycle import GCycleEngine
    from mtp import RBLK, CBLK
    Swap = r6h.Swap
    DEEP_TRIG = max(1, int(getattr(_mtp, "DEEP_TRIG", 1)))
    R6 = r6_serve.R6Scheduler(E)

    # ---- per-slot trunk engines for the T=1 prefill path (fixed handles ->
    # built ONCE; every eager op of slot s runs under Swap(s)) ----
    ST.busy = True; ST.rpc = "boot"; _beat()   # R3-04b: batch boot is watched
    try:
      with Swap(1):
        if hasattr(E, "_seq"): del E._seq
        Gs1 = GCycleEngine(E); Gs1.build()
      if hasattr(E, "_seq"): del E._seq   # nothing rebuilds the trunk seqs after this
      dev.synchronize()
      _beat()
    finally:
      ST.busy = False; ST.rpc = None
    slog(op="r6_batch_boot", graphs=len(R6.g), trunk_engines=2)

    class _BStream:
      def __init__(self, s):
        self.s = s; self.reset()
      def reset(self):
        self.model_id = None     # R3-42: (model_id, cid) residency keying
        self.convo_id = None; self.conn = None; self.fed = []
        self.mode = None; self.pos_cache = 0; self.cur_cache = 0
        self.pc_last_end = 0; self.pc_last_hkey = None; self.pc_protect = set()
        self.dirty = False; self.used = False; self.last_used = time.time()
        self.cancel_req = False; self.gen = None; self.cur_override = None
        self.deep = 0; self.hitrun = 0
    streams = [ _BStream(0), _BStream(1) ]
    by_conn = {}
    pending = collections.deque()
    ST.bengine = None            # None | "sess" | "r6" (transition anchoring)
    ST.barrier = False           # a prefill-class op owns the GPU

    def _trunk_G(s): return G if s == 0 else Gs1

    def _reprotect():
      if PC is None: return
      u = set()
      for st in streams: u |= st.pc_protect
      PC.set_protect(u)          # W3: under PC.lock (V-41)

    def _pc_node_done_st(st, hk, B):
      st.pc_last_end = B; st.pc_last_hkey = hk; st.pc_protect.add(hk)
      _reprotect()

    def _pc_ingest_cb_st(st, toks):
      def cb(pos_after, final_b):
        want = (pos_after % pcache.STRIDE == 0 and pos_after >= pcache.STRIDE) or \
               (pos_after == final_b and pos_after >= pcache.HASH_BLK)
        if not want: return
        A = st.pc_last_end
        if pos_after - A < pcache.HASH_BLK: return
        t0 = time.perf_counter()
        # T=1-world boundary state (REC1/xA64 do not exist): dhd = the draft
        # chain's running hidden (hd_d1, slot bank under the swap); hlast =
        # the trunk hidden after the last fed token (x0 — the trunk graph
        # always lands its final hidden in x0 for the head read).
        dhd = E.P.down_at("hd_d1", 0, 5120, np.float32)
        hlast = E.P.down_at("x0", 0, 5120, np.float32)
        cur = int(E.P.down_at("tok_slot", 0, 1)[0])
        node = pcache.capture_node(E, A, pos_after, "midprefill", fed_prefix=toks,
                                   parent=st.pc_last_hkey, dhd=dhd, hlast=hlast, cur=cur)
        PC.write_node(node)
        _pc_node_done_st(st, node["hkey"], pos_after)
        slog(op="pc_ingest", slot=st.s, pos=pos_after, win=[A, pos_after],
             secs=round(time.perf_counter() - t0, 2), wq=PC.wq.qsize())
      return cb

    def _pc_turn_end_st(st, pos):
      if PC is None: return
      if pos % pcache.HASH_BLK != 0 or pos - st.pc_last_end < pcache.HASH_BLK: return
      if pos > len(st.fed): return
      try:
        t0 = time.perf_counter()
        with Swap(st.s):
          node = pcache.capture_node(E, st.pc_last_end, pos, "turnend", fed_prefix=st.fed, parent=st.pc_last_hkey)
          PC.write_node(node)
        _pc_node_done_st(st, node["hkey"], pos)
        slog(op="pc_turn_end_ingest", slot=st.s, pos=pos, win_secs=round(time.perf_counter() - t0, 2))
      except Exception as e:
        slog(op="pc_turn_end_error", slot=st.s, error=repr(e))

    # L7: the batch-local `class _PrefillCancelled(Exception)` shadow is
    # DELETED — ONE module-level class is raised by the checkpoints and caught
    # by the handlers (two exception types for one protocol was how an abort
    # only one path repaired became possible).

    def _seed_hist_sw(fed):
      # caller must be inside Swap(st.s)
      if LK_ACTIVE and len(fed):
        E.P.win_up("tok_hist", 0, np.asarray(fed, dtype=np.int32))
        dev.synchronize()

    # ---------------- L7 abort-safety (batch halves) ----------------
    def _b_gpu_rpc_entry():
      """L7 FIX 1+5 (batch): clean cancel slate + global graph-submit budget
      spend-down at barrier-prefill start (rotated fence)."""
      ST.cancel = False; ST.cancel_armed_by = None
      if GLOBAL_REBUILD_EVERY and getattr(dev, "global_cycle_ctr", 0) >= GLOBAL_REBUILD_EVERY:
        _fence_all_rebuild()

    def _b_cancel_checkpoint(st, stage):
      """L7 FIX 2 (batch): per-stream cancel checkpoint — the only legal raise.
      Drain-then-raise; the barrier handler re-syncs as the belt."""
      if not (st.cancel_req or ST.cancel):
        return
      _beat()
      dev.synchronize()
      slog(op="abort", stage=stage, slot=st.s, tl=tl(),
           armed_by=ST.cancel_armed_by, cycles_since_rebuild=ST.cycles_since_rebuild,
           gctr=getattr(dev, "global_cycle_ctr", -1))
      raise _PrefillCancelled()

    def _mk_prog_b(st, conn, rid):
      def prog(done, total, stage="prefill_t1"):
        _beat()
        send(conn, {"id": rid, "event": "prefill_progress", "stage": stage,
                    "done": done, "total": total}, abort_on_fail=True,
             cancel_fn=lambda: (setattr(st, "cancel_req", True),
                                setattr(ST, "cancel_armed_by", ("send_stalled", st.s))))
        _b_cancel_checkpoint(st, stage)
      return prog

    # ---------------- barrier prefill (per stream, swap-scoped) ----------------
    def _b_fresh(st, conn, rid, toks):
      prog = _mk_prog_b(st, conn, rid)     # L7: beat+send; checkpoint raises
      slog(op="prefill_fresh", stage="begin", slot=st.s, tl=tl(), n=len(toks))
      with Swap(st.s):
        E.reset_fresh(toks[0])
        E.stload_trunk()   # FRESH runs from zero GDN state (M1-B FIX #1/#2)
        for _i in E.gdn_idx: E._mfill(f"conv{_i}_1", 0, CBLK)
        dev.synchronize()
      ingest = _pc_ingest_cb_st(st, toks) if (PC is not None and len(toks) >= 64) else None
      CH = max(64, BATCH_PF_CHUNK)
      # The draft KV fills INTERLEAVED per chunk (trunk chunk, then the draft
      # chain over the same span seeded from hd_d1) — same total work as the
      # one-shot fill, but the chain state sits at every chunk boundary so a
      # mid-prefill node capture has its dhd, and kv_d rows trail the trunk.
      # NB: each chunk opens its OWN Swap scope — between chunks P.d is
      # canonical so _interleave()'s batch_step reads emits/graph state by
      # their CANONICAL names (under Swap(1) P.d["emit"] would resolve to
      # emit_s1: the name-keyed readout hazard).
      dhd_np = None
      for i in range(0, len(toks), CH):
        _b_cancel_checkpoint(st, "pf_chunk")   # L7: quiescent (prev chunk + fill_draft synced)
        with Swap(st.s):
          E.prefill_t1(_trunk_G(st.s), toks[i:i + CH], prog=prog)
          E.fill_draft(toks[i:i + CH], start_pos=i, seed_hd=dhd_np,
                       prog=lambda d, t: prog(d, t, "fill_draft"))
          dhd_np = E.P.down_at("hd_d1", 0, 5120, np.float32)
          pos_after = int(E.P.down_at("pos_slot", 0, 1)[0])
          if ingest is not None and pos_after % pcache.HASH_BLK == 0:
            ingest(pos_after, len(toks))
        _interleave()      # fairness: other streams' decode + Q drain between chunks
      with Swap(st.s):
        E.stseed_spec(len(toks) & 1)
        newcur = int(E.P.down_at("tok_slot", 0, 1)[0])
        posn = int(E.P.down_at("pos_slot", 0, 1)[0])
        E.P.win_up("cur_slot", 0, np.array([newcur], dtype=np.int32))
        E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
        E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
        E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
        dev.synchronize()
        st.fed = list(toks); st.mode = "FRESH"
        _seed_hist_sw(toks)
        if PC is not None: st.pc_last_end = 0; st.pc_last_hkey = None; st.pc_protect = set(); _reprotect()
        if ingest is not None and posn % pcache.HASH_BLK == 0:
          ingest(posn, len(toks))     # final node at the deepest 64-block
        st.pos_cache = posn; st.cur_cache = newcur
      r = {"pos": st.pos_cache, "cur": st.cur_cache, "fed": len(toks), "mode": "FRESH", "cached_tokens": 0}
      slog(op="prefill_fresh", stage="done", slot=st.s, tl=tl(), **r)
      return r

    def _b_followup(st, conn, rid, toks):
      with Swap(st.s):
        if st.cur_override is not None:
          E.P.win_up("cur_slot", 0, np.array([int(st.cur_override)], dtype=np.int32))
          st.cur_override = None
        cur = int(E.P.down_at("cur_slot", 0, 1)[0])
        _beat()
        pos0 = int(E.P.down_at("pos_slot", 0, 1)[0])
        slog(op="follow_up", stage="begin", slot=st.s, tl=tl(), n=len(toks), cur=cur, pos0=pos0)
        prog = _mk_prog_b(st, conn, rid)     # L7: beat+send; checkpoint raises
        # R6 VRAM LAW: the PF chunk path faults with the batch banks resident —
        # FOLLOW_UP always takes the T=1 trunk path here (batch=False).
        newcur, posn, nd = E.follow_up(_trunk_G(st.s), toks,
                                       log=lambda s, **kw: slog(op="follow_up", stage=s, slot=st.s, tl=tl(), **kw),
                                       prog=prog, batch=False)
        st.fed = st.fed + [cur] + toks
        _seed_hist_sw(st.fed)
        st.mode = "FOLLOW_UP"
        st.pos_cache = posn; st.cur_cache = newcur
      r = {"pos": posn, "cur": newcur, "fed": nd}
      slog(op="follow_up", stage="done", slot=st.s, tl=tl(), **r)
      return r

    def _b_autocache(st, conn, rid, p, toks):
      cache_key = p.get("cache_key") or p.get("prompt_cache_key")
      ttl = p.get("cache_ttl") or p.get("prompt_cache_ttl")
      min_cov = max(pcache.MIN_HIT, int(0.5 * len(toks)))
      t0 = time.perf_counter()
      B, chain = PC.lookup(toks, min_hit=min_cov)
      if not chain:
        slog(op="pc_lookup", stage="miss", slot=st.s, tl=tl(), n=len(toks), secs=round(time.perf_counter() - t0, 2))
        return None
      slog(op="pc_lookup", stage="hit", slot=st.s, tl=tl(), B=B, nodes=len(chain),
           cached_gb=round(sum(int(e["bytes"]) for _, e in chain) / 1e9, 2))
      PC.touch(chain)
      if cache_key: PC.pin(chain, str(cache_key)[:128], ttl=ttl)
      st.pc_protect = set(hk for hk, _ in chain); _reprotect()
      prog = _mk_prog_b(st, conn, rid)     # L7: beat+send; checkpoint raises
      with Swap(st.s):
        t0 = time.perf_counter()
        try:
          B2, cur = pcache.restore_chain(E, chain, beat=_beat); assert B2 == B
        except pcache.NodeCorrupt as nc:
          slog(op="pc_corrupt", slot=st.s, hkey=(nc.hkey or "")[:16], why=nc.why, tl=tl())
          PC.quarantine(nc.hkey)
          return None       # -> _b_fresh (validation precedes any upload)
        st.fed = list(toks[:B]); st.mode = "CACHE_HIT"; st.convo_id = p.get("conversation_id")
        _seed_hist_sw(st.fed)
        _pc_node_done_st(st, chain[-1][0], B)
        if len(toks) > B:
          E.P.win_up("cur_slot", 0, np.array([int(toks[B])], dtype=np.int32))
          newcur, posn, nd = E.follow_up(_trunk_G(st.s), toks[B + 1:],
                                         log=lambda s, **kw: slog(op="follow_up", stage=s, slot=st.s, tl=tl(), **kw),
                                         prog=prog, batch=False)
          st.fed = list(toks)
          _seed_hist_sw(st.fed)
          slog(op="pc_tail", stage="done", slot=st.s, tl=tl(), n=nd, secs=round(time.perf_counter() - t0, 1))
        else:
          newcur, posn = cur, B
        st.pos_cache = posn; st.cur_cache = newcur
      r = {"pos": posn, "cur": newcur, "fed": len(st.fed), "mode": "CACHE_HIT", "cached_tokens": B}
      slog(op="prefill_cache_hit", stage="done", slot=st.s, tl=tl(), pos=posn, cur=newcur, cached_tokens=B)
      return r

    # ---------------- slot selection / binding ----------------
    def _pick_slot(p):
      cid = p.get("conversation_id")
      mid = p.get("model_id")      # R3-42: same cid across models must not
      mode = p.get("mode", "FRESH")  # cross-match (pre-MoE: always equal)
      if mode == "FOLLOW_UP":
        for st in streams:
          if st.convo_id == cid and not st.gen and st.model_id == mid:
            return st
        raise ValueError(f"FOLLOW_UP conversation_id mismatch: no resident conversation "
                         f"{cid!r} (model {mid!r}) on any slot (evicted? dirty? use FRESH)")
      for st in streams:      # a slot already pinned to this conv continues it
        if cid is not None and st.convo_id == cid and not st.gen and st.model_id == mid:
          return st
      for st in streams:      # pristine slot (park counts as evictable)
        if not st.used and not st.gen:
          return st
      idle = [st for st in streams if not st.gen]
      if idle:
        return min(idle, key=lambda st: st.last_used)    # LRU eviction
      raise ValueError("engine busy: all batch slots are generating")

    def _finish_gen(st, kind, last_r=None):
      g = st.gen; st.gen = None; st.last_used = time.time()
      conn = g["conn"]
      if st.conn is not conn:
        # R3-16: the generating conn was unbound at teardown (client gone) —
        # terminal frames to a dead fd (or a reused fd's NEW owner) are
        # skipped; the result is only in slog.
        slog(op="generate", stage="terminal_skipped_dead_conn", slot=st.s, kind=kind)
        return
      pos = (last_r or {}).get("pos_new", st.pos_cache)
      try:
        if kind == "cancelled":
          send(conn, {"id": g["rid"], "event": "cancelled", "tokens": g["all_toks"], "cycles": g["k"]},
               cancel_fn=lambda: None)
          send(conn, {"id": g["rid"], "ok": True, "result": {"cancelled": True, "tokens": g["all_toks"], "cycles": g["k"]}},
               cancel_fn=lambda: None)
          slog(op="generate", stage="cancelled", slot=st.s, cycles=g["k"], ntoks=len(g["all_toks"]), tl=tl())
          return
        if kind == "stop":
          _pc_turn_end_st(st, pos)
        send(conn, {"id": g["rid"], "event": "done", "tokens": g["all_toks"], "cycles": g["k"], "pos": pos,
                    "stop": kind == "stop",
                    "usage": {"tokens": len(g["all_toks"]), "cycles": g["k"],
                              "secs": round(time.perf_counter() - g["t0"], 2)}}, cancel_fn=lambda: None)
        res = {"tokens": g["all_toks"], "cycles": g["k"]}
        if kind == "stop": res["stop"] = True
        send(conn, {"id": g["rid"], "ok": True, "result": res}, cancel_fn=lambda: None)
        slog(op="generate", stage="done_stop" if kind == "stop" else "done_max",
             slot=st.s, cycles=g["k"], ntoks=len(g["all_toks"]), stop=kind == "stop")
      except Exception as e:
        slog(op="generate", stage="terminal_send_error", slot=st.s, error=repr(e))

    def _process_emit(st, r):
      g = st.gen
      # W4.3 emit-sequence monotonicity (per stream): violation = dirty stream
      # + error event + cancel the generate (never stream silently).
      _viol = _emit_seq_violation(st.pos_cache, r)
      if _viol is not None:
        st.dirty = True
        slog(op="emit_seq_violation", slot=st.s, violation=_viol, k=g["k"], tl=tl())
        try:
          send(g["conn"], {"id": g["rid"], "event": "error", "error": f"emit_seq_violation: {_viol}"},
               cancel_fn=lambda: None)
        except Exception: pass
        _finish_gen(st, "cancelled")
        return
      st.pos_cache = r["pos_new"]
      g["k"] += 1; g["k_left"] -= 1
      g["all_toks"] += r["tokens"]
      st.fed.extend(r["tokens"])                    # V-21 extend
      # READOUT-ORDER law: next cycle's mode comes from THIS completed emit
      st.hitrun = st.hitrun + 1 if r["hit"] >= 9 else 0
      st.deep = 1 if st.hitrun >= DEEP_TRIG else 0
      _beat()
      if CYCLE_SLOG: slog(op="cycle", slot=st.s, k=g["k"], tl=tl(), pos=r["pos_new"], m=r["m"], deep=st.deep)
      if st.conn is None or st.conn is not g["conn"]:
        g["cancel"] = True                  # R3-16: unbound/dead client — stop
        return
      try:
        _cyc = r.get("cycle", r.get("cyc", 0))
        send(st.conn, {"id": g["rid"], "event": "cycle", "cycle": _cyc, "pos": r["pos_new"],
                       "tokens": r["tokens"]}, abort_on_fail=True,
             cancel_fn=lambda: g.update(cancel=True) if st.gen else None)
      except Exception as _e:
        slog(op="cycle_send_error", slot=st.s, k=g["k"], error=repr(_e))
      if st.gen is None: return                      # send-abort already finished us? (defensive)
      if g["stops"] & set(r["tokens"]):
        _finish_gen(st, "stop", r); return
      if g["k_left"] <= 0:
        _finish_gen(st, "max", r); return

    _fence_i = [0]
    def _fence_all_rebuild():
      # ROTATED fence: one graph set per event (canonical -> solo1 -> b35 ->
      # b53 -> b55). ParityGraph allocates a fresh nolru kernargs slab per
      # build that is never released — a FULL fence every 232 cycles leaked
      # ~22 slabs/fence and exhausted host-mapped memory after ~250 fences
      # (alloc_sysmem IndexError, generates cancelled). Rotation buys 4-5x;
      # the durable fix is ParityGraph ka-slab reuse (documented follow-up).
      # A FAILED fence is non-fatal: the old graphs remain valid objects, the
      # counter still resets (retry on the next window).
      t0 = time.perf_counter()
      which = _fence_i[0] % 5; _fence_i[0] += 1
      slog(op="gen_rebuild_global", stage="begin", which=which, cycles=ST.cycles_since_rebuild, tl=tl())
      _beat()
      try:
        dev.synchronize()        # kimi(b).8: fence EVERYTHING before graph drops
        if which == 0:
          _rebuild_capacity_asserts()
          E.build_graphs()
          sess.begin(); ST.bengine = None
        else:
          R6.rebuild_one(which - 1)
          R6.begin()
        ST.cycles_since_rebuild = 0
        dev.global_cycle_ctr = 0         # L7 FIX 5: the fence resets the GLOBAL submit budget
        ST.rebuild_fails = 0            # R3-06: reset ONLY on success
        slog(op="gen_rebuild_global", stage="done", which=which,
             secs=round(time.perf_counter() - t0, 2), tl=tl())
      except Exception as e:
        # R3-06: old graphs stay live (retry next window) BUT the counter no
        # longer resets on failure — 3 failed fences in a row (or >=800
        # cycles while failing) = the reboot-class escalation gap: exit
        # cleanly inside the safety margin, with attribution.
        ST.rebuild_fails += 1
        slog(op="gen_rebuild_global", stage="failed_nonfatal", which=which,
             fails=ST.rebuild_fails, error=repr(e))
        if ST.rebuild_fails >= 3 or ST.cycles_since_rebuild >= 800:
          slog(op="rebuild_budget_exhausted", fails=ST.rebuild_fails,
               cycles_since_rebuild=ST.cycles_since_rebuild, tl=tl())
          _clean_exit(1, "rebuild_budget_exhausted", flush=True)
      _beat()

    def batch_step():
      gens = [st for st in streams if st.gen]
      for st in gens:                     # per-stream cancel (checked pre-submit)
        if st.gen["cancel"] or st.cancel_req:
          _finish_gen(st, "cancelled")
      gens = [st for st in streams if st.gen]
      if not gens: return
      _beat()
      if len(gens) == 1 and gens[0].s == 0:
        # slot-0 solo: the CANONICAL DecodeSession (full deep-K + its own
        # data-dependent graph-set selection). Transition = begin() re-anchor.
        if ST.bengine != "sess":
          sess.begin(); ST.bengine = "sess"
        res = {0: sess.step()}
        streams[0].deep = sess.deep       # composition continuity on re-batch
      else:
        if ST.bengine != "r6":
          R6.begin(); ST.bengine = "r6"
        res = R6.step([st.s for st in gens], {st.s: st.deep for st in gens})
        if len(gens) == 1:
          gens[0].deep = 0                # solo k2: no deep next cycle from here
      ST.cycles_since_rebuild += 1        # THE GLOBAL counter (any composition)
      for st in gens:
        if st.gen is None: continue
        if st.conn is None and st.gen is not None:
          st.gen["cancel"] = True          # dead client (unbind hook fired)
        _process_emit(st, res[st.s])
      if BATCH_REBUILD_EVERY and (ST.cycles_since_rebuild >= BATCH_REBUILD_EVERY
          or (GLOBAL_REBUILD_EVERY and getattr(dev, "global_cycle_ctr", 0) >= GLOBAL_REBUILD_EVERY)):
        _fence_all_rebuild()   # L7 FIX 5: prefill chunk-graph debt is now visible here too
      ST.busy = ST.barrier or any(st.gen is not None for st in streams)
      ST.rpc = ("prefill" if ST.barrier else "generate") if ST.busy else None

    def _interleave():
      """Between prefill chunks: drain the Q (generate attaches; a second
      prefill parks in `pending`; R3-18: snapshot ops park too — a snapshot
      dequeued here used to reset_snapshot+win_up the canonical banks the
      barrier prefill was using), then one batch step for the others."""
      while True:
        try: conn2, req2 = Q.get_nowait()
        except queue.Empty: break
        m2 = req2.get("method"); p2 = req2.get("params") or {}
        if m2 in ("snapshot_save", "snapshot_load") or (m2 == "prefill" and "snapshot" in p2):
          if len(pending) < MAX_PENDING:
            pending.append((conn2, req2, m2))
          else:
            send(conn2, {"id": req2.get("id"), "ok": False,
                         "error": f"engine pending queue full ({MAX_PENDING}); retry shortly"})
            slog(op="pending_full", method=m2, qsize=len(pending))
          continue
        try: b_handle(conn2, req2)
        except Exception as e:
          slog(op="interleave_rpc_error", error=repr(e))
      if any(st.gen for st in streams):
        try:
          batch_step()
        except Exception as e:
          slog(op="interleave_step_error", error=repr(e))

    # ---------------- batch RPC handlers ----------------
    def b_status(p):
      s0 = streams[0]
      return {"ready": ST.ready, "ctxk": CTXK, "busy": ST.busy, "rpc": ST.rpc,
              "batch_b": 2, "barrier": ST.barrier,
              "pos": s0.pos_cache, "mode": s0.mode, "cur": s0.cur_cache,
              "fed_len": len(s0.fed), "fed_tail": s0.fed[-64:],
              "conversation_id": s0.convo_id,
              "queue": Q.qsize(), "dirty": s0.dirty,
              "keepalive_s": KEEPALIVE_S, "keep_len": len(E.P._keep),
              "uptime_s": round(time.time() - ST.t0, 1),
              "config_fp": ST.config_fp, "lookup_k": os.getenv("LOOKUP_K"),
        "ka": _ka_slog_fields(),
              "pf_prefill": os.getenv("PF_PREFILL"),
              "model_id": _MODEL_ID,          # R3-42 (pre-MoE: env or default)
              "cycles_since_rebuild": ST.cycles_since_rebuild,
              "cycle_cap": max(1, min(4096, CTXK - max(0, int(streams[0].pos_cache)))),  # R3-34
              "pc": _pc_status(),             # R3-28
              "streams": [{"slot": st.s, "conversation_id": st.convo_id,
                           "model_id": st.model_id,
                           "pos": st.pos_cache, "cur": st.cur_cache, "mode": st.mode,
                           "fed_len": len(st.fed), "generating": bool(st.gen),
                           "dirty": st.dirty, "used": st.used} for st in streams]}

    def _b_prefill(conn, rid, p):
      if ST.barrier:
        if len(pending) >= MAX_PENDING:      # R3-13: bounded, LOUD rejection
          send(conn, {"id": rid, "ok": False,
                      "error": f"engine pending queue full ({MAX_PENDING}); retry shortly"})
          slog(op="pending_full", method="prefill", qsize=len(pending))
          return
        pending.append((conn, {"id": rid, "method": "prefill", "params": p}, "prefill"))
        slog(op="prefill", stage="queued_behind_barrier", qsize=len(pending))
        return
      # R3-08 engine belt: a conversation already bound to a GENERATING slot
      # must never be bound to a second slot (the API holds the conv lock for
      # the stream lifetime; this catches any TOCTOU window above it).
      _cid = p.get("conversation_id")
      if _cid is not None:
        for _st in streams:
          if _st.convo_id == _cid and _st.gen is not None:
            send(conn, {"id": rid, "ok": False,
                        "error": f"conversation busy: {_cid!r} is generating on "
                                 f"slot {_st.s}; wait for the stream to end"})
            slog(op="rpc_rejected", method="prefill", error="conversation_busy",
                 convo=_cid, slot=_st.s)
            return
      try:
        st = _pick_slot(p)
      except ValueError as e:
        send(conn, {"id": rid, "ok": False, "error": str(e)})
        slog(op="rpc_rejected", method="prefill", error=str(e))
        return
      if st.convo_id is not None and p.get("conversation_id") != st.convo_id:
        slog(op="slot_evict", slot=st.s, old_convo=st.convo_id, new_convo=p.get("conversation_id"),
             old_fed=len(st.fed))
      st.convo_id = p.get("conversation_id")
      st.model_id = p.get("model_id")      # R3-42
      st.conn = conn; by_conn[conn] = st.s
      st.used = True; st.last_used = time.time(); st.cancel_req = False
      st.dirty = False; st.gen = None; st.cur_override = p.get("cur")
      _b_gpu_rpc_entry()                   # L7 FIX 1+5: clean slate + budget spend-down
      ST._pf_stream = st                   # L7: the mtp.CANCEL_CHECK target stream
      ST.barrier = True; ST.rpc = "prefill"; ST.busy = True; _beat()
      toks = [int(t) for t in p["ids"]]
      try:
        if p.get("mode", "FRESH") == "FOLLOW_UP":
          r = _b_followup(st, conn, rid, toks)
        elif PC is not None and p.get("mode", "FRESH") in ("FRESH", "AUTO_CACHE"):
          r = _b_autocache(st, conn, rid, p, toks)
          if r is None:
            r = _b_fresh(st, conn, rid, toks)
        else:
          r = _b_fresh(st, conn, rid, toks)
        send(conn, {"id": rid, "ok": True, "result": r})
      except _PrefillCancelled:
        # L7 FIX 4 (batch): THE FENCE — the barrier must not drop while the
        # device may still execute the aborted prefill's kernels; the slot is
        # fully invalidated (unbound + unfed + dirty) so only a FRESH prefill
        # can ever re-use it (FOLLOW_UP cannot match a None convo).
        _beat()
        t_ab = time.perf_counter()
        dev.synchronize()
        st.dirty = True
        st.convo_id = None; st.fed = []; st.mode = None
        st.pos_cache = 0; st.cur_cache = 0
        send(conn, {"id": rid, "ok": False, "error": "cancelled"})
        slog(op="prefill", stage="cancelled", slot=st.s, tl=tl(),
             fence_ms=round((time.perf_counter() - t_ab) * 1e3, 1),
             armed_by=ST.cancel_armed_by, gctr=getattr(dev, "global_cycle_ctr", -1))
      except Exception as e:
        import traceback; traceback.print_exc()
        try: dev.synchronize()          # L7: no in-flight leak on the fault path either
        except Exception: pass
        st.dirty = True
        try: send(conn, {"id": rid, "ok": False, "error": str(e)})
        except Exception: pass
        slog(op="rpc_error", method="prefill", slot=st.s, error=repr(e), tb=traceback.format_exc()[-2000:])
        if "Device fault" in repr(e) or "device hang" in repr(e).lower():
          slog(op="device_fault", stage="exiting")
          _clean_exit(1, "device_fault", flush=True)
        raise
      finally:
        ST._pf_stream = None             # L7: detach the checkpoint target
        ST.barrier = False; ST.rpc = None
        ST.busy = any(st.gen for st in streams)
        # NB: do NOT reset the global rebuild counter here — the R6_PF_T1
        # prefill is GRAPH-CLASS work (trunk graph replays), not the eager-
        # class work that reset the dext budget in the M1-C law. The soak
        # wedged deterministically at ~4000-4500 continuous mixed graph
        # cycles with counter resets; ONLY the fence-all rebuild resets it.
        _beat()

    def _b_generate(conn, rid, p):
      s = by_conn.get(conn)
      if s is None:
        send(conn, {"id": rid, "ok": False, "error": "no prefill on this connection (send prefill first)"})
        return
      st = streams[s]
      if st.gen is not None:
        send(conn, {"id": rid, "ok": False, "error": "slot already generating"})
        return
      if st.dirty:
        send(conn, {"id": rid, "ok": False, "error": "slot state dirty; re-prefill (FRESH) first"})
        return
      mc = int(p.get("max_cycles", 60)); stops = set(int(t) for t in p.get("stop_token_ids", []))
      st.gen = {"rid": rid, "conn": conn, "stops": stops, "k_left": mc, "k": 0,
                "all_toks": [], "t0": time.perf_counter(), "cancel": False}
      ST.busy = True; ST.rpc = "generate"
      slog(op="generate", stage="attached", slot=st.s, tl=tl(), max_cycles=mc, convo=st.convo_id)
      # NO reply here: cycle events flow, the terminal sends done/cancelled + result.

    def b_handle(conn, req):
      rid = req.get("id"); m = req.get("method"); p = req.get("params") or {}
      idle_before = time.time() - ST.last_rpc; ST.last_rpc = time.time()
      slog(op="rpc", method=m, tl=tl(), idle_before=round(idle_before, 1), busy=ST.busy,
           q=Q.qsize(), rid=str((p or {}).get("rid") or "")[:40])   # R3-48
      if m in _PRIV_METHODS or (m == "prefill" and "snapshot" in p):
        ok, why = check_admin(p)
        if not ok:
          send(conn, {"id": rid, "ok": False, "error": f"admin required: {why}"})
          slog(op="admin_denied", method=m)
          return None
      if m in ("generate", "prefill") and "snapshot" not in p:
        # per-stream pos for the FOLLOW_UP overflow / max_cycles caps
        st_pos = 0
        if m == "generate":
          s = by_conn.get(conn)
          if s is not None: st_pos = streams[s].pos_cache
        else:
          cid = p.get("conversation_id")
          if p.get("mode", "FRESH") == "FOLLOW_UP":
            cand = [st for st in streams if st.convo_id == cid]
            st_pos = cand[0].pos_cache if cand else 0
        err, p2 = validate_rpc(m, p, CTXK, ST.vocab or (1 << 30), st_pos)
        if err is not None:
          send(conn, {"id": rid, "ok": False, "error": f"invalid params: {err}"})
          slog(op="rpc_rejected", method=m, error=err)
          return None
        p = p2
      gpu_rpc = m in _GPU_RPC
      if gpu_rpc: _beat()
      try:
        if m == "status":
          r = b_status(p); send(conn, {"id": rid, "ok": True, "result": r}); return r
        if m == "prefill":
          if "snapshot" in p:
            # snapshot restore = full canonical-world clobber: R3-18 widens
            # the guard to barrier/pending/ANY-used — the old check passed
            # MID-BARRIER (slot-0 barrier prefill sets used+ST.barrier while
            # _interleave drains this RPC between chunks: same-thread clobber)
            if ST.barrier or pending or any(st.used for st in streams) or any(st.gen for st in streams):
              send(conn, {"id": rid, "ok": False, "error": "snapshot prefill rejected: batch slots in use"})
              slog(op="prefill_snapshot", stage="rejected_batch")
              return None
            r = _snapshot_prefill(conn, rid, p)
            send(conn, {"id": rid, "ok": True, "result": r})
            return r
          _b_prefill(conn, rid, p)
          return None
        if m == "generate":
          _b_generate(conn, rid, p)
          return None
        if m == "snapshot_save" or m == "snapshot_load":
          if ST.barrier or pending or any(st.used for st in streams) or any(st.gen for st in streams):
            send(conn, {"id": rid, "ok": False, "error": f"{m} rejected: batch slots in use (multi-stream residency)"})
            slog(op=m, stage="rejected_batch")
            return None
          r = h_snapshot_save(p) if m == "snapshot_save" else h_snapshot_load(p)
          st0 = streams[0]
          st0.fed = list(ids); st0.mode = "snapshot"; st0.convo_id = None
          st0.used = True; st0.dirty = False; st0.gen = None
          st0.pc_last_end = 0; st0.pc_last_hkey = None; st0.pc_protect = set(); _reprotect()
          _resync_slot0()
          send(conn, {"id": rid, "ok": True, "result": r})
          return r
        if m == "shutdown":
          send(conn, {"id": rid, "ok": True, "result": {"bye": True}})
          dev.synchronize()
          slog(op="shutdown", stage="clean_exit")
          _clean_exit(0, "shutdown_rpc", flush=True, staydown=True)
        raise ValueError(f"unknown method {m}")
      except Exception as e:
        import traceback; traceback.print_exc()
        slog(op="rpc_error", method=m, error=repr(e), tb=traceback.format_exc()[-2000:])
        try: send(conn, {"id": rid, "ok": False, "error": str(e)})
        except Exception: pass
        if "Device fault" in repr(e) or "device hang" in repr(e).lower():
          slog(op="device_fault", stage="exiting")
          _clean_exit(1, "device_fault", flush=True)
        return None

    def _snapshot_prefill(conn, rid, p):
      _beat()
      slog(op="prefill_snapshot", stage="begin", tl=tl(), path=p["snapshot"])
      m = json.load(open(p["snapshot"] + "/meta.json"))
      E.reset_snapshot(p["snapshot"], int(m["cur0"]), int(m["P"]))
      st = streams[0]
      st.mode = "snapshot"; st.fed = list(ids); st.convo_id = p.get("conversation_id")
      st.used = True; st.dirty = False
      seed_hist(ids)
      st.pos_cache = int(E.P.down_at("pos_slot", 0, 1)[0]); st.cur_cache = int(E.P.down_at("cur_slot", 0, 1)[0])
      r = {"pos": st.pos_cache, "cur": st.cur_cache, "fed": 0}
      slog(op="prefill_snapshot", stage="done", tl=tl(), **r)
      return r

    def _resync_slot0():
      st = streams[0]
      st.pos_cache = int(E.P.down_at("pos_slot", 0, 1)[0])
      st.cur_cache = int(E.P.down_at("cur_slot", 0, 1)[0])

    # hooks for the listener threads
    ST.inline_status = lambda req_id: {"id": req_id, "ok": True, "result": {
        "ready": ST.ready, "ctxk": CTXK, "busy": ST.busy, "pos": streams[0].pos_cache,
        "rpc": ST.rpc, "mode": streams[0].mode, "cur": streams[0].cur_cache,
        "fed_len": len(streams[0].fed), "fed_tail": streams[0].fed[-16:],
        "conversation_id": streams[0].convo_id, "dirty": streams[0].dirty,
        "batch_b": 2, "barrier": ST.barrier,
        "queue": Q.qsize(), "keepalive_s": KEEPALIVE_S,
        "uptime_s": round(time.time() - ST.t0, 1),
        "config_fp": ST.config_fp, "lookup_k": os.getenv("LOOKUP_K"),
        "ka": _ka_slog_fields(),
        "pf_prefill": os.getenv("PF_PREFILL"),
        "cycles_since_rebuild": ST.cycles_since_rebuild,
        "streams": [{"slot": st.s, "conversation_id": st.convo_id, "pos": st.pos_cache,
                     "cur": st.cur_cache, "fed_len": len(st.fed), "mode": st.mode,
                     "generating": bool(st.gen), "dirty": st.dirty} for st in streams]}}
    def _cancel_hook(conn):
      s = by_conn.get(conn)
      if s is None:
        ST.cancel = True      # pre-binding cancel: legacy global (harmless)
        ST.cancel_armed_by = ("cancel_rpc", None)   # L7: attribution
        return
      st = streams[s]
      if st.gen is not None: st.gen["cancel"] = True
      st.cancel_req = True
      ST.cancel_armed_by = ("cancel_rpc", st.s)     # L7: attribution
    ST.cancel_hook = _cancel_hook
    def _unbind_hook(conn):
      s = by_conn.pop(conn, None)
      if s is not None and streams[s].conn is conn:
        streams[s].conn = None
      if pending:      # R3-13: dead conns' queued work never reaches the GPU
        dq = collections.deque(e for e in pending if e[0] is not conn)
        if len(dq) != len(pending):
          slog(op="pending_purged_dead_conn", before=len(pending), after=len(dq))
          pending.clear(); pending.extend(dq)
    ST.unbind_hook = _unbind_hook

    # slot-0 mirror of the park state (the park ran on canonical = slot 0)
    streams[0].fed = list(ids); streams[0].pos_cache = ST.pos_cache; streams[0].cur_cache = ST.cur_cache
    streams[0].mode = "park"

    threading.Thread(target=listener, daemon=True).start()
    # (watchdog + heartbeat already started pre-boot — R3-04b)
    slog(op="r6_batch_ready", batch_b=2, rebuild_every=BATCH_REBUILD_EVERY, pf_chunk=BATCH_PF_CHUNK)
    nka = 0
    while True:
      # pending RPCs first (queued while another prefill ran / parked by
      # _interleave); dispatched through b_handle so guards re-apply
      while pending and not ST.barrier:
        conn2, req2, m2 = pending.popleft()
        try: b_handle(conn2, req2)
        except Exception as e:
          slog(op="pending_rpc_error", method=m2, error=repr(e))
      any_gen = any(st.gen for st in streams)
      try:
        conn, req = Q.get(timeout=0 if any_gen else KEEPALIVE_S)
      except queue.Empty:
        conn = None
      if conn is not None:
        try: b_handle(conn, req)
        except Exception as e:
          slog(op="rpc_error", method=req.get("method"), error=repr(e))
        continue
      if any(st.gen for st in streams):
        try:
          batch_step()
        except Exception as e:
          import traceback; traceback.print_exc()
          slog(op="batch_step_error", error=repr(e), tb=traceback.format_exc()[-2000:])
          if "Device fault" in repr(e) or "device hang" in repr(e).lower():
            slog(op="device_fault", stage="exiting_from_batch_step")
            _clean_exit(1, "device_fault_batch_step", flush=True)
          for st in streams:
            if st.gen is not None:
              try: _finish_gen(st, "cancelled")
              except Exception: pass
          ST.busy = False; ST.rpc = None
        continue
      try:
        keepalive(); nka += 1
        if nka % 50 == 1: slog(op="keepalive", n=nka, tl=tl())
      except Exception as e:
        slog(op="keepalive_error", error=repr(e), tl=tl())
        if "Device fault" in repr(e):
          slog(op="device_fault", stage="exiting_from_keepalive")
          _clean_exit(1, "device_fault_keepalive", flush=True)
        time.sleep(1)

  # ---- health probe + park: MOVED BELOW the listener/watchdog defs (R3-04b)
  # so the watchdog thread can start BEFORE boot — a wedged boot (the
  # ~2-3min pc_boot_ingest, the park, the snapshot reset) used to hang
  # forever with busy=False and NOTHING watching it. ----

  Q = queue.Queue(maxsize=MAX_Q_DEPTH)   # R3-13: bounded, loud rejection
  # hooks the listener threads consult (legacy defaults; the batch path
  # overrides them BEFORE the threads start)
  ST.inline_status = lambda req_id: {"id": req_id, "ok": True, "result": {
      "ready": ST.ready, "ctxk": CTXK, "busy": ST.busy, "pos": ST.pos_cache,
      "rpc": ST.rpc, "mode": ST.mode, "cur": ST.cur_cache, "fed_len": len(ST.fed),
      "fed_tail": ST.fed[-16:], "conversation_id": ST.convo_id,
      "dirty": ST.dirty,
      "queue": Q.qsize() if Q else 0, "keepalive_s": KEEPALIVE_S,
      "uptime_s": round(time.time() - ST.t0, 1),
      "config_fp": ST.config_fp, "lookup_k": os.getenv("LOOKUP_K"),
      "pf_prefill": os.getenv("PF_PREFILL"),
      "ka": _ka_slog_fields(),           # TLX P10 pool counters (F2/F3)
      "cycles_since_rebuild": ST.cycles_since_rebuild,
      "model_id": _MODEL_ID,               # R3-42 (L2 fix: inline path missed)
      "cycle_cap": max(1, min(4096, CTXK - max(0, int(ST.pos_cache)))),  # R3-34
      "pc": _pc_status()}}                 # R3-28 (L2 fix)
  ST.active_conn = None
  def _legacy_cancel_hook(conn):
    """R3-21: legacy cancel is scoped to the conn that OWNS the active RPC —
    any same-uid process used to be able to abort the active generate."""
    if ST.active_conn is not None and conn is not ST.active_conn:
      slog(op="cancel_scoped_ignored")
      return
    ST.cancel = True
    ST.cancel_armed_by = ("cancel_rpc", ST.rpc)     # L7: attribution
  ST.cancel_hook = _legacy_cancel_hook
  ST.unbind_hook = None                                        # batch: conn->slot cleanup

  _conn_count = {"n": 0}
  _conn_g = threading.Lock()
  def listener_conn(conn):
    with _conn_g:
      _conn_count["n"] += 1
      n_now = _conn_count["n"]
    if n_now > MAX_CONNS:            # R3-51: bound the per-conn thread count
      try: conn.close()
      except Exception: pass
      with _conn_g:
        _conn_count["n"] -= 1
      slog(op="conn_cap_exceeded", active=n_now, cap=MAX_CONNS)
      return
    # V-24: peer-credential gate. LOCAL_PEERCRED gives the peer euid; a peer
    # outside the allowlist is dropped before a single byte is read. When the
    # credential query is unsupported the 0600 socket mode remains the boundary.
    uid = _peer_uid(conn)
    if uid is not None and uid not in ALLOWED_PEER_UIDS:
      slog(op="socket_peer_denied", peer_uid=uid)
      try: conn.close()
      except Exception: pass
      return
    if uid is None:
      if not PEERCRED_LENIENT:      # R3-21: FAIL-CLOSED (was: accept on perms)
        slog(op="socket_peer_denied", note="LOCAL_PEERCRED unavailable; fail-closed")
        try: conn.close()
        except Exception: pass
        return
      slog(op="socket_peer_unknown", note="lenient mode: 0600 perms only")
    buf = b""
    conn.settimeout(SEND_TIMEOUT_S)   # V-30: send-stall protection; recv below
    try:                               # tolerates the timeout as "no request yet"
      while True:
        try:
          chunk = conn.recv(65536)
        except socket.timeout:
          continue
        if not chunk: break
        buf += chunk
        # V-31: unbounded line buffer = jetsam of the GPU python (= reboot
        # class on this box). A no-newline stream past the cap drops the conn.
        if len(buf) > MAX_LINE_BYTES:
          slog(op="line_too_long", pending=len(buf), cap=MAX_LINE_BYTES, dropped=True)
          break
        while b"\n" in buf:
          line, buf = buf.split(b"\n", 1)
          if not line.strip(): continue
          if len(line) > MAX_LINE_BYTES:
            slog(op="line_too_long", line=len(line), cap=MAX_LINE_BYTES, dropped=True)
            return
          try: req = json.loads(line)
          except Exception: continue
          if req.get("method") == "cancel":
            ST.cancel_hook(conn)   # side-channel flag (no ack write)
          elif req.get("method") == "status" and ST.ready:
            # LOCK-FREE inline status from host caches: /health must answer
            # DURING a generate (the Q is serialized behind GPU work).
            try: send(conn, ST.inline_status(req.get("id")))
            except Exception: pass
          else:
            try:
              Q.put_nowait((conn, req))    # R3-13: Q is CAPPED, LOUDLY
            except queue.Full:
              slog(op="engine_queue_full", dropped=req.get("method"), q=Q.qsize())
              try:
                send(conn, {"id": req.get("id"), "ok": False,
                            "error": f"engine queue full ({MAX_Q_DEPTH}); retry shortly"})
              except Exception: pass
    except Exception: pass
    finally:
      with _conn_g:
        _conn_count["n"] -= 1
      if ST.unbind_hook is not None:
        try: ST.unbind_hook(conn)
        except Exception: pass
      _mark_dead(conn)   # R3-16: close + dead flag under the send lock

  def listener():
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(SOCK): os.unlink(SOCK)   # unlink-before-bind (stale socket never blocks boot)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o600)   # V-24: owner-only connect (umask-independent)
    srv.listen(16)
    while True:
      try:
        conn, _ = srv.accept()
        threading.Thread(target=listener_conn, args=(conn,), daemon=True).start()
      except Exception as e:
        log("listener:", e); time.sleep(0.5)

  def heartbeat():
    n = 0
    while True:
      time.sleep(2)
      with _LOGL:
        line = json.dumps({"ts": round(time.time(),3), "hb": n, "tl": tl(), "busy": ST.busy})
        for p in (LOGF, LOGF_PERSIST):
          try:
            with open(p, "a") as f: f.write(line + "\n")
          except Exception: pass
      n += 1
      if n % 300 == 0:   # V-38b: ~10min — rotate the forever-growing heartbeats
        _log_rotate((LOGF, LOGF_PERSIST), 20 * 1024 * 1024)

  def watchdog():
    """W2 serving-half watchdog (the gcycle/ops_nv halves are W4):
      (a) an RPC owns the GPU (ST.busy) and no beat for TLX_STEP_TIMEOUT_S ->
          the wait is wedged -> mark dirty + clean-exit(1) (the DEVICE-FAULT
          LAW: a wedged daemon heals only via supervisor reboot).
      (b) R3-02 idle-arm: NOT busy and no successful keepalive/beat for
          3x KEEPALIVE_S -> the keepalive wedge class (a SILENT zombie that
          answered /health ready+busy=False forever on a dead GPU) -> exit
          loudly as keepalive_stale. A healthily-serving daemon beats every
          cycle, so this can only fire on a truly idle wedge."""
    iv = min(5.0, max(0.05, STEP_TIMEOUT_S / 10.0))
    while True:
      time.sleep(iv)
      try:
        now = time.time()
        if ST.busy and (now - ST.step_beat) > STEP_TIMEOUT_S:
          ST.dirty = True
          slog(op="watchdog", stage="step_timeout", rpc=ST.rpc,
               overdue_s=round(now - ST.step_beat, 1), timeout_s=STEP_TIMEOUT_S)
          _clean_exit(1, "watchdog_step_timeout", flush=True)
        if not ST.busy and (now - ST.last_keepalive_ok) > 3 * KEEPALIVE_S:
          slog(op="watchdog", stage="keepalive_stale",
               stale_s=round(now - ST.last_keepalive_ok, 1),
               keepalive_s=KEEPALIVE_S, rpc=ST.rpc)
          _clean_exit(1, "keepalive_stale", flush=True)
      except Exception as e:
        slog(op="watchdog_error", error=repr(e))

  # ---- R3-04b: watchdog + heartbeat from the VERY START (boot included) ----
  threading.Thread(target=watchdog, daemon=True).start()
  threading.Thread(target=heartbeat, daemon=True).start()

  # ---- health probe + park (busy + beats: a boot wedge is WATCHED) --------
  ST.busy = True; ST.rpc = "boot"; _beat()
  try:
    keepalive()
    _beat()
    E.reset_snapshot(SNAP, CUR0, P0)
    seed_hist(ids)   # R5d: the parked 100k conversation is lookup-visible (reset_spec law)
    _sync_caches()
    if PC is not None:
      try:
        _pc_boot_ingest()   # R1: node@P0 from snap files + device draft KV (~2-3min first boot)
      except Exception as e:
        slog(op="pc_boot_error", stage="ingest", error=repr(e))
  finally:
    ST.busy = False; ST.rpc = None
  ST.ready = True
  ST.fed = list(ids)
  slog(op="boot", stage="daemon_attached", park_pos=P0)
  log(f"daemon attached; health probe ok; parked at pos {P0}; listening on {SOCK}")

  if BATCH_B >= 2 and r6h is not None:
    _batch_main(Q, listener, heartbeat, watchdog)
    return

  threading.Thread(target=listener, daemon=True).start()
  threading.Thread(target=heartbeat, daemon=True).start()
  threading.Thread(target=watchdog, daemon=True).start()
  nka = 0
  while True:
    try:
      conn, req = Q.get(timeout=KEEPALIVE_S)
    except queue.Empty:
      try:
        keepalive(); nka += 1
        if nka % 50 == 1: slog(op="keepalive", n=nka, tl=tl())
      except Exception as e:
        slog(op="keepalive_error", error=repr(e), tl=tl())
        if "Device fault" in repr(e):
          slog(op="device_fault", stage="exiting_from_keepalive")
          _clean_exit(1, "device_fault_keepalive", flush=True)   # V-37
        time.sleep(1)
      continue
    nka = 0
    handle(conn, req)
