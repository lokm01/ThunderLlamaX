#!/bin/zsh
# TLX W2 (ledger V-26/V-27/V-29/V-36) + R3 (R3-04a/R3-05/R3-26) — engine daemon
# supervisor wrapper. launchd (or an operator) runs THIS script; it runs the
# engine python as a CHILD so exits can be counted, and it owns the breaker +
# GPU-lock etiquette.
#
# W2 fixes vs the M1-B wrapper:
#   V-26  breaker state PERSISTS at <logs>/ (was /tmp — wiped by the very
#         fault-reboots the breaker exists for); counts CHILD EXITS
#         (crash-class), never starts; stays down via the marker file.
#   V-27  env is SINGLE-SOURCED from ops/env.canonical (sha256 digest logged)
#         — a launchd relaunch can no longer boot the stale M1-era env subset
#         (the LOOKUP_K=0-class silent slow-path degradation).
#   V-29  GPU-lock liveness is PID-based (kill -0 on the lock's pid), never a
#         name-filtered pgrep. MULTI-AGENT RULE: this box runs other GPU
#         agents (R6-class benchmarks); a lock owned by ANY live pid is never
#         removed. A lock without a parseable pid is never auto-removed
#         either — inspect manually (the writer convention is "<pid> ...").
#   V-36  an operator RPC shutdown exits 0 AND leaves the staydown marker —
#         a clean exit with the marker present is NOT a crash; KeepAlive
#         throttles through wrapper refusals until `enginectl clear-breaker`.
#
# R3 fixes (round-3 ledger):
#   R3-04a wrapper READY-DEADLINE (~900s, TLX_BOOT_DEADLINE_S): a child alive
#         but not ready (no engine socket) past the deadline records boot_fail
#         + staydown and DOES NOT SIGTERM the possibly-wedged GPU python (the
#         GPU-EXIT/REBOOT law — the operator is paged to inspect/kill -9).
#   R3-05 consecutive-CRASH breaker (the 600s window could never hold 3
#         ready-then-wedge boot cycles of >=530s each): the consecutive
#         counter counts trailing crash-class exits whose ready-uptime was
#         < TLX_BREAKER_READY_MIN_S (a crash after a >=N-minute healthy run
#         resets the run); the WINDOW now guards BOOT_FAIL loops only; every
#         exit line records ready_uptime for attribution; an operator SIGTERM
#         (the wrapper's trap forwarding) is recorded operator-class and
#         never counted as a crash.
#   R3-26 GPU-lock removal is an EXACT first-field pid match (the old
#         `grep -q "^$$"` prefix match deleted the live lock of e.g. pid
#         12345 when the wrapper was 1234 — defeating the V-29 multi-agent
#         rule).
#
# Dry-run/testable: TLX_ENGINE_CMD (stub command), TLX_OPS_ROOT, TLX_LOGS_DIR,
# TLX_GPU_LOCK, TLX_SOURCE_ONLY=1 (define helpers, skip the main flow) let the
# whole logic run without touching the GPU.
set -u

ENGINE_DIR="${TLX_ENGINE_DIR:-~/tinygrad-metal/engine0}"
OPS_ROOT="${TLX_OPS_ROOT:-$ENGINE_DIR/ops}"
LOGS_DIR="${TLX_LOGS_DIR:-$ENGINE_DIR/../logs}"
ENGINE_SOCK="${TLX_ENGINE_SOCK:-/tmp/llm-engine.sock}"
LOCK="${TLX_GPU_LOCK:-/tmp/nv_usb4.lock}"
ENVFILE="$OPS_ROOT/env.canonical"
# TLX P8 multi-model: registry + durable swap-intent state. The wrapper is the
# SOLE writer of current_model (next_model promotion at boot); enginectl switch
# is the sole writer of next_model/swap_in_progress (atomic + fsync + dir-fsync
# — the L5 law: file-fsync alone does not survive the GPU-EXIT hard reset).
REGISTRY="$OPS_ROOT/model_registry.json"
STATEDIR="$OPS_ROOT/state"
MODEL_ENV_D="$OPS_ROOT/env.canonical.d"
ENV_COMMON="$OPS_ROOT/env.common"
STAYDOWN="$LOGS_DIR/llm_engine_staydown"
CRASHLOG="$LOGS_DIR/llm_engine_crashes.log"
CRASHLOCK="$CRASHLOG.lock"
PIDFILE="$LOGS_DIR/engine.pid"
OPSLOG="$LOGS_DIR/engine_ops.log"
BREAKER_N=${TLX_BREAKER_N:-3}
BREAKER_WINDOW_S=${TLX_BREAKER_WINDOW_S:-600}
# R3-05: a crash counts toward the CONSECUTIVE breaker only if the run stayed
# ready less than this (steady-state transients reset the streak)
BREAKER_READY_MIN_S=${TLX_BREAKER_READY_MIN_S:-300}
# R3-04a: child alive but not ready past this -> boot_fail + staydown, child LEFT RUNNING
BOOT_DEADLINE_S=${TLX_BOOT_DEADLINE_S:-900}

mkdir -p "$LOGS_DIR" 2>/dev/null || true
now=$(date +%s)

opslog() { # tiny structured line for forensics
  print -r -- "{\"ts\":$now,\"ev\":\"$1\",\"msg\":\"$2\"}" >> "$OPSLOG" 2>/dev/null || true
}

# ---- P8: durable state writes (atomic + fsync + dir-fsync; the L5 law) -------
_state_write() {  # _state_write <name> <content>
  /usr/bin/python3 - "$STATEDIR" "$1" "$2" <<'PYEOF'
import os, sys, tempfile
sd, name, val = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(sd, exist_ok=True)
fd, tmp = tempfile.mkstemp(dir=sd, prefix=f".{name}.")
with os.fdopen(fd, "w") as f:
    f.write(val if val.endswith("\n") else val + "\n")
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, os.path.join(sd, name))
dfd = os.open(sd, os.O_RDONLY)
try: os.fsync(dfd)
finally: os.close(dfd)
PYEOF
}
_state_read() { head -1 "$STATEDIR/$1" 2>/dev/null | tr -d '[:space:]'; }
_state_clear() { rm -f "$STATEDIR/$1" 2>/dev/null || true; }

# ---- P8: registry validation (prints "ok <engine_host> <env_file> <model_path>"
#         or "err <reason>"; exit code 0 iff the model is bootable) ------------
_registry_check() {  # _registry_check <model_id> <strict_env:0|1>
  /usr/bin/python3 - "$REGISTRY" "$MODEL_ENV_D" "$1" "${TLX_ENGINE_DIR:-~/tinygrad-metal/engine0}" "$2" <<'PYEOF'
import json, os, sys
reg, envd, mid, engdir, strict_env = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
try:
    r = json.load(open(reg))
except Exception as e:
    print(f"err registry unreadable: {e}"); sys.exit(1)
m = r.get("models", {}).get(mid)
if m is None:
    print(f"err unknown model id '{mid}' (known: {', '.join(sorted(r.get('models', {})))})"); sys.exit(1)
# ENV FILE: HARD failure for an ARMED swap intent (D3: an intent whose model
# cannot boot must quarantine, keep current, refuse); SOFT for the current
# model (a missing per-model env falls back to the monolithic env.canonical —
# the pre-P8 rollback path).
envf = os.path.join(envd, os.path.basename(m["env_file"]))
if strict_env == "1" and not os.path.isfile(envf):
    print(f"err env file missing: {envf}"); sys.exit(1)
mp = m.get("model_path", "")
if mp and not os.path.isfile(mp):
    print(f"err model file missing: {mp}"); sys.exit(1)
host = m.get("engine_host", "test_w100k.py")
# host resolution: the ops-tree root (sandbox batteries put stub hosts there),
# then the REAL engine dir (production: registry hosts are engine0-relative)
if not (os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(envd)), host)) or os.path.isfile(host)
        or os.path.isfile(os.path.join(engdir, host))):
    print(f"err engine host missing: {host}"); sys.exit(1)
print(f"ok {host} {m['env_file']} {mp}"); sys.exit(0)
PYEOF
}

# ---- P8 helpers only below; the model-selection FLOW runs after the
# TLX_SOURCE_ONLY gate (test batteries source the helpers without the flow).

# ---- R3-05/R3-07: crashlog mutations under an flock shared with enginectl
# clear-breaker (macOS ships no flock(1): python3 fcntl). ---------------------
_crashlog() {  # _crashlog <append|filter|consecutive> [arg]
  local mode=$1; shift
  /usr/bin/python3 - "$CRASHLOG" "$CRASHLOCK" "$mode" "$@" <<'PYEOF'
import fcntl, os, re, subprocess, sys
path, lockpath, mode = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]
open(lockpath, "a").close()
with open(lockpath, "r+") as lf:
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    if mode == "append":
        with open(path, "a") as f:
            f.write(args[0] + "\n")
            f.flush(); os.fsync(f.fileno())   # L5: survive the GPU-EXIT hard reset
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    elif mode == "filter":   # bound growth WITHOUT erasing consecutive history
        window = int(args[0])
        try:
            lines = [l for l in open(path).read().splitlines() if l.split()]
        except Exception:
            lines = []
        now = __import__("time").time()
        fresh = [l for l in lines if l.split()[0].isdigit()
                 and int(l.split()[0]) > now - window]
        # R3-05: the consecutive breaker needs the trailing run — always keep
        # the last 64 entries even when older than the window (the old
        # window-only filter erased the history every spaced-out crash).
        keep = fresh if len(fresh) >= 64 else lines[-64:]
        tmp = path + ".f"
        with open(tmp, "w") as f:
            f.write("".join(l + "\n" for l in keep))
            f.flush(); os.fsync(f.fileno())   # L5
        os.replace(tmp, path)
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    elif mode == "consecutive":  # trailing run of crash-class exits with ruptime < min
        minru = int(args[0])
        run = 0
        try:
            for line in open(path).read().splitlines():
                f = line.split()
                if len(f) >= 3 and f[0].isdigit():
                    cls = f[2]
                    ru = 0
                    for tok in f[3:]:
                        if tok.startswith("ruptime="):
                            ru = int(float(tok.split("=", 1)[1] or 0))
                    if cls == "crash" and ru < minru:
                        run += 1
                    else:
                        run = 0
        except Exception:
            pass
        print(run)
PYEOF
}

# ---- R3-26: exact first-field pid match on the lock (testable helper) -------
_lock_owner_matches() {  # _lock_owner_matches <lockfile> <pid> -> 0 iff owned
  local lfile=$1 want=$2 owner
  [ -f "$lfile" ] || return 1
  owner=$(head -1 "$lfile" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/^pid=//' | awk '{print $1}')
  [ "$owner" = "$want" ]
}

# TLX_SOURCE_ONLY=1: tests source the helpers above without running the flow.
if [ "${TLX_SOURCE_ONLY:-0}" = "1" ]; then
  return 0 2>/dev/null || true
fi

# ---- P8: model selection (next_model > current_model > registry default) -----
# Runs BEFORE the staydown check: a boot with a VALID armed next_model is a
# swap boot — promotion clears the swap-authored staydown (the operator asked
# for this stop). An INVALID armed next_model is a swap failure: staydown
# (swap_failed) + refuse; current_model is left untouched (rollback = boot the
# previous model on the next wrapper start after clear-breaker).
MODEL_ID=""
P8_ENGINE_HOST="test_w100k.py"
P8_MODEL_ENV=""
P8_LEGACY=0
if [ ! -f "$REGISTRY" ]; then
  # PRE-P8 INSTALL (no registry): full legacy behavior — monolithic env, the
  # default engine host, no model-id validation (the R3 battery's sandbox and
  # any pre-P8 rollback boot land here). TLX_MODEL_ID stays unset unless the
  # env carries it.
  P8_LEGACY=1
  opslog wrapper_start "p8=registry_absent legacy boot"
fi
_next=$(_state_read next_model)
_cur=$(_state_read current_model)
if [ -z "$P8_LEGACY" ] || [ "$P8_LEGACY" = "0" ]; then
  if [ -z "$_cur" ]; then
    _cur=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("default_model",""))' "$REGISTRY" 2>/dev/null || echo "")
  fi
fi
if [ -z "$P8_LEGACY" ] || [ "$P8_LEGACY" = "0" ]; then :; else _next=""; fi
if [ -n "$_next" ]; then
  _chk=$(_registry_check "$_next" 1)
  if [ "${_chk%% *}" = "ok" ]; then
    _was_swap=0
    [ -f "$STATEDIR/swap_in_progress" ] && _was_swap=1
    _state_write current_model "$_next"
    _state_clear next_model
    _state_clear swap_in_progress
    MODEL_ID="$_next"
    P8_MODEL_ENV="$MODEL_ENV_D/$(print -r -- "$_chk" | awk '{print $3}' | awk -F/ '{print $NF}')"
    P8_ENGINE_HOST="$(print -r -- "$_chk" | awk '{print $2}')"
    if [ "$_was_swap" = "1" ] && [ -f "$STAYDOWN" ]; then
      rm -f "$STAYDOWN"   # the swap-authored stop marker — the swap supersedes it
    fi
    opslog swap_promoted "$_next"
    echo "[engine_daemon] P8 swap: promoted next_model='$_next' — booting it now." >&2
  else
    echo "swap_failed: armed next_model '$_next' invalid ($_chk)" > "$STAYDOWN"
    opslog swap_failed "next_model '$_next': ${_chk#err }"
    echo "[engine_daemon] P8 swap FAILED validation: ${_chk#err }" >&2
    echo "[engine_daemon] staydown set; current_model unchanged ('$_cur'); fix or rm $STATEDIR/next_model, then clear-breaker." >&2
    exit 17
  fi
fi
if [ -z "$MODEL_ID" ] && { [ -z "$P8_LEGACY" ] || [ "$P8_LEGACY" = "0" ]; }; then
  _chk=$(_registry_check "$_cur" 0)
  if [ "${_chk%% *}" != "ok" ]; then
    echo "swap_failed: current_model '$_cur' invalid ($_chk)" > "$STAYDOWN"
    opslog model_invalid "'$_cur': ${_chk#err }"
    echo "[engine_daemon] P8: current model '$_cur' failed validation: ${_chk#err } — staying down." >&2
    exit 17
  fi
  MODEL_ID="$_cur"
  P8_MODEL_ENV="$MODEL_ENV_D/$(print -r -- "$_chk" | awk '{print $3}' | awk -F/ '{print $NF}')"
  P8_ENGINE_HOST="$(print -r -- "$_chk" | awk '{print $2}')"
fi

# --- circuit breaker: staydown marker + R3-05 arithmetic ----------------------
if [ -f "$STAYDOWN" ]; then
  echo "[engine_daemon] STAYDOWN marker present ($STAYDOWN) — refusing to start." >&2
  echo "[engine_daemon] re-enable with: enginectl clear-breaker" >&2
  exit 11
fi
_crashlog filter "$BREAKER_WINDOW_S" 2>/dev/null || true
consec=$(_crashlog consecutive "$BREAKER_READY_MIN_S" 2>/dev/null || echo 0)
if [ "${consec:-0}" -ge "$BREAKER_N" ]; then
  echo "breaker: $consec consecutive crash-class exits (ruptime < ${BREAKER_READY_MIN_S}s) at $(date)" > "$STAYDOWN"
  echo "[engine_daemon] circuit breaker: $consec consecutive crashes — staying down ($STAYDOWN)" >&2
  opslog breaker_trip "$consec consecutive crashes"
  exit 12
fi

# --- GPU lock (V-29: pid liveness, multi-agent-safe) --------------------------
# NOTE: the ENGINE-LEVEL lock is an flock at $TMPDIR/nv_usb4.lock (system.py
# flock_acquire) — auto-released when its holder dies. THIS marker lock is the
# wrapper-visible ownership convention for multi-agent etiquette.
if [ -f "$LOCK" ]; then
  # first whitespace-separated token of line 1, optional 'pid=' prefix;
  # must be a pure positive integer (a misparsed live-owner lock MUST NOT
  # fall through to the stale path — that was the V-29 bug class)
  lockpid=$(head -1 "$LOCK" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/^pid=//' | awk '{print $1}')
  case "$lockpid" in
    ''|*[!0-9]*) lockpid="" ;;
    0) lockpid="" ;;
  esac
  if [ -n "$lockpid" ]; then
    if kill -0 "$lockpid" 2>/dev/null; then
      echo "[engine_daemon] GPU lock $LOCK held by LIVE pid $lockpid — refusing to start." >&2
      ps -p "$lockpid" -o pid,uid,etime,command 2>/dev/null >&2 || true
      opslog lock_refused "live owner pid $lockpid"
      exit 13
    fi
    echo "[engine_daemon] removing STALE GPU lock $LOCK (pid $lockpid dead)" >&2
    opslog lock_stale_removed "pid $lockpid"
    rm -f "$LOCK"
  else
    echo "[engine_daemon] GPU lock $LOCK has no parseable pid — NOT removing it." >&2
    echo "[engine_daemon] verify no other GPU agent is live, then: rm -f $LOCK" >&2
    opslog lock_refused "unparseable lock content"
    exit 14
  fi
fi

# --- canonical env (V-27 + P8): shared env.common + the per-model env file ---
# P8: the effective env = env.common ∪ env.canonical.d/<model_id>.env (variable
# set identical to the pre-P8 monolithic env.canonical for the dense model).
# Rollback: delete env.common or env.canonical.d/ and the wrapper falls back to
# sourcing the monolithic env.canonical exactly as before.
if [ -f "$ENV_COMMON" ] && [ -f "$P8_MODEL_ENV" ]; then
  set -a
  source "$ENV_COMMON"
  source "$P8_MODEL_ENV"
  set +a
  [ -n "$MODEL_ID" ] && export TLX_MODEL_ID="$MODEL_ID"
  env_digest=$( (cat "$ENV_COMMON"; cat "$P8_MODEL_ENV") | grep -v '^[[:space:]]*#' | sort | shasum -a 256 | cut -d' ' -f1)
  opslog wrapper_start "model=$MODEL_ID env_digest ${env_digest:0:16} host=$P8_ENGINE_HOST"
else
  if [ ! -f "$ENVFILE" ]; then
    echo "[engine_daemon] missing $ENV_COMMON/$P8_MODEL_ENV AND $ENVFILE — refusing (env must be single-sourced)" >&2
    exit 15
  fi
  echo "[engine_daemon] P8 env split incomplete — FALLING BACK to monolithic $ENVFILE (pre-P8 behavior)" >&2
  set -a
  source "$ENVFILE"
  set +a
  [ -n "$MODEL_ID" ] && export TLX_MODEL_ID="$MODEL_ID"
  env_digest=$(cat "$ENVFILE" | grep -v '^[[:space:]]*#' | sort | shasum -a 256 | cut -d' ' -f1)
  opslog wrapper_start "model=$MODEL_ID env_digest ${env_digest:0:16} fallback=monolithic"
fi
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
export DOCKER_HOST="${DOCKER_HOST:-unix://~/.colima/default/docker.sock}"

# --- run the engine as a CHILD (V-26: count exits, not starts) -----------------
print -r -- "$$ $(date +%s)" > "$LOCK"
cd "$ENGINE_DIR" || { rm -f "$LOCK"; exit 1; }
if [ -n "${TLX_ENGINE_CMD:-}" ]; then
  zsh -c "$TLX_ENGINE_CMD" &
else
  "${TLX_PYTHON:-~/tg311/bin/python}" -u "$P8_ENGINE_HOST" &
fi
child=$!
print -r -- "$child" > "$PIDFILE"

termed=0
fwd() { kill -TERM "$child" 2>/dev/null || true; }
trap 'fwd; termed=1' TERM INT   # R3-05: SIGTERM-after-attempted-graceful is operator-class

# R3-04a: ready deadline — a wedged BOOT must not hang unnoticed forever.
# NOT a SIGTERM: killing a live GPU python is the GPU-EXIT/REBOOT law; the
# child is left RUNNING and the operator is paged (staydown + loud log).
boot_start=$(date +%s)
ready=0; t_ready=0
while kill -0 "$child" 2>/dev/null; do
  if [ "$ready" -eq 0 ] && [ -S "$ENGINE_SOCK" ]; then
    ready=1; t_ready=$(date +%s)
  fi
  if [ "$ready" -eq 0 ] && [ $(( $(date +%s) - boot_start )) -gt "$BOOT_DEADLINE_S" ]; then
    now=$(date +%s)
    _crashlog append "$now 0 boot_fail ready=0 ruptime=0 boot_deadline=${BOOT_DEADLINE_S}s child_left_running=$child"
    echo "breaker: boot deadline ${BOOT_DEADLINE_S}s exceeded at $(date) (child $child LEFT RUNNING — operator action required)" > "$STAYDOWN"
    opslog boot_deadline "not ready after ${BOOT_DEADLINE_S}s; child left RUNNING pid $child"
    echo "[engine_daemon] BOOT DEADLINE: child alive but not ready after ${BOOT_DEADLINE_S}s —" >&2
    echo "[engine_daemon] boot_fail recorded, staydown set, GPU python LEFT RUNNING (do not SIGTERM a wedged GPU process; inspect, then kill -9 + cold-cycle)." >&2
    exit 16
  fi
  sleep 2
done
wait "$child"
code=$?
now=$(date +%s)
rm -f "$PIDFILE"
# R3-26: remove the lock only on an EXACT first-field pid match (the old
# prefix grep deleted e.g. pid 12345's live lock when the wrapper was 1234)
if _lock_owner_matches "$LOCK" "$$"; then
  rm -f "$LOCK"
fi

# crash-class = nonzero exit that is not an operator SIGTERM. Exit 0 + staydown
# marker = operator shutdown (serve.py _clean_exit wrote the marker) -> NOT a
# crash. R3-05: every line carries cls + ready_uptime for attribution.
ruptime=0
[ "$t_ready" -gt 0 ] && ruptime=$(( now - t_ready ))
if [ "$code" -ne 0 ]; then
  cls=crash
  [ "$ready" -eq 0 ] && cls=boot_fail
  [ "$termed" -eq 1 ] && cls=operator
  _crashlog append "$now $code $cls ready=$ready ruptime=$ruptime"
  opslog child_exit "code $code class $cls ready $ready ruptime $ruptime"
  # trip the breaker IMMEDIATELY when this recording reaches a threshold
  # (R3-05: consecutive crash-class for crash loops; the window guards
  # boot_fail loops only — the marker exists right away; launchd's throttled
  # restarts then refuse)
  consec=$(_crashlog consecutive "$BREAKER_READY_MIN_S" 2>/dev/null || echo 0)
  nboot=$(awk -v n=$now 'NF>=3 && $1 > n-'"$BREAKER_WINDOW_S"' && $3 == "boot_fail"' "$CRASHLOG" 2>/dev/null | wc -l | tr -d ' ')
  if { [ "${consec:-0}" -ge "$BREAKER_N" ] || [ "${nboot:-0}" -ge "$BREAKER_N" ]; } && [ ! -f "$STAYDOWN" ]; then
    echo "breaker: ${consec} consecutive crashes / ${nboot} boot_fails at $(date)" > "$STAYDOWN"
    opslog breaker_trip "consec $consec boot_fail $nboot (post-record)"
    echo "[engine_daemon] circuit breaker: consec=$consec boot_fail=$nboot — staying down ($STAYDOWN)" >&2
  fi
elif [ -f "$STAYDOWN" ]; then
  opslog child_exit "code 0 requested (staydown present)"
else
  opslog child_exit "code 0 clean ruptime $ruptime"
fi
exit "$code"
