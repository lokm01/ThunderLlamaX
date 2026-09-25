# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
# TLX W2 (ledger V-26/V-27/V-29/V-36) — engine daemon supervisor wrapper.
# launchd (or an operator) runs THIS script; it runs the engine python as a
# CHILD so exits can be counted, and it owns the breaker + GPU-lock etiquette.
#
# Fixes vs the M1-B wrapper:
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
# Dry-run/testable: TLX_ENGINE_CMD (stub command), TLX_OPS_ROOT, TLX_LOGS_DIR,
# TLX_GPU_LOCK overrides let the whole logic run without touching the GPU.
set -u

ENGINE_DIR="${TLX_ENGINE_DIR:-~/tinygrad-metal/engine0}"
OPS_ROOT="${TLX_OPS_ROOT:-$ENGINE_DIR/ops}"
LOGS_DIR="${TLX_LOGS_DIR:-$ENGINE_DIR/../logs}"
ENGINE_SOCK="${TLX_ENGINE_SOCK:-/tmp/llm-engine.sock}"
LOCK="${TLX_GPU_LOCK:-/tmp/nv_usb4.lock}"
ENVFILE="$OPS_ROOT/env.canonical"
STAYDOWN="$LOGS_DIR/llm_engine_staydown"
CRASHLOG="$LOGS_DIR/llm_engine_crashes.log"
PIDFILE="$LOGS_DIR/engine.pid"
OPSLOG="$LOGS_DIR/engine_ops.log"
BREAKER_N=${TLX_BREAKER_N:-3}
BREAKER_WINDOW_S=${TLX_BREAKER_WINDOW_S:-600}

mkdir -p "$LOGS_DIR"
now=$(date +%s)

opslog() { # tiny structured line for forensics
  print -r -- "{\"ts\":$now,\"ev\":\"$1\",\"msg\":\"$2\"}" >> "$OPSLOG" 2>/dev/null || true
}

# --- circuit breaker (V-26: persistent state; crash-class EXITS only) --------
if [ -f "$STAYDOWN" ]; then
  echo "[engine_daemon] STAYDOWN marker present ($STAYDOWN) — refusing to start." >&2
  echo "[engine_daemon] re-enable with: enginectl clear-breaker" >&2
  exit 11
fi
# keep only recent entries; count nonzero exit codes as crashes (class+ready kept for forensics)
if [ -f "$CRASHLOG" ]; then
  awk -v n=$now 'NF>=2 && $1 > n-'"$BREAKER_WINDOW_S"'' "$CRASHLOG" > "$CRASHLOG.f" 2>/dev/null
  mv "$CRASHLOG.f" "$CRASHLOG" 2>/dev/null
  ncrash=$(awk -v n=$now 'NF>=2 && $1 > n-'"$BREAKER_WINDOW_S"' && $2 != 0' "$CRASHLOG" 2>/dev/null | wc -l | tr -d ' ')
else
  ncrash=0
fi
if [ "${ncrash:-0}" -ge "$BREAKER_N" ]; then
  echo "breaker: $ncrash crash-class exits within ${BREAKER_WINDOW_S}s at $(date)" > "$STAYDOWN"
  echo "[engine_daemon] circuit breaker: $ncrash crash-class exits in ${BREAKER_WINDOW_S}s — staying down ($STAYDOWN)" >&2
  opslog breaker_trip "$ncrash exits"
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

# --- canonical env (V-27): ONE sourced file ------------------------------------
if [ ! -f "$ENVFILE" ]; then
  echo "[engine_daemon] missing $ENVFILE — refusing (env must be single-sourced)" >&2
  exit 15
fi
set -a
source "$ENVFILE"
set +a
env_digest=$(cat "$ENVFILE" | grep -v '^[[:space:]]*#' | sort | shasum -a 256 | cut -d' ' -f1)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
export DOCKER_HOST="${DOCKER_HOST:-unix://~/.colima/default/docker.sock}"
opslog wrapper_start "env_digest ${env_digest:0:16}"

# --- run the engine as a CHILD (V-26: count exits, not starts) -----------------
print -r -- "$$ $(date +%s)" > "$LOCK"
cd "$ENGINE_DIR" || { rm -f "$LOCK"; exit 1; }
if [ -n "${TLX_ENGINE_CMD:-}" ]; then
  zsh -c "$TLX_ENGINE_CMD" &
else
  "${TLX_PYTHON:-~/tg311/bin/python}" -u test_w100k.py &
fi
child=$!
print -r -- "$child" > "$PIDFILE"

fwd() { kill -TERM "$child" 2>/dev/null || true; }
trap fwd TERM INT

ready=0
while kill -0 "$child" 2>/dev/null; do
  [ -S "$ENGINE_SOCK" ] && ready=1
  sleep 2
done
wait "$child"
code=$?
now=$(date +%s)
rm -f "$PIDFILE"
# remove the lock only if WE still own it (content starts with our pid)
if [ -f "$LOCK" ] && head -1 "$LOCK" | grep -q "^$$"; then
  rm -f "$LOCK"
fi

# crash-class = nonzero exit. Exit 0 + staydown marker = operator shutdown
# (serve.py _clean_exit wrote the marker) -> NOT a crash.
if [ "$code" -ne 0 ]; then
  cls=crash
  [ "$ready" -eq 0 ] && cls=boot_fail
  echo "$now $code $cls ready=$ready" >> "$CRASHLOG"
  opslog child_exit "code $code class $cls ready $ready"
  # trip the breaker IMMEDIATELY when this recording reaches the threshold
  # (the marker exists right away; launchd's throttled restarts then refuse)
  ncr=$(awk -v n=$now 'NF>=2 && $1 > n-'"$BREAKER_WINDOW_S"' && $2 != 0' "$CRASHLOG" 2>/dev/null | wc -l | tr -d ' ')
  if [ "${ncr:-0}" -ge "$BREAKER_N" ] && [ ! -f "$STAYDOWN" ]; then
    echo "breaker: $ncr crash-class exits within ${BREAKER_WINDOW_S}s at $(date)" > "$STAYDOWN"
    opslog breaker_trip "$ncr exits (post-record)"
    echo "[engine_daemon] circuit breaker: $ncr crash-class exits — staying down ($STAYDOWN)" >&2
  fi
elif [ -f "$STAYDOWN" ]; then
  opslog child_exit "code 0 requested (staydown present)"
else
  opslog child_exit "code 0 clean"
fi
exit "$code"
