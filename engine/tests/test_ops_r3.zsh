#!/bin/zsh
# TLX R3 ops battery (GPU-free): engine_daemon.sh breaker/boot-deadline/lock
# semantics + enginectl stop-order/clear-breaker (R3-04a/05/07/26).
# Run:  zsh engine0/tests/test_ops_r3.zsh
set -u
HERE=${0:A:h}
ENG0=${HERE:h}
OPS="$ENG0/ops"
PASS=0; FAIL=0; FAILED=""

t() { # t <name> <assert-cmd...>
  local name=$1; shift
  if "$@" >/dev/null 2>&1; then
    print -r -- "PASS  $name"; PASS=$((PASS+1))
  else
    print -r -- "FAIL  $name"; FAIL=$((FAIL+1)); FAILED="$FAILED $name"
  fi
}

_sandbox() { # fresh TLX_OPS_ROOT/LOGS/GPU lock sandbox; echoes the dir
  local d=$(mktemp -d "${TMPDIR:-/tmp}/tlx_ops_r3_XXXXXX")
  mkdir -p "$d/ops" "$d/logs"
  cp "$OPS/env.canonical" "$d/ops/env.canonical" 2>/dev/null || \
    print "TLX_ADMIN_TOKEN=t\nTLX_MODEL_PATH=/nonexistent" > "$d/ops/env.canonical"
  print -r -- "$d"
}

# ==============================================================================
# R3-26: exact-pid lock ownership (the prefix-match bug deleted pid 12345's
# live lock when the wrapper was 1234)
# ==============================================================================
test_lock_owner_exact() {
  local d=$(_sandbox); local lock="$d/gpu.lock"
  print -r -- "12345 1700000000" > "$lock"
  ( TLX_SOURCE_ONLY=1 source "$OPS/engine_daemon.sh"
    # the OLD code (grep -q "^1234") matched the prefix; the helper must not
    if _lock_owner_matches "$lock" 1234; then exit 1; fi
    _lock_owner_matches "$lock" 12345 || exit 2
    print -r -- "pid=12345 x" > "$lock"
    _lock_owner_matches "$lock" 12345 || exit 3
    if _lock_owner_matches "$lock" 99999; then exit 4; fi
    if _lock_owner_matches "$d/nope.lock" 12345; then exit 5; fi
    exit 0 )
  local rc=$?
  rm -rf "$d"
  return $rc
}
t test_r3_26_lock_owner_exact test_lock_owner_exact

# ==============================================================================
# R3-04a: boot deadline — child alive but never ready -> boot_fail recorded +
# staydown + exit 16, and the child is LEFT RUNNING (no SIGTERM of a GPU py)
# ==============================================================================
test_boot_deadline() {
  local d=$(_sandbox)
  local sock="$d/engine.sock"
  ( TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" TLX_ENGINE_SOCK="$sock" \
    TLX_GPU_LOCK="$d/gpu.lock" TLX_BOOT_DEADLINE_S=2 TLX_BREAKER_N=3 \
    TLX_ENGINE_CMD="sleep 30" zsh "$OPS/engine_daemon.sh" ) 
  local rc=$?
  local ok=0
  [ "$rc" -eq 16 ] || ok=1
  [ -f "$d/logs/llm_engine_staydown" ] || ok=2
  grep -q "boot_fail" "$d/logs/llm_engine_crashes.log" 2>/dev/null || ok=3
  grep -q "child_left_running" "$d/logs/llm_engine_crashes.log" 2>/dev/null || ok=4
  # the stub child must still be alive (operator's job now)
  pgrep -f "sleep 30" >/dev/null || ok=5
  pkill -f "tlx_ops_r3_sleeper_marker_none" 2>/dev/null || true
  # clean up the stub child
  local kid=$(pgrep -f "^sleep 30$" | head -1)
  [ -n "$kid" ] && kill -9 "$kid" 2>/dev/null
  rm -rf "$d"
  return $ok
}
t test_r3_04a_boot_deadline test_boot_deadline

# ==============================================================================
# R3-05: CONSECUTIVE crash breaker trips even when the exits are spaced
# beyond the 600s window (the ready-then-wedge boot cycle is >=530s); a
# crash after a >=READY_MIN ready run resets the streak
# ==============================================================================
# stub engine: binds the real unix socket (ready), stays ready S seconds,
# exits with CODE — the ready-then-crash boot-cycle shape
_cat_stub() {
  cat > "$1" <<'PYEOF'
import socket, sys, time
sock, secs, code = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
try: os.unlink(sock)
except Exception: pass
import os
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(sock); s.listen(1)
time.sleep(secs)
sys.exit(code)
PYEOF
}

test_consecutive_breaker() {
  local d=$(_sandbox)
  local sock="$d/engine.sock"
  _cat_stub "$d/stub.py"
  local env=(TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" TLX_ENGINE_SOCK="$sock"
             TLX_GPU_LOCK="$d/gpu.lock" TLX_BREAKER_N=3
             TLX_BREAKER_WINDOW_S=1 TLX_BREAKER_READY_MIN_S=300)
  # stub: binds the socket (ready, observed by the 2s poll), stays ready 3s,
  # then crash-exits 7. ruptime ~1-3s < 300 -> consecutive crash; the exits
  # are ~4s apart so the 1s WINDOW path can NEVER see 3 of them (that was
  # the old, never-tripping arithmetic).
  local cmd="/usr/bin/python3 $d/stub.py $sock 3 7"
  local rc1=0 rc2=0 rc3=0 rc4=0
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; rc1=$?
  rm -f "$sock"
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; rc2=$?
  rm -f "$sock"
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; rc3=$?
  local ok=0
  [ "$rc1" -eq 7 ] && [ "$rc2" -eq 7 ] || ok=1       # first two crash through
  [ "$rc3" -eq 7 ] || ok=2                           # third RUNS (2 prior only) and
  [ -f "$d/logs/llm_engine_staydown" ] || ok=3       # trips the breaker AT RECORD
  grep -q "ruptime=" "$d/logs/llm_engine_crashes.log" 2>/dev/null || ok=4
  grep -q " crash " "$d/logs/llm_engine_crashes.log" 2>/dev/null || ok=5
  # the NEXT start refuses (consecutive counter, window be damned)
  env $env TLX_ENGINE_CMD="exit 0" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; rc4=$?
  # refused either by the staydown marker (11) or the breaker counter (12)
  [ "$rc4" -eq 11 ] || [ "$rc4" -eq 12 ] || ok=6
  rm -f "$sock"; rm -rf "$d"
  return $ok
}
t test_r3_05_consecutive_breaker test_consecutive_breaker

test_long_ready_crash_resets_streak() {
  # a crash AFTER a >=READY_MIN ready run is a steady-state transient: the
  # consecutive streak must reset (no trip on the 3rd such exit)
  local d=$(_sandbox)
  local sock="$d/engine.sock"
  _cat_stub "$d/stub.py"
  local env=(TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" TLX_ENGINE_SOCK="$sock"
             TLX_GPU_LOCK="$d/gpu.lock" TLX_BREAKER_N=3
             TLX_BREAKER_WINDOW_S=1 TLX_BREAKER_READY_MIN_S=1)
  # stub stays ready 3.2s (t_ready at the 2s poll -> ruptime ~1.2 >= min=1)
  local cmd="/usr/bin/python3 $d/stub.py $sock 3.2 7"
  local r1=0 r2=0 r3=0
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; r1=$?
  rm -f "$sock"
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; r2=$?
  rm -f "$sock"
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1; r3=$?
  local ok=0
  [ "$r1" -eq 7 ] && [ "$r2" -eq 7 ] && [ "$r3" -eq 7 ] || ok=1
  [ ! -f "$d/logs/llm_engine_staydown" ] || ok=2
  rm -rf "$d"
  return $ok
}
t test_r3_05_long_ready_crash_resets test_long_ready_crash_resets_streak

# ==============================================================================
# R3-07: operator SIGTERM class — the trap-forwarded exit is recorded
# operator-class and never counts toward the crash breaker
# ==============================================================================
test_operator_term_not_crash() {
  local d=$(_sandbox)
  local sock="$d/engine.sock"
  _cat_stub "$d/stub.py"
  local env=(TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" TLX_ENGINE_SOCK="$sock"
             TLX_GPU_LOCK="$d/gpu.lock" TLX_BREAKER_N=2 TLX_BREAKER_READY_MIN_S=300)
  local cmd="/usr/bin/python3 $d/stub.py $sock 60 0"
  env $env TLX_ENGINE_CMD="$cmd" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1 &
  local wpid=$!
  sleep 2                      # ready by now
  kill -TERM "$wpid" 2>/dev/null
  wait "$wpid" 2>/dev/null
  local ok=0
  grep -q "operator" "$d/logs/llm_engine_crashes.log" 2>/dev/null || ok=1
  # no crash entries -> the breaker did not trip
  if grep -q " crash " "$d/logs/llm_engine_crashes.log" 2>/dev/null; then ok=2; fi
  [ ! -f "$d/logs/llm_engine_staydown" ] || ok=3
  # a next start is NOT refused
  env $env TLX_ENGINE_CMD="exit 0" zsh "$OPS/engine_daemon.sh" >/dev/null 2>&1 || ok=4
  rm -f "$sock"; rm -rf "$d"
  return $ok
}
t test_r3_07_operator_term_class test_operator_term_not_crash

# ==============================================================================
# R3-07: enginectl stop ORDER — the launchd branch must attempt the socket
# shutdown BEFORE any unload (source-order assertion; the functional stop
# sequence is a live-window item) + clear-breaker takes the flock
# ==============================================================================
test_enginectl_stop_order() {
  local ec=$(cat "$OPS/enginectl")
  local ok=0
  # stop_engine: socket_shutdown call appears BEFORE the launchctl unload
  local stop_body=$(sed -n '/^stop_engine()/,/^}/p' <<<"$ec")
  local sh_line=$(grep -n "socket_shutdown" <<<"$stop_body" | head -1 | cut -d: -f1)
  local ul_line=$(grep -n "_svc_stop_launchd" <<<"$stop_body" | head -1 | cut -d: -f1)
  [ -n "$sh_line" ] && [ -n "$ul_line" ] && [ "$sh_line" -lt "$ul_line" ] || ok=1
  # and the launchd check no longer RETURNS before the socket block
  grep -q "launchd-managed: unloading" <<<"$stop_body" || ok=2
  # clear-breaker runs under the flock
  grep -q "flock" "$OPS/enginectl" || ok=3
  return $ok
}
t test_r3_07_enginectl_stop_order test_enginectl_stop_order

print "================================================================"
print "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || { print "FAILED:$FAILED"; exit 1 }
exit 0
