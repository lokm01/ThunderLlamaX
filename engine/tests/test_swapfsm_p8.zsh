#!/bin/zsh
# TLX P8 swap-FSM battery (GPU-free) — activates the R3 W-E.4 scaffold's D1-D4
# against the REAL wrapper/enginectl logic (sandboxed TLX_OPS_ROOT + stub
# TLX_ENGINE_CMD engines; never touches the GPU or the live services).
#   D1 current_model writes are atomic (tmp+rename+fsync+dir-fsync) — no torn
#      write can silently boot the DEFAULT while the API "verifies" a swap.
#   D2 the swap path never deletes a BREAKER-authored staydown (only the
#      swap-in-progress boot clears its own marker).
#   D3 intents validate against model_registry.json BEFORE touching
#      current_model; invalid intent -> keep current + staydown + refuse.
#   D4 the wrapper records the booted MODEL_ID (opslog) and exports
#      TLX_MODEL_ID to the child env (the engine's model_id status).
# Plus: the promotion order (next_model > current > default), the rollback
# fallback (env split missing -> monolithic env.canonical), and the
# enginectl switch validation refusals.
# Run:  zsh engine0/tests/test_swapfsm_p8.zsh
set -u
HERE=${0:A:h}
ENG0=${HERE:h}
OPS="$ENG0/ops"
PASS=0; FAIL=0; FAILED=""

t() { local name=$1; shift
  if "$@" >/dev/null 2>&1; then print -r -- "PASS  $name"; PASS=$((PASS+1))
  else print -r -- "FAIL  $name"; FAIL=$((FAIL+1)); FAILED="$FAILED $name"; fi }

# ---- sandbox: a full ops tree with two stub models ---------------------------
# modelA = the default (env split present); modelB = a second model with its
# own env file + host; modelBAD = registry entry whose env file is MISSING.
_sandbox() {
  local d=$(mktemp -d "${TMPDIR:-/tmp}/tlx_p8_fsm_XXXXXX")
  mkdir -p "$d/ops/state" "$d/logs" "$d/ops/env.canonical.d"
  cp "$OPS/model_registry.json" "$d/ops/" 2>/dev/null || return 1
  # bare clone: env.common is generated from the published example — use the
  # example when the operator's real file is absent (rig behavior intact)
  if [[ -f "$OPS/env.common" ]]; then
    cp "$OPS/env.common" "$d/ops/"
  else
    cp "$OPS/env.common.example" "$d/ops/env.common"
  fi
  # dense env from the real split
  cp "$OPS/env.canonical.d/qwen3.8-27b-egpu.env" "$d/ops/env.canonical.d/" 2>/dev/null || return 1
# a SECOND stub model so the swap has a target: registry-edit it in
# (bare clone: the dense model's real GGUF does not ship — point the sandbox
# copy at the same stub so the wrapper's model-file validation is hermetic)
/usr/bin/python3 - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]
r = json.load(open(os.path.join(d, "ops/model_registry.json")))
r["models"]["stub-b"] = {
    "display_name": "stub model B", "engine_host": "stub_host_b.sh",
    "model_path": os.path.join(d, "stub.gguf"),
    "env_file": "env.canonical.d/stub-b.env",
    "ctxk": 4096, "max_output_tokens": 512,
    "pcache_root": "/tmp/none", "pcache_quota_gb": 1}
dense_env = os.path.join(d, "ops/env.canonical.d/qwen3.8-27b-egpu.env")
if os.path.exists(dense_env):
    lines = open(dense_env).read().splitlines()
    out, seen = [], False
    for ln in lines:
        if ln.startswith("TLX_MODEL_PATH="):
            out.append("TLX_MODEL_PATH=%s/stub.gguf" % d); seen = True
        else:
            out.append(ln)
    if not seen:
        out.append("TLX_MODEL_PATH=%s/stub.gguf" % d)
    open(dense_env, "w").write("\n".join(out) + "\n")
if "qwen3.8-27b-egpu" in r["models"]:
    r["models"]["qwen3.8-27b-egpu"]["model_path"] = os.path.join(d, "stub.gguf")
json.dump(r, open(os.path.join(d, "ops/model_registry.json"), "w"), indent=1)
open(os.path.join(d, "stub.gguf"), "wb").write(b"GGUF....")
open(os.path.join(d, "ops/env.canonical.d/stub-b.env"), "w").write(
    "SKV=1\nTLX_MODEL_PATH=%s/stub.gguf\n" % d)
open(os.path.join(d, "stub_host_b.sh"), "w").write("#!/bin/sh\necho B-BOOTED $TLX_MODEL_ID\n")
os.chmod(os.path.join(d, "stub_host_b.sh"), 0o755)
PY
  print -r -- "$d"
}

_wrap() { # _wrap <sandbox> <extra-env...> --  : run the wrapper to completion
  local d=$1; shift
  ( cd "$d" && env TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" \
      TLX_GPU_LOCK="$d/gpu.lock" TLX_ENGINE_SOCK="$d/eng.sock" \
      TLX_ENGINE_DIR="$ENG0" \
      TLX_ENGINE_CMD='echo CHILD-RAN TLX_MODEL_ID=$TLX_MODEL_ID > '"$d"'/child.env; exit 0' \
      "$@" zsh "$OPS/engine_daemon.sh" ) >/dev/null 2>&1
}

# ==============================================================================
# D4 + default boot: no state -> registry default; child sees TLX_MODEL_ID
# ==============================================================================
test_default_boot_exports_model_id() {
  local d=$(_sandbox) || return 1
  _wrap "$d"
  grep -q "TLX_MODEL_ID=qwen3.8-27b-egpu" "$d/child.env" 2>/dev/null
}

# ==============================================================================
# D1+promotion: armed next_model -> promoted to current_model; next cleared;
# swap_in_progress cleared; the child boots the NEW model id
# ==============================================================================
test_promotion() {
  local d=$(_sandbox) || return 1
  /usr/bin/python3 - "$d" <<'PY'
import os, sys
sd = os.path.join(sys.argv[1], "ops/state")
open(os.path.join(sd, "next_model"), "w").write("stub-b\n")
open(os.path.join(sd, "swap_in_progress"), "w").write("0 from=a to=b\n")
open(os.path.join(sd, "current_model"), "w").write("qwen3.8-27b-egpu\n")
PY
  _wrap "$d"
  [ "$(head -1 "$d/ops/state/current_model")" = "stub-b" ] \
    && [ ! -f "$d/ops/state/next_model" ] \
    && [ ! -f "$d/ops/state/swap_in_progress" ] \
    && grep -q "TLX_MODEL_ID=stub-b" "$d/child.env" 2>/dev/null
}

# ==============================================================================
# D2: a swap boot clears a SWAP-authored staydown; a breaker-authored staydown
# (no swap_in_progress) still refuses the boot
# ==============================================================================
test_swap_clears_staydown() {
  local d=$(_sandbox) || return 1
  mkdir -p "$d/logs"; print "breaker: x" > "$d/logs/llm_engine_staydown"
  /usr/bin/python3 - "$d" <<'PY'
import os, sys
sd = os.path.join(sys.argv[1], "ops/state")
open(os.path.join(sd, "next_model"), "w").write("stub-b\n")
open(os.path.join(sd, "swap_in_progress"), "w").write("0 from=a to=b\n")
PY
  _wrap "$d"
  [ ! -f "$d/logs/llm_engine_staydown" ] && grep -q "TLX_MODEL_ID=stub-b" "$d/child.env" 2>/dev/null
}
test_breaker_staydown_refuses() {
  local d=$(_sandbox) || return 1
  mkdir -p "$d/logs"; print "breaker: x" > "$d/logs/llm_engine_staydown"
  _wrap "$d"
  [ ! -f "$d/child.env" ]   # refused BEFORE any child ran
}

# ==============================================================================
# D3: invalid armed intent (unknown id / missing env file) -> current_model
# KEPT, staydown written, exit 17, no child
# ==============================================================================
test_invalid_intent_quarantined() {
  local d=$(_sandbox) || return 1
  /usr/bin/python3 - "$d" <<'PY'
import os, sys
sd = os.path.join(sys.argv[1], "ops/state")
open(os.path.join(sd, "current_model"), "w").write("qwen3.8-27b-egpu\n")
open(os.path.join(sd, "next_model"), "w").write("no-such-model\n")
PY
  ( cd "$d" && env TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" \
      TLX_GPU_LOCK="$d/gpu.lock" TLX_ENGINE_SOCK="$d/eng.sock" \
      TLX_ENGINE_DIR="$ENG0" \
      zsh "$OPS/engine_daemon.sh" ) >/dev/null 2>&1
  local rc=$?
  [ "$rc" = "17" ] && [ "$(head -1 "$d/ops/state/current_model")" = "qwen3.8-27b-egpu" ] \
    && [ -f "$d/logs/llm_engine_staydown" ] && [ ! -f "$d/child.env" ]
}
test_missing_env_intent_fails() {
  local d=$(_sandbox) || return 1
  /usr/bin/python3 - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]
r = json.load(open(os.path.join(d, "ops/model_registry.json")))
r["models"]["stub-b"]["env_file"] = "env.canonical.d/GONE.env"
json.dump(r, open(os.path.join(d, "ops/model_registry.json"), "w"))
sd = os.path.join(d, "ops/state")
open(os.path.join(sd, "current_model"), "w").write("qwen3.8-27b-egpu\n")
open(os.path.join(sd, "next_model"), "w").write("stub-b\n")
PY
  ( cd "$d" && env TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" \
      TLX_GPU_LOCK="$d/gpu.lock" TLX_ENGINE_SOCK="$d/eng.sock" \
      TLX_ENGINE_DIR="$ENG0" \
      zsh "$OPS/engine_daemon.sh" ) >/dev/null 2>&1
  local rc=$?
  [ "$rc" = "17" ] && [ "$(head -1 "$d/ops/state/current_model")" = "qwen3.8-27b-egpu" ]
}

# ==============================================================================
# rollback: env split removed -> monolithic env.canonical fallback boots fine
# ==============================================================================
test_monolithic_fallback() {
  local d=$(_sandbox) || return 1
  rm -rf "$d/ops/env.canonical.d"
  # bare clone: the monolithic env.canonical ships as env.canonical.example
  if [[ -f "$OPS/env.canonical" ]]; then
    cp "$OPS/env.canonical" "$d/ops/env.canonical"
  else
    cp "$OPS/env.canonical.example" "$d/ops/env.canonical"
  fi
  _wrap "$d"
  grep -q "TLX_MODEL_ID=qwen3.8-27b-egpu" "$d/child.env" 2>/dev/null
}

# ==============================================================================
# enginectl switch validation: unknown id refused BEFORE any intent is armed
# ==============================================================================
test_switch_refuses_unknown() {
  local d=$(_sandbox) || return 1
  local out
  out=$(env TLX_OPS_ROOT="$d/ops" TLX_LOGS_DIR="$d/logs" \
    TLX_ENGINE_SOCK="$d/nonexistent.sock" zsh "$OPS/enginectl" switch no-such-model 2>&1)
  local rc=$?
  [ "$rc" = "2" ] && [ ! -f "$d/ops/state/next_model" ]
}

# ==============================================================================
# current_model persistence: an explicit current_model (not the default) boots
# without any next_model
# ==============================================================================
test_current_persists() {
  local d=$(_sandbox) || return 1
  /usr/bin/python3 - "$d" <<'PY'
import os, sys
open(os.path.join(sys.argv[1], "ops/state/current_model"), "w").write("stub-b\n")
PY
  _wrap "$d"
  grep -q "TLX_MODEL_ID=stub-b" "$d/child.env" 2>/dev/null
}

t default_boot_exports_model_id test_default_boot_exports_model_id
t promotion_next_model test_promotion
t swap_clears_staydown test_swap_clears_staydown
t breaker_staydown_refuses test_breaker_staydown_refuses
t invalid_intent_quarantined test_invalid_intent_quarantined
t missing_env_intent_fails test_missing_env_intent_fails
t monolithic_fallback test_monolithic_fallback
t switch_refuses_unknown test_switch_refuses_unknown
t current_persists test_current_persists

print -r -- "swap-FSM P8 battery: $PASS pass, $FAIL fail"
[ "$FAIL" = "0" ] || { print -r -- "FAILED:$FAILED"; exit 1 }
exit 0
