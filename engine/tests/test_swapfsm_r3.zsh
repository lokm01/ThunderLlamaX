#!/bin/zsh
# TLX R3 W-E.4 SCAFFOLD: swap-FSM crash-step tests — build when the MoE swap
# lands (MM_PLAN). The D1-D4 hardening items these will assert (ledger §3.1):
#   D1 current_model writes are atomic (tmp+rename+fsync+dir-fsync) — a torn
#      write must NEVER silently boot the DEFAULT model while the API
#      "verifies" a swap that never happened.
#   D2 the swap path never deletes a BREAKER-authored staydown (distinct,
#      content-tagged markers).
#   D3 intents validate against model_registry.json BEFORE touching
#      current_model; invalid intent -> keep current + quarantine intent +
#      boot current (fail-operational).
#   D4 crashlog lines carry the booted MODEL_ID (per-model EXPECTED_FP).
# Method: SIGKILL a stub swap-runner at each state transition (drain,
# post-intent, mid-shutdown-RPC, boot) -> the supervisor reaches a
# deterministic terminal state.
print "swap-FSM battery: SCAFFOLD (activates with the MoE swap; D1-D4 above)"
exit 0
