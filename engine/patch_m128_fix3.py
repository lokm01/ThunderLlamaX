# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c fix 3: the M128 tail must clear BOTH ambient flags — with _M64ON left
True, _pf_graphs builds the M64 plan/graphs while the 34-row tail runs the M32
path -> the M32 chunk replays the M64 graph on ids64 (the P15 law-2 bug
re-committed; 8k F 3.0e-1 first-div-0; the 7680 no-tail control 60/60 EXACT)."""
def patch(path, subs):
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        assert n == 1, (n, old[:80])
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("[patch]", len(subs), "edits")

patch("~/tinygrad-metal/engine0/pf_prefill.py", [(
'''    _s128, _s64 = _M128ON, _M64ON
    m128_set(False)   # the tail MUST run the M64/M32 plans (graph-cache ambient-flag law)
    try:''',
'''    _s128, _s64 = _M128ON, _M64ON
    m128_set(False); m64_set(False)   # P15 law-2: clear BOTH flags — the tail runs the
    try:                              # M32 plan AND the M32 graphs (m64 left on = the M64 graph)''')])
