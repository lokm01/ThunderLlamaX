#!/usr/bin/env python3
"""P9 FIX v3: feed the prefill's `cur` (first response token) through the
visible detok path ONCE, before the generate event loop. Indentation matches
the 2-space style (statements at the try:'s level)."""

P = "~/tinygrad-metal/engine0/api_server.py"
src = open(P).read()

ANCHOR = "    try:\n      while True:\n        r = c.recv()\n        ev = r.get(\"event\")"

FIX = (
    "    # P9 EVAL FIX (first-token loss): in every prefill mode the ENGINE's\n"
    "    # first RESPONSE token is the prefill result's `cur` (predicted at the\n"
    "    # boundary hidden, held in cur_slot -- deliberately NOT part of the fed\n"
    "    # stream nor the cycle emits; serve.py: \"generate appends emits,\n"
    "    # FOLLOW_UP appends [cur]+delta\"). The cycle events start at the SECOND\n"
    "    # response token, so `cur` must be fed through the visible path ONCE,\n"
    "    # before the event loop, or every completion loses its first token\n"
    "    # (one-token answers came back EMPTY; found by the P9 needle eval +\n"
    "    # eval/probe_engine.py).\n"
    "    _cur0 = _pr.get(\"cur\")\n"
    "    if _cur0 is not None:\n"
    "      _cur0 = int(_cur0)\n"
    "      result[\"tokens\"].append(_cur0)\n"
    "      _feed_token(_cur0)\n"
    "      if _cur0 in stopset:\n"
    "        R[\"reusable\"] = False\n"
    "        return bail(\"stop\")\n"
    "      if not split.in_think and result[\"visible\"] >= max_tokens:\n"
    "        return bail(\"length\")\n"
    + ANCHOR
)

if "P9 EVAL FIX (first-token loss)" in src:
    print("already patched")
    raise SystemExit(0)
assert src.count(ANCHOR) == 1, f"anchor count {src.count(ANCHOR)}"
src = src.replace(ANCHOR, FIX)
open(P, "w").write(src)
print("patched ok")
