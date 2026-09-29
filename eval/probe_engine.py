#!/usr/bin/env python3
"""P9 diagnostic: probe the ENGINE socket directly with the same rendered
prompt the API would send; print the raw cycle tokens + detok. Splits
engine-side vs API-side for the first-token-loss bug."""
import json, socket, sys

sys.path.insert(0, "~/tinygrad-metal/engine0")
import api_server as A

MSGS = [{"role": "user", "content": "Reply with the exact number 12345."}]
TVARS = {"reasoning_effort": "medium", "enable_thinking": False}
rtext = A._render_msgs(MSGS, True, TVARS) if hasattr(A, "_render_msgs") else None
if rtext is None:
    rtext = A._template_render(MSGS, True, TVARS)
ids = A.TOK.encode(rtext)
print("rendered:", repr(rtext))
print("ids:", ids[:20], "... n=", len(ids))

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(300)
s.connect("/tmp/llm-engine.sock")
f = s.makefile("rwb")

def send(o):
    f.write((json.dumps(o) + "\n").encode()); f.flush()

send({"id": 10, "method": "prefill", "params": {
    "mode": "FRESH", "ids": ids, "conversation_id": "p9-probe-firsttok",
    "model_id": "qwen3.8-27b-egpu", "rid": "p9-probe-firsttok"}})
pr = None
while True:
    r = json.loads(f.readline())
    if r.get("id") == 10 and "event" not in r:
        pr = r
        break
print("prefill result:", {k: v for k, v in (pr.get("result") or {}).items() if k != "fed"})
send({"id": 11, "method": "generate", "params": {"max_cycles": 24, "rid": "p9-probe-firsttok"}})
toks_all = []
while True:
    r = json.loads(f.readline())
    if r.get("id") == 11:
        if r.get("event") == "cycle":
            t = r.get("tokens") or []
            toks_all += t
            print("cycle tokens:", t, "->", repr(A.TOK.decode(t)))
        elif r.get("event") in ("done", "cancelled"):
            print("terminal:", r.get("event"), "stop:", r.get("stop"),
                  "tokens:", r.get("tokens"))
            break
        elif r.get("event") == "error":
            print("ERROR EVENT:", r)
            break
        elif not r.get("ok", True):
            print("RPC ERROR:", r)
            break
print("ALL ENGINE TOKENS:", toks_all)
print("DETOK FULL:", repr(A.TOK.decode(toks_all)))
