# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 RUNG B: build the prose corpus (re-encoded reply streams) for the
graded-LOOKUP offline sim. Uses api_server zero-GPU tokenizer; reply tail
encode is exact at the special-token boundary (M1-C re-encode law: model
splits are not BPE-canonical — acceptable statistical noise for the sim)."""
import json, sys, types
import numpy as np
sys.path.insert(0, "~/tinygrad-metal/engine0")
# stub the serving-only imports (api_server is zero-GPU but imports fastapi at mod level)
class _Any:
  def __call__(self, *a, **kw): return _Any()
  def __getattr__(self, k): return _Any()
class _Stub(types.ModuleType):
  def __getattr__(self, k): return _Any()
for m in ("fastapi", "fastapi.responses", "uvicorn", "httpx",
          "starlette", "starlette.concurrency"):
  mod = _Stub(m); mod.__path__ = []   # make it a package so submodule lookups short-circuit
  sys.modules[m] = mod
from api_server import load_tokenizer as _lt
TOK, TEMPLATE = _lt()

def reply_ids(user_text, reply_text):
  # generation header ends at "<|im_start|>assistant\n" — the reply tail
  hdr = TEMPLATE.render(messages=[{"role": "user", "content": user_text}], add_generation_prompt=True)
  full = hdr + reply_text
  ih, ifull = TOK.encode(hdr), TOK.encode(full)
  assert ifull[:len(ih)] == ih, "hdr not a prefix (unexpected)"
  return ifull[len(ih):]

def main():
  out = {}
  for nm, q in (("prose1", "Write a vivid paragraph about a lighthouse in a storm."),
                ("prose2", "Explain in detail how a turbofan jet engine works, from intake to exhaust. Write several paragraphs of fluent technical prose.")):
    txt = json.load(open(f"~/{nm}.json")) if False else json.load(open(f"~/r8_{nm}.json"))
    ids = reply_ids(q, txt)
    out[nm] = ids
    print(f"[corpus] {nm}: {len(ids)} reply toks; first 12 {ids[:12]}")
    # repetition sanity: 8-gram self-overlap count within the reply
    s = set(); dup = 0
    for i in range(len(ids) - 8):
      k = tuple(ids[i:i+8])
      if k in s: dup += 1
      s.add(k)
    print(f"[corpus] {nm}: dup 8-grams within reply: {dup}/{max(len(ids)-8,1)}")
  np.save("~/r8_prose_ids.npy", np.array(out["prose1"] + out["prose2"], dtype=np.int64))
  print(f"[corpus] combined {len(out[chr(112)+chr(114)+chr(111)+chr(115)+chr(101)+chr(49)]) + len(out[chr(112)+chr(114)+chr(111)+chr(115)+chr(101)+chr(50)])} toks -> ~/r8_prose_ids.npy")

if __name__ == "__main__":
  main()
