# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Gate (g): template render + encode must be BIT-STABLE vs the fork's own
tokenizer path (tinygrad.llm.cli.SimpleTokenizer + jinja2 with the fork's exact
env setup). Run under ~/tg311/bin/python (has tinygrad + jinja2; no GPU use).
Prints JSON: {"ids": [...], "text": "..."} for the canonical test messages."""
import sys, json
sys.path.insert(0, "~/tinygrad-src")
GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

# minimal gguf KV parser (same format logic as api_server; independent copy)
import struct
def parse(path):
  f = open(path, "rb"); assert f.read(4) == b"GGUF"
  struct.unpack("<I", f.read(4)); struct.unpack("<Q", f.read(8))
  n_kv, = struct.unpack("<Q", f.read(8))
  def rs():
    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
  def rv(t):
    m = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),6:("<f",4),10:("<Q",8),11:("<q",8),12:("<d",8)}
    if t in m: return struct.unpack(m[t][0], f.read(m[t][1]))[0]
    if t == 7: return bool(f.read(1)[0])
    if t == 8: return rs()
    if t == 9:
      et, = struct.unpack("<I", f.read(4)); cnt, = struct.unpack("<Q", f.read(8))
      return [rv(et) for _ in range(cnt)]
    raise ValueError(t)
  kv = {}
  for _ in range(n_kv):
    k = rs(); t, = struct.unpack("<I", f.read(4)); kv[k] = rv(t)
  f.close(); return kv

kv = parse(GGUF)
from tinygrad.llm.cli import SimpleTokenizer
import jinja2
tok = SimpleTokenizer.from_gguf_kv(kv)
env = jinja2.Environment()
env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
tmpl = env.from_string(kv['tokenizer.chat_template'])
msgs = [
  {"role": "system", "content": "You are a concise assistant."},
  {"role": "user", "content": "Hi! What is 2+2?"},
  {"role": "assistant", "content": "It is 4."},
  {"role": "user", "content": "And 3+3? Also: naïve café ☕ emoji test."},
]
text = tmpl.render(messages=msgs, add_generation_prompt=True)
ids = tok.encode(text)
print(json.dumps({"ids": ids, "text": text, "eot": tok.eot_id, "eos": tok.eos_id}))
