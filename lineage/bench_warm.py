# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time, os, argparse
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf')
ap.add_argument("--lengths", type=lambda s:[int(x) for x in s.split(',')], default="1024,4096,16384,32768,65536,100000")
ap.add_argument("--decode", type=int, default=5, help="tokens to decode per measurement")
ap.add_argument("--warm", type=int, default=3, help="warmup tokens")
args = ap.parse_args()

t0 = time.time()
model, kv = Transformer.from_gguf(args.model, 300000)
print(f"[load] {time.time()-t0:.1f}s", flush=True)

tok = SimpleTokenizer.from_gguf_kv(kv)
env = jinja2.Environment(); env.filters['tojson'] = lambda obj,**k: json.dumps(obj,**k)
env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
env.globals['eos_token'] = tok.decode([tok.eos_id])

# filler tokens that tokenize to readable but neutral text (large enough unit)
fill = "The quick brown fox jumps over the lazy dog. " * 30  # ~29 tokens per rep

def fill_to(n_tokens):
  """Return a token list of approximately n_tokens."""
  ids, reps = [], 0
  fids = tok.encode(fill)
  while len(ids) < n_tokens and reps < 5000:
    ids = ids + fids
    reps += 1
  return ids[:n_tokens]

def decode_at(ctx_len):
  """Prefill to ctx_len, warm, then measure decode tok/s."""
  # Build a prompt that yields ~ctx_len tokens of context
  ids = [tok.bos_id if tok.bos_id is not None else 0] + fill_to(ctx_len - 8)
  prompt_len = len(ids)
  # Warm the graph with a couple tokens
  warm = list(model.generate(ids[:64], chunk_size=32, temperature=0.0))
  # Now prefill to full length and decode --- measure steady decode
  # generate() appends and maintains state internally; easier: just run generate from prompt
  # Prefill graph first call may JIT; do it, discard
  _ = list(model.generate(ids[:min(prompt_len, 512)], temperature=0.0))
  return prompt_len

# Measure decode at each target length by generating a running sequence
lengths = args.lengths
results = []
# generate a single long stream; measure tok/s over windows ending at each length
stream_buf = [tok.bos_id if tok.bos_id is not None else 0] + fill_to(64)
# We'll use model.generate to accumulate context and time windows
gen = model.generate(list(stream_buf), chunk_size=32, temperature=0.0)
# warm
cur_len = len(stream_buf)
toks_past = []
t_next = next(gen)  # advance once to force graph build
toks_past.append(t_next)
cur_len += 1
for target in lengths:
  while cur_len < target - 1:
    t = next(gen)
    toks_past.append(t)
    cur_len += 1
  # measure a window at this context length
  times = []
  for _ in range(args.decode):
    s = time.time()
    t = next(gen)
    toks_past.append(t)
    times.append(time.time() - s)
    cur_len += 1
  dt = sum(times)/len(times)
  tokps = 1.0/dt
  results.append((cur_len - args.decode, tokps))
  print(f"ctx~{cur_len-args.decode:>7}  decode {tokps:6.2f} tok/s  ({dt*1000:.0f} ms/tok)  [warm]", flush=True)

print("done", flush=True)
