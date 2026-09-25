# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor
import tinygrad.engine.jit as J

# ---- intercept graph batching decisions ----
_orig_create = J.create_graph_call
BATCHES=[]
def spy_create(batch):
    BATCHES.append(len(batch))
    return _orig_create(batch)
J.create_graph_call = spy_create

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)

text="Machine learning is a field of study in artificial intelligence concerned with the development "*12
ids=[0]+tok.encode(text)
gen=model.generate(list(ids), chunk_size=32, temperature=0.0)
for _ in range(6): next(gen)
print(f"\n== CAPTURE DIAG ==\ngraph batches formed: {len(BATCHES)}", flush=True)
if BATCHES:
    import collections
    c=collections.Counter(BATCHES)
    print("size histogram:", dict(sorted(c.items())), flush=True)
    print(f"kernels covered by graphs: {sum(BATCHES)}", flush=True)
print(f"total kernels/token (GC): {GlobalCounters.kernel_count}", flush=True)

# quick decode sanity
N=6
w0=pc()
for _ in range(N): next(gen)
wall=pc()-w0
print(f"decode: {N/wall:.2f} tok/s ({wall/N*1e3:.1f} ms/tok)", flush=True)
print("DONE", flush=True)
