# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Generate a longer greedy baseline (200 tokens) into spec_base200.json."""
import sys, os, json, time
sys.path.insert(0, "~/tinygrad-metal")
os.environ.setdefault("JIT", "1")
from mtp_config import MTPConfig
CFG = MTPConfig.load()
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
model, kv = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", CFG.max_context)
tok = SimpleTokenizer.from_gguf_kv(kv)
ids = [0] + tok.encode(CFG.prompt)
t0 = time.perf_counter()
gen = model.generate(ids, chunk_size=32, temperature=0.0)
outs = []
while len(outs) < 200:
    outs.append(next(gen))
json.dump(outs, open("~/tinygrad-metal/spec_base200.json", "w"))
print(f"BASE200 done in {time.perf_counter()-t0:.1f}s: {outs[:12]}")
