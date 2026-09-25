# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Generalized greedy baseline generator.
Env: MTP_PROMPT_FILE / MTP_PROMPT_TEXT / MTP_MAXCTX / GEN_N / GEN_OUT
"""
import sys, os, json, time
sys.path.insert(0, "~/tinygrad-metal")
os.environ.setdefault("JIT", "1")
from mtp_config import MTPConfig
CFG = MTPConfig.load()
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
MAXCTX = int(os.getenv("MTP_MAXCTX", "0")) or CFG.max_context
model, kv = Transformer.from_gguf(os.getenv("MTP_MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"), MAXCTX)
tok = SimpleTokenizer.from_gguf_kv(kv)
if os.getenv("MTP_HEAD_INT8"):
    from head_i8 import install as _i8install
    _i8install(model)   # baseline MUST use the same int8 head as the MTP runs
_prompt = CFG.prompt
if os.getenv("MTP_PROMPT_FILE"): _prompt = open(os.getenv("MTP_PROMPT_FILE")).read()
elif os.getenv("MTP_PROMPT_TEXT"): _prompt = os.getenv("MTP_PROMPT_TEXT")
ids = [0] + tok.encode(_prompt)
N = int(os.getenv("GEN_N", "60"))
OUT = os.getenv("GEN_OUT", "~/tinygrad-metal/spec_basex.json")
t0 = time.perf_counter()
gen = model.generate(ids, chunk_size=32, temperature=0.0)
outs = []
while len(outs) < N: outs.append(next(gen))
json.dump(outs, open(OUT, "w"))
dt = time.perf_counter() - t0
print(f"BASEX done in {dt:.1f}s ({dt/N*1e3:.0f} ms/tok incl prefill {len(ids)}): {outs[:12]}")
