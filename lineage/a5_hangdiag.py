# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""p1_sanity + hang diagnosis: on RuntimeError dump last sig_prof_records (kernel names)."""
import sys, os, json
sys.path.insert(0, "~/tinygrad-metal")
os.environ.setdefault("JIT", "2")
from mtp_config import MTPConfig
CFG = MTPConfig.load()
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
model, kv = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", CFG.max_context)
tok = SimpleTokenizer.from_gguf_kv(kv)
ids = [0] + tok.encode(CFG.prompt)
gen = model.generate(ids, chunk_size=32, temperature=0.0)
outs = []
try:
    while len(outs) < 12: outs.append(next(gen))
    print("COMPLETED", outs)
except Exception as e:
    dev = None
    try:
        from tinygrad.device import Device
        dev = Device["NV"]
    except Exception: pass
    print("HANG/ERR:", type(e).__name__, str(e)[:200])
    if dev is not None:
        recs = getattr(dev, "sig_prof_records", [])
        print(f"total records: {len(recs)}")
        for r in recs[-24:]:
            try: print("  ", r[2], "dt_us=", float(r[1].timestamp) - float(r[0].timestamp))
            except Exception: print("  ", r)
    raise
