# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""100k T=3 probe kernel histogram from the repro's sig_prof_records."""
import os, sys, time, json, collections
os.environ.setdefault("DEV", "NV")
os.environ.update({"MTP_T3_LAZY": "1", "MTP_SEQ_ATTN": "1", "MTP_PROBE_RO": "1", "MTP_A3C_OFF": "1",
                   "MTP_EMB_GATHER": "1", "MTP_HEAD16_DIRECT": "1"})
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad.llm.model import Transformer, flush_step_states
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device

MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
model, kv = Transformer.from_gguf(MODEL, 100352)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
Device["NV"].synchronize()
print("[loaded]", flush=True)

sp = UOp.variable("start_pos", 0, 100351)
dev = Device["NV"]

def fwd3(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for _bi, b in enumerate(model.blk):
        x = b(x, start_pos)
        if (_bi + 1) % 8 == 0:
            x = x.contiguous().realize()
    h = flush_step_states(x.contiguous(), model.blk).contiguous()
    return h, model.output(model.output_norm(h).half())

toks = Tensor([[11, 22, 33]], dtype=dtypes.int32).contiguous().realize()
pos = 94208
for _ in range(3):
    h, lg = fwd3(toks, sp.bind(pos)); Tensor.realize(h, lg); pos += 3
Device["NV"].synchronize()
recs = getattr(dev, "sig_prof_records", [])
agg = collections.Counter(); tot = collections.Counter()
for st, en, name, dname, pk in recs:
    try: dt_us = float(en.timestamp) - float(st.timestamp)
    except Exception: continue
    if dt_us > 0:
        agg[name] += dt_us/1e3; tot[name] += 1
rows = sorted(agg.items(), key=lambda x: -x[1])
total = sum(agg.values())
print(f"\n[100k T=3 eager] kern_exec={total:.1f}ms kernels={sum(tot.values())}", flush=True)
for nm, tm in rows[:25]:
    print(f"  {tm*1000/tot[nm]:9.1f}us x{tot[nm]:<4} {tm:8.3f}ms  {nm[:100]}", flush=True)
