# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""3-block chain validation: fwd3_split vs the stock model forward on the SAME
live states, real weights. Compares the final hidden exactly."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.device import Device
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock

os.environ.setdefault("BEAM", "1")
os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1")
os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_MAXCTX", "2048")
model = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", 2048)[0]
dev = Device["NV"]
DIM = model.blk[0].attn_norm.weight.shape[0]

import fwd3_split
fwd3_split.init(model, T=3)

rng = np.random.default_rng(7)
T = 3
# init live GDN states the model's native way, then snapshot
for b in model.blk:
    if isinstance(b, GatedDeltaNetBlock):
        b._init_state(Tensor.zeros(1, T, DIM))
# zeros initial states (both sides identical, no dtype games)

toks = Tensor(rng.integers(0, 1000, (1, T)).astype(np.int32)).contiguous().realize()
sp = 100

# ---- snapshot ALL in-place state (attn KV + GDN) before the reference ----
_snap = []
for b in model.blk:
    _snap.append((getattr(b, "cache_kv", None).clone().realize() if getattr(b, "cache_kv", None) is not None else None,
                  b.conv_state.clone().realize() if hasattr(b, "conv_state") else None,
                  b.recurrent_state.clone().realize() if hasattr(b, "recurrent_state") else None))
import copy
# reference TRUNK hidden (embed + blocks, stop before output_norm/head — same as fwd3_split)
_rx = model.token_embd(toks).float()
for _b in model.blk:
    _rx = _b(_rx, sp).contiguous().realize()
ref_h = _rx.float().numpy()

# restore ALL state for the split run (same starting point as the reference)
for b, (kv, cv, rs) in zip(model.blk, _snap):
    if kv is not None: b.cache_kv.assign(kv).realize()
    if cv is not None: b.conv_state.assign(cv).realize()
    if rs is not None: b.recurrent_state.assign(rs).realize()

# ---- split path ----
h = fwd3_split.fwd3_split(model, toks, sp)
got = h.float().numpy()

rel = np.abs(got - ref_h).max() / max(np.abs(ref_h).max(), 1e-9)
print(f"3-block-chain(actually full model) h relerr = {rel:.2e}")
print("VERDICT:", "PASS" if rel < 1e-3 else "FAIL")
