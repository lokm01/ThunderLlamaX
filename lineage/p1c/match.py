# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("JIT","2")
sys.path.insert(0,"~/tinygrad-src")
L=64
MODEL="~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
from tinygrad.llm.model import Transformer
model,_ = Transformer.from_gguf(MODEL,L)
targets={34119680:"A",11141120:"B",20480:"C",1024:"D",17408:"E"}
seen=set()
def scan(tag, obj):
    for n,p in vars(obj).items():
        if not hasattr(p,"nbytes"): continue
        nb=p.nbytes
        if nb in targets and (n,nb) not in seen:
            seen.add((n,nb)); print(f"{tag}.{n}: {nb} B dtype={p.dtype} shape={p.shape}")
scan("blk0", model.blk[0]); scan("blk48", model.blk[48])
for n,p in vars(model).items():
    if hasattr(p,"nbytes") and p.nbytes in targets: print(f"model.{n}: {p.nbytes} B shape={p.shape}")
