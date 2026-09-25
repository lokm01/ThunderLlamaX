# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys
sys.path.insert(0,"~/tinygrad-src")
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.gguf import gguf_load
def v(): return round(GlobalCounters.mem_used_per_device.get("NV",0)/1e9,2)
print("start", v(), flush=True)
m,kv = Tensor.from_gguf if False else (None,None)
from tinygrad.llm.model import Transformer
model,_ = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", 1024)
print("after from_gguf", v(), flush=True)
kv2, sd = gguf_load("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
print("after gguf_load", v(), flush=True)
x = Tensor.kaiming_uniform(17408, 5120, dtype="float32")
x.realize()
print("after one kaiming fp32", v(), flush=True)
