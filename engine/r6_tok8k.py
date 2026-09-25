# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from api_server import parse_gguf_kv, SimpleTokenizer, GGUF_PATH
kv = parse_gguf_kv(GGUF_PATH)
tok = SimpleTokenizer.from_gguf_kv(kv)
text = open("~/prompt8k.txt").read()
ids = tok.encode(text)
print("prompt8k tokens:", len(ids), "first:", ids[:8])
np.save("~/r6_p8k_ids.npy", np.array(ids, dtype=np.int64))
