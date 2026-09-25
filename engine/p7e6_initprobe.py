# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C
print("[1] importing ok; device:", dev, flush=True)
E = TrunkEngineW1C(theta=1e7)
print("[2] engine built", flush=True)
dev.synchronize()
print("[3] post-init sync OK", flush=True)
P = E.P
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
print("[4] pos_slot ok", flush=True)
for n, i in enumerate(E.attn_idx):
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
    dev.synchronize()
    print(f"[5.{n}] kv{i} zeroed ({i})", flush=True)
print("[6] ALL KV ZEROED", flush=True)
