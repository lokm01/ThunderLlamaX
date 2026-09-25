# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Per-block anatomy: kernels + ms for ONE GDN block and ONE attn block at T=1.
Reads real weights from the model, isolates each block in its own TinyJit,
prints 'jit execs' kernel count (DEBUG>=1) and steady replay time."""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, TransformerBlock
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import DEBUG, getenv
import tinygrad.helpers as H

# force DEBUG>=1 so CapturedJit prints "jit execs N calls"
os.environ["DEBUG"] = os.environ.get("DEBUG", "1")

MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
model, kv = Transformer.from_gguf(MODEL, 256)
print("[load] done", flush=True)

v = UOp.variable("sp", 0, 254)
gdn = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
att = next(b for b in model.blk if isinstance(b, TransformerBlock))

def bench_block(b, tag):
    def _f(x, sp): return b(x, sp).contiguous()
    j = TinyJit(_f)
    x = Tensor.kaiming_uniform(1, 1, 5120).half().contiguous().realize()
    for i in range(3): j(x.clone(), v.bind(i))
    H.DEBUG.value = 1
    j(x.clone(), v.bind(10))  # prints "jit execs N calls"
    H.DEBUG.value = 0
    # time
    N = 50
    for _ in range(5): j(x.clone(), v.bind(20))
    from tinygrad.device import Device
    Device["NV"].synchronize()
    t0 = time.perf_counter()
    for i in range(N): j(x.clone(), v.bind(30 + i))
    Device["NV"].synchronize()
    dt = (time.perf_counter() - t0) / N
    print(f"[{tag}] {dt*1e3:.3f} ms/block", flush=True)
    return dt

d_gdn = bench_block(gdn, "GDN block T=1")
d_att = bench_block(att, "ATT block T=1")
print(f"[model est] 48*GDN + 16*ATT = {(48*d_gdn+16*d_att)*1e3:.1f} ms (vs measured 89.3 full model)", flush=True)
print("DONE_ANATOMY", flush=True)
