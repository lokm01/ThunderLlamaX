# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P0 GDN fused-chain microbench: T=1 _attention of ONE real GDN block.
Noise reduction: 20 calls batched inside ONE TinyJit (JIT=2 graphless), 100 timed
replays; report ms/block (= wall/20) and x48 extrapolation. Also single-call eager
timing for contrast (block_anatomy's flawed mode).
Run: DEV=NV ~/tg311/bin/python gdn_chain_bench.py
"""
import os, sys, time, json
os.environ.setdefault("JIT", "2")
sys.path.insert(0, "~/tinygrad-src")
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
NBATCH = int(os.getenv("NBATCH", "20"))
ITS = int(os.getenv("ITERS", "100"))
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit

model, _ = Transformer.from_gguf(MODEL, 2048)
cfg = model.blk[-1].config
gdns = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
b0 = gdns[0]
for b in gdns:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    b.conv_state.assign(Tensor.rand(b.conv_state.shape)).realize()
    b.recurrent_state.assign(Tensor.rand(b.recurrent_state.shape)).realize()
Device["NV"].synchronize()

sp = UOp.variable("start_pos", 1, 2047)
BLKS = gdns[:NBATCH]   # NBATCH DISTINCT blocks, one call each - exactly mirrors one
                       # production T=1 token (48 blocks x 1 call). Re-calling the SAME
                       # block 20x in one graph trips 'cycle detected in assign graph'.
XS = [Tensor.rand(1, 1, cfg.dim).realize() for _ in range(NBATCH)]

# --- batched-in-one-jit timing ---
def chain(*xs):
    # full block.__call__ chained via residual stream - the exact pattern the stock
    # rollout_jit uses (proven safe); _attention-direct TinyJits are a documented
    # device-fault trip-wire.
    x = xs[0]
    outs = []
    for b in BLKS:
        x = b(x, sp.bind(123))
        outs.append(x.contiguous())
    return outs
jfn = TinyJit(chain)
XS = XS[:1]  # single seed input; blocks chain sequentially
outs = jfn(*XS); Tensor.realize(*outs)            # capture
for _ in range(5): outs = jfn(*XS); Tensor.realize(*outs)  # warm
Device["NV"].synchronize()
t0 = time.perf_counter()
for _ in range(ITS):
    outs = jfn(*XS); Tensor.realize(*outs[-1])
Device["NV"].synchronize()
dt = time.perf_counter() - t0
ms_block = dt / ITS / NBATCH * 1e3   # NBATCH sequential full blocks / time

# --- single-call eager (contrast; includes launch tax) ---
x1 = XS[0]
for _ in range(5): b0._attention(x1, sp.bind(123)).realize()
Device["NV"].synchronize()
t0 = time.perf_counter()
for _ in range(30): b0._attention(x1, sp.bind(123)).realize()
Device["NV"].synchronize()
single_ms = (time.perf_counter() - t0) / 30 * 1e3

res = {"ms_per_block_batched20": round(ms_block, 3), "x48_blocks_ms": round(ms_block*48, 1),
       "ms_per_call_single_eager": round(single_ms, 3)}
json.dump(res, open(os.path.expanduser("~/tinygrad-metal/gdnbench.json"), "w"), indent=1)
print("== GDN CHAIN BENCH ==")
for k, v in res.items(): print(f"{k}: {v}")
print("DONE_GDN")
