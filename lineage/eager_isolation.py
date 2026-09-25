# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Isolate the eager-work probe penalty: probe timing after (a) nothing, (b) 1 tiny eager
kernel, (c) full draft, (d) sync-only. Run 4 probe replays in a row at cycle 3."""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
os.environ.setdefault("JIT", "2")
L = int(os.getenv("HIST_L", "1024"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

from tinygrad.llm.model import Transformer
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context

model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()
print("[load] ok", flush=True)

v_sp = UOp.variable("mtp_sp", 0, L - 4)
def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    from tinygrad.llm.model import flush_step_states
    return flush_step_states(x.contiguous(), model.blk).contiguous()

probe_j = TinyJit(_fwd)
toks = Tensor([[100, 200, 300]], dtype="int32").contiguous().realize()

with Context(JIT=1):
    h = probe_j(toks, v_sp.bind(10)).realize()
Device["NV"].synchronize()

def t_probe(tag):
    t0 = time.perf_counter()
    with Context(JIT=1):
        h = probe_j(toks, v_sp.bind(12)).realize()
    Device["NV"].synchronize()
    print(f"[{tag}] {time.perf_counter()-t0:.3f}s", flush=True)
    return h

# warm replay
t_probe("warm")

# (a) back-to-back
t_probe("a-back-to-back")

# (b) one tiny eager kernel then probe
_tiny = Tensor([1.0, 2.0]).realize()
t0 = time.perf_counter()
_tiny2 = (_tiny + 1).realize()
Device["NV"].synchronize()
print(f"[tiny-eager] {time.perf_counter()-t0:.4f}s", flush=True)
t_probe("b-after-1-tiny-eager")

# (c) sync only
Device["NV"].synchronize()
t_probe("c-after-sync-only")

# (d) 100 tiny eager kernels then probe
t0 = time.perf_counter()
acc = _tiny
for i in range(100):
    acc = (acc + 1).realize()
Device["NV"].synchronize()
print(f"[100-eager] {time.perf_counter()-t0:.3f}s", flush=True)
t_probe("d-after-100-tiny-eager")

# (e) one T=1 model fwd (eager JIT=2, like the draft) then probe
t0 = time.perf_counter()
with Context(JIT=2):
    _fwd(Tensor([[42]], dtype="int32").contiguous(), v_sp.bind(15)).realize()
Device["NV"].synchronize()
print(f"[t1-eager-fwd] {time.perf_counter()-t0:.3f}s", flush=True)
t_probe("e-after-t1-eager-fwd")
print("DONE", flush=True)
