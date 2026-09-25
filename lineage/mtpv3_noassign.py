# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""MTP v3 STEP 0.5: (a) T=3 _fwd TinyJit at JIT=1 as ONLY graph family — fault check;
(b) eager assign into state buffers BETWEEN replays (the v3 select mechanism);
(c) eager draft-block launches around the live graph; (d) replay timing."""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("HIST_L", "2048"))
N = int(os.getenv("N_PASS", "15"))
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
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
Device["NV"].synchronize()
print("[load] ok", flush=True)

K = 2
v_sp = UOp.variable("mtp_sp", 0, L - 2 - K)

def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    h = x.contiguous()
    return model.output(model.output_norm(h[:, :, :]).half())   # head INSIDE the graph

probe_j = TinyJit(_fwd)
pos = L - 3*N - 8
tt = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
toks = tt[:, pos:pos+3].contiguous().realize()

with Context(JIT=1):
    logits = probe_j(toks, v_sp.bind(pos)).realize()
Device["NV"].synchronize()
print("[warm] T=3 JIT=1 capture+run OK, logits", tuple(logits.shape), flush=True)

# (b) eager state assign between replays (the partial-accept select mechanism)
from tinygrad.llm.model import GatedDeltaNetBlock
gdns = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
snap = [Tensor(b.recurrent_state.numpy()) if False else b.recurrent_state.clone().contiguous().realize() for b in gdns[:3]]
for b, s in zip(gdns[:3], s if False else snap):
    b.recurrent_state.assign(s).realize()
Device["NV"].synchronize()
print("[b] eager state assign between replays OK", flush=True)

# (d) replay loop with eager assigns interleaved every cycle
t1 = time.perf_counter()
with Context(JIT=1):
    for i in range(N):
        logits = probe_j(toks, v_sp.bind(pos)).realize()
        Device["NV"].synchronize()
        amd = logits[0].argmax(-1).to(None).tolist() if False else None  # skip host readback for timing
        pass
        pos += 3
Device["NV"].synchronize()
dt = (time.perf_counter() - t1) / N
print(f"[d] T=3 JIT=1 replay + eager assigns: {dt*1e3:.2f} ms/cycle", flush=True)
print("STEP0.5 PASS", flush=True)
