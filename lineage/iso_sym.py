# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Does the symbolic-slice TinyJit rebind sp/toks on replay? Compare eager vs jit hidden."""
import os, time, sys
sys.path.insert(0, "~/tinygrad-src")
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device

MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
model, kv = Transformer.from_gguf(MODEL, 256)
print("[load] done", flush=True)

v_sp = UOp.variable("sp", 0, 254)
v_toks = UOp.variable("toks", 1, 3)

def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    return x.contiguous()
jit = TinyJit(_fwd)

def fwd(pos, toks, nt):
    t_full = Tensor([toks + [0] * (256 - nt)], dtype="int32").reshape(1, 256).contiguous()
    sp_b, nt_b = v_sp.bind(pos), v_toks.bind(nt)
    h = jit(t_full[:, sp_b:sp_b+nt_b], sp_b).realize()
    Device["NV"].synchronize()
    return h

def eager(pos, toks, nt):
    t = Tensor([toks], dtype="int32").contiguous()
    x = model.token_embd(t).float()
    for b in model.blk:
        b._init_state(x)
        hh = x + b._attention(b.attn_norm(x), pos)
        x = (hh + b._feed_forward(b.ffn_norm(hh))).contiguous().realize()
    return x

# seq of tokens; compare eager vs jit at each pos (jit: capture then replay)
seq = [11, 150, 999, 42, 7]
for i, tid in enumerate(seq):
    he = eager(i, [tid], 1)
    hj = fwd(i, [tid], 1)
    d = float((he - hj).abs().max().item())
    print(f"pos={i} tok={tid} jit_cnt={jit.cnt} maxdiff={d:.4e}", flush=True)

# same pos twice (pure replay)
he = eager(2, [999], 1); hj = fwd(2, [999], 1)
print(f"replay pos=2 maxdiff={float((he-hj).abs().max().item()):.4e}", flush=True)
print("ISO_SYM DONE", flush=True)
