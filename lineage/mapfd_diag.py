# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""MAP_SYSMEM_FD mapping count diagnostic v2: T=1 base vs T=3 probe (JIT=1 capture).
Env: DEV=NV BEAM=1 MTP_MAPFD_DIAG=1 [MODE=base|probe] [L=context_length]
Mirrors mtp_v3.py's probe construction exactly (stable-buffer inputs, same env recipe).
"""
import os, sys
MODE = os.getenv("MODE", "base")
L = int(os.getenv("L", "8192"))
NTOK = int(os.getenv("NTOK", "5"))

# MTP env must be set BEFORE tinygrad imports (model.py knobs read at import/block-init)
if MODE == "probe":
    os.environ.update({
        "MTP_T3_LAZY": "1", "MTP_SEQ_ATTN": "1", "MTP_STEP_STATES": "1",
        "MTP_PROBE_RO": "1", "MTP_EMB_GATHER": "1", "MTP_BLKGROUP": "8",
    })
    if L > 1024: os.environ["MTP_A3C_OFF"] = "1"
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("JIT", "1")

sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")

import numpy as np
from tinygrad.llm.model import Transformer, flush_step_states
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.engine.jit import TinyJit

MODEL = os.getenv("MTP_MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
print(f"[load] L={L} mode={MODE}", flush=True)
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()
dev = Device["NV"]

import tinygrad.runtime.support.system as _sys_mod
def mfd(): return getattr(_sys_mod, "_MAPFD_TOTAL", 0)

ids = [0] + [42] * min(32, L // 4)
sp0 = len(ids)
v_sp = UOp.variable("start_pos", 0, L - 1)
tokbuf = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
dev.allocator._copyin(tokbuf.uop.buf_uop.buffer._bufs["NV"],
                      memoryview(np.zeros((1, L), dtype=np.int32).tobytes()).cast("B"))
temp = Tensor([0.0])
print(f"[pre] mapfd={mfd()}", flush=True)

if MODE == "base":
    out = None
    for i in range(NTOK + 2):
        inp = tokbuf[:, sp0+i:sp0+i+1] if out is None else out
        out = model(inp, v_sp.bind(sp0 + i), temp).realize()
        Device["NV"].synchronize()
        print(f"[base tok {i}] mapfd={mfd()}", flush=True)
else:
    def _fwd3(tokens, start_pos, want_logits=False):
        if getattr(model, "emb_rows", None) is not None:
            from tinygrad.llm.gguf import ggml_data_to_tensor as _g2t
            _n = model.emb_out_dim
            x = _g2t(model.emb_rows[tokens.reshape(-1)].cast(dtypes.uint8),
                     _n * tokens.numel(), model.emb_ggml_type) \
                  .reshape(tokens.shape[0], tokens.shape[1], _n).cast(dtypes.float32)
        else:
            x = model.token_embd(tokens).float()
        for _bi, b in enumerate(model.blk):
            x = b(x, start_pos)
            if (_bi + 1) % 8 == 0:
                x = x.contiguous().realize()
        h = flush_step_states(x.contiguous(), model.blk).contiguous()
        if want_logits: return h, model.output(model.output_norm(h).half())
        return h

    probe_j = TinyJit(_fwd3)
    _TOK = Tensor([[1, 2, 3]], dtype=dtypes.int32).contiguous().realize()
    for i in range(4):
        try:
            with Context(JIT=1):
                ret = probe_j(_TOK, v_sp.bind(sp0 + i), True)
                h = ret[0].contiguous().realize()
            Device["NV"].synchronize()
            print(f"[probe call {i}] mapfd={mfd()}", flush=True)
        except Exception as e:
            print(f"[probe call {i}] FAILED: {type(e).__name__}: {e} | mapfd={mfd()}", flush=True)
            break

print("DONE", flush=True)
