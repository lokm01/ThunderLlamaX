# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, time
sys.path.insert(0,"~/tinygrad-src")
from tinygrad import Tensor, UOp, TinyJit, nn
from tinygrad.llm.model import TransformerBlock
pc=time.perf_counter

cfg_ok = True
class C: pass
# build config matching qwen35 attention block minimally
from dataclasses import replace as _repl
from tinygrad.llm.model import TransformerConfig
base = TransformerConfig(
  num_blocks=1, dim=5120, hidden_dim=17408, n_heads=24, n_kv_heads=4,
  norm_eps=1e-6, vocab_size=1000, head_dim=256, v_head_dim=256, rope_theta=1000000.0, rope_dim=256,
  max_context=1024, qk_norm=256)
cfg = _repl(base, qk_norm=256)

blk = TransformerBlock(cfg)
v = UOp.variable("spx", 0, 1022)
jd = TinyJit(lambda t, h, sp: blk(blk.attn_norm(h.cat(h, dim=-1)[:, :, :5120]), sp))

h = Tensor.zeros(1, 1, 5120).float()
t = Tensor([[7]], dtype="int32")
for spv in (14, 15, 16, 17):
    o = jd(t, h, v.bind(spv))
    o = o.realize()
    print(f"call sp={spv} OK out={float(o.float().sum().item()):.3f}", flush=True)
print("REPRO DONE", flush=True)
