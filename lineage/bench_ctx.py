# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Decode-speed vs context-length benchmark (timing-only).

Per length L (fresh process):
  - load model with max_context=L (cache_kv sized L; 100k expected to OOM)
  - SYNTH-FILL attn KV caches with device-side rand (values irrelevant to
    timing; avoids denormals/NaN pathologies; avoids 100ms/token real prefill)
  - warm 8 tokens, then time N=50 rollout tokens via stock model() T=1 path
    (rollout_jit, same as generate(); tokens fed back on-device, no host sync)
Method note: GDN state cost is context-independent; only the 16 full-attn
layers' KV reads scale with L (expected +L*128KB/447GB/s ms/token).
"""
import os, sys, time, json
sys.path.insert(0, "~/tinygrad-src")

L = int(sys.argv[1])
N = int(os.getenv("MEASURE_N", "50"))
MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")  # P1e: env-plumb for alternate quant e2e runs

from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.helpers import GlobalCounters
from tinygrad.device import Device

res = {"L": L}
t0 = time.perf_counter()
try:
    model, kv = Transformer.from_gguf(MODEL, L)
except Exception as e:
    res["error"] = f"load_oom: {type(e).__name__}: {str(e)[:120]}"
    print("RESULT " + json.dumps(res), flush=True)
    sys.exit(0)
res["load_s"] = round(time.perf_counter() - t0, 1)

cfg = model.blk[-1].config
# init + synth-fill state
t_fill = time.perf_counter()
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None:
        ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()  # P2: cast rand to cache dtype (fp16 KV)
Device["NV"].synchronize()
res["fill_s"] = round(time.perf_counter() - t_fill, 1)

pos0 = L - N - 8
t = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
temp = Tensor([0.0])
sp = UOp.variable("start_pos", 0, L - 1)

pos = pos0
out = None
for _ in range(8):  # warmup (captures rollout_jit)
    inp = t[:, pos:pos+1] if out is None else out
    out = model(inp, sp.bind(pos), temp).realize()
    pos += 1
Device["NV"].synchronize()

t1 = time.perf_counter()
for _ in range(N):
    out = model(out, sp.bind(pos), temp).realize()
    pos += 1
Device["NV"].synchronize()
dt = (time.perf_counter() - t1) / N
res["ms_per_tok"] = round(dt * 1e3, 2)
res["tok_s"] = round(1 / dt, 2)
res["vram_gb"] = round(GlobalCounters.mem_used_per_device.get("NV", 0) / 1e9, 2)
_kvdt = next((b.cache_kv.dtype.itemsize for b in model.blk if hasattr(b, "cache_kv")), 4)
res["kv_bytes_per_tok"] = round(L * 131072 * _kvdt / 4)
res["kv_gb"] = round(res["kv_bytes_per_tok"] / 1e9, 2)
print("RESULT " + json.dumps(res), flush=True)
