# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import json, time
import jinja2
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.function import _function
import collections

REPORT=collections.Counter()   # (funcname, dev, shape, dtype) -> count

_orig_call = _function.__call__
def spy_call(self, *args, **kwargs):
    before_params = None
    # run original while recording param creation: easiest is post-hoc via ret uop scan
    ret = _orig_call(self, *args, **kwargs)
    try:
        t = ret[0] if isinstance(ret, tuple) else ret
        uops = list(t.uop.toposort())[:100000]
        for u in uops:
            if u.op.name == "BUFFER" or (hasattr(u.op,'name') and u.op.name=="BUFFER"): pass
    except Exception:
        pass
    return ret

# simpler: hook param_like to catch PYTHON-device params being created during function calls
from tinygrad.uop.ops import UOp as _U
_orig_param_like = _U.param_like
DEPTH=[0]
def spy_param_like(self, i):
    r = _orig_param_like(self, i)
    try:
        if getattr(r.arg, 'device', None) == 'PYTHON':
            REPORT[(self.dtype.name, self.max_numel())] += 1
    except Exception: pass
    return r
_U.param_like = spy_param_like

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)

# count only during decode captures: reset now
REPORT.clear()
tok = SimpleTokenizer.from_gguf_kv(kv)
text="Machine learning is a field of study in artificial intelligence concerned with the development "*12
ids=[0]+tok.encode(text)
gen=model.generate(list(ids), chunk_size=32, temperature=0.0)
for _ in range(6): next(gen)
REPORT.clear()
for _ in range(2): next(gen)

print("\n== PYTHON-device params created per 2 decode tokens ==")
for (dt,n),c in sorted(REPORT.items(), key=lambda kv:-kv[1])[:20]:
    print(f"  x{c:4d}  dtype={dt} numel={n}")
print("DONE", flush=True)
