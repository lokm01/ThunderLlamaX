# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""T=3 kernel histogram: JIT=2 graphless forward under the lazy gate — per-kernel attribution."""
import os, sys, time, collections
sys.path.insert(0, "~/tinygrad-src")
os.environ["JIT"] = "2"
L = int(os.getenv("HIST_L", "2048"))
N = int(os.getenv("MEASURE_N", "20"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device

model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
Device["NV"].synchronize()
print("[load] ok", flush=True)

v_sp = UOp.variable("mtp_sp", 0, L - 4)
pos = L - 3*N - 8
tt = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
temp = Tensor([0.0])
dev = Device["NV"]

def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    h = x.contiguous()
    return model.output(model.output_norm(h[:, :, :]).half())

def drain():
    recs = getattr(dev, "sig_prof_records", [])
    dev.sig_prof_records = []
    return recs

# warmup
for _ in range(3):
    out = _fwd(tt[:, pos:pos+3].contiguous(), v_sp.bind(pos)).realize()
    pos += 3
Device["NV"].synchronize()
drain()

agg = collections.defaultdict(lambda: [0, 0.0])
wall0 = time.perf_counter()
for _ in range(N):
    drain()
    t0 = time.perf_counter()
    out = _fwd(tt[:, pos:pos+3].contiguous(), v_sp.bind(pos)).realize()
    Device["NV"].synchronize()
    dt = time.perf_counter() - t0
    recs = drain()
    for r in recs:
        nm = getattr(r, "name", None) or str(r)
        # records: (name, t_us) pairs or objects; handle tuple form
        try:
            name, tus = r
        except Exception:
            name, tus = nm, 0.0
        agg[name][0] += 1
        agg[name][1] += float(tus) / 1e6 if float(tus) > 1e-3 else float(tus) * 1e-6  # us->s guard
    pos += 3
wall = (time.perf_counter() - wall0) / N
tot = sum(v[1] for v in agg.values())
print(f"== T3 HIST == wall {wall*1e3:.1f} ms/pass, exec-sum {tot*1e3:.1f} ms, kernels {sum(v[0] for v in agg.values())//N}", flush=True)
for name, (cnt, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:24]:
    print(f"{cnt//N:5d}x  {ms/N*1e3:8.2f} ms/pass  {(ms/max(cnt,1))*1e6:8.1f} us/ea  {name}", flush=True)
