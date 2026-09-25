# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""T=3 probe kernel histogram (in-model, eager JIT=2 + KERNEL_HIST=1). Decides P2:
megakernel payoff vs 1:1 a3b substitution of the scan pieces."""
import os, sys, time, json
os.environ.setdefault("JIT", "2")
os.environ.update({"MTP_T3_LAZY": "1", "MTP_SEQ_ATTN": "1", "MTP_PROBE_RO": "1", "MTP_A3C_OFF": "1"})
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("HIST_L", "2048"))
N = int(os.getenv("MEASURE_N", "10"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

from tinygrad.llm.model import Transformer, flush_step_states
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
Device["NV"].synchronize()
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

sp = UOp.variable("mtp_sp", 0, L - 4)
dev = Device["NV"]

def fwd3(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for _bi, b in enumerate(model.blk):
        x = b(x, start_pos)
        if (_bi + 1) % 8 == 0:
            x = x.contiguous().realize()
    h = flush_step_states(x.contiguous(), model.blk).contiguous()
    return h, model.output(model.output_norm(h).half())

def drain():
    recs = getattr(dev, "sig_prof_records", [])
    out = []
    for st, en, name, dname, pk in recs:
        try: dt_us = float(en.timestamp) - float(st.timestamp)
        except Exception: continue
        if dt_us > 0: out.append((name, dt_us / 1e3))
    del dev.sig_prof_records[:]
    return out

toks = Tensor([[11, 22, 33]], dtype=dtypes.int32).contiguous().realize()
pos = L - 130
for _ in range(4):  # warm/compile
    h, lg = fwd3(toks, sp.bind(pos)); Tensor.realize(h, lg); pos += 3
Device["NV"].synchronize(); drain()

per, walls = [], []
for i in range(N):
    t1 = time.perf_counter()
    h, lg = fwd3(toks, sp.bind(pos))
    Tensor.realize(h, lg)
    Device["NV"].synchronize()
    walls.append(time.perf_counter() - t1)
    per.append(drain()); pos += 3

agg = {}
for tok in per:
    for name, dt in tok:
        e = agg.setdefault(name, [0, 0.0]); e[0] += 1; e[1] += dt
n = len(per)
wall_ms = sum(walls)/n*1e3
kern_ms = sum(dt for tok in per for _, dt in tok)/n
nkern = sum(e[0] for e in agg.values())/n
print(f"\n[T=3 probe @L={L}] wall={wall_ms:.1f}ms kern_exec={kern_ms:.1f}ms kernels={nkern:.0f}", flush=True)
rows = sorted(((nm, c//n, tm/n) for nm, (c, tm) in agg.items()), key=lambda r: -r[2])
for nm, c, tm in rows[:25]:
    print(f"  {tm*1000/c:8.1f}us x{c:<4} {tm:8.3f}ms  {nm[:110]}", flush=True)
json.dump({"wall_ms": wall_ms, "kern_ms": kern_ms, "nkern": nkern,
           "top": [{"name": nm[:120], "count": c, "total_ms": round(tm,4)} for nm, c, tm in rows[:60]]},
          open("~/tinygrad-metal/kernhist_t3.json", "w"), indent=1)
print("DONE", flush=True)
