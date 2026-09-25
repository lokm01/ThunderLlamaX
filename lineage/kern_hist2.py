# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P0 option (a): PROFILE=1 JIT=1 - per-kernel timestamps from INSIDE HCQGraph replay."""
import os, sys, time, json, pickle
os.environ["PROFILE"] = "1"; os.environ["VIZ"] = "0"
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("HIST_L", "2048")); NTOK = int(os.getenv("MEASURE_N", "10"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.device import ProfileGraphEvent

model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape)).realize()
Device["NV"].synchronize()

pos0 = L - NTOK - 8 - 64
t = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
temp = Tensor([0.0]); sp = UOp.variable("start_pos", 0, L - 1)
dev = Device["NV"]
pos = pos0; out = None
walls = []
for i in range(8 + NTOK):
    t1 = time.perf_counter()
    inp = t[:, pos:pos+1] if out is None else out
    out = model(inp, sp.bind(pos), temp).realize()
    dev.synchronize()
    walls.append(time.perf_counter() - t1)
    pos += 1
print(f"[decode] warm wall {sum(walls[:8])/8*1e3:.1f} ms/tok; measured {sum(walls[8:])/NTOK*1e3:.1f} ms/tok", flush=True)

evs = list(dev.profile_events)
graphs = [e for e in evs if isinstance(e, ProfileGraphEvent)]
print(f"[events] total={len(evs)} graphs={len(graphs)}", flush=True)
# use last NTOK graph replays
per_name = {}
n_used = 0
for g in graphs[-NTOK:]:
    n_used += 1
    ts = [float(x) for x in g.sigs]
    for ent in g.ents:
        st, en = ts[ent.st_id], ts[ent.en_id]
        if en <= st: continue
        e = per_name.setdefault(ent.name if isinstance(ent.name,str) else ent.name.display_name, [0, 0.0])
        e[0] += 1; e[1] += (en - st)
rows = sorted(per_name.items(), key=lambda kv: -kv[1][1])
tot = sum(v[1] for _, v in rows)
nk = sum(v[0] for _, v in rows)
print(f"[agg] {n_used} replays, {nk/n_used:.0f} kernels/replay, kernel_sum={tot/n_used/1e3:.2f} ms (ticks assumed us)")
print(f"{'kernel':<60}{'cnt':>6}{'ms/tok':>10}{'us/ea':>9}")
for nm, (c, tm) in rows[:25]:
    print(f"{nm[:59]:<60}{c//n_used:>6}{tm/n_used/1e3:>10.3f}{tm/c:>9.1f}")
json.dump({"kernel_exec_ms_tok_graph": round(tot/n_used/1e3, 2),
           "kernels_per_tok": round(nk/n_used, 1),
           "top": [{"name": nm[:120], "count": c // n_used, "total_ms": round(tm/n_used/1e3, 4), "mean_us": round(tm/c, 1)} for nm, (c, tm) in rows]},
          open("/tmp/kernhist_graph.json", "w"), indent=1)
os._exit(0)  # skip viz dump machinery
