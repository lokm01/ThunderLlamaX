# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P0 kern_hist.py - per-kernel attribution of one steady decode token.
Requires fork patch KERNEL_HIST=1 (hcq.py appends per-kernel HCQ signal-timestamp
records to dev.sig_prof_records even outside PROFILE).
Run: DEV=NV BEAM=1 JIT=2 KERNEL_HIST=1 MEASURE_N=50 ~/tg311/bin/python kern_hist.py
NOTE: JIT=2 (graphless) is REQUIRED so kernels launch individually; kernel names
and device times are identical to the JIT=1 production path (same schedule).
"""
import os, sys, time, json
os.environ.setdefault("JIT", "2")
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("HIST_L", "2048"))
N = int(os.getenv("MEASURE_N", "50"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import GlobalCounters

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, L)
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
Device["NV"].synchronize()

pos0 = L - N - 8 - 64
t = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
temp = Tensor([0.0])
sp = UOp.variable("start_pos", 0, L - 1)
dev = Device["NV"]

def drain():
    recs = getattr(dev, "sig_prof_records", [])
    out = []
    for st, en, name, dname, pk in recs:
        # CALIBRATED on this rig (2026-08-24, diag_sig.py): dext signal timestamps
        # are MICROSECONDS, not ns as hcq.py's /1e6 assumes (4096^2 fp32 matmul:
        # 31159 ticks vs 31306 us wall). Fork built-in PROFILE timing is 1000x off.
        try: dt_us = float(en.timestamp) - float(st.timestamp)
        except Exception: continue
        if dt_us > 0: out.append((name, dt_us / 1e3))  # us -> ms
    del dev.sig_prof_records[:]
    return out

pos = pos0
out = None
for _ in range(8):  # warmup (captures rollout_jit); discard records
    inp = t[:, pos:pos+1] if out is None else out
    out = model(inp, sp.bind(pos), temp).realize()
    pos += 1
Device["NV"].synchronize()
drain()

per_tok = []
walls = []
for i in range(N):
    t1 = time.perf_counter()
    out = model(out, sp.bind(pos), temp).realize()
    Device["NV"].synchronize()
    walls.append(time.perf_counter() - t1)
    per_tok.append(drain())
    pos += 1

agg = {}
ntok_kern = []
for tok in per_tok:
    ntok_kern.append(sum(dt for _, dt in tok))
    for name, dt in tok:
        e = agg.setdefault(name, [0, 0.0])
        e[0] += 1; e[1] += dt

n = len(per_tok)
wall_ms = sum(walls) / n * 1e3
kern_ms = sum(ntok_kern) / n
nkern = sum(e[0] for e in agg.values()) / n
res = {
    "note": "timestamps us-calibrated; wall is graphless JIT=2 (inflated by launch tax); production wall = bench_ctx JIT=1",
    "L": L, "N": n,
    "wall_ms_tok_graphlessJIT2": round(wall_ms, 2),
    "kernel_exec_ms_tok": round(kern_ms, 3),
    "gap_ms_tok": round(wall_ms - kern_ms, 2),
    "kernels_per_tok": round(nkern, 1),
    "kernel_busy_pct_of_wall": round(100 * kern_ms / wall_ms, 1),
}
rows = sorted(((name, c // n, tm / n, (tm / c) * 1e3) for name, (c, tm) in agg.items()),
              key=lambda r: -r[2])
res["top30"] = [{"name": nm[:120], "count": c, "total_ms": round(tm, 4), "mean_us": round(mu, 1)}
                for nm, c, tm, mu in rows[:30]]

# generic grouping heuristics (refined manually from top30 dump)
import re
def grp(name):
    nl = name.lower()
    if re.search(r"gemv|matmul|gemm|_mm|w_.*x.*_", nl): return "GEMV?"
    if re.search(r"rmsnorm|norm|silu|sigmoid|softplus|exp|mul|add|log", nl): return "elementwise/norm?"
    return "other"
groups = {}
for nm, c, tm, mu in rows:
    g = grp(nm)
    e = groups.setdefault(g, {"count": 0, "total_ms": 0.0})
    e["count"] += c; e["total_ms"] = round(e["total_ms"] + tm, 4)
res["groups_auto"] = groups

json.dump(res, open(os.path.expanduser("~/tinygrad-metal/kernhist.json"), "w"), indent=1)
print("== KERN HIST ==")
for k, v in res.items():
    if k not in ("top30", "groups_auto"): print(f"{k}: {v}")
print(f"{'kernel':<70}{'cnt/tok':>8}{'ms/tok':>10}{'us/ea':>9}")
for nm, c, tm, mu in rows[:20]:
    print(f"{nm[:69]:<70}{c:>8}{tm:>10.3f}{mu:>9.1f}")
