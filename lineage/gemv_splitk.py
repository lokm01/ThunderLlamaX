# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""GEMV rescue microbench: can relayout/split-K get the bad shapes to >=300 GB/s?

Shapes from Qwen3.8-27B decode (W stored as Linear: (N,K) row-major, y = x@W.T):
  ssm_out/o_proj: K=6144 N=5120 (18 GB/s baseline)
  qkv_gdn:        K=5120 N=10240 (59)
  q_attn:         K=5120 N=12288
  kv_attn:        K=5120 N=2048
  gate_up:        K=5120 N=34816 (212)
  down:           K=17408 N=5120 (112)
  lm_head:        K=5120 N=248320 (control, expect ~438)

Variants:
  A: plain      y = x(1,K) @ W.T        (Linear layout, baseline)
  B: wt         y = x(1,K) @ Wt(K,N)    (W transposed-contiguous one-time relayout)
  C: splitK-S   y = (xs(S,1,Ks) @ Wt(S,Ks,N)).sum(0)   S in 16/32/64
Timing: 3 warmup + 30 timed, .realize() + device sync each iter.
"""
import os, sys, time
sys.path.insert(0, "~/tinygrad-src")
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv

SHAPES = [
    ("ssm_out",  6144,  5120),
    ("o_proj",   6144,  5120),
    ("qkv_gdn",  5120, 10240),
    ("q_attn",   5120, 12288),
    ("kv_attn",  5120,  2048),
    ("gate_up",  5120, 34816),
    ("down",    17408,  5120),
    ("lm_head",  5120, 248320),
]
ITERS = 30

def bench(fn, nbytes):
    for _ in range(3): fn()
    Device["NV"].synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    Device["NV"].synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    return dt, nbytes / dt / 1e9

print(f"[cfg] BEAM={getenv('BEAM',0)} JIT_BATCH_SIZE={getenv('JIT_BATCH_SIZE',32)}", flush=True)
# replay overhead floor: empty jit
W0 = Tensor.ones(8, 8).half().contiguous().realize()
je = TinyJit(lambda: (W0 @ W0).realize())
je(); je()
Device["NV"].synchronize(); t0 = time.perf_counter()
for _ in range(ITERS): je()
Device["NV"].synchronize()
print(f"[overhead] empty jit replay: {(time.perf_counter()-t0)/ITERS*1e3:.3f} ms/call", flush=True)
results = {}
for name, K, N in SHAPES:
    W  = Tensor.kaiming_uniform(N, K).half().contiguous().realize()   # Linear layout
    Wt = W.permute(1, 0).contiguous().realize()                        # (K,N) relayout
    x  = Tensor.kaiming_uniform(1, K).half().contiguous().realize()
    ref = (x @ W.T).realize()
    nbytes = N * K * 2
    row = {}
    # A: plain
    j = TinyJit(lambda xx: (xx @ W.T).realize())
    j(x.clone()); j(x.clone())
    dt, g = bench(lambda: j(x.clone()), nbytes)
    ok = bool(((j(x.clone()).realize() - ref).abs().cast(dtypes.float32).max().item()) < 0.5)
    row["A"] = (g, ok); print(f"{name:9s} K={K:6d} N={N:6d} A plain      {g:7.1f} GB/s {'OK' if ok else 'BAD'}", flush=True)
    # B: Wt
    j = TinyJit(lambda xx: (xx @ Wt).realize())
    j(x.clone()); j(x.clone())
    dt, g = bench(lambda: j(x.clone()), nbytes)
    ok = bool(((j(x.clone()).realize() - ref).abs().cast(dtypes.float32).max().item()) < 0.5)
    row["WT"] = (g, ok); print(f"{name:9s} {'':14s} B wt         {g:7.1f} GB/s {'OK' if ok else 'BAD'}", flush=True)
    # C: split-K on Wt
    for S in (16, 32, 64):
        if K % S: continue
        Ks = K // S
        Wsk = Wt.reshape(S, Ks, N).contiguous().realize()  # (S,Ks,N) contiguous
        def run(xx, S=S, Ks=Ks, Wsk=Wsk):
            return (xx.reshape(S, 1, Ks) @ Wsk).sum(0).realize()
        try:
            j = TinyJit(run)
            j(x.clone()); j(x.clone())
            dt, g = bench(lambda: j(x.clone()), nbytes)
            ok = bool(((j(x.clone()).realize().flatten() - ref.flatten()).abs().cast(dtypes.float32).max().item()) < 0.5)
        except Exception as e:
            g, ok = 0.0, False
            print(f"  SK{S} fail: {str(e)[:80]}", flush=True)
        row[f"SK{S}"] = (g, ok)
        print(f"{name:9s} {'':14s} C splitK{S:<4d} {g:7.1f} GB/s {'OK' if ok else 'BAD'}", flush=True)
    results[name] = row
    del W, Wt, Wsk, x, ref

print("\n== SUMMARY (GB/s) ==", flush=True)
for name, row in results.items():
    best = max((g, v) for v, (g, ok) in row.items() if ok)
    print(f"{name:9s} best={best[0]:7.1f} ({best[1]})  " + "  ".join(f"{v}={g:.0f}" for v, (g, _) in row.items()), flush=True)
print("DONE_GEMV", flush=True)
