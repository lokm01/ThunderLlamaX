# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# W0 fallback discrimination: pages-vs-handles test for the ~50-piece-graph capture wall.
# Captures N distinct TinyJit graph families (distinct shapes -> distinct HCQGraphs),
# each a trivial add kernel over a stable buffer of BP_MIB MiB (big) or tiny KB (small).
# Keeps every jit + buffer alive (mirrors capture holding graph_cache), replays each twice.
# If BIG faults at much lower N than TINY -> budget ~ mapped pages (DMA/pagetable/dext).
# If same N -> per-graph object budget. If neither -> wall couples to real model captures.
import os, sys, gc, time
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

N    = int(os.getenv("BP_N", "150"))
MIB  = float(os.getenv("BP_MIB", "64"))     # per-family stable buffer size in MiB (0 => 256 KiB tiny)
REPL = int(os.getenv("BP_REPL", "2"))

keep_j, keep_b = [], []
t0 = time.time()
try:
  for i in range(N):
    rows = 2048 + 7 * i                       # distinct shape per family -> distinct graph
    if MIB > 0:
      cols = max(1, int(MIB * (1 << 20) / (2 * rows)))
    else:
      cols = max(1, (256 * 1024) // (2 * rows))
    x = Tensor.arange(rows * cols, dtype=dtypes.half).reshape(rows, cols).contiguous().clone()
    keep_b.append(x)
    @TinyJit
    def fam(x=x):
      return (x + 1).realize()
    for _ in range(3): out = fam(x)           # capture passes
    for _ in range(REPL): out = fam(x)        # replays
    keep_j.append(fam)
    del out
    if i % 10 == 0 or i == N - 1:
      print(f"[BP] family {i+1}/{N} rows={rows} cols={cols} buf={(rows*cols*2)>>20}MiB elapsed={time.time()-t0:.1f}s", flush=True)
  print(f"[BP] ALL {N} FAMILIES CLEAN ({time.time()-t0:.1f}s) mode={big if MIB>0 else tiny}", flush=True)
except Exception as e:
  print(f"[BP] FAULT at family {i+1}/{N} elapsed={time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:300]}", flush=True)
  sys.exit(2)
