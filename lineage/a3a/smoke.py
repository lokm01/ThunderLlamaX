# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
import numpy as np
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_nv import NVProgram

cub = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke.cubin"), "rb").read()
dev = Device["NV"]

def mkbuf(data, spec=None):
  b = dev.allocator.alloc(len(data), spec or BufferSpec())
  dev.allocator._copyin(b, memoryview(data).cast("B"))
  return b

# signature: 2 buf placeholders + int32 n + int32 c
sig = [(None,0,dtypes.uint8,())]*2 + [(None,0,dtypes.int32,()),(None,0,dtypes.int32,())]
prg = NVProgram(dev, TinyELF(lib=cub, name="smoke", target=None, signature=tuple(sig)))

def launch(xb, yb, n, c, gs, ls):
  # nvcc SASS reads NTID from c[0][0..8]; the fork's cbuf_0 leaves it zero
  prg.cbuf_0[0], prg.cbuf_0[1], prg.cbuf_0[2] = ls
  prg(xb, yb, global_size=gs, local_size=ls, vals=(n,c), wait=True)

n, c = 4096, 7
x = np.arange(n, dtype=np.float32)
xb = mkbuf(np.ascontiguousarray(x).tobytes())
yb = dev.allocator.alloc(4*n, BufferSpec())
for _ in range(3):
  launch(xb, yb, n, c, (n//256,1,1), (256,1,1))
mv = memoryview(bytearray(4*n))
dev.allocator._copyout(mv, yb)
y = np.frombuffer(mv, dtype=np.float32)
ok = np.allclose(y, x+c)
print("[smoke] correct:", ok, "maxerr:", float(np.abs(y-(x+c)).max()))
assert ok

INNER, RUNS = 200, 5
best = 1e9
for r in range(RUNS):
  dev.synchronize()
  q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1)
  for _ in range(INNER):
    args = prg.fill_kernargs((xb, yb), (n, c))
    q.exec(prg, args, (n//256,1,1), (256,1,1))
  t0 = time.perf_counter()
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()
  best = min(best, (time.perf_counter()-t0)/INNER)
print(f"[smoke] batched per-launch: {best*1e6:.1f} us (incl QMD chain)")
print("SMOKE OK")
