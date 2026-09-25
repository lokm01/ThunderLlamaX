# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""One-shot mapping decoder: rows 0-127 qs-byte low nibble, 128-255 high nibble,
256-511 qh byte/bit toggles (with qs byte0=1), 512-523 scale-byte activation.
out[r] encodes the paired element index k (or bit membership)."""
import os, sys
os.environ["MTP_A3_OVERRIDE"] = "~/tinygrad-metal/a3b/override.json"
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
import tinygrad.codegen.a3b as a3b

def probe_validate(renderer, orig_prg, new_prg, roles, sig_args, new_dims):
    from tinygrad.device import Device, BufferSpec
    dev = Device[renderer.target.device]
    N = roles["N"]
    Wb = np.zeros((N, 3520), dtype=np.uint8)
    W3 = Wb.reshape(N, 20, 176)
    W3[:, :, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)      # d = 1.0
    for r in range(0, 128):
        W3[r, :, 48 + r] = 1                                    # qs byte r, low nibble 1
    for r in range(128, 256):
        W3[r, :, 48 + (r - 128)] = 0x20                         # qs byte, high nibble 2
    for r in range(256, 512):
        b, m = (r - 256) >> 3, (r - 256) & 7
        W3[r, :, 48] = 1                                        # element k=0's byte (per my qs map)
        W3[r, :, 16 + b] |= (1 << m)
    for r in range(512, 524):
        W3[r, :, 48:] = 1                                       # all qs low = 1
        W3[r, :, 4:16] = 0
        W3[r, :, 4 + (r - 512)] = 1                             # only scale byte i active
    x = np.arange(1, 5121, dtype=np.float32) / 512.0
    wdata = {}
    wdata[roles["W"][1]] = Wb.tobytes()
    wdata[roles["x"][1]] = x.tobytes()
    wdata[roles["nw"][1]] = np.ones((5120,), dtype=np.float32).tobytes()
    wdata[roles["s"][1]] = np.array([1.0], dtype=np.float32).tobytes()
    wdata[roles["inn"][1]] = np.zeros((3 * N,), dtype=np.float32).tobytes()
    bufs, allocs, yidx = [], [], None
    oname = roles["out"][1]
    for ctype, name, numel in sig_args:
        if ctype == "const int": continue
        nbytes = numel * (2 if ctype == "half*" else 4)
        b_ = dev.allocator.alloc(nbytes, BufferSpec())
        if name == oname: yidx = len(bufs); data = memoryview(bytearray(nbytes)).cast("B")
        elif name in wdata: data = memoryview(wdata[name]).cast("B")
        else: data = memoryview(np.zeros(max(numel, 1), dtype=np.float32).tobytes()).cast("B")
        dev.allocator._copyin(b_, data)
        bufs.append(b_); allocs.append((b_, nbytes))
    rt_orig = dev.runtime(orig_prg.to_elf())
    og, ol = orig_prg.arg.launch_dims({})
    rt_orig(*bufs, global_size=tuple(og), local_size=tuple(ol), vals=(7,), wait=True)
    mv = memoryview(bytearray(4 * roles["out"][2])); dev.allocator._copyout(mv, bufs[yidx])
    y = np.frombuffer(mv, dtype=np.float32)
    for b_, nb in allocs:
        try: dev.allocator.free(b_, nb, BufferSpec())
        except Exception: pass
    res = {"qs_low": {}, "qs_high": {}, "qh": {}, "sc": {}}
    for r in range(128):
        res["qs_low"][r] = round(float(y[3 * N + r]) * 512, 3)      # = (k+1) -> k
    for r in range(128, 256):
        res["qs_high"][r - 128] = round(float(y[3 * N + r]) * 512 / 2, 3)  # = (k+1)
    for r in range(256, 512):
        b, m = (r - 256) >> 3, (r - 256) & 7
        v = float(y[3 * N + r]) * 512
        res["qh"][(b, m)] = round(v, 3)                              # 1 -> no bit; 17 -> bit at k=0
    for r in range(512, 524):
        res["sc"][r - 512] = round(float(y[3 * N + r]), 3)
    print("MAP " + repr(res), flush=True)
    return True, 0.0

a3b._validate_q5 = probe_validate
os.environ.setdefault("JIT", "2")
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
L = 1024
model, kv = Transformer.from_gguf("~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf", L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
    ck = getattr(b, "cache_kv", None)
    if ck is not None: ck.assign(Tensor.rand(ck.shape).cast(ck.dtype)).realize()
pos = L - 12
t = Tensor.zeros(1, L, dtype=dtypes.int32).contiguous().realize()
temp = Tensor([0.0])
sp = UOp.variable("start_pos", 0, L - 1)
out = None
for i in range(3):
    inp = t[:, pos:pos+1] if out is None else out
    out = model(inp, sp.bind(pos), temp).realize()
    pos += 1
Device["NV"].synchronize()
