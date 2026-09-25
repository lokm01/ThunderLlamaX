# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""MTP_HEAD_INT8: g128-int8 lm_head quantized STREAMING from the lazy Q5 dequant.
Shared by mtp_v3.py and gen_basex.py (baseline must regenerate under the same head).
No fp16 head tensor is ever materialized (bypasses the 2.54GB free-leak; saves 1.27GB).
"""
import os, time
import numpy as np

class _ShapeShim:
    def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype

class Int8Head:
    def __init__(self, q, s, dim, vocab):
        self.q, self.s, self.dim, self.vocab = q, s, dim, vocab
        self.weight = _ShapeShim((vocab, dim), q.dtype)
    def __call__(self, x):   # x: (..., dim) half -> (..., vocab) half
        from tinygrad import dtypes
        outs = []
        B = 16384
        for r0 in range(0, self.vocab, B):
            r1 = min(r0 + B, self.vocab)
            qb = self.q[r0:r1].cast(dtypes.float16).reshape(-1, 40, 128)
            sb = self.s[r0:r1].reshape(-1, 40, 1)
            wb = (qb * sb).reshape(-1, self.dim)
            outs.append(x.matmul(wb.T))
        return outs[0].cat(*outs[1:], dim=-1) if len(outs) > 1 else outs[0]

def install(model, log=print):
    """model: the Transformer (with lazy-Q5 output.weight). Returns the Int8Head."""
    import os as _os
    assert _os.getenv("MTP_HEAD_INT8"), "install() only under MTP_HEAD_INT8=1"
    from tinygrad import dtypes
    from tinygrad.tensor import Tensor
    from tinygrad.device import Device
    t0 = time.perf_counter()
    Wl = model.output.weight
    V, D = Wl.shape[0], Wl.shape[1]
    dev = Device["NV"]
    _cq, _cs = "~/tinygrad-metal/head_i8_q.npy", "~/tinygrad-metal/head_i8_s.npy"
    import os.path as _p
    if _p.exists(_cq) and _p.exists(_cs):
        from tinygrad.tensor import Tensor as _T
        q_buf = _T(np.load(_cq), device="NV").contiguous().realize()
        s_buf = _T(np.load(_cs), device="NV").contiguous().realize()
        dev.synchronize()
        head = Int8Head(q_buf, s_buf, D, V)
        model.output = head
        log(f"[head-i8] loaded from disk cache ({time.perf_counter()-t0:.1f}s)")
        return head
    q_buf = Tensor.zeros(V, D, dtype=dtypes.int8, device="NV").contiguous().realize()
    s_buf = Tensor.zeros(V, D // 128, dtype=dtypes.float16, device="NV").contiguous().realize()
    CH = 4096
    for r0 in range(0, V, CH):
        r1 = min(r0 + CH, V)
        wf = Wl[r0:r1].cast(dtypes.float32).contiguous().realize().numpy()
        w4 = wf.reshape(-1, 128)
        mx = np.abs(w4).max(axis=1, keepdims=True)
        sc = np.maximum(mx / 127.0, 1e-12).astype(np.float16)
        qi = np.clip(np.rint(w4 / sc.astype(np.float32)), -127, 127).astype(np.int8)
        dev.allocator._copyin(q_buf[r0:r1].contiguous().realize().uop.buf_uop.buffer._bufs["NV"],
                              memoryview(qi.tobytes()).cast("B"))
        dev.allocator._copyin(s_buf[r0:r1].contiguous().realize().uop.buf_uop.buffer._bufs["NV"],
                              memoryview(sc.reshape(r1 - r0, -1).tobytes()).cast("B"))
    dev.synchronize()
    # disk-cache the quantized tensors (the streaming quantize costs ~18min/process)
    qn = np.zeros((V, D), dtype=np.int8); sn = np.zeros((V, D // 128), dtype=np.float16)
    _o = 0
    for r0 in range(0, V, CH):
        r1 = min(r0 + CH, V)
        wf2 = q_buf[r0:r1].contiguous().realize().numpy()
        qn[r0:r1] = wf2
        sn[r0:r1] = s_buf[r0:r1].contiguous().realize().numpy()
    np.save(_cq, qn); np.save(_cs, sn)
    head = Int8Head(q_buf, s_buf, D, V)
    model.output = head
    log(f"[head-i8] quantized {V}x{D} streaming + cached ({time.perf_counter()-t0:.1f}s)")
    return head
