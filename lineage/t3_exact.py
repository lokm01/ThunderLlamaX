# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""T=3 numerics isolation: h3[:, j] must equal T=1 eager hidden after token j.
Env matrix: MTP_T3_LAZY / MTP_SEQ_ATTN / MTP_STEP_STATES — this script prints which are on.
Reference: T=1 graphless forwards at JIT=2 (exact, gated 12/12 every session)."""
import os, sys, time
import numpy as np
sys.path.insert(0, "~/tinygrad-src")
L = int(os.getenv("HIST_L", "512"))
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
print(f"[cfg] T3_LAZY={os.getenv('MTP_T3_LAZY','0')} SEQ_ATTN={os.getenv('MTP_SEQ_ATTN','0')} "
      f"STEP_STATES={os.getenv('MTP_STEP_STATES','0')} KV_CHUNK={os.getenv('MTP_KV_CHUNK','0')}", flush=True)

from tinygrad.llm.model import Transformer
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context

model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk:
    b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()
print("[load] ok", flush=True)

v_sp = UOp.variable("mtp_sp", 0, L - 4)

def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    from tinygrad.llm.model import flush_step_states
    return flush_step_states(x.contiguous(), model.blk).contiguous()

# reference: sequential T=1 graphless. IMPORTANT: snapshot states after T1 x _sp so the
# probe starts from the same baseline the T1 chain had at position _sp (the original
# harness advanced through all 6 tokens first = over-advanced states = bogus comparison).
import random
random.seed(7)
seq = [random.randrange(1000, 50000) for _ in range(6)]
_sp0 = int(os.getenv("SP", "3"))
gdns_all = [b for b in model.blk if hasattr(b, "conv_state")]
snaps = [(b.conv_state.clone().contiguous().realize(), b.recurrent_state.clone().contiguous().realize()) for b in gdns_all]
h_ref = []
with Context(JIT=2):
    for i, tid in enumerate(seq):
        t = Tensor([[tid]], dtype="int32").contiguous()
        h = _fwd(t, v_sp.bind(i)).realize()
        h_ref.append(h.numpy()[0, -1])   # (dim,)
# restore states to after-position-_sp0 (they are currently after 5)
if _sp0 > 0:
    # rebuild by replaying 0.._sp0-1 from scratch
    for b in model.blk:
        if hasattr(b, "conv_state"):
            b.conv_state.assign(Tensor.zeros_like(b.conv_state)).realize()
            b.recurrent_state.assign(Tensor.zeros_like(b.recurrent_state)).realize()
    Device["NV"].synchronize()
    with Context(JIT=2):
        for i in range(_sp0):
            _fwd(Tensor([[seq[i]]], dtype="int32").contiguous(), v_sp.bind(i)).realize()
    Device["NV"].synchronize()

# T=3 probe at sp=3 over [seq[3], seq[4], seq[5]] — h3[:, j] should equal h_ref[3+j]
probe_j = _fwd if os.getenv("PROBE_JIT")=="0" else TinyJit(_fwd)
from tinygrad.llm.model import GatedDeltaNetBlock
gd = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
if os.getenv("STATE_TEST"):
    import numpy as np
    def snap():
        return gd.conv_state.numpy().copy(), gd.recurrent_state.numpy().copy()
    def reset():
        for b in model.blk:
            if isinstance(b, GatedDeltaNetBlock):
                b.conv_state.assign(Tensor.zeros_like(b.conv_state)).realize()
                b.recurrent_state.assign(Tensor.zeros_like(b.recurrent_state)).realize()
        Device["NV"].synchronize()
    # path A: T1 x3
    with Context(JIT=2):
        for i in range(3):
            _fwd(Tensor([[seq[i]]], dtype="int32").contiguous(), v_sp.bind(i)).realize()
    csA, rsA = snap()
    reset()
    # path B: T3 @0
    with Context(JIT=2):
        _fwd(Tensor([seq[0:3]], dtype="int32").contiguous(), v_sp.bind(0)).realize()
    csB, rsB = snap()
    relc = float(np.abs(csA - csB).max() / max(np.abs(csA).max(), 1e-9))
    relr = float(np.abs(rsA - rsB).max() / max(np.abs(rsA).max(), 1e-9))
    print("STATE conv relerr=%.2e  recurrent relerr=%.2e" % (relc, relr), flush=True)
    print("conv SAME" if relc < 0.01 else "conv DIFFERS", "|", "rec SAME" if relr < 0.01 else "rec DIFFERS", flush=True)
    sys.exit(0)
if os.getenv("CHAIN_TEST"):
    t0 = Tensor([seq[0:3]], dtype="int32").contiguous()
    with Context(JIT=2):
        _fwd(t0, v_sp.bind(0)).realize()
    h_ref2 = []
    for i, tid in enumerate(seq[3:6], start=3):
        t = Tensor([[tid]], dtype="int32").contiguous()
        with Context(JIT=2):
            h = _fwd(t, v_sp.bind(i)).realize()
        h_ref2.append(h.numpy()[0, -1])
    for j in range(3):
        r1, r2 = h_ref[3 + j], h_ref2[j]
        rel = float(np.abs(r1 - r2).max() / max(np.abs(r1).max(), 1e-9))
        verdict = "SAME" if rel < 0.02 else "T3-WRITES-BAD-STATES"; print("[chain row %d] relerr=%.2e %s" % (j, rel, verdict), flush=True)
    sys.exit(0)
_sp = int(os.getenv("SP","3"))
toks = Tensor([seq[_sp:_sp+3]], dtype="int32").contiguous()
with Context(JIT=int(os.getenv("PROBE_JIT","1"))):
    h3 = probe_j(toks, v_sp.bind(int(os.getenv("SP","3")))).realize()
Device["NV"].synchronize()
h3_np = h3.numpy()[0]
for j in range(3):
    ref = h_ref[_sp + j]
    got = h3_np[j]
    rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
    print(f"[row {j}] relerr={rel:.2e} {'OK' if rel < 0.02 else 'WRONG'}  "
          f"argmax_head_ref={int(np.argmax(ref))} got_head_absmax_idx={int(np.argmax(np.abs(got)))}", flush=True)
ok = all(float(np.abs(h3_np[j] - h_ref[_sp+j]).max() / max(np.abs(h_ref[3+j]).max(), 1e-9)) < 0.02 for j in range(3))
print("NUMERICS:", "EXACT-OK" if ok else "WRONG", flush=True)
