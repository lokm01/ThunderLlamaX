# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Patch mtp_v3.py: replace eager draft_step with a JIT=2 TinyJit draft (stable-buffer inputs).

Draft was 88ms/step (per-op sync overhead: ~5 realize() round-trips @16ms each over PCIe).
This wraps embed+norms+eh_proj+attn-block+ffn+head in ONE TinyJit replayed graphless (JIT=2,
no second HCQGraph family — the probe stays the only JIT=1 graph). Inputs go through stable
buffers mutated via allocator copyin (the proven TOKBUF pattern from probe()).
Run: python3 patch_draft.py  (idempotent: skips if DRAFT_JIT marker present)
"""
import re

P = "~/tinygrad-metal/mtp_v3.py"
src = open(P).read()
if "DRAFT_JIT" in src:
    print("already patched"); raise SystemExit(0)

# 1) replace draft_step with TinyJit version (keep old as draft_step_eager)
old_ds = '''def draft_step(pe, hm, pos):
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
    hdj = attn_eager(draft.blk, xin, pos)
    with Context(JIT=2):
        lgj = model.output(draft.head_norm(hdj).half()).realize()
    import numpy as np
    return int(np.argmax(lgj.numpy()[0, -1])), hdj[:, -1:, :].contiguous().realize()'''
new_ds = '''def draft_step_eager(pe, hm, pos):
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
    hdj = attn_eager(draft.blk, xin, pos)
    with Context(JIT=2):
        lgj = model.output(draft.head_norm(hdj).half()).realize()
    import numpy as np
    return int(np.argmax(lgj.numpy()[0, -1])), hdj[:, -1:, :].contiguous().realize()

# ---- DRAFT_JIT: one TinyJit for the whole draft step (JIT=2 graphless; no 2nd graph family) ----
def _draft_fwd(tid_t, hm_t, sp):
    pe = model.token_embd(tid_t).float()
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm_t), dim=-1))
    b = draft.blk
    x = xin
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), sp)
    hd = hh + b._feed_forward(b.ffn_norm(hh))
    lg = model.output(draft.head_norm(hd).half())[:, -1:, :]
    return lg, hd[:, -1:, :]
draft_j = TinyJit(_draft_fwd)
_TIDB = [None]; _HMB = [None]
def _dbuf(t): return t.uop.buf_uop.buffer._bufs["NV"]

def draft_step(tid, hm, pos, _warm=False):
    """hm: [1,1,dim] fp32 Tensor (or None for zeros at pos 0). Returns (argmax, hd Tensor)."""
    import numpy as np
    dev = Device["NV"]
    if hm is None:
        hm = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    hm_c = hm.contiguous().realize()
    if _TIDB[0] is None:
        _TIDB[0] = Tensor([[int(tid)]], dtype="int32").contiguous().realize()
        _HMB[0] = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    dev.allocator._copyin(_dbuf(_TIDB[0]), memoryview(np.asarray([[int(tid)]], dtype=np.int32).tobytes()).cast("B"))
    dev.allocator._copyin(_dbuf(_HMB[0]), memoryview(hm_c.numpy().astype(np.float32).tobytes()).cast("B"))
    with Context(JIT=2):
        lg, hd = draft_j(_TIDB[0], _HMB[0], v_sp.bind(pos))
        lg = lg.contiguous().realize()
        hd = hd.contiguous().realize()
    Device["NV"].synchronize()
    return int(np.argmax(lg.numpy()[0, -1])), hd'''
assert old_ds in src, "draft_step body drifted"
src = src.replace(old_ds, new_ds, 1)

# 2) update the three call sites: they pass embed(tid)/hm tensors; new signature takes token id
# cycle: pe, hm = embed(cur), h_seed ; draft_step(pe, hm, pos+j) ; pe, hm = embed(pj), hdj
old_c1 = """    props = []
    pe, hm = embed(cur), h_seed
    for j in range(K):
        pj, hdj = draft_step(pe, hm, pos + j)
        props.append(pj)
        pe, hm = embed(pj), hdj"""
new_c1 = """    props = []
    hm = h_seed
    for j in range(K):
        pj, hdj = draft_step(cur if j == 0 else props[-1], hm, pos + j)
        props.append(pj)
        hm = hdj"""
assert old_c1 in src, "cycle draft loop drifted"
src = src.replace(old_c1, new_c1, 1)

# post-accept draft fill: draft_step(embed(props[j]), hA[:, j:j+1, :].contiguous().realize(), pos+1+j)
old_c2 = """        for j in range(m):
            draft_step(embed(props[j]), hA[:, j:j+1, :].contiguous().realize(), pos + 1 + j)"""
new_c2 = """        for j in range(m):
            draft_step(props[j], hA[:, j:j+1, :].contiguous().realize(), pos + 1 + j)"""
assert old_c2 in src, "post-accept fill drifted"
src = src.replace(old_c2, new_c2, 1)

# prefill draft fill: draft_step(embed(tid), hm, i) with hm = zeros if i==0
old_c3 = """for i, tid in enumerate(ids):
    hm = zeros if i == 0 else hs[i - 1]
    draft_step(embed(tid), hm, i)"""
new_c3 = """for i, tid in enumerate(ids):
    hm = zeros if i == 0 else hs[i - 1]
    if i < 3:
        draft_step_eager(embed(tid), hm, i)   # warm draft KV through the eager path first
    else:
        draft_step(tid, hm, i)"""
assert old_c3 in src, "prefill fill drifted"
src = src.replace(old_c3, new_c3, 1)

# ensure TinyJit import exists
if "from tinygrad.engine.jit import TinyJit" not in src:
    src = src.replace("from tinygrad.tensor import Tensor",
                      "from tinygrad.tensor import Tensor\nfrom tinygrad.engine.jit import TinyJit", 1)

open(P, "w").write(src)
import ast; ast.parse(src)
print("mtp_v3.py patched with DRAFT_JIT, parses OK")
