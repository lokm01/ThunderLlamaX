# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Minimal 100k probe-fault repro: checkpoint-resumed states, ONE eager T=3 forward.
Fault bisect: sp ladder + knob ladder (STEP_STATES / PROBE_RO / logits).
Env: REPRO_SP=<sp> REPRO_SS=0/1 REPRO_RO=0/1 REPRO_LOGITS=0/1 (defaults: 94208, 1, 1, 1)
"""
import os, sys, time
REPRO_SP = int(os.getenv("REPRO_SP", "94208"))
SS = os.getenv("REPRO_SS", "1") == "1"
RO = os.getenv("REPRO_RO", "1") == "1"
LOGI = os.getenv("REPRO_LOGITS", "1") == "1"
L = 100352
os.environ.update({
    "MTP_T3_LAZY": "1", "MTP_SEQ_ATTN": "1", "MTP_A3C_OFF": "1",
    "MTP_EMB_GATHER": "1", "MTP_HEAD16_DIRECT": "1", "MTP_A3_OVERRIDE": "~/tinygrad-metal/a3b/override.json",
})
if SS: os.environ["MTP_STEP_STATES"] = "1"
if RO: os.environ["MTP_PROBE_RO"] = "1"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")

import numpy as np
from tinygrad.llm.model import Transformer, flush_step_states
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context

MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
for b in model.blk: b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()
print(f"[load] {time.perf_counter()-t0:.1f}s sp={REPRO_SP} SS={SS} RO={RO} LOGITS={LOGI}", flush=True)

# restore checkpoint states (pos 94208 content; sp sweep just changes read extent)
CK = "~/ckpt100k"
for i, b in enumerate(model.blk):
    if getattr(b, "cache_kv", None) is not None:
        b.cache_kv.assign(Tensor(np.load(f"{CK}/kv_{i}.npy")).cast(b.cache_kv.dtype)).realize()
    if hasattr(b, "conv_state"):
        b.conv_state.assign(Tensor(np.load(f"{CK}/cv_{i}.npy")).cast(b.conv_state.dtype)).realize()
        b.recurrent_state.assign(Tensor(np.load(f"{CK}/rc_{i}.npy")).cast(b.recurrent_state.dtype)).realize()
Device["NV"].synchronize()
print("[ckpt] states restored", flush=True)

from tinygrad.llm.gguf import ggml_data_to_tensor as _g2t
v_sp = UOp.variable("start_pos", 0, L - 1)

# ---- REPRO_SLICE=1: build the draft vocab slice EXACTLY like mtp_v3 (the
# counter + gather over the fp16 head) — the untested mtp_v3 delta at the
# SS-stash fault moment. ----
if os.getenv("REPRO_SLICE") == "1":
    import collections as _col, numpy as _nps
    _cnt = _col.Counter([0] + [42] * 512)
    for _b in range(60): _cnt[1000 + _b] += 1
    _slice_ids = sorted(i for i, _ in _cnt.most_common())[:40960]
    _slice_ids_t = Tensor(_nps.asarray(_slice_ids, dtype=_nps.int32), device="NV").contiguous().realize()
    _w16s = model.output.weight.cast(dtypes.float16).contiguous().realize()
    _dh = _w16s[_slice_ids_t].contiguous().realize()
    Device["NV"].synchronize()
    print(f"[slice] N={len(_slice_ids)} gathered OK", flush=True)

# ---- REPRO_DRAFT=1: load the draft module exactly like mtp_v3 (layout shift test;
# no draft FORWARD — only the fp16 weight realizes + dkv-style state init). ----
if os.getenv("REPRO_DRAFT") == "1":
    import struct as _st
    from tinygrad.helpers import prod as _prod
    from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT
    from tinygrad import nn as _nn
    from tinygrad.nn.state import load_state_dict as _lsd
    from tinygrad.llm.model import TransformerBlock as _TB
    def _extract(path, prefix):
        r = open(path, "rb"); assert r.read(4) == b"GGUF"
        _st.unpack("<I", r.read(4)); nt = _st.unpack("<Q", r.read(8))[0]; nk = _st.unpack("<Q", r.read(8))[0]
        align = 32
        def _rs():
            n = _st.unpack("<Q", r.read(8))[0]; return r.read(n).decode()
        def _rv(typ):
            if typ == 8: return _rs()
            if typ == 9:
                it = _st.unpack("<I", r.read(4))[0]; n = _st.unpack("<Q", r.read(8))[0]
                return [_rv(it) for _ in range(n)]
            fmt = {0:"c",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[typ]
            return _st.unpack("<" + fmt, r.read(_st.calcsize("<" + fmt)))[0]
        for _ in range(nk):
            k = _rs(); t = _st.unpack("<I", r.read(4))[0]; v = _rv(t)
            if k == "general.alignment": align = int(v)
        infos = []
        for _ in range(nt):
            nm = _rs(); nd = _st.unpack("<I", r.read(4))[0]
            dims = tuple(_st.unpack("<Q", r.read(8))[0] for _ in range(nd))
            typ = _st.unpack("<I", r.read(4))[0]; off = _st.unpack("<Q", r.read(8))[0]
            infos.append((nm, dims, typ, off))
        data_start = (r.tell()+align-1)//align*align
        out = {}
        for nm, dims, typ, off in infos:
            if not nm.startswith(prefix): continue
            n = _prod(dims)
            if typ in _GGML_NATIVE: nbytes = _GGML_NATIVE[typ].itemsize*n
            else:
                ne, nb = _GGML_QUANT[typ]; nbytes = (n//ne)*nb
            r.seek(data_start+off); raw = r.read(nbytes)
            t8 = Tensor(np.frombuffer(raw, np.uint8).copy())
            out[nm] = _g2t(t8, n, typ).reshape(*reversed(dims))
        r.close(); return out
    _sd = _extract(MODEL, "blk.64")
    class _D: pass
    draft = _D()
    draft.blk = _TB(cfg)
    draft.enorm = _nn.RMSNorm(cfg.dim, cfg.norm_eps)
    draft.hnorm = _nn.RMSNorm(cfg.dim, cfg.norm_eps)
    draft.eh_proj = _nn.Linear(2*cfg.dim, cfg.dim, bias=False)
    draft.head_norm = _nn.RMSNorm(cfg.dim, cfg.norm_eps)
    def _w16(name): return _sd["blk.64." + name].cast(dtypes.float16).contiguous()
    _remap = {
     "blk.attn_norm.weight": _w16("attn_norm.weight"),
     "blk.attn_q.weight": _w16("attn_q.weight"), "blk.attn_k.weight": _w16("attn_k.weight"),
     "blk.attn_v.weight": _w16("attn_v.weight"), "blk.attn_output.weight": _w16("attn_output.weight"),
     "blk.attn_q_norm.weight": _w16("attn_q_norm.weight"), "blk.attn_k_norm.weight": _w16("attn_k_norm.weight"),
     "blk.ffn_norm.weight": _w16("post_attention_norm.weight"),
     "blk.ffn_gate.weight": _w16("ffn_gate.weight"), "blk.ffn_up.weight": _w16("ffn_up.weight"),
     "blk.ffn_down.weight": _w16("ffn_down.weight"),
     "enorm.weight": _sd["blk.64.nextn.enorm.weight"].cast(dtypes.float16).contiguous(),
     "hnorm.weight": _sd["blk.64.nextn.hnorm.weight"].cast(dtypes.float16).contiguous(),
     "head_norm.weight": _sd["blk.64.nextn.shared_head_norm.weight"].cast(dtypes.float16).contiguous(),
     "eh_proj.weight": _sd["blk.64.nextn.eh_proj.weight"].cast(dtypes.float16).contiguous(),
    }
    _lsd(draft, _remap, verbose=False)
    for _p in _remap.values(): _p.realize()
    del _sd, _remap
    draft.blk._init_state(Tensor.zeros(1, 1, cfg.dim))
    # draft KV full-ctx allocation like mtp_v3 (draft.blk cache at maxctx via first call state)
    Device["NV"].synchronize()
    print("[draft] loaded (layout-shift test)", flush=True)

def fwd3_nochk(tokens, start_pos):
    # chunk-mode forward: no logits, no flush_step_states hidden handling beyond stock
    _n = model.emb_out_dim
    x = _g2t(model.emb_rows[tokens.reshape(-1)].cast(dtypes.uint8),
             _n * tokens.numel(), model.emb_ggml_type) \
          .reshape(tokens.shape[0], tokens.shape[1], _n).cast(dtypes.float32)
    for _bi, b in enumerate(model.blk):
        x = b(x, start_pos)
        if (_bi + 1) % 8 == 0:
            x = x.contiguous().realize()
    return x.contiguous()

def fwd3(tokens, start_pos):
    _n = model.emb_out_dim
    x = _g2t(model.emb_rows[tokens.reshape(-1)].cast(dtypes.uint8),
             _n * tokens.numel(), model.emb_ggml_type) \
          .reshape(tokens.shape[0], tokens.shape[1], _n).cast(dtypes.float32)
    for _bi, b in enumerate(model.blk):
        x = b(x, start_pos)
        if (_bi + 1) % 8 == 0:
            x = x.contiguous().realize()
    h = flush_step_states(x.contiguous(), model.blk).contiguous()
    if LOGI:
        lg = model.output(model.output_norm(h).half())
        return h, lg
    return h, None

# ---- REPRO_ADV: advance N T=8 chunks from the checkpoint (the mtp_v3 prefill tail,
# with RO/SS popped like mtp_v3 does), then restore knobs. Fault-bisect step. ----
ADV = int(os.getenv("REPRO_ADV", "0"))
if ADV > 0:
    _ro = os.environ.pop("MTP_PROBE_RO", None)
    _ss = os.environ.pop("MTP_STEP_STATES", None)
    import numpy as _npa
    dev0 = Device["NV"]
    for c0 in range(94208, 94208 + ADV * 8, 8):
        chunk = [4471, 6545, 9956, 2002, 4471, 12253, 29877, 4471]
        toks_c = Tensor.zeros(1, 8, dtype=dtypes.int32).contiguous().realize()
        dev0.allocator._copyin(toks_c.uop.buf_uop.buffer._bufs["NV"],
                               memoryview(_npa.asarray([chunk]*1, dtype=_npa.int32).tobytes()).cast("B"))
        with Context(JIT=2):
            hl = fwd3_nochk(toks_c, v_sp.bind(c0)).realize()
        if (c0 - 94208) % 64 == 0: print(f"[adv] chunk@{c0}", flush=True)
    Device["NV"].synchronize()
    if _ro is not None: os.environ["MTP_PROBE_RO"] = _ro
    if _ss is not None: os.environ["MTP_STEP_STATES"] = _ss
    dev0.allocator.free_cache(); __import__("gc").collect()
    dev0.allocator.free_cache(); __import__("gc").collect()
    print(f"[adv] {ADV} chunks advanced + drained", flush=True)

REPRO_TLEN = int(os.getenv("REPRO_TLEN", "3"))
_toks_map = {2: [4471, 6545], 3: [4471, 6545, 9956], 4: [4471, 6545, 9956, 2002]}
toks = Tensor([_toks_map.get(REPRO_TLEN, _toks_map[3])], dtype=dtypes.int32).contiguous().realize()

if os.getenv("REPRO_HEADROWS") == "1":
    t4 = time.perf_counter()
    _hh = Tensor(np.random.RandomState(0).normal(0, 0.5, (1, 1, cfg.dim)).astype(np.float32))
    _hh = Tensor(_hh.numpy()).contiguous().realize()
    with Context(JIT=2):
        _lg = model.output(model.output_norm(_hh).half()).realize()
    _am = int(np.argmax(_lg.numpy()[0, 0]))
    Device["NV"].synchronize()
    print(f"[headrows] CLEAN argmax={_am} ({time.perf_counter()-t4:.1f}s)", flush=True)

# ---- REPRO_DRAFTSTEP=1: run mtp_v3's draft_step (eager T=1 over the 97k draft KV)
# right before the probe capture — the async-fault-attribution suspect. ----
if os.getenv("REPRO_DRAFTSTEP") == "1" and os.getenv("REPRO_DRAFT") == "1":
    def _draft_fwd(tid_t, hm_t, sp):
        pe = model.token_embd(tid_t).float()
        xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm_t), dim=-1))
        b = draft.blk
        x = xin; b._init_state(x)
        hh = x + b._attention(b.attn_norm(x), sp)
        hd = hh + b._feed_forward(b.ffn_norm(hh))
        lg = model.output(draft.head_norm(hd).half())[:, -1:, :]
        return lg, hd[:, -1:, :]
    if os.getenv("REPRO_TINYDRAFT") == "1":
        from tinygrad.engine.jit import TinyJit as _TJ
        dj = _TJ(_draft_fwd)
    _hm = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    _tidb = Tensor([[4471]], dtype=dtypes.int32).contiguous().realize()
    import numpy as _npd
    for j, tid in enumerate([4471, 6545]):
        t3 = time.perf_counter()
        if os.getenv("REPRO_TINYDRAFT") == "1":
            dev0 = Device["NV"]
            dev0.allocator._copyin(_tidb.uop.buf_uop.buffer._bufs["NV"],
                                   memoryview(_npd.asarray([[tid]], dtype=_npd.int32).tobytes()).cast("B"))
            with Context(JIT=2):
                lgj, hdj = dj(_tidb, _hm, v_sp.bind(REPRO_SP + j))
                hdj = hdj.contiguous().realize()
            am = int(np.argmax(lgj.contiguous().realize().numpy()[0, -1]))
            hd = hdj
        else:
            pe = model.token_embd(Tensor([[tid]], dtype=dtypes.int32)).float().contiguous().realize()
            xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(_hm), dim=-1))
            b = draft.blk
            x = xin; b._init_state(x)
            with Context(JIT=2):
                hh = x + b._attention(b.attn_norm(x), v_sp.bind(REPRO_SP + j))
                hd = (hh + b._feed_forward(b.ffn_norm(hh))).contiguous().realize()
            am = -1
        _hm = hd[:, -1:, :].contiguous().realize()
        Device["NV"].synchronize()
        print(f"[dstep] {j} CLEAN am={am} ({time.perf_counter()-t3:.1f}s)", flush=True)

USE_JIT = os.getenv("REPRO_JIT", "0") == "1"
print(f"[fwd] starting T=3 forward (JIT={1 if USE_JIT else 2})...", flush=True)
t1 = time.perf_counter()
if not USE_JIT:
    with Context(JIT=2):
        h, lg = fwd3(toks, v_sp.bind(REPRO_SP))
        h = h.contiguous().realize()
        if lg is not None: lg = lg.contiguous().realize()
    Device["NV"].synchronize()
    print(f"[fwd] CLEAN-eager ({time.perf_counter()-t1:.1f}s) h[0,0,:4]={h.numpy()[0,0,:4]}", flush=True)
else:
    from tinygrad.engine.jit import TinyJit
    pj = TinyJit(fwd3)
    # call 0: eager (cnt=0). call 1: CAPTURE — the fault site.
    for i, spv in enumerate([REPRO_SP, REPRO_SP + 3, REPRO_SP + 6]):
        t2 = time.perf_counter()
        with Context(JIT=1):
            r = pj(toks, v_sp.bind(spv))
            hh = r[0].contiguous().realize()
            if LOGI: _lgt = r[1].contiguous().realize()
        Device["NV"].synchronize()
        print(f"[jit call {i}] sp={spv} CLEAN ({time.perf_counter()-t2:.1f}s)", flush=True)
    lg = None
print("DONE", flush=True)
