# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""MTP v3 driver: ONE JIT=1 T=K+1 probe graph + eager draft + step-state select.

Env: MTP_T3_LAZY=1 MTP_SEQ_ATTN=1 MTP_STEP_STATES=1 (+ A3 override).
DO NOT set MTP_KV_CHUNK (chunked split-K attention = known-wrong at T>1; every run with it
produced garbage amd — 2026-08-29 six-hour root chase, see PERFLOG).
Gates: greedy 60/60 vs spec_base.json (same contract as v223/v231).

Architecture (vs v223): no commit re-forward — the probe graph exposes per-step GDN
states (step_rec_buf/step_conv_buf); partial accept selects [m] via eager assign between
replays (the proven cache_kv-fill pattern). Probe = the ONLY JIT=1 graph family.
"""
import os, sys, time, json
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")

from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad import nn, dtypes
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context, getenv
from mtp_config import MTPConfig

CFG = MTPConfig.load()
K = int(getenv("MTP_K3", 0)) or CFG.K
NTOK = int(getenv("NTOK", "60"))
MODEL = os.getenv("MTP_MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(MODEL, int(os.getenv("MTP_MAXCTX", "0")) or CFG.max_context)
cfg = model.blk[-1].config
from tinygrad.llm.cli import SimpleTokenizer
tok = SimpleTokenizer.from_gguf_kv(kv)
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

# ---- draft (ported from v223; extract_tensors inlined — do NOT import mtp_spec,
#      it loads a second model = +12.6GB VRAM) ----
import struct
import numpy as _np
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_NATIVE, _GGML_QUANT
def _rstr(r):
    n = struct.unpack("<Q", r.read(8))[0]; return r.read(n).decode()
def _rd_val(r, typ):
    if typ == 8: return _rstr(r)
    if typ == 9:
        it = struct.unpack("<I", r.read(4))[0]; n = struct.unpack("<Q", r.read(8))[0]
        return [_rd_val(r, it) for _ in range(n)]
    fmt = {0:"c",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[typ]
    return struct.unpack("<" + fmt, r.read(struct.calcsize("<" + fmt)))[0]
def extract_tensors(path, prefix):
    r = open(path, "rb"); assert r.read(4) == b"GGUF"
    struct.unpack("<I", r.read(4)); nt = struct.unpack("<Q", r.read(8))[0]; nk = struct.unpack("<Q", r.read(8))[0]
    align = 32
    for _ in range(nk):
        k = _rstr(r); t = struct.unpack("<I", r.read(4))[0]; v = _rd_val(r, t)
        if k == "general.alignment": align = int(v)
    infos = []
    for _ in range(nt):
        nm = _rstr(r); nd = struct.unpack("<I", r.read(4))[0]
        dims = tuple(struct.unpack("<Q", r.read(8))[0] for _ in range(nd))
        typ = struct.unpack("<I", r.read(4))[0]; off = struct.unpack("<Q", r.read(8))[0]
        infos.append((nm, dims, typ, off))
    data_start = (r.tell()+align-1)//align*align
    out = {}
    for nm, dims, typ, off in infos:
        if not nm.startswith(prefix): continue
        n = prod(dims)
        if typ in _GGML_NATIVE: nbytes = _GGML_NATIVE[typ].itemsize*n
        else:
            ne, nb = _GGML_QUANT[typ]; nbytes = (n//ne)*nb
        r.seek(data_start+off); raw = r.read(nbytes)
        t8 = Tensor(_np.frombuffer(raw, _np.uint8).copy())
        out[nm] = ggml_data_to_tensor(t8, n, typ).reshape(*reversed(dims))
    r.close()
    return out
sd = extract_tensors(MODEL, "blk.64")
class Draft: pass
draft = Draft()
draft.blk = model.blk[0].__class__(cfg) if False else None
# build a full-attn TransformerBlock for the draft (blk.64 is a standard attn block)
from tinygrad.llm.model import TransformerBlock
draft.blk = TransformerBlock(cfg)
draft.enorm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
draft.hnorm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
draft.eh_proj = nn.Linear(2*cfg.dim, cfg.dim, bias=False)
draft.head_norm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
def w16(name): return sd["blk.64." + name].cast(dtypes.float16).contiguous()
remap = {
 "blk.attn_norm.weight": w16("attn_norm.weight"),
 "blk.attn_q.weight": w16("attn_q.weight"), "blk.attn_k.weight": w16("attn_k.weight"),
 "blk.attn_v.weight": w16("attn_v.weight"), "blk.attn_output.weight": w16("attn_output.weight"),
 "blk.attn_q_norm.weight": w16("attn_q_norm.weight"), "blk.attn_k_norm.weight": w16("attn_k_norm.weight"),
 "blk.ffn_norm.weight": w16("post_attention_norm.weight"),
 "blk.ffn_gate.weight": w16("ffn_gate.weight"), "blk.ffn_up.weight": w16("ffn_up.weight"),
 "blk.ffn_down.weight": w16("ffn_down.weight"),
 "enorm.weight": sd["blk.64.nextn.enorm.weight"].cast(dtypes.float16).contiguous(),
 "hnorm.weight": sd["blk.64.nextn.hnorm.weight"].cast(dtypes.float16).contiguous(),
 "head_norm.weight": sd["blk.64.nextn.shared_head_norm.weight"].cast(dtypes.float16).contiguous(),
 "eh_proj.weight": sd["blk.64.nextn.eh_proj.weight"].cast(dtypes.float16).contiguous(),
}
load_state_dict(draft, remap, verbose=False)
for p in remap.values(): p.realize()
del sd, remap
print("[draft loaded]", flush=True)

gdns = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
v_sp = UOp.variable("mtp_sp", 0, (int(os.getenv("MTP_MAXCTX", "0")) or CFG.max_context) - 2 - K)

_BLKG = int(os.getenv("MTP_BLKGROUP", "0"))   # >0: realize x every G blocks during
# eager chunk forwards -- the whole-64-block single schedule stages ALL weights
# concurrently (5.09GB arena at 8k -> capacity faults). Group-realize caps it.

def _fwd3(tokens, start_pos, want_logits=False):
    # MTP_EMB_GATHER: mtp_v3 bypasses Transformer.forward, so the gather fix must be
    # applied HERE too (the raw-table path; identical math, KBs not the 5.09GB arena).
    if os.getenv("MTP_EMB_GATHER") and getattr(model, "emb_rows", None) is not None:
        from tinygrad.llm.gguf import ggml_data_to_tensor as _g2t
        _n = model.emb_out_dim
        x = _g2t(model.emb_rows[tokens.reshape(-1)].cast(dtypes.uint8),
                 _n * tokens.numel(), model.emb_ggml_type) \
              .reshape(tokens.shape[0], tokens.shape[1], _n).cast(dtypes.float32)
    else:
        x = model.token_embd(tokens).float()
    for _bi, b in enumerate(model.blk):
        x = b(x, start_pos)
        if _BLKG and (_bi + 1) % _BLKG == 0:
            x = x.contiguous().realize()
    from tinygrad.llm.model import flush_step_states
    h = flush_step_states(x.contiguous(), model.blk).contiguous()   # (1, T, dim) trunk hiddens
    # NOTE: in-graph .argmax(-1) over vocab FUSES with the lm_head lazy-dequant; the fused
    # reduce adopts the RAW GGUF byte view as buffer -> 5.09GB f32 bitcast -> OOM/fault.
    # Return logits; argmax host-side on the realized tensor.
    if want_logits:
        lg = model.output(model.output_norm(h).half())          # (1,T,V) fp16, plain tensor
        if os.getenv("MTP_PROBE_AM"):
            # tie-safe in-graph argmax over the PLAIN fp16 head output (safe: not the
            # lazy-dequant that caused the 5.09GB fusion; max/where/arange only).
            mx = lg.max(-1, keepdim=True)
            am = ((lg == mx) * Tensor.arange(lg.shape[-1]).reshape(1, 1, -1)).max(-1).cast(dtypes.int32)
            return h, lg, am
        return h, lg
    return h

def _fwd1(tokens, start_pos):
    return _fwd3(tokens, start_pos)

probe_j = TinyJit(_fwd3)
_TOKBUF = [None]
_SPLIT_FWD = None
_SPLIT_HEAD_J = None
def probe(toks, pos):
    # stable input BUFFER: t.assign() swaps the uop/buffer -> jit re-capture every cycle.
    # Write tokens straight into the captured buffer via allocator copyin instead.
    import numpy as _np
    dev = Device["NV"]
    if _TOKBUF[0] is None:
        _TOKBUF[0] = Tensor([toks], dtype="int32").contiguous().realize()
    else:
        b = _TOKBUF[0].uop.buf_uop.buffer._bufs["NV"]
        dev.allocator._copyin(b, memoryview(_np.asarray([toks], dtype=_np.int32).tobytes()).cast("B"))
    if os.getenv("MTP_SCAN_SPLIT"):
        # the split-graph path: the fragment chain (fwd3_split) replaces probe_j.
        global _SPLIT_FWD
        if _SPLIT_FWD is None:
            import fwd3_split as _fs
            _fs.init(model, T=K + 1)
            _SPLIT_FWD = _fs
        # FIX-1: the eager attn fragments append T rows to cache_kv with no
        # graph-driven rollback — snapshot ONLY the written window [pos, pos+K+1)
        # (the full-clone was 392MB/layer at 100k = the transition OOM); the
        # cycle code restores it every cycle (the commit re-forward re-appends).
        global _KV_SNAP
        _KV_SNAP = []
        _KV_SNAP_WIN[0] = pos
        _KV_SNAP_WIN[1] = pos + K + 1
        for b in model.blk:
            kv = getattr(b, "cache_kv", None)
            if kv is None or not hasattr(kv, "shape"): _KV_SNAP.append(None); continue
            _KV_SNAP.append(kv[:, :, pos:pos + K + 1, :].clone().realize())
        print(f"[probe-split call pos={pos}]", flush=True)
        h = _SPLIT_FWD.fwd3_split(model, _TOKBUF[0], pos, v_sp=v_sp).contiguous().realize()
        print("[split-stage] fwd done", flush=True)
        # fold norm+head into a captured TinyJit (the eager output_norm on the
        # split h faults — the eager-arena class; capture makes it graph-safe)
        global _SPLIT_HEAD_J
        if _SPLIT_HEAD_J is None:
            from tinygrad.engine.jit import TinyJit as _TJ
            @_TJ
            def _head_fn(hh: Tensor):
                return model.output(model.output_norm(hh.half()))
            _SPLIT_HEAD_J = _head_fn
        lg = _SPLIT_HEAD_J(h).contiguous().realize()
        print("[split-stage] head done", flush=True)
        Device["NV"].synchronize()
        print("[split-stage] synced post-head", flush=True)
        if os.getenv("MTP_PROBE_AM"):
            mx = lg.max(-1, keepdim=True)
            am = ((lg == mx) * Tensor.arange(lg.shape[-1]).reshape(1, 1, -1)).max(-1).cast(dtypes.int32).contiguous().realize()
            print("[split-stage] argmax done", flush=True)
            Device["NV"].synchronize()
            print("[split-stage] synced post-am", flush=True)
            return (h, am)
        lg.contiguous().realize()
        return (h, lg)
    print(f"[probe call cnt={probe_j.cnt} pos={pos}]", flush=True)
    _am = None
    with Context(JIT=2 if os.getenv("MTP_PROBE_J2") else 1):
        ret = probe_j(_TOKBUF[0], v_sp.bind(pos), True)
        h = ret[0].contiguous().realize()
        if os.getenv("MTP_PROBE_AM"):
            _am = ret[2].contiguous().realize()
        else:
            ret[1].contiguous().realize()
    return (h, _am) if _am is not None else (h, ret[1])

_KV_SNAP_WIN = [0, 0]
def kv_restore():
    """FIX-1: restore the attn-layer KV WINDOW snapshots taken before a split
    probe (every cycle: the commit re-forward re-appends the authoritative rows)."""
    global _KV_SNAP
    if os.getenv("MTP_SCAN_SPLIT") and globals().get("_KV_SNAP") is not None:
        for b, kv in zip(model.blk, _KV_SNAP):
            if kv is not None:
                b.cache_kv[:, :, _KV_SNAP_WIN[0]:_KV_SNAP_WIN[1], :].assign(kv).realize()

def fwd1_eager(tid, pos):
    t = Tensor([[int(tid)]], dtype="int32").contiguous()
    with Context(JIT=2):
        h = _fwd1(t, v_sp.bind(pos)).realize()
    return h

# MTP_COMMIT_JIT: the commit re-forward as its OWN early-captured JIT=1 T=1 family
# (NOBIND removed the one-graph-family cap). Replaces ~2-17s/cyc eager re-forwards
# with ~0.1-0.3s graph replays at 100k.
_CMTBUF = [None]
# MTP_HEAD_J: lazy-head logits as their own early-captured jit (lets the fp16
# head be DROPPED before the early captures — freeing 2.54GB for the SS-stash
# capture arena at 100k; the graph binds the raw-Q5 head with in-graph dequant).
_head_j = [None]
_HJBUF = [None]
def _headj_fwd(h_t):
    return model.output(model.output_norm(h_t).half())
def head_logits(h):
    """h: (1,T,dim) fp32 contiguous tensor -> logits (1,T,V) fp16 (graph replay)."""
    if _head_j[0] is None:
        _HJBUF[0] = h.contiguous().realize()
        _head_j[0] = TinyJit(_headj_fwd)
        for _ in range(2):
            with Context(JIT=1):
                r = _head_j[0](_HJBUF[0]).contiguous().realize()
    import numpy as _nphj
    dev = Device["NV"]
    _hh = h.contiguous().realize()
    dev.allocator._copyin(_HJBUF[0].uop.buf_uop.buffer._bufs["NV"],
                          memoryview(_hh.numpy().astype(_nphj.float32).tobytes()).cast("B"))
    with Context(JIT=1):
        return _head_j[0](_HJBUF[0]).contiguous().realize()

_CMTBUF2 = None
# MTP_COMMIT2: T=2 commit family — m==1 partial accepts commit [cur,p1] in ONE
# replay instead of two T=1 replays.
_T2BUF = [None]
if os.getenv("MTP_COMMIT2"):
    def _cmt2_fwd(tid_t, sp): return _fwd1(tid_t, sp)
    commit2_j = TinyJit(_cmt2_fwd)
    def commit2_step(toks, pos):
        import numpy as _npc3
        dev = Device["NV"]
        if _T2BUF[0] is None:
            _T2BUF[0] = Tensor([[int(t) for t in toks]], dtype="int32").contiguous().realize()
        dev.allocator._copyin(_T2BUF[0].uop.buf_uop.buffer._bufs["NV"],
                              memoryview(_npc3.asarray([[int(t) for t in toks]], dtype=_npc3.int32).tobytes()).cast("B"))
        with Context(JIT=1):
            h = commit2_j(_T2BUF[0], v_sp.bind(pos))
            h = h.contiguous().realize()
        return h
# MTP_DTAIL_JIT: the draft's eager slice-head tail (norm + (N,5120)@(5120,1) GEMV +
# realize + numpy, ~25-30ms of launch tax per step) as its own JIT=1 family.
_dtail_j = [None]
_HNBUF = [None]
def _dtail_fwd(hd_t):
    hn = draft.head_norm(hd_t).half()                      # (1,1,dim)
    lg_s = (_draft_head_sliced @ hn.reshape(-1, 1)).T.reshape(1, 1, -1)  # (1,1,Nslice)
    # tie-safe arithmetic argmax (in-graph .argmax UOps lose dtype under capture
    # and can return out-of-range garbage on some renders):
    mx = lg_s.max(-1, keepdim=True)
    am = ((lg_s == mx) * Tensor.arange(lg_s.shape[-1]).reshape(1, 1, -1)).max(-1).cast(dtypes.int32)
    return am
def draft_tail(hd):
    if _dtail_j[0] is None:
        _HNBUF[0] = hd.contiguous().realize()
        _dtail_j[0] = TinyJit(_dtail_fwd)
        for _ in range(2):   # cnt0 + cnt1 capture
            with Context(JIT=1):
                r = _dtail_j[0](_HNBUF[0])
                r = r.contiguous().realize()
        # note: capture ran on the CURRENT hd content; replays re-copyin below
    import numpy as _npdt
    dev = Device["NV"]
    dev.allocator._copyin(_HNBUF[0].uop.buf_uop.buffer._bufs["NV"],
                          memoryview(hd.contiguous().realize().numpy().astype(_npdt.float32).tobytes()).cast("B"))
    with Context(JIT=1):
        r = _dtail_j[0](_HNBUF[0]).contiguous().realize()
    return int(_sl_np[int(r.numpy().reshape(-1)[0])])   # map slice idx -> token id
if os.getenv("MTP_COMMIT_JIT"):
    def _cmt_fwd(tid_t, sp_t): return _fwd1(tid_t, sp_t)
    commit_j = TinyJit(_cmt_fwd)
    def commit_step(tid, pos):
        import numpy as _npc2
        dev = Device["NV"]
        if _CMTBUF[0] is None:
            _CMTBUF[0] = Tensor([[int(tid)]], dtype="int32").contiguous().realize()
        dev.allocator._copyin(_CMTBUF[0].uop.buf_uop.buffer._bufs["NV"],
                              memoryview(_npc2.asarray([[int(tid)]], dtype=_npc2.int32).tobytes()).cast("B"))
        with Context(JIT=1):
            h = commit_j(_CMTBUF[0], v_sp.bind(pos))
            h = h.contiguous().realize()
        return h

def head_rows(h, n):
    if "_HEAD_LAZY_ORIG" not in globals() and model.output.weight.dtype != dtypes.float16 and _head_j[0] is not None:
        lg = head_logits(h[:, :n, :].contiguous())
        return amds_from_logits(lg, n)
    with Context(JIT=2):
        lg = model.output(model.output_norm(h[:, :n, :]).half()).realize()
    return amds_from_logits(lg, n)

def amds_from_logits(lg, n):
    import numpy as np
    lg_np = lg.numpy()
    return [int(np.argmax(lg_np[0, j])) for j in range(n)], lg

def embed(tok_id):
    return model.token_embd(Tensor([[int(tok_id)]], dtype="int32")).float().contiguous().realize()

def attn_eager(b, x, pos):
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), v_sp.bind(pos) if not isinstance(pos, UOp) else pos)
    return (hh + b._feed_forward(b.ffn_norm(hh))).contiguous()

def draft_step_eager(pe, hm, pos):
    if os.getenv("MTPDBG"):
        print(f"[dbg] pe={pe.shape} hm={hm.shape} pos={pos}", flush=True)
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
    hdj = attn_eager(draft.blk, xin, pos)
    with Context(JIT=2):
        if _DRAFT_SLICE_N > 0 and "_draft_head_sliced" in globals():
            lgj = (_draft_head_sliced @ draft.head_norm(hdj).half()[:, -1:, :].reshape(-1, 1)).T.reshape(1, 1, -1).realize()
            import numpy as np
            return _draft_argmax(lgj), hdj[:, -1:, :].contiguous().realize()
        lgj = model.output(draft.head_norm(hdj).half()).realize()
    import numpy as np
    return int(np.argmax(lgj.numpy()[0, -1])), hdj[:, -1:, :].contiguous().realize()

# ctx>1024 schedules insert a COPY into the lm_head lazy-dequant chain that MATERIALIZES
# the raw-GGUF base view (5.09GB to-EOF extent; with a3 override: misaligned f32 reads ->
# SM fault; without: 6.82GB OOM. ctx-1024 schedules fuse and are clean). FIX: realize the
# head weight ONCE as fp16 (+1.67GB VRAM) — plain tensor, no lazy chain, no copy.
# 2026-08-30 FINAL DIAGNOSIS (AGENTS.md): at ctx>1024 the GPU faults with SM
# Misaligned Address REGARDLESS of override on/off or head fp16/lazy — the trigger is
# ALLOCATOR-LAYOUT-DEPENDENT BUFFER ALIGNMENT (a vectorized kernel gets a misaligned
# base at the 2048 layout). Layout-shuffling workarounds do NOT fix it. Proven-clean
# config remains ctx<=1024 (200/200 exact).
if (int(os.getenv("MTP_MAXCTX", "0") or CFG.max_context)) > 1024 and not os.getenv("MTP_HEAD_INT8") and not os.getenv("MTP_HEAD_LAZY"):
    # ctx>1024 notes (2026-08-30): MTP_A3C_OFF=1 REQUIRED (misaligned reads).
    # fp16-head workaround REMOVED: probe capture scratch ARENA legitimately peaks
    # ~6.8GB at ctx 2048; +2.54GB fp16 head on top OOMs 24GB VRAM. Without it fits.
    assert os.getenv("MTP_A3C_OFF"), "ctx>1024 requires MTP_A3C_OFF=1"
    if 2048 < (int(os.getenv("MTP_MAXCTX", "0") or CFG.max_context)):
        # at 8k..32k the fp16 head fits; at >32k (100k) VRAM = model 12.6G + KV 6.4G leaves
    # no room for +2.54G -> stock lazy head there (first use stages once).
        # capture arena and breaks VRAM. Materialize the head fp16 ONCE: the transient
        # staging copy happens at startup (fits), and the arena drops the 5.09GB slot.
        _w16 = model.output.weight.cast(dtypes.float16).contiguous().realize()
        model.output.weight = _w16
        print("[ctx] >2048 mode: a3c-off + fp16 head (arena de-staged)", flush=True)
    else:
        print("[ctx] >1024 mode: a3c-off, stock head", flush=True)
# MTP_HEAD16_CHUNKED: the stock lm_head lazy-dequant schedule materializes the FULL
# fp32 head (248320x5120x4B = 5.09GB) in the schedule arena during ANY eager head call
# (the 100k chunk-0 killer: 12.6 model + 6.4 KV + 5.09 arena > 24GB). Cast the head to
# fp16 ONCE at load, in vocab-row SLICES so no single schedule exceeds ~700MB arena.
# Steady state: +2.54GB resident fp16 head, zero dequant arenas.
if os.getenv("MTP_HEAD16_DIRECT") and not os.getenv("MTP_HEAD_INT8") and not os.getenv("MTP_HEAD_LAZY"):
    import time as _t_d
    _t0d = _t_d.perf_counter()
    _HEAD_LAZY_ORIG = model.output.weight   # MTP_HEAD_DROP: restore pre-capture
    model.output.weight = model.output.weight.cast(dtypes.float16).contiguous().realize()
    Device["NV"].synchronize()   # full drain: without it the next tiny copyin waits on a
    # stale b_timeline slot (run17: 32B token copyin wedged 30s right after the cast)
    print("[head] DIRECT fp16 cast OK+synced ({:.1f}s)".format(_t_d.perf_counter() - _t0d), flush=True)
elif os.getenv("MTP_HEAD16_CHUNKED"):
    import time as _t_h
    _t0h = _t_h.perf_counter()
    _W = model.output.weight
    _rows, _cols = _W.shape[0], _W.shape[1]
    _SLICE = 4096    # small slices keep the copyin ring shallow (620MB slices wedged at #7)
    _parts = []
    from tinygrad import dtypes as _dt_h
    from tinygrad.tensor import Tensor as _T_h
    import numpy as _np_h
    _parts = []
    for _r0 in range(0, _rows, _SLICE):
        _r1 = min(_r0 + _SLICE, _rows)
        _parts.append(_W[_r0:_r1].cast(_dt_h.float16).contiguous().realize().numpy())
        print(f"[head] slice {_r0}:{_r1} cast+cpu", flush=True)
    _buf = _T_h.zeros(_rows, _cols, dtype=_dt_h.float16, device="NV").contiguous().realize()
    for _i, _p in enumerate(_parts):
        _r0, _r1 = _i * _SLICE, min((_i + 1) * _SLICE, _rows)
        _buf[_r0:_r1].assign(_T_h(_p, device="NV")).realize()
        print(f"[head] slice {_r0}:{_r1} uploaded", flush=True)
    del _parts
    model.output.weight = _buf
    print(f"[head] fp16 head cast in {(_rows + _SLICE - 1)//_SLICE} slices ({_t_h.perf_counter()-_t0h:.1f}s)", flush=True)

# ---- DRAFT_JIT: one TinyJit for the whole draft step (JIT=2 graphless; no 2nd graph family) ----
def _draft_fwd(tid_t, hm_t, sp):
    # MTP_HD_INGRAPH: write the step's hd back into the hm INPUT buffer (the
    # select_states uop-store pattern) — the next step's replay reads it with
    # ZERO host hops (kills the ~17ms store-realize sync per step).
    _ing = os.getenv("MTP_HD_INGRAPH")
    pe = model.token_embd(tid_t).float()
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm_t), dim=-1))
    b = draft.blk
    x = xin
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), sp)
    hd = hh + b._feed_forward(b.ffn_norm(hh))
    if _DRAFT_SLICE_N > 0 and "_draft_head_sliced" in globals():
        if os.getenv("MTP_DRAFT_TAILIN"):
            # REGRESSION at 100k (run16: draft 71->100ms/cycle); default OFF.
            hn = draft.head_norm(hd[:, -1:, :]).half()
            lg_s = (_draft_head_sliced @ hn.reshape(-1, 1)).T.reshape(1, 1, -1)
            return lg_s, hd[:, -1:, :]
        return hd[:, -1:, :], hd[:, -1:, :]   # eager slice tail (run15 semantics)
    hd_last = hd[:, -1:, :]
    if _ing:
        # in-graph write-back: hm_t is a VIEW of _HMB's buffer — store hd into it
        hd_last = Tensor(hm_t.uop.after(hm_t.uop.store(hd_last.cast(dtypes.float32).uop)))
    lg = model.output(draft.head_norm(hd).half())[:, -1:, :]
    return lg, hd_last
draft_j = TinyJit(_draft_fwd)
_TIDB = [None]; _HMB = [None]
def _dbuf(t): return t.uop.buf_uop.buffer._bufs["NV"]

def draft_step(tid, hm, pos, _warm=False):
    """hm: [1,1,dim] fp32 Tensor (or None for zeros at pos 0). Returns (argmax, hd Tensor)."""
    import numpy as np, time as _tmod
    _dp = {"t0": _tmod.perf_counter()}
    dev = Device["NV"]
    if hm is None:
        hm = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    if _TIDB[0] is None:
        _TIDB[0] = Tensor([[int(tid)]], dtype="int32").contiguous().realize()
        _HMB[0] = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    _dp["t1"] = _tmod.perf_counter()
    dev.allocator._copyin(_dbuf(_TIDB[0]), memoryview(np.asarray([[int(tid)]], dtype=np.int32).tobytes()).cast("B"))
    if os.getenv("MTP_DRAFT_DEVCOPY"):
        hm_sink = Tensor(_HMB[0].uop.after(_HMB[0].uop.store(hm_c.cast(dtypes.float32).uop)))
        hm_sink.realize()
    elif hm is _HMB[0] and os.getenv("MTP_HSEED_SEL"):
        pass   # _HMB already holds hm (select_final stored hA[:,m]); no hop needed
    elif os.getenv("MTP_HM_DEVSTORE") and not os.getenv("MTP_HD_INGRAPH"):
        # dperf: every full sync costs ~17ms (TB/dext roundtrip). Enqueue the hm
        # store from the ALREADY-REALIZED hm (no hm_c copy); realize to enqueue;
        # queue order feeds the replay.
        _hms = Tensor(_HMB[0].uop.after(_HMB[0].uop.store(hm.cast(dtypes.float32).contiguous().uop)))
        _hms.realize()
    else:
        hm_c = hm.contiguous().realize()
        dev.allocator._copyin(_dbuf(_HMB[0]), memoryview(hm_c.numpy().astype(np.float32).tobytes()).cast("B"))
    _dp["t2"] = _tmod.perf_counter()
    with Context(JIT=1 if os.getenv("MTP_DRAFT_JIT1") else 2):
        r0, hd = draft_j(_TIDB[0], _HMB[0], v_sp.bind(pos))
        hd = hd.contiguous().realize()
    _dp["t3"] = _tmod.perf_counter()
    if _DRAFT_SLICE_N > 0 and "_draft_head_sliced" in globals():
        if os.getenv("MTP_DRAFT_TAILIN"):
            lg_s = r0.contiguous().realize()   # computed in-graph
        elif os.getenv("MTP_DTAIL_JIT"):
            _t4d = _tmod.perf_counter()
            pj = draft_tail(hd)
            Device["NV"].synchronize()
            if os.getenv("MTP_DPERF"):
                print(f"[dperf] hm_rlz={(_dp['t1']-_dp['t0'])*1e3:.1f} copyin={(_dp['t2']-_dp['t1'])*1e3:.1f} "
              f"replay={(_dp['t3']-_dp['t2'])*1e3:.1f} tail={(_tmod.perf_counter()-_t4d)*1e3:.1f}", flush=True)
            return pj, hd
        else:
            hn = draft.head_norm(hd).half()
            lg_s = (_draft_head_sliced @ hn.reshape(-1, 1)).T.reshape(1, 1, -1).contiguous().realize()
        Device["NV"].synchronize()
        return _draft_argmax(lg_s), hd
    lg = r0.contiguous().realize()
    Device["NV"].synchronize()
    if os.getenv("MTP_DPERF"):
        print(f"[dperf] hm_rlz={(_dp['t1']-_dp['t0'])*1e3:.1f} copyin={(_dp['t2']-_dp['t1'])*1e3:.1f} "
              f"replay={(_dp['t3']-_dp['t2'])*1e3:.1f} tail={(_tmod.perf_counter()-_dp['t4'])*1e3:.1f}", flush=True)
    return int(np.argmax(lg.numpy()[0, -1])), hd

# ---- MTP_DCHAIN: the WHOLE draft phase (K steps: emb -> eh_proj -> block -> sliced
# head -> argmax -> NEXT-token emb, chained ON DEVICE) as ONE JIT=1 family.
# Kills the per-step host roundtrips (hm/hd numpy hops) and per-step syncs.
# Positions via the v_sp VARIABLE + j (graph-constant offsets); props read back
# once as K ints. Requires MTP_DRAFT_SLICE.
_dchain_j = [None]
def _dchain_fwd(tid_t, hm_t, sp_var):
    props = []
    hm = hm_t
    tid = tid_t
    for j in range(K):
        if j == 0:
            pe = model.token_embd(tid).float()            # TRUE token (host-provided)
        else:
            from tinygrad.llm.gguf import ggml_data_to_tensor as _g2t_dc
            pe = _g2t_dc(_draft_emb_sliced[tid.reshape(-1)].cast(dtypes.uint8),
                         cfg.dim, model.emb_ggml_type).reshape(1, 1, cfg.dim).cast(dtypes.float32)  # slice-space emb
        xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
        b = draft.blk
        x = xin; b._init_state(x)
        hh = x + b._attention(b.attn_norm(x), sp_var + j)
        hd = hh + b._feed_forward(b.ffn_norm(hh))
        hn = draft.head_norm(hd[:, -1:, :]).half()
        lg_s = (_draft_head_sliced @ hn.reshape(-1, 1)).T.reshape(1, 1, -1)
        # argmax via ordinary ops (in-graph argmax UOps lose dtype under capture):
        mx = lg_s.max(-1, keepdim=True)
        am = ((lg_s == mx) * Tensor.arange(lg_s.shape[-1]).reshape(1, 1, -1)).max(-1).cast(dtypes.int32)  # (1,1); max not sum: ties stay in-range
        props.append(am)
        tid = am  # STAY in slice-index space; the next emb comes from the sliced table
        hm = hd[:, -1:, :]
    return props[0].reshape(1).cat(*[p.reshape(1) for p in props[1:]], dim=0) if len(props) > 1 else props[0].reshape(1)

_draft_emb_sliced = None  # built lazily on first dchain (needs emb_rows); (Nslice, row_bytes) raw
def draft_chain(tid, hm, pos):
    global _draft_emb_sliced
    if _TIDB[0] is None:
        _TIDB[0] = Tensor([[int(tid)]], dtype="int32").contiguous().realize()
        _HMB[0] = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    if _draft_emb_sliced is None and model.emb_rows is not None:
        _draft_emb_sliced = model.emb_rows[_slice_ids_t].contiguous().realize()  # one-time raw-row gather
    """hm: (1,1,dim) fp32 device tensor. Returns the K proposed token ids (host list)."""
    import numpy as _npdc
    dev = Device["NV"]
    if _dchain_j[0] is None:
        _dchain_j[0] = TinyJit(_dchain_fwd)
        _ec = hm.contiguous().realize()
        for _ in range(2):
            with Context(JIT=1):
                r = _dchain_j[0](_TIDB[0], _ec, v_sp.bind(pos))
                r = r.contiguous().realize()
    dev.allocator._copyin(_TIDB[0].uop.buf_uop.buffer._bufs["NV"],
                          memoryview(_npdc.asarray([[int(tid)]], dtype=_npdc.int32).tobytes()).cast("B"))
    dev.allocator._copyin(_HMB[0].uop.buf_uop.buffer._bufs["NV"],
                          memoryview(hm.contiguous().realize().numpy().astype(_npdc.float32).tobytes()).cast("B"))
    with Context(JIT=1):
        r = _dchain_j[0](_TIDB[0], _HMB[0], v_sp.bind(pos)).contiguous().realize()
    dev.synchronize()
    return [int(_sl_np[t]) for t in r.numpy()]   # slice idx -> token id (host map)

_sel_j: dict = {}
_selfin_j = [None]
_hA_ref = [None]
def _mk_selfinal():
    # MTP_HSEED_SEL: also store the probe's hA[:, m] hidden into _HMB — the draft's
    # next-cycle hm — killing the h_seed eager realize (~17ms TB sync).
    # m==K full-accept fast path: the probe's RO-scratch FINAL states are exactly the
    # committed states (all K+1 tokens accepted) -> 96 in-place stores, no re-forward.
    def _sel():
        sinks = []
        for b in gdns:
            sinks.append(Tensor(b.conv_state.uop.after(b.conv_state.uop.store(
                b.probe_scratch_conv.cast(b.conv_state.dtype).uop))))
            sinks.append(Tensor(b.recurrent_state.uop.after(b.recurrent_state.uop.store(
                b.probe_scratch_rec.cast(b.recurrent_state.dtype).uop))))
        if os.getenv("MTP_HSEED_SEL") and _hA_ref[0] is not None:
            sinks.append(Tensor(_HMB[0].uop.after(_HMB[0].uop.store(
                _hA_ref[0][:, -1:, :].cast(dtypes.float32).uop))))
        return tuple(sinks)
    return TinyJit(_sel)
def select_final():
    # MTP_SEL_JIT1: run as a captured JIT=1 graph (one replay) instead of 96 eager
    # launches + sync (~68ms -> ~8ms). Stable buffers (probe RO scratch -> live).
    if _selfin_j[0] is None: _selfin_j[0] = _mk_selfinal()
    with Context(JIT=1 if os.getenv("MTP_SEL_JIT1") else 2):
        Tensor.realize(*_selfin_j[0]())
    if not os.getenv("MTP_SEL_ASYNC") and not os.getenv("MTP_SEL_JIT1"):
        Device["NV"].synchronize()   # JIT=1 replay's realize drains already
    # MTP_SEL_ASYNC: the stores are enqueued; the NEXT phase (draft copyin of the
    # next cycle) queue-drains them — saves one ~17ms TB sync roundtrip/cycle.
def _mk_sel(m):
    def _sel():
        sinks = []
        for b in gdns:
            sinks.append(Tensor(b.conv_state.uop.after(b.conv_state.uop.store(
                b.step_conv_buf[m].cast(b.conv_state.dtype).uop))))
            sinks.append(Tensor(b.recurrent_state.uop.after(b.recurrent_state.uop.store(
                b.step_rec_buf[m].cast(b.recurrent_state.dtype).uop))))
        return tuple(sinks)
    return TinyJit(_sel)

def select_states(m):
    # per-m TinyJit: 96 uop.stores re-lowered every cycle cost ~80ms of launch tax;
    # cached lowering replays them graphlessly in one batched realize.
    if m not in _sel_j: _sel_j[m] = _mk_sel(m)
    with Context(JIT=2):
        sinks = _sel_j[m]()
        Tensor.realize(*sinks)
    Device["NV"].synchronize()

if os.getenv("MTP_HEAD_INT8"):
    from head_i8 import install as _i8install
    _i8install(model)

# ---- MTP_CKPT: checkpoint/resume for the chunked prefill (crash-resilient) ----
import os as _os_k
_CKPT_DIR = os.getenv("MTP_CKPT_DIR", "~/ckpt100k")
_CKPT_EVERY = int(os.getenv("MTP_CKPT_EVERY", "8192"))

def _ckpt_save(pos_done, hid):
    import numpy as _np_k, pickle as _pk
    _os_k.makedirs(_CKPT_DIR, exist_ok=True)
    st = {"pos": pos_done}
    arrs = {}
    for _i, b in enumerate(model.blk):
        if getattr(b, "cache_kv", None) is not None:
            arrs[f"kv_{_i}"] = b.cache_kv.numpy()
        if hasattr(b, "conv_state"):
            arrs[f"cv_{_i}"] = b.conv_state.numpy()
            arrs[f"rc_{_i}"] = b.recurrent_state.numpy()
    arrs["dkv"] = draft.blk.cache_kv.numpy()
    arrs["hid"] = hid.numpy()
    with open(f"{_CKPT_DIR}/state.pkl", "wb") as f: _pk.dump(st, f)
    for k, v in arrs.items(): _np_k.save(f"{_CKPT_DIR}/{k}.npy", v)
    with open(f"{_CKPT_DIR}/pos.txt", "w") as f: f.write(str(pos_done))
    print(f"[ckpt] saved @{pos_done}", flush=True)

def _ckpt_load():
    import numpy as _np_k
    if not os.path.exists(f"{_CKPT_DIR}/pos.txt"): return 0, None
    pos_done = int(open(f"{_CKPT_DIR}/pos.txt").read())
    # blocks create their state buffers lazily in _init_state; ensure they exist
    _z = Tensor.zeros(1, 1, cfg.dim)
    for b in model.blk: b._init_state(_z)
    draft.blk._init_state(_z)
    for _i, b in enumerate(model.blk):
        if getattr(b, "cache_kv", None) is not None:
            _a = _np_k.load(f"{_CKPT_DIR}/kv_{_i}.npy")
            b.cache_kv.assign(Tensor(_a).cast(b.cache_kv.dtype)).realize()
        if hasattr(b, "conv_state"):
            b.conv_state.assign(Tensor(_np_k.load(f"{_CKPT_DIR}/cv_{_i}.npy")).cast(b.conv_state.dtype)).realize()
            b.recurrent_state.assign(Tensor(_np_k.load(f"{_CKPT_DIR}/rc_{_i}.npy")).cast(b.recurrent_state.dtype)).realize()
    if not os.getenv("MTP_CKPT_NODKV"):
        draft.blk.cache_kv.assign(Tensor(_np_k.load(f"{_CKPT_DIR}/dkv.npy")).cast(draft.blk.cache_kv.dtype)).realize()
    hid = Tensor(_np_k.load(f"{_CKPT_DIR}/hid.npy")).cast(dtypes.float32).contiguous().realize()
    print(f"[ckpt] resumed @{pos_done}", flush=True)
    return pos_done, hid

# ---- prefill ----
_prompt_txt = CFG.prompt
if os.getenv("MTP_PROMPT_FILE"):
    _prompt_txt = open(os.getenv("MTP_PROMPT_FILE")).read()
elif os.getenv("MTP_PROMPT_TEXT"):
    _prompt_txt = os.getenv("MTP_PROMPT_TEXT")
ids = [0] + tok.encode(_prompt_txt)
base = json.load(open(os.getenv("MTP_BASE_JSON", CFG.base_json)))
# ---- MTP_DRAFT_SLICE: draft head restricted to top-N vocab rows (syv-ai: 40,960 rows
# covers ~97.5% of outputs on this model). Greedy exactness is UNAFFECTED: the probe's
# full-vocab amd decides emissions; a draft proposal outside the slice only shortens m.
# Slice ids = most frequent prompt tokens + all baseline output tokens (deterministic).
_DRAFT_SLICE_N = int(os.getenv("MTP_DRAFT_SLICE", "0"))
def _build_draft_slice():
    """MTP_SLICE_LATE=1 defers this to AFTER the early captures: the head gather
    POISONS the allocator for the SS-stash T=3 capture at 100k (bisected root
    cause of the historical stash-fault class, 2026-09-08, pfr_ss1 repro)."""
    global _draft_head_sliced, _slice_ids_t, _sl_np, _draft_argmax
    import collections as _col, numpy as _np_s
    _cnt = _col.Counter(ids)
    _extra = []
    for _b in (base or [])[:NTOK]: _cnt[int(_b)] += 1   # keep baseline outputs selectable
    _slice_ids = sorted(i for i, _ in _cnt.most_common() if i < model.output.weight.shape[0])
    _slice_ids = _slice_ids[:_DRAFT_SLICE_N]
    _slice_ids_t = Tensor(_np.asarray(_slice_ids, dtype=_np_s.int32), device="NV").contiguous().realize()
    _sl_np = _np_s.asarray(_slice_ids, dtype=_np_s.int64)
    if hasattr(model.output, "q"):   # Int8Head installed: gather int8 rows + dequant
        _qh = model.output.q[_slice_ids_t].cast(dtypes.float16).reshape(-1, 40, 128)
        _sh = model.output.s[_slice_ids_t].reshape(-1, 40, 1)
        _draft_head_sliced = (_qh * _sh).reshape(-1, cfg.dim).contiguous().realize()
        print(f"[draft-slice] N={len(_slice_ids)} from INT8 head", flush=True)
        _w_src = None
    else:
        _w_src = model.output.weight
    if _w_src is not None:
        if _w_src.dtype != dtypes.float16:
            # LAZY head (post-drop): gather ROWS first (row-gather-dequant, the
            # emb-gather pattern) — casting the whole lazy head would materialize 5.09GB.
            _draft_head_sliced = _w_src[_slice_ids_t].cast(dtypes.float16).contiguous().realize()
        else:
            _draft_head_sliced = _w_src[_slice_ids_t].contiguous().realize()   # (N, dim) fp16
        print(f"[draft-slice] N={len(_slice_ids)} materialized ({float(_draft_head_sliced.nbytes())/1e6:.0f}MB)", flush=True)
    def _draft_argmax(lg_slice):
        return int(_sl_np[int(_np_s.argmax(lg_slice.numpy()[0, -1]))])

if _DRAFT_SLICE_N > 0 and not os.getenv("MTP_SLICE_LATE"):
    _build_draft_slice()

t_p = time.perf_counter()
zeros = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()

# NOTE: a one-shot T=8 draft-block forward here requested a 5.09GB phantom buffer
# (lazy symbolic view in the T>1 feed-forward chain, 2026-08-30). The proven T=1
# draft_step path is used per token instead — exact, jit-cached, ~21ms/token.

if os.getenv("MTP_CHUNKED_PREFILL"):
    # T=32 chunks through the exact T>1 stack (all tokens REAL -> state stores correct,
    # PROBE_RO toggled OFF around chunk forwards). masked-SDPA path for T>8 (SEQ_ATTN
    # gate covers T<=8 only). Draft KV filled chunk-wise (one forward per 32 tokens).
    import numpy as _npc
    CH = 8   # inside the SEQ_ATTN T<=8 gate; T=32 masked-SDPA path unproven
    _ro = os.environ.pop("MTP_PROBE_RO", None)   # chunk forwards must advance live state
    os.environ.setdefault("MTP_BLKGROUP", "8")
    globals()["_BLKG"] = int(os.environ["MTP_BLKGROUP"])
    _ss = os.environ.pop("MTP_STEP_STATES", None)  # T=8 stash during eager chunks poisons the
    # later JIT=1 probe capture (device fault; composite-repro bisected 2026-08-30 — clean
    # with stash off during advance, fault with it on). The probe stashes its own T=3 set.
    dev0 = Device["NV"]
    h_last = None
    hid_prev = None
    _resume_pos, _resume_hid = (0, None) if not os.getenv("MTP_CKPT") else _ckpt_load()
    _start = _resume_pos if _resume_pos > 0 else 0
    hid_prev = _resume_hid if _start > 0 else None
    if os.getenv("MTP_EARLY_CAPTURE") and _start > 0:
        # Capture the probe graph YOUNG (before the advance churns the allocator): the
        # 100k fault is allocation-history-dependent (probe cnt0 faults after the full
        # mtp_v3 prefill sequence, but the identical capture is CLEAN in a fresh
        # process — probe_fault_repro pfr4/11). States/KV are address-stable buffers
        # the advance mutates in place, so a graph captured at _start replays
        # correctly at any later pos. Bonus: crash-recovery never re-captures.
        import numpy as _npec
        # The commit-family capture runs T=1 forwards that ADVANCE the live GDN
        # recurrent/conv state (no RO at T=1). The recurrent state is NOT positional —
        # the advance won't overwrite it — so snapshot/restore around the captures
        # (run12: the corruption diverged greedy at token 5 => 59/60).
        if os.getenv("MTP_HEAD_DROP") and "_HEAD_LAZY_ORIG" in globals():
            # drop the 2.54GB fp16 head BEFORE the captures (the stash arena needs
            # it at 100k); head_rows at prefill-end uses the head_j lazy family.
            _h16t = model.output.weight   # fp16 Tensor — capture BEFORE swap
            model.output.weight = _HEAD_LAZY_ORIG
            globals().pop("_HEAD_LAZY_ORIG", None)
            # FORCE-FREE: the head's Buffer object survives in closure/UOp-graph refs
            # so __del__ never runs and the opaque never enters the LRU (the 2.54GB
            # leak). Explicit deallocate() returns the device memory NOW.
            try:
                _h16t.uop.buf_uop.buffer.deallocate()
                print("[early-cap] head buffer FORCE-deallocated", flush=True)
            except Exception as _e:
                print(f"[early-cap] force-free failed: {type(_e).__name__}: {_e}", flush=True)
            for _hn in ("_w16", "_w_src", "_HEAD16_ORIG"):
                globals().pop(_hn, None)   # module locals pin the fp16 head (the 2.54GB leak)
            # nuclear: drop EVERY disposable module-level underscore global (some name
            # still pins the fp16 head — dict-with-str-keys referrer, not yet identified)
            _keep = ("_CKPT_DIR", "_CKPT_EVERY", "_DRAFT_SLICE_N", "_draft_argmax",
                     "_head_j", "_HJBUF", "_dchain_j", "_sel_j", "_selfin_j",
                     "_TOKBUF", "_TIDB", "_HMB", "_CMTBUF", "_T2BUF", "_draft_head_sliced",
                     "_slice_ids_t", "_sl_np", "_BLKG", "_ec_toks", "_ec_hm", "_ec_snap",
                     "_cmt_fwd", "_cmt2_fwd", "_dchain_fwd", "_dtail_j", "_HNBUF",
                     "_HEAD_LAZY_ORIG", "_build_draft_slice", "_draft_emb_sliced")
            from tinygrad.tensor import Tensor as _Tpurge
            from tinygrad.uop.ops import UOp as _Upurge
            for _gn, _gv in list(globals().items()):
                if _gn.startswith("_") and not _gn.startswith("__") and _gn not in _keep \
                   and isinstance(_gv, (_Tpurge, _Upurge)):
                    try: del globals()[_gn]
                    except Exception: pass
            import gc as _gc_hd
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); _gc_hd.collect()
            Device["NV"].allocator.free_cache(); _gc_hd.collect()
            def _toplive2(tag):
                import gc as _g2
                from tinygrad.device import Buffer
                _bs = [b for b in _g2.get_objects() if isinstance(b, Buffer) and b.is_initialized()]
                _sz = {}
                for b in _bs:
                    _szk = b.size * (getattr(b.dtype, "itemsize", 4))
                    _sz[_szk] = _sz.get(_szk, 0) + 1
                for k, v in sorted(_sz.items(), key=lambda x: -x[0])[:5]:
                    print(f"[vrdbg{tag}] {k/1e9:.3f}GB x{v}", flush=True)
                for b in _bs:
                    if getattr(b, "device", "") == "NV" and b.size * 4 > 2e9:
                        rr = [type(r).__name__ for r in _g2.get_referrers(b)][:6]
                        print(f"[vrdbg{tag}] big-buf referrers: {rr}", flush=True)
                        break
            _toplive2("post")
            # hunt: identify WHAT the 2.5GB buffer belongs to
            import gc as _g3
            from tinygrad.device import Buffer as _B3
            for b in _g3.get_objects():
                if isinstance(b, _B3) and b.is_initialized() and getattr(b, "device", "") == "NV" and b.size * 4 > 2e9:
                    for r in _g3.get_referrers(b):
                        if isinstance(r, list):
                            print(f"[vrdbgX] in list len={len(r)} sample={[type(x).__name__ for x in r[:4]]}", flush=True)
                        elif isinstance(r, dict):
                            ks = list(r.keys())[:4]
                            print(f"[vrdbgX] in dict keys={[type(k).__name__ for k in ks]}", flush=True)
                    break
            # name the holder: find which module-global NAME (transitively) holds the buffer
            import gc as _g4
            from tinygrad.device import Buffer as _B4
            _tgt = next((b for b in _g4.get_objects()
                         if isinstance(b, _B4) and b.is_initialized()
                         and getattr(b, "device", "") == "NV" and b.size * 4 > 2e9), None)
            if _tgt is not None:
                import types as _ty4
                def _named_path(o, path, seen, depth=0):
                    if depth > 5 or id(o) in seen: return
                    seen.add(id(o))
                    for r in _g4.get_referrers(o):
                        tn = type(r).__name__
                        if isinstance(r, dict) and "__name__" in r and isinstance(r.get("__name__"), str):
                            _hits = [k for k, v in r.items() if not k.startswith("__") and (v is o or (hasattr(v, "uop") and getattr(v, "uop", None) is o))]
                            print(f"[lh] MODULE {r['__name__']} direct-hits={_hits[:6]} via {path[-1]}", flush=True)
                            continue
                        if isinstance(r, _ty4.ModuleType):
                            print(f"[lh] MOD-OBJ {r.__name__} via {path[-1]}", flush=True); continue
                        if isinstance(r, list):
                            _named_path(r, path + [f"list[{len(r)}]"], seen, depth+1); continue
                        if isinstance(r, dict):
                            _hits = [k for k, v in list(r.items())[:20000] if v is o]
                            if _hits: print(f"[lh] DICT holds keys {_hits[:6]} via {path[-1]}", flush=True)
                            _named_path(r, path + ["dict"], seen, depth+1); continue
                        if callable(r) and hasattr(r, "__qualname__"):
                            print(f"[lh] FUNC {r.__qualname__} via {path[-1]}", flush=True)
                            _named_path(r, path + [r.__qualname__], seen, depth+1); continue
                        print(f"[lh] {tn} via {path[-1]}", flush=True)
                _named_path(_tgt, ["buffer"], set())
            print("[early-cap] fp16 head DROPPED pre-capture", flush=True)
            _z_hj = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
            if not os.getenv("MTP_HEADPROBE") and not os.getenv("MTP_HEAD_DROP"):
                # HEADPROBE derives cur/h_seed from probe replays — no head_j needed.
                # The eager warmup materializes the lazy-head dequant arena (eats the
                # freed 2.54GB), so SKIP it entirely when HEADPROBE is on.
                # MTP_HEAD_DROP: head_logits MATERIALIZES the 2543MB fp16 head right
                # after the drop (alloc-stack caught it, 3ff0ea0+); skip it too — the
                # probe graph binds the lazy Q5 head.
                head_logits(_z_hj)   # capture the lazy-head logits family NOW (young allocator)
            # its cnt0 EAGER pass materializes the lazy-head dequant arena; drain the
            # residue or the stash capture OOMs at the same 22.42GB (run27/28).
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); _gc_hd.collect()
            Device["NV"].allocator.free_cache(); _gc_hd.collect()
            print("[early-cap] head_j captured + drained", flush=True)
        _ec_snap = [(b.conv_state.clone().realize(), b.recurrent_state.clone().realize()) for b in gdns] \
            if gdns else []
        _ec_toks = [0] * (K + 1)   # K-generic: the probe graph must be T=K+1
        _TOKBUF[0] = Tensor([_ec_toks], dtype="int32").contiguous().realize()
        # the capture MUST see the probe env (RO scratch states + per-step stash):
        # temporarily restore what the chunk-prefill prologue popped.
        if _ro is not None and os.getenv("MTP_EC_RO", "1") == "1": os.environ["MTP_PROBE_RO"] = _ro
        if _ss is not None and os.getenv("MTP_EC_SS", "1") == "1": os.environ["MTP_STEP_STATES"] = _ss
        print(f"[early-cap] env RO={os.getenv('MTP_EC_RO','1')} SS={os.getenv('MTP_EC_SS','1')}", flush=True)
        if os.getenv("MTP_DRAFT_JIT1") and os.getenv("MTP_EC_SS", "0") != "1":   # route2: SS captures probe-first
            import numpy as _npdj
            _HMB[0] = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
            _ec_hm = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
            for _dj, _dsp in enumerate([_start, _start + 1]):
                if os.getenv("MTP_EC_DRAIN") == "2":
                    Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
                    print("[early-cap] drain-call", flush=True)
                _t0 = time.perf_counter()
                print(f"[early-cap] draft call {_dj} sp={_dsp}", flush=True)
                draft_step(0, _ec_hm, _dsp)
                print(f"[early-cap] draft {_dj} CLEAN ({time.perf_counter()-_t0:.1f}s)", flush=True)
        if os.getenv("MTP_EC_DRAIN"):
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
            print("[early-cap] drain", flush=True)
        for _eci, _ecsp in enumerate([_start, _start + 3]):
            if os.getenv("MTP_EC_DRAIN") == "2":
                Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
                print("[early-cap] drain-call", flush=True)
            _t0 = time.perf_counter()
            print(f"[vram-trace] pre-probe-call used={getattr(Device['NV'].allocator, '_MemPool__used', 'n/a')}", flush=True)
            print(f"[early-cap] probe call {_eci} cnt={probe_j.cnt} sp={_ecsp}", flush=True)
            with Context(JIT=1):
                _r = probe_j(_TOKBUF[0], v_sp.bind(_ecsp), True)
                _ = _r[0].contiguous().realize(); _ = _r[1].contiguous().realize()
            Device["NV"].synchronize()
            print(f"[early-cap] call {_eci} CLEAN ({time.perf_counter()-_t0:.1f}s)", flush=True)
        if os.getenv("MTP_EC_DRAIN"):
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
            print("[early-cap] drain", flush=True)
        if os.getenv("MTP_COMMIT_JIT"):
            for _cj, _csp in enumerate([_start, _start + 1]):
                _t0 = time.perf_counter()
                print(f"[early-cap] commit call {_cj} sp={_csp}", flush=True)
                commit_step(0, _csp)
                print(f"[early-cap] commit {_cj} CLEAN ({time.perf_counter()-_t0:.1f}s)", flush=True)
        if os.getenv("MTP_EC_DRAIN"):
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
            print("[early-cap] drain", flush=True)
        if os.getenv("MTP_SEL_JIT1"):
            if _HMB[0] is None:
                _HMB[0] = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
            if os.getenv("MTP_HSEED_SEL"):
                _hA_ref[0] = _r[0]   # the probe's stable h output buffer
            _t0 = time.perf_counter()
            select_final(); select_final()   # cnt0 + cnt1 capture
            print(f"[early-cap] select_final captured ({time.perf_counter()-_t0:.1f}s)", flush=True)
        if os.getenv("MTP_COMMIT2"):
            for _c2, _c2sp in enumerate([_start, _start + 2]):
                _t0 = time.perf_counter()
                print(f"[early-cap] commit2 call {_c2} sp={_c2sp}", flush=True)
                commit2_step([0, 0], _c2sp)
                print(f"[early-cap] commit2 {_c2} CLEAN ({time.perf_counter()-_t0:.1f}s)", flush=True)
        if os.getenv("MTP_SCAN_SPLIT"):
            # young-allocator capture: the split fragments (96 families) render
            # NOW (young allocator, post-drain) instead of at the transition (the
            # post-prefill state faults the late-block captures at 100k).
            import fwd3_split as _fs0
            if _SPLIT_FWD is None:
                _fs0.init(model, T=K + 1)
                globals()["_SPLIT_FWD"] = _fs0
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
            _t0y = time.perf_counter()
            try:
                _ecy = _fs0.fwd3_split(model, _TOKBUF[0], _start, v_sp=v_sp)
                Device["NV"].synchronize()
                print(f"[early-cap] SPLIT fragments captured ({time.perf_counter()-_t0y:.1f}s)", flush=True)
            except RuntimeError as _e:
                if "SPLIT_ABORT_AT" in str(_e):
                    print(f"[early-cap] SPLIT chunk A done ({_e}); cache warm — exiting", flush=True)
                    import sys as _sysex
                    _sysex.exit(0)
                raise
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
        if os.getenv("MTP_DRAFT_JIT1") and os.getenv("MTP_EC_SS", "0") == "1":   # route2: SS order
            import numpy as _npdj
            _HMB[0] = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
            _ec_hm = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
            for _dj, _dsp in enumerate([_start, _start + 1]):
                if os.getenv("MTP_EC_DRAIN") == "2":
                    Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
                    print("[early-cap] drain-call", flush=True)
                _t0 = time.perf_counter()
                print(f"[early-cap] draft call {_dj} sp={_dsp}", flush=True)
                draft_step(0, _ec_hm, _dsp)
                print(f"[early-cap] draft {_dj} CLEAN ({time.perf_counter()-_t0:.1f}s)", flush=True)
        if os.getenv("MTP_EC_DRAIN"):
            Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); __import__("gc").collect()
            print("[early-cap] drain", flush=True)
        for _b_, (_sc, _sr) in zip(gdns, _ec_snap):
            _b_.conv_state.assign(_sc).realize()
            _b_.recurrent_state.assign(_sr).realize()
        Device["NV"].synchronize()
        if _ro is not None: os.environ.pop("MTP_PROBE_RO", None)   # advance wants them off
        if _ss is not None: os.environ.pop("MTP_STEP_STATES", None)
        dev0.allocator.free_cache(); __import__("gc").collect()
        print("[early-cap] probe graph CAPTURED pre-advance + GDN state restored", flush=True)
    if _DRAFT_SLICE_N > 0 and os.getenv("MTP_SLICE_LATE") and "_draft_head_sliced" not in globals():
        # build the slice AFTER the early captures (young allocator: the head gather
        # poisons the stash capture) but BEFORE the chunk loop (the prefill draft
        # fill must use the slice — an eager FULL-head call would materialize the
        # 5.09GB arena; under MTP_HEAD_LAZY there is no fp16 head at all).
        import gc as _gc_l
        Device["NV"].synchronize()
        _build_draft_slice()
        Device["NV"].allocator.free_cache(); _gc_l.collect()
        print("[slice-late] built post-capture pre-prefill", flush=True)
    for c0 in range(_start, len(ids), CH):
        chunk = ids[c0:c0 + CH]
        toks_c = Tensor.zeros(1, len(chunk), dtype=dtypes.int32).contiguous().realize()
        dev0.allocator._copyin(toks_c.uop.buf_uop.buffer._bufs["NV"], memoryview(_npc.asarray([chunk], dtype=_npc.int32).tobytes()).cast("B"))
        with Context(JIT=2):
            h_last = _fwd3(toks_c, v_sp.bind(c0)).realize()   # (1, T, dim); advances state+KV
        # draft fill for this chunk needs PREV-token hiddens: use this chunk h h[:, :-1] with
        # the previous chunk tail; position i uses hm = hidden of token i-1
        if not os.getenv("MTP_CHUNK_NODRAFT"):
            for j, t_ in enumerate(chunk):
                if c0 + j == 0: hm = zeros
                elif j == 0: hm = hid_prev          # prev chunk tail (NOT h_last[:,-1:0,:]=empty)
                else: hm = h_last[:, j-1:j, :].contiguous().realize()
                if os.getenv("MTP_CHUNK_EAGERDRAFT"):
                    draft_step_eager(embed(t_), hm, c0 + j)   # no jit replay interleave
                else:
                    draft_step(t_, hm, c0 + j)
        hid_prev = h_last[:, -1:, :].contiguous().realize()
        dev0.synchronize()
        if os.getenv("MTP_CKPT") and (c0 + len(chunk)) % _CKPT_EVERY < CH:
            _ckpt_save(c0 + len(chunk), hid_prev)
        if c0 % 16 == 0:
            # eager per-chunk schedules alloc sp-SIZED intermediates (grow with c0) -> the
            # LRU buffer cache holds every historical size forever -> TLSF fragmentation
            # death at ~14GB used. Drain the cache periodically during prefill.
            if c0 % 128 == 0:
                print(f"[chunk {c0}/{len(ids)}] {time.perf_counter()-t_p:.1f}s", flush=True)
    if os.getenv("MTP_CKPT") and _resume_pos >= len(ids) and hid_prev is None:
        hid_prev = _resume_hid
    if _ro is not None: os.environ["MTP_PROBE_RO"] = _ro
    if _ss is not None: os.environ["MTP_STEP_STATES"] = _ss
    # NUCLEAR cleanup between prefill and head: save h_last to numpy (KBs), delete ALL
    # intermediate tensors, drain LRU, GC (twice — GC frees Python refs that hold buffers,
    # so a second free_cache catches those), then rebuild h_last from numpy and compute.
    # runs 31/32/33 all died here at ~22.1GB + 1.86GB needed.
    dev0.synchronize()
    # SEQ_ATTN_BATCH leaves larger qk/softmax intermediates in the LRU at the last
    # chunk; drain BEFORE the numpy handoff too (run17d/e faulted here otherwise).
    dev0.allocator.free_cache()
    import gc as _gc_t
    _gc_t.collect()
    dev0.allocator.free_cache()
    _hl_np = hid_prev.numpy()
    _cur_h = h_last[:, -1:, :].contiguous().realize().numpy()
    del h_last, hid_prev, zeros
    if "hm" in dir(): del hm
    for _v in list(locals().keys()):
        if _v.startswith("_") and not _v.startswith("__") and _v not in ("_CKPT_DIR", "_CKPT_EVERY", "_hl_np", "_cur_h"):
            try: del _v
            except: pass
    dev0.allocator.free_cache()
    import gc as _gc_p
    _gc_p.collect()
    dev0.allocator.free_cache()
    _gc_p.collect()
    dev0.synchronize()
    hid_prev = Tensor(_hl_np).cast(dtypes.float32).contiguous().realize()
    h_last = Tensor(_cur_h).cast(dtypes.float32).reshape(1, 1, -1).contiguous().realize()
    if os.getenv("MTP_HEADPROBE") and "_HEAD_LAZY_ORIG" not in globals() and model.output.weight.dtype != dtypes.float16:
        # head dropped (lazy Q5 bound in-graph): derive cur + h_seed from ONE probe
        # replay on the last prompt token. KV writes land at pos..pos+2 (>= prompt
        # end, rewritten by cycle 1 anyway); GDN states -> RO scratch (live intact).
        _hp_toks = [ids[-1]] + [0] * K
        hA_hp, lgA_hp = probe(_hp_toks, len(ids) - 1)
        _lg_np = lgA_hp.numpy()[0]
        cur = int(_lg_np[0].argmax())
        h_seed = hA_hp[:, 0:1, :].contiguous().realize()
        print(f"[headprobe] cur={cur} (probe-replay path)", flush=True)
    else:
        h_seed = h_last[:, -1:, :].contiguous().realize()
        cur = head_rows(h_last[:, -1:, :], 1)[0][0]
    # warm draft jit + select jits happen naturally in cycle 1
else:
    hs = []
    for i, tid in enumerate(ids):
        h = fwd1_eager(tid, i)
        hs.append(h)
    h_seed = hs[-1].contiguous().realize()
    cur = head_rows(hs[-1], 1)[0][0]
    for i, tid in enumerate(ids):
        hm = zeros if i == 0 else hs[i - 1]
        if i < 3:
            draft_step_eager(embed(tid), hm, i)   # warm draft KV through the eager path first
        else:
            draft_step(tid, hm, i)
print(f"[prefill] {len(ids)} tok, cur={cur}, {time.perf_counter()-t_p:.1f}s", flush=True)
# ---- W3: engine0 snapshot dump (MTP_SNAP100K=1) ----
# Post-prefill state: all len(ids) tokens fed; cur = first predicted token.
# The ENGINE seeds pos_slot=len(ids), tok_slot=cur (shifted contract, see W2_100K.md).
if os.getenv("MTP_SNAP100K"):
    import numpy as _np_s
    _S = "~/snap100k"
    os.makedirs(_S, exist_ok=True)
    for _i, b in enumerate(model.blk):
        _kv = getattr(b, "cache_kv", None)
        if _kv is not None and hasattr(_kv, "shape"):
            _np_s.save(f"{_S}/kv_{_i}.npy", _kv.numpy().reshape(2, 4, 100352, 256).astype(_np_s.float16))
            print(f"[snap100k] kv_{_i}", flush=True)
        elif isinstance(b, GatedDeltaNetBlock):
            _np_s.save(f"{_S}/conv_{_i}.npy", b.conv_state.float().numpy().reshape(-1).astype(_np_s.float32))
            _np_s.save(f"{_S}/rc_{_i}.npy", b.recurrent_state.float().numpy().reshape(-1).astype(_np_s.float32))
    _np_s.save(f"{_S}/ids.npy", _np_s.array(ids, dtype=_np_s.int64))
    import json as _json_s
    _json_s.dump({"P": len(ids), "CTXK": 100352, "cur0": int(cur)}, open(f"{_S}/meta.json", "w"))
    print("[snap100k] prefill state saved", flush=True)

# ---- verify probe states buffers exist (MTP_STEP_STATES created them at first T>1 fwd) ----
# (they materialize on first probe call)

# pre-capture drain: the cleanup above + head/draft/select warm leave LRU buffers
# behind; the T=3 capture at 100k OOMs by ~1.4GB without this (22.29GB used).
import gc as _gc_c
Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); _gc_c.collect()
Device["NV"].allocator.free_cache(); _gc_c.collect()
print(f"[pre-capture] drained; used={Device['NV'].allocator._MemPool__used if hasattr(Device['NV'].allocator, '_MemPool__used') else 'n/a'}", flush=True)
if os.getenv("MTP_HEAD_DROP") and "_HEAD_LAZY_ORIG" in globals():
    # free the 2.54GB fp16 head before the T=3 capture: the captured graph binds the
    # lazy Q5 head (in-graph fused dequant, raw bytes resident in the model) instead.
    # Requires MTP_DRAFT_SLICE (draft_j's captured graph would pin the fp16 buffer).
    assert os.getenv("MTP_DRAFT_SLICE"), "MTP_HEAD_DROP requires MTP_DRAFT_SLICE"
    model.output.weight = _HEAD_LAZY_ORIG
    globals().pop("_HEAD_LAZY_ORIG", None)
    import gc as _gc_h
    Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); _gc_h.collect()
    Device["NV"].allocator.free_cache(); _gc_h.collect()
    print("[pre-capture] fp16 head DROPPED (probe graph binds lazy Q5 head)", flush=True)
    print(f"[vram-trace] post-head-drop used={getattr(Device['NV'].allocator, '_MemPool__used', 'n/a')}", flush=True)

if os.getenv("MTP_HIST_DUMP"):
    # aggregate the LAST ~6 prefill chunks' kernel records (T=8 eager, same
    # GEMV/scan shapes as the probe; attention shapes are the 100k ones)
    import collections as _chist
    _recs = getattr(Device["NV"], "sig_prof_records", [])
    _tail = _recs[-6 * 1900:]
    _agg = collections.Counter(); _cnt = collections.Counter()
    for st, en, name, dname, pk in _tail:
        try: _dt = float(en.timestamp) - float(st.timestamp)
        except Exception: continue
        if _dt > 0: _agg[name] += _dt/1e3; _cnt[name] += 1
    _rows = sorted(_agg.items(), key=lambda x: -x[1])[:30]
    print(f"[hist100k] last-chunks kern_exec={sum(_agg.values())/6:.1f}ms/chunk top:", flush=True)
    for nm, tm in _rows:
        print(f"  {tm*1000/_cnt[nm]:9.1f}us x{_cnt[nm]//6:<4} {tm/6:8.3f}ms  {nm[:100]}", flush=True)
    del _recs[:]

if os.getenv("MTP_EC_DRAIN") == "2":
    # transition drain: all 5 advance chunks are clean but the first probe replay
    # at K=3 faults without this (the 3rd fault surface, 6b37508)
    import gc as _gc_t
    Device["NV"].synchronize(); Device["NV"].allocator.free_cache(); _gc_t.collect()
    Device["NV"].allocator.free_cache(); _gc_t.collect()
    print("[transition] drained pre-steady", flush=True)
outs = []
pos = len(ids)
n_acc = n_prop = n_cyc = 0
t_probe = t_draft = t_head = t_sel = 0.0
t_all = time.perf_counter()
per_cyc = []
while len(outs) < NTOK:
    n_cyc += 1
    td0 = time.perf_counter()
    props = []
    hm = h_seed
    if os.getenv("MTP_NODRAFT_DECODE"):
        props = [cur] * K   # no draft execution: pure probe-cycling bisect
    elif os.getenv("MTP_DCHAIN") and _DRAFT_SLICE_N > 0:
        props = draft_chain(cur, hm, pos)
    else:
        for j in range(K):
            pj, hdj = draft_step(cur if j == 0 else props[-1], hm, pos + j)
            props.append(pj)
            hm = hdj
    t_draft += time.perf_counter() - td0

    tp0 = time.perf_counter()
    hA, _lg = probe([cur] + props, pos)   # _lg: logits (or int32 am under MTP_PROBE_AM)
    amA, _am = _lg, (_lg if os.getenv("MTP_PROBE_AM") else None)
    if os.getenv("MTP_HSEED_SEL"): _hA_ref[0] = hA   # live ref for the select jit
    t_probe += time.perf_counter() - tp0
    if os.getenv("MTP_TRACE_GAP"):
        d = Device["NV"]
        print(f"[gap] cyc{n_cyc} after-probe tv={d.timeline_value} sv={d.timeline_signal.value} gap={d.timeline_value - d.timeline_signal.value}", flush=True)
    if n_cyc == 3:
        tp = time.perf_counter() - tp0
        print(f"[cyc {n_cyc} probe-t] {tp:.3f}s", flush=True)
    th0 = time.perf_counter()
    if os.getenv("MTP_PROBE_AM") and _am is not None:
        # NB: without PROBE_AM, probe() returns (h, LOGITS) — _am holds lg here!
        amds = [int(t) for t in _am.numpy().reshape(-1)]   # 12B readback (in-graph argmax)
    elif os.getenv("MTP_AMDS_DEV"):
        # S1: argmax on DEVICE over the realized logits (eager small kernel ~1-5ms),
        # then a 12-byte D2H — replaces the 3MB numpy roundtrip (~45ms/cycle).
        _am_t = amA[0][:K + 1].argmax(-1).cast(dtypes.int32).contiguous().realize()
        amds = [int(x) for x in _am_t.numpy()]
    else:
        amds = [int(x) for x in amA.numpy()[0][:K + 1].argmax(axis=-1)]
    t_head += time.perf_counter() - th0

    m = 0
    for i, p in enumerate(props):
        if amds[i] == p: m = i + 1
        else: break
    bonus = amds[m]
    n_acc += m; n_prop += K

    ts0 = time.perf_counter()
    if os.getenv("MTP_SCAN_SPLIT"):
        print("[cycle-stage] pre-kv-restore", flush=True)
        kv_restore()   # every cycle: the commit re-forward provides the authoritative KV
                        # advance; the split probe's eager attn writes are always scratch
        print("[cycle-stage] kv restored", flush=True)
    if os.getenv("MTP_COMMIT_MODE"):
        if os.getenv("MTP_SELFIN") and m == K and not os.getenv("MTP_SCAN_SPLIT"):
            select_final()
            _h_last = None   # h_seed falls through to hA[:, m:m+1] below
        else:
            _cmt_toks = [cur] + props[:m]
            _h_last = None
            if os.getenv("MTP_COMMIT2") and len(_cmt_toks) == 2:
                _h_last = commit2_step(_cmt_toks, pos)
            else:
                for _j, _t_ in enumerate(_cmt_toks):
                    _h_last = commit_step(_t_, pos + _j) if os.getenv("MTP_COMMIT_JIT") else fwd1_eager(_t_, pos + _j)
        t_sel += time.perf_counter() - ts0
    else:
        select_states(m)
        t_sel += time.perf_counter() - ts0
    if os.getenv("MTP_TRACE_GAP"):
        d = Device["NV"]
        print(f"[gap] cyc{n_cyc} after-select tv={d.timeline_value} sv={d.timeline_signal.value} gap={d.timeline_value - d.timeline_signal.value}", flush=True)
    if os.getenv("MTP_HSEED_SEL") and os.getenv("MTP_COMMIT_MODE") and _h_last is None:
        pass   # select_final stored hA into _HMB; the draft reads _HMB directly
    elif os.getenv("MTP_COMMIT_MODE") and _h_last is not None:
        h_seed = _h_last.contiguous().realize()
    elif os.getenv("MTP_HSEED_SEL") and m == K:
        h_seed = _HMB[0]   # select_final stored hA[:, K] into _HMB in-graph
    else:
        h_seed = hA[:, m:m+1, :].contiguous().realize()
    if m >= 1 and not os.getenv("MTP_NODRAFT_DECODE"):
        for j in range(m):
            draft_step(props[j], hA[:, j:j+1, :].contiguous().realize(), pos + 1 + j)

    outs.append(cur)
    outs.extend(props[:m])
    cur = bonus
    pos += m + 1
    if n_cyc <= 6 or n_cyc % 5 == 0 or os.getenv("MTP_ALLCYC"):
        print(f"[cyc {n_cyc}] m={m}/{K} props={props} amd={amds} bonus={bonus} pos={pos} "
              f"acc={n_acc/max(n_prop,1):.2f}", flush=True)
    per_cyc.append(m + 1)

    # warm probe bookkeeping: probe timing includes realize+sync
    if len(outs) >= NTOK: break

dt = time.perf_counter() - t_all
gen = outs[:NTOK]
match = gen == base[:NTOK]
if not match:
    for _i, (a, b) in enumerate(zip(gen, base[:NTOK])):
        if a != b:
            print(f"[first-mismatch] idx={_i} gen={a} base={b} (token {_i+1})", flush=True)
            break
print(f"[emit] {gen[:12]}...", flush=True)
print(f"[base] {base[:12]}...", flush=True)
print(f"GREEDY MATCH: {sum(1 for a, b in zip(gen, base[:NTOK]) if a == b)}/{NTOK}"
      f"{' (FULL)' if match else ''}", flush=True)
print(f"decode: {NTOK/dt:.2f} tok/s ({dt/NTOK*1e3:.1f} ms/tok) n={NTOK} "
      f"acc={n_acc}/{n_prop} cyc={n_cyc} tok/cyc={NTOK/max(n_cyc,1):.2f}", flush=True)
print(f"split: probe={t_probe*1e3:.0f}ms draft={t_draft*1e3:.0f} head={t_head*1e3:.0f} sel={t_sel*1e3:.0f}", flush=True)
