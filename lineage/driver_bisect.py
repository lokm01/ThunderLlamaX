# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Driver-component bisection: which of draft/head/select triggers the 4.4s probe?
Runs the real driver loop with components disabled via env flags."""
import os, sys, time, json
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")

NO_DRAFT = os.getenv("NO_DRAFT", "") == "1"
NO_HEAD = os.getenv("NO_HEAD", "") == "1"
NO_SELECT = os.getenv("NO_SELECT", "") == "1"

from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, TransformerBlock, flush_step_states
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import nn, dtypes
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context, getenv
from mtp_config import MTPConfig

CFG = MTPConfig.load()
K = CFG.K
NTOK = 6
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

model, kv = Transformer.from_gguf(MODEL, CFG.max_context)
cfg = model.blk[-1].config
from tinygrad.llm.cli import SimpleTokenizer
tok = SimpleTokenizer.from_gguf_kv(kv)

# draft setup (same as mtp_v3)
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
        out[nm] = ggml_data_to_tensor(Tensor(_np.frombuffer(raw, _np.uint8).copy()), n, typ).reshape(*reversed(dims))
    r.close()
    return out
sd = extract_tensors(MODEL, "blk.64")
class Draft: pass
draft = Draft()
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
v_sp = UOp.variable("mtp_sp", 0, CFG.max_context - 2 - K)

def _fwd3(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    return flush_step_states(x.contiguous(), model.blk).contiguous()

probe_j = TinyJit(_fwd3)
_TOKBUF = [None]
def probe(toks, pos):
    dev = Device["NV"]
    if _TOKBUF[0] is None:
        _TOKBUF[0] = Tensor([toks], dtype="int32").contiguous().realize()
    else:
        b = _TOKBUF[0].uop.buf_uop.buffer._bufs["NV"]
        dev.allocator._copyin(b, memoryview(_np.asarray([toks], dtype=_np.int32).tobytes()).cast("B"))
    with Context(JIT=1):
        h = probe_j(_TOKBUF[0], v_sp.bind(pos)).realize()
    return h

def fwd1_eager(tid, pos):
    t = Tensor([[int(tid)]], dtype="int32").contiguous()
    with Context(JIT=2):
        h = _fwd3(t, v_sp.bind(pos)).realize()
    return h

def head_rows(h, n):
    with Context(JIT=2):
        lg = model.output(model.output_norm(h[:, :n, :]).half()).realize()
    _np.linalg.norm  # noqa
    lg_np = lg.numpy()
    return [int(_np.argmax(lg_np[0, j])) for j in range(n)], lg

def embed(tok_id):
    return model.token_embd(Tensor([[int(tok_id)]], dtype="int32")).float().contiguous().realize()

def attn_eager(b, x, pos):
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), v_sp.bind(pos) if not isinstance(pos, UOp) else pos)
    return (hh + b._feed_forward(b.ffn_norm(hh))).contiguous()

def draft_step(pe, hm, pos):
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
    hdj = attn_eager(draft.blk, xin, pos)
    with Context(JIT=2):
        lgj = model.output(draft.head_norm(hdj).half()).realize()
    return int(_np.argmax(lgj.numpy()[0, -1])), hdj[:, -1:, :].contiguous().realize()

def select_states(m):
    sinks = []
    for b in gdns:
        sinks.append(Tensor(b.conv_state.uop.after(b.conv_state.uop.store(
            b.step_conv_buf[m].cast(b.conv_state.dtype).uop))))
        sinks.append(Tensor(b.recurrent_state.uop.after(b.recurrent_state.uop.store(
            b.step_rec_buf[m].cast(b.recurrent_state.dtype).uop))))
    Tensor.realize(*sinks)
    Device["NV"].synchronize()

ids = [0] + tok.encode(CFG.prompt)
t_p = time.perf_counter()
hs = []
for i, tid in enumerate(ids):
    h = fwd1_eager(tid, i)
    hs.append(h)
h_seed = hs[-1].contiguous().realize()
cur = head_rows(hs[-1], 1)[0][0]
zeros = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
if not NO_DRAFT:
    for i, tid in enumerate(ids):
        hm = zeros if i == 0 else hs[i - 1]
        draft_step(embed(tid), hm, i)
print(f"[prefill] {len(ids)} tok, cur={cur}, {time.perf_counter()-t_p:.1f}s", flush=True)

outs = []
pos = len(ids)
n_cyc = 0
probe_t = []
t_all = time.perf_counter()
while len(outs) < NTOK:
    n_cyc += 1
    if NO_DRAFT:
        props = [100, 200]  # garbage — correctness irrelevant, timing only
    else:
        props = []
        pe, hm = embed(cur), h_seed
        for j in range(K):
            pj, hdj = draft_step(pe, hm, pos + j)
            props.append(pj)
            pe, hm = embed(pj), hdj

    t0 = time.perf_counter()
    hA = probe([cur] + props, pos)
    Device["NV"].synchronize()
    probe_t.append(time.perf_counter() - t0)
    if n_cyc >= 2:
        print(f"[cyc {n_cyc}] probe {probe_t[-1]:.3f}s", flush=True)

    if NO_HEAD:
        amds = [cur, 1, 2]
    else:
        amds, _ = head_rows(hA, K + 1)
    m = 0
    for i, p in enumerate(props):
        if amds[i] == p: m = i + 1
        else: break
    bonus = amds[m]

    if not NO_SELECT:
        select_states(m)
    h_seed = hA[:, m:m+1, :].contiguous().realize()
    if m >= 1 and not NO_DRAFT:
        for j in range(m):
            draft_step(embed(props[j]), hA[:, j:j+1, :].contiguous().realize(), pos + 1 + j)

    outs.append(cur)
    outs.extend(props[:m])
    cur = bonus
    pos += m + 1

import statistics
steady = probe_t[1:] if len(probe_t) > 1 else probe_t
print(f"BISECT: NO_DRAFT={NO_DRAFT} NO_HEAD={NO_HEAD} NO_SELECT={NO_SELECT} | "
      f"probe steady median={statistics.median(steady):.3f}s mean={sum(steady)/len(steady):.3f}s", flush=True)
