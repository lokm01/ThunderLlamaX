# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Instrument draft_step: where do the 98ms go? Times each sub-operation."""
import os, sys, time
import numpy as np
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")

from tinygrad.llm.model import Transformer, TransformerBlock
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad import nn, dtypes
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.helpers import Context, getenv
from mtp_config import MTPConfig
import struct
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_NATIVE, _GGML_QUANT

CFG = MTPConfig.load()
MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

model, kv = Transformer.from_gguf(MODEL, CFG.max_context)
cfg = model.blk[-1].config
print("[load] ok", flush=True)

# draft setup (same as mtp_v3)
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
        out[nm] = ggml_data_to_tensor(Tensor(np.frombuffer(raw, np.uint8).copy()), n, typ).reshape(*reversed(dims))
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
print("[draft loaded]", flush=True)

v_sp = UOp.variable("mtp_sp", 0, CFG.max_context - 4)

def embed(tok_id):
    return model.token_embd(Tensor([[int(tok_id)]], dtype="int32")).float().contiguous().realize()

def attn_eager(b, x, pos):
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), v_sp.bind(pos) if not isinstance(pos, UOp) else pos)
    return (hh + b._feed_forward(b.ffn_norm(hh))).contiguous()

# create inputs
pe = embed(1049)
hm = Tensor.rand(1, 1, cfg.dim).contiguous().realize()
pos = 15


def _full_step(pe, hm, pos):
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
    hdj = attn_eager(draft.blk, xin, pos)
    lg = model.output(draft.head_norm(hdj).half()).realize()
    return lg, hdj[:, -1:, :].contiguous().realize()

def _full_step_np(pe, hm, pos):
    lg, _ = _full_step(pe, hm, pos)
    return lg.numpy()

# WARMUP: run each component twice (first = compile, second = cached)
print("\n=== WARMUP (compile) ===", flush=True)
for tag, fn in [
    ("embed", lambda: embed(1049)),
    ("enorm", lambda: draft.enorm(pe).realize()),
    ("hnorm", lambda: draft.hnorm(hm).realize()),
    ("cat", lambda: draft.enorm(pe).cat(draft.hnorm(hm), dim=-1).realize()),
    ("eh_proj", lambda: draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1)).realize()),
    ("attn_eager", lambda: attn_eager(draft.blk, draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1)), pos).realize()),
    ("head_norm", lambda: draft.head_norm(attn_eager(draft.blk, draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1)), pos)).half().realize()),
    ("output(lm_head)", lambda: model.output(draft.head_norm(attn_eager(draft.blk, draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1)), pos)).half()).realize()),
    ("FULL draft_step", lambda: _full_step(pe, hm, pos)),
]:
    t0 = time.perf_counter()
    fn()
    Device["NV"].synchronize()
    print(f"  {tag}: {time.perf_counter()-t0:.3f}s", flush=True)

# TIMED: each component individually, 5 reps
print("\n=== TIMED (median of 5) ===", flush=True)
x_in = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1)).contiguous().realize()
hdj = attn_eager(draft.blk, x_in, pos).contiguous().realize()
hn = draft.head_norm(hdj).half().contiguous().realize()

for tag, fn, n in [
    ("embed(1049)", lambda: embed(1049), 5),
    ("enorm(pe)", lambda: draft.enorm(pe).realize(), 20),
    ("hnorm(hm)", lambda: draft.hnorm(hm).realize(), 20),
    ("enorm.cat(hnorm)", lambda: draft.enorm(pe).cat(draft.hnorm(hm), dim=-1).realize(), 20),
    ("eh_proj(cat)", lambda: draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1)).realize(), 20),
    ("attn_norm(x_in)", lambda: draft.blk.attn_norm(x_in).realize(), 20),
    ("_attn(an)", lambda: draft.blk._attention(draft.blk.attn_norm(x_in), v_sp.bind(pos)).realize(), 10),
    ("full_attn_eager", lambda: attn_eager(draft.blk, x_in, pos).realize(), 10),
    ("head_norm(hdj).half", lambda: draft.head_norm(hdj).half().realize(), 20),
    ("lm_head(hn)", lambda: model.output(hn).realize(), 10),
    ("numpy_sync", lambda: model.output(hn).realize().numpy(), 10),
    ("FULL_step", lambda: _full_step_np(pe, hm, pos), 5),
]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        Device["NV"].synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"  {tag}: median={times[n//2]*1e3:.2f}ms  min={times[0]*1e3:.2f}ms", flush=True)

print("\nDONE", flush=True)
