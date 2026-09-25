# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Draft-step position ladder: exact sp where the draft jit faults.
Uses mtp_v3's own draft loading (verbatim)."""
import os, sys, time, struct
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
import numpy as _np
L = int(os.getenv("REPRO_L", "2048"))
os.environ.setdefault("MTP_T3_LAZY", "1")

from mtp_config import MTPConfig
CFG = MTPConfig.load()
from tinygrad.llm.model import Transformer
from tinygrad.tensor import Tensor
from tinygrad import dtypes, nn
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device
from tinygrad.helpers import Context, prod
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_NATIVE, _GGML_QUANT
import tinygrad.llm.model as M

MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
model, kv = Transformer.from_gguf(MODEL, L)
cfg = model.blk[-1].config
print(f"[load] maxctx={L} cfg.max_context={cfg.max_context}", flush=True)

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
    r.close(); return out

sd = extract_tensors(MODEL, "blk.64")
class Draft: pass
draft = Draft()
draft.blk = M.TransformerBlock(cfg)
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

v_sp = UOp.variable("rsp", 0, L - 3)
for b in model.blk: b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()

def _draft_fwd(tid_t, hm_t, sp):
    pe = model.token_embd(tid_t).float()
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm_t), dim=-1))
    b = draft.blk
    x = xin
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), sp)
    hd = hh + b._feed_forward(b.ffn_norm(hh))
    lg = model.output(draft.head_norm(hd).half())[:, -1:, :]
    return lg.argmax(-1), hd[:, -1:, :]

draft_j = TinyJit(_draft_fwd)
_TID = Tensor([[1]], dtype=dtypes.int32).contiguous().realize()
_HM = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
dev = Device["NV"]

pos = int(os.getenv("REPRO_START", "900"))
t0 = time.perf_counter()
while pos < L - 2:
    dev.allocator._copyin(_TID.uop.buf_uop.buffer._bufs["NV"], memoryview(_np.asarray([[1]], dtype=_np.int32).tobytes()).cast("B"))
    with Context(JIT=2):
        am, hd = draft_j(_TID, _HM, v_sp.bind(pos))
        am = am.contiguous().realize(); hd = hd.contiguous().realize()
    dev.synchronize()
    print(f"[pos {pos}] ok ({time.perf_counter()-t0:.1f}s)", flush=True)
    pos += 8
print("COMPLETED", flush=True)
