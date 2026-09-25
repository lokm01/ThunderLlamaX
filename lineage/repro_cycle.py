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

v_sp = UOp.variable("rsp", 0, L - 5)
for b in model.blk: b._init_state(Tensor.zeros(1, 1, cfg.dim))
Device["NV"].synchronize()

from tinygrad.llm.model import flush_step_states
def _fwd3(tokens, start_pos, want_logits=False):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    h = flush_step_states(x.contiguous(), model.blk).contiguous()
    if want_logits: return h, model.output(model.output_norm(h).half()).argmax(-1)
    return h

probe_j = TinyJit(_fwd3)
_TOK = Tensor([[1, 2, 3]], dtype=dtypes.int32).contiguous().realize()
dev = Device["NV"]
import os as _os
_os.environ["MTP_PROBE_RO"] = "1"

pos0 = int(_os.getenv("REPRO_START", "1030"))
t0 = time.perf_counter()
# fast-advance states to pos0 via T=8 chunk forwards (PROBE_RO off = chunk semantics)
_os.environ.pop("MTP_PROBE_RO", None)
if _os.getenv("REPRO_ADV_NOSTASH"): _os.environ.pop("MTP_STEP_STATES", None)
sp = 0
while sp < pos0:
    Tn = min(8, pos0 - sp)
    toks = Tensor.zeros(1, Tn, dtype=dtypes.int32).contiguous().realize()
    dev.allocator._copyin(toks.uop.buf_uop.buffer._bufs["NV"], memoryview(_np.zeros((1, Tn), dtype=_np.int32).tobytes()).cast("B"))
    with Context(JIT=2):
        _hl = _fwd3(toks, v_sp.bind(sp)).realize()
    dev.synchronize()
    # mimic mtp chunk fills: realized h_last slices + eager draft steps
    zeros = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    for j in range(Tn):
        if sp + j == 0: hm = zeros
        elif j == 0: hm = hid_prev
        else: hm = _hl[:, j-1:j, :].contiguous().realize()
        pe = model.token_embd(Tensor([[1]], dtype=dtypes.int32)).float().contiguous().realize()
        xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
        b_ = draft.blk
        b_._init_state(xin)
        hh = xin + b_._attention(b_.attn_norm(xin), v_sp.bind(sp + j))
        hdj = (hh + b_._feed_forward(b_.ffn_norm(hh))).contiguous()
        lgj = model.output(draft.head_norm(hdj).half()).realize()
    hid_prev = _hl[:, -1:, :].contiguous().realize()
    sp += Tn
    if sp % 256 == 0: print(f"[advance {sp}] {time.perf_counter()-t0:.1f}s", flush=True)
_os.environ["MTP_PROBE_RO"] = "1"
if _os.getenv("REPRO_ADV_NOSTASH"): _os.environ["MTP_STEP_STATES"] = "1"
print(f"[advanced to {pos0}] {time.perf_counter()-t0:.1f}s", flush=True)

# real cycle phase: draft K=2 -> probe -> select -> h_seed, repeating
def attn_eager(b, x, pos):
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), v_sp.bind(pos) if not isinstance(pos, UOp) else pos)
    return (hh + b._feed_forward(b.ffn_norm(hh))).contiguous()
def _draft_fwd(tid_t, hm_t, spv):
    pe = model.token_embd(tid_t).float()
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm_t), dim=-1))
    b = draft.blk
    x = xin
    b._init_state(x)
    hh = x + b._attention(b.attn_norm(x), spv)
    hd = hh + b._feed_forward(b.ffn_norm(hh))
    lg = model.output(draft.head_norm(hd).half())[:, -1:, :]
    return lg.argmax(-1), hd[:, -1:, :]
draft_j2 = TinyJit(_draft_fwd)
_TID = Tensor([[1]], dtype=dtypes.int32).contiguous().realize()
_HM = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
gdns = [b for b in model.blk if isinstance(b, M.GatedDeltaNetBlock)]
def select_states(m):
    sinks = []
    for b in gdns:
        sinks.append(Tensor(b.conv_state.uop.after(b.conv_state.uop.store(b.step_conv_buf[m].cast(b.conv_state.dtype).uop))))
        sinks.append(Tensor(b.recurrent_state.uop.after(b.recurrent_state.uop.store(b.step_rec_buf[m].cast(b.recurrent_state.dtype).uop))))
    Tensor.realize(*sinks)
    dev.synchronize()

pos = pos0
cyc = 0
while pos < L - 6:
    cyc += 1
    hm0 = _HM
    props = []
    import os as _os2
    for j in range(0 if _os2.getenv("REPRO_NODRAFT") else 2):
        dev.allocator._copyin(_TID.uop.buf_uop.buffer._bufs["NV"], memoryview(_np.asarray([[1]], dtype=_np.int32).tobytes()).cast("B"))
        with Context(JIT=2):
            am2, hd2 = draft_j2(_TID, hm0, v_sp.bind(pos + j))
            am2 = am2.contiguous().realize(); hd2 = hd2.contiguous().realize()
        props.append(int(am2.numpy()[0, -1]))
        hm0 = hd2
    dev.allocator._copyin(_TOK.uop.buf_uop.buffer._bufs["NV"], memoryview(_np.asarray([[1] + props], dtype=_np.int32).tobytes()).cast("B"))
    with Context(JIT=1):
        ret = probe_j(_TOK, v_sp.bind(pos), True)
        hA = ret[0].contiguous().realize(); amA = ret[1].contiguous().realize()
    dev.synchronize()
    amds = [int(x) for x in amA.numpy()[0]]
    m = 0
    for i, p in enumerate(props):
        if amds[i] == p: m = i + 1
        else: break
    if m < 2: select_states(m)
    h_seed = hA[:, m:m+1, :].contiguous().realize()
    print(f"[cyc {cyc} pos {pos}] m={m} props={props} amd={amds[:3]} ({time.perf_counter()-t0:.1f}s)", flush=True)
    pos += m + 1
print("COMPLETED", flush=True)
