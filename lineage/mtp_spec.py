# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""MTP speculative decoding v223 — EXACT (60/60 greedy), post-review fixes (D7/D10/D14).

v206 (the ONLY clean exact run): stock block.__call__ (implicit GDN/KV),
per-T TinyJits (T=1 prefill/commit AND T=K+1 probe both live), JIT=2
(no HCQGraph batching), GDN rewind = snapshot + uop.store restore + sequential
T=1 commit through the SAME T=1 jit. Correct text, 16 tokens, no fault.

DO NOT: JIT=1 (timeline hangs on restore), forward_all/call_mtp TinyJits
(T=1-after-T=3 device fault), symbolic-slice jit (replay substitution broken).

Usage: python mtp_spec.py base|spec [N]   Env: MTP_K, MTP_DBG
"""
import sys, json, time, os, struct
import numpy as _np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_config import MTPConfig
if "JIT" not in os.environ:
    os.environ["JIT"] = "2"
CFG = MTPConfig.load()
MODE = sys.argv[1] if len(sys.argv) > 1 else "spec"
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else CFG.n_gen

from tinygrad.llm.model import Transformer, TransformerBlock, Linear, GatedDeltaNetBlock
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT, _GGML_NATIVE
from tinygrad.helpers import GlobalCounters, getenv, Context
from tinygrad.tensor import Tensor
from tinygrad import dtypes, nn
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit

def mark(m):
    if CFG.dbg: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def vram():
    return {k: round(v/1e9, 2) for k, v in GlobalCounters.mem_used_per_device.items()}

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
    from tinygrad.helpers import prod
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

t0 = time.perf_counter()
MODEL_PATH = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
model, kv = Transformer.from_gguf(MODEL_PATH, CFG.max_context)
print(f"[cfg] JIT={getenv('JIT',2)} BEAM={getenv('BEAM',0)} K={CFG.K}", flush=True)
print(f"[load] {time.perf_counter()-t0:.1f}s VRAM={vram()}", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)
ids = [0] + tok.encode(CFG.prompt)
print(f"[prompt] {len(ids)} tokens", flush=True)

if MODE == "base":
    gen = model.generate(list(ids), chunk_size=32, temperature=0.0)
    outs = []
    while len(outs) < N_GEN: outs.append(next(gen))
    json.dump(outs, open(CFG.base_json, "w"))
    print("BASE OK:", " ".join(tok.decode([t]) for t in outs[:40]), flush=True)
    print("DONE_BASE", flush=True); sys.exit(0)

sd = extract_tensors(MODEL_PATH, "blk.64")
cfg = model.blk[-1].config
class Draft: pass
draft = Draft()
draft.blk = TransformerBlock(cfg)
draft.enorm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
draft.hnorm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
draft.eh_proj = Linear(2*cfg.dim, cfg.dim, bias=False)
draft.head_norm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
def w16(name): return sd[f"blk.64.{name}"].cast(dtypes.float16).contiguous()
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
print(f"[draft loaded] VRAM={vram()}", flush=True)

K = CFG.K
v_sp = UOp.variable("mtp_sp", 0, model.max_context - 2 - K)  # probe writes pos..pos+K
gdn_blocks = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
_PROBE_JIT = int(getenv("MTP_PROBE_JIT", 2))   # 2 = graphless (SAFE, proven 60/60); 1 hangs
_HEAD_JIT = int(getenv("MTP_T1_JIT", 2))        # 2 = graphless (safe); 1 breaks with 2 families

def _fwd(tokens, start_pos):
    x = model.token_embd(tokens).float()
    for b in model.blk:
        x = b(x, start_pos)
    return x.contiguous()

_fwd_j = {}
def fwd(toks: list[int], pos: int) -> Tensor:
    T = len(toks)
    key = (T, T == K + 1)
    if key not in _fwd_j:
        _fwd_j[key] = TinyJit(_fwd)
        print(f"[jit] new stock-forward T={T} probe={T == K + 1}", flush=True)
    t = Tensor([toks], dtype="int32").contiguous()
    # JIT level from env: 2 = graphless (SAFE default, proven 60/60); 1 HANGS
    # (two graph families blow the MAP_SYSMEM_FD cap — see AGENTS.md MTP FINAL STATUS)
    with Context(JIT=_PROBE_JIT if T == K + 1 else _HEAD_JIT):
        h = _fwd_j[key](t, v_sp.bind(pos)).realize()
    if int(h.shape[1]) != T:
        raise RuntimeError(f"fwd T={h.shape[1]} expected {T}")
    return h

def head_rows(h: Tensor, n: int):
    # ONE batched GEMM over all rows, then per-row argmax on host
    lg = model.output(model.output_norm(h[:, :n, :]).half()).realize()
    am = lg.argmax(-1).realize()
    return [int(x) for x in am.flatten().tolist()[:n]]

def snap_gdn():
    out = []
    for b in gdn_blocks:
        if not hasattr(b, "conv_state"):
            b._init_state(Tensor.zeros(1, 1, cfg.dim))
        out.append((b.conv_state.contiguous().clone().realize(),
                    b.recurrent_state.contiguous().clone().realize()))
    return out

def restore_gdn(snaps):
    for b, (cv, rs) in zip(gdn_blocks, snaps):
        Tensor(b.conv_state.uop.after(b.conv_state.uop.store(cv.uop))).realize()
        Tensor(b.recurrent_state.uop.after(b.recurrent_state.uop.store(rs.uop))).realize()

def verify_restore(snaps, nblk=2):
    """D9 guard: restore must actually land. CRITICAL LESSON #3 says uop.store
    restores silently no-op under graph capture — this catches that immediately
    instead of letting output quality drift."""
    for b, (cv, _rs) in list(zip(gdn_blocks, snaps))[:nblk]:
        got = b.conv_state.contiguous().realize()
        d = float((got - cv).abs().cast(dtypes.float32).max().item())
        if d > 1e-2:
            raise RuntimeError(f"restore_gdn NO-OP detected (conv_state maxdiff={d:.3f}) — capture semantics changed?")

def attn_eager(b, x, pos):
    """pos int|UOp — ALWAYS pass bound var so kernels cache across positions."""
    b._init_state(x)
    if not isinstance(pos, UOp):
        pos = v_sp.bind(pos)
    hh = x + b._attention(b.attn_norm(x), pos)
    return (hh + b._feed_forward(b.ffn_norm(hh))).contiguous()

def embed(tok_id: int) -> Tensor:
    return model.token_embd(Tensor([[int(tok_id)]], dtype="int32")).float().contiguous().realize()

def draft_step(pe: Tensor, hm: Tensor, pos: int):
    xin = draft.eh_proj(draft.enorm(pe).cat(draft.hnorm(hm), dim=-1))
    hdj = attn_eager(draft.blk, xin, pos)
    lgj = model.output(draft.head_norm(hdj).half())[:, -1:, :]
    return lgj.realize(), hdj.realize()

def run_prefill(ids):
    t_all = time.perf_counter()
    hs = []   # trunk hiddens per position (D8: draft fill uses the TRUNK residual
              # stream per MTPLX contract; in-chain draft-hidden is only for
              # speculative positions, which the prompt fill has none of)
    for i, tid in enumerate(ids):
        h = fwd([tid], i)
        hs.append(h[:, -1:, :].contiguous().realize())
        if (i + 1) in (1, len(ids)) or (i + 1) % 4 == 0:
            print(f"[prefill] {i+1}/{len(ids)} {time.perf_counter()-t_all:.1f}s VRAM={vram()}", flush=True)
    cur = head_rows(hs[-1], 1)[0]
    zeros = Tensor.zeros(1, 1, cfg.dim, dtype=dtypes.float32).contiguous().realize()
    for i, tid in enumerate(ids):
        hm = zeros if i == 0 else hs[i - 1]   # h_{i-1} = trunk hidden after pos i-1
        draft_step(embed(tid), hm, i)
    print(f"[prefill done] cur={cur} {time.perf_counter()-t_all:.1f}s VRAM={vram()}", flush=True)
    return hs[-1], cur

def spec_generate(ids, h, cur, stats):
    pos = len(ids)
    h_seed = h[:, -1:, :].contiguous().realize()
    n_acc = n_prop = n_cyc = 0
    t_d = t_p = t_h = t_c = t_r = 0.0
    while True:
        n_cyc += 1
        t_cyc = time.perf_counter()
        props = []
        td0 = time.perf_counter()
        pe = embed(cur)
        hm = h_seed
        for j in range(K):
            lgj, hdj = draft_step(pe, hm, pos + j)
            props.append(int(lgj.flatten().argmax().item()))
            pe = embed(props[-1])
            hm = hdj[:, -1:, :].contiguous().realize()
        t_d += time.perf_counter() - td0
        mark(f"props={props} pos={pos}")

        snap = snap_gdn()
        tp0 = time.perf_counter()
        hA = fwd([cur] + props, pos)
        t_p += time.perf_counter() - tp0
        th0 = time.perf_counter()
        amds = head_rows(hA, K + 1)
        t_h += time.perf_counter() - th0
        mark(f"amd={amds}")

        m = 0
        for i, p in enumerate(props):
            if amds[i] == p: m = i + 1
            else: break
        bonus = amds[m]
        n_acc += m
        n_prop += K

        tc0 = time.perf_counter()
        if m == K:
            hC = hA
        else:
            restore_gdn(snap)
            if n_cyc == 1 or getenv("MTP_CHECK_RESTORE", "") == "all":
                verify_restore(snap)
            h_last = None
            seq = [cur] + props[:m]
            for j, tid in enumerate(seq):
                h_last = fwd([tid], pos + j)
            hC = h_last
        t_c += time.perf_counter() - tc0
        h_seed = hC[:, -1:, :].contiguous().realize()

        tr0 = time.perf_counter()
        if m >= 1:
            # hA[:, j] IS the trunk hidden after accepted-prefix position j (probe ran
            # the same tokens) — hC is only T=1 deep for m<K and would empty-slice at j>=1
            for j in range(m):
                draft_step(embed(props[j]), hA[:, j:j+1, :].contiguous().realize(), pos + 1 + j)
        t_r += time.perf_counter() - tr0

        pos += m + 1
        dt = time.perf_counter() - t_cyc
        rate = (n_acc / n_prop) if n_prop else 0.0
        stats.update(acc=rate, n_acc=n_acc, n_prop=n_prop, n_cyc=n_cyc)
        print(f"[cyc {n_cyc}] m={m}/{K} props={props} amd={amds} bonus={bonus} pos={pos} {dt*1e3:.0f}ms "
              f"(d{t_d/n_cyc*1e3:.0f} p{t_p/n_cyc*1e3:.0f} h{t_h/n_cyc*1e3:.0f} c{t_c/n_cyc*1e3:.0f} r{t_r/n_cyc*1e3:.0f}) "
              f"acc={rate:.2f} ({n_acc}/{n_prop})", flush=True)
        if CFG.accept_log:
            with open(CFG.accept_log, "a") as af:
                af.write(json.dumps({"cyc": n_cyc, "pos": pos, "m": m, "props": props, "amd": amds, "ms": dt*1e3})+"\n")
        for tkn in [cur] + props[:m]:
            yield tkn
        cur = bonus

mark("running prefill")
h, cur = run_prefill(ids)
STATS = {}
yield_gen = spec_generate(ids, h, cur, STATS)
outs = []
t0 = time.perf_counter(); n_tok = 0
try:
    for t in yield_gen:
        outs.append(t); n_tok += 1
        dt = time.perf_counter()-t0
        print(f"tok {n_tok:3d}: {dt/n_tok*1e3:7.1f} ms/tok ({n_tok/dt:.2f} tok/s) last={outs[-1]}", flush=True)
        if n_tok >= N_GEN: break
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"\n[RUNTIME STOP] {type(e).__name__}: {str(e)[:200]}", flush=True)
dt = time.perf_counter()-t0
print(f"\n== SPEC RESULT ==", flush=True)
if n_tok:
    print(f"decode: {n_tok/dt:.2f} tok/s ({dt/n_tok*1e3:.1f} ms/tok) n={n_tok}", flush=True)
if STATS.get("n_prop"):
    import math
    p, n = STATS["acc"], STATS["n_prop"]
    ci = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0
    print(f"acceptance: {p:.2f} ({STATS['n_acc']}/{STATS['n_prop']} proposals, K={K}, 95% CI ±{ci:.2f})", flush=True)
print("generated:", " ".join(tok.decode([t]) for t in outs), flush=True)
try:
    base = json.load(open(CFG.base_json))
    match = sum(1 for a, b in zip(base, outs) if a == b)
    print(f"GREEDY MATCH: {match}/{min(len(base), len(outs))}", flush=True)
except FileNotFoundError:
    print("GREEDY MATCH: no baseline file", flush=True)
json.dump(outs, open(CFG.out_json, "w"))
print("DONE_SPEC", flush=True)
