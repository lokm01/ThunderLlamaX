# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, json, time
import numpy as np
import jinja2
from tinygrad.llm.model import Transformer, TransformerBlock, Linear, GatedDeltaNetBlock
from tinygrad.nn import RMSNorm
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT, _GGML_NATIVE
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad import dtypes, nn
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import TinyJit

MODE = sys.argv[1] if len(sys.argv)>1 else "spec"
N_GEN = int(sys.argv[2]) if len(sys.argv)>2 else 60
K = int(getenv("SPEC_K", "4"))
f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter

def mark(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------------- targeted GGUF extraction (blk.64 only) ----------------
import struct
def _rstr(r):
    n=struct.unpack("<Q",r.read(8))[0]; return r.read(n).decode()
def _rd_val(r, typ):
    if typ==8: return _rstr(r)
    if typ==9:
        it=struct.unpack("<I",r.read(4))[0]; n=struct.unpack("<Q",r.read(8))[0]
        return [_rd_val(r,it) for _ in range(n)]
    fmt={0:"c",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[typ]
    nb=struct.calcsize("<"+fmt)
    return struct.unpack("<"+fmt, r.read(nb))[0]
def extract_tensors(path, prefix):
    r=open(path,"rb"); assert r.read(4)==b"GGUF"
    struct.unpack("<I",r.read(4)); nt=struct.unpack("<Q",r.read(8))[0]; nk=struct.unpack("<Q",r.read(8))[0]
    align=32
    for _ in range(nk):
        k=_rstr(r); t=struct.unpack("<I",r.read(4))[0]; v=_rd_val(r,t)
        if k=="general.alignment": align=int(v)
    infos=[]
    for _ in range(nt):
        nm=_rstr(r); nd=struct.unpack("<I",r.read(4))[0]
        dims=tuple(struct.unpack("<Q",r.read(8))[0] for _ in range(nd))
        typ=struct.unpack("<I",r.read(4))[0]; off=struct.unpack("<Q",r.read(8))[0]
        infos.append((nm,dims,typ,off))
    data_start=(r.tell()+align-1)//align*align
    out={}
    from tinygrad.helpers import prod as _prod
    for nm,dims,typ,off in infos:
        if not nm.startswith(prefix): continue
        n=_prod(dims)
        if typ in _GGML_NATIVE: nbytes=_GGML_NATIVE[typ].itemsize*n
        else:
            ne,nb=_GGML_QUANT[typ]; nbytes=(n//ne)*nb
        r.seek(data_start+off); raw=r.read(nbytes)
        t8=Tensor(np.frombuffer(raw,np.uint8).copy())
        out[nm]=ggml_data_to_tensor(t8, n, typ).reshape(*reversed(dims))
    r.close()
    return out

# ---------------- load main model ----------------
t0=pc()
model, kv = Transformer.from_gguf(f, 1024)
print(f"[load] {pc()-t0:.1f}s", flush=True)
tok = SimpleTokenizer.from_gguf_kv(kv)

prompt="The theory of relativity transformed our understanding of space and time."
ids=[0]+tok.encode(prompt)

# ---------------- draft module (blk.64 nextn) ----------------
sd = extract_tensors(f, "blk.64")
cfg = model.blk[-1].config
class Draft: pass
draft = Draft()
draft.blk = TransformerBlock(cfg)
draft.enorm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
draft.hnorm = nn.RMSNorm(cfg.dim, cfg.norm_eps)
draft.eh_proj = Linear(2*cfg.dim, cfg.dim, bias=False)
draft.head_norm = RMSNorm(cfg.dim, cfg.norm_eps)
def w(name): return sd[f"blk.64.{name}"]
remap={
 "blk.attn_norm.weight": w("attn_norm.weight"),
 "blk.attn_q.weight": w("attn_q.weight"), "blk.attn_k.weight": w("attn_k.weight"),
 "blk.attn_v.weight": w("attn_v.weight"), "blk.attn_output.weight": w("attn_output.weight"),
 "blk.attn_q_norm.weight": w("attn_q_norm.weight"), "blk.attn_k_norm.weight": w("attn_k_norm.weight"),
 "blk.ffn_norm.weight": w("post_attention_norm.weight"),
 "blk.ffn_gate.weight": w("ffn_gate.weight"), "blk.ffn_up.weight": w("ffn_up.weight"),
 "blk.ffn_down.weight": w("ffn_down.weight"),
 "enorm.weight": sd["blk.64.nextn.enorm.weight"],
 "hnorm.weight": sd["blk.64.nextn.hnorm.weight"],
 "head_norm.weight": sd["blk.64.nextn.shared_head_norm.weight"],
 "eh_proj.weight": sd["blk.64.nextn.eh_proj.weight"],
}
for name, ten in remap.items():
    parts = name.split(".")
    obj = draft
    for pp in parts[:-1]: obj = getattr(obj, pp)
    setattr(obj, parts[-1], ten.cast(dtypes.float16).contiguous())
del sd, remap
import gc; gc.collect()
print("[draft loaded]", flush=True)

# ---------------- jits ----------------
v_sp_p = UOp.variable("sp_prefill", 0, model.max_context-2)
v_sp_v = UOp.variable("sp_verify", 0, model.max_context-2)
v_sp_d = UOp.variable("sp_draft", 0, model.max_context-2)
v_tok_n = UOp.variable("tokn", 1, 33)

def dfwd(tok_t, h_prev, start_pos):
    e = model.token_embd(tok_t).float()
    xin = draft.eh_proj(draft.enorm(e).cat(draft.hnorm(h_prev), dim=-1))
    hd = draft.blk(xin, start_pos)   # native FFNBlock.__call__ (attn block)
    return model.output(draft.head_norm(hd).half())[:, -1:, :], hd[:, -1:, :]

prefill_jit = TinyJit(lambda t, sp, *st: model.forward_all(t, sp, st))
probe_jit   = TinyJit(lambda t, sp, *st: model.forward_all(t, sp, st))
draft_jit   = TinyJit(dfwd)

# batched draft prompt-fill: hnorm = main hidden BEFORE each position (first uses zeros)
def _prompt_fill_body(tok_ids, start_pos, h_main_win, hm_first):
    x = model.token_embd(tok_ids).float()          # (1,T,D)
    h_prev = None
    for i in range(tok_ids.shape[1]):              # i = position k
        hm = hm_first if i == 0 else h_main_win[:, i-1:i, :]
        e = x[:, i:i+1, :]
        xin = draft.eh_proj(draft.enorm(e).cat(draft.hnorm(hm), dim=-1))
        hd = draft.blk(xin, start_pos+i)
        h_prev = hd
    return h_prev
prompt_fill_jit = TinyJit(lambda tokids, sp, hmwin, hm0: _prompt_fill_body(tokids, sp, hmwin, hm0))

def head_last(h):
    # cast ACTIVATION to fp16 (weight already fp16) -> pure fp16 GEMM, no 5GB fp32 weight cast
    return model.output(model.output_norm(h).half())

def states_zeros():
    b0=None
    for b in model.blk:
        if isinstance(b, GatedDeltaNetBlock):
            b0=b
            if not hasattr(b,"conv_state"): b._init_state(Tensor.zeros(1,1,cfg.dim))
    n=sum(1 for b in model.blk if isinstance(b, GatedDeltaNetBlock))
    cv=Tensor.zeros(n, *b0.conv_state.shape, dtype=b0.conv_state.dtype)
    rs=Tensor.zeros(n, *b0.recurrent_state.shape, dtype=b0.recurrent_state.dtype)
    return cv, rs

def run_prefill(ids):
    CH=32; pos=0
    t = Tensor(ids + [0]*(model.max_context-len(ids)), dtype="int32").reshape(1, model.max_context)
    lg_last=None; states=states_zeros()
    while pos < len(ids):
        n=min(CH, len(ids)-pos)
        h,cv,rs = prefill_jit(t[:, pos:pos+n].contiguous(), v_sp_p.bind(pos), *states)
        h,cv,rs = h.realize(), cv.realize(), rs.realize()
        states=(cv,rs)
        pos+=n; lg_last=head_last(h[:, -1:, :]).realize()
    # draft KV prefill: feed every prompt token through the draft block at its position,
    # with hnorm input = the main hidden BEFORE that position (MTP semantics).
    # draft dh chain starts from the first main hidden.
    # batched draft KV fill covering positions 0..T-1 INCLUDING the first gen token (cur).
    # MTP at position k: inputs (e(t_k), hnorm(h_main[k-1])); position 0 hnorm = zeros.
    hh = h
    cur = int(lg_last[0,-1].argmax().item())
    tlen = len(ids)
    tokwin = Tensor([ids+[cur]], dtype="int32")
    hm_zero = hh[:, :1, :] * 0.0
    dh2 = prompt_fill_jit(tokwin, v_sp_d.bind(0), hh, hm_zero)
    dh2 = dh2.realize()
    if getenv("MTPDBG"): print(f"[prefill] draft KV filled for positions 0..{tlen} (incl cur={cur})", flush=True)
    return h, cur, (states[0].contiguous().clone(), states[1].contiguous().clone())

def spec_generate(ids):
    h, cur, states = run_prefill(list(ids))
    pos = len(ids)
    h_seed = h[:, -1:, :].contiguous().clone()
    if getenv("MTPDBG"):
        print(f"[start] cur={cur} (baseline first-gen should be 1049)", flush=True)
        def dprobe(prev_tok, hm, sp, tag):
            xin = draft.eh_proj(draft.enorm(model.token_embd(Tensor([[prev_tok]],dtype="int32")).float()).cat(draft.hnorm(hm), dim=-1))
            hd = draft.blk(xin, v_sp_d.bind(sp))
            lg = model.output(draft.head_norm(hd).half()).realize()
            v,i = lg[0,-1].topk(3)
            print(f"[dp] {tag}: top3={[int(x) for x in i.tolist()]}", flush=True)
        t13 = ids[len(ids)-1]
        dprobe(cur,   h[:, -1:, :], 14, "e(cur)+h13")
        dprobe(cur,   h[:, -2:-1, :], 14, "e(cur)+h12")
        dprobe(t13,   h[:, -1:, :], 14, "e(t13)+h13")
        dprobe(t13,   h[:, -2:-1, :], 14, "e(t13)+h12")
        dprobe(cur,   h[:, -1:, :], 13, "e(cur)+h13@sp13")
        # quantitative sensitivity: how much do h-inputs move the logits?
        import numpy as _np
        def lgvec(prev_tok, hm):
            xin = draft.eh_proj(draft.enorm(model.token_embd(Tensor([[prev_tok]],dtype="int32")).float()).cat(draft.hnorm(hm), dim=-1))
            hd = draft.blk(xin, v_sp_d.bind(14))
            return model.output(draft.head_norm(hd).half()).realize()[0,-1,:]
        l_base   = lgvec(cur, h[:, -2:-1, :])
        l_h13    = lgvec(cur, h[:, -1:, :])
        l_t13    = lgvec(t13, h[:, -2:-1, :])
        d_h   = float((l_base.float()-l_h13.float()).square().sum().sqrt().item())
        d_tok = float((l_base.float()-l_t13.float()).square().sum().sqrt().item())
        nrm   = float(l_base.float().square().sum().sqrt().item())
        hn_out = draft.hnorm(h[:, -2:-1, :]).float()
        hn2   = draft.hnorm(h[:, -1:, :]).float()
        d_hn  = float((hn_out-hn2).square().sum().sqrt().item())
        csim  = float((hn_out.flatten()*hn2.flatten()).sum()/(hn_out.square().sum().sqrt()*hn2.square().sum().sqrt()).item())
        hnm   = float(hn_out.abs().max().item())
        print(f"[sens] ||L||={nrm:.2f}  d(L,h12->h13)={d_h:.4f}  d(L,t13->cur)={d_tok:.2f}  ratio_h/tok={d_h/max(d_tok,1e-9):.4f}", flush=True)
        print(f"[sens] ||hnorm(h12)-hnorm(h13)||={d_hn:.4f} cossim={csim:.4f} hnorm_absmax={hnm:.4f}", flush=True)
        print(f"[sens] h12[:4]={[[round(float(x),3) for x in h[0,-2,:4].tolist()]]} h13[:4]={[[round(float(x),3) for x in h[0,-1,:4].tolist()]]}", flush=True)
        print(f"[sens] hnorm.w[:4]={[[round(float(x),4) for x in draft.hnorm.weight[:4].tolist()]]} dtype={draft.hnorm.weight.dtype}", flush=True)
        hn_out = draft.hnorm(h[:, -1:, :]).float()
        en_out = draft.enorm(model.token_embd(Tensor([[cur]],dtype="int32")).float()).float()
        print(f"[diag] h_seed[:3]={[[round(float(x),3) for x in h[0,0,:3].tolist()]]}", flush=True)
        print(f"[diag] hnorm(h_seed) absmax={hn_out.abs().max().item():.4f} sumsq={float((hn_out**2).sum()):.4f}", flush=True)
        print(f"[diag] enorm(e_cur) absmax={en_out.abs().max().item():.4f}", flush=True)
        print(f"[diag] hnorm.w absmax={draft.hnorm.weight.abs().max().item():.4f} mean={draft.hnorm.weight.float().mean().item():.4f} shape={tuple(draft.hnorm.weight.shape)}", flush=True)
        print(f"[diag] enorm.w absmax={draft.enorm.weight.abs().max().item():.4f} shape={tuple(draft.enorm.weight.shape)}", flush=True)
        print(f"[diag] eh_proj.w shape={tuple(draft.eh_proj.weight.shape)} absmax={draft.eh_proj.weight.abs().max().item():.4f} dtype={draft.eh_proj.weight.dtype}", flush=True)
        # half-split stats of eh_proj output: first half (e-cols) vs second half (h-cols) contribution
        W = draft.eh_proj.weight.float()   # (5120, 10240)
        e_contrib = (W[:, :5120].abs().mean()).item()
        h_contrib = (W[:, 5120:].abs().mean()).item()
        print(f"[diag] eh_proj col-half mean|W|: e-half={e_contrib:.4f} h-half={h_contrib:.4f}", flush=True)
    yield cur
    n_accept=[0,0]  # accepted, cycles
    while True:
        # --- draft K proposals ---
        props=[]; hd=h_seed; prev=Tensor([[cur]], dtype="int32")
        for j in range(K):
            lgj, hd = draft_jit(prev, hd, v_sp_d.bind(pos+j))
            lgj, hd = lgj.realize(), hd.realize()
            pj = int(lgj[0,-1].argmax().item())
            if getenv("MTPDBG") and j==0:
                vals, idx5 = lgj[0,-1].topk(5)
                print(f"[drift] draft@pos{pos} top5={[int(x) for x in idx5.tolist()]}", flush=True)
                # batch comparison: extend the prompt window with cur (positions 0..14 in ONE call)
                ext_tok = Tensor([ids+[cur]], dtype="int32")
                hext = h.cat(h_seed, dim=1)   # (1,15,D): h[0..13] + h_seed(h[13])? need h_main for pos14 = h_seed
                # batch-fill body needs (tok, sp, h_main_win, hm0); run eagerly for ONE more position
                btok = Tensor([[cur]], dtype="int32")
                hm_prev2 = h_seed
                xin2 = draft.eh_proj(draft.enorm(model.token_embd(btok).float()).cat(draft.hnorm(hm_prev2), dim=-1))
                hd2 = draft.blk(xin2, v_sp_d.bind(pos))
                lg2 = model.output(draft.head_norm(hd2).half()).realize()
                v2, i2 = lg2[0,-1].topk(5)
                print(f"[drift2] eager draft@pos{pos} top5={[int(x) for x in i2.tolist()]}", flush=True)
            hd = hd.contiguous().clone()
            props.append(pj); prev = Tensor([[pj]], dtype="int32")
        if getenv("MTPDBG"): print(f"[sp] props={props}", flush=True)
        # --- probe [cur, p1..pK] at pos ---
        tv = Tensor([[cur]+props], dtype="int32").contiguous()
        if getenv("MTPDBG"): print(f"[sp] probe start pos={pos} tv={tv.shape}", flush=True)
        hA,cvA,rsA = probe_jit(tv, v_sp_v.bind(pos), *states)
        if getenv("MTPDBG"): print("[sp] probe returned", flush=True)
        hA,cvA,rsA = hA.realize(), cvA.realize(), rsA.realize()
        if getenv("MTPDBG"): print("[sp] probe realized", flush=True)
        stA=(cvA,rsA)
        if getenv("MTPDBG"): print("[sp] heads", flush=True)
        amd=[int(head_last(hA[:, i:i+1, :]).argmax().item()) for i in range(K+1)]
        if getenv("MTPDBG"): print(f"[sp] amd={amd}", flush=True)
        m=0
        for i,p in enumerate(props):
            if amd[i]==p: m=i+1
            else: break
        bonus=amd[m]
        n_accept[0]+=m; n_accept[1]+=1
        if m == K:
            states = (stA[0].contiguous().clone(), stA[1].contiguous().clone())
        else:
            # commit pass: re-run accepted prefix only -> exact states at pos+m+1
            ct = Tensor([[cur]+props[:m]], dtype="int32").contiguous()
            if getenv("MTPDBG"): print("[sp] commit", flush=True)
            _,cvC,rsC = probe_jit(ct, v_sp_v.bind(pos), *states)
            cvC,rsC = cvC.realize(), rsC.realize()
            if getenv("MTPDBG"): print("[sp] committed", flush=True)
            stC=(cvC,rsC)
            states = (stC[0].contiguous().clone(), stC[1].contiguous().clone())
        h_seed = hA[:, m:m+1, :].contiguous().clone()
        pos += m+1
        if n_accept[1]%5==0:
            print(f"  [cycle] pos={pos} m={m} avg_acc={n_accept[0]/max(1,n_accept[1]):.2f}", flush=True)
        for tkn in props[:m]:
            cur = tkn; yield tkn
        cur = bonus; yield bonus

if MODE=="base":
    gen=model.generate(list(ids), chunk_size=32, temperature=0.0)
    outs=[]
    while len(outs)<N_GEN: outs.append(next(gen))
    open("/tmp/spec_base.json","w").write(json.dumps(outs))
    print("BASE OK:", " ".join(tok.decode([t]) for t in outs[:40]), flush=True)
    print("DONE_BASE", flush=True); sys.exit(0)

gen = spec_generate(ids)
outs=[]
t0=pc(); n_tok=0
for t in gen:
    outs.append(t); n_tok+=1
    if n_tok%10==0:
        dt=pc()-t0
        print(f"tok {n_tok:3d}: {dt/n_tok*1e3:7.1f} ms/tok ({n_tok/dt:.2f} tok/s)", flush=True)
    if n_tok>=N_GEN: break
dt=pc()-t0
print(f"\n== SPEC RESULT ==\ndecode: {n_tok/dt:.2f} tok/s ({dt/n_tok*1e3:.1f} ms/tok)", flush=True)
print("generated:", " ".join(tok.decode([t]) for t in outs), flush=True)
try:
    base=json.loads(open("/tmp/spec_base.json").read())
    match=sum(1 for a,b in zip(base,outs) if a==b)
    print(f"GREEDY MATCH: {match}/{min(len(base),len(outs))}", flush=True)
except FileNotFoundError: pass
print("DONE_SPEC", flush=True)
