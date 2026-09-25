# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c rung: the M=128 trunk (PF_M128=1). 128-row chunks; GEMMs = the proven
m64 cubins at 2 M-blocks (the M-grid: mb = blockIdx.x / NGRID reads x/out rows
mb*MTILE — one launch per class per chunk); scan = WY-C32 NC=4 (one triple per
128-row chunk); attention = ONE w128h window (ROWS=128 HRP=1 — per-row math
VERBATIM from t32/w64, causal-mask law: a 128-row window == 2 concatenated
64-row windows; scratch 312*128 = 39936 rows); norms/emb g=128 row-per-CTA;
pre = pre64 x2 on 64-row half-views; tail r%128 -> the M64 path (which tails
r%64 -> M32). Bit-identity by construction at every stage (same kernels, same
per-row op order; the M-grid second pass is the P7B-proven mechanism)."""
import sys

def patch(path, subs, mode="rw"):
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        assert n == 1, (path, n, old[:90])
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"[patch] {path}: {len(subs)} edits OK")

BASE = "~/tinygrad-metal/engine0"

M128_CODE = '''

# ====================== R2c: M=128 TRUNK MACHINERY ======================

M128 = os.getenv("PF_M128", "0") == "1"
assert not M128 or (M64 and SCANC and SCANC_N2 and ATTNW), "PF_M128 requires the M64/WY-N2/wide-attn config"
_M128ON = M128

def m128_set(on):
  global _M128ON
  _M128ON = bool(on)

M128_CUBINS = ["pfaw_w128h_s13_100k", "pfcw128h_s13",
               "pfca_c32_nc4_nw16", "pfcb_c32_nc4_nw8", "pfcz_c32_nc4_nw8"] + \\
              (["pfk_smul128"] if FFNSPLIT else [])

def ensure128(E):
  """R2c: the 128-row trunk plan. GEMMs = m64 cubins x2 M-blocks; scan =
  WY-C32 NC=4; attention = one w128h window; norms/emb g=128 (row-per-CTA);
  pre = pre64 x2. Coexists with the M64/M32 plans (the tail chain)."""
  if getattr(E, "_pf_plan128", None) is not None:
    return
  ensure64(E)   # cubins + W7 + the M64 world (the tail path)
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  W7 = getattr(E, "_pf_W7", {})
  _l = E._pf_m128loaded = getattr(E, "_pf_m128loaded", set())
  for n in M128_CUBINS:
    if n in _l: continue
    _l.add(n)
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=_sc_entry(n) if n.endswith("_100k") else n,
                                   target=dev.renderer.target, signature=tuple()))
  M = 128
  for nm, nb, dt, v in [
      ("xA128", M*5120*4, np.float32, 7.7e31), ("xB128", M*5120*4, np.float32, 7.7e31),
      ("xh128", M*5120*2, np.float16, 7.7), ("hh128", M*5120*4, np.float32, 7.7e31),
      ("hhx128", M*5120*2, np.float16, 7.7), ("attn_out128", M*5120*2, np.float16, 7.7),
      ("qkv128", M*10240*2, np.float16, 7.7), ("gate128", M*6144*2, np.float16, 7.7),
      ("z128", M*6144*2, np.float16, 7.7), ("gact128", M*17408*2, np.float16, 7.7),
      ("araw128", M*48*4, np.float32, 7.7e31), ("braw128", M*48*4, np.float32, 7.7e31),
      ("qrow128", M*12288*2, np.float16, 7.7), ("krow128", M*1024*2, np.float16, 7.7),
      ("vrow128", M*1024*2, np.float16, 7.7), ("qw128", M*24*256*2, np.float16, 7.7),
      ("ao128", M*6144*2, np.float16, 7.7)]:
    P.poison(nm, nb, dt, v)
  if FFNSPLIT:
    P.poison("ag128", M*17408*2, np.float16, 7.7)
    P.poison("au128", M*17408*2, np.float16, 7.7)
  P.up("ids128", np.zeros(M, dtype=np.int32))
  P.up("pos_arr128", np.zeros(1, dtype=np.int32))
  P.up("pos_w128", np.zeros(8, dtype=np.int32))
  P.poison("scscr128", 48 * 4 * 125712, np.uint8, 0xAB)   # WY C=32 NC=4 scratch
  P.poison("sco128", M*6144*4, np.float32, 7.7e31)
  P.poison("pmW128", 39968*4, np.float32, 7.7e31)   # w128h: 312 CTAs x 128 rows (+32 pad)
  P.poison("psW128", 39968*4, np.float32, 7.7e31)
  P.poison("pAW128", 39968*256*4, np.float32, 7.7e31)
  dev.synchronize(); P._keep.clear()
  plan = []
  def A(p, *a, g=1, ls=LS):
    plan.append((p, a, g, ls))
  def V(nm, off, sz):
    return d[nm].offset(offset=off, size=sz)
  A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids128"], d["xA128"], g=128)
  cur = 0
  for i in range(64):
    xin = d["xA128"] if cur == 0 else d["xB128"]
    xout = d["xB128"] if cur == 0 else d["xA128"]
    if i in E.qtypes:
      A(pr["pfk_n16"], xin, W[("nw1", i)], d["xh128"], g=128)
      _r7qkv = ("k", i) in W7 and (E.qtypes[i] == 14 or ("q", i) in W7)
      if M64QKV and _r7qkv:
        mq64 = "pfg3m_attnqkvq6_r7_m64_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m64_nw8k128"
        qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
        A(pr[mq64], qw_, W7[("k", i)], W[("v", i)], d["gridf"], d["xh128"],
          d["qrow128"], d["krow128"], d["vrow128"], g=448, ls=(256, 1, 1))
      else:
        for p in range(4):
          A(pr["pfg2_attnqkvq6_m32_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_m32_hm_nw8k128"],
            W[("q", i)], W[("k", i)], W[("v", i)], d["gridf"],
            V("xh128", p*32*5120*2, 32*5120*2), V("qrow128", p*32*12288*2, 32*12288*2),
            V("krow128", p*32*1024*2, 32*1024*2), V("vrow128", p*32*1024*2, 32*1024*2), g=224, ls=LS)
      for p in range(2):   # pre64 x2 on 64-row half-views (pos_w128[p*4:(p+1)*4])
        A(pr["pfk_pre64_100k"], V("qrow128", p*64*12288*2, 64*12288*2), V("krow128", p*64*1024*2, 64*1024*2),
          V("vrow128", p*64*1024*2, 64*1024*2), W[("qnw", i)], W[("knw", i)], d["freqs"],
          d[f"kv{i}"], d[f"sc{i}"], d["pos_arr128"], V("qw128", p*64*6144*2, 64*6144*2), g=24)
      A(pr["pfaw_w128h_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw128"], d["pos_w128"],
        d["pmW128"], d["psW128"], d["pAW128"], g=4*S13*6, ls=(512, 1, 1))
      A(pr["pfcw128h_s13"], d["pmW128"], d["psW128"], d["pAW128"], d["qrow128"], d["ao128"], g=24)
      for p in range(4):   # attn o-proj: classic m32 (iq3s not in packed7)
        A(pr["pfg_iq3s_m32_hm_nw8k128"], W[("o", i)], d["grid512"],
          V("ao128", p*32*6144*2, 32*6144*2), V("attn_out128", p*32*5120*2, 32*5120*2), g=80, ls=LS)
    else:
      j = E.gdn_idx.index(i)
      if ABW:
        A(pr["pfk_ab16w"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
          d["xh128"], d["araw128"], d["braw128"], g=384, ls=(1024, 1, 1))
      else:
        A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
          d["xh128"], d["araw128"], d["braw128"], g=128*13)
      if ("gate", i) in W7:
        A(pr["pfg3m_gdnqg_r7_m64_nw8k128"], W[("qkv", i)], W7[("gate", i)], d["gridf"], d["xh128"],
          d["qkv128"], d["gate128"], g=512, ls=(256, 1, 1))
      else:
        for p in range(4):
          A(pr["pfg2_gdnqg_m32_hm_nw16k128"], W[("qkv", i)], W[("gate", i)], d["gridf"],
            V("xh128", p*32*5120*2, 32*5120*2), V("qkv128", p*32*10240*2, 32*10240*2),
            V("gate128", p*32*6144*2, 32*6144*2), g=128, ls=(512, 1, 1))
      # WY-C32 NC=4: ONE launch triple per 128-row chunk
      scwpb = d["scwp"].offset(offset=j*47200*4, size=47200*4)
      A(pr["pfca_c32_nc4_nw16"], scwpb, d[f"conv{i}_0"],
        d["qkv128"], d["araw128"], d["braw128"], d["scscr128"], g=48*4, ls=(512, 1, 1))
      A(pr["pfcb_c32_nc4_nw8"], d["scscr128"], d[f"rec{i}"], d["sco128"], g=192)
      A(pr["pfcz_c32_nc4_nw8"], d["sco128"], d["gate128"],
        scwpb.offset(offset=41056*4, size=6144*4), d["z128"], d["qkv128"],
        d[f"conv{i}_0"], g=48*4*4)
      if (not E.gdn_oq8[i]) and ("out", i) in W7:
        A(pr["pfg3_iq3o_r7_m64_nw8k128"], W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], g=160, ls=LS)
      else:
        on = "pfg_q8o_m32_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_m32_hm_nw8k128"
        for p in range(4):
          A(pr[on], W[("out", i)], d["gridf"], V("z128", p*32*6144*2, 32*6144*2),
            V("attn_out128", p*32*5120*2, 32*5120*2), g=80, ls=LS)
    A(pr["pfk_hh16"], xin, d["attn_out128"], W[("nw2", i)], d["hh128"], d["hhx128"], g=128)
    if ("fg", i) in W7 and FFNSPLIT:
      A(pr["pfg3_fgp_r7_m64_nw8k128"], W7[("fg", i)], d["gridf"], d["hhx128"], d["ag128"], g=544, ls=LS)
      A(pr["pfg3_fup_r7_m64_nw8k128"], W7[("fu", i)], d["gridf"], d["hhx128"], d["au128"], g=544, ls=LS)
      A(pr["pfk_smul128"], d["ag128"], d["au128"], d["gact128"], g=1088, ls=LS)
    elif ("fg", i) in W7:
      A(pr["pfg3_ffn_r7_m64_nw4k128"], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx128"],
        d["gact128"], g=1088, ls=(128, 1, 1))
    else:
      for p in range(4):
        A(pr["pfg_ffn_m32_nt32_hm_nw4k128" if NT32 else "pfg_ffn_m32_hm_nw8k128"], W[("fg", i)], W[("fu", i)], d["gridf"],
          V("hhx128", p*32*5120*2, 32*5120*2), V("gact128", p*32*17408*2, 32*17408*2),
          g=544 if NT32 else 272, ls=(128, 1, 1) if NT32 else LS)
    if ("fd", i) in W7:
      A(pr["pfg3_iq3d_r7_m64_nw8k128"], W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], xout, g=160, ls=LS)
    else:
      for p in range(4):
        A(pr["pfg_iq3d_m32_res_hm_nw8k128"], W[("fd", i)], d["gridf"],
          V("gact128", p*32*17408*2, 32*17408*2), V("hh128", p*32*5120*4, 32*5120*4),
          xout.offset(offset=p*32*5120*4, size=32*5120*4), g=80, ls=LS)
    cur ^= 1
  E._pf_plan128 = plan
  E._pf_last128 = d["xA128"] if cur == 0 else d["xB128"]
  dev.synchronize()
  print(f"[r2c] M128 plan ready: {len(plan)} launches/chunk "
        f"(scan=WY-C32-NC4, attn=w128h, ffn={'split' if FFNSPLIT else 'fused'}, pre=pre64x2)", flush=True)

def _pf_dfill_seq128(E):
  """8 x 16-row draft-fill windows (even count -> last window rows in REC1)."""
  d, W, pr = E.P.d, E.W, E.pr
  seq = []
  for k in range(8):
    xk = d["xA128"].offset(offset=k*16*5120*4, size=16*5120*4)
    idk = d["ids128"].offset(offset=k*16*4, size=16*4)
    pk = d["pos_w128"].offset(offset=k*4, size=4)
    w, wn = (d["REC0"], d["REC1"]) if (k & 1) == 0 else (d["REC1"], d["REC0"])
    seq.append((pr["pfk_rec16"], (xk, w, wn), 17, LS))
    seq.append((pr["pfk_emb16"], (W[("emb", 0)], d["grid512"], idk, d["e16f_d"]), 16, LS))
    seq.append((pr["pfd_dnorm16"], (d["e16f_d"], w, d["d_enw"], d["d_hnw"], d["cat16_d"]), 16, LS))
    seq.append((pr["pfg_ehd_res_hm_nw8k128"], (d["d_eh"], d["gridf"], d["cat16_d"], d["zed5k16"], d["xin_d16"]), 80, LS))
    seq.append((pr["pfk_n16"], (d["xin_d16"], d["d_nw1"], d["xh_d16"]), 16, LS))
    seq.append((pr["pfg2_dqkv_hm_nw8k128"], (d["d_q"], d["d_k"], d["d_v"], d["gridf"], d["xh_d16"],
                                             d["qrow_d16"], d["krow_d16"], d["vrow_d16"]), 224, LS))
    seq.append((pr["pfk_pre16_100k"], (d["qrow_d16"], d["krow_d16"], d["vrow_d16"], d["d_qnw"], d["d_knw"],
                                        d["freqs"], d["kv_d"], d["sc_d"], pk, d["qw16_d"]), 24, LS))
  return seq

def prefill_batch_m128(E, G, ids, prog=None, log=None, chunk_times=None, on_chunk=None):
  """R2c: 128-row chunks. r = N % 128 tail delegates to the M64 path (which
  tails r%64 -> M32). Same post-conditions + the P15 tail laws (running pos
  BEFORE delegation; ambient-flag graph keying)."""
  ensure128(E)
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  pos0 = int(P.down_at("pos_slot", 0, 1)[0])
  N = len(ids)
  nc = N // 128
  r = N - 128 * nc
  P.win_up("tok_hist", pos0 * 4, np.array([int(t) for t in ids], dtype=np.int32))
  t0 = time.perf_counter()
  for c in range(nc):
    tc = time.perf_counter()
    p0 = pos0 + 128 * c
    P.win_up("ids128", 0, np.array([int(t) for t in ids[128*c:128*c+128]], dtype=np.int32))
    P.win_up("pos_arr128", 0, np.array([p0], dtype=np.int32))
    P.win_up("pos_w128", 0, np.array([p0 + 16*k for k in range(8)], dtype=np.int32))
    if os.getenv("PF_PG", "1") == "1":
      vend = _pf_submit_chunk(E, p0)
      if os.getenv("PG_WAIT", "1") == "1":
        dev.timeline_signal.wait(vend)
    else:
      _run_plan(E._pf_plan128)
    if chunk_times is not None:
      chunk_times.append((p0, (time.perf_counter() - tc) * 1e3))
    if on_chunk is not None:
      on_chunk(pos0 + 128 * (c + 1))
    if prog is not None and (c % 2 == 0 or c == nc - 1):
      prog(128 * (c + 1), N)
    if log is not None and (c % 32 == 31 or c == nc - 1):
      log("prefill_batch128_chunk", k=c + 1, n=nc, t=round(time.perf_counter() - t0, 1))
  if r > 0:
    if log is not None: log("prefill_batch128_tail_m64", n=r)
    P.win_up("pos_slot", 0, np.array([pos0 + 128*nc], dtype=np.int32))
    _s128, _s64 = _M128ON, _M64ON
    m128_set(False)   # the tail MUST run the M64/M32 plans (graph-cache ambient-flag law)
    try:
      prefill_batch(E, G, ids[128*nc:], prog=(lambda k, n: prog(128*nc + k, N)) if prog is not None else None,
                    log=None, chunk_times=chunk_times)
    finally:
      m128_set(_s128); m64_set(_s64)
    dev.synchronize()
    return time.perf_counter() - t0
  xr = E._pf_last128.offset(offset=127 * 5120 * 4, size=5120 * 4)
  P.win_up("pos_slot", 0, np.array([pos0 + N - 1], dtype=np.int32))
  pr["pfk_n16"](xr, W[("onw", 0)], d["xh"], global_size=(1, 1, 1), local_size=LS)
  pr["head8"](W[("head", 0)], d["xh"], d["logits"], global_size=(VOCAB // 8, 1, 1), local_size=LS)
  pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1, 1, 1), local_size=LS, wait=True)
  if DFILL:
    # 8 windows/chunk (even) -> last window rows in REC1 (same law)
    hlast = P.down_at("REC1", 16*5120*4, 5120, np.float32)
    P.win_up("hd_d1", 0, hlast)
    P._keep.clear()
  dev.synchronize()
  return time.perf_counter() - t0
'''

patch(f"{BASE}/pf_prefill.py", [
  # append the M128 machinery before prefill_batch
  ('def prefill_batch(E, G, ids, prog=None, log=None, chunk_times=None, on_chunk=None):',
   M128_CODE + '\ndef prefill_batch(E, G, ids, prog=None, log=None, chunk_times=None, on_chunk=None):'),
  # dispatch
  ('''  if _M64ON and M32 and len(ids) >= 64:   # P15: the M=64 trunk (tails r%64 -> M32 below)
    return prefill_batch_m64(E, G, ids, prog=prog, log=log, chunk_times=chunk_times, on_chunk=on_chunk)''',
   '''  if _M128ON and M128 and len(ids) >= 128:  # R2c: the M=128 trunk (tails r%128 -> M64)
    return prefill_batch_m128(E, G, ids, prog=prog, log=log, chunk_times=chunk_times, on_chunk=on_chunk)
  if _M64ON and M32 and len(ids) >= 64:   # P15: the M=64 trunk (tails r%64 -> M32 below)
    return prefill_batch_m64(E, G, ids, prog=prog, log=log, chunk_times=chunk_times, on_chunk=on_chunk)'''),
  # graph cache: m128 key + plan/dffill selection
  ('''def _pf_graphs(E):
  m64 = bool(_M64ON) and getattr(E, "_pf_plan64", None) is not None
  key = (M32, DFILL, G3M, NT32, ATTN32, A4, HYB, ATTN_THR, N32, PRE32, SCAN32, PERSIST, PERSIST_NAME, m64, ATTNW, SCANC, SCANC_N2, M64QKV, ABW, FFNSPLIT)''',
   '''def _pf_graphs(E):
  m64 = bool(_M64ON) and not _M128ON and getattr(E, "_pf_plan64", None) is not None
  m128 = bool(_M128ON) and M128 and getattr(E, "_pf_plan128", None) is not None
  key = (M32, DFILL, G3M, NT32, ATTN32, A4, HYB, ATTN_THR, N32, PRE32, SCAN32, PERSIST, PERSIST_NAME, m64, m128, ATTNW, SCANC, SCANC_N2, M64QKV, ABW, FFNSPLIT)'''),
  ('''    if m64:
      full = list(E._pf_plan64) + (_pf_dfill_seq64(E) if (M32 and DFILL) else [])
    else:
      full = list(E._pf_plan) + (_pf_dfill_seq(E) if (M32 and DFILL) else [])''',
   '''    if m128:
      full = list(E._pf_plan128) + (_pf_dfill_seq128(E) if (M32 and DFILL) else [])
    elif m64:
      full = list(E._pf_plan64) + (_pf_dfill_seq64(E) if (M32 and DFILL) else [])
    else:
      full = list(E._pf_plan) + (_pf_dfill_seq(E) if (M32 and DFILL) else [])'''),
  ('''    _p26 = getattr(E, "_pf_plan26_64", None) if m64 else getattr(E, "_pf_plan26", None)''',
   '''    _p26 = None if m128 else (getattr(E, "_pf_plan26_64", None) if m64 else getattr(E, "_pf_plan26", None))'''),
])

src = open(f"{BASE}/pcache.py").read()
if "PF_M128" not in src:
  old = '"PF_DR7", "PF_FFNSPLIT")'
  assert src.count(old) == 1
  src = src.replace(old, '"PF_DR7", "PF_FFNSPLIT", "PF_M128")')
  open(f"{BASE}/pcache.py", "w").write(src)
  print("[patch] pcache.py: PF_M128 added")
print("[patch all done]")
