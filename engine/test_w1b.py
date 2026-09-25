# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-b: validation + gate run.
MODE=attnval (default): ONE attention block vs stock on REAL weights, T=1, poison-first.
MODE=trunk: restore /tmp/w1b_state.npz -> 60-token greedy -> agreement + timing + attribution."""
import os, sys, time, json
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
MODE = os.getenv("MODE", "attnval")
GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

def rel(a, b): return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))

if MODE == "attnval":
  os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
  os.environ.setdefault("MTP_A3C_OFF", "1")
  os.environ.setdefault("MTP_EMB_GATHER", "1")
  os.environ.setdefault("MTP_MAXCTX", "2048")
  from tinygrad import Tensor
  from tinygrad.llm.model import Transformer, TransformerBlock
  from tinygrad.device import Device
  print("[loading stock model...]", flush=True)
  model = Transformer.from_gguf(GGUF, 2048)[0]
  blk = next(b for b in model.blk if isinstance(b, TransformerBlock))
  cfg = blk.config
  print(f"[cfg] n_heads={cfg.n_heads} n_kv={cfg.n_kv_heads} head_dim={cfg.head_dim} "
        f"v_head={cfg.v_head_dim} qk_norm={cfg.qk_norm} gate={cfg.attn_output_gate} "
        f"rope_dim={cfg.rope_dim} theta={cfg.rope_theta}", flush=True)
  assert (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.qk_norm, bool(cfg.attn_output_gate)) == (24, 4, 256, 256, True), "structure drift"
  theta = float(cfg.rope_theta)
  L = 1500
  rng = np.random.default_rng(7)
  x_np = (rng.standard_normal((1, 1, 5120)) * 0.2).astype(np.float32)
  kv_np = np.zeros((2, 1, 4, 2048, 256), np.float16)
  kv_np[:, :, :, :L] = (rng.standard_normal((2, 1, 4, L, 256)) * 0.1).astype(np.float16)
  # ---- stock reference ----
  x_in = Tensor(x_np).contiguous().realize()
  blk._init_state(x_in)
  blk.cache_kv.assign(Tensor(kv_np)).realize()
  y_ref = blk(x_in, L).float().numpy().reshape(-1)
  k_ref = blk.cache_kv[0, 0, :, L, :].float().numpy().reshape(-1)
  v_ref = blk.cache_kv[1, 0, :, L, :].float().numpy().reshape(-1)
  print(f"[ref] |y|max={np.abs(y_ref).max():.4f} |k|max={np.abs(k_ref).max():.4f}", flush=True)
  # ---- engine (block 3 only) ----
  import engine0
  engine0.QUANT[21] = (256, 110)
  from engine0 import Bufs, parse_gguf, read_raw, iq3_grid_f32, dev
  from trunk import iq3s_grid_f32
  from tinygrad.device import TinyELF
  from tinygrad.runtime.ops_nv import NVProgram
  P = Bufs(); ds, infos = parse_gguf(); i = 3; pre = f"blk.{i}."
  P.up("gridf", iq3_grid_f32()); P.up("grid512", iq3s_grid_f32())
  P.up("freqs", (1.0/(theta**(np.arange(0, 64, 2, dtype=np.float64)/64.0))).astype(np.float32))
  W = {n: P.up("w_"+n, np.frombuffer(read_raw(infos[pre+tname], ds), dtype=np.uint8))
       for n, tname in [("q", "attn_q.weight"), ("k", "attn_k.weight"), ("v", "attn_v.weight"), ("o", "attn_output.weight")]}
  for n in ["fg", "fu", "fd"]:
    t = {"fg": "ffn_gate", "fu": "ffn_up", "fd": "ffn_down"}[n]
    W[n] = P.up("w_"+n, np.frombuffer(read_raw(infos[pre+f"{t}.weight"], ds), dtype=np.uint8))
  for n, t in [("nw1", "attn_norm"), ("nw2", "post_attention_norm"), ("qnw", "attn_q_norm"), ("knw", "attn_k_norm")]:
    W[n] = P.up("w_"+n, np.frombuffer(read_raw(infos[pre+f"{t}.weight"], ds), dtype="<f4"))
  for nm, nb, dt, pv in [("x", 5120*4, np.float32, 7.7e31), ("xh", 5120*2, np.float16, 7.7),
                         ("qrow", 12288*2, np.float16, 7.7), ("k_row", 1024*2, np.float16, 7.7),
                         ("v_row", 1024*2, np.float16, 7.7), ("ao_row", 6144*2, np.float16, 7.7),
                         ("attn_out", 5120*2, np.float16, 7.7), ("hh", 5120*4, np.float32, 7.7e31),
                         ("hhx", 5120*2, np.float16, 7.7), ("gact", 17408*2, np.float16, 7.7),
                         ("y", 5120*4, np.float32, 7.7e31)]:
    P.poison(nm, nb, dt, pv)
  P.up("x", x_np.reshape(-1))
  P.up("kv3", kv_np.reshape(-1))
  P.up("pos", np.array([L], dtype=np.int32))
  dev.synchronize()
  pr = {}
  for n in ["k0_norm", "k1_q5", "k1_iq3", "k3m_hh", "k3b_ffn", "k3c_down", "a_q6", "a_kv", "a_attn", "a_o"]:
    lib = open(f"~/tinygrad-metal/engine0/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  LS = (256, 1, 1); d = P.d
  qt = infos[pre+"attn_q.weight"][0]
  print(f"[engine] blk3 q type = {qt}", flush=True)
  def LN(name, fn):
    if os.getenv("DBG"): print(f"[launch] {name}", flush=True)
    fn(bool(os.getenv("DBG")))
  LN("k0_norm", lambda w: pr["k0_norm"](d["x"], W["nw1"], d["xh"], global_size=(1,1,1), local_size=LS, wait=w))
  if qt == 14: LN("a_q6", lambda w: pr["a_q6"](W["q"], d["xh"], d["qrow"], global_size=(1536,1,1), local_size=LS, wait=w))
  else: LN("k1_iq3", lambda w: pr["k1_iq3"](W["q"], d["gridf"], d["xh"], d["qrow"], global_size=(1536,1,1), local_size=LS, wait=w))
  LN("a_kv", lambda w: pr["a_kv"](W["k"], W["v"], d["gridf"], d["xh"], d["k_row"], d["v_row"], global_size=(256,1,1), local_size=LS, wait=w))
  LN("a_attn", lambda w: pr["a_attn"](d["qrow"], d["k_row"], d["v_row"], W["qnw"], W["knw"], d["freqs"], d["kv3"], d["pos"], d["ao_row"], global_size=(24,1,1), local_size=LS, wait=w))
  LN("a_o", lambda w: pr["a_o"](W["o"], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=w))
  LN("k3m_hh", lambda w: pr["k3m_hh"](d["x"], d["attn_out"], W["nw2"], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS, wait=w))
  LN("k3b_ffn", lambda w: pr["k3b_ffn"](W["fg"], W["fu"], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS, wait=w))
  pr["k3c_down"](W["fd"], d["gridf"], d["gact"], d["hh"], d["y"], global_size=(640,1,1), local_size=LS, wait=True)
  got_y = P.down("y", (5120,))
  kv_got = P.down("kv3", (2*4*2048*256,), np.float16).astype(np.float32).reshape(2, 4, 2048, 256)
  got_k = kv_got[0, :, L, :].reshape(-1); got_v = kv_got[1, :, L, :].reshape(-1)
  r_y, r_k, r_v = rel(got_y, y_ref), rel(got_k, k_ref), rel(got_v, v_ref)
  print(f"[val] y relerr={r_y:.2e}  k[L] relerr={r_k:.2e}  v[L] relerr={r_v:.2e}", flush=True)
  ok = max(r_y, r_k, r_v) < 1e-3
  print(f"[val] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
  if not ok:
    qrow = P.down("qrow", (12288,), np.float16).astype(np.float32)
    ao = P.down("ao_row", (6144,), np.float16).astype(np.float32)
    # --- stage oracle 1: stock q GEMV ---
    q_ref = blk.attn_q(blk.attn_norm(x_in)).numpy().reshape(-1).astype(np.float32)
    print(f"[dbg] q GEMV relerr={rel(qrow, q_ref):.3e}")
    print(f"[dbg] qrow[:6]={qrow[:6]}\n      q_ref[:6]={q_ref[:6]}", flush=True)
    # --- stage oracle 2: numpy attention from q_ref over engine kv ---
    qref_h = q_ref.reshape(24, 2, 256)          # [h][q|gate][d]
    kvn = kv_np.reshape(2, 4, 2048, 256).astype(np.float32)
    fr = (1.0/(theta**(np.arange(0, 64, 2, dtype=np.float64)/64.0))).astype(np.float32)
    ang = np.float32(L) * fr                    # (32,)
    cs, sn = np.cos(ang), np.sin(ang)
    ao_np = np.zeros((24, 256), np.float32)
    qnw_np = np.frombuffer(read_raw(infos[pre+"attn_q_norm.weight"], ds), dtype="<f4")
    for h in range(24):
      kvh = h // 6
      q0 = qref_h[h, 0].astype(np.float32)
      ss = float((q0*q0).mean())
      qn = (q0 * (1.0/np.sqrt(ss + 1e-6))).astype(np.float16).astype(np.float32) * qnw_np
      kn = kvn[0, kvh, L]                        # engine-written K row == stock k_ref (validated)
      qroped = qn.copy()
      qroped[:32] = qn[:32]*cs - qn[32:64]*sn
      qroped[32:64] = qn[32:64]*cs + qn[:32]*sn
      sc = (kvn[0, kvh, :L+1] @ qroped) * 0.0625
      p = np.exp(sc - sc.max()); p /= p.sum()
      out = p @ kvn[1, kvh, :L+1]
      g = 1.0/(1.0+np.exp(-qref_h[h, 1].astype(np.float32)))
      ao_np[h] = out * g
    print(f"[dbg] numpy-ao vs engine ao relerr={rel(ao, ao_np.reshape(-1)):.3e}", flush=True)
    ao_eng = ao.reshape(24, 256)
    print(f"[dbg] ao eng[0,:4]={ao_eng[0,:4]} np[0,:4]={ao_np[0,:4]}", flush=True)
    np.save("/tmp/w1b_got.npy", got_y); np.save("/tmp/w1b_refy.npy", y_ref)
    np.save("/tmp/w1b_gotk.npy", got_k); np.save("/tmp/w1b_refk.npy", k_ref)
    np.save("/tmp/w1b_qrow.npy", qrow); np.save("/tmp/w1b_qref.npy", q_ref)
  sys.exit(0 if ok else 1)

# ---------------- MODE=trunk ----------------
from trunk import TrunkEngine, CTX
from engine0 import dev
snap = np.load(os.getenv("SNAP", "~/w1b_state_2k.npz"))
theta = float(snap["theta"].reshape(-1)[0])
print("[trunk] loading engine weights (~12.8GB)...", flush=True)
t0 = time.perf_counter()
E = TrunkEngine(theta)
print(f"[trunk] engine loaded in {time.perf_counter()-t0:.1f}s", flush=True)
E.restore(snap)
P0 = int(snap["P"].reshape(-1)[0]); base = snap["base_out"].reshape(-1).tolist()
NTOK = 60

E.run_tokens(NTOK)
hist = E.P.down("tok_hist", (CTX+128,), np.int32)
eng = hist[P0-1:P0-1+NTOK]
agree = int((eng == np.array(base)).sum())
first_div = next((k for k in range(NTOK) if eng[k] != base[k]), None)
print(f"[agree] engine vs stock baseline: {agree}/{NTOK} exact, first divergence at token {first_div}", flush=True)
print(f"[engine toks] {eng.tolist()}", flush=True)
print(f"[base   toks] {base}", flush=True)

print("== timing (60-token runs, one wait at end) ==", flush=True)
times = []
for rep in range(3):
  E.restore(snap)
  ts = [time.perf_counter()]
  E._submit_only = True
  E.run_tokens(NTOK)
  ts.append(time.perf_counter())          # after last submit (final wait inside)
  dt = ts[1]-ts[0]
  times.append(dt)
  print(f"[time] rep{rep}: {dt*1e3:.1f} ms -> {NTOK/dt:.2f} tok/s", flush=True)
best = min(times)
print(f"[time] BEST: {best*1e3:.1f} ms/60tok = {best/NTOK*1e3:.2f} ms/tok = {NTOK/best:.2f} tok/s", flush=True)

print("== attribution (60x pipelined per phase) ==", flush=True)
E.restore(snap)
d = E.P.d
def phase(fn):
  for _ in range(3): fn()
  dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(60): fn(); dev.synchronize()
  E.pr["k0_norm"](d["x0"], E.W[("nw1", E.gdn_idx[0])], d["xh"], global_size=(1,1,1), local_size=(256,1,1), wait=True)
  return (time.perf_counter()-t0)/60*1e3
def f_gdn():
  for i in E.gdn_idx: E.gdn(i, d["x0"], d["x1"], f"conv{i}_0", f"conv{i}_1")
def f_attn():
  for i in E.attn_idx: E.attn(i, d["x0"], d["x1"])
def f_head():
  E.head(d["x0"])
def f_embed():
  E.pr["h_embed"](E.W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=(256,1,1))
for nm, fn in [("gdn48", f_gdn), ("attn16", f_attn), ("head+argmax", f_head), ("embed", f_embed)]:
  print(f"[attr] {nm:12s} {phase(fn):8.2f} ms/token", flush=True)
print("[done]", flush=True)
