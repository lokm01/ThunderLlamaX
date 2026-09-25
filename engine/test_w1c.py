# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-c validation.
MODE=kval (default): bit-equal check of every wide kernel vs its W1-b original on
REAL weights (poison-first between runs). Wide kernels must be BIT-IDENTICAL (same
decoded values + same fp op order; only load widths changed).
MODE=trunk: TrunkEngineW1C 60-token greedy vs stock baseline + timing."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
MODE = os.getenv("MODE", "kval")
BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)

def rel(a, b): return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))

if MODE == "kval":
  from engine0 import Bufs, parse_gguf, read_raw, iq3_grid_f32, dev
  from trunk import iq3s_grid_f32
  from tinygrad.device import TinyELF
  from tinygrad.runtime.ops_nv import NVProgram
  rng = np.random.default_rng(11)
  P = Bufs(); ds, infos = parse_gguf()
  P.up("gridf", iq3_grid_f32()); P.up("grid512", iq3s_grid_f32())
  _pc = {}
  def prog_load(n):
    if n not in _pc:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      _pc[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    return _pc[n]
  W = {}
  def upraw(key, name):
    W[key] = P.up("w_"+key, np.frombuffer(read_raw(infos[name], ds), dtype=np.uint8))
  PK = "~/tinygrad-metal/engine0/packed"
  def uppk(key, fname):
    W[key] = P.up("wp_"+key, np.ascontiguousarray(np.load(f"{PK}/{fname}.npy")))
  def upboth(key, tname, fname):
    upraw(key, tname)                      # W[key] = RAW (for OLD kernels)
    uppk(key + "p", fname)                 # W[key+"p"] = PACKED (for NEW kernels)
  # GDN block 0: qkv raw Q5 (both); gate/fg/fu/fd + block8 ssm_out RAW+PACKED
  upraw("qkv", "blk.0.attn_qkv.weight")
  upboth("gate", "blk.0.attn_gate.weight", "gate0")
  upboth("fg", "blk.0.ffn_gate.weight", "fg0")
  upboth("fu", "blk.0.ffn_up.weight", "fu0")
  upboth("fd", "blk.0.ffn_down.weight", "fd0")
  upboth("out3", "blk.8.ssm_out.weight", "out8")
  upboth("q6", "blk.3.attn_q.weight", "q3")
  upboth("q3", "blk.11.attn_q.weight", "q11")
  upboth("k", "blk.3.attn_k.weight", "k3")
  upraw("v", "blk.3.attn_v.weight"); upraw("o", "blk.3.attn_output.weight")
  # head: first 1024 rows of output.weight (3520B each)
  head_raw = read_raw(infos["output.weight"], ds)[:1024*3520]
  W["head"] = P.up("w_head", np.frombuffer(head_raw, dtype=np.uint8))
  # scratch (poisoned, then filled with plausible inputs)
  for nm, nb, dt, pv in [("xh", 5120*2, np.float16, 7.7), ("qkv_row", 10240*2, np.float16, 7.7),
                         ("gate_row", 6144*2, np.float16, 7.7), ("attn_out", 5120*2, np.float16, 7.7),
                         ("gact", 17408*2, np.float16, 7.7), ("y", 5120*4, np.float32, 7.7e31),
                         ("qrow", 12288*2, np.float16, 7.7), ("k_row", 1024*2, np.float16, 7.7),
                         ("v_row", 1024*2, np.float16, 7.7), ("ao_row", 6144*2, np.float16, 7.7),
                         ("hh", 5120*4, np.float32, 7.7e31), ("logits", 1024*2, np.float16, 7.7),
                         ("z", 6144*2, np.float16, 7.7)]:
    P.poison(nm, nb, dt, pv)
  P.up("xh", (rng.standard_normal(5120)*0.3).astype(np.float16))
  P.up("hh", (rng.standard_normal(5120)*0.4).astype(np.float32))
  P.up("z", (rng.standard_normal(6144)*0.3).astype(np.float16))
  P.up("gact", (rng.standard_normal(17408)*0.2).astype(np.float16))
  dev.synchronize()
  d = P.d
  POIS = {"xh": None}
  def cmp(name, out, shape, dt, old_launch, new_launch):
    pval = np.full(int(np.prod(shape)), -333.0 if dt == np.float32 else -33.3, dtype=dt)
    P.up(out, pval); dev.synchronize()
    old_launch(True)
    ref = P.down(out, shape, dt)
    P.up(out, pval); dev.synchronize()
    new_launch(True)
    got = P.down(out, shape, dt)
    ib = np.uint16 if dt == np.float16 else np.uint32
    biteq = bool(np.array_equal(ref.view(ib), got.view(ib)))
    maxd = float(np.abs(ref.astype(np.float64) - got.astype(np.float64)).max())
    print(f"[kval] {name:8s} bit-equal={biteq} maxdiff={maxd:.3e} relerr={rel(got, ref):.2e}", flush=True)
    return biteq
  ok = True
  # q5g8 (grid 2048)
  ok &= cmp("q5g8", "qkv_row", (10240,), np.float16,
    lambda w: prog_load("k1_q5g")(W["qkv"], W["gate"], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS, wait=w),
    lambda w: prog_load("q5g8")(W["qkv"], W["gatep"], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS, wait=w))
  ok &= cmp("q5g8g", "gate_row", (6144,), np.float16,
    lambda w: prog_load("k1_q5g")(W["qkv"], W["gate"], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS, wait=w),
    lambda w: prog_load("q5g8")(W["qkv"], W["gatep"], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS, wait=w))
  # head8 (first 1024 rows -> grid 128)
  ok &= cmp("head8", "logits", (1024,), np.float16,
    lambda w: prog_load("k1_q5")(W["head"], d["xh"], d["logits"], global_size=(128,1,1), local_size=LS, wait=w),
    lambda w: prog_load("head8")(W["head"], d["xh"], d["logits"], global_size=(128,1,1), local_size=LS, wait=w))
  # ffn8 (grid 2176)
  ok &= cmp("ffn8", "gact", (17408,), np.float16,
    lambda w: prog_load("k3b_ffn")(W["fg"], W["fu"], d["gridf"], d["xh"], d["gact"], global_size=(2176,1,1), local_size=LS, wait=w),
    lambda w: prog_load("ffn8")(W["fgp"], W["fup"], d["gridf"], d["xh"], d["gact"], global_size=(2176,1,1), local_size=LS, wait=w))
  # down8 (grid 640)
  ok &= cmp("down8", "y", (5120,), np.float32,
    lambda w: prog_load("k3c_down")(W["fd"], d["gridf"], d["gact"], d["hh"], d["y"], global_size=(640,1,1), local_size=LS, wait=w),
    lambda w: prog_load("down8")(W["fdp"], d["gridf"], d["gact"], d["hh"], d["y"], global_size=(640,1,1), local_size=LS, wait=w))
  # op38 (grid 640)
  ok &= cmp("op38", "attn_out", (5120,), np.float16,
    lambda w: prog_load("k3a_iq3")(W["out3"], d["gridf"], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=w),
    lambda w: prog_load("op38")(W["out3p"], d["gridf"], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=w))
  # aq6k8 / aq3k8 (grid 1792): q, k, v outs
  for qk, wq in [("aq6k8", "q6"), ("aq3k8", "q3")]:
    old = "a_qkv_q6" if qk == "aq6k8" else "a_qkv_iq3"
    ok &= cmp(qk+".q", "qrow", (12288,), np.float16,
      lambda w, old=old, wq=wq: prog_load(old)(W[wq], W["k"], W["v"], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=w),
      lambda w, qk=qk, wq=wq: prog_load(qk)(W[wq+"p"], W["kp"], W["v"], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=w))
    ok &= cmp(qk+".k", "k_row", (1024,), np.float16,
      lambda w, old=old, wq=wq: prog_load(old)(W[wq], W["k"], W["v"], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=w),
      lambda w, qk=qk, wq=wq: prog_load(qk)(W[wq+"p"], W["kp"], W["v"], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=w))
    ok &= cmp(qk+".v", "v_row", (1024,), np.float16,
      lambda w, old=old, wq=wq: prog_load(old)(W[wq], W["k"], W["v"], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=w),
      lambda w, qk=qk, wq=wq: prog_load(qk)(W[wq+"p"], W["kp"], W["v"], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=w))
  # ao8 (grid 640)
  ok &= cmp("ao8", "attn_out", (5120,), np.float16,
    lambda w: prog_load("a_o")(W["o"], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=w),
    lambda w: prog_load("ao8")(W["o"], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=w))
  print(f"[kval] VERDICT: {'ALL BIT-EQUAL' if ok else 'MISMATCH'}", flush=True)
  sys.exit(0 if ok else 1)

if MODE == "trunk":
  from trunk_w1c import TrunkEngineW1C
  from engine0 import dev
  from trunk import CTX
  snap = np.load(os.getenv("SNAP", "~/w1b_state_2k.npz"))
  theta = float(snap["theta"].reshape(-1)[0])
  print("[trunk] loading engine weights (~12.8GB)...", flush=True)
  t0 = time.perf_counter()
  E = TrunkEngineW1C(theta)
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
  if agree < NTOK:
    print(f"[engine toks] {eng.tolist()}", flush=True)
    print(f"[base   toks] {base}", flush=True)
  print("== timing (60-token runs, one wait at end) ==", flush=True)
  times = []
  for rep in range(3):
    E.restore(snap)
    ts = [time.perf_counter()]
    E._submit_only = True
    E.run_tokens(NTOK)
    ts.append(time.perf_counter())
    dt = ts[1]-ts[0]
    times.append(dt)
    print(f"[time] rep{rep}: {dt*1e3:.1f} ms -> {NTOK/dt:.2f} tok/s", flush=True)
  best = min(times)
  print(f"[time] BEST: {best*1e3:.1f} ms/60tok = {best/NTOK*1e3:.2f} ms/tok = {NTOK/best:.2f} tok/s", flush=True)
  print("== attribution (60x pipelined per phase) ==", flush=True)
  E.restore(snap)
  dd = E.P.d
  def phase(fn):
    for _ in range(3): fn()
    dev.synchronize()
    t0 = time.perf_counter()
    for _ in range(60): fn(); dev.synchronize()
    return (time.perf_counter()-t0)/60*1e3
  def f_gdn():
    for i in E.gdn_idx: E.gdn(i, dd["x0"], dd["x1"], f"conv{i}_0", f"conv{i}_1")
  def f_attn():
    for i in E.attn_idx: E.attn(i, dd["x0"], dd["x1"])
  def f_head():
    E.head(dd["x0"])
  def f_embed():
    E.pr["h_embed"](E.W[("emb",0)], dd["grid512"], dd["tok_slot"], dd["x0"], global_size=(1,1,1), local_size=LS)
  for nm, fn in [("gdn48", f_gdn), ("attn16", f_attn), ("head+argmax", f_head), ("embed", f_embed)]:
    print(f"[attr] {nm:12s} {phase(fn):8.2f} ms/token", flush=True)
  print("[done]", flush=True)
