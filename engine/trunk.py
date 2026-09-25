# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""engine0 W1-b: full T=1 decode trunk (48 GDN + 16 attn blocks + embed + head +
argmax), all static raw NV buffers, per-kernel cubins, device-resident greedy loop
(argmax -> tok_slot -> h_embed; pos from pos_slot; no host sync inside the loop)."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
import engine0
from engine0 import Bufs, parse_gguf, read_raw, iq3_grid_f32, dev

# engine0.QUANT[21] is (128,110) but the file says IQ3_S = (256,110) (W1A never
# loaded a type-21 tensor; token_embd/o_proj byte sizes prove 256/110).
engine0.QUANT[21] = (256, 110)

BASE = "~/tinygrad-metal/engine0"
DIM = 5120
# M1-A fixed handles: ALL mutable state allocated ONCE at boot at full engine ctx
# (SKV_CTXK; default 2304 = mtp.CTXK default) with the RUN dtype — a later P.up
# of kv/tok_hist would realloc -> stale graph handles + VRAM orphans (the ~755MB
# class). KV8=1 allocates uint8 kv + fp16 scale slabs instead of the fp16 slab.
CTX = int(os.getenv("SKV_CTXK", "2304"))
_TKV8 = os.getenv("KV8", "0") == "1"
VOCAB = 248320
GDN_CUBINS = ["k0_norm","k1_q5","k1_iq3","k1_ab","k2_scan","k2b_z","k3a_oproj","k3m_hh","k3b_ffn","k3c_down"]
NEW_CUBINS = ["a_q6","a_kv","a_attn","a_o","h_embed","h_argmax","k3a_iq3","k0ab","k2s","a_qkv_q6","a_qkv_iq3","k1_q5g"]
LS = (256,1,1)

def iq3s_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3s_grid
  vals = np.array([(w >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.float32)
  assert vals.size == 2048
  return vals

class TrunkEngine:
  """All 64 blocks + embed + head. Weights ~12.8GB on NV. Host RAM guarded by
  batched uploads (copyin -> synchronize -> drop numpy refs)."""
  def __init__(self, theta=10000.0):
    self.P = Bufs(); P = self.P
    ds, infos = parse_gguf()
    self.infos = infos
    self.attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
    assert len(self.attn_idx) == 16, self.attn_idx
    self.gdn_idx = [i for i in range(64) if i not in set(self.attn_idx)]
    self.qtypes = {}
    self.gdn_oq8 = {}
    # shared tables
    P.up("gridf", iq3_grid_f32())
    P.up("grid512", iq3s_grid_f32())
    freqs = (1.0 / (float(theta) ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
    P.up("freqs", freqs)
    # shared scratch
    for nm, nb, dt, pv in [("x0", DIM*4, np.float32, 7.7e31), ("x1", DIM*4, np.float32, 7.7e31),
                           ("xh", DIM*2, np.float16, 7.7), ("qkv_row", 10240*2, np.float16, 7.7),
                           ("gate_row", 6144*2, np.float16, 7.7), ("alpharaw", 48*4, np.float32, 7.7e31),
                           ("betaraw", 48*4, np.float32, 7.7e31), ("q", 48*128*4, np.float32, 7.7e31),
                           ("k", 48*128*4, np.float32, 7.7e31), ("v", 48*128*4, np.float32, 7.7e31),
                           ("core", 6144*4, np.float32, 7.7e31), ("z", 6144*2, np.float16, 7.7),
                           ("attn_out", DIM*2, np.float16, 7.7), ("hh", DIM*4, np.float32, 7.7e31),
                           ("hhx", DIM*2, np.float16, 7.7), ("gact", 17408*2, np.float16, 7.7),
                           ("qrow", 12288*2, np.float16, 7.7), ("k_row", 1024*2, np.float16, 7.7),
                           ("v_row", 1024*2, np.float16, 7.7), ("ao_row", 6144*2, np.float16, 7.7),
                           ("logits", VOCAB*2, np.float16, 7.7)]:
      P.poison(nm, nb, dt, pv)
    P.up("tok_slot", np.zeros(1, dtype=np.int32))
    P.up("pos_slot", np.zeros(1, dtype=np.int32))
    P.up("tok_hist", np.full(CTX+256, -1, dtype=np.int32))
    # per-block weights + state
    self.W = {}
    for i in self.gdn_idx:
      pre = f"blk.{i}."
      self.W[("qkv",i)] = self._upraw(pre+"attn_qkv.weight", ds)
      self.W[("gate",i)] = self._upraw(pre+"attn_gate.weight", ds)
      self.W[("out",i)] = self._upraw(pre+"ssm_out.weight", ds)
      self.gdn_oq8[i] = (infos[pre+"ssm_out.weight"][0] == 8)
      self.W[("fg",i)] = self._upraw(pre+"ffn_gate.weight", ds)
      self.W[("fu",i)] = self._upraw(pre+"ffn_up.weight", ds)
      self.W[("fd",i)] = self._upraw(pre+"ffn_down.weight", ds)
      self._upf32(("alpha",i), np.frombuffer(read_raw(infos[pre+"ssm_alpha.weight"], ds), dtype="<f4").reshape(48,5120))
      self._upf32(("beta",i), np.frombuffer(read_raw(infos[pre+"ssm_beta.weight"], ds), dtype="<f4").reshape(48,5120))
      self._upf32(("convw",i), np.frombuffer(read_raw(infos[pre+"ssm_conv1d.weight"], ds), dtype="<f4").reshape(10240,4))
      self._upf32(("dtb",i), np.frombuffer(read_raw(infos[pre+"ssm_dt.bias"], ds), dtype="<f4"))
      self._upf32(("ssma",i), np.frombuffer(read_raw(infos[pre+"ssm_a"], ds), dtype="<f4"))
      self._upf32(("nw1",i), np.frombuffer(read_raw(infos[pre+"attn_norm.weight"], ds), dtype="<f4"))
      self._upf32(("nw2",i), np.frombuffer(read_raw(infos[pre+"post_attention_norm.weight"], ds), dtype="<f4"))
      self._upf32(("snw",i), np.frombuffer(read_raw(infos[pre+"ssm_norm.weight"], ds), dtype="<f4"))
      for c in (0,1): P.poison(f"conv{i}_{c}", 3*10240*4, np.float32, 7.7e31)
      P.poison(f"rec{i}", 48*128*128*4, np.float32, 7.7e31)
      if i % 8 == 0: self._flush()
    for i in self.attn_idx:
      pre = f"blk.{i}."
      qt = infos[pre+"attn_q.weight"][0]
      assert qt in (14, 18), (i, qt)
      self.qtypes[i] = qt
      self.W[("q",i)] = self._upraw(pre+"attn_q.weight", ds)
      self.W[("k",i)] = self._upraw(pre+"attn_k.weight", ds)
      self.W[("v",i)] = self._upraw(pre+"attn_v.weight", ds)
      self.W[("o",i)] = self._upraw(pre+"attn_output.weight", ds)
      self.W[("fg",i)] = self._upraw(pre+"ffn_gate.weight", ds)
      self.W[("fu",i)] = self._upraw(pre+"ffn_up.weight", ds)
      self.W[("fd",i)] = self._upraw(pre+"ffn_down.weight", ds)
      self._upf32(("nw1",i), np.frombuffer(read_raw(infos[pre+"attn_norm.weight"], ds), dtype="<f4"))
      self._upf32(("nw2",i), np.frombuffer(read_raw(infos[pre+"post_attention_norm.weight"], ds), dtype="<f4"))
      self._upf32(("qnw",i), np.frombuffer(read_raw(infos[pre+"attn_q_norm.weight"], ds), dtype="<f4"))
      self._upf32(("knw",i), np.frombuffer(read_raw(infos[pre+"attn_k_norm.weight"], ds), dtype="<f4"))
      if _TKV8:
        P.up(f"kv{i}", np.zeros(2*4*CTX*256, dtype=np.uint8))       # biased int8
        P.up(f"sc{i}", np.zeros(2*4*CTX*8, dtype=np.float16))
      else:
        P.poison(f"kv{i}", 2*4*CTX*256*2, np.float16, 7.7)
      if i % 4 == 0: self._flush()
    # head + embed
    self.W[("head",0)] = self._upraw("output.weight", ds)
    self._upf32(("onw",0), np.frombuffer(read_raw(infos["output_norm.weight"], ds), dtype="<f4"))
    self.W[("emb",0)] = self._upraw("token_embd.weight", ds)
    self._flush()
    # programs (per-kernel cubins)
    self.pr = {}
    for n in GDN_CUBINS + NEW_CUBINS:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      self.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

  def _upraw(self, name, ds):
    arr = np.frombuffer(read_raw(self.infos[name], ds), dtype=np.uint8)
    return self.P.up(name.replace(".", "_"), arr)
  def _upf32(self, key, arr):
    self.W[key] = self.P.up(f"w_{key[0]}_{key[1]}", np.ascontiguousarray(arr))
  def _flush(self):
    dev.synchronize()
    self.P._keep.clear()   # DMA flushed; free host numpy (16GB host RAM guard)

  # ---- block launches ----
  def gdn(self, i, xin, xout, csrc, cdst, wait=False):
    W, pr, d = self.W, self.pr, self.P.d
    pr["k0_norm"](xin, W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS)
    pr["k1_q5"](W[("qkv",i)], d["xh"], d["qkv_row"], global_size=(1280,1,1), local_size=LS)
    pr["k1_iq3"](W[("gate",i)], d["gridf"], d["xh"], d["gate_row"], global_size=(768,1,1), local_size=LS)
    pr["k1_ab"](W[("alpha",i)], W[("beta",i)], d["xh"], d["alpharaw"], d["betaraw"], global_size=(12,1,1), local_size=LS)
    pr["k2_scan"](d[csrc], d[cdst], d["qkv_row"], d["gate_row"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                  d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d[f"rec{i}"], d["core"], global_size=(48,1,1), local_size=LS)
    pr["k2b_z"](d["core"], d["gate_row"], W[("snw",i)], d["z"], global_size=(48,1,1), local_size=LS)
    if self.gdn_oq8[i]: pr["k3a_oproj"](W[("out",i)], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    else: pr["k3a_iq3"](W[("out",i)], d["gridf"], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr["k3b_ffn"](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr["k3c_down"](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def attn(self, i, xin, xout, wait=False):
    W, pr, d = self.W, self.pr, self.P.d
    pr["k0_norm"](xin, W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS)
    if self.qtypes[i] == 14:
      pr["a_q6"](W[("q",i)], d["xh"], d["qrow"], global_size=(1536,1,1), local_size=LS)
    else:
      pr["k1_iq3"](W[("q",i)], d["gridf"], d["xh"], d["qrow"], global_size=(1536,1,1), local_size=LS)
    pr["a_kv"](W[("k",i)], W[("v",i)], d["gridf"], d["xh"], d["k_row"], d["v_row"], global_size=(256,1,1), local_size=LS)
    pr["a_attn"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"],
                 d[f"kv{i}"], d["pos_slot"], d["ao_row"], global_size=(24,1,1), local_size=LS)
    pr["a_o"](W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr["k3b_ffn"](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr["k3c_down"](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def head(self, xin, wait=False):
    d, W, pr = self.P.d, self.W, self.pr
    pr["k0_norm"](xin, W[("onw",0)], d["xh"], global_size=(1,1,1), local_size=LS)
    pr["k1_q5"](W[("head",0)], d["xh"], d["logits"], global_size=(VOCAB//8,1,1), local_size=LS)
    pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1,1,1), local_size=LS, wait=wait)

  def _build_seqs(self):
    """Prebuild the whole-token launch sequence per conv parity (host submission
    was ~78us/launch with per-call arg building; cached tuples cut it to the floor).
    Must be rebuilt after restore() (buffer handles change)."""
    d, W, pr = self.P.d, self.W, self.pr
    self._seq = {}
    for par in (0, 1):
      seq = [(pr["h_embed"], (W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"]), (1,))]
      cur = 0
      for i in range(64):
        xin, xout = d[f"x{cur}"], d[f"x{cur^1}"]
        if i in self.qtypes:
          qkname = "a_qkv_q6" if self.qtypes[i] == 14 else "a_qkv_iq3"
          a = [(pr["k0_norm"], (xin, W[("nw1",i)], d["xh"]), (1,)),
               (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"]), (1792,)),
               (pr["a_attn"], (d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], d["pos_slot"], d["ao_row"]), (24,))]
          a.append((pr["a_o"], (W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"]), (640,)))
          a.append((pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)))
          a.append((pr["k3b_ffn"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)))
          a.append((pr["k3c_down"], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,)))
        else:
          csrc, cdst = d[f"conv{i}_{par}"], d[f"conv{i}_{par^1}"]
          a = [(pr["k0ab"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh"], d["alpharaw"], d["betaraw"]), (13,)),
               (pr["k1_q5g"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"]), (2048,)),
               (pr["k2s"], (csrc, cdst, d["qkv_row"], d["gate_row"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                            d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d[f"rec{i}"], d["core"],
                            W[("snw",i)], d["z"]), (48,))]
          if self.gdn_oq8[i]:
            a.append((pr["k3a_oproj"], (W[("out",i)], d["z"], d["attn_out"]), (640,)))
          else:
            a.append((pr["k3a_iq3"], (W[("out",i)], d["gridf"], d["z"], d["attn_out"]), (640,)))
          a.append((pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)))
          a.append((pr["k3b_ffn"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)))
          a.append((pr["k3c_down"], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,)))
        seq += a
        cur ^= 1
      seq.append((pr["k0_norm"], (d["x0"], W[("onw",0)], d["xh"]), (1,)))
      seq.append((pr["k1_q5"], (W[("head",0)], d["xh"], d["logits"]), (VOCAB//8,)))
      seq.append((pr["h_argmax"], (d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"]), (1,)))
      self._seq[par] = seq

  def token(self, it, wait=False):
    """One decode step from the cached launch sequence. Mid-token dev.synchronize:
    >512 pipelined launches fault the dext (W1A proven max 480)."""
    if not hasattr(self, "_seq"): self._build_seqs()
    seq = self._seq[it & 1]
    for n, (p, a, g) in enumerate(seq):
      p(*a, global_size=(g[0],1,1), local_size=LS, wait=(wait and n == len(seq)-1))
      if n == 230: dev.synchronize()   # pipelined-launch ceiling ~312 proven on this dext

  def run_tokens(self, n):
    if getattr(self, "_submit_only", False):
      for it in range(n): self.token(it, wait=False)
      dev.synchronize()
      self._submit_only = False
      return
    for it in range(n):
      self.token(it, wait=(it == n-1))

  # ---- state restore (bootstrap snapshot) ----
  def restore(self, snap):
    P = self.P
    for j, i in enumerate(self.gdn_idx):
      P.up(f"conv{i}_0", snap[f"conv{j}"].astype(np.float32).reshape(-1))
      P.up(f"rec{i}", snap[f"rec{j}"].astype(np.float32).reshape(-1))
      if j % 8 == 0: self._flush()
    for j, i in enumerate(self.attn_idx):
      kvs = snap[f"kv{j}"].astype(np.float16).reshape(-1)
      # W3 FIX: attention kernels address KV slabs with CTXK=2304 stride; the
      # stock snapshot is 2048 rows -> undersized buffer = OOB slab writes (the
      # device-fault class at the 2nd spk_pre1). Pad to 2304 (restore_mtp parity).
      CTXK_PAD = 2304
      if kvs.size == 2*4*CTXK_PAD*256:
        P.up(f"kv{i}", kvs)
      else:
        kvbig = np.full(2*4*CTXK_PAD*256, 7.7, dtype=np.float16)
        kvbig[:kvs.size] = kvs
        P.up(f"kv{i}", kvbig)
      if j % 4 == 0: self._flush()
    P.up("tok_slot", np.array([int(snap["ids"].reshape(-1)[-1])], dtype=np.int32))
    P.up("pos_slot", np.array([int(snap["P"].reshape(-1)[0]) - 1], dtype=np.int32))
    P.up("tok_hist", np.full(CTX+256, -1, dtype=np.int32))
    self._flush()
    if hasattr(self, "_seq"): del self._seq   # buffer handles changed
