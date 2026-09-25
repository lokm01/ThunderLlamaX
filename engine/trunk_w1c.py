# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""engine0 W1-c: TrunkEngine with WIDE-LOAD GEMV kernels (u64/u32/u16 loads; no
repack — layouts unchanged, math bit-identical, see w1c.cu)."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from trunk import TrunkEngine, LS, DIM, CTX, VOCAB, GDN_CUBINS, NEW_CUBINS
from engine0 import read_raw
from engine0 import dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
W1C_CUBINS = ["q5g8", "head8", "ffn8", "down8", "op38", "aq6k8", "aq3k8", "ao8"]

# R2c decode-r7 (PF_DR7=1): fg/fu/fd upload as packed7 and the decode/spec GEMVs
# read the SAME plane prefill's G3M tier uses (r7d.cu ports — BIT-IDENTICAL,
# r7d_test.py det x2 nz=0). The packed originals are NEVER uploaded -> the
# both-live VRAM wall is gone and prefill gets full m64 fg/fu coverage.
DR7 = bool(int(os.getenv("PF_DR7", "0")))
R7D_CUBINS = ["ffn8r7", "down8r7", "ffn8v3r7", "down8nw32v3r7", "ffn8v8r7", "down8nw32v8r7"]

class TrunkEngineW1C(TrunkEngine):
  PACKED = "~/tinygrad-metal/engine0/packed"

  def __init__(self, theta=10000.0):
    self._packed_phase = True
    self._r7native = set()   # R2c: (t, i) uploaded as packed7 (decode reads the same plane)
    super().__init__(theta)
    self._packed_phase = False
    for n in W1C_CUBINS:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      self.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    if DR7:
      for n in R7D_CUBINS:
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        self.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if len(self._r7native) != 3 * 64:
        raise RuntimeError(f"PF_DR7: expected 192 r7-native tensors, got {len(self._r7native)}")
    # ---- W3 split-KV (env-gated; SKV=1 SKV_CTXK=2304|100352) ----
    import os as _os
    self.SKV = bool(int(_os.getenv("SKV", "0")))
    self.CTXK = int(_os.getenv("SKV_CTXK", "2304"))
    self.SKV_S = int(_os.getenv("SKV_S", "32"))
    self.KV8 = bool(int(_os.getenv("KV8", "0")))
    self.QH = bool(int(_os.getenv("QH", "0")))   # W2F: half2-QK K1 + KPRE qw16
    if self.SKV:
      suf = "2k" if self.CTXK <= 2304 else "100k"
      skv_k = _os.getenv("SKV_K", "a")   # "a" = W3 K1S | "g2" = SKV-G (spk_g.cu)
      if self.KV8:
        assert suf == "100k", "KV8 int8-KV kernels built for 100k only"
        if self.QH:
          k1n = ("spk_g4nw32qh1p_100k" if bool(int(_os.getenv("PVH", "0"))) else "spk_g4nw32qh1_100k")
          pren = "spk_pre1qh_100k"
        else:
          k1n, pren = "spk_g4nw32qa1_100k", "spk_pre1q_100k"
      else:
        k1n = (f"spk_{skv_k}a1_" if skv_k != "a" else "spk_a1_") + suf
        pren = "spk_pre1_" + suf
      if suf == "2k": cn = "spk_c1"
      elif skv_k == "a": cn = "spk_c1_100k"
      else: cn = f"spk_c1g{0 if self.SKV_S == 256 else self.SKV_S}_100k".replace("g0", "g")
      for key, n in (("spk_pre1", pren), ("spk_a1", k1n), ("spk_c1", cn)):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      P = self.P
      S = self.SKV_S
      self.SKV_LS = (1024,1,1) if "nw32" in k1n else (768,1,1) if "nw24" in k1n else (512,1,1) if "nw16" in k1n else LS
      P.poison("qw1", 24*256*4, np.float32, 7.7e31)
      if self.QH: P.poison("qw16_1", 24*256*2, np.float16, 7.7)
      P.poison("pm1", 4*S*6*4, np.float32, 7.7e31)
      P.poison("ps1", 4*S*6*4, np.float32, 7.7e31)
      P.poison("pA1", 4*S*6*256*4, np.float32, 7.7e31)
      self._flush()

  # packed-weight loading: divert gate/fg/fu/fd/iq3-ssm_out/q/k to engine0/packed/*.npy
  def _upraw(self, name, ds):
    if getattr(self, "_packed_phase", False):
      parts = name.split(".")
      if len(parts) >= 4:
        blk, sub = int(parts[1]), parts[2]
        t = {"attn_gate": "gate", "ffn_gate": "fg", "ffn_up": "fu", "ffn_down": "fd",
             "ssm_out": "out", "attn_q": "q", "attn_k": "k"}.get(sub)
        if t == "out" and self.infos[name][0] != 18: t = None
        if t is not None:
          if DR7 and t in ("fg", "fu", "fd"):
            p7 = f"{BASE}/packed7/{t}{blk}.npy"
            self._r7native.add((t, blk))
            return self.P.up(name.replace(".", "_"), np.ascontiguousarray(np.load(p7)))
          return self.P.up(name.replace(".", "_"), np.ascontiguousarray(np.load(f"{self.PACKED}/{t}{blk}.npy")))
    arr = np.frombuffer(read_raw(self.infos[name], ds), dtype=np.uint8)
    return self.P.up(name.replace(".", "_"), arr)

  def _ffnk(self, i):
    return "ffn8r7" if ("fg", i) in self._r7native else "ffn8"
  def _downk(self, i):
    return "down8r7" if ("fd", i) in self._r7native else "down8"

  def gdn(self, i, xin, xout, csrc, cdst, wait=False):
    W, pr, d = self.W, self.pr, self.P.d
    pr["k0_norm"](xin, W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS)
    pr["q5g8"](W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS)
    pr["k2s"](d[csrc], d[cdst], d["qkv_row"], d["gate_row"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
              d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d[f"rec{i}"], d["core"],
              W[("snw",i)], d["z"], global_size=(48,1,1), local_size=LS)
    if self.gdn_oq8[i]: pr["k3a_oproj"](W[("out",i)], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    else: pr["op38"](W[("out",i)], d["gridf"], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr[self._ffnk(i)](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr[self._downk(i)](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def attn(self, i, xin, xout, wait=False):
    W, pr, d = self.W, self.pr, self.P.d
    pr["k0_norm"](xin, W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS)
    qkname = "aq6k8" if self.qtypes[i] == 14 else "aq3k8"
    pr[qkname](W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS)
    if self.SKV:
      sca = (d[f"sc{i}"],) if self.KV8 else ()
      pr["spk_pre1"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"],
                  d[f"kv{i}"], sca[0] if sca else None, d["pos_slot"], d["qw1"], global_size=(24,1,1), local_size=LS) if False else pr["spk_pre1"](*((d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"]) + sca + (d["pos_slot"], d["qw1"]) + ((d["qw16_1"],) if self.QH else ())), global_size=(24,1,1), local_size=LS)
      pr["spk_a1"](*((d[f"kv{i}"],) + sca + ((d["qw16_1"],) if self.QH else d["qw1"],) + (d["pos_slot"], d["pm1"], d["ps1"], d["pA1"])), global_size=(4*self.SKV_S,1,1), local_size=self.SKV_LS)
      pr["spk_c1"](d["pm1"], d["ps1"], d["pA1"], d["qrow"], d["ao_row"], global_size=(24,1,1), local_size=LS)
    else:
      pr["a_attn"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"],
                   d[f"kv{i}"], d["pos_slot"], d["ao_row"], global_size=(24,1,1), local_size=LS)
    pr["ao8"](W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr[self._ffnk(i)](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr[self._downk(i)](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def head(self, xin, wait=False):
    d, W, pr = self.P.d, self.W, self.pr
    pr["k0_norm"](xin, W[("onw",0)], d["xh"], global_size=(1,1,1), local_size=LS)
    pr["head8"](W[("head",0)], d["xh"], d["logits"], global_size=(VOCAB//8,1,1), local_size=LS)
    pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1,1,1), local_size=LS, wait=wait)

  def _build_seqs(self):
    d, W, pr = self.P.d, self.W, self.pr
    self._seq = {}
    for par in (0, 1):
      seq = [(pr["h_embed"], (W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"]), (1,))]
      cur = 0
      for i in range(64):
        xin, xout = d[f"x{cur}"], d[f"x{cur^1}"]
        if i in self.qtypes:
          qkname = "aq6k8" if self.qtypes[i] == 14 else "aq3k8"
          a = [(pr["k0_norm"], (xin, W[("nw1",i)], d["xh"]), (1,)),
               (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"]), (1792,))]
          if self.SKV:
            sca = (d[f"sc{i}"],) if self.KV8 else ()
            a += [(pr["spk_pre1"], (d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw1"], *((d["qw16_1"],) if self.QH else ())), (24,)),
                  (pr["spk_a1"], (d[f"kv{i}"], *sca, *((d["qw16_1"],) if self.QH else d["qw1"]), d["pos_slot"], d["pm1"], d["ps1"], d["pA1"]), (4*self.SKV_S,)),
                  (pr["spk_c1"], (d["pm1"], d["ps1"], d["pA1"], d["qrow"], d["ao_row"]), (24,))]
          else:
            a += [(pr["a_attn"], (d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], d["pos_slot"], d["ao_row"]), (24,))]
          a += [(pr["ao8"], (W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"]), (640,)),
               (pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)),
               (pr[self._ffnk(i)], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)),
               (pr[self._downk(i)], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,))]
        else:
          csrc, cdst = d[f"conv{i}_{par}"], d[f"conv{i}_{par^1}"]
          a = [(pr["k0ab"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh"], d["alpharaw"], d["betaraw"]), (13,)),
               (pr["q5g8"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"]), (2048,)),
               (pr["k2s"], (csrc, cdst, d["qkv_row"], d["gate_row"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                            d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d[f"rec{i}"], d["core"],
                            W[("snw",i)], d["z"]), (48,))]
          if self.gdn_oq8[i]:
            a.append((pr["k3a_oproj"], (W[("out",i)], d["z"], d["attn_out"]), (640,)))
          else:
            a.append((pr["op38"], (W[("out",i)], d["gridf"], d["z"], d["attn_out"]), (640,)))
          a.append((pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)))
          a.append((pr[self._ffnk(i)], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)))
          a.append((pr[self._downk(i)], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,)))
        seq += a
        cur ^= 1
      seq.append((pr["k0_norm"], (d["x0"], W[("onw",0)], d["xh"]), (1,)))
      seq.append((pr["head8"], (W[("head",0)], d["xh"], d["logits"]), (VOCAB//8,)))
      seq.append((pr["h_argmax"], (d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"]), (1,)))
      self._seq[par] = seq
# appended loader for trunk_w1c.py

def pf_ffn16_program(name="pfg_ffn_hm_nw8k128"):
  """P1 prefill pGEMM (PF_GEMM=1 env gate): fused FFN gate+up M=16 batched kernel.
  Validated vs ffn8 row-by-row on real packed weights (F-norm ~5e-4, W2H-HMMA class).
  Launch: pf(w_fg, w_fu, gridf, x16[16][5120] fp16, out16[16][17408] fp16,
  global_size=(17408//64,1,1), local_size=(256,1,1))"""
  if os.getenv("PF_GEMM") != "1":
    raise RuntimeError("pf_ffn16_program requires PF_GEMM=1")
  lib = open(f"{BASE}/{name}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=name, target=dev.renderer.target, signature=tuple()))
