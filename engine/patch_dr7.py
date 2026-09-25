# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c decode-r7 wiring patch (fail-loud: every replacement asserts exactly 1 hit).
Applies to: trunk_w1c.py, mtp.py, pf_prefill.py, pcache.py. Idempotent-ish: greps first.
"""
import sys

def patch(path, subs):
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        assert n == 1, (path, n, old[:90])
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"[patch] {path}: {len(subs)} edits OK")

BASE = "~/tinygrad-metal/engine0"

# ---------------- trunk_w1c.py ----------------
patch(f"{BASE}/trunk_w1c.py", [
  # (a) header: DR7 flag + cubin list
  ('W1C_CUBINS = ["q5g8", "head8", "ffn8", "down8", "op38", "aq6k8", "aq3k8", "ao8"]',
   '''W1C_CUBINS = ["q5g8", "head8", "ffn8", "down8", "op38", "aq6k8", "aq3k8", "ao8"]

# R2c decode-r7 (PF_DR7=1): fg/fu/fd upload as packed7 and the decode/spec GEMVs
# read the SAME plane prefill's G3M tier uses (r7d.cu ports — BIT-IDENTICAL,
# r7d_test.py det x2 nz=0). The packed originals are NEVER uploaded -> the
# both-live VRAM wall is gone and prefill gets full m64 fg/fu coverage.
DR7 = bool(int(os.getenv("PF_DR7", "0")))
R7D_CUBINS = ["ffn8r7", "down8r7", "ffn8v3r7", "down8nw32v3r7", "ffn8v8r7", "down8nw32v8r7"]'''),
  # (b) __init__: _r7native before super().__init__ (weights load inside it)
  ('''    self._packed_phase = True
    super().__init__(theta)
    self._packed_phase = False
    for n in W1C_CUBINS:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      self.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))''',
   '''    self._packed_phase = True
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
        raise RuntimeError(f"PF_DR7: expected 192 r7-native tensors, got {len(self._r7native)}")'''),
  # (c) _upraw: fg/fu/fd diversion to packed7
  ('''        if t is not None:
          return self.P.up(name.replace(".", "_"), np.ascontiguousarray(np.load(f"{self.PACKED}/{t}{blk}.npy")))''',
   '''        if t is not None:
          if DR7 and t in ("fg", "fu", "fd"):
            p7 = f"{BASE}/packed7/{t}{blk}.npy"
            self._r7native.add((t, blk))
            return self.P.up(name.replace(".", "_"), np.ascontiguousarray(np.load(p7)))
          return self.P.up(name.replace(".", "_"), np.ascontiguousarray(np.load(f"{self.PACKED}/{t}{blk}.npy")))'''),
  # (d) helpers after _upraw
  ('''  def gdn(self, i, xin, xout, csrc, cdst, wait=False):
    W, pr, d = self.W, self.pr, self.P.d
    pr["k0_norm"](xin, W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS)
    pr["q5g8"](W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS)''',
   '''  def _ffnk(self, i):
    return "ffn8r7" if ("fg", i) in self._r7native else "ffn8"
  def _downk(self, i):
    return "down8r7" if ("fd", i) in self._r7native else "down8"

  def gdn(self, i, xin, xout, csrc, cdst, wait=False):
    W, pr, d = self.W, self.pr, self.P.d
    pr["k0_norm"](xin, W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS)
    pr["q5g8"](W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS)'''),
  # (e) gdn() ffn/down selection
  ('''    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr["ffn8"](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr["down8"](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def attn(self, i, xin, xout, wait=False):''',
   '''    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr[self._ffnk(i)](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr[self._downk(i)](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def attn(self, i, xin, xout, wait=False):'''),
  # (f) attn() ffn/down selection (the block after the ao8 launch in attn())
  ('''    pr["ao8"](W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr["ffn8"](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr["down8"](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def head(self, xin, wait=False):''',
   '''    pr["ao8"](W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
    pr[self._ffnk(i)](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS)
    pr[self._downk(i)](W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout, global_size=(640,1,1), local_size=LS, wait=wait)

  def head(self, xin, wait=False):'''),
  # (g) _build_seqs: attn-branch
  ('''          a += [(pr["ao8"], (W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"]), (640,)),
               (pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)),
               (pr["ffn8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)),
               (pr["down8"], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,))]''',
   '''          a += [(pr["ao8"], (W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"]), (640,)),
               (pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)),
               (pr[self._ffnk(i)], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)),
               (pr[self._downk(i)], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,))]'''),
  # (h) _build_seqs: gdn-branch (the one ending with xout at the seq level)
  ('''          a.append((pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)))
          a.append((pr["ffn8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)))
          a.append((pr["down8"], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,)))
        seq += a''',
   '''          a.append((pr["k3m_hh"], (xin, d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"]), (1,)))
          a.append((pr[self._ffnk(i)], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"]), (2176,)))
          a.append((pr[self._downk(i)], (W[("fd",i)], d["gridf"], d["gact"], d["hh"], xout), (640,)))
        seq += a'''),
])

# ---------------- mtp.py ----------------
patch(f"{BASE}/mtp.py", [
  # import DR7
  ('from trunk_w1c import TrunkEngineW1C',
   'from trunk_w1c import TrunkEngineW1C, DR7'),
  # K2 probe selections (probe_g): ffn + down (2 lines, contiguous)
  ('''             (pr["ffn8v4" if K3 else ("ffn8v_3" if GEMVV else "ffn8_3")], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
             (pr["down8nw32_4" if K3 else ("down8nw32_3" if GEMVV else "down8_3")], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160 if (K3 or GEMVV) else 640)]
      else:''',
   '''             (pr["ffn8v4" if K3 else (("ffn8v3r7" if DR7 else "ffn8v_3") if GEMVV else "ffn8_3")], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
             (pr["down8nw32_4" if K3 else (("down8nw32v3r7" if DR7 else "down8nw32_3") if GEMVV else "down8_3")], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160 if (K3 or GEMVV) else 640)]
      else:'''),
  ('''        a.append((pr["ffn8v4" if K3 else ("ffn8v_3" if GEMVV else "ffn8_3")], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_4" if K3 else ("down8nw32_3" if GEMVV else "down8_3")], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160 if (K3 or GEMVV) else 640))''',
   '''        a.append((pr["ffn8v4" if K3 else (("ffn8v3r7" if DR7 else "ffn8v_3") if GEMVV else "ffn8_3")], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_4" if K3 else (("down8nw32v3r7" if DR7 else "down8nw32_3") if GEMVV else "down8_3")], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160 if (K3 or GEMVV) else 640))'''),
  # _probe8_seq selections
  ('''              (pr["ffn8v8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32_8"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:''',
   '''              (pr["ffn8v8r7" if DR7 else "ffn8v8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32v8r7" if DR7 else "down8nw32_8"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:'''),
  ('''        a.append((pr["ffn8v8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_8"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))''',
   '''        a.append((pr["ffn8v8r7" if DR7 else "ffn8v8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32v8r7" if DR7 else "down8nw32_8"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))'''),
  # fail-loud guard at build_graphs entry
  ('''  def build_graphs(self):
    d, W, pr = self.P.d, self.W, self.pr''',
   '''  def build_graphs(self):
    d, W, pr = self.P.d, self.W, self.pr
    if DR7:
      # r7d.cu ports exist ONLY for the live families: K2-probe GEMVV (ffn8v_3 /
      # down8nw32_3) and the K=7 deep probe (ffn8v8 / down8nw32_8). Anything else
      # would read packed7 bytes through packed-layout kernels = silent garbage.
      assert GEMVV and not K3 and LOOKUP_K in (0, 7), \\
        "PF_DR7 requires GEMVV=1, K3 off, LOOKUP_K in (0,7)"'''),
])

# ---------------- pf_prefill.py ----------------
patch(f"{BASE}/pf_prefill.py", [
  # alias native r7 into W7 before the budget swap
  ('''    if os.getenv("PF_G3M_ORDER", "") == "ffn":   # P15: fg/fu-first (the m64 ffn rate probe)
      cands.sort(key=lambda c: (0 if c[0] == 2 else 1, -c[1]))
    else:
      cands.sort(key=lambda c: (c[0], -c[1]))
    used = 0; nk = 0
    for prio, sz, keys in cands:
      if used + sz > budget:
        continue
      for t, i, p in keys:
        W7[(t, i)] = P.up(f"r7_{t}_{i}", np.load(p)); nk += 1
      used += sz
      if nk % 16 == 15:
        dev.synchronize(); P._keep.clear()''',
   '''    if os.getenv("PF_G3M_ORDER", "") == "ffn":   # P15: fg/fu-first (the m64 ffn rate probe)
      cands.sort(key=lambda c: (0 if c[0] == 2 else 1, -c[1]))
    else:
      cands.sort(key=lambda c: (c[0], -c[1]))
    # R2c decode-r7: (t,i) the trunk ALREADY uploaded as packed7 (PF_DR7 — decode
    # reads the same plane) — alias the SAME buffers into W7: zero extra VRAM, so
    # the budget admits every remaining class => FULL m64 coverage.
    nat = getattr(E, "_r7native", set())
    for (t, i) in sorted(nat):
      W7[(t, i)] = E.W[(t, i)]
    used = 0; nk = len(W7)
    for prio, sz, keys in cands:
      if all((t, i) in W7 for (t, i, p) in keys):
        continue   # already native (or already uploaded)
      if used + sz > budget:
        continue
      for t, i, p in keys:
        if (t, i) in W7: continue
        W7[(t, i)] = P.up(f"r7_{t}_{i}", np.load(p)); nk += 1
      used += sz
      if nk % 16 == 15:
        dev.synchronize(); P._keep.clear()'''),
  # fail-loud guards: classic fallbacks would misread packed7 bytes
  ('''def ensure(E):
  if getattr(E, "_pf_plan", None) is not None:
    return''',
   '''def ensure(E):
  if getattr(E, "_pf_plan", None) is not None:
    return
  if getattr(E, "_r7native", None) and not G3M:
    raise RuntimeError("PF_DR7 requires the G3M tier (PF_G3M=1): the packed originals are not resident")'''),
  ('''def ensure_sc(E):
  if getattr(E, "_pfsc_plan", None) is not None:
    return''',
   '''def ensure_sc(E):
  if getattr(E, "_pfsc_plan", None) is not None:
    return
  if getattr(E, "_r7native", None):
    raise RuntimeError("PF_DR7 is incompatible with the SC path (PF_SUPER)")'''),
])

# ---------------- pcache.py ----------------
src = open(f"{BASE}/pcache.py").read()
if "PF_DR7" not in src:
  old = '"PF_ATTN32", "PF_N32", "PF_PRE32", "PF_SCAN32", "PF_A4", "PF_SCANC")'
  assert src.count(old) == 1
  src = src.replace(old, '"PF_ATTN32", "PF_N32", "PF_PRE32", "PF_SCAN32", "PF_A4", "PF_SCANC", "PF_DR7")')
  open(f"{BASE}/pcache.py", "w").write(src)
  print("[patch] pcache.py: PF_DR7 added to _ENV_KEYS")
else:
  print("[patch] pcache.py: PF_DR7 already present")
print("[patch all done]")
