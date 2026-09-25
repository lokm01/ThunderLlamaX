# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX W4.4 manifest EMITTER — generates manifests/manifest_k{K}.json for the
shipped rung set from the gen scripts' ground truth:
  - trunk kernel tables parsed from the m{M}.cu sources (the gen_mN audit
    product — names + launch bounds are what the generators verified);
  - accept/acceptsel/lookup siblings + the deep attention set (ROWS=M) names;
  - scratch/emit/dring layout laws (mtp.py constants, K-derived).

Retrofit run (no GPU, no engine import):
    python3 gen_manifest.py            # writes manifests/ for k2 + k4..k10
Each manifest carries the per-kernel launch law (local_size from the cubin's
__launch_bounds__ — cross-checked live by the W4.1 exec tripwire).
"""
import json, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(BASE, "manifests")
KER = re.compile(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(')

def m_cu_kernels(M, expect=None):
  """Kernel (name, launch_bounds) table from m{M}.cu — the generator-audited set.
  m3.cu predates the single-file convention (10 kernels; ao8_3/aq*/aattn3/
  head8_3/amx3 live in their own .cu files) — the K0 manifest adds those."""
  src = open(f"{BASE}/m{M}.cu").read()
  ks = KER.findall(src)
  assert len(ks) == (expect or (10 if M == 3 else 14)), \
    f"m{M}.cu: expected {expect or (10 if M == 3 else 14)} kernels, found {len(ks)}: {[k[1] for k in ks]}"
  return {n: int(b) for b, n in ks}

# GEMVV=1 swaps these M3-family kernels for the v2 half2 variants (W2D L1).
GEMVV_ALT = {"q5g8_3": "q5g8v_3", "ffn8_3": "ffn8v_3", "down8_3": "down8nw32_3",
             "op38_3": "op38nw32_3", "k3ao3": "k3aonw32_3", "ao8_3": "ao8nw32_3",
             "aq3k8_3": "aq3k8v_3", "head8_3": "head8v_3"}

def dr7_pair(M):
  """The DR7 ffn/down cubin names for rung M (None where none were built).
  Shipped DR7 rungs: M=3 (ffn8v_3-world), M=8, M=10, M=11 (r7-named in the cu),
  M=9 via ffn8v9r7/down8nw32v9r7."""
  if M in (8, 9, 10, 11): return [f"ffn8v{M}r7", f"down8nw32v{M}r7"]
  if M == 3: return ["ffn8v3r7", "down8nw32v3r7"]
  return []

def emit_manifest(K):
  if K == 0:
    # the legacy k2 world: M3 probe + accept/acceptsel (+ optional R3 lookup)
    trunk = dict(m_cu_kernels(3))
    trunk.update({n: 256 for n in ("ao8_3", "aq6k8_3", "aq3k8_3", "aattn3", "head8_3", "amx3")})
    man = {"K": 0, "M": 3, "rung": "k2-legacy",
           "trunk_cu": trunk,
           "trunk_gemvv_alt": GEMVV_ALT,
           "dr7_cu": dr7_pair(3),
           "accept": "accept", "acceptsel": "acceptsel",
           "lookup": "lookup_nw32",            # R3 LOOKUP=1 (not under LOOKUP_K)
           "attn_deep": [],
           "scratch_conv": [], "scratch_rec": [],
           "drings": 2, "emit_stop_off": 5, "emit_words": 8,
           "amds_grid": 3}
  else:
    M = K + 1
    man = {"K": K, "M": M, "rung": f"deep-k{K}",
           "trunk_cu": m_cu_kernels(M),
           "trunk_gemvv_alt": {},
           "dr7_cu": dr7_pair(M),
           "accept": f"accept{M}k", "acceptsel": f"acceptsel{M}k",
           "lookup": f"lookup{M}_nw32",
           "attn_deep": [f"spk_pre{M}qh_100k", f"spk_g4nw32hm{M}_100k", f"spk_c{M}g_100k"],
           # k2s{M} conv windows t=4..K -> conv5x..conv{M}x; rec t=5..K -> rec6x..rec{M}x
           "scratch_conv": [f"conv{i}x" for i in range(5, M + 1)],
           "scratch_rec": [f"rec{i}x" for i in range(6, M + 1)],
           "drings": K, "emit_stop_off": K + 3, "emit_words": K + 6,
           "amds_grid": M}
    # existence checks against the built cubins (the retrofit gate)
    for role in ("accept", "acceptsel", "lookup"):
      n = man[role]
      assert os.path.exists(f"{BASE}/{n}.cubin"), f"K={K}: {role} cubin {n}.cubin MISSING on disk"
    for n in man["attn_deep"]:
      assert os.path.exists(f"{BASE}/{n}.cubin"), f"K={K}: deep attention cubin {n}.cubin MISSING"
  os.makedirs(MDIR, exist_ok=True)
  p = os.path.join(MDIR, f"manifest_k{'k2' if K == 0 else K}.json")
  json.dump(man, open(p, "w"), indent=1, sort_keys=True)
  print(f"[manifest] K={K} M={man['M']} -> {os.path.basename(p)} "
        f"(trunk={len(man['trunk_cu'])} acc={man['accept']} lu={man['lookup']})")
  return man

if __name__ == "__main__":
  ks = [int(a) for a in sys.argv[1:]] or [0, 4, 5, 6, 7, 8, 9, 10]
  for k in ks: emit_manifest(k)
  # every emitted manifest must self-validate through the loader
  sys.path.insert(0, BASE)
  from rung_manifest import load_manifest, emit_offsets, audit_k2s
  for k in ks:
    man = load_manifest(k)
    assert man["emit_stop_off"] == emit_offsets(k)["stop"]
    if k > 0:
      src = open(f"{BASE}/m{k+1}.cu").read()
      audit_k2s(src, k + 1)
      print(f"[manifest] k2s{k+1} textual audits PASS (2 barriers, t-loop, rec-chain)")
  print("[manifest] ALL RUNGS EMITTED + AUDITED")
