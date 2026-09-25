# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c rung: split-plane FFN on the M64 trunk (PF_FFNSPLIT=1). The fused
pfg3_ffn_r7_m64_nw4k128 is smem-forced to nw4 (2 W planes, 18.3% MFU); the
split runs fg and fu as SINGLE-plane m64 nw8 GEMMs (34816B, the gdnqg-class
shape) + the pfk_smul64 epilogue — BIT-IDENTICAL by construction (plain
epilogue writes (half)acc; smul applies the identical fused sequence;
r2c_ffnsplit_test.py det x2 nz=0)."""
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

patch(f"{BASE}/pf_prefill.py", [
  # knob
  ('M64QKV = os.getenv("PF_M64QKV", "0") == "1"   # R2b rung-6 retry: the attnqkv m64 twins (P7E4-quarantined on the SC path; the M64 trunk is a different world)',
   '''M64QKV = os.getenv("PF_M64QKV", "0") == "1"   # R2b rung-6 retry: the attnqkv m64 twins (P7E4-quarantined on the SC path; the M64 trunk is a different world)
FFNSPLIT = os.getenv("PF_FFNSPLIT", "0") == "1"  # R2c: split-plane FFN m64 (fg/fu as single-plane nw8 GEMMs + pfk_smul64; bit-identical)'''),
  # ensure64: load the split cubins
  ('''  if SCANC:
    _wy = ["pfca_c32_nc2_nw16", "pfcb_c32_nc2_nw8", "pfcz_c32_nc2_nw8"] if SCANC_N2 else           ["pfca_c32_nc1_nw16", "pfcb_c32_nc1_nw8", "pfcz_c32_nc1_nw8"]''',
   '''  if FFNSPLIT:
    for n in ["pfg3_fgp_r7_m64_nw8k128", "pfg3_fup_r7_m64_nw8k128", "pfk_smul64"]:
      if n in _m64l: continue
      _m64l.add(n)
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  if SCANC:
    _wy = ["pfca_c32_nc2_nw16", "pfcb_c32_nc2_nw8", "pfcz_c32_nc2_nw8"] if SCANC_N2 else           ["pfca_c32_nc1_nw16", "pfcb_c32_nc1_nw8", "pfcz_c32_nc1_nw8"]'''),
  # scratch: ag64/au64
  ('''  P.up("ids64", np.zeros(M, dtype=np.int32))
  P.up("pos_arr64", np.zeros(1, dtype=np.int32))     # pre64: [pos0]
  P.up("pos_w64", np.zeros(4, dtype=np.int32))       # 4x16-row attention windows''',
   '''  if FFNSPLIT:
    P.poison("ag64", M*17408*2, np.float16, 7.7)
    P.poison("au64", M*17408*2, np.float16, 7.7)
  P.up("ids64", np.zeros(M, dtype=np.int32))
  P.up("pos_arr64", np.zeros(1, dtype=np.int32))     # pre64: [pos0]
  P.up("pos_w64", np.zeros(4, dtype=np.int32))       # 4x16-row attention windows'''),
  # plan: the M64 ffn branch
  ('''    if ("fg", i) in W7:
      A(pr["pfg3_ffn_r7_m64_nw4k128"], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx64"],
        d["gact64"], g=544, ls=(128, 1, 1))
    else:''',
   '''    if ("fg", i) in W7 and FFNSPLIT:
      A(pr["pfg3_fgp_r7_m64_nw8k128"], W7[("fg", i)], d["gridf"], d["hhx64"], d["ag64"], g=272, ls=LS)
      A(pr["pfg3_fup_r7_m64_nw8k128"], W7[("fu", i)], d["gridf"], d["hhx64"], d["au64"], g=272, ls=LS)
      A(pr["pfk_smul64"], d["ag64"], d["au64"], d["gact64"], g=544, ls=LS)
    elif ("fg", i) in W7:
      A(pr["pfg3_ffn_r7_m64_nw4k128"], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx64"],
        d["gact64"], g=544, ls=(128, 1, 1))
    else:'''),
  # graph-cache key
  ('  key = (M32, DFILL, G3M, NT32, ATTN32, A4, HYB, ATTN_THR, N32, PRE32, SCAN32, PERSIST, PERSIST_NAME, m64, ATTNW, SCANC, SCANC_N2, M64QKV, ABW)',
   '  key = (M32, DFILL, G3M, NT32, ATTN32, A4, HYB, ATTN_THR, N32, PRE32, SCAN32, PERSIST, PERSIST_NAME, m64, ATTNW, SCANC, SCANC_N2, M64QKV, ABW, FFNSPLIT)'),
])

src = open(f"{BASE}/pcache.py").read()
if "PF_FFNSPLIT" not in src:
  old = '"PF_SCANC", "PF_DR7")'
  assert src.count(old) == 1
  src = src.replace(old, '"PF_SCANC", "PF_DR7", "PF_FFNSPLIT")')
  open(f"{BASE}/pcache.py", "w").write(src)
  print("[patch] pcache.py: PF_FFNSPLIT added")
print("[patch all done]")
