#!/usr/bin/env python3
"""MM SESSION D -- wire MM_PFW (the wide PF attention) into production:
  1. Rig7.__init__: load spkqw4_98304 (+ QMD dyn-smem patch) when present.
  2. build_seq7: the PFW arm (spka + spkqw4 + spkc(NP=S)) for PF graphs.
  3. engine0/svc_fp.py: MM_PFW joins the config-fingerprint env keys.
Idempotent (marker-scanned)."""
import re, sys

LIB = "~/tinygrad-metal/MM_P7_lib.py"
SVC = "~/tinygrad-metal/engine0/svc_fp.py"

src = open(LIB).read()

LOADER = '''
        # ---- SESSION D (L4): the row-grouped wide PF attention (spkqw4 --
        # POC 2.01x @96k, 1.81x @16k; Tier-2 reassociation class, gated
        # MM_PFW). Cubin PREBUILT (engine0/mm/mm_build_d.zsh). The QMD
        # dyn-smem patch (Session-A law) MUST land before any graph build.
        D_LSZ = {"spkqw4_98304": (256, 1, 1)}
        _nd = 0
        for _sym, _lsz in D_LSZ.items():
            _cb = f"{BASE}/MM_D_{_sym}.cubin"
            if not os.path.exists(_cb):
                continue
            _lib = open(_cb, "rb").read()
            _prg = NVProgram(dev, TinyELF(lib=_lib, name=_sym,
                                          target=dev.renderer.target,
                                          signature=(self.INT_SIG,)))
            _dyn = max(4 * 8 * 256 * 4, 64 * 528)     # the qq/tile union
            _prg.qmd.write(shared_memory_size=((_prg.shmem_usage + 127) // 128 * 128) + _dyn,
                           min_sm_config_shared_mem_size=100 * 1024 // 4096 + 1,
                           target_sm_config_shared_mem_size=100 * 1024 // 4096 + 1)
            self.K[_sym] = _prg
            L56.LSZ[_sym] = _lsz; _nd += 1
        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-D: {_nd}/1 wide-PF-attn programs", flush=True)
'''

if "session-D" not in src:
    anchor = '        print(f"[rig7] session-C: {_nc}/2 programs (mma M-GEMM + dn fold)", flush=True)\n'
    assert anchor in src, "session-C loader anchor not found"
    src = src.replace(anchor, anchor + LOADER, 1)
    print("[wire] loader block inserted")
else:
    print("[wire] loader block present")

# ---- the PFW arm in build_seq7 ----
OLD = '''            if spk is not None and spk.startswith("s"):
                R = spk[1:]
                seq.append((f"spka256m_{R}", (B["kqb"], B["vqb"], rig.SPTB[(ai, "m")]), (2*P,), ()))
                scr = rig.SCR_PF if pf else rig.SCR_DEC
                seq.append((f"spkq256s_{R}", (B["qgb"], scr, ptbl), (16*P, S), (S,)))
                seq.append(("spkc256", (B["qgb"], B["ayb"], scr), (16*P,), (8*S,)))'''
NEW = '''            scr = rig.SCR_PF if pf else rig.SCR_DEC
            # SESSION D (L4): the row-grouped wide PF attention -- PF graphs
            # only (decode keeps the stock split kernel). Partials land in
            # the NP=S layout -> spkc256 merges exactly the S real slots.
            # Tier-2 numerics (reassociation within splits, S pinned);
            # kill-switch MM_PFW=0 restores the stock pair byte-for-byte.
            PFW = (pf and os.getenv("MM_PFW", "0") == "1" and P in (256, 64)
                   and spk is not None and spk.startswith("s")
                   and f"spkqw4_{spk[1:]}" in rig.K)
            if PFW:
                R = spk[1:]
                seq.append((f"spka256m_{R}", (B["kqb"], B["vqb"], rig.SPTB[(ai, "m")]), (2*P,), ()))
                seq.append((f"spkqw4_{R}", (B["qgb"], scr, ptbl), (P // 4, 2 * S), (S,)))
                seq.append(("spkc256", (B["qgb"], B["ayb"], scr), (16*P,), (S,)))
            elif spk is not None and spk.startswith("s"):
                R = spk[1:]
                seq.append((f"spka256m_{R}", (B["kqb"], B["vqb"], rig.SPTB[(ai, "m")]), (2*P,), ()))
                seq.append((f"spkq256s_{R}", (B["qgb"], scr, ptbl), (16*P, S), (S,)))
                seq.append(("spkc256", (B["qgb"], B["ayb"], scr), (16*P,), (8*S,)))'''
if "spkqw4_" not in src.split("def build_seq7")[1]:
    assert OLD in src, "build_seq7 attention anchor not found"
    src = src.replace(OLD, NEW, 1)
    print("[wire] PFW arm wired into build_seq7")
else:
    print("[wire] PFW arm present")

open(LIB, "w").write(src)

# ---- svc_fp key ----
svc = open(SVC).read()
if '"MM_PFW"' not in svc:
    svc = svc.replace('"MM_PFT",', '"MM_PFT", "MM_PFW",', 1)
    open(SVC, "w").write(svc)
    print("[wire] svc_fp key added")
else:
    print("[wire] svc_fp key present")
print("[wire] done")
