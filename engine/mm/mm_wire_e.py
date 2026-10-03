#!/usr/bin/env python3
"""MM SESSION E -- wire MM_PFU (the gathered-row-list mma routed gate+up)
into production:
  1. Rig7.__init__: load the gxu_gm cubin when present (plain loader --
     static 39KB smem, no QMD dyn patch needed).
  2. build_seq7: the PFU arm replaces the gxm_up launch (IQ3_S lane only;
     the 1 IQ4_XS layer keeps gxm_up4) for PF graphs.
  3. svc_fp.py: MM_PFU joins the config-fingerprint env keys (Tier-2
     numerics -> pcache namespace flip: ONE cold rebuild on first boot).
Idempotent (marker-scanned). POC evidence: engine0/mm/mm_e1_poc.json --
relerr 4.1e-4, det x2, isolated 1.43x, in-chunk d=49.1ms/chunk (x1.86
implied) at 2k."""
import sys

BASE = "~/tinygrad-metal"
LIB = BASE + "/MM_P7_lib.py"
SVC = BASE + "/engine0/svc_fp.py"
src = open(LIB).read()

# ---- 1. loader: session-E block after the session-D block ----
OLD_LD = '''        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-D: {_nd}/1 wide-PF-attn programs", flush=True)
'''
NEW_LD = '''        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-D: {_nd}/1 wide-PF-attn programs", flush=True)

        # ---- SESSION E (routed-up mma): the gathered-row-list gate+up
        # (gxu_gm -- M=W-rows dense mma tiles, N=gathered 16-token tiles
        # from the mmsort8 bins; POC 1.43x isolated / x1.86 implied
        # in-chunk, F 4.1e-4, det x2; gate MM_PFU). Static 39KB smem.
        E_LSZ = {"gxu_gm": (256, 1, 1)}
        _ne = 0
        for _sym, _lsz in E_LSZ.items():
            _cb = f"{BASE}/MM_E_{_sym}.cubin"
            if not os.path.exists(_cb):
                continue
            _lib = open(_cb, "rb").read()
            self.K[_sym] = NVProgram(dev, TinyELF(lib=_lib, name=_sym,
                                                  target=dev.renderer.target,
                                                  signature=()))
            L56.LSZ[_sym] = _lsz; _ne += 1
        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-E: {_ne}/1 routed-up mma programs", flush=True)
'''
if "gxu_gm" not in src:
    assert OLD_LD in src, "session-D loader tail anchor not found"
    src = src.replace(OLD_LD, NEW_LD, 1)
    print("[wireE] loader added")
else:
    print("[wireE] loader present")

# ---- 2. the PFU flag ----
OLD_F = '''    PFK = (pf and os.getenv("MM_PFK", "0") == "1" and P in (256, 64)
           and "pgmq8k2_r8192" in rig.K)
    if PFG or PFM or PFT:'''
NEW_F = '''    PFK = (pf and os.getenv("MM_PFK", "0") == "1" and P in (256, 64)
           and "pgmq8k2_r8192" in rig.K)
    # SESSION E: the gathered-row-list routed gate+up mma
    PFU = (pf and os.getenv("MM_PFU", "0") == "1" and P in (256, 64)
           and "gxu_gm" in rig.K)
    if PFG or PFM or PFT or PFU:'''
if "MM_PFU" not in src:
    assert OLD_F in src, "PFK flag anchor not found"
    src = src.replace(OLD_F, NEW_F, 1)
    print("[wireE] PFU flag added")
else:
    print("[wireE] flag present")

# ---- 3. _need extension ----
OLD_N = '''                ["pgmq8m32"] if PFT else [])'''
NEW_N = '''                ["pgmq8m32"] if PFT else []) + (
                ["gxu_gm"] if PFU else [])'''
if "gxu_gm\"] if PFU" not in src:
    assert OLD_N in src, "_need anchor not found"
    src = src.replace(OLD_N, NEW_N, 1)
    print("[wireE] _need extended")
else:
    print("[wireE] _need present")

# ---- 4. the gxm_up arm ----
OLD_A = '''                gup = "gxm_up4" if upk == "gx8e256up4" else "gxm_up"
                gex = (rig.iq4nl,) if gup == "gxm_up4" else (rig.gridf,)
                seq.append((gup, (rig.PTB_UP[L], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                  rig.PLISTB, B["hnb"]) + gex + (B["actb"],), 1024, ()))'''
NEW_A = '''                gup = "gxm_up4" if upk == "gx8e256up4" else "gxm_up"
                gex = (rig.iq4nl,) if gup == "gxm_up4" else (rig.gridf,)
                # SESSION E (MM_PFU): the IQ3_S routed gate+up becomes the
                # gathered-row-list mma (gxu_gm) -- grid 2048 = 256 experts
                # x 8 W-row-tiles, token-tiles looped in-kernel; eoff/plist
                # drive everything (items/nit UNUSED by the mma path).
                # Tier-2: F 4.1e-4 + CE-gated. The 1 IQ4_XS layer keeps
                # gxm_up4. Kill-switch MM_PFU=0 = gxm_up verbatim.
                if PFU and gup == "gxm_up":
                    seq.append(("gxu_gm", (rig.PTB_UP[L], rig.EOFFB, rig.PLISTB,
                                           B["hnb"], rig.gridf, B["actb"]), 2048, ()))
                else:
                    seq.append((gup, (rig.PTB_UP[L], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                      rig.PLISTB, B["hnb"]) + gex + (B["actb"],), 1024, ()))'''
if "gxu_gm\", (rig.PTB_UP[L]" not in src:
    assert OLD_A in src, "gxm_up arm anchor not found"
    src = src.replace(OLD_A, NEW_A, 1)
    print("[wireE] gxm_up arm wired")
else:
    print("[wireE] arm present")

open(LIB, "w").write(src)

# ---- 5. svc_fp key ----
ssrc = open(SVC).read()
OLD_S = '''    "MM_PFT", "MM_PFW", "MM_PFK",'''
NEW_S = '''    "MM_PFT", "MM_PFW", "MM_PFK",
    # MM SESSION E: the gathered-row-list routed gate+up mma (Tier-2
    # numerics move, F~4e-4 class). config_fp key -> pcache namespace
    # flip: ONE cold MoE pcache rebuild on first boot (slogged).
    "MM_PFU",'''
if "MM_PFU" not in ssrc:
    assert OLD_S in ssrc, "svc_fp anchor not found"
    ssrc = ssrc.replace(OLD_S, NEW_S, 1)
    open(SVC, "w").write(ssrc)
    print("[wireE] svc_fp key added")
else:
    print("[wireE] svc_fp present")
print("[wireE] done")
