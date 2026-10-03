#!/usr/bin/env python3
"""MM SESSION E port 2 -- wire the remaining Session-E kernels:
  MM_PFD: the gathered dn mma (gxd_gm; the IQ4_NL lane -- the 3 Q6_K
          layers keep gxm_dn6)
  MM_PFR: the shared-expert mma pair (shgm512/sdm2048; the gold ROUTER
          stays stock rt8e256 -- the M-batch port FAILED the bit-exact
          contract and its 2.5ms pool did not justify Tier-2 erosion)
  MM_PFS: the split-row scan (k2s36h_{256,64} + k2nz36; Tier-2 yss
          regroup only, S bit-exact)
All Tier-2 (F/CE-gated) except the scan's S path. Loader + arms + svc_fp.
ONE SYMBOL PER CUBIN (the loader law -- 2-symbol cubins mispick metadata
-> OOR faults). Idempotent (marker-scanned)."""
import sys

BASE = "~/tinygrad-metal"
LIB = BASE + "/MM_P7_lib.py"
SVC = BASE + "/engine0/svc_fp.py"
src = open(LIB).read()

# ---- 1. loader: extend the session-E block ----
OLD_LD = '''        E_LSZ = {"gxu_gm": (256, 1, 1)}'''
NEW_LD = '''        E_LSZ = {"gxu_gm": (256, 1, 1), "gxd_gm": (256, 1, 1),
                 "shgm512": (256, 1, 1), "sdm2048": (256, 1, 1),
                 "k2s36h_256": (256, 1, 1), "k2s36h_64": (256, 1, 1),
                 "k2nz36": (128, 1, 1)}'''
if "gxd_gm" not in src:
    assert OLD_LD in src, "session-E loader anchor not found"
    src = src.replace(OLD_LD, NEW_LD, 1)
    # the loader loop reads MM_E_{sym}.cubin -- symbol names match files
    print("[wireE2] loader extended")
else:
    print("[wireE2] loader present")

# scan scratch (alloc once, after the E loader block)
OLD_SC = '''        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-E: {_ne}/1 routed-up mma programs", flush=True)'''
NEW_SC = '''        self.LSZ7 = dict(L56.LSZ)
        # SESSION E scratch: the split-row scan's YQB/YSSB (+4MB)
        if "k2s36h_256" in self.K:
            self.YQB = self.alloc(256 * 4096 * 4); self.keep.append(self.YQB)
            self.YSSB = self.alloc(256 * 64 * 4); self.keep.append(self.YSSB)
        dev.synchronize()
        print(f"[rig7] session-E: {_ne} routed-up/dn/shared/scan programs", flush=True)'''
if "self.YQB" not in src:
    assert OLD_SC in src, "session-E loader tail anchor not found"
    src = src.replace(OLD_SC, NEW_SC, 1)
    print("[wireE2] scan scratch added")
else:
    print("[wireE2] scratch present")

# ---- 2. flags ----
OLD_F = '''    PFU = (pf and os.getenv("MM_PFU", "0") == "1" and P in (256, 64)
           and "gxu_gm" in rig.K)
    if PFG or PFM or PFT or PFU:'''
NEW_F = '''    PFU = (pf and os.getenv("MM_PFU", "0") == "1" and P in (256, 64)
           and "gxu_gm" in rig.K)
    # SESSION E port 2: the gathered dn mma (PFD), the shared-expert mma
    # pair (PFR), the split-row scan (PFS)
    PFD = (pf and os.getenv("MM_PFD", "0") == "1" and P in (256, 64)
           and "gxd_gm" in rig.K)
    PFR = (pf and os.getenv("MM_PFR", "0") == "1" and P in (256, 64)
           and "shgm512" in rig.K and "sdm2048" in rig.K)
    PFS = (pf and os.getenv("MM_PFS", "0") == "1" and P in (256, 64)
           and f"k2s36h_{P}" in rig.K and "k2nz36" in rig.K)
    if PFG or PFM or PFT or PFU or PFD or PFR or PFS:'''
if "MM_PFD" not in src:
    assert OLD_F in src, "flag anchor not found"
    src = src.replace(OLD_F, NEW_F, 1)
    print("[wireE2] flags added")
else:
    print("[wireE2] flags present")

# ---- 3. _need ----
OLD_N = '''                ["gxu_gm"] if PFU else [])'''
NEW_N = '''                ["gxu_gm"] if PFU else []) + (
                ["gxd_gm"] if PFD else []) + (
                ["shgm512", "sdm2048"] if PFR else []) + (
                [f"k2s36h_{P}", "k2nz36"] if PFS else [])'''
if "gxd_gm\"] if PFD" not in src:
    assert OLD_N in src, "_need anchor not found"
    src = src.replace(OLD_N, NEW_N, 1)
    print("[wireE2] _need extended")
else:
    print("[wireE2] _need present")

# ---- 4a. the dn arm ----
OLD_D = '''                gdn = "gxm_dn6" if dnk == "gx8e256dn6" else (
                    "gxm_dnf" if "gxm_dnf" in rig.K else "gxm_dn")'''
NEW_D = '''                gdn = "gxm_dn6" if dnk == "gx8e256dn6" else (
                    "gxm_dnf" if "gxm_dnf" in rig.K else "gxm_dn")
                # SESSION E (MM_PFD): the IQ4_NL dn lane -> the gathered
                # mma (grid 4096 = 256 experts x 16 W-row-tiles). Tier-2
                # (F 3.8e-4 class); Q6_K layers keep gxm_dn6.
                if PFD and gdn == "gxm_dnf":
                    gdn = "gxd_gm"'''
if "gxd_gm\" if PFD" not in src and "MM_PFD\": the IQ4_NL" not in src:
    assert OLD_D in src, "dn arm anchor not found"
    src = src.replace(OLD_D, NEW_D, 1)
    print("[wireE2] dn arm wired")
else:
    print("[wireE2] dn arm present")

# the launch: gxd_gm takes (PTB, EOFFB, PLISTB, actb, iq4nl, partsb) grid 4096
OLD_DL = '''                dex = () if gdn == "gxm_dn6" else (rig.iq4nl,)
                seq.append((gdn, (rig.PTB_DN[L], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                  rig.PLISTB, B["actb"]) + dex + (B["partsb"],), 1024, ()))'''
NEW_DL = '''                if gdn == "gxd_gm":
                    seq.append((gdn, (rig.PTB_DN[L], rig.EOFFB, rig.PLISTB,
                                      B["actb"], rig.iq4nl, B["partsb"]), 4096, ()))
                else:
                    dex = () if gdn == "gxm_dn6" else (rig.iq4nl,)
                    seq.append((gdn, (rig.PTB_DN[L], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                      rig.PLISTB, B["actb"]) + dex + (B["partsb"],), 1024, ()))'''
if "gdn == \"gxd_gm\":" not in src:
    assert OLD_DL in src, "dn launch anchor not found"
    src = src.replace(OLD_DL, NEW_DL, 1)
    print("[wireE2] dn launch wired")
else:
    print("[wireE2] dn launch present")

# ---- 4b. the shared pair arm ----
OLD_S = '''                seq.append(("shgu32", (w["sg"], w["su"], B["hnb"], rig.ACTSHB), 512 // 8, (P,)))
                seq.append(("shdn32", (w["sd"], rig.ACTSHB, B["shb"]), 2048 // 8, (P,)))'''
NEW_S = '''                # SESSION E (MM_PFR): the shared-expert mma pair (isolated
                # 2.8x/3.3x, in-chunk -15.1ms; Tier-2 F class)
                if PFR:
                    seq.append(("shgm512", (w["sg"], w["su"], B["hnb"], rig.ACTSHB),
                                (P // 32) * (512 // 64), (P,)))
                    seq.append(("sdm2048", (w["sd"], rig.ACTSHB, B["shb"]),
                                (P // 32) * (2048 // 64), (P,)))
                else:
                    seq.append(("shgu32", (w["sg"], w["su"], B["hnb"], rig.ACTSHB), 512 // 8, (P,)))
                    seq.append(("shdn32", (w["sd"], rig.ACTSHB, B["shb"]), 2048 // 8, (P,)))'''
if "shgm512\", (w[\"sg\"]" not in src:
    assert OLD_S in src, "shared pair anchor not found"
    src = src.replace(OLD_S, NEW_S, 1)
    print("[wireE2] shared pair wired")
else:
    print("[wireE2] shared pair present")

# ---- 4c. the scan arm ----
OLD_K = '''            else:
                seq.append((tg, (w["cw"], B["qkvsb"], rig.CSV[gi], B["qkvsb"]), (32,), ()))
                seq.append((tk, (B["qkvsb"], B["abb"], w["al"], w["dt"], w["sn"], B["zb"], rig.SV[gi], B["gyb"]), (32,), ()))'''
if OLD_K not in src:
    OLD_K = '''                seq.append((tg, (w["cw"], B["qkvsb"], rig.CSV[gi], B["qkvsb"]), (32,), ()))
                seq.append((tk, (B["qkvsb"], B["abb"], w["al"], w["dt"], w["sn"], B["zb"], rig.SV[gi], B["gyb"]), (32,), ()))'''
NEW_K = '''                seq.append((tg, (w["cw"], B["qkvsb"], rig.CSV[gi], B["qkvsb"]), (32,), ()))
                # SESSION E (MM_PFS): the split-row scan (2 CTAs/head, the
                # norm apply split out; yss regroup only -- S bit-exact)
                if PFS and tk == "k2s36_256":
                    seq.append(("k2s36h_256", (B["qkvsb"], B["abb"], w["al"], w["dt"],
                                               rig.SV[gi], rig.YQB, rig.YSSB), 64, ()))
                    seq.append(("k2nz36", (rig.YQB, rig.YSSB, w["sn"], B["zb"],
                                           B["gyb"]), P * 32, (P,)))
                elif PFS and tk == "k2s36_64":
                    seq.append(("k2s36h_64", (B["qkvsb"], B["abb"], w["al"], w["dt"],
                                              rig.SV[gi], rig.YQB, rig.YSSB), 64, ()))
                    seq.append(("k2nz36", (rig.YQB, rig.YSSB, w["sn"], B["zb"],
                                           B["gyb"]), P * 32, (P,)))
                else:
                    seq.append((tk, (B["qkvsb"], B["abb"], w["al"], w["dt"], w["sn"], B["zb"], rig.SV[gi], B["gyb"]), (32,), ()))'''
if "k2s36h_256\", (B[\"qkvsb\"]" not in src:
    assert OLD_K in src, "scan anchor not found"
    src = src.replace(OLD_K, NEW_K, 1)
    print("[wireE2] scan wired")
else:
    print("[wireE2] scan present")

open(LIB, "w").write(src)

# ---- 5. svc_fp keys ----
ssrc = open(SVC).read()
OLD_SV = '''    "MM_PFU",'''
NEW_SV = '''    "MM_PFU",
    # MM SESSION E port 2: the gathered dn mma, the shared-expert mma pair,
    # the split-row scan (all Tier-2; pcache namespace flip on first boot)
    "MM_PFD", "MM_PFR", "MM_PFS",'''
if "MM_PFD" not in ssrc:
    assert OLD_SV in ssrc, "svc_fp anchor not found"
    ssrc = ssrc.replace(OLD_SV, NEW_SV, 1)
    open(SVC, "w").write(ssrc)
    print("[wireE2] svc_fp keys added")
else:
    print("[wireE2] svc_fp present")
print("[wireE2] done")
