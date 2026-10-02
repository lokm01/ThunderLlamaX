#!/usr/bin/env python3
"""MM SESSION D -- wire MM_PFK (the k=2048 trunk mma ports) into production:
  1. Rig7.__init__: load the pgmq8k2_r{8192,4096,512} cubins when present.
  2. build_seq7: the PFK arms replace the gvs32k2048 qkv/z (GDN) and
     q/k/v (ATTN) launches for PF graphs.
  3. svc_fp.py: MM_PFK joins the config-fingerprint env keys.
Idempotent (marker-scanned)."""
import sys

LIB = "~/tinygrad-metal/MM_P7_lib.py"
SVC = "~/tinygrad-metal/engine0/svc_fp.py"
src = open(LIB).read()

# ---- loader: extend the session-D block ----
OLD_LD = '''        D_LSZ = {"spkqw4_98304": (256, 1, 1)}'''
NEW_LD = '''        D_LSZ = {"spkqw4_98304": (256, 1, 1),
                 "pgmq8k2_r8192": (256, 1, 1), "pgmq8k2_r4096": (256, 1, 1),
                 "pgmq8k2_r512": (256, 1, 1)}'''
if "pgmq8k2_r8192" not in src:
    assert OLD_LD in src, "session-D loader anchor not found"
    src = src.replace(OLD_LD, NEW_LD, 1)
    # the loader loop reads MM_D_{sym}.cubin -- symbol names match files
    print("[wire2] loader extended")
else:
    print("[wire2] loader present")

# ---- GDN qkv/z arm ----
OLD_G = '''            if PFM:
                # G2-measured: qkv/z win (1.54x/1.36x); ab is 0.92x -> stock.
                seq.append(("gvs32k2048", (w["qkv"], B["hnb"], B["qkvb"]), 8192 // 8, (8192, P)))
                seq.append(("gvs32k2048", (w["z"], B["hnb"], B["zb"]), 4096 // 8, (4096, P)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))'''
NEW_G = '''            if PFM:
                # G2-measured: qkv/z win (1.54x/1.36x); ab is 0.92x -> stock.
                # SESSION D (port 2): the k=2048 mma M-GEMMs (POC 4.6x/5.2x
                # isolated, F ~2.8e-4) when MM_PFK=1 + cubins present.
                if PFK:
                    seq.append(("pgmq8k2_r8192", (w["qkv"], B["hnb"], B["qkvb"]),
                                (P // 32) * 128, (P,)))
                    seq.append(("pgmq8k2_r4096", (w["z"], B["hnb"], B["zb"]),
                                (P // 32) * 64, (P,)))
                else:
                    seq.append(("gvs32k2048", (w["qkv"], B["hnb"], B["qkvb"]), 8192 // 8, (8192, P)))
                    seq.append(("gvs32k2048", (w["z"], B["hnb"], B["zb"]), 4096 // 8, (4096, P)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))'''
if "pgmq8k2_r8192\", (w[\"qkv\"]" not in src:
    assert OLD_G in src, "GDN qkv/z anchor not found"
    src = src.replace(OLD_G, NEW_G, 1)
    print("[wire2] GDN qkv/z arm wired")
else:
    print("[wire2] GDN arm present")

# ---- ATTN q/k/v arm ----
OLD_A = '''            if PFM:
                seq.append(("gvs32k2048", (w["q"], B["hnb"], B["qgb"]), 8192 // 8, (8192, P)))
                seq.append(("gvs32k2048", (w["k"], B["hnb"], B["kqb"]), 512 // 8, (512, P)))
                seq.append(("gvs32k2048", (w["v"], B["hnb"], B["vqb"]), 512 // 8, (512, P)))'''
NEW_A = '''            if PFM:
                if PFK:
                    seq.append(("pgmq8k2_r8192", (w["q"], B["hnb"], B["qgb"]),
                                (P // 32) * 128, (P,)))
                    seq.append(("pgmq8k2_r512", (w["k"], B["hnb"], B["kqb"]),
                                (P // 32) * 8, (P,)))
                    seq.append(("pgmq8k2_r512", (w["v"], B["hnb"], B["vqb"]),
                                (P // 32) * 8, (P,)))
                else:
                    seq.append(("gvs32k2048", (w["q"], B["hnb"], B["qgb"]), 8192 // 8, (8192, P)))
                    seq.append(("gvs32k2048", (w["k"], B["hnb"], B["kqb"]), 512 // 8, (512, P)))
                    seq.append(("gvs32k2048", (w["v"], B["hnb"], B["vqb"]), 512 // 8, (512, P)))'''
if "pgmq8k2_r8192\", (w[\"q\"]" not in src:
    assert OLD_A in src, "ATTN q/k/v anchor not found"
    src = src.replace(OLD_A, NEW_A, 1)
    print("[wire2] ATTN q/k/v arm wired")
else:
    print("[wire2] ATTN arm present")

# ---- the PFK flag next to PFM ----
OLD_F = '''    PFT = pf and os.getenv("MM_PFT", "0") == "1" and P in (256, 64)'''
NEW_F = '''    PFT = pf and os.getenv("MM_PFT", "0") == "1" and P in (256, 64)
    # SESSION D (port 2): the k=2048 trunk mma ports (qkv/z/q/k/v)
    PFK = (pf and os.getenv("MM_PFK", "0") == "1" and P in (256, 64)
           and "pgmq8k2_r8192" in rig.K)'''
if "MM_PFK" not in src:
    assert OLD_F in src, "PFT flag anchor not found"
    src = src.replace(OLD_F, NEW_F, 1)
    print("[wire2] PFK flag added")
else:
    print("[wire2] PFK flag present")

open(LIB, "w").write(src)

svc = open(SVC).read()
if '"MM_PFK"' not in svc:
    svc = svc.replace('"MM_PFW",', '"MM_PFW", "MM_PFK",', 1)
    open(SVC, "w").write(svc)
    print("[wire2] svc_fp key added")
else:
    print("[wire2] svc_fp key present")
print("[wire2] done")
