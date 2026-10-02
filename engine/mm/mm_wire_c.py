#!/usr/bin/env python3
"""MM SESSION C wiring patch -- MM_PFT (the out/o k=4096 mma M-GEMM) + the
gxm_dnf act-fold. Exact-string replacements; FAILS LOUD on stale anchors.
Idempotent."""
import os, sys

BASE = "~/tinygrad-metal"
EDITS = []


def edit(path, old, new, tag):
    EDITS.append((path, old, new, tag))


# ---------------- MM_P7_lib.py ----------------
P7 = BASE + "/MM_P7_lib.py"

edit(P7,
'''        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-B: {_nb}/12 programs + grouped/seat-loop scratch", flush=True)
''',
'''        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-B: {_nb}/12 programs + grouped/seat-loop scratch", flush=True)

        # ---- SESSION C (MoE prefill residual attack): the L1b mma M-GEMM
        # (pgmq8m32 -- POC 7.4-8.2x, Tier-2 numerics F~1.5e-4, gate MM_PFT)
        # + the routed-dn act fold (gxm_dnf -- BIT-EXACT 1.32-1.36x, rides
        # MM_PFG; rollback = remove the cubin). Cubins PREBUILT
        # (engine0/mm/mm_build_c.zsh -- nvcc needs the colima container).
        C_LSZ = {"pgmq8m32": (256, 1, 1), "gxm_dnf": (256, 1, 1)}
        C_VALS = {"pgmq8m32": 1}
        _nc = 0
        for _sym, _lsz in C_LSZ.items():
            _cb = f"{BASE}/MM_C_{_sym}.cubin"
            if not os.path.exists(_cb):
                continue
            _lib = open(_cb, "rb").read()
            _sig = tuple(self.INT_SIG for _ in range(C_VALS.get(_sym, 0)))
            self.K[_sym] = NVProgram(dev, TinyELF(lib=_lib, name=_sym,
                                                  target=dev.renderer.target,
                                                  signature=_sig))
            L56.LSZ[_sym] = _lsz; _nc += 1
        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-C: {_nc}/2 programs (mma M-GEMM + dn fold)", flush=True)
''', "rig7-c")

edit(P7,
'''    PFG = pf and os.getenv("MM_PFG", "0") == "1" and P in (256, 64)
    PFM = pf and os.getenv("MM_PFM", "0") == "1" and P in (256, 64)
    if PFG or PFM:
        _need = ["mmsort8", "gxm_up", "gxm_up4", "gxm_dn", "gxm_dn6",
                 "gvs32k2048", "shgu32", "shdn32"] + (
                ["gconv36_64", "k2s36_64"] if tg == "gconv36_64" else [])
''',
'''    PFG = pf and os.getenv("MM_PFG", "0") == "1" and P in (256, 64)
    PFM = pf and os.getenv("MM_PFM", "0") == "1" and P in (256, 64)
    PFT = pf and os.getenv("MM_PFT", "0") == "1" and P in (256, 64)
    if PFG or PFM or PFT:
        _need = ["mmsort8", "gxm_up", "gxm_up4", "gxm_dn", "gxm_dn6",
                 "gvs32k2048", "shgu32", "shdn32"] + (
                ["gconv36_64", "k2s36_64"] if tg == "gconv36_64" else []) + (
                ["pgmq8m32"] if PFT else [])
''', "seq7-pft-flag")

edit(P7,
'''            # out/o stay STOCK: the k=4096 pair already streams at ~750
            # GB/s in isolation (G2: the seat-loop port is 0.96x).
            seq.append(("gv8k4096r", (w["out"], B["gyb"], hin, hmid), (64,), (2048,)))
''',
'''            # SESSION C (L1b): the out k=4096 pair -> the pgmq8m32 mma
            # M-GEMM when MM_PFT=1 (POC 7.4-8.2x isolated; the stock runs
            # ~212 GB/s IN-GRAPH = the latency floor the tile amortization
            # beats). Tier-2 numerics: F ~1.5e-4 vs the stock (fp16
            # operands + mma order) -- the F-bank re-baseline gates the ship.
            if PFT:
                seq.append(("pgmq8m32", (w["out"], B["gyb"], hin, hmid), (P // 32) * 32, (P,)))
            else:
                seq.append(("gv8k4096r", (w["out"], B["gyb"], hin, hmid), (64,), (2048,)))
''', "seq7-out-mma")

edit(P7,
'''            seq.append(("gv8k4096r", (w["o"], B["ayb"], hin, hmid), (64,), (2048,)))
''',
'''            if PFT:
                seq.append(("pgmq8m32", (w["o"], B["ayb"], hin, hmid), (P // 32) * 32, (P,)))
            else:
                seq.append(("gv8k4096r", (w["o"], B["ayb"], hin, hmid), (64,), (2048,)))
''', "seq7-o-mma")

edit(P7,
'''                gdn = "gxm_dn6" if dnk == "gx8e256dn6" else "gxm_dn"
''',
'''                # SESSION C: the dn act-restaging fold (BIT-EXACT, 1.32-1.36x
                # POC) preferred when its cubin is loaded; no env key (same
                # outputs bit-for-bit -> pcache nodes stay valid).
                gdn = "gxm_dn6" if dnk == "gx8e256dn6" else (
                    "gxm_dnf" if "gxm_dnf" in rig.K else "gxm_dn")
''', "seq7-dnf")

# ---------------- svc_fp.py ----------------
SF = BASE + "/engine0/svc_fp.py"
edit(SF,
'''    "MM_PFG", "MM_PFM", "MM_PF64",
)
''',
'''    "MM_PFG", "MM_PFM", "MM_PF64",
    # MM SESSION C: the out/o mma M-GEMM (NUMERICS MOVE -- F~1.5e-4 bank,
    # the dense-M32 precedent class). config_fp key -> pcache namespace
    # flip: ONE cold MoE pcache rebuild on first boot (slogged).
    "MM_PFT",
)
''', "svcfp-pft")


def main():
    done = failed = skipped = 0
    for path, old, new, tag in EDITS:
        src = open(path).read()
        if new in src:
            print(f"[wireC] {tag}: already applied -- skip")
            skipped += 1
            continue
        if old not in src:
            print(f"[wireC] {tag}: ANCHOR NOT FOUND in {path}")
            failed += 1
            continue
        open(path, "w").write(src.replace(old, new, 1))
        print(f"[wireC] {tag}: applied")
        done += 1
    print(f"[wireC] done={done} skipped={skipped} failed={failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
