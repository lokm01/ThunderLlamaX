#!/usr/bin/env python3
"""MM SESSION B wiring patch -- applies the production edits for MM_PFG /
MM_PFM / MM_PF64 (build_seq7 + Rig7 + test_moe36 + serve_moe + svc_fp + the
MoE env file). Exact-string replacements; FAILS LOUD if any anchor is stale.
Idempotent: skips an edit whose NEW text is already present."""
import os, sys

BASE = "~/tinygrad-metal"
EDITS = []

def edit(path, old, new, tag):
    EDITS.append((path, old, new, tag))

# ---------------- MM_P7_lib.py ----------------
P7 = BASE + "/MM_P7_lib.py"

edit(P7,
'''        }
        dev.synchronize()
        print("[rig7] cpu-mapped ctl + pf buffers + split scratch ready", flush=True)
''',
'''        }
        dev.synchronize()
        print("[rig7] cpu-mapped ctl + pf buffers + split scratch ready", flush=True)

        # ---- SESSION B (MoE prefill campaign): grouped-expert (MM_PFG) +
        # seat-loop trunk (MM_PFM) programs + their PF scratch. Cubins are
        # PREBUILT (engine0/mm/mm_build_b.zsh -- nvcc needs the colima
        # container); a gated build_seq7 without them fails LOUD.
        self.EOFFB = self.alloc(257 * 4)          # expert bin offsets [257] i32
        self.PLISTB = self.alloc(2048 * 2)        # sorted pair ids [2048] u16
        self.ITEMSB = self.alloc(4096 * 4)        # item descriptors (e|rs|m0) u32
        self.NITB = self.alloc(4)                 # item count u32
        self.ACTSHB = self.alloc(256 * 512 * 4)   # shared-expert act [seats][512]
        B_LSZ = {"mmsort8": (1024, 1, 1), "gxm_up": (512, 1, 1), "gxm_up4": (512, 1, 1),
                 "gxm_dn": (1024, 1, 1), "gxm_dn6": (1024, 1, 1),
                 "gvs32k2048": (256, 1, 1), "gvs32k4096r": (256, 1, 1), "gvsab": (128, 1, 1),
                 "shgu32": (256, 1, 1), "shdn32": (256, 1, 1),
                 "gconv36_64": (256, 1, 1), "k2s36_64": (256, 1, 1)}
        B_VALS = {"mmsort8": 1, "gvs32k2048": 2, "gvs32k4096r": 2,
                  "gvsab": 1, "shgu32": 1, "shdn32": 1}
        _nb = 0
        for _sym, _lsz in B_LSZ.items():
            _cb = (f"{BASE}/MM_P34_{_sym}.cubin" if _sym in ("gconv36_64", "k2s36_64")
                   else f"{BASE}/MM_B_{_sym}.cubin")
            if not os.path.exists(_cb):
                continue
            _lib = open(_cb, "rb").read()
            _sig = tuple(self.INT_SIG for _ in range(B_VALS.get(_sym, 0)))
            self.K[_sym] = NVProgram(dev, TinyELF(lib=_lib, name=_sym,
                                                  target=dev.renderer.target,
                                                  signature=_sig))
            L56.LSZ[_sym] = _lsz; _nb += 1
        self.LSZ7 = dict(L56.LSZ)
        dev.synchronize()
        print(f"[rig7] session-B: {_nb}/12 programs + grouped/seat-loop scratch", flush=True)
''', "rig7-buffers")

edit(P7,
'''    rig._P2D = P
    seq = [("embg248", (rig.EMB, idsb, B["hA"]), max(1, (P+31)//32), (P,))]
''',
'''    rig._P2D = P
    # SESSION B gates (kill-switch byte-identity: default OFF = the exact
    # stock seq). PFG: grouped experts + shared M-batch + the sort. PFM: the
    # seat-loop trunk GEMVs. PF graphs only (P in 256/64, %32==0 -- the
    # gvs/gvsab seat staging assumes seats % 32 == 0).
    PFG = pf and os.getenv("MM_PFG", "0") == "1" and P in (256, 64)
    PFM = pf and os.getenv("MM_PFM", "0") == "1" and P in (256, 64)
    if PFG or PFM:
        _need = ["mmsort8", "gxm_up", "gxm_up4", "gxm_dn", "gxm_dn6",
                 "gvs32k2048", "gvs32k4096r", "gvsab", "shgu32", "shdn32"] + (
                ["gconv36_64", "k2s36_64"] if tg == "gconv36_64" else [])
        _miss = [s for s in _need if s not in rig.K]
        if _miss:
            raise RuntimeError(f"MM_PFG/MM_PFM=1 but session-B programs missing: {_miss} "
                               "(run engine0/mm/mm_build_b.zsh; cubins load at Rig7 boot)")
    seq = [("embg248", (rig.EMB, idsb, B["hA"]), max(1, (P+31)//32), (P,))]
''', "seq7-flags")

edit(P7,
'''        if L in GDN_LAYERS:
            gi = GDN_LAYERS.index(L)
            seq.append(("gv8k2048p", (w["qkv"], B["hnb"], B["qkvb"]), (256,), (8192,)))
            seq.append(("gv8k2048p", (w["z"], B["hnb"], B["zb"]), (128,), (4096,)))
            seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))
''',
'''        if L in GDN_LAYERS:
            gi = GDN_LAYERS.index(L)
            if PFM:
                seq.append(("gvs32k2048", (w["qkv"], B["hnb"], B["qkvb"]), 8192 // 8, (8192, P)))
                seq.append(("gvs32k2048", (w["z"], B["hnb"], B["zb"]), 4096 // 8, (4096, P)))
                seq.append(("gvsab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (16, P // 32), (P,)))
            else:
                seq.append(("gv8k2048p", (w["qkv"], B["hnb"], B["qkvb"]), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["z"], B["hnb"], B["zb"]), (128,), (4096,)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))
''', "seq7-gdn-trunk")

edit(P7,
'''            ptbl = rig.SPTB[ai]
            seq.append(("gv8k2048p", (w["q"], B["hnb"], B["qgb"]), (256,), (8192,)))
            seq.append(("gv8k2048p", (w["k"], B["hnb"], B["kqb"]), (16,), (512,)))
            seq.append(("gv8k2048p", (w["v"], B["hnb"], B["vqb"]), (16,), (512,)))
''',
'''            ptbl = rig.SPTB[ai]
            if PFM:
                seq.append(("gvs32k2048", (w["q"], B["hnb"], B["qgb"]), 8192 // 8, (8192, P)))
                seq.append(("gvs32k2048", (w["k"], B["hnb"], B["kqb"]), 512 // 8, (512, P)))
                seq.append(("gvs32k2048", (w["v"], B["hnb"], B["vqb"]), 512 // 8, (512, P)))
            else:
                seq.append(("gv8k2048p", (w["q"], B["hnb"], B["qgb"]), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["k"], B["hnb"], B["kqb"]), (16,), (512,)))
                seq.append(("gv8k2048p", (w["v"], B["hnb"], B["vqb"]), (16,), (512,)))
''', "seq7-attn-trunk")

edit(P7,
'''            seq.append(("gv8k4096r", (w["out"], B["gyb"], hin, hmid), (64,), (2048,)))
''',
'''            if PFM:
                seq.append(("gvs32k4096r", (w["out"], B["gyb"], hin, hmid), 2048 // 8, (2048, P)))
            else:
                seq.append(("gv8k4096r", (w["out"], B["gyb"], hin, hmid), (64,), (2048,)))
''', "seq7-out")

edit(P7,
'''            seq.append(("gv8k4096r", (w["o"], B["ayb"], hin, hmid), (64,), (2048,)))
''',
'''            if PFM:
                seq.append(("gvs32k4096r", (w["o"], B["ayb"], hin, hmid), 2048 // 8, (2048, P)))
            else:
                seq.append(("gv8k4096r", (w["o"], B["ayb"], hin, hmid), (64,), (2048,)))
''', "seq7-o")

edit(P7,
'''        else:
            seq.append(("rt8e256", (w["rt"], w["wsh"], B["hnb"], B["eidsb"], B["gatesb"], B["sgb"]), (P,), ()))
            seq.append(("shexp8", (w["sg"], w["su"], w["sd"], B["hnb"], B["shb"]), (P,), ()))
            upbufs = (rig.PTB_UP[L], B["eidsb"], B["hnb"], rig.iq4nl, B["actb"]) if upk == "gx8e256up4" else (rig.PTB_UP[L], B["eidsb"], B["hnb"], rig.gridf, B["actb"])
            seq.append((upk, upbufs, (P*8,), ()))
            dnbufs = (rig.PTB_DN[L], B["eidsb"], B["actb"], B["partsb"]) if dnk == "gx8e256dn6" else (rig.PTB_DN[L], B["eidsb"], B["actb"], rig.iq4nl, B["partsb"])
            seq.append((dnk, dnbufs, (P*8,), ()))
''',
'''        else:
            if PFG:
                # GOLD ROUTER UNTOUCHED (rt8e256 keeps the bit-exact top-8
                # contract); mmsort8 builds eoff/plist/items; the grouped
                # gxm_* walk expert bins (slab staged once per
                # row-block-chunk instead of once per pair); the shared
                # expert becomes the shgu32+shdn32 M-batched pair. The
                # cmbz2048 combine is UNTOUCHED (plist scatter writes the
                # same [pair] slots the pair-walk wrote).
                seq.append(("rt8e256", (w["rt"], w["wsh"], B["hnb"], B["eidsb"], B["gatesb"], B["sgb"]), (P,), ()))
                seq.append(("mmsort8", (B["eidsb"], rig.EOFFB, rig.PLISTB, rig.ITEMSB, rig.NITB), 1, (P * 8,)))
                seq.append(("shgu32", (w["sg"], w["su"], B["hnb"], rig.ACTSHB), 512 // 8, (P,)))
                seq.append(("shdn32", (w["sd"], rig.ACTSHB, B["shb"]), 2048 // 8, (P,)))
                gup = "gxm_up4" if upk == "gx8e256up4" else "gxm_up"
                gex = (rig.iq4nl,) if gup == "gxm_up4" else (rig.gridf,)
                seq.append((gup, (rig.PTB_UP[L], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                  rig.PLISTB, B["hnb"]) + gex + (B["actb"],), 1024, ()))
                gdn = "gxm_dn6" if dnk == "gx8e256dn6" else "gxm_dn"
                dex = () if gdn == "gxm_dn6" else (rig.iq4nl,)
                seq.append((gdn, (rig.PTB_DN[L], rig.ITEMSB, rig.NITB, rig.EOFFB,
                                  rig.PLISTB, B["actb"]) + dex + (B["partsb"],), 1024, ()))
            else:
                seq.append(("rt8e256", (w["rt"], w["wsh"], B["hnb"], B["eidsb"], B["gatesb"], B["sgb"]), (P,), ()))
                seq.append(("shexp8", (w["sg"], w["su"], w["sd"], B["hnb"], B["shb"]), (P,), ()))
                upbufs = (rig.PTB_UP[L], B["eidsb"], B["hnb"], rig.iq4nl, B["actb"]) if upk == "gx8e256up4" else (rig.PTB_UP[L], B["eidsb"], B["hnb"], rig.gridf, B["actb"])
                seq.append((upk, upbufs, (P*8,), ()))
                dnbufs = (rig.PTB_DN[L], B["eidsb"], B["actb"], B["partsb"]) if dnk == "gx8e256dn6" else (rig.PTB_DN[L], B["eidsb"], B["actb"], rig.iq4nl, B["partsb"])
                seq.append((dnk, dnbufs, (P*8,), ()))
''', "seq7-moe")

# ---------------- test_moe36.py ----------------
TM = BASE + "/engine0/test_moe36.py"

edit(TM,
'''        seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False,
                           spk=spk, S=PF_S, pf=True)
        self.gr_pf = mkgraph(rig, seqpf, "sv_pf")
        self.dev = rig.dev
''',
'''        seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False,
                           spk=spk, S=PF_S, pf=True)
        self.gr_pf = mkgraph(rig, seqpf, "sv_pf")
        # ---- SESSION B (L5): the PF-64 tail graph -- the len%256 tail at
        # chunk rate instead of ~52ms/token T1 cycles (the GSM8K-class TTFT
        # is ~70% tail). MM_PF64 kill-switch; needs the _64 scan cubins.
        self.pf64_on = os.getenv("MM_PF64", "0") == "1" and "gconv36_64" in rig.K
        self.gr_pf64 = None
        if self.pf64_on:
            seqpf64 = build_seq7(rig, 64, "gconv36_64", "k2s36_64", with_head=False,
                                 spk=spk, S=PF_S, pf=True)
            self.gr_pf64 = mkgraph(rig, seqpf64, "sv_pf64")
            print("[moe36] PF-64 tail graph built (len%256 tail at chunk rate)", flush=True)
        self.dev = rig.dev
''', "tm-pf64")

edit(TM,
'''    def eager_head_cur(self, seat=None, pf=False):
''',
'''    def pf64_feed_chunk(self, chunk, pos0):
        """One 64-seat PF tail chunk (same cpu-fold control; seats 0..63 of
        the 256-seat PFB buffers). Zeros beyond seat 63 are never read."""
        assert len(chunk) == 64, f"pf64 chunk must be 64 (got {len(chunk)})"
        a = self.np.zeros(256, dtype=self.np.int32)
        a[:64] = self.np.asarray(chunk, dtype=self.np.int32)
        self.rig.pf_ids_view[:] = memoryview(self.np.ascontiguousarray(a).data)
        self.rig.pos_view[0] = int(pos0)
        self.gr_pf64.step()

    def eager_head_cur(self, seat=None, pf=False):
''', "tm-pf64-feed")

edit(TM,
'''        runners = [self.gr1, self.gr2, self.gr8, self.gr_pf]
''',
'''        runners = [self.gr1, self.gr2, self.gr8, self.gr_pf]
        if getattr(self, "pf64_on", False) and self.gr_pf64 is not None:
            runners.append(self.gr_pf64)
''', "tm-fence")

# ---------------- serve_moe.py ----------------
SM = BASE + "/engine0/serve_moe.py"

edit(SM,
'''      _pc_maybe_ingest(pos0 + i, full_toks)
    pf_ms = (time.perf_counter() - t0) * 1e3
    for q in range(i, n):
''',
'''      _pc_maybe_ingest(pos0 + i, full_toks)
      last_seat = PF_CHUNK - 1
    # SESSION B (L5): the 1..255 tail runs 64-seat PF chunks at chunk rate
    # (bit-exact vs the per-token T1 tail by the chunk-256 construction)
    # before the final <64 per-token tail.
    if getattr(eng, "pf64_on", False) and getattr(eng, "gr_pf64", None) is not None:
      while n - i >= 64:
        cancel_checkpoint("prefill_pf64_chunk")
        eng.pf64_feed_chunk([int(x) for x in delta[i:i + 64]], pos0 + i)
        i += 64; last_seat = 63
        _beat()
        if prog is not None:
          prog(i, n, "prefill_pf")
        if (pos0 + i) % PF_CHUNK == 0:
          _pc_maybe_ingest(pos0 + i, full_toks)
    pf_ms = (time.perf_counter() - t0) * 1e3
    for q in range(i, n):
''', "sm-pf64-tail")

edit(SM,
'''    if n > 0:
      # P10: where the LAST fed token's hidden lives (the bit-exact
      # chunk-256 class: PF seat 255 == the T1 hidden by construction).
      feed_hidden[0] = ("hA", 0) if i < n else ("pf", PF_CHUNK - 1)
    return chunks, pf_ms
''',
'''    if n > 0:
      # P10: where the LAST fed token's hidden lives (the bit-exact
      # chunk-256 class: PF seat 255 == the T1 hidden by construction).
      # SESSION B: the final chunk may be a 64 (seat 63) -- last_seat tracks
      # the final chunk's last seat; the T1 tail (i < n) anchors at hA[0].
      feed_hidden[0] = ("hA", 0) if i < n else ("pf", last_seat)
    return chunks, pf_ms
''', "sm-last-seat")

edit(SM,
'''    n = len(delta); i = 0; chunks = 0
    t0 = time.perf_counter()
''',
'''    n = len(delta); i = 0; chunks = 0
    last_seat = PF_CHUNK - 1
    t0 = time.perf_counter()
''', "sm-seat-init")

# ---------------- svc_fp.py ----------------
SF = BASE + "/engine0/svc_fp.py"
edit(SF,
'''    "TLX_EAGLE_K",
)
''',
'''    "TLX_EAGLE_K",
    # MM SESSION B (MoE prefill): the grouped-expert / seat-loop / PF64-tail
    # graph knobs -- they change the PF graph KERNEL SET (outputs bit-exact
    # by the G2/F1b gates, but pcache nodes must never cross kernel-set
    # boundaries). Cache-invalidation event: ONE cold MoE pcache rebuild on
    # the first boot after this ships (slogged).
    "MM_PFG", "MM_PFM", "MM_PF64",
)
''', "svcfp-keys")

def main():
    done = failed = skipped = 0
    for path, old, new, tag in EDITS:
        src = open(path).read()
        if new in src:
            print(f"[wire] {tag}: already applied -- skip")
            skipped += 1
            continue
        if old not in src:
            print(f"[wire] {tag}: ANCHOR NOT FOUND in {path}")
            failed += 1
            continue
        open(path, "w").write(src.replace(old, new, 1))
        print(f"[wire] {tag}: applied")
        done += 1
    print(f"[wire] done={done} skipped={skipped} failed={failed}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
