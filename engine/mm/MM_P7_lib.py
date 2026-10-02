#!/usr/bin/env python3
"""MM P7 harness library — the 140-cross (host fold) + split-KV + prefill.

PART 1 (the host fold): every _copyin/_copyout on this stack does a FULL
dev.synchronize() (hcq.py _copyin/_copyout) — the P6 spec cycle paid ~6 of
them (~14ms). The fold:
  - idsb / POSB / AMDB / EB allocated cpu_access=True -> host writes/reads
    via cpu_view() DIRECTLY (the cmdq hw_page pattern; zero GPU commands).
  - acc36 computes the greedy accept m ON DEVICE from AMDB + the drafts
    (= idsb[1..K]) -> mb (selc36 slot) + the EMIT BLOCK eb[2] = (m, amds[m]).
  - selc36 chained as the probe graph TAIL -> the IN-GRAPH COMMIT.
  Per-cycle GPU interactions: ONE graph submit + timeline wait.

PART 2 (split-KV): spkq256s (warp-per-position, online max-rescale, zero
syncthreads in the hot loop; grid (16*T, S)) + spkc256 (combine + gate).
The numpy mirror spkq_h_split_ref models the exact kernel order (lane j-asc
partials, 5-xor tree, per-(split,warp) online rescale, i-asc combine).

PART 3 (prefill): the chunk-256 graph = the SAME per-seat kernels at P=256
(gconv36_256/k2s36_256 T-chains + P-flat everything else) -> chunked feed
is BIT-EXACT vs the per-token T1 feed by construction (the gate proves it).
"""
import os, sys, time
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
import MM_P56_lib as L56
from MM_P56_lib import KAPool, MG, GraphRunner
from MM_P34_ports import (_f, EPS, F32, SCA, rmszc_ref, rope_apply, _tree_warp8,
                          GDN_LAYERS, ATTN_LAYERS)

# ---- THE COHERENCE PATCH: pass BufferSpec.uncached through to the RM alloc.
# cpu_access alone maps VRAM with GPU_CACHEABLE_YES -> host writes race GPU
# reads and GPU writes race host reads (the H0a/H0b stale class: 88/120
# acc36 trials, spec probes on stale ids). uncached allocates LOCATION_PCI
# sysmem + GPU_CACHEABLE_NO (the PMA / copy-staging proven-coherent class).
# The stock NVAllocator._alloc DROPS the flag -> monkeypatch.
from tinygrad.runtime.ops_nv import NVAllocator as _NVA
def _nv_alloc_uncached(self, size, options):
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access,
                                host=options.host, uncached=options.uncached)
_NVA._alloc = _nv_alloc_uncached

# ---------------- P7 cubin specs (nvcc via the colima container) ----------------
NVCC_ENV = L56.NVCC_ENV
P7_SPECS = [
    # (stem, symbol, variants, extra -D flags, 1024-thread?)
    ("MM_P7_acc36",     "acc36",     [None], []),
    ("MM_P7_spkc256",   "spkc256",   [None], []),
    ("MM_P7_spkq256s",  "spkq256s",  ["4096", "16384", "65536", "98304"], ["-DCTXS={V}"]),
    ("MM_P34_gconv36",  "gconv36",   ["256"], []),
    ("MM_P34_k2s36",    "k2s36",     ["256"], []),
    ("MM_P34_spka256",  "spka256m",  ["98304"], ["-DCTXS={V}", "-DRENAMED"]),
    ("MM_P6_h6kam",     "h6kam",     ["1"], ["-DPP={V}"]),
]

def build_p7_cubins(only=None):
    import subprocess
    built = {}
    for stem, sym, variants, extra in P7_SPECS:
        for v in (variants if variants else [None]):
            name = sym if v is None else f"{sym}_{v}"
            if only and name not in only: continue
            cb = f"{BASE}/{stem}_{v}.cubin" if v else f"{BASE}/{stem}.cubin"
            if not os.path.exists(cb):
                cmd = f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={cb} {BASE}/{stem}.cu"
                if v:
                    cmd += " " + " ".join(x.replace("{V}", v) for x in extra)
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=NVCC_ENV)
                if r.returncode:
                    print(f"[build FAIL {name}]\n" + r.stderr[-3000:]); sys.exit(1)
                print(f"[build] {name}", flush=True)
            built[name] = cb
    return built

def audit_cubin(path, name):
    import subprocess
    r = subprocess.run(f"cuobjdump -res-usage {path}", shell=True, capture_output=True, text=True, env=NVCC_ENV)
    for l in r.stdout.splitlines():
        lu = l.upper()
        if "SPILL" in lu and "STORES" not in lu:
            print(f"[audit {name}] {l.strip()}", flush=True)

LSZ_P7 = {"acc36": (256,1,1), "spkc256": (256,1,1),
          "spkq256s_4096": (256,1,1), "spkq256s_16384": (256,1,1),
          "spkq256s_65536": (256,1,1), "spkq256s_98304": (256,1,1),
          "gconv36_256": (256,1,1), "k2s36_256": (256,1,1),
          "spka256m_98304": (256,1,1), "h6kam_1": (1024,1,1)}
SCALAR_P7 = {"acc36", "spkc256", "spkq256s_4096", "spkq256s_16384",
             "spkq256s_65536", "spkq256s_98304"}
# MG._build resolves local sizes via the MM_P56_lib.LSZ table -> extend it.
L56.LSZ.update(LSZ_P7)

# ============================================================================
# THE RIG (P7): cpu-mapped control + prefill buffers + split scratch
# ============================================================================
class Rig7(L56.Rig):
    def __init__(self, ctx_alloc=98304, load_p6=True):
        L56.CTX_ALLOC = ctx_alloc          # the KV/rope sizing (P7: 96k default)
        super().__init__(load_p6=load_p6)
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_nv import NVProgram
        dev = self.dev
        built = build_p7_cubins()
        for name, cb in built.items():
            lib = open(cb, "rb").read()
            sig = (self.INT_SIG,) if (name in SCALAR_P7 or name in L56.SCALAR_TAIL) else tuple()
            self.K[name] = NVProgram(dev, TinyELF(lib=lib, name=name, target=dev.renderer.target, signature=sig))
            if name.startswith(("acc36", "spkq256s", "spkc256", "k2s36_256", "gconv36_256")):
                audit_cubin(cb, name)
        dev.synchronize()
        print(f"[rig7] +{len(built)} P7 programs", flush=True)
        self.LSZ7 = dict(L56.LSZ); self.LSZ7.update(LSZ_P7)

        # ---- cpu-mapped control/emit (THE HOST FOLD) ----
        # uncached=True: sysmem-backed, GPU-uncached -> bidirectionally coherent
        BS = self.BufferSpec
        def cpu_map(nbytes):
            b = dev.allocator.alloc(nbytes, BS(cpu_access=True, nolru=True, uncached=True))
            self.keep.append(b); return b
        self.idsb = cpu_map(9*4)      # decode ctl ids [cur + drafts]
        self.ids_view = self.idsb.cpu_view().view(size=36, fmt="i")
        self.pf_idsb = cpu_map(256*4)
        self.pf_ids_view = self.pf_idsb.cpu_view().view(size=1024, fmt="i")
        self.POSB = cpu_map(4)
        self.pos_view = self.POSB.cpu_view().view(size=4, fmt="i")
        self.AMDB = cpu_map(9*4)      # amred36 writes; host + acc36 read
        self.am_view = self.AMDB.cpu_view().view(size=36, fmt="i")
        self.EB = cpu_map(8)          # acc36 emit block (m, amds[m])
        self.eb_view = self.EB.cpu_view().view(size=8, fmt="i")
        self.eb_view[0] = -1; self.eb_view[1] = -1
        # THE SPTB TABLES BAKE POSB.va_addr -- REBUILD them against the
        # cpu-mapped POSB (the super() tables point at the old device POSB).
        for ai in range(10):
            Lv = ATTN_LAYERS[ai]
            self.SPTB[ai] = self.up(np.array([self.W[Lv]["kw"].va_addr, self.W[Lv]["qw"].va_addr,
                                self.COSB.va_addr, self.SINB.va_addr,
                                self.KVQ[ai].va_addr, self.KVS[ai].va_addr,
                                self.VVQ[ai].va_addr, self.VVS[ai].va_addr,
                                self.POSB.va_addr], dtype=np.uint64))
            self.SPTB[(ai, "m")] = self.up(np.array([self.W[Lv]["kw"].va_addr, self.W[Lv]["qw"].va_addr,
                                self.COSB.va_addr, self.SINB.va_addr,
                                self.KVQ[ai].va_addr, self.KVS[ai].va_addr,
                                self.VVQ[ai].va_addr, self.VVS[ai].va_addr,
                                self.POSB.va_addr, self.SPSCR.va_addr], dtype=np.uint64))
        # split scratch: decode (P<=9, S<=32) + prefill (P=256, S=8)
        self.SCR_DEC = self.alloc(16*9*32*8*258*4)
        self.SCR_PF  = self.alloc(16*256*8*8*258*4)
        # prefill tensor buffers (P=256 mirrors; idsb stays the pf ctl above)
        PF = 256
        def pf(nbytes): b = self.alloc(nbytes); return b
        self.PFB = {
            "hA": pf(PF*2048*4), "hB": pf(PF*2048*4), "hnb": pf(PF*2048*4),
            "qkvb": pf(PF*8192*4), "zb": pf(PF*4096*4), "abb": pf(PF*64*4),
            "qkvsb": pf(PF*8192*4), "gyb": pf(PF*4096*4),
            "qgb": pf(PF*8192*4), "kqb": pf(PF*512*4), "vqb": pf(PF*512*4),
            "ayb": pf(PF*4096*4), "eidsb": pf(PF*8*2), "gatesb": pf(PF*8*4),
            "sgb": pf(PF*4), "actb": pf(PF*8*512*4), "partsb": pf(PF*8*2048*2),
            "shb": pf(PF*2048*4), "normhb": pf(PF*2048*4),
        }
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
        B_LSZ = {"mmsort8": (1024, 1, 1), "gxm_up": (256, 1, 1), "gxm_up4": (256, 1, 1),
                 "gxm_dn": (256, 1, 1), "gxm_dn6": (256, 1, 1),
                 "gvs32k2048": (256, 1, 1), "gvs32k4096r": (256, 1, 1), "gvsab": (128, 1, 1),
                 "shgu32": (256, 1, 1), "shdn32": (256, 1, 1),
                 "gconv36_64": (256, 1, 1), "k2s36_64": (256, 1, 1)}
        B_VALS = {"mmsort8": 1, "gvs32k2048": 2, "gvs32k4096r": 2,
                  "gvsab": 1, "shgu32": 1, "shdn32": 1}
        B_FILE = {"gvs32k2048": "MM_B_gvs32", "gvs32k4096r": "MM_B_gvs32r"}
        _nb = 0
        for _sym, _lsz in B_LSZ.items():
            _stem = B_FILE.get(_sym)
            if _stem is None:
                _stem = ("MM_P34_" if _sym in ("gconv36_64", "k2s36_64") else "MM_B_") + _sym
            _cb = f"{BASE}/{_stem}.cubin"
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

    # ---------- feed helpers (zero-GPU-command) ----------
    def feed(self, tid, pos):
        self.ids_view[0] = int(tid)
        self.pos_view[0] = int(pos)
    def feed9(self, ids, pos):
        v = self.ids_view
        for i in range(9): v[i] = int(ids[i])
        self.pos_view[0] = int(pos)

# ============================================================================
# build_seq7 — the extended seq builder (split-KV + prefill P + in-graph tail)
# ============================================================================
def build_seq7(rig, P, tg, tk, with_head=True, head_mode="full", spk=None,
               slots=False, S=8, pf=False, tail_acc_K=0):
    """spk: None -> spkq256 (CTXS=1024 serial); 'm<N>' -> spkq256m_N serial;
    's<N>' -> spkq256s_N + spkc256 SPLIT (S runtime splits, scratch SCR_DEC/PF).
    tail_acc_K: >0 -> append acc36 + selc36 (the folded commit; K = drafts).
    pf: use the P=256 tensor buffers + pf_idsb."""
    K = rig.K
    idsb = rig.pf_idsb if pf else rig.idsb
    B = rig.PFB if pf else {n: getattr(rig, n) for n in
        ["hA","hB","hnb","qkvb","zb","abb","qkvsb","gyb","qgb","kqb","vqb",
         "ayb","eidsb","gatesb","sgb","actb","partsb","shb","normhb"]}
    rig._P2D = P
    # SESSION B gates (kill-switch byte-identity: default OFF = the exact
    # stock seq). PFG: grouped experts + shared M-batch + the sort. PFM: the
    # seat-loop trunk GEMVs. PF graphs only (P in 256/64, %32==0 -- the
    # gvs/gvsab seat staging assumes seats % 32 == 0).
    PFG = pf and os.getenv("MM_PFG", "0") == "1" and P in (256, 64)
    PFM = pf and os.getenv("MM_PFM", "0") == "1" and P in (256, 64)
    PFT = pf and os.getenv("MM_PFT", "0") == "1" and P in (256, 64)
    if PFG or PFM or PFT:
        _need = ["mmsort8", "gxm_up", "gxm_up4", "gxm_dn", "gxm_dn6",
                 "gvs32k2048", "shgu32", "shdn32"] + (
                ["gconv36_64", "k2s36_64"] if tg == "gconv36_64" else []) + (
                ["pgmq8m32"] if PFT else [])
        _miss = [s for s in _need if s not in rig.K]
        if _miss:
            raise RuntimeError(f"MM_PFG/MM_PFM=1 but session-B programs missing: {_miss} "
                               "(run engine0/mm/mm_build_b.zsh; cubins load at Rig7 boot)")
    seq = [("embg248", (rig.EMB, idsb, B["hA"]), max(1, (P+31)//32), (P,))]
    for L in range(40):
        w = rig.W[L]
        hin = B["hA"]; hmid = B["hB"]
        seq.append(("rmsz2048g", (hin, w["an"], B["hnb"]), (P,), ()))
        if L in GDN_LAYERS:
            gi = GDN_LAYERS.index(L)
            if PFM:
                # G2-measured: qkv/z win (1.54x/1.36x); ab is 0.92x -> stock.
                seq.append(("gvs32k2048", (w["qkv"], B["hnb"], B["qkvb"]), 8192 // 8, (8192, P)))
                seq.append(("gvs32k2048", (w["z"], B["hnb"], B["zb"]), 4096 // 8, (4096, P)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))
            else:
                seq.append(("gv8k2048p", (w["qkv"], B["hnb"], B["qkvb"]), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["z"], B["hnb"], B["zb"]), (128,), (4096,)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], B["hnb"], B["abb"]), (P,), ()))
            if slots:
                seq.append((tg, (w["cw"], B["qkvb"], rig.CSV[gi], B["qkvsb"], rig.CSLOTV[gi]), (32,), ()))
                seq.append((tk, (B["qkvsb"], B["abb"], w["adt"], w["sn"], B["zb"], rig.SV[gi], B["gyb"], rig.SLOTV[gi]), (32,), ()))
            else:
                seq.append((tg, (w["cw"], B["qkvb"], rig.CSV[gi], B["qkvsb"]), (32,), ()))
                seq.append((tk, (B["qkvsb"], B["abb"], w["al"], w["dt"], w["sn"], B["zb"], rig.SV[gi], B["gyb"]), (32,), ()))
            # SESSION C (L1b): the out k=4096 pair -> the pgmq8m32 mma
            # M-GEMM when MM_PFT=1 (POC 7.4-8.2x isolated; the stock runs
            # ~212 GB/s IN-GRAPH = the latency floor the tile amortization
            # beats). Tier-2 numerics: F ~1.5e-4 vs the stock (fp16
            # operands + mma order) -- the F-bank re-baseline gates the ship.
            if PFT:
                seq.append(("pgmq8m32", (w["out"], B["gyb"], hin, hmid), (P // 32) * 32, (P,)))
            else:
                seq.append(("gv8k4096r", (w["out"], B["gyb"], hin, hmid), (64,), (2048,)))
        else:
            ai = ATTN_LAYERS.index(L)
            ptbl = rig.SPTB[ai]
            if PFM:
                seq.append(("gvs32k2048", (w["q"], B["hnb"], B["qgb"]), 8192 // 8, (8192, P)))
                seq.append(("gvs32k2048", (w["k"], B["hnb"], B["kqb"]), 512 // 8, (512, P)))
                seq.append(("gvs32k2048", (w["v"], B["hnb"], B["vqb"]), 512 // 8, (512, P)))
            else:
                seq.append(("gv8k2048p", (w["q"], B["hnb"], B["qgb"]), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["k"], B["hnb"], B["kqb"]), (16,), (512,)))
                seq.append(("gv8k2048p", (w["v"], B["hnb"], B["vqb"]), (16,), (512,)))
            if spk is not None and spk.startswith("s"):
                R = spk[1:]
                seq.append((f"spka256m_{R}", (B["kqb"], B["vqb"], rig.SPTB[(ai, "m")]), (2*P,), ()))
                scr = rig.SCR_PF if pf else rig.SCR_DEC
                seq.append((f"spkq256s_{R}", (B["qgb"], scr, ptbl), (16*P, S), (S,)))
                seq.append(("spkc256", (B["qgb"], B["ayb"], scr), (16*P,), (8*S,)))
            else:
                spkan = "spka256" if spk is None else f"spka256m_{spk[1:] if spk.startswith('m') else spk}"
                seq.append((spkan, (B["kqb"], B["vqb"], rig.SPTB[(ai, "m")] if spk else ptbl), (2*P,), ()))
                spkn = "spkq256" if spk is None else f"spkq256m_{spk[1:] if spk.startswith('m') else spk}"
                pt = ptbl if spk is None else rig.SPTB[(ai, "m")]
                seq.append((spkn, (B["qgb"], B["ayb"], pt), (16*P,), ()))
            if PFT:
                seq.append(("pgmq8m32", (w["o"], B["ayb"], hin, hmid), (P // 32) * 32, (P,)))
            else:
                seq.append(("gv8k4096r", (w["o"], B["ayb"], hin, hmid), (64,), (2048,)))
        seq.append(("rmsz2048g", (hmid, w["pn"], B["hnb"]), (P,), ()))
        upk = "gx8e256up4" if rig.man["routed"][L]["types"]["gate"] == "IQ4_XS" else "gx8e256up"
        dnk = "gx8e256dn6" if rig.man["routed"][L]["types"]["down"] == "Q6_K" else "gx8e256dn"
        # ---- PB1 (Part B step 1): THE PAIRWISE MoE FUSION (MM_FUSE2=1) ----
        # rt8e256+shexp8 -> rtsh8 (one launch) and gx8e256up+gx8e256dn ->
        # gxdn8 (the act passes through smem instead of the global actb).
        # VERBATIM body merges => bit-exact by construction (gated). The
        # launch-serialization law (~0.094ms/kernel) prices this at
        # -2 launches/layer = -80/cycle on the P=1 T1 graph (~-7.5ms).
        if os.getenv("MM_FUSE2", "0") == "1" and P == 1 and upk == "gx8e256up" and dnk == "gx8e256dn":
            if "rtsh8" not in rig.K:
                raise RuntimeError("MM_FUSE2=1 but the PB1 fused programs are not loaded "
                                   "(MM_P7_lib.load_pb1_fused)")
            seq.append(("rtsh8", (w["rt"], w["wsh"], w["sg"], w["su"], w["sd"], B["hnb"],
                                  B["eidsb"], B["gatesb"], B["sgb"], B["shb"]), (P,), ()))
            seq.append(("gxdn8", (rig.PTB_UP[L], rig.PTB_DN[L], B["eidsb"], B["hnb"],
                                  rig.gridf, rig.iq4nl, B["partsb"]), (P*8,), ()))
        else:
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
                # SESSION C: the dn act-restaging fold (BIT-EXACT, 1.32-1.36x
                # POC) preferred when its cubin is loaded; no env key (same
                # outputs bit-for-bit -> pcache nodes stay valid).
                gdn = "gxm_dn6" if dnk == "gx8e256dn6" else (
                    "gxm_dnf" if "gxm_dnf" in rig.K else "gxm_dn")
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
        seq.append(("cmbz2048", (B["partsb"], B["gatesb"], B["sgb"], B["shb"], hmid, hin), (P,), ()))
    if with_head and head_mode == "full":
        seq.append(("rmsz2048g", (B["hA"], rig.ONORM, B["normhb"]), (P,), ()))
        seq.append(("h6k2048", (rig.HEAD, B["normhb"], rig.logitsb), (7760,), (248320,)))
    elif with_head and head_mode == "am":
        hk = f"h6kam_{P}"
        seq.append(("rmsz2048g", (B["hA"], rig.ONORM, B["normhb"]), (P,), ()))
        seq.append((hk, (rig.HEAD, B["normhb"], rig.PARTB), (7760,), ()))
        seq.append(("amred36", (rig.PARTB, rig.AMDB), (1,), (P,)))
    if tail_acc_K:
        drafts_v = idsb.offset(offset=4, size=8*4)
        seq.append(("acc36", (rig.AMDB, drafts_v, rig.MB, rig.EB), (1,), (tail_acc_K,)))
        seq.append(("selc36", (rig.SLOTS, rig.CSLOTS, rig.MB, rig.SALL, rig.CSALL), (8040,), ()))
    def _grid(n, g):
        # THE SILENT-NO-LAUNCH LAW (P7): MG._build does (grid+(1,))[:3] for
        # tuples -- a 1-tuple becomes a malformed 2-tuple global_size and the
        # graph NEVER EXECUTES (top1=0 = logitsb never written). L56's _grid
        # normalizes 1-tuples -> scalars; pass through ONLY the real 2D grids.
        if isinstance(g, tuple) and len(g) == 2: return g   # (16P, S) split grids
        gx = g[0] if isinstance(g, tuple) else g
        if n in ("gv8k2048p", "gv8k4096r") and P > 1:
            return (gx, P)
        return gx
    return [(n, b, _grid(n, g), v) for n, b, g, v in seq]

def mkgraph(rig, seq, tag, fence_every=1024):
    LSZ = rig.LSZ7
    return GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq], tag, fence_every=fence_every)


def load_pb1_fused(rig):
    """PB1: load the fused MoE kernels (rtsh8 + gxdn8) + their local sizes.
    Call BEFORE build_seq7 when MM_FUSE2=1 (the daemon host + the gates)."""
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    dev = rig.dev
    for stem, sym in (("MM_PB1_rtsh8", "rtsh8"), ("MM_PB1_gxdn8", "gxdn8")):
        lib = open(f"{BASE}/{stem}.cubin", "rb").read()
        rig.K[sym] = NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                                            signature=tuple()))
        L56.LSZ[sym] = (1024, 1, 1)
    rig.LSZ7 = dict(L56.LSZ)
    dev.synchronize()
    print("[pb1] fused MoE kernels loaded (rtsh8 + gxdn8)", flush=True)

# ============================================================================
# THE SPLIT-KV NUMPY ANCHOR (mirrors spkq256s+spkc256 exactly)
# ============================================================================
def spkq_h_split_ref(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos, h, S):
    """Kernel-order mirror of spkq256s+spkc256 (the P7 split-anchor).
    Dot: WARP-PER-POSITION, lane l covers dims l*8..l*8+7 (j-asc partials,
    TWO multiplies per elem: k*ks then q*that), 5-xor tree over the 32 lanes
    (no warp-asc -- one warp owns the position), *0.0625.
    Splits s = [sL/S,(s+1)L/S); warp w takes begin+w+8i (i asc); per warp
    ONLINE max-rescale; partial (m,z,out[256]) at index s*8+w; combine i-asc:
    w_i=expf(m_i-m*), Z+=w_i*z_i, acc+=w_i*out_i, y=(acc/Z)*sg."""
    q = _f(qg[h*512:h*512+256]); gate = _f(qg[h*512+256:h*512+512])
    qn = rmszc_ref(q, qw)
    qr = rope_apply(qn[None, :], cos[None, :], sin[None, :])[0]
    L = pos + 1
    kd = _f(Kq[:L].astype(np.float32) * np.repeat(Ks[:L], 128, axis=1))
    # per-lane j-asc partials: lane l <- dims l*8..l*8+7
    lanes = np.arange(32)
    lp = np.zeros((L, 32), dtype=np.float32)
    for j in range(8):
        lp = _f(lp + _f(qr[lanes*8 + j][None, :] * kd[:, lanes*8 + j]))
    for o in (16, 8, 4, 2, 1):
        lp = _f(lp + lp[:, np.arange(32) ^ o])
    scores = _f(lp[:, 0] * SCA)
    vd = _f(Vq[:L].astype(np.float32) * np.repeat(Vs[:L], 128, axis=1))
    NP = 8 * S
    ms = np.full(NP, -3.402823466e38, dtype=np.float32)
    zs = np.zeros(NP, dtype=np.float32)
    outs = np.zeros((NP, 256), dtype=np.float32)
    for s in range(S):
        begin = s * L // S; end = (s + 1) * L // S
        for w in range(8):
            i = s * 8 + w
            m = F32(-3.402823466e38); z = F32(0.0); acc = np.zeros(256, dtype=np.float32)
            for p in range(begin + w, end, 8):
                mn = F32(max(m, scores[p]))
                r = np.exp(_f(m - mn), dtype=np.float32)
                e = np.exp(_f(scores[p] - mn), dtype=np.float32)
                z = _f(_f(z * r) + e)
                acc = _f(acc * r + _f(e * vd[p]))
                m = mn
            ms[i] = m; zs[i] = z; outs[i] = acc
    mstar = ms[0]
    for i in range(1, NP):
        if ms[i] > mstar: mstar = ms[i]
    Z = F32(0.0); acc = np.zeros(256, dtype=np.float32)
    for i in range(NP):
        wv = np.exp(_f(ms[i] - mstar), dtype=np.float32)
        Z = _f(Z + _f(wv * zs[i]))
        acc = _f(acc + _f(wv * outs[i]))
    sg = _f(_f(1.0) / (_f(1.0) + np.exp(-gate, dtype=np.float32)))
    return _f(_f(acc / Z) * sg)

# the anchor dispatch hook (ladder rebase/first-64 use the split ref)
_SPLIT_S = [0]
def install_split_anchor(S):
    import MM_P34_ports as P34
    if not hasattr(P34, "_orig_spkq_h_ref"):
        P34._orig_spkq_h_ref = P34.spkq_h_ref
        def _disp(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos, h):
            if _SPLIT_S[0]:
                return spkq_h_split_ref(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos, h, _SPLIT_S[0])
            return P34._orig_spkq_h_ref(qg, qw, Kq, Ks, Vq, Vs, cos, sin, pos, h)
        P34.spkq_h_ref = _disp
    _SPLIT_S[0] = S

# ============================================================================
# THE FOLDED SPEC ENGINE (per cycle: 2 cpu-view writes + 1 submit/wait + 1 read)
# ============================================================================
def mk_lookup():
    # the P6 incremental 4-gram lookup (import from the P6 module)
    import MM_P6_run as P6
    return P6.Lookup()

class SpecEngine7:
    """K-mix T1/D2/D8 with the folded host path. gr1 = T1 (am head optional
    via am1=True + rig.am_view read; else full head + logits readback)."""
    def __init__(self, rig, gr1, gr2, gr8, am1=False):
        self.rig = rig; self.gr1 = gr1; self.gr2 = gr2; self.gr8 = gr8
        self.am1 = am1
    def t1_top(self):
        if self.am1:
            return int(self.rig.am_view[0])
        return int(np.argmax(self.rig.dn(self.rig.logitsb, (248320,))))
    def feed_prompt(self, ids):
        rig = self.rig
        rig.reset_states(1024)
        lk = mk_lookup()
        for pos, tid in enumerate(ids):
            rig.feed(tid, pos)
            self.gr1.step()
            lk.append(int(tid))
        t = self.t1_top()
        lk.append(t)
        return t, lk
    def reset_stats(self):
        self.stats = {"cyc": 0, "cyc_t1": 0, "cyc_d2": 0, "cyc_d8": 0, "tok": 0,
                      "ms": 0.0, "ms_t1": 0.0, "ms_d2": 0.0, "ms_d8": 0.0,
                      "hits": 0, "msum": 0, "m_hist": [], "n_hist": [], "wait_ms": 0.0}
    def generate(self, t, lk, pos0, ntok, mode="spec"):
        rig = self.rig
        self.reset_stats()
        gen = [t]; pos = pos0
        t0 = time.perf_counter()
        while len(gen) < ntok:
            cur = lk.h[-1]
            n, drafts = (0, [])
            if mode == "spec":
                n, drafts = lk.scan(kmax=8)
            if n >= 8 and len(drafts) >= 8:
                K = 8; gr = self.gr8
            elif n >= 4 and len(drafts) >= 2:
                K = 2; gr = self.gr2
            else:
                K = 0
            if K == 0:
                tc = time.perf_counter()
                rig.feed(cur, pos)
                self.gr1.step()
                t2 = self.t1_top()
                self.stats["ms_t1"] += time.perf_counter() - tc
                self.stats["cyc_t1"] += 1
                gen.append(t2); lk.append(t2); pos += 1
            else:
                tc = time.perf_counter()
                rig.feed9([cur] + list(drafts[:K]) + [0]*(8-K), pos)
                tw = time.perf_counter()
                gr.step()
                self.stats["wait_ms"] += time.perf_counter() - tw
                m = int(rig.eb_view[0]); bnd = int(rig.eb_view[1])
                emitted = list(drafts[:m]) + [bnd]
                gen.extend(emitted)
                for e in emitted: lk.append(e)
                pos += m + 1
                self.stats[f"ms_{'d8' if K==8 else 'd2'}"] += time.perf_counter() - tc
                self.stats[f"cyc_{'d8' if K==8 else 'd2'}"] += 1
                self.stats["hits"] += 1
                self.stats["msum"] += m
                self.stats["m_hist"].append(m)
                self.stats["n_hist"].append(n)
            self.stats["cyc"] += 1
        dt = time.perf_counter() - t0
        self.stats["tok"] = len(gen)
        self.stats["ms"] = dt * 1e3
        return gen

# ============================================================================
# THE PREFILL FEEDER (chunk-256; tail per-token T1)
# ============================================================================
class PrefillFeed:
    def __init__(self, rig, gr_pf, gr1):
        self.rig = rig; self.gr_pf = gr_pf; self.gr1 = gr1
    def feed(self, ids, pos0=0, verbose=False):
        rig = self.rig
        ids = [int(x) for x in ids]
        p = 0; chunks = 0
        t0 = time.perf_counter()
        while p + 256 <= len(ids):
            rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(ids[p:p+256], dtype=np.int32).data)
            rig.pos_view[0] = pos0 + p
            self.gr_pf.step()
            p += 256; chunks += 1
        pf_ms = (time.perf_counter() - t0) * 1e3
        for q in range(p, len(ids)):          # tail: per-token T1
            rig.feed(ids[q], pos0 + q)
            self.gr1.step()
        return chunks, pf_ms
