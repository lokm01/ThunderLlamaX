#!/usr/bin/env python3
"""MM P5+P6 shared harness library: program load (P34+P6 cubins, build-on-
demand via the nvcc container), weight upload, buffers at CTX_ALLOC=65536,
the KA-SLAB POOL (P5.3: one host-mapped arena carved across ALL graph
variants -- the KERNARGS-SLAB LEAK law mitigation), MG graphs with rebuild
cadence (the ~950-cycle dext budget), build_seq (T1/D2/D8 + head modes +
spkq256m ladder swaps), and the run helpers.

The GPU process law: ONE process per boot-session; every stage resumable
via the progress file.
"""
import os, sys, time, json, hashlib, subprocess
import numpy as np

os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
PACK_DIR = "~/models36/packed"
CTX_ALLOC = 65536
PMAX = 9

from MM_P2_ports import (load_router, load_wsh, load_shexp_raw, iq4nl_f32,
    iq3s_grid_f32, bank_reader, PACK)
from MM_P34_ports import (GDN_LAYERS, ATTN_LAYERS, load_f32, load_q8)

LSZ = {"gconv36_1": (256,1,1), "gconv36_3": (256,1,1), "gconv36_9": (256,1,1),
       "gconv36s_3": (256,1,1), "gconv36s_9": (256,1,1),
       "k2s36_1": (256,1,1), "k2s36_3": (256,1,1), "k2s36_9": (256,1,1),
       "k2s36s_3": (256,1,1), "k2s36s_9": (256,1,1),
       "selc36": (256,1,1), "amred36": (256,1,1),
       "h6kam_3": (1024,1,1), "h6kam_9": (1024,1,1),
       "spkq256m_4096": (256,1,1), "spkq256m_16384": (256,1,1), "spkq256m_65536": (256,1,1),
       "spka256m_4096": (256,1,1), "spka256m_16384": (256,1,1), "spka256m_65536": (256,1,1),
       "embg248": (1024,1,1), "rmsz2048g": (256,1,1), "gv8k2048p": (1024,1,1),
       "gv8k4096r": (1024,1,1), "gvf32ab": (1024,1,1),
       "spka256": (256,1,1), "spkq256": (256,1,1), "rt8e256": (1024,1,1),
       "shexp8": (1024,1,1), "gx8e256up": (1024,1,1), "gx8e256up4": (1024,1,1),
       "gx8e256dn": (1024,1,1), "gx8e256dn6": (1024,1,1), "cmbz2048": (256,1,1),
       "h6k2048": (1024,1,1)}

NVCC_ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
                DOCKER_HOST="unix://~/.colima/default/docker.sock")

# (stem, symbol, variant-macro or None, extra -D flags)
CUBIN_SPECS = [
    ("MM_P34_gconv36", "gconv36", ["1", "3", "9"], []),
    ("MM_P34_k2s36",   "k2s36",   ["1", "3", "9"], []),
    ("MM_P34_gvf32ab", "gvf32ab", [None], []),
    ("MM_P34_spka256", "spka256", [None], []),
    ("MM_P34_spkq256", "spkq256", [None], []),
    ("MM_P34_h6k2048", "h6k2048", [None], []),
    ("MM_P34_cmbz2048", "cmbz2048", [None], []),
    ("MM_P34_gv8k4096r", "gv8k4096r", [None], []),
    ("MM_P34_gv8k2048p", "gv8k2048p", [None], []),
    ("MM_P34_rmsz2048", "rmsz2048g", [None], []),
    ("MM_P6_k2s36s",   "k2s36s",   ["3", "9"], []),
    ("MM_P6_gconv36s", "gconv36s", ["3", "9"], []),
    ("MM_P6_selc36",   "selc36",   [None], []),
    ("MM_P6_h6kam",    "h6kam",    ["3", "9"], ["-DPP={V}"]),
    ("MM_P6_amred36",  "amred36",  [None], []),
    ("MM_P6_spkq256m", "spkq256m", ["4096", "16384", "65536"], ["-DCTXS={V}"]),
    ("MM_P34_spka256", "spka256m", ["4096", "16384", "65536"], ["-DCTXS={V}"]),
]
EXTRA_PROGS = [
    ("MM_P1_rt8e256", "rt8e256"), ("MM_P1_gx8e256up", "gx8e256up"),
    ("MM_P2_gx8e256up4", "gx8e256up4"), ("MM_P2_gx8e256dn", "gx8e256dn"),
    ("MM_P2_gx8e256dn6", "gx8e256dn6"), ("MM_P2_shexp8", "shexp8"),
    ("MM_P2_embg248", "embg248"),
]

SCALAR_TAIL = {"gv8k2048p", "gv8k4096r", "h6k2048", "embg248", "amred36"}

def build_cubins(only=None):
    built = {}
    for stem, sym, variants, extra in CUBIN_SPECS:
        for v in (variants if variants else [None]):
            name = sym if v is None else f"{sym}_{v}"
            if only and name not in only: continue
            cb = f"{BASE}/{stem}_{v}.cubin" if v else f"{BASE}/{stem}.cubin"
            if not os.path.exists(cb):
                cmd = f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={cb} {BASE}/{stem}.cu"
                if v:
                    ds = [f"-DTMAX={v}"] + [x.replace("{V}", v) for x in extra]
                    cmd += " " + " ".join(ds)
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=NVCC_ENV)
                if r.returncode:
                    print(r.stderr[-3000:]); sys.exit(1)
            built[name] = cb
    return built

def audit_cubin(path, name):
    r = subprocess.run(f"cuobjdump -res-usage {path}", shell=True, capture_output=True, text=True, env=NVCC_ENV)
    for l in r.stdout.splitlines():
        lu = l.upper()
        if "SPILL" in lu and "STORES" not in lu:
            m = [w for w in lu.split() if "SPILL" in w]
            print(f"[audit {name}] {l.strip()}", flush=True)

class Rig:
    """The one-GPU-process rig: programs + weights + buffers + graphs."""
    def __init__(self, build_only=None, load_p6=True):
        from engine0 import dev
        from tinygrad.device import BufferSpec, TinyELF
        from tinygrad.runtime.ops_nv import NVProgram
        from tinygrad.uop.ops import UOp
        from tinygrad.dtype import dtypes
        from tinygrad.helpers import round_up
        from tinygrad.runtime.ops_nv import NVComputeQueue, nv_wait_timeline
        self.dev = dev; self.BufferSpec = BufferSpec; self.UOp = UOp
        self.dtypes = dtypes; self.round_up = round_up
        self.NVComputeQueue = NVComputeQueue; self.nv_wait_timeline = nv_wait_timeline
        self.keep = []
        INT_SIG = (None, 4, dtypes.int32, ())
        self.INT_SIG = INT_SIG
        built = build_cubins(build_only)
        K = {}
        for name, cb in built.items():
            lib = open(cb, "rb").read()
            K[name] = NVProgram(dev, TinyELF(lib=lib, name=name, target=dev.renderer.target,
                          signature=(INT_SIG,) if name in SCALAR_TAIL else tuple()))
            if name.startswith(("k2s36s", "gconv36s", "selc36", "h6kam", "amred36", "spkq256m")):
                audit_cubin(cb, name)
        for stem, sym in EXTRA_PROGS:
            lib = open(f"{BASE}/{stem}.cubin", "rb").read()
            K[sym] = NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                          signature=(INT_SIG,) if sym in SCALAR_TAIL else tuple()))
        self.K = K
        dev.synchronize()
        print(f"[rig] {len(K)} programs loaded", flush=True)
        self._upload_weights()
        self._alloc_buffers(load_p6)
        self.pool = KAPool(dev, self.keep)
        self.n_replays_since_fence = 0
        self.fence_count = 0

    # ---------- host<->dev ----------
    def up(self, a):
        a = np.ascontiguousarray(a); self.keep.append(a)
        b = self.dev.allocator.alloc(a.nbytes, self.BufferSpec())
        self.dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def up_big(self, np_getter, nbytes):
        b = self.dev.allocator.alloc(nbytes, self.BufferSpec())
        CH = 64 << 20; off = 0
        while off < nbytes:
            n = min(CH, nbytes - off)
            a = np.ascontiguousarray(np_getter(off, n))
            self.dev.allocator._copyin(b.offset(offset=off, size=n), memoryview(a.data).cast("B"))
            self.dev.synchronize(); del a; off += n
        return b
    def alloc(self, nbytes):
        b = self.dev.allocator.alloc(nbytes, self.BufferSpec()); self.keep.append(b); return b
    def dn(self, b, shape, dtype=np.float32):
        n = int(np.prod(shape))
        mv = memoryview(bytearray(int(n)*np.dtype(dtype).itemsize)).cast("B")
        self.dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()

    def _upload_weights(self):
        import os
        t0 = time.time()
        def file_uploader(path):
            f = open(path, "rb")
            def getter(off, n):
                f.seek(off); return np.frombuffer(f.read(n), dtype=np.uint8)
            return getter
        self.gridf = self.up(iq3s_grid_f32())
        self.iq4nl = self.up(iq4nl_f32())
        man = bank_reader()
        self.man = man
        PTB_UP = {}; PTB_DN = {}
        for L in range(40):
            meta = man["routed"][L]; fl = meta["files"]
            if len(fl) == 1:
                f = fl[0]
                b = self.up_big(file_uploader(os.path.join(PACK, f["file"])), f["bytes"])
                PTB_UP[L] = self.up(np.array([b.va_addr + e*f["slab"] + f["offsets"]["gate"] for e in range(256)], dtype=np.uint64))
                PTB_DN[L] = self.up(np.array([b.va_addr + e*f["slab"] + f["offsets"]["down"] for e in range(256)], dtype=np.uint64))
            else:
                fg = next(f for f in fl if "gate" in f["offsets"])
                fd = next(f for f in fl if "down" in f["offsets"])
                bg = self.up_big(file_uploader(os.path.join(PACK, fg["file"])), fg["bytes"])
                bd = self.up_big(file_uploader(os.path.join(PACK, fd["file"])), fd["bytes"])
                PTB_UP[L] = self.up(np.array([bg.va_addr + e*fg["slab"] + fg["offsets"]["gate"] for e in range(256)], dtype=np.uint64))
                PTB_DN[L] = self.up(np.array([bd.va_addr + e*fd["slab"] + fd["offsets"]["down"] for e in range(256)], dtype=np.uint64))
        self.PTB_UP, self.PTB_DN = PTB_UP, PTB_DN
        W = {}
        for L in range(40):
            w = {}
            w["an"] = self.up(load_f32(L, "attn_norm_weight"))
            w["pn"] = self.up(load_f32(L, "post_attention_norm_weight"))
            w["rt"] = self.up(load_router(L)); w["wsh"] = self.up(load_wsh(L))
            wg, wu, wd = load_shexp_raw(L)
            w["sg"] = self.up(wg); w["su"] = self.up(wu); w["sd"] = self.up(wd)
            if L in GDN_LAYERS:
                w["qkv"] = self.up(load_q8(L, "attn_qkv_weight", 8192))
                w["z"] = self.up(load_q8(L, "attn_gate_weight", 4096))
                w["al"] = self.up(load_f32(L, "ssm_a")); w["dt"] = self.up(load_f32(L, "ssm_dt_bias"))
                w["adt"] = self.up(np.concatenate([load_f32(L, "ssm_a"), load_f32(L, "ssm_dt_bias")]))
                w["wa"] = self.up(load_f32(L, "ssm_alpha_weight")); w["wb"] = self.up(load_f32(L, "ssm_beta_weight"))
                w["cw"] = self.up(load_f32(L, "ssm_conv1d_weight"))
                w["sn"] = self.up(load_f32(L, "ssm_norm_weight"))
                w["out"] = self.up(load_q8(L, "ssm_out_weight", 2048, 4096))
            else:
                w["q"] = self.up(load_q8(L, "attn_q_weight", 8192))
                w["k"] = self.up(load_q8(L, "attn_k_weight", 512))
                w["v"] = self.up(load_q8(L, "attn_v_weight", 512))
                w["qw"] = self.up(load_f32(L, "attn_q_norm_weight"))
                w["kw"] = self.up(load_f32(L, "attn_k_norm_weight"))
                w["o"] = self.up(load_q8(L, "attn_output_weight", 2048, 4096))
            W[L] = w
        self.W = W
        self.EMB = self.up_big(file_uploader(f"{PACK}/trunk/token_embd_weight.bin"), 248320*2176)
        self.HEAD = self.up_big(file_uploader(f"{PACK}/trunk/output_weight.bin"), 248320*1680)
        self.ONORM = self.up(load_f32(None, "output_norm_weight"))
        self.dev.synchronize()
        print(f"[rig] banks+trunk uploaded in {time.time()-t0:.0f}s", flush=True)

    def _alloc_buffers(self, load_p6):
        from MM_P34_ports import rope_tables
        dev = self.dev; BS = self.BufferSpec
        CTX = CTX_ALLOC
        P = PMAX
        self.hA = self.alloc(P*2048*4); self.hB = self.alloc(P*2048*4); self.hnb = self.alloc(P*2048*4)
        self.qkvb = self.alloc(P*8192*4); self.zb = self.alloc(P*4096*4); self.abb = self.alloc(P*64*4)
        self.qkvsb = self.alloc(P*8192*4); self.gyb = self.alloc(P*4096*4)
        self.qgb = self.alloc(P*8192*4); self.kqb = self.alloc(P*512*4); self.vqb = self.alloc(P*512*4)
        self.ayb = self.alloc(P*4096*4)
        self.eidsb = self.alloc(P*8*2); self.gatesb = self.alloc(P*8*4); self.sgb = self.alloc(P*4)
        self.actb = self.alloc(P*8*512*4); self.partsb = self.alloc(P*8*2048*2); self.shb = self.alloc(P*2048*4)
        self.normhb = self.alloc(P*2048*4); self.logitsb = self.alloc(248320*4)
        self.idsb = self.up(np.zeros(P, dtype=np.int32))
        self.POSB = self.up(np.array([0], dtype=np.int32))
        self.SALL = self.alloc(30*32*128*128*4)
        self.SV = [self.SALL.offset(offset=L*32*128*128*4, size=32*128*128*4) for L in range(30)]
        self.CSALL = self.alloc(30*8192*3*4)
        self.CSV = [self.CSALL.offset(offset=L*8192*3*4, size=8192*3*4) for L in range(30)]
        self.KVQ = {}; self.KVS = {}; self.VVQ = {}; self.VVS = {}
        for ai in range(10):
            self.KVQ[ai] = self.up(np.zeros((2, CTX, 256), dtype=np.int8))
            self.VVQ[ai] = self.up(np.zeros((2, CTX, 256), dtype=np.int8))
            self.KVS[ai] = self.up(np.zeros((2, CTX, 2), dtype=np.float32))
            self.VVS[ai] = self.up(np.zeros((2, CTX, 2), dtype=np.float32))
        cos, sin = rope_tables(CTX)
        self.COSB, self.SINB = self.up(cos), self.up(sin)
        # scratch for spkq256m: (16*P_MAX) rows x CTX_ALLOC fp32 -- the row
        # stride is the KERNEL'S CTXS (rung), so worst case = 144 x 65536
        # (the 16k rung at P=9 overflowed the old 16x65536 sizing -> device
        # fault; 4k fit by luck: 143*4096 < 1M)
        self.SPSCR = self.alloc(144*CTX*4)
        self.SPTB = {}
        for ai in range(10):
            L = ATTN_LAYERS[ai]
            self.SPTB[ai] = self.up(np.array([self.W[L]["kw"].va_addr, self.W[L]["qw"].va_addr, self.COSB.va_addr, self.SINB.va_addr,
                                self.KVQ[ai].va_addr, self.KVS[ai].va_addr, self.VVQ[ai].va_addr, self.VVS[ai].va_addr,
                                self.POSB.va_addr], dtype=np.uint64))
            self.SPTB[(ai, "m")] = self.up(np.array([self.W[L]["kw"].va_addr, self.W[L]["qw"].va_addr, self.COSB.va_addr, self.SINB.va_addr,
                                self.KVQ[ai].va_addr, self.KVS[ai].va_addr, self.VVQ[ai].va_addr, self.VVS[ai].va_addr,
                                self.POSB.va_addr, self.SPSCR.va_addr], dtype=np.uint64))
        self.ZS = np.zeros(30*32*128*128, dtype=np.float32).tobytes()
        self.ZCS = np.zeros(30*8192*3, dtype=np.float32).tobytes()
        self._zkv = {}
        if load_p6:
            self.SLOTS = self.alloc(9*30*32*128*128*4)      # [9][30][2MiB]
            self.CSLOTS = self.alloc(9*30*8192*3*4)         # [9][30][96KiB]
            self.SLOTV = [self.SLOTS.offset(offset=L*32*128*128*4, size=32*128*128*4) for L in range(30)]
            self.CSLOTV = [self.CSLOTS.offset(offset=L*8192*3*4, size=8192*3*4) for L in range(30)]
            self.AMDB = self.alloc(9*4)                      # argmax per probe seat
            self.PARTB = self.alloc(9*7760*8)                  # h6kam packed partials [9][7760] u64
            self.MB = self.up(np.array([0], dtype=np.int32)) # m slot (device int)
        dev.synchronize()
        print("[rig] buffers allocated", flush=True)

    def reset_states(self, n=1024):
        dev = self.dev
        dev.allocator._copyin(self.SALL, memoryview(self.ZS))
        dev.allocator._copyin(self.CSALL, memoryview(self.ZCS))
        if n not in self._zkv:
            self._zkv[n] = np.zeros((2, n, 256), dtype=np.int8).tobytes()
            self._zkv[(n, "s")] = np.zeros((2, n, 2), dtype=np.float32).tobytes()
        zq, zs = self._zkv[n], self._zkv[(n, "s")]
        for ai in range(10):
            dev.allocator._copyin(self.KVQ[ai], memoryview(zq))
            dev.allocator._copyin(self.VVQ[ai], memoryview(zq))
            dev.allocator._copyin(self.KVS[ai], memoryview(zs))
            dev.allocator._copyin(self.VVS[ai], memoryview(zs))

    # ---------- seq builder ----------
    def build_seq(self, P, tg, tk, with_head=True, head_mode="full", spk=None, slots=False):
        """head_mode: full = h6k2048 logits (P must be 1); am = rmsz+h6kam_P+amred36.
        spk: None -> spkq256 (ctx<=1024 class); 'm4096'/'m16384'/'m65536' -> spkq256m_<n>.
        slots: True -> gconv36s/k2s36s with per-t state slots (spec trains)."""
        K = self.K
        self._P2D = P
        seq = [("embg248", (self.EMB, self.idsb, self.hA), max(1, (P+31)//32), (P,))]
        for L in range(40):
            w = self.W[L]
            hin = self.hA; hmid = self.hB
            seq.append(("rmsz2048g", (hin, w["an"], self.hnb), (P,), ()))
            if L in GDN_LAYERS:
                gi = GDN_LAYERS.index(L)
                seq.append(("gv8k2048p", (w["qkv"], self.hnb, self.qkvb), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["z"], self.hnb, self.zb), (128,), (4096,)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], self.hnb, self.abb), (P,), ()))
                if slots:
                    # tg/tk carry the T-variant names (gconv36s_3/9, k2s36s_3/9)
                    seq.append((tg, (w["cw"], self.qkvb, self.CSV[gi], self.qkvsb, self.CSLOTV[gi]), (32,), ()))
                    seq.append((tk, (self.qkvsb, self.abb, w["adt"], w["sn"], self.zb, self.SV[gi], self.gyb, self.SLOTV[gi]), (32,), ()))
                else:
                    seq.append((tg, (w["cw"], self.qkvb, self.CSV[gi], self.qkvsb), (32,), ()))
                    seq.append((tk, (self.qkvsb, self.abb, w["al"], w["dt"], w["sn"], self.zb, self.SV[gi], self.gyb), (32,), ()))
                seq.append(("gv8k4096r", (w["out"], self.gyb, hin, hmid), (64,), (2048,)))
            else:
                ai = ATTN_LAYERS.index(L)
                ptbl = self.SPTB[ai] if spk is None else self.SPTB[(ai, "m")]
                spkn = "spkq256" if spk is None else f"spkq256m_{spk}"
                seq.append(("gv8k2048p", (w["q"], self.hnb, self.qgb), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["k"], self.hnb, self.kqb), (16,), (512,)))
                seq.append(("gv8k2048p", (w["v"], self.hnb, self.vqb), (16,), (512,)))
                spkan = "spka256" if spk is None else f"spka256m_{spk}"
                seq.append((spkan, (self.kqb, self.vqb, self.SPTB[ai]), (2*P,), ()))
                seq.append((spkn, (self.qgb, self.ayb, ptbl), (16*P,), ()))
                seq.append(("gv8k4096r", (w["o"], self.ayb, hin, hmid), (64,), (2048,)))
            seq.append(("rmsz2048g", (hmid, w["pn"], self.hnb), (P,), ()))
            seq.append(("rt8e256", (w["rt"], w["wsh"], self.hnb, self.eidsb, self.gatesb, self.sgb), (P,), ()))
            seq.append(("shexp8", (w["sg"], w["su"], w["sd"], self.hnb, self.shb), (P,), ()))
            upk = "gx8e256up4" if self.man["routed"][L]["types"]["gate"] == "IQ4_XS" else "gx8e256up"
            upbufs = (self.PTB_UP[L], self.eidsb, self.hnb, self.iq4nl, self.actb) if upk == "gx8e256up4" else (self.PTB_UP[L], self.eidsb, self.hnb, self.gridf, self.actb)
            seq.append((upk, upbufs, (P*8,), ()))
            dnk = "gx8e256dn6" if self.man["routed"][L]["types"]["down"] == "Q6_K" else "gx8e256dn"
            dnbufs = (self.PTB_DN[L], self.eidsb, self.actb, self.partsb) if dnk == "gx8e256dn6" else (self.PTB_DN[L], self.eidsb, self.actb, self.iq4nl, self.partsb)
            seq.append((dnk, dnbufs, (P*8,), ()))
            seq.append(("cmbz2048", (self.partsb, self.gatesb, self.sgb, self.shb, hmid, hin), (P,), ()))
        if with_head and head_mode == "full":
            seq.append(("rmsz2048g", (self.hA, self.ONORM, self.normhb), (P,), ()))
            seq.append(("h6k2048", (self.HEAD, self.normhb, self.logitsb), (7760,), (248320,)))
        elif with_head and head_mode == "am":
            seq.append(("rmsz2048g", (self.hA, self.ONORM, self.normhb), (P,), ()))
            seq.append((f"h6kam_{P}", (self.HEAD, self.normhb, self.PARTB), (7760,), ()))
            seq.append(("amred36", (self.PARTB, self.AMDB), (1,), (P,)))
        # THE 2D-GRID LAW: gv8k2048p/gv8k4096r are (rows/32, P) grids with
        # seat = blockIdx.y -- a 1D grid runs seat 0 ONLY (the G3 amds[1:]==0
        # root cause; the S8 smoke was det-but-seat-0-only). All other
        # P-batched kernels parallelize flat (blockIdx.x / warps).
        def _grid(n, g):
            gx = g[0] if isinstance(g, tuple) else g
            if n in ("gv8k2048p", "gv8k4096r") and self._P2D and self._P2D > 1:
                return (gx, self._P2D)
            return gx
        return [(n, b, _grid(n, g), v) for n, b, g, v in seq]

# ============================ P5.3: THE KA-SLAB POOL ============================
class KAPool:
    """One host-mapped arena carved across ALL graph builds (the leak-law
    mitigation): slab count target 1-2 per process instead of 1 per graph.
    Kernargs are written at build/rebuild time; replays only READ them, so
    disjoint slices in one mapping are safe."""
    def __init__(self, dev, keep):
        self.dev = dev; self.keep = keep
        self.buf = None; self.off = 0; self.slabs = 0; self.carves = 0
    def carve(self, nbytes):
        SLAB = 1 << 20
        if self.buf is None or self.off + nbytes > self.buf.size:
            alloc = max(nbytes, SLAB) + (1 << 20)
            self.buf = self.dev.allocator.alloc(alloc, __import__('tinygrad').device.BufferSpec(cpu_access=True, nolru=True))
            self.keep.append(self.buf); self.off = 0; self.slabs += 1
            print(f"[kapool] new slab #{self.slabs} ({alloc} B)", flush=True)
        b = self.buf.offset(offset=self.off, size=nbytes)
        self.off += (nbytes + 7) & ~7
        self.carves += 1
        return b

class MG:
    """The explicit-kernargs multi-graph (wait-each discipline)."""
    _seq = 0
    def __init__(self, rig, seq, tag):
        from tinygrad.device import BufferSpec
        from tinygrad.runtime.ops_nv import NVComputeQueue
        dev = rig.dev
        self.rig = rig; self.tag = tag
        MG._seq += 1; self.uid = MG._seq
        self.prev = rig.UOp.variable(f"{tag}_p{self.uid}", 0, 0xffffffff, dtype=rig.dtypes.uint32)
        self.cur  = rig.UOp.variable(f"{tag}_c{self.uid}", 0, 0xffffffff, dtype=rig.dtypes.uint32)
        per = max(rig.round_up(p.kernargs_alloc_size, 8) for p, a, g, v in seq)
        self.kb = per*len(seq)
        self.ka = rig.pool.carve(self.kb + 8)
        self.seq = seq
        self._build()
        rig.pool_audit_note = f"slabs={rig.pool.slabs} carves={rig.pool.carves} bytes={rig.pool.off}"
    def _build(self):
        from tinygrad.runtime.ops_nv import NVComputeQueue
        rig = self.rig; dev = rig.dev
        q = NVComputeQueue(); q.memory_barrier()
        q.wait(dev.timeline_signal, self.prev)
        off = 0
        for p, bufs, grid, vals in self.seq:
            ab = self.ka.offset(offset=off, size=p.kernargs_alloc_size)
            st = p.fill_kernargs(tuple(bufs), tuple(vals), kernargs=ab)
            q.exec(p, st, (grid + (1,))[:3] if isinstance(grid, tuple) else (grid,1,1), LSZ[p.name])
            off += rig.round_up(p.kernargs_alloc_size, 8)
        q.signal(dev.timeline_signal, self.cur)
        self.q = q
    def rebuild(self):
        """Re-fill kernargs + re-record into the SAME ka slice (the 1024-cycle
        fence: resets the dext continuous-replay budget at quiescent points)."""
        self.rig.dev.synchronize()
        self._build()
    def submit(self, pv, cv):
        self.q.submit(self.rig.dev, {self.prev.expr: int(pv), self.cur.expr: int(cv)})

class GraphRunner:
    """Alternating MG pair + wait-each + the fence cadence."""
    def __init__(self, rig, seq, tag, fence_every=1024):
        self.rig = rig
        self.ga = MG(rig, seq, tag + "a")
        self.gb = MG(rig, seq, tag + "b")
        self.turn = 0
        self.fence_every = fence_every
        self.n = 0
    def step(self):
        m = self.ga if self.turn == 0 else self.gb
        self.turn = 1 - self.turn
        v = self.rig.dev.next_timeline()
        m.submit(v - 1, v)
        self.rig.nv_wait_timeline(self.rig.dev, v, what="step", timeout_s=120.0)
        self.n += 1; self.rig.n_replays_since_fence += 1
        if self.fence_every and self.rig.n_replays_since_fence >= self.fence_every:
            self.fence()
    def fence(self):
        self.ga.rebuild(); self.gb.rebuild()
        self.rig.n_replays_since_fence = 0
        self.rig.fence_count += 1

def feed_token(rig, tid, pos, P=1):
    rig.dev.allocator._copyin(rig.idsb, memoryview(np.full(P, tid, dtype=np.int32).tobytes()))
    rig.dev.allocator._copyin(rig.POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))

def t1_argmax(rig):
    return int(np.argmax(rig.dn(rig.logitsb, (248320,))))

def amds_read(rig, P):
    return [int(x) for x in rig.dn(rig.AMDB, (P,), np.int32)]
