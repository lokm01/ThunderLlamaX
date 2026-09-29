#!/usr/bin/env python3
"""MM P9 — THE MTP K=4 CHAIN for the MoE engine (the prose lever).

The MTP layer (blk.40 nextn, llama.cpp graph_mtp semantics decoded in P8S)
drafts K=4 tokens per cycle through its own attn+MoE layer + the SHARED head;
the P=5 probe verifies [cur + 4 MTP drafts] with the proven slots/acc36/
selc36 machinery (bit-exact vs T1 by construction, the D2/D8 class).

THE ONE-CYCLE-LATE CHAIN PROTOCOL (the EAGLE/llama.cpp standard shape):
  after a probe cycle at pos with accept m (eb = (m, bnd)):
    seed    : (hA[m], ids[m]) @ pos+m     -- the TRUE trunk pair (the last
              committed position); writes MTP-KV row pos+m TRUE; its own
              argmax c1 predicts pos+m+1 = the already-known bnd (discarded,
              the seed runner carries NO head).
    step 1  : (h0, bnd) @ pos+m+1          -- the TRUE boundary token; writes
              row pos+m+1 TRUE; argmax = draft1 (predicts pos+m+2).
    steps 2-4: (h_j, c_j) @ pos+m+2..      -- own guesses; drafts 2-4.
  THE KV INVARIANT (the trunk's own protocol transplanted): every row a
  chain reads was written by a true seed, an accepted draft's own step, or
  this chain's own earlier step (spka runs before spq within each step);
  rejected-draft rows are overwritten by the next seed before any read.
  D8/D2 lookup cycles leave gaps (never-written rows stay zero) -- the
  zero-prefix conditioning CLASS the P8S anchor measured (alpha 0.875 quote:
  its chain_kv was fresh zeros; the MTP attention self-dilutes at depth).

Kernels: gxk3e256up (Q3_K gate+up, VERBATIM the validated numpy dq port),
gxk4e256dn (Q4_K down), gv8k4096r2 (eh_proj over the [e_n || h_n] halves),
rowcp2048 (the committed-hidden carry). Everything else = the proven trunk
families pointed at the MTP's own weights/KV/controls.

Usage (MM_P9_run.py drives the stages; see its docstring for the gates).
"""
import os, sys, time
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
os.environ.setdefault("DEV", "NV")

from MM_P7_lib import build_seq7, mkgraph, SpecEngine7
import MM_P56_lib as L56
from MM_P56_lib import MG, GraphRunner

# ---- THE KAPool SLAB COHERENCE LAW (P9, empirical): MGs carved from the
# SHARED pool slab WEDGE (SKEDCHECK16_CTA_THREAD_DIMENSION_ZERO -- stale
# kernargs/QMDs read through the GPU-CACHEABLE slab mapping) once the GPU
# has prior traffic over the slab; only the boot-time graph set (kernargs
# written before any execution) is safe there. Late-built graphs need a
# DEDICATED uncached ka (sysmem, GPU_CACHEABLE_NO -- bidirectionally
# coherent, the cpu-fold control class). Proven: trunk graphs (pool,
# boot-written) step fine interleaved with dedicated-ka MTP graphs.
class MGUnc(MG):
    def __init__(self, rig, seq, tag):
        from tinygrad.device import BufferSpec
        dev = rig.dev
        self.rig = rig; self.tag = tag
        MG._seq += 1; self.uid = MG._seq
        self.prev = rig.UOp.variable(f"{tag}_p{self.uid}", 0, 0xffffffff, dtype=rig.dtypes.uint32)
        self.cur  = rig.UOp.variable(f"{tag}_c{self.uid}", 0, 0xffffffff, dtype=rig.dtypes.uint32)
        per = max(rig.round_up(p.kernargs_alloc_size, 8) for p, a, g, v in seq)
        self.kb = per*len(seq)
        self.ka = dev.allocator.alloc(self.kb + 8, BufferSpec(cpu_access=True, nolru=True, uncached=True))
        rig.keep.append(self.ka)
        self.seq = seq
        self._build()

class GraphRunnerUnc(GraphRunner):
    def __init__(self, rig, seq, tag, fence_every=0):
        self.rig = rig
        self.ga = MGUnc(rig, seq, tag + "a")
        self.gb = MGUnc(rig, seq, tag + "b")
        self.turn = 0
        self.fence_every = fence_every
        self.n = 0

PACK = os.path.expanduser("~/models36/packed/qwen3.6-35b-a3b-iq4_xs")
TR = os.path.join(PACK, "trunk")

# ---- P9 cubin programs (built by MM_P9_build.sh / the nvcc container) ----
P9_SPECS = [
    ("MM_P9_gxk3e256up", "gxk3e256up", (1024, 1, 1), False),
    ("MM_P9_gxk4e256dn", "gxk4e256dn", (1024, 1, 1), False),
    ("MM_P9_gv8k4096r2", "gv8k4096r2", (1024, 1, 1), True),
    ("MM_P9_rowcp2048",  "rowcp2048",  (256, 1, 1),  True),
]

# the MTP trunk tensor map: name -> (rows, row_bytes) for Q8_0, or f32 len
_Q8_SHAPES = {
    "attn_q_weight": (8192, 2176), "attn_k_weight": (512, 2176),
    "attn_v_weight": (512, 2176), "attn_output_weight": (2048, 4352),
    "ffn_gate_shexp_weight": (512, 2176), "ffn_up_shexp_weight": (512, 2176),
    "ffn_down_shexp_weight": (2048, 544), "nextn_eh_proj_weight": (2048, 4352),
}
_F32_NAMES = ["attn_norm_weight", "post_attention_norm_weight",
              "attn_q_norm_weight", "attn_k_norm_weight",
              "nextn_enorm_weight", "nextn_hnorm_weight",
              "nextn_shared_head_norm_weight", "ffn_gate_inp_shexp_weight"]


class MtpRig:
    """The MTP layer on a booted Rig7: weights + own KV + controls + the
    SEED / CHAIN-A / CHAIN-B graph runners (P=1 steps; A: H0->H1, B: H1->H0).
    S: the split-S for the MTP attention (default = the decode S)."""

    def __init__(self, rig, S=None):
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_nv import NVProgram
        self.rig = rig
        dev = rig.dev
        CTX = L56.CTX_ALLOC
        self.S = int(S) if S else 32
        rung = f"spka256m_{CTX}", f"spkq256s_{CTX}"
        assert rung[0] in rig.K and rung[1] in rig.K, f"rung {CTX} cubins missing from rig.K"
        # ---- programs ----
        INT_SIG = rig.INT_SIG
        for stem, sym, lsz, scalar in P9_SPECS:
            lib = open(f"{BASE}/{stem}.cubin", "rb").read()
            rig.K[sym] = NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                                                signature=(INT_SIG,) if scalar else tuple()))
            L56.LSZ[sym] = lsz
        # ---- the P=5 probe variants (built on demand; not in CUBIN_SPECS) ----
        for cb, sym, lsz in ((f"{BASE}/MM_P6_gconv36s_5.cubin", "gconv36s_5", (256, 1, 1)),
                             (f"{BASE}/MM_P6_k2s36s_5.cubin", "k2s36s_5", (256, 1, 1)),
                             (f"{BASE}/MM_P6_h6kam_5.cubin", "h6kam_5", (1024, 1, 1))):
            lib = open(cb, "rb").read()
            rig.K[sym] = NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                                                signature=tuple()))
            L56.LSZ[sym] = lsz
        rig.LSZ7 = dict(L56.LSZ)
        # ---- weights ----
        def raw(name):
            return np.fromfile(os.path.join(TR, f"blk_40_{name}.bin"), dtype=np.uint8)
        W = {}
        for name, (rows, rowb) in _Q8_SHAPES.items():
            a = raw(name); assert a.nbytes == rows * rowb, (name, a.nbytes, rows * rowb)
            W[name] = rig.up(a.reshape(rows, rowb))
        for name in _F32_NAMES:
            W[name] = rig.up(raw(name).view(np.float32).copy())
        W["ffn_gate_inp_weight"] = rig.up(raw("ffn_gate_inp_weight").view(np.float32).reshape(256, 2048).copy())
        self.W = W
        # short handles (the seq uses these)
        self.wmap = {
            "enorm": W["nextn_enorm_weight"], "hnorm": W["nextn_hnorm_weight"],
            "shn": W["nextn_shared_head_norm_weight"], "anorm": W["attn_norm_weight"],
            "pnorm": W["post_attention_norm_weight"], "qw": W["attn_q_norm_weight"],
            "kw": W["attn_k_norm_weight"], "eh": W["nextn_eh_proj_weight"],
            "wq": W["attn_q_weight"], "wk": W["attn_k_weight"], "wv": W["attn_v_weight"],
            "wo": W["attn_output_weight"], "rt": W["ffn_gate_inp_weight"],
            "wsh": W["ffn_gate_inp_shexp_weight"], "sg": W["ffn_gate_shexp_weight"],
            "su": W["ffn_up_shexp_weight"], "sd": W["ffn_down_shexp_weight"],
        }
        # ---- the routed L40 bank + pointer tables ----
        rec = rig.man["routed"][40]["files"][0]
        def _getter(off, n):
            with open(os.path.join(PACK, rec["file"]), "rb") as f:
                f.seek(off); return np.frombuffer(f.read(n), dtype=np.uint8)
        bank = rig.up_big(_getter, rec["bytes"])
        self.ptb_up = rig.up(np.array(
            [bank.va_addr + e * rec["slab"] + rec["offsets"]["gate"] for e in range(256)], dtype=np.uint64))
        self.ptb_dn = rig.up(np.array(
            [bank.va_addr + e * rec["slab"] + rec["offsets"]["down"] for e in range(256)], dtype=np.uint64))
        # ---- buffers (P=1) ----
        A = rig.alloc
        self.eb = A(2048 * 4); self.en = A(2048 * 4); self.hn = A(2048 * 4)
        self.xj = A(2048 * 4); self.hnb = A(2048 * 4); self.qgb = A(8192 * 4)
        self.kqb = A(512 * 4); self.vqb = A(512 * 4); self.ayb = A(4096 * 4)
        self.mid = A(2048 * 4); self.outb = A(2048 * 4)
        self.actb = A(8 * 512 * 4); self.partsb = A(8 * 2048 * 2); self.shb = A(2048 * 4)
        self.eidsb = A(8 * 2); self.gatesb = A(8 * 4); self.sgb = A(4)
        self.zero = rig.up(np.zeros(2048, dtype=np.float32))
        self.hseed = A(2048 * 4); self.h0 = A(2048 * 4); self.h1 = A(2048 * 4)
        self.partb = A(7760 * 8)
        # ---- own KV (int8 g128, the trunk layout) + split scratch ----
        self.KVQ = rig.up(np.zeros((2, CTX, 256), dtype=np.int8))
        self.VVQ = rig.up(np.zeros((2, CTX, 256), dtype=np.int8))
        self.KVS = rig.up(np.ones((2, CTX, 2), dtype=np.float32))
        self.VVS = rig.up(np.ones((2, CTX, 2), dtype=np.float32))
        self.scr = A(16 * 1 * self.S * 8 * 258 * 4)
        # ---- cpu-mapped controls (the host-fold pattern) ----
        BS = rig.BufferSpec
        def cpu_map(nbytes):
            b = dev.allocator.alloc(nbytes, BS(cpu_access=True, nolru=True, uncached=True))
            rig.keep.append(b); return b
        self.idsb = cpu_map(4);   self.ids_view = self.idsb.cpu_view().view(size=4, fmt="i")
        self.posb = cpu_map(4);   self.pos_view = self.posb.cpu_view().view(size=4, fmt="i")
        self.amdb = cpu_map(4);   self.am_view = self.amdb.cpu_view().view(size=4, fmt="i")
        self.ids_view[0] = 0; self.pos_view[0] = 0; self.am_view[0] = -1
        # ---- the aux pointer table (9 slots, the spka/spkq contract) ----
        self.sptb = rig.up(np.array([
            self.wmap["kw"].va_addr, self.wmap["qw"].va_addr,
            rig.COSB.va_addr, rig.SINB.va_addr,
            self.KVQ.va_addr, self.KVS.va_addr, self.VVQ.va_addr, self.VVS.va_addr,
            self.posb.va_addr], dtype=np.uint64))
        # ---- the runners (DEDICATED uncached ka -- the pool coherence law) --
        M = self
        def seq(hin, hout, with_head):
            return [(rig.K[n], b, g, v) for n, b, g, v in M._step_seq(hin, hout, with_head)]
        self.r_seed = GraphRunnerUnc(rig, seq(self.hseed, self.h0, False), "mtp_seed")
        self.r_a = GraphRunnerUnc(rig, seq(self.h0, self.h1, True), "mtp_a")
        self.r_b = GraphRunnerUnc(rig, seq(self.h1, self.h0, True), "mtp_b")
        dev.synchronize()
        print(f"[mtp] layer 40 loaded; chain runners built (S={self.S}, ctx {CTX})", flush=True)

    def _step_seq(self, hin, hout, with_head):
        rig = self.rig; w = self.wmap; S = self.S; CTX = L56.CTX_ALLOC
        seq = [
            ("embg248", (rig.EMB, self.idsb, self.eb), 1, (1,)),
            ("rmsz2048g", (self.eb, w["enorm"], self.en), 1, ()),
            ("rmsz2048g", (hin, w["hnorm"], self.hn), 1, ()),
            ("gv8k4096r2", (w["eh"], self.en, self.hn, self.zero, self.xj), 64, (2048,)),
            ("rmsz2048g", (self.xj, w["anorm"], self.hnb), 1, ()),
            ("gv8k2048p", (w["wq"], self.hnb, self.qgb), 256, (8192,)),
            ("gv8k2048p", (w["wk"], self.hnb, self.kqb), 16, (512,)),
            ("gv8k2048p", (w["wv"], self.hnb, self.vqb), 16, (512,)),
            (f"spka256m_{CTX}", (self.kqb, self.vqb, self.sptb), 2, ()),
            (f"spkq256s_{CTX}", (self.qgb, self.scr, self.sptb), (16, S), (S,)),
            ("spkc256", (self.qgb, self.ayb, self.scr), 16, (8 * S,)),
            ("gv8k4096r", (w["wo"], self.ayb, self.xj, self.mid), 64, (2048,)),
            ("rmsz2048g", (self.mid, w["pnorm"], self.hnb), 1, ()),
            ("rt8e256", (w["rt"], w["wsh"], self.hnb, self.eidsb, self.gatesb, self.sgb), 1, ()),
            ("shexp8", (w["sg"], w["su"], w["sd"], self.hnb, self.shb), 1, ()),
            ("gxk3e256up", (self.ptb_up, self.eidsb, self.hnb, self.actb), 8, ()),
            ("gxk4e256dn", (self.ptb_dn, self.eidsb, self.actb, self.partsb), 8, ()),
            ("cmbz2048", (self.partsb, self.gatesb, self.sgb, self.shb, self.mid, self.outb), 1, ()),
            ("rmsz2048g", (self.outb, w["shn"], hout), 1, ()),
        ]
        if with_head:
            seq.append(("h6kam_1", (rig.HEAD, hout, self.partb), 7760, ()))
            seq.append(("amred36", (self.partb, self.amdb), 1, (1,)))
        return seq

    # ---- state helpers ----
    def reset_kv(self, n=1024):
        """Zero the MTP KV prefix (the zero-prefix conditioning class)."""
        dev = self.rig.dev
        if n == "all":
            n = L56.CTX_ALLOC
        zq = np.zeros((2, n, 256), dtype=np.int8).tobytes()
        zs = np.ones((2, n, 2), dtype=np.float32).tobytes()
        for b, data in ((self.KVQ, zq), (self.VVQ, zq), (self.KVS, zs), (self.VVS, zs)):
            dev.allocator._copyin(b, memoryview(data))
        dev.synchronize()

    # ---- serving-side dirty tracking ----
    # run_chain maintains _kv_max = (highest row ever written)+1; rows beyond
    # were never written since alloc (zeros), so a FRESH conversation only
    # needs to zero [0, _kv_max) -- a quote-class conversation costs ~50KB
    # instead of the full 100MB CTX_ALLOC reset.
    _kv_max = -1

    def reset_dirty(self):
        """FRESH conversations: zero exactly the chain-written prefix."""
        if self._kv_max < 0:
            return 0
        n = min(self._kv_max, L56.CTX_ALLOC)
        self.reset_kv(n)
        self._kv_max = -1
        return n


def run_chain(rig, M, seed_row, seed_tok, seed_pos, true_tok, src=None):
    """One MTP chain: seed (src[seed_row], seed_tok) @ seed_pos + 4 head steps
    (step 1 processes the TRUE boundary token). Returns drafts [d1..d4]
    (predictions for seed_pos+2 .. seed_pos+5). src: an [N][2048] f32 hidden
    buffer (default rig.hA; the serving bridge passes rig.PFB["hA"] row 255
    when the last feed was a PF chunk -- the bit-exact chunk-256 class)."""
    src_buf = rig.hA if src is None else src
    rig.K["rowcp2048"](src_buf, M.hseed, global_size=(1, 1, 1),
                       local_size=(256, 1, 1), vals=(int(seed_row),), wait=True)
    M.ids_view[0] = int(seed_tok)
    M.pos_view[0] = int(seed_pos)
    M.r_seed.step()
    tok = int(true_tok)
    drafts = []
    for j, r in enumerate((M.r_a, M.r_b, M.r_a, M.r_b)):
        M.ids_view[0] = tok
        M.pos_view[0] = int(seed_pos) + 1 + j
        r.step()
        tok = int(M.am_view[0])
        drafts.append(tok)
    M._kv_max = max(M._kv_max, int(seed_pos) + 5)
    return drafts


def write_seed_hidden(rig, M, h_vec):
    """Gate helper: place an explicit seed hidden (bypassing the rowcp)."""
    rig.dev.allocator._copyin(M.hseed, memoryview(np.ascontiguousarray(h_vec, dtype=np.float32).data))
    rig.dev.synchronize()


# ============================================================================
# THE SPEC ENGINE (the mtp mode: D8 hit / D2 mid / P5+chain miss)
# ============================================================================
class SpecEngineMTP(SpecEngine7):
    """mode: 't1' | 'spec' (D8/D2/T1, the P8 baseline) | 'mtp' (D8/D2/P5+chain)
    | 'p5lk' (D8 / P5-with-lookup-drafts / T1 -- the P5-graph isolation gate)."""

    def __init__(self, rig, gr1, gr2, gr8, gr5, M, am1=True):
        super().__init__(rig, gr1, gr2, gr8, am1=am1)
        self.gr5 = gr5
        self.M = M

    def _t1_cycle(self, lk, gen, pos):
        rig = self.rig
        tc = time.perf_counter()
        cur = lk.h[-1]
        rig.feed(cur, pos)
        self.gr1.step()
        t2 = self.t1_top()
        self.stats["ms_t1"] += time.perf_counter() - tc
        self.stats["cyc_t1"] += 1
        gen.append(t2); lk.append(t2)
        return t2

    def generate(self, t, lk, pos0, ntok, mode="mtp"):
        """THE STALE-RECOVERY DISCIPLINE: lookup cycles (D8/D2) do NOT run
        the chain (the quote path stays untouched); they mark the chain
        STALE (a deferred chain cannot seed -- hA[m] is overwritten by later
        cycles). The first miss after a lookup streak pays ONE T1 cycle,
        whose hA[0] re-anchors the chain (seed = (hA[0], cur) @ pos-1)."""
        rig = self.rig; M = self.M
        self.reset_stats()
        st = self.stats
        st.update({"cyc_p5": 0, "ms_p5": 0.0, "ms_chain": 0.0, "m5_hist": [],
                   "chain_n": 0, "stale_rec": 0})
        gen = [t]; pos = pos0
        drafts_mtp = None
        if mode == "mtp":
            # the initial chain: seeded from the prompt's last hidden/token
            tc = time.perf_counter()
            drafts_mtp = run_chain(rig, M, 0, lk.h[-2], pos0 - 1, t)
            st["ms_chain"] += time.perf_counter() - tc
            st["chain_n"] += 1
        t0 = time.perf_counter()
        while len(gen) < ntok:
            cur = lk.h[-1]
            n, drafts = (0, [])
            if mode in ("spec", "mtp", "p5lk"):
                n, drafts = lk.scan(kmax=8)
            if n >= 8 and len(drafts) >= 8:
                K = 8; gr = self.gr8
            elif n >= 4 and len(drafts) >= 2 and mode != "p5lk":
                K = 2; gr = self.gr2
            elif mode == "p5lk" and n >= 4 and len(drafts) >= 4:
                K = 4; gr = self.gr5            # the P5 isolation arm
            else:
                K = 0
            if K != 0:
                # ---- lookup cycle (D8 / D2 / p5lk) -- the proven paths ----
                tc = time.perf_counter()
                use = (drafts[:K] + [0] * (8 - K))
                rig.feed9([cur] + use, pos)
                gr.step()
                m = int(rig.eb_view[0]); bnd = int(rig.eb_view[1])
                emitted = list(drafts[:m]) + [bnd]
                if K == 4:
                    st["cyc_p5"] += 1; st["ms_p5"] += time.perf_counter() - tc
                    st["m5_hist"].append(m)
                else:
                    st[f"ms_{'d8' if K==8 else 'd2'}"] += time.perf_counter() - tc
                    st[f"cyc_{'d8' if K==8 else 'd2'}"] += 1
                    if m > 0:
                        st["hits"] += 1; st["msum"] += m; st["m_hist"].append(m)
                gen.extend(emitted)
                for e in emitted: lk.append(e)
                pos += m + 1
                st["cyc"] += 1
                if mode == "mtp":
                    drafts_mtp = None            # STALE (deferred recovery)
                continue
            # ---- K == 0 (miss) ----
            if mode == "mtp":
                if drafts_mtp is None:
                    # STALE RECOVERY: one T1 cycle re-anchors hA[0] = h(pos)
                    self._t1_cycle(lk, gen, pos)
                    pos += 1
                    st["stale_rec"] += 1
                    tc = time.perf_counter()
                    drafts_mtp = run_chain(rig, M, 0, cur, pos - 1, gen[-1])
                    st["ms_chain"] += time.perf_counter() - tc
                    st["chain_n"] += 1
                    continue
                # ---- THE P5+MTP CYCLE ----
                tc = time.perf_counter()
                ids = [cur] + list(drafts_mtp) + [0] * 4
                rig.feed9(ids, pos)
                self.gr5.step()
                m = int(rig.eb_view[0]); bnd = int(rig.eb_view[1])
                emitted = list(drafts_mtp[:m]) + [bnd]
                gen.extend(emitted)
                for e in emitted: lk.append(e)
                pos += m + 1
                st["cyc_p5"] += 1
                st["ms_p5"] += time.perf_counter() - tc
                st["m5_hist"].append(m)
                tc = time.perf_counter()
                drafts_mtp = run_chain(rig, M, m, ids[m], pos - 1, bnd)
                st["ms_chain"] += time.perf_counter() - tc
                st["chain_n"] += 1
                st["cyc"] += 1
                continue
            self._t1_cycle(lk, gen, pos)
            pos += 1
        dt = time.perf_counter() - t0
        self.stats["tok"] = len(gen)
        self.stats["ms"] = dt * 1e3
        return gen
