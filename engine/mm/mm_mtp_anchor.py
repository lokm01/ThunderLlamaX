#!/usr/bin/env python3
"""TLX PART-B — the MTP (blk.40 nextn) numpy anchor + the offline alpha
measurement (mm_mtp_anchor).

THE REFERENCE (llama.cpp qwen35moe graph_mtp, verbatim semantics):
  x   = eh_proj @ concat([enorm(emb(t)), hnorm(h_trunk_normed)])   (e FIRST)
  attn: rmszc(attn_norm) -> q[16x(Q256|gate256)]/k[2x256]/v -> q/k-norm ->
        partial-RoPE(64) -> own-KV append (int8) -> attn -> *sigmoid(gate) ->
        o-proj + residual
  MoE : rmszc(post_norm) -> router fp32 top8 renorm + sigmoid-gated shared
  out : + residual -> shared_head_norm -> the SHARED head (output.weight)

NEW DQ PORTS (numpy, VERBATIM from ggml-quants.c dequantize_row_q3_K/q4_K —
the MTP experts are K-quants, unlike the trunk's IQ formats):
  dq_q3_k / dq_q4_k

ALPHA MEASUREMENT (the Part-B decision number): the trunk anchor's greedy
stream over prose passages; per committed position the MTP chain drafts K=4
(the true-row maintenance protocol); alpha_i = P(draft_i == greedy);
E[m]@K=4 = sum(P(m >= j)) — the offline ceiling for the serving-path speedup.

Usage:
  ~/tg311/bin/python mm_mtp_anchor.py dq        # the dq port self-tests
  ~/tg311/bin/python mm_mtp_anchor.py alpha     # the alpha measurement
"""
import os, sys, json
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")

from MM_P34_ports import (Anchor, fresh_state, rmszc_ref, rope_apply, rope_tables,
                          spka_ref, spkq_h_ref, load_f32, load_q8, kv_quant,
                          GDN_LAYERS, ATTN_LAYERS, _f)
from MM_P2_ports import dq_q8_0, dq_q6_k
from MM_P0_d2_repack import _f16
PACK = os.path.expanduser("~/models36/packed/qwen3.6-35b-a3b-iq4_xs")
CTX = 2048


def silu_np(x):
    return x / (1.0 + np.exp(-np.float64(x))) if False else _f(x / (1.0 + np.exp(np.float64(-x))))


# ===================== the K-quant dq ports (verbatim ggml-quants.c) =========
def _get_scale_min_k4(j, q):
    """q: uint8[12] scales. returns (d, m) uint8 pairs, ggml verbatim."""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def dq_q4_k_np(raw, kdim):
    """raw [n][144]; block: d f16 | dmin f16 | scales[12] | qs[128]. VERBATIM
    dequantize_row_q4_K: 4 x 64-elem groups; lo/hi nibbles per group."""
    n = raw.shape[0]
    nb = kdim >> 8
    out = np.empty((n, kdim), dtype=np.float32)
    raw = raw.reshape(n, nb, 144)
    for i in range(n):
        y = out[i]
        blk = raw[i]
        d = _f16(blk[:, 0:2]).astype(np.float32)
        mn = _f16(blk[:, 2:4]).astype(np.float32)
        q = blk[:, 16:144].reshape(nb, 4, 32)      # qs: 4 x 32 per superblock
        for b in range(nb):
            sc = blk[b, 4:16]
            isv = 0
            for jj in range(4):
                s0, m0 = _get_scale_min_k4(isv + 0, sc)
                d1 = d[b] * s0; m1 = mn[b] * m0
                s1, m1b = _get_scale_min_k4(isv + 1, sc)
                d2 = d[b] * s1; m2 = mn[b] * m1b
                qq = q[b, jj]
                base = b * 256 + jj * 64
                y[base:base + 32] = d1 * (qq[0:32] & 0xF) - m1
                y[base + 32:base + 64] = d2 * (qq[0:32] >> 4) - m2
                isv += 2
    return out


def dq_q3_k_np(raw, kdim):
    """raw [n][110]; block: d f16 | hmask[32] | qs[64] | scales[12]. VERBATIM
    dequantize_row_q3_K: aux-uint32 scale unpack; 2 x 128; 4 shifts x m."""
    n = raw.shape[0]
    nb = kdim >> 8
    out = np.empty((n, kdim), dtype=np.float32)
    raw = raw.reshape(n, nb, 110)
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0f0f0f0f)
    for i in range(n):
        y = out[i]
        blk = raw[i]
        d_all = _f16(blk[:, 108:110]).astype(np.float32)   # d comes LAST in q3_K
        for b in range(nb):
            # block_q3_K = { hmask[32] | qs[64] | scales[12] | d[2] }
            a0, a1, tmp = np.frombuffer(blk[b, 96:108].tobytes(), dtype="<u4")
            n2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
            n3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
            n0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
            n1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
            scales = np.array([n0, n1, n2, n3], dtype=np.uint32).view(np.int8)  # 16
            q = blk[b, 32:96]                   # qs[64]
            hm = blk[b, 0:32]                   # hmask[32]
            m = 1
            isv = 0
            for nn_ in range(2):                # 2 x 128
                shift = 0
                for j in range(4):
                    qseg = q[nn_ * 32: nn_ * 32 + 32]
                    hseg = hm      # the SAME 32 hmask bytes serve both 128-groups
                                   # (m's 8 bits span them: 1,2,4,8 | 16,32,64,128)
                    base = b * 256 + nn_ * 128 + j * 32
                    dl = d_all[b] * (scales[isv] - 32); isv += 1
                    y[base:base + 16] = dl * ((qseg[0:16] >> shift & 3).astype(np.int8) -
                                              np.where(hseg[0:16] & m, 0, 4))
                    dl = d_all[b] * (scales[isv] - 32); isv += 1
                    y[base + 16:base + 32] = dl * ((qseg[16:32] >> shift & 3).astype(np.int8) -
                                                    np.where(hseg[16:32] & m, 0, 4))
                    shift += 2
                    m <<= 1
    return out


# ===================== the MTP layer reference ================================
class MtpLayer:
    """The blk.40 nextn block (numpy, trunk-anchor order)."""

    def __init__(self, ctx=CTX):
        self.ctx = ctx
        self.cos, self.sin = rope_tables(ctx)
        L = 40
        self.eh = dq_q8_0(load_q8(L, "nextn_eh_proj_weight", 2048, 4096), 4096)   # [2048, 4096]
        self.enorm = load_f32(L, "nextn_enorm_weight")
        self.hnorm = load_f32(L, "nextn_hnorm_weight")
        self.shn = load_f32(L, "nextn_shared_head_norm_weight")
        self.anorm = load_f32(L, "attn_norm_weight")
        self.pnorm = load_f32(L, "post_attention_norm_weight")
        self.qw = load_f32(L, "attn_q_norm_weight")
        self.kw = load_f32(L, "attn_k_norm_weight")
        self.wq = dq_q8_0(load_q8(L, "attn_q_weight", 8192), 2048)
        self.wk = dq_q8_0(load_q8(L, "attn_k_weight", 512), 2048)
        self.wv = dq_q8_0(load_q8(L, "attn_v_weight", 512), 2048)
        self.wo = dq_q8_0(load_q8(L, "attn_output_weight", 2048, 4096), 4096)
        self.router = load_f32(L, "ffn_gate_inp_weight").reshape(256, 2048)  # F32
        self.wsh = load_f32(L, "ffn_gate_inp_shexp_weight")
        self.shg = dq_q8_0(load_q8(L, "ffn_gate_shexp_weight", 512), 2048)
        self.shu = dq_q8_0(load_q8(L, "ffn_up_shexp_weight", 512), 2048)
        self.shd = dq_q8_0(load_q8(L, "ffn_down_shexp_weight", 2048, 512), 512)
        self.types = {"gate": "Q3_K", "up": "Q3_K", "down": "Q4_K"}
        self._exp = {}
        # the shared embed/head come from the trunk anchor (lazily shared)
        self._emb_cache = {}

    def embed(self, tid, trunk):
        return trunk.embed_row(int(tid))

    def experts(self, e, trunk_man):
        if e in self._exp:
            return self._exp[e]
        from MM_P2_ports import routed_rows
        meta = trunk_man["routed"][40]
        rec = meta["files"][0]
        rowb = {"gate": 110 * 8, "up": 110 * 8, "down": 144 * 2}[None] if False else None
        bank = np.fromfile(os.path.join(PACK, rec["file"]), dtype=np.uint8)
        slab, expb, moff = rec["slab"], rec["expb"], rec["offsets"]
        def rows(mat, kdim):
            rb = {"gate": 110, "up": 110, "down": 144}[
                None] if False else {"gate": 110 * (2048 // 256), "up": 110 * (2048 // 256),
                                      "down": 144 * (512 // 256)}[mat]
            nrow = {"gate": 512, "up": 512, "down": 2048}[mat]
            return bank[e * slab + moff[mat]: e * slab + moff[mat] + rb * nrow].reshape(nrow, rb)
        Wg = dq_q3_k_np(rows("gate", 2048), 2048)
        Wu = dq_q3_k_np(rows("up", 2048), 2048)
        Wd = dq_q4_k_np(rows("down", 512), 512)
        if len(self._exp) > 64:
            self._exp.clear()
        self._exp[e] = (Wg, Wu, Wd)
        return self._exp[e]

    def forward(self, h_normed, token, pos, kv, trunk, trunk_man):
        """h_normed: the trunk's output_norm hidden at the seed position.
        kv: [(Kq,Ks,Vq,Vs) x 2] fp-quant arrays at [ctx]. Appends row pos.
        Returns (h_nextn, logits)."""
        e = self.embed(token, trunk)
        # THE 2048 norms are ZERO-CENTERED ((1+w); the Anchor's norm_zc carries
        # the engine's op order) — rmszc_ref is the per-head-256 q/k norm ONLY
        e_n = trunk.norm_zc(e, self.enorm)
        h_n = trunk.norm_zc(_f(h_normed), self.hnorm)
        x = np.concatenate([e_n, h_n]).astype(np.float32)
        xj = _f(self.eh @ x)
        # ---- attention ----
        hn = trunk.norm_zc(xj, self.anorm)
        qg = _f(self.wq @ hn)
        kq = _f(self.wk @ hn)
        vq = _f(self.wv @ hn)
        for j in range(2):
            Kq, Ks, Vq, Vs = kv[j]
            spka_ref(_f(kq[j * 256:(j + 1) * 256]), _f(vq[j * 256:(j + 1) * 256]),
                     self.qw, self.kw, self.cos[pos], self.sin[pos], pos, Kq, Ks, Vq, Vs)
        yh = np.empty(4096, dtype=np.float32)
        for hh in range(16):
            j = hh >> 3
            Kq, Ks, Vq, Vs = kv[j]
            yh[hh * 256:(hh + 1) * 256] = spkq_h_ref(qg, self.qw, Kq, Ks, Vq, Vs,
                                                     self.cos[pos], self.sin[pos], pos, hh)
        mid = _f(xj + _f(yh @ self.wo.T))
        # ---- MoE ----
        hn2 = trunk.norm_zc(mid, self.pnorm)
        lg = _f(self.router @ hn2)
        l = lg.copy(); ids = []; ex = []
        for r in range(8):
            be = int(np.argmax(l)); ids.append(be)
            ex.append(np.exp(np.float64(l[be] - lg.max()))); l[be] = np.float32(-3.4e38)
        gates = _f(np.asarray(ex, dtype=np.float64) / np.sum(ex)).astype(np.float32)
        sg = _f(_f(1.0) / (_f(1.0) + np.exp(-np.float64(self.wsh @ hn2))))
        y = np.zeros(2048, dtype=np.float32)
        for r in range(8):
            Wg, Wu, Wd = self.experts(ids[r], trunk_man)
            t = _f(silu_np(_f(Wg @ hn2)) * _f(Wu @ hn2))
            y = _f(y + _f(gates[r] * _f(Wd @ t)))
        shg = silu_np(_f(self.shg @ hn2)); shu = _f(self.shu @ hn2)
        sh = _f(self.shd @ _f(shg * shu))
        out = _f(mid + _f(y + _f(sg * sh)))
        h_nextn = trunk.norm_zc(out, self.shn)
        logits = trunk.head_dot(h_nextn)
        return h_nextn, logits


# ===================== the alpha measurement ==================================
def mtp_kv(ctx=CTX):
    lay = []
    for j in range(2):
        Kq = np.zeros((ctx, 256), dtype=np.int8); Ks = np.ones((ctx, 2), dtype=np.float32)
        Vq = np.zeros((ctx, 256), dtype=np.int8); Vs = np.ones((ctx, 2), dtype=np.float32)
        lay.append((Kq, Ks, Vq, Vs))
    return lay


def run_alpha(passages=None, kmax=4, ntok=24):
    """THE ALPHA MEASUREMENT (the Part-B decision number).

    Protocol (the llama.cpp graph_mtp semantics; the serving chain's offline
    ceiling): the trunk anchor runs the TRUE greedy stream; at each committed
    base position p the MTP chain seeds with (h_trunk(p), t_p) at position p
    (writing its own KV row) and drafts K tokens; m(p) = the longest prefix
    matching the truth stream. alpha_j = P(d_j correct | d_1..d_{j-1} correct);
    E[m]@K = mean accepted drafts + 1 boundary token per cycle.

    KV maintenance: each measurement chain's SEED row p is the true pair
    (idempotent); rows < p were written true by prior seeds; draft rows > p
    are overwritten by later seeds before being read. Same shape as serving.
    """
    from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
    tok = SimpleTokenizer.from_gguf_kv(parse_gguf_kv(os.path.expanduser(
        "~/models36/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")))
    if passages is None:
        # passage 0 = the QUOTE wiring check (the alphabet repeat: alpha MUST
        # be ~1.0 — a low value there = an anchor wiring bug, not model truth)
        passages = [
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A B C D E F G H I J K L M N O P Q",
            "The old lighthouse keeper had seen many storms in his forty years, but nothing",
            "In economics, inflation is best understood not as prices rising but as",
            "She opened the letter slowly, knowing that whatever was written inside would",
        ]
    anc = Anchor(ctx=CTX, fp16_partials=True)
    # THE 16GB RAM WALL: run beside the LIVE daemon — cap the expert LRU
    # (384 x 12.6MB = 4.8GB thrashes with the daemon resident; 96 = 1.2GB)
    import MM_P34_ports as _P34
    _P34._AN_LRU = None  # marker; the monkeypatch below is the real cap
    class _AnchorCap(Anchor):
        def _experts(self, L, ids):
            out = Anchor._experts(self, L, ids)
            return out
    # simplest real cap: patch the eviction threshold on the instance path
    orig_exp = anc._experts
    def capped_experts(L, ids):
        out = orig_exp(L, ids)
        while len(anc._exp) > 96:
            k = anc._exp_order.pop(0); del anc._exp[k]
        return out
    anc._experts = capped_experts
    man = json.load(open(os.path.join(PACK, "manifest.json")))
    mtp = MtpLayer(ctx=CTX)
    tot_n = 0
    tot_acc = [0] * kmax           # accepted-at-depth-j counts (j-1 prior accepts)
    tot_m = 0.0
    for pi, ptxt in enumerate(passages):
        ids = tok.encode(ptxt)
        S, CS, KV = fresh_state(CTX)
        hs, stream = [], []
        h = None
        for pos, t in enumerate(ids):
            top1, _, h = anc.forward_token(int(t), pos, S, CS, KV)
            hs.append(h.copy()); stream.append(int(t))
        for g in range(ntok + kmax):
            top1, _, h = anc.forward_token(int(stream[-1]), len(stream) - 1, S, CS, KV)
            hs.append(h.copy()); stream.append(int(top1))
        chain_kv = mtp_kv()
        n = 0; acc = [0] * kmax; msum = 0.0
        # bases p: from len(ids)-1 (the first prefill boundary) while p+kmax < len(stream)
        for p in range(len(ids) - 1, len(stream) - kmax - 1):
            hh, lg = mtp.forward(hs[p], stream[p], p, chain_kv, anc, man)
            drafts = [int(np.argmax(lg))]
            for j in range(1, kmax):
                hh, lg = mtp.forward(hh, drafts[-1], p + j, chain_kv, anc, man)
                drafts.append(int(np.argmax(lg)))
            m = 0
            while m < kmax and drafts[m] == stream[p + 1 + m]:
                acc[m] += 1
                m += 1
            msum += m
            n += 1
            if n % 8 == 0:
                print(f"    [p{pi}] {n} bases: running E[acc] {msum/n:.2f} "
                      f"a1 {acc[0]/n:.2f}", flush=True)
        tot_n += n
        for j in range(kmax):
            tot_acc[j] += acc[j]
        em = msum / max(1, n)
        print(f"  passage {pi} ({n} bases): E[accepted]@K{kmax} = {em:.2f} "
              f"-> E[tok/cycle] = {em + 1:.2f} | cond-alpha = "
              f"{[round(acc[j] / max(1, acc[j - 1] if j else n), 3) for j in range(kmax)]}", flush=True)
    em = sum(tot_acc) / max(1, tot_n)
    print(f"ALPHA MEASUREMENT: {tot_n} bases | P(accept1) = {tot_acc[0]/max(1,tot_n):.3f} | "
          f"E[accepted]@K4 = {em:.2f} | E[tokens/cycle] = {em + 1:.2f} "
          f"(the 39-53 tok/s target needs E[tok/cycle] 2.2-3.1 at ~57ms/cycle)", flush=True)


def run_dq_tests():
    rng = np.random.default_rng(11)
    # Q4_K synthetic: build a block from known d/dmin/scales/qs, dequant, compare
    nb = 4
    blk = np.zeros((1, nb, 144), dtype=np.uint8)
    ref = np.zeros((nb, 256), dtype=np.float32)
    for b in range(nb):
        d0 = rng.uniform(0.01, 0.05); mn0 = rng.uniform(-0.02, 0.02)
        blk[0, b, 0:2] = np.frombuffer(np.float16(d0).tobytes(), dtype=np.uint8)
        blk[0, b, 2:4] = np.frombuffer(np.float16(mn0).tobytes(), dtype=np.uint8)
        d = float(np.float16(d0)); mn = float(np.float16(mn0))   # the format's precision
        sc = rng.integers(0, 64, 12).astype(np.uint8)
        blk[0, b, 4:16] = sc
        qs = rng.integers(0, 256, 128).astype(np.uint8)
        blk[0, b, 16:144] = qs
        for jj in range(4):
            s0, m0 = _get_scale_min_k4(2 * jj, sc)
            s1, m1b = _get_scale_min_k4(2 * jj + 1, sc)
            for l in range(32):
                ref[b, jj * 64 + l] = d * s0 * (qs[jj * 32 + l] & 0xF) - mn * m0
                ref[b, jj * 64 + 32 + l] = d * s1 * (qs[jj * 32 + l] >> 4) - mn * m1b
    got = dq_q4_k_np(blk.reshape(1, -1), 256 * nb)
    nz = int((got[0].reshape(nb, 256) != ref).sum())
    print(f"dq_q4_k synthetic roundtrip: {'BIT-EXACT' if nz == 0 else f'DIFF nz={nz}'}")
    # real-weight plausibility: L40 down rows
    from MM_P2_ports import routed_rows
    man = json.load(open(os.path.join(PACK, "manifest.json")))
    rec = man["routed"][40]["files"][0]
    bank = np.fromfile(os.path.join(PACK, rec["file"]), dtype=np.uint8)
    slab, moff = rec["slab"], rec["offsets"]
    rb = 144 * (512 // 256)
    rows = bank[0 * slab + moff["down"]: 0 * slab + moff["down"] + rb * 2048].reshape(2048, rb)
    w = dq_q4_k_np(rows[:64], 512)
    print(f"dq_q4_k real L40 down e0: shape {w.shape} std {w.std():.4f} finite {np.isfinite(w).all()}")
    rb3 = 110 * (2048 // 256)
    rowsg = bank[moff["gate"]: moff["gate"] + rb3 * 512].reshape(512, rb3)
    wg = dq_q3_k_np(rowsg[:64], 2048)
    print(f"dq_q3_k real L40 gate e0: shape {wg.shape} std {wg.std():.4f} finite {np.isfinite(wg).all()}")
    # expert weight stats vs the trunk's IQ3_S experts (sanity: same magnitude class)
    rg3 = routed_rows(3, "gate", [0])
    from MM_P2_ports import dq_iq3_s
    w3 = dq_iq3_s(rg3, 2048)
    print(f"   trunk L3 IQ3_S gate e0 std {w3.std():.4f} (the magnitude class reference)")


def mtp_smoke():
    """A 5-second MtpLayer.forward shape/finite check on synthetic inputs
    (no trunk phase) — catches wiring bugs before the expensive runs."""
    import numpy as np
    mtp = MtpLayer(ctx=256)
    class _T:  # the trunk stub: just the pieces MtpLayer touches
        pass
    anc = Anchor.__new__(Anchor)          # no heavy init
    anc.ctx = 256
    from MM_P34_ports import rope_tables as _rt
    anc.cos, anc.sin = _rt(256)
    W_head = np.zeros((16, 2048), dtype=np.float32); W_head[3, 0] = 1.0
    anc.head_dot = lambda h: W_head @ h
    anc.embed_row = lambda tid: np.full(2048, 0.01 * ((int(tid) % 7) + 1), dtype=np.float32)
    anc.norm_zc = Anchor.norm_zc.__get__(anc)
    man = json.load(open(os.path.join(PACK, "manifest.json")))
    kv = mtp_kv(256)
    h = np.full(2048, 0.02, dtype=np.float32)
    hh, lg = mtp.forward(h, 5, 10, kv, anc, man)
    print(f"mtp_smoke: h_nextn finite {np.isfinite(hh).all()} |logits| {np.abs(lg).max():.3f} "
          f"argmax {int(np.argmax(lg))} | row10 written "
          f"{any(np.any(kv[j][0][10] != 0) for j in range(2))}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dq"
    if mode == "dq":
        run_dq_tests()
    elif mode == "smoke":
        mtp_smoke()
    elif mode == "alpha":
        run_alpha()
