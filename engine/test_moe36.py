#!/usr/bin/env python3
"""TLX P8: the Qwen3.6-35B-A3B (MoE) daemon HOST — the serving bridge.

The MM P5-P7 harness (MM_P7_lib: Rig7 / build_seq7 / GraphRunner) is
Tier-1-gated in-harness (60/60 battery, 119.6 tok/s quote-class, split-KV
96k green, chunk-256 prefill bit-exact). THIS host adapts it to the
serve.py socket contract (prefill/generate/status/cancel via serve_moe)
so the MoE serves through api_server.py exactly like the dense engine.

Boot shape (the HOST-PROCESS law): launchd runs the wrapper, the wrapper
runs `python -u test_moe36.py` under the qwen3.6-35b-a3b-egpu env; the rig
boots (weights ~15GB + cubin builds on first run), the four graph pairs
build (T1-am / D2 / D8 / PF-256, all split-KV rung 98304, the proven
classes: decode S=32 + PF S=8), then serve_moe.run_daemon_moe attaches.

Env (ops/env.canonical.d/qwen3.6-35b-a3b-egpu.env):
  MM_PACKED    the packed weights dir (the repack manifest feeds config_fp)
  PC_ENABLED/PC_ROOT/PC_QUOTA_GB   the per-model prompt cache
  MM_SPEC=0    pure-T1 daemon (the spec gate's T1 arm / a kill switch)
  MM_MTP=0     kill switch for the MTP K=4 chain (P10; default ON — the
               miss path becomes P5+MTP instead of T1: prose 19.6 -> 42-47)
  MM_DEC_S/MM_PF_S  split-S classes (defaults 32/8 — the gated configs)
"""
import os, sys

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
os.environ.setdefault("DEV", "NV")

CTXK = int(os.getenv("MM_CTXS", "98304"))
DEC_S = int(os.getenv("MM_DEC_S", "32"))
PF_S = int(os.getenv("MM_PF_S", "8"))
RUNG = CTXK            # the split cubin rung covers every L <= CTXK
MTP_ON = os.getenv("MM_MTP", "1") == "1"   # P10: the MTP K=4 chain (prose lever)


class MoeServeEngine:
    """The facade serve_moe drives: rig + graphs + the host-fold primitives."""

    def __init__(self):
        import numpy as np
        from MM_P7_lib import Rig7, build_seq7, mkgraph
        self.np = np
        print(f"[moe36] booting Rig7 (ctx {CTXK}, split rung {RUNG})...", flush=True)
        self.rig = rig = Rig7(ctx_alloc=CTXK, load_p6=True)
        print("[moe36] rig up; building graphs (t1-am/d2/d8/pf-256)...", flush=True)
        spk = f"s{RUNG}"
        seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True,
                          head_mode="am", spk=spk, S=DEC_S)
        self.gr1 = mkgraph(rig, seq1, "sv_t1")
        seq2 = build_seq7(rig, 3, "gconv36s_3", "k2s36s_3", with_head=True,
                          head_mode="am", slots=True, spk=spk, S=DEC_S, tail_acc_K=2)
        self.gr2 = mkgraph(rig, seq2, "sv_d2")
        seq8 = build_seq7(rig, 9, "gconv36s_9", "k2s36s_9", with_head=True,
                          head_mode="am", slots=True, spk=spk, S=DEC_S, tail_acc_K=8)
        self.gr8 = mkgraph(rig, seq8, "sv_d8")
        seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False,
                           spk=spk, S=PF_S, pf=True)
        self.gr_pf = mkgraph(rig, seqpf, "sv_pf")
        self.dev = rig.dev
        # ---- P10: THE MTP K=4 CHAIN (MM_P9_mtp.MtpRig + the P=5 probe) ----
        # All late-built graphs use GraphRunnerUnc (dedicated uncached ka --
        # the KAPool slab coherence law). fence_every=1024 matches the trunk
        # runners' dext-budget discipline.
        self.M = None; self.gr5 = None; self.mtp_on = False
        if MTP_ON:
            from MM_P9_mtp import MtpRig, GraphRunnerUnc
            self.M = MtpRig(rig, S=DEC_S)
            for r in (self.M.r_seed, self.M.r_a, self.M.r_b):
                r.fence_every = 1024
            seq5 = build_seq7(rig, 5, "gconv36s_5", "k2s36s_5", with_head=True,
                              head_mode="am", slots=True, spk=spk, S=DEC_S, tail_acc_K=4)
            self.gr5 = GraphRunnerUnc(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq5],
                                      "sv_p5", fence_every=1024)
            self.mtp_on = True
            print("[moe36] MTP layer + P5 probe graph built (the mtp mode is live)", flush=True)
        print("[moe36] graphs built; the engine facade is live", flush=True)

    # ---- control-plane helpers (the host fold: cpu_view reads/writes) ----
    def feed(self, tid, pos):
        self.rig.feed(int(tid), int(pos))

    def feed_step(self, tid, pos):
        """One T1 cycle: control write + graph + am read."""
        self.rig.feed(int(tid), int(pos))
        self.gr1.step()
        return int(self.rig.am_view[0])

    def feed9(self, ids, pos):
        self.rig.feed9(ids, int(pos))

    def pf_feed_chunk(self, chunk, pos0):
        """One 256-chunk PF feed (the chunk graph reads pf_idsb + POSB)."""
        assert len(chunk) == 256, f"pf chunk must be 256 (got {len(chunk)})"
        self.rig.pf_ids_view[:] = memoryview(
            self.np.ascontiguousarray(self.np.asarray(chunk, dtype=self.np.int32)).data)
        self.rig.pos_view[0] = int(pos0)
        self.gr_pf.step()

    def eager_head_cur(self, seat=None, pf=False):
        """Top-1 after the boundary hidden: the eager rmsz+h6k pair. seat+pf:
        a PFB seat of the last PF chunk (seat 255 = the chunk's last token).
        No args: rig.hA row 0 — valid right after a P=1 (T1/tail) step."""
        rig = self.rig
        np = self.np
        if pf:
            hin = rig.PFB["hA"].offset(offset=int(seat) * 2048 * 4, size=2048 * 4)
        else:
            hin = rig.hA
        rig.K["rmsz2048g"](hin, rig.ONORM, rig.normhb,
                           global_size=(1, 1, 1), local_size=(256, 1, 1), wait=True)
        rig.K["h6k2048"](rig.HEAD, rig.normhb, rig.logitsb,
                         global_size=(7760, 1, 1), local_size=(1024, 1, 1),
                         vals=(248320,), wait=True)
        return int(np.argmax(rig.dn(rig.logitsb, (248320,))))

    def keepalive_probe(self):
        """Tiny eager probe (NOT a graph): host-write am -> acc36 K=0 -> the
        cpu-mapped eb read must echo (0, sentinel). Exercises the GPU, the
        uncached control path, and the timeline wait — the health-probe
        pattern (a silent zombie must die LOUD, not answer /health)."""
        rig = self.rig
        SENT = 1234567
        rig.am_view[0] = SENT
        dv = rig.idsb.offset(offset=4, size=8 * 4)
        rig.K["acc36"](rig.AMDB, dv, rig.MB, rig.EB,
                       global_size=(1, 1, 1), local_size=(256, 1, 1), vals=(0,), wait=True)
        m, b = int(rig.eb_view[0]), int(rig.eb_view[1])
        assert (m, b) == (0, SENT), f"keepalive probe mismatch: ({m},{b}) != (0,{SENT})"

    # ---- P10: the MTP chain surface (serve_moe drives these) ----
    def mtp_run_chain(self, seed_src, seed_row, seed_tok, seed_pos, true_tok):
        """One chain from a hidden source: 'hA' = rig.hA[seed_row] (a decode
        graph's seat hidden), 'pf' = PFB['hA'][seed_row] (a PF chunk seat —
        the bit-exact chunk-256 class, seat 255 = the chunk's last token)."""
        from MM_P9_mtp import run_chain
        src = self.rig.hA if seed_src == "hA" else self.rig.PFB["hA"]
        return run_chain(self.rig, self.M, int(seed_row), int(seed_tok),
                         int(seed_pos), int(true_tok), src=src)

    def mtp_reset_kv(self):
        """FRESH conversations: zero exactly the chain-written MTP-KV prefix."""
        return self.M.reset_dirty()

    def fence_all(self):
        """The ~950-cycle budget reset: rebuild every runner's kernargs at a
        quiescent point (the E.build_graphs() equivalent). P10: the P5 probe
        + the three MTP chain runners fence WITH the trunk set."""
        runners = [self.gr1, self.gr2, self.gr8, self.gr_pf]
        if self.mtp_on:
            runners += [self.gr5, self.M.r_seed, self.M.r_a, self.M.r_b]
        for gr in runners:
            gr.fence()


def main():
    eng = MoeServeEngine()
    import serve_moe
    serve_moe.run_daemon_moe(eng, CTXK)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
