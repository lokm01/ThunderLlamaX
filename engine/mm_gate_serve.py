#!/usr/bin/env python3
"""TLX P8 MoE SERVING GATES (mm_gate_serve) — the conformance battery against
the LIVE MoE daemon (test_moe36.py + serve_moe on /tmp/llm-engine.sock).

  G1  boot/status shape
  G2  THE TIER-1 BANK THROUGH THE DAEMON: the mm_p5 bank60 prompts, spec vs
      t1 per prompt (32-tok gens, prefix-exact) + det-x2 spots
  G3  spec == t1 at depth (doc100k 4k PF-chunk feed, 32-tok gens)
  G4  pcache: FRESH ingest -> AUTO_CACHE hit (cached_tokens) -> the restored
      continuation == the FRESH continuation, bit-level state agreement
  G5  quote-class tok/s through the daemon (expect ~110-120)
  G6  prose tok/s through the daemon (expect ~19)
  G7  long-cycle fence sanity (cyc counters in status)
  G8  P10 THE MISSION GATE: bank60 with the MTP mode ON (mtp==t1 + det-x2)
  G9  P10 perf: quote/prose classes through the DEFAULT (mtp) mode
  G10 P10 the PF-only cur regression (fresh cur == pcache node cur)

Usage (the daemon must be UP): ~/tg311/bin/python mm_gate_serve.py G1,G2,...
"""
import os, sys, time, json, socket
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
SOCK = "/tmp/llm-engine.sock"
A60 = os.path.expanduser("~/mm_p5_anchor60.npz")
DOC = os.path.expanduser("~/mm_p5_doc100k_ids.npy")
PROG = os.getenv("MM_GATE_PROG", os.path.expanduser("~/mm_p8s_progress.txt"))
NTOK = 32


def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)


def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())


class Eng:
    def __init__(self, timeout=1200.0):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(SOCK)
        self.buf = b""; self.n = 0
    def send(self, method, params):
        self.n += 1
        self.s.sendall((json.dumps({"id": self.n, "method": method, "params": params}) + "\n").encode())
        return self.n
    def recv(self):
        while b"\n" not in self.buf:
            ch = self.s.recv(1 << 20)
            if not ch: raise RuntimeError("engine closed")
            self.buf += ch
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)
    def rpc(self, method, params):
        rid = self.send(method, params)
        while True:
            r = self.recv()
            if r.get("id") == rid and "event" not in r:
                if not r.get("ok"):
                    raise RuntimeError(f"{method}: {r.get('error')}")
                return r["result"]
    def prefill(self, ids, mode="FRESH", cid=None, **kw):
        p = {"mode": mode, "ids": [int(x) for x in ids]}
        if cid: p["conversation_id"] = cid
        p.update(kw)
        return self.rpc("prefill", p)
    def generate(self, mc=NTOK, stops=(), force=None, collect=False):
        p = {"max_cycles": mc, "stop_token_ids": list(stops)}
        if force: p["force_mode"] = force
        rid = self.send("generate", p)
        toks = []; evs = 0; t0 = time.perf_counter()
        while True:
            r = self.recv()
            if r.get("id") != rid:
                continue
            if r.get("event") == "cycle":
                evs += 1; toks += r.get("tokens", [])
            elif r.get("event") in ("done", "cancelled"):
                dt = time.perf_counter() - t0
                return toks, r, dt
            elif r.get("event") == "error":
                raise RuntimeError(f"generate error event: {r.get('error')}")
            elif "event" not in r:
                if not r.get("ok"):
                    raise RuntimeError(f"generate: {r.get('error')}")
                return toks, r, time.perf_counter() - t0
    def close(self):
        try: self.s.close()
        except Exception: pass


def g1():
    e = Eng()
    st = e.rpc("status", {})
    ok = (st.get("ready") is True and st.get("model_id") == "qwen3.6-35b-a3b-egpu"
          and st.get("ctxk") == 98304 and st.get("config_fp"))
    record("G1", f"status ready={st.get('ready')} model={st.get('model_id')} ctxk={st.get('ctxk')} "
                 f"fp={str(st.get('config_fp'))[:16]} pc={bool(st.get('pc'))} "
                 f"spec={st.get('spec')} {'OK' if ok else 'FAIL'}")
    e.close()


def g2():
    a = np.load(A60, allow_pickle=True)
    ids60 = [np.asarray(x, dtype=np.int32) for x in a["ids"]]
    e = Eng()
    mism = []
    for pi, ids in enumerate(ids60):
        e.prefill(ids, cid=f"b-t1-{pi}")
        t_t1, _, _ = e.generate(force="t1")
        e.prefill(ids, cid=f"b-sp-{pi}")
        t_sp, _, _ = e.generate(force="spec")
        if t_t1 != t_sp[:len(t_t1)]:
            bad = next(i for i, (x, y) in enumerate(zip(t_t1, t_sp)) if x != y) if t_sp[:len(t_t1)] != t_t1 else -1
            mism.append((pi, bad))
        if pi % 10 == 0:
            print(f"    [G2] {pi}: {'EXACT' if not mism or mism[-1][0] != pi else 'MISMATCH'}", flush=True)
    det = True
    for pi in range(0, 60, 15):
        e.prefill(ids60[pi], cid=f"b-det-{pi}")
        g1_, _, _ = e.generate(force="spec")
        e.prefill(ids60[pi], cid=f"b-det2-{pi}")
        g2_, _, _ = e.generate(force="spec")
        det &= (g1_ == g2_)
    record("G2", f"TIER-1 BANK60 THROUGH THE DAEMON: {60 - len(mism)}/60 spec==t1 prefix-exact, "
                 f"det-x2 {'OK' if det else 'FAIL'} mism={mism[:6]}")
    e.close()


def g3():
    doc = np.load(DOC).astype(np.int32)
    N = 4096
    e = Eng()
    e.prefill(doc[:N], cid="g3a")
    t_t1, _, _ = e.generate(force="t1")
    e.prefill(doc[:N], cid="g3b")
    t_sp, _, _ = e.generate(force="spec")
    ok = t_t1 == t_sp[:len(t_t1)]
    det = True
    e.prefill(doc[:N], cid="g3c")
    g1_, _, _ = e.generate(force="spec")
    e.prefill(doc[:N], cid="g3d")
    g2_, _, _ = e.generate(force="spec")
    det = (g1_ == g2_)
    record("G3", f"spec==t1 @4k-ctx (PF-chunk feed): {'EXACT' if ok else 'MISMATCH'} "
                 f"first16={t_t1[:8]} det-x2 {'OK' if det else 'FAIL'}")
    e.close()


def g4():
    rng = np.random.default_rng(7)
    base = [int(x) for x in rng.integers(0, 248320, size=2048)]
    ext = [int(x) for x in rng.integers(0, 248320, size=64)]
    full = base + ext
    e = Eng()
    e.prefill(full, cid="g4f")            # ingest at 1024, 2048
    t_fr, _, _ = e.generate(force="t1")
    st = e.rpc("status", {})
    pc = st.get("pc") or {}
    r = e.prefill(full, mode="AUTO_CACHE", cid="g4c")
    hit = r.get("mode") == "CACHE_HIT" and r.get("cached_tokens") == 2048
    t_hi, _, _ = e.generate(force="t1")
    ok = t_hi == t_fr
    record("G4", f"PCACHE: nodes={pc.get('entries')} cached_tokens={r.get('cached_tokens')} "
                 f"hit={'OK' if hit else 'FAIL'} continuation {'EXACT' if ok else 'MISMATCH vs FRESH'} "
                 f"(fresh {str(t_fr)[:24]} hit {str(t_hi)[:24]})")
    e.close()


def g56():
    from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
    tok = SimpleTokenizer.from_gguf_kv(parse_gguf_kv(os.path.expanduser(
        "~/models36/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")))
    from MM_P34_anchor import PROMPTS as P20
    doc_txt = open(os.path.expanduser("~/prompt100k.txt"), encoding="utf-8", errors="replace").read()
    code = ("def process(items):\n    out = []\n    for it in items:\n        if it is not None:\n"
            "        out.append(it.strip())\n    return out\n")
    batteries = [
        ("quote-alpha", P20[7], 48),
        ("quote-code", f"{code}\nThe same function again:\ndef process(items):", 48),
        ("prose-0", P20[0], 48),
        ("prose-9", P20[9], 48),
    ]
    e = Eng()
    rows = []
    for tag, txt, n in batteries:
        ids = tok.encode(txt)
        e.prefill(ids, cid=f"p-{tag}")
        toks, done, dt = e.generate(mc=n + 8)
        ntok = len(toks)
        rows.append(f"{tag}: {ntok / dt:.1f} tok/s ({ntok} toks / {dt:.1f}s)")
        print(f"    [G56] {rows[-1]}", flush=True)
    record("G5G6", " | ".join(rows))
    e.close()


def g7():
    e = Eng()
    st = e.rpc("status", {})
    record("G7", f"fence sanity: cycles_since_rebuild={st.get('cycles_since_rebuild')} "
                 f"pc={st.get('pc')} spec={st.get('spec')}")
    e.close()


# ---- P10 (the MTP wire): the mission gates -----------------------------------

def g8():
    """THE P10 MISSION GATE: the Tier-1 bank with the MTP mode ON —
    force='mtp' (D8 hit / D2 mid / P5+chain miss / T1 stale-recovery) vs
    force='t1', prefix-exact, + det-x2 spots."""
    a = np.load(A60, allow_pickle=True)
    ids60 = [np.asarray(x, dtype=np.int32) for x in a["ids"]]
    e = Eng()
    mism = []
    for pi, ids in enumerate(ids60):
        e.prefill(ids, cid=f"m-t1-{pi}")
        t_t1, _, _ = e.generate(force="t1")
        e.prefill(ids, cid=f"m-mt-{pi}")
        t_mt, _, _ = e.generate(force="mtp")
        if t_t1 != t_mt[:len(t_t1)]:
            bad = next(i for i, (x, y) in enumerate(zip(t_t1, t_mt)) if x != y) if t_mt[:len(t_t1)] != t_t1 else -1
            mism.append((pi, bad))
        if pi % 10 == 0:
            print(f"    [G8] {pi}: {'EXACT' if not mism or mism[-1][0] != pi else 'MISMATCH'}", flush=True)
    det = True
    for pi in range(0, 60, 15):
        e.prefill(ids60[pi], cid=f"m-det-{pi}")
        g1_, _, _ = e.generate(force="mtp")
        e.prefill(ids60[pi], cid=f"m-det2-{pi}")
        g2_, _, _ = e.generate(force="mtp")
        det &= (g1_ == g2_)
    record("G8", f"P10 TIER-1 BANK60 MTP-ON THROUGH THE DAEMON: {60 - len(mism)}/60 mtp==t1 "
                 f"prefix-exact, det-x2 {'OK' if det else 'FAIL'} mism={mism[:6]}")
    e.close()


def g9():
    """The P10 perf battery: the DEFAULT mode (MTP) through the daemon vs the
    P8S G5G6 numbers (quote-alpha 98.1 / quote-code 96.3 / prose 19.1)."""
    from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
    tok = SimpleTokenizer.from_gguf_kv(parse_gguf_kv(os.path.expanduser(
        "~/models36/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")))
    from MM_P34_anchor import PROMPTS as P20
    doc_txt = open(os.path.expanduser("~/prompt100k.txt"), encoding="utf-8", errors="replace").read()
    passage = doc_txt[3000:3900]; half = passage[:len(passage)//2]
    qp_docx2 = f"Here is a passage:\n{passage}\nNow repeat the passage exactly, word for word:\n{half}"
    code = ("def process(items):\n    out = []\n    for it in items:\n        if it is not None:\n"
            "            out.append(it.strip())\n    return out\n")
    batteries = [
        ("quote-alpha", P20[7], 48),
        ("quote-code", f"{code}\nThe same function again:\ndef process(items):", 48),
        ("quote-docx2", qp_docx2, 48),
        ("prose-0", P20[0], 48),
        ("prose-1", P20[1], 48),
        ("prose-9", P20[9], 48),
    ]
    e = Eng()
    rows = []
    for tag, txt, n in batteries:
        ids = tok.encode(txt)
        e.prefill(ids, cid=f"m9-{tag}")
        toks, done, dt = e.generate(mc=n + 8)
        ntok = len(toks)
        rows.append(f"{tag}: {ntok / dt:.1f} tok/s ({ntok} toks / {dt:.1f}s)")
        print(f"    [G9] {rows[-1]}", flush=True)
    st = e.rpc("status", {})
    record("G9", "P10 PERF (default=mtp) | " + " | ".join(rows) + f" | spec={st.get('spec')}")
    e.close()


def g10():
    """The PF-ONLY CUR regression (the P10 latent-bug fix): a 1024-token
    PF-only feed must derive cur from PFB seat 255. Truth = the pcache node
    meta cur (the ingest path always used the seat-based eager head), read
    back through a full AUTO_CACHE hit."""
    rng = np.random.default_rng(11)
    toks = [int(x) for x in rng.integers(1000, 200000, size=1024)]
    e = Eng()
    r1 = e.prefill(toks, cid="g10a")            # 4 PF chunks, ingest at 1024
    t_fr, _, _ = e.generate(mc=4)
    r2 = e.prefill(toks, mode="AUTO_CACHE", cid="g10b")
    hit = r2.get("mode") == "CACHE_HIT" and r2.get("cached_tokens") == 1024
    ok = hit and r1["cur"] == r2["cur"]
    record("G10", f"PF-ONLY CUR: fresh cur={r1['cur']} node cur={r2.get('cur')} "
                  f"hit={'OK' if hit else 'FAIL'} cur {'MATCH (fix live)' if r1['cur'] == r2.get('cur') else 'MISMATCH (BUG)'} "
                  f"{'OK' if ok else 'FAIL'}")
    e.close()


def main():
    stages = os.environ.get("GATES", "G1,G2,G3,G4,G56,G7").split(",")
    fns = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G56": g56, "G7": g7,
           "G8": g8, "G9": g9, "G10": g10}
    for s in stages:
        if s in fns and os.environ.get("FORCE", "0") != "1" and done(s):
            print(f"[skip] {s} (done)"); continue
        fns[s]()
    print("[GATES DONE]", flush=True)


if __name__ == "__main__":
    main()
