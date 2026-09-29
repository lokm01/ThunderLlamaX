#!/usr/bin/env python3
"""P9 EVAL — teacher-forced NLL/perplexity on the DENSE rig (standalone).

Mirrors test_w100k.py's daemon boot EXACTLY through build_graphs (snapshot
load, T1 ref decode, slice+init_draft, fill_draft — the proven order), then
scores each corpus domain via pcache.fresh_prefill(ingest=scorer):

  ingest(pos_after, final64) fires at every 128-chunk quiescent boundary
  (PF_M128 trunk; on_chunk -> ingest in pcache.fresh_prefill). The chunk's
  trunk hiddens sit in E._pf_last128 (xA128/xB128 per the 64-blocks-even law;
  row r = token pos_after-128+r, pre-final-norm). Per row we run the proven
  final-head pair from pf_prefill.prefill_batch_m128's own tail:
      pr['pfk_n16'](xr, W[('onw',0)], d['xh'])     (RMS norm)
      pr['head8'](W[('head',0)], d['xh'], d['logits'])
  download logits fp32, NLL of the next token in float64.

Gap-detection handles 128 (M128) and 64 (M64) chunks; other gaps are skipped
+ logged (tails). The last row of the last chunk has no target -> skipped.

Run wrapper: eval/ppl_run.sh dense   (env.common minus M1A_SERVE + dense env)
Output: eval/results/ppl_dense.json
"""
import os, sys, time, json, collections

os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
os.environ.pop("M1A_SERVE", None)          # NEVER attach the daemon

BASE = "~/tinygrad-metal"
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE)

import numpy as np
from mtp import MTPEngine, RBLK, CBLK, SLICE, CTXK
import mtp as _mtpmod
from trunk import CTX as TRUNK_CTX
from engine0 import dev
from gcycle import GCycleEngine

DATA = BASE + "/eval/data"
OUT = BASE + "/eval/results/ppl_dense.json"
SNAP = os.getenv("SNAPDIR", "~/snap100k")

PROSE_TARGET = int(os.getenv("PPL_PROSE_TOK", "32768"))
CODE_TARGET = int(os.getenv("PPL_CODE_TOK", "10240"))
PRIVATE_TARGET = int(os.getenv("PPL_PRIVATE_TOK", "12288"))

DOMAINS_ALL = ("prose", "code", "prose_private", "code2")


def corpus_domains():
    """Domain list via DOMAINS env (comma-separated; default prose,code).
    ids come from eval/data/ppl_<name>_ids.json (tok_prep.py)."""
    want = [d.strip() for d in os.getenv("DOMAINS", "prose,code").split(",") if d.strip()]
    targets = {"prose": PROSE_TARGET, "code": CODE_TARGET,
               "prose_private": PRIVATE_TARGET, "code2": CODE_TARGET}
    return [(d, None, targets[d]) for d in want if d in DOMAINS_ALL]


def logsumexp(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def main():
    KV8 = os.getenv("KV8", "0") == "1"
    QH = os.getenv("QH", "0") == "1"
    REFSUF = ("_kv8_qh" if QH else "_kv8") if KV8 else ""

    meta = json.load(open(f"{SNAP}/meta.json"))
    P0 = int(meta["P"]); assert CTXK == int(meta["CTXK"])
    CUR0 = int(meta["cur0"])
    ids = np.load(f"{SNAP}/ids.npy").tolist()
    print(f"[ppl_dense] boot: P={P0} cur0={CUR0} prompt ids {len(ids)}", flush=True)

    t0 = time.perf_counter()
    E = MTPEngine(theta=1e7)
    print(f"[ppl_dense] engine loaded {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- load100k (verbatim from test_w100k) ----
    P = E.P
    assert TRUNK_CTX == CTXK
    E.load_snapshot_kv(SNAP, progress=True)
    for j, i in enumerate(E.gdn_idx):
        P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
        P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
        if j % 8 == 0: E._flush()
    P.win_up("tok_slot", 0, np.array([CUR0], dtype=np.int32))
    P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
    E._flush()
    dev.synchronize()
    print(f"[ppl_dense] snapshot loaded {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- T1 reference (the daemon-boot-proven path; ref feeds the slice) ----
    G = GCycleEngine(E)
    G.build(); dev.synchronize()
    NTOK = 60
    G.run_tokens(NTOK, wait_each=True)
    h = E.P.down("tok_hist", (CTXK + 256,), np.int32)
    ref = h[P0:P0 + NTOK].tolist()
    assert all(t >= 0 for t in ref), "T1 reference incomplete"
    print(f"[ppl_dense] T1 ref done {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- slice + draft (verbatim) ----
    seen, sl = set(), []
    for t in ref + ids:
        if t not in seen: seen.add(t); sl.append(t)
    for t, _ in collections.Counter(ids).most_common():
        if t not in seen: seen.add(t); sl.append(t)
    base = sl[:]
    while len(sl) < SLICE: sl += base
    sl = sl[:SLICE]
    E.init_draft(sl)
    E.fill_draft(ids)
    E.build_graphs()
    dev.synchronize()
    print(f"[ppl_dense] graphs built {time.perf_counter()-t0:.1f}s — boot complete", flush=True)

    import pcache as _pc_mod
    import pf_prefill
    PFLS = pf_prefill.LS
    VOCAB = pf_prefill.VOCAB
    print(f"[ppl_dense] pcache/pf_prefill ready {time.perf_counter()-t0:.1f}s", flush=True)

    results = {}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))   # merge across partial runs
        except Exception:
            results = {}
    for name, text, target in corpus_domains():
        cids = [int(t) for t in json.load(open(f"{DATA}/ppl_{name}_ids.json"))][:target]
        print(f"[ppl_dense] {name}: {len(cids)} raw tokens", flush=True)

        st = {"nll": 0.0, "cnt": 0, "greedy": 0, "scored": 0, "skipped": []}
        tA = time.perf_counter()

        def ingest(pos_after, final64, cids=cids, st=st):
            gap = pos_after - st["scored"]
            buf = None
            if gap == 128 and getattr(E, "_pf_last128", None) is not None:
                buf = E._pf_last128; rows = gap; rbytes = 5120 * 4
            elif gap == 64 and getattr(E, "_pf_last64", None) is not None:
                buf = E._pf_last64; rows = gap; rbytes = 5120 * 4
            else:
                st["skipped"].append((st["scored"], gap)); return
            for r in range(rows):
                pos = st["scored"] + r
                if pos + 1 >= len(cids):
                    break
                tgt = cids[pos + 1]
                xr = buf.offset(offset=r * rbytes, size=rbytes)
                if st["cnt"] < 3:
                    print(f"[ppl_dense] rowdbg r={r} launching pfk_n16...", flush=True)
                E.pr["pfk_n16"](xr, E.W[("onw", 0)], E.P.d["xh"],
                                global_size=(1, 1, 1), local_size=PFLS)
                if st["cnt"] < 3:
                    dev.synchronize()
                    print(f"[ppl_dense] rowdbg r={r} pfk ok; launching head8...", flush=True)
                E.pr["head8"](E.W[("head", 0)], E.P.d["xh"], E.P.d["logits"],
                              global_size=(VOCAB // 8, 1, 1), local_size=PFLS, wait=True)
                if st["cnt"] < 3:
                    dev.synchronize()
                    print(f"[ppl_dense] rowdbg r={r} head8 ok; downloading...", flush=True)
                lg = E.P.down_at("logits", 0, VOCAB, np.float32).astype(np.float64)
                if st["cnt"] < 3:
                    print(f"[ppl_dense] rowdbg r={r} download ok absmax={np.abs(lg).max():.1f}",
                          flush=True)
                st["nll"] += logsumexp(lg) - lg[tgt]
                st["greedy"] += int(np.argmax(lg) == tgt)
                st["cnt"] += 1
            st["scored"] = pos_after
            if st["cnt"] and (pos_after % 4096 == 0):
                el = time.perf_counter() - tA
                print(f"[ppl_dense] {name} pos={pos_after}/{len(cids)} "
                      f"nll/tok={st['nll']/st['cnt']:.4f} greedy={st['greedy']/st['cnt']:.3f} "
                      f"rate={st['cnt']/el:.1f} tok/s", flush=True)

        _pc_mod.fresh_prefill(E, G, cids, ingest=ingest,
                              log=lambda s, **kw: print(f"[pc] {s} {kw}", flush=True)
                              if s in ("start", "done") or kw.get("k", 0) % 64 == 0 else None)
        dt = time.perf_counter() - tA
        res = {
            "n_tokens": st["cnt"],
            "nll_per_token": round(st["nll"] / max(1, st["cnt"]), 6),
            "ppl": round(float(np.exp(st["nll"] / max(1, st["cnt"]))), 4),
            "greedy_acc": round(st["greedy"] / max(1, st["cnt"]), 4),
            "wall_s": round(dt, 1),
            "score_rate_tps": round(st["cnt"] / dt, 1),
            "skipped_gaps": st["skipped"][:8],
        }
        results[name] = res
        print(f"[ppl_dense] {name} DONE {json.dumps(res)}", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)

    print("[ppl_dense] ALL DONE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
