#!/usr/bin/env python3
"""TLX DRAFTER Phase 0 (S1) — GSM8K-30 class-B traces for chain_sim.
ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
SPDX-License-Identifier: MIT
Copyright (c) 2026 lokm01

30 GSM8K battery-format transcripts (the 4-shot primer + test question +
the reference answer, plain-text continuation = the pilot's cloud-data
format), each teacher-forced through the DENSE engine via pcache.fresh_prefill
with per-128-chunk hidden capture (fp32 pre-final-norm rows = h_seed
semantics) + the end-of-trace draft kv_d/sc_d int8 state. PF_W4A8=0.

Run inside the dense GPU window AFTER ppl_dense's globals dump:
  python3 eval/dump_gsm8k.py --out ~/trace_dump/gsm8k --n 30
"""
import os, sys, time, json

os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
os.environ.pop("M1A_SERVE", None)          # NEVER attach the daemon

BASE = "~/tinygrad-metal"
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE)

import numpy as np
import argparse
from mtp import MTPEngine, SLICE, CTXK
from trunk import CTX as TRUNK_CTX
from engine0 import dev
from gcycle import GCycleEngine
import pcache as _pc_mod
import pf_prefill
import trace_dump_lib as tdl

SNAP = os.getenv("SNAPDIR", "~/snap100k")
DATA = BASE + "/eval/data"


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--tokcap", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # tokenizer: tg311 may lack the `tokenizers` wheel — pre-encoded ids win
    # (pre-encode with system python3: eval/gsm8k_pretok.py, no GPU needed).
    pre_ids_path = f"{DATA}/gsm8k_dump_ids.json"
    tok = None
    if os.path.exists(pre_ids_path):
        pre_ids = json.load(open(pre_ids_path))
    else:
        from api_server import load_tokenizer
        gguf = os.getenv("TLX_MODEL_PATH") or "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
        tok, _ = load_tokenizer(gguf)
        pre_ids = None

    shots = load_jsonl(f"{DATA}/gsm8k_train.jsonl")[:a.shots]
    tests = load_jsonl(f"{DATA}/gsm8k_test.jsonl")[:a.n]
    primer = "\n\n".join(f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots) + "\n\n"

    meta = json.load(open(f"{SNAP}/meta.json"))
    P0 = int(meta["P"]); assert CTXK == int(meta["CTXK"])
    ids = np.load(f"{SNAP}/ids.npy").tolist()

    E = MTPEngine(theta=1e7)
    E.load_snapshot_kv(SNAP, progress=False)
    for j, i in enumerate(E.gdn_idx):
        E.P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
        E.P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
        if j % 8 == 0:
            E._flush()
    E.P.win_up("tok_slot", 0, np.array([int(meta["cur0"])], dtype=np.int32))
    E.P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
    E._flush()
    dev.synchronize()
    print("[gsm8k_dump] engine booted", flush=True)

    # slice + draft (the daemon-boot order — SAME canonical slice as serve;
    # ref ids from the CACHED T1 stream when present, else a live 60-run)
    import collections
    G = GCycleEngine(E)
    G.build(); dev.synchronize()
    NTOK = 60
    refsuf = ("_kv8_qh" if os.getenv("QH", "0") == "1" else "_kv8") if os.getenv("KV8", "0") == "1" else ""
    refp = f"{SNAP}/engine_t1_ref{refsuf}.npy"
    if os.path.exists(refp):
        ref = np.load(refp).tolist()
        print("[gsm8k_dump] cached T1 ref for slice", flush=True)
    else:
        G.run_tokens(NTOK, wait_each=True)
        h = E.P.down("tok_hist", (CTXK + 256,), np.int32)
        ref = h[P0:P0 + NTOK].tolist()
    assert all(t >= 0 for t in ref)
    seen, sl = set(), []
    for t in ref + ids:
        if t not in seen:
            seen.add(t); sl.append(t)
    for t, _ in collections.Counter(ids).most_common():
        if t not in seen:
            seen.add(t); sl.append(t)
    base = sl[:]
    while len(sl) < SLICE:
        sl += base
    E.init_draft(sl[:SLICE])
    E.fill_draft(ids)
    E.build_graphs()
    dev.synchronize()
    print("[gsm8k_dump] graphs built — boot complete", flush=True)

    summary = []
    for k, ex in enumerate(tests):
        if pre_ids is not None:
            cids = [int(t) for t in pre_ids[k]][:a.tokcap]
        else:
            text = primer + f"Question: {ex['question']}\nAnswer: {ex['answer']}"
            cids = [int(t) for t in tok.encode(text)][:a.tokcap]
        if len(cids) < 256:
            summary.append({"idx": k, "n": len(cids), "skip": "short"})
            continue
        out = f"{a.out}/t{k:02d}"
        os.makedirs(out, exist_ok=True)
        hid_buf = []
        t0 = time.perf_counter()

        def _ing(pos_after, _f, hid_buf=hid_buf):
            if getattr(E, "_pf_last128", None) is None:
                return
            bufname = "xA128" if E._pf_last128 is E.P.d["xA128"] else "xB128"
            arr = E.P.down_at(bufname, 0, 128 * 5120, np.float32).reshape(128, 5120)
            hid_buf.append((pos_after - 128, arr.copy()))

        _pc_mod.fresh_prefill(E, G, cids, ingest=_ing,
                              log=lambda s, **kw: print(f"[pc] {s}", flush=True) if s in ("start", "done") else None)
        np.save(f"{out}/ids.npy", np.array(cids, dtype=np.int32))
        posn = int(E.P.down_at("pos_slot", 0, 1, np.int32)[0])
        tdl.dump_kvd(out, E, "end", min(posn, CTXK))
        if hid_buf:
            pos_arr = np.array([p for p, _ in hid_buf])
            hid = np.concatenate([x for _, x in hid_buf])
            np.save(f"{out}/span_pos.npy", pos_arr)
            np.save(f"{out}/hiddens.npy", hid)
        tdl.finish_trace(out, dict(kind="class_b_gsm8k", idx=k, n_tokens=len(cids),
                                   pos_end=posn, pf_w4a8=int(os.getenv("PF_W4A8", "0") or 0)))
        dt = time.perf_counter() - t0
        summary.append({"idx": k, "n": len(cids), "pos_end": posn, "wall_s": round(dt, 1)})
        print(f"[gsm8k_dump] t{k:02d}: {len(cids)} toks in {dt:.0f}s", flush=True)

    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(f"[gsm8k_dump] DONE {len(summary)} traces -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
