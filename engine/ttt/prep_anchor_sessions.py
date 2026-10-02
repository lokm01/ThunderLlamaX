#!/usr/bin/env python3

# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX DRAFTER Phase 2 (anchor-scale Stage B) — session prompt prep (zero GPU).

Tokenizes the FRESH novel corpus (~/anchor_books, bookcorpusopen indie novels
— disjoint from the r8 corpus AND from Stage-A sh_books training books) into
per-session engine id streams:

  <out>/sessions.json          [{f, split, target, title, n, off}]
  <out>/<f>_ids.npy            int32 prompt ids (header + book crop)

Prompt shape (mirrors the serve novel-prose class): the Qwen chat header
(user: "continue the story" instruction + book text) with add_generation_prompt;
the header stays at absolute position 0; the book body is cropped to the
session target with a deterministic per-book offset (position diversity is the
point — the corpus-only run proved <=16k positions don't reach 98k).

Run with system python3 (needs the api_server tokenizer cache path only):
  python3 prep_anchor_sessions.py --books ~/anchor_books --out ~/anchor_sessions
"""
import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np

# stub the serving-only imports (r8_corpus.py pattern — api_server is zero-GPU
# but imports fastapi at module level)
class _Any:
    def __call__(self, *a, **kw):
        return _Any()
    def __getattr__(self, k):
        return _Any()

class _Stub(types.ModuleType):
    def __getattr__(self, k):
        return _Any()

for m in ("fastapi", "fastapi.responses", "fastapi.middleware",
          "fastapi.middleware.cors", "uvicorn", "uvicorn.logging", "httpx",
          "http", "starlette", "starlette.concurrency", "starlette.requests",
          "starlette.responses", "starlette.middleware",
          "starlette.middleware.trustedhost", "starlette.middleware.cors"):
    mod = _Stub(m)
    mod.__path__ = []
    sys.modules[m] = mod

BASE = "/Users/lokm/tinygrad-metal"
sys.path.insert(0, BASE + "/engine0")
from api_server import load_tokenizer  # noqa: E402

GGUF = os.getenv("TLX_MODEL_PATH") or BASE + "/models/Qwen3.8-27B-IQ3_XXS.gguf"

INSTR = ("Continue the following story. Write the next part of the story in "
         "the same voice, seamlessly continuing the narrative:\n\n")

TAIL_SKIP = 2048   # never crop into the last 2k tokens (book-end artifacts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=os.path.expanduser("~/anchor_books"))
    ap.add_argument("--out", default=os.path.expanduser("~/anchor_sessions"))
    ap.add_argument("--max-ids", type=int, default=99400,
                    help="hard cap on total prompt ids (ctxk 100352 - decode headroom)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = json.load(open(f"{a.books}/manifest.json"))
    tok, template = load_tokenizer(GGUF)

    rows = []
    for bi, b in enumerate(man["books"]):
        f = b["f"]
        target = min(int(b["target"]), a.max_ids)
        text = open(f"{a.books}/{f}.txt").read()
        hdr = template.render(messages=[{"role": "user", "content": INSTR}],
                              add_generation_prompt=True)
        hdr_ids = tok.encode(hdr)
        body_ids = tok.encode(text)
        keep = target - len(hdr_ids)
        lo = 0
        hi = max(lo, len(body_ids) - keep - TAIL_SKIP)
        rng = np.random.default_rng(1000 + bi)      # deterministic per book
        off = int(rng.integers(lo, hi + 1)) if hi > lo else 0
        ids = np.array(hdr_ids + body_ids[off:off + keep], dtype=np.int32)
        assert len(ids) <= a.max_ids, (f, len(ids))
        np.save(f"{a.out}/{f}_ids.npy", ids)
        # novelty guard vs the r8 eval corpus: no 64-gram id overlap
        r8 = np.load("/Users/lokm/r8_prose_ids.npy")
        s8 = {tuple(r8[i:i + 8]) for i in range(len(r8) - 8)}
        sids = ids.tolist()
        dup = sum(1 for i in range(len(sids) - 8) if tuple(sids[i:i + 8]) in s8)
        rows.append(dict(f=f, split=b["split"], target=target, title=b["title"],
                         n=int(len(ids)), off=off, body_len=len(body_ids),
                         r8_8gram_overlap=dup))
        print(f"[prep] {f} split={b['split']:5s} target={target} ids={len(ids)} "
              f"off={off} body={len(body_ids)} r8_overlap_8g={dup}", flush=True)

    json.dump(dict(rows=rows, instr=INSTR, gguf=GGUF, tail_skip=TAIL_SKIP,
                   books_src=man.get("src", "?")),
              open(f"{a.out}/sessions.json", "w"), indent=1)
    ntr = sum(1 for r in rows if r["split"] == "train")
    nca = sum(1 for r in rows if r["split"] == "canary")
    tot = sum(r["n"] for r in rows)
    print(f"[prep] DONE {len(rows)} sessions ({ntr} train / {nca} canary), "
          f"{tot/1e6:.2f}M prompt ids", flush=True)


if __name__ == "__main__":
    main()
