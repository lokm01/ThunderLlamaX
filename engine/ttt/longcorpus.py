# TLX DRAFTER Phase 1 — long-context corpus builder (rental or Mac).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""The G1 fix: long-context training documents. Emits JSONL rows:
  {"text": <full doc>, "cls": "book", "max_new": 0}          — teacher-forced
  {"text": <prompt slice>, "cls": "sg_long", "max_new": N}   — self-gen long

Sources (fallback chain, ALL modern — NOT memorized classics per the plan):
  1. lucadiliello/bookcorpusopen  (17k full indie novels)
  2. HuggingFaceFW/fineweb streaming, char-length filter (long web docs)
  3. deepmind/pg19 parquet        (LAST resort — classics; flagged in meta)

Books are filtered by Qwen-token length (>= min-tok), dedup'd by prefix,
shuffled (seeded), split into disjoint sets: teacher-forced docs vs self-gen
prompt slices. Self-gen prompts ask for a continuation in the same style —
the on-distribution long-ctx class (serve decodes the model's OWN prose).
"""
import argparse
import json
import os
import random


def load_books(n_max, min_chars, tokenizer, cache="/root/data/books.jsonl"):
    if os.path.exists(cache):
        rows = [json.loads(l) for l in open(cache)]
        print(f"[books] cache {len(rows)}")
        return rows
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("lucadiliello/bookcorpusopen", split="train")
        idx = list(range(len(ds)))
        random.Random(0).shuffle(idx)
        for i in idx:
            t = ds[i]["text"]
            if t and len(t) >= min_chars:
                rows.append({"text": t, "cls": "book", "max_new": 0})
            if len(rows) >= n_max * 3:
                break
        print(f"[books] bookcorpusopen: {len(rows)} raw >= {min_chars} chars")
    except Exception as e:
        print(f"[books] bookcorpusopen FAILED: {str(e)[:200]}")
    if len(rows) < n_max:
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
            seen = 0
            for ex in ds:
                seen += 1
                t = ex.get("text") or ""
                if len(t) >= min_chars * 2:      # web docs need more chars (noise)
                    rows.append({"text": t, "cls": "web", "max_new": 0})
                if len(rows) >= n_max * 3 or seen > 400000:
                    break
            print(f"[books] fineweb supplement: {len(rows)} ({seen} scanned)")
        except Exception as e:
            print(f"[books] fineweb FAILED: {str(e)[:200]}")
    # dedup by 200-char prefix
    seen_pre = set()
    ded = []
    for r in rows:
        k = r["text"][:200]
        if k not in seen_pre:
            seen_pre.add(k)
            ded.append(r)
    rows = ded
    # token-length filter (fast batch tokenizer, no specials)
    lens = []
    B = 64
    texts = [r["text"][:400000] for r in rows]
    for i in range(0, len(texts), B):
        enc = tokenizer(texts[i:i + B], add_special_tokens=False)["input_ids"]
        lens.extend(len(e) for e in enc)
        if (i // B) % 20 == 0:
            print(f"[books] toklen {i}/{len(texts)}", flush=True)
    ok = [(r, L) for r, L in zip(rows, lens) if L >= 60000]
    ok.sort(key=lambda x: -x[1])          # longest first (high-offset windows)
    out = [r for r, L in ok]
    with open(cache, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    import numpy as np
    print(f"[books] {len(out)} docs >=60k tok; len dist p50/p90/max: "
          f"{np.percentile([L for _, L in ok], [50, 90, 100]) if ok else '-'} "
          f"{dict(Counter(r['cls'] for r in out))}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tf", default="data/books_tf.jsonl", help="teacher-forced docs out")
    ap.add_argument("--out-sg", default="data/books_sg.jsonl", help="self-gen prompt out")
    ap.add_argument("--out-prose", default="data/prose_short.jsonl", help="short prose continuation prompts")
    ap.add_argument("--n-tf", type=int, default=170, help="teacher-forced books")
    ap.add_argument("--n-sg", type=int, default=80, help="self-gen prompt slices")
    ap.add_argument("--n-prose", type=int, default=800, help="short prose prompts")
    ap.add_argument("--min-chars", type=int, default=240000, help="~60k tokens")
    ap.add_argument("--sg-prompt-tok", type=int, default=36000)
    ap.add_argument("--sg-max-new", type=int, default=2560)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    books = load_books(a.n_tf + a.n_sg + a.n_prose + 40, a.min_chars, tok)
    assert len(books) >= a.n_tf + a.n_sg, f"only {len(books)} long docs — relax --min-chars"
    rng = random.Random(7)
    rng.shuffle(books)
    tf, sg, ps = books[:a.n_tf], books[a.n_tf:a.n_tf + a.n_sg], books[a.n_tf + a.n_sg:]
    os.makedirs(os.path.dirname(a.out_tf) or ".", exist_ok=True)
    with open(a.out_tf, "w") as f:
        for r in tf:
            f.write(json.dumps(r) + "\n")
        print(f"[books] {len(tf)} -> {a.out_tf}")
    chars = int(a.sg_prompt_tok * 3.7)
    with open(a.out_sg, "w") as f:
        for r in sg:
            cut = r["text"][:chars]
            dot = cut.rfind(". ")
            if dot > chars // 2:
                cut = cut[:dot + 1]
            f.write(json.dumps({"text": f"Continue the following passage of a novel in the same style.\n\n{cut}",
                                "cls": "sg_long", "max_new": a.sg_max_new}) + "\n")
    print(f"[books] self-gen prompts (~{chars} chars) -> {a.out_sg}")
    with open(a.out_prose, "w") as f:
        for r in ps[:a.n_prose]:
            cut = r["text"][20000:20000 + rng.randint(1800, 12000)]   # mid-book slices
            dot = cut.rfind(". ")
            if dot > 800:
                cut = cut[:dot + 1]
            f.write(json.dumps({"text": f"Continue the following passage of a novel in the same style:\n\n{cut}\n\n",
                                "cls": "prose", "max_new": 512}) + "\n")
    print(f"[books] {min(len(ps), a.n_prose)} short prose prompts -> {a.out_prose}")


if __name__ == "__main__":
    main()
