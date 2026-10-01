# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — prompt pool builder (runs on the rental).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Builds a mixed prompt pool (JSONL {text, cls, max_new}) for teacher
generation: GSM8K-train + MetaMath-class reasoning (the battery workload),
MBPP/code, UltraChat-class chat, prose-continuation, doc-QA, and a long-context
subset. Pilot ~3.5k prompts; full run 40-80k (scale args).
Contamination is a non-issue: the drafter proposes, the probe verifies."""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prompts.jsonl")
    ap.add_argument("--n-reason", type=int, default=1200)
    ap.add_argument("--n-code", type=int, default=500)
    ap.add_argument("--n-chat", type=int, default=700)
    ap.add_argument("--n-prose", type=int, default=600)
    ap.add_argument("--n-docqa", type=int, default=300)
    ap.add_argument("--n-long", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    rng = random.Random(a.seed)
    rows = []

    def add(text, cls, mx):
        text = text.strip()
        if 40 < len(text) < 60000:
            rows.append({"text": text, "cls": cls, "max_new": mx})

    # ---- reasoning: GSM8K train (the battery class) ----
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        idx = rng.sample(range(len(ds)), min(a.n_reason, len(ds)))
        for i in idx:
            q = ds[i]["question"]
            add(f"{q}\n\nReason step by step, then give the final numeric answer after '#### '.",
                "gsm8k", 512)
        print(f"[prompts] gsm8k {min(a.n_reason, len(ds))}")
    except Exception as e:
        print(f"[prompts] gsm8k FAILED: {e}")
    # ---- reasoning: MetaMathQA slice ----
    try:
        n_meta = max(200, a.n_reason // 4)
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        idx = rng.sample(range(len(ds)), min(n_meta, len(ds)))
        for i in idx:
            add(ds[i]["query"], "metamath", 512)
        print(f"[prompts] metamath {n_meta}")
    except Exception as e:
        print(f"[prompts] metamath FAILED: {e}")
    # ---- code: MBPP ----
    try:
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", trust_remote_code=False)
        except Exception:
            ds = load_dataset("sahil2801/CodeAlpaca", split="train")  # fallback code class
            global _code_alpaca
            _code_alpaca = True
        idx = rng.sample(range(len(ds)), min(a.n_code, len(ds)))
        for i in idx:
            t = ds[i]
            if "prompt" in t and "test" in t:
                add(f"Write a Python function for the following task.\n\nTask: {t['prompt']}\n\n"
                    f"Test examples: {str(t['test'])[:400]}\n\nProvide the function implementation:", "mbpp", 512)
            else:
                add(f"Write Python code for the following task.\n\n{t['instruction']}\n\n"
                    + (f"Input: {t['input'][:400]}\n\n" if t.get("input") else "") + "Solution:", "code", 512)
        print(f"[prompts] mbpp {min(a.n_code, len(ds))}")
    except Exception as e:
        print(f"[prompts] mbpp FAILED: {e}")
    # ---- chat: UltraChat ----
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        idx = rng.sample(range(len(ds)), min(a.n_chat, len(ds)))
        for i in idx:
            msgs = ds[i]["messages"]
            u = next((m["content"] for m in msgs if m["role"] == "user"), None)
            if u:
                add(u, "ultrachat", 512)
        print(f"[prompts] ultrachat {min(a.n_chat, len(ds))}")
    except Exception as e:
        print(f"[prompts] ultrachat FAILED: {e}")
    # ---- prose: PG19 (parquet branch) -> wikitext-103 -> TinyStories ----
    prose_ds = None
    for spec in (("deepmind/pg19", "refs/convert/parquet"), ("Salesforce/wikitext", "wikitext-103-raw-v1", None), ("roneneldan/TinyStories", None)):
        try:
            prose_ds = load_dataset(spec[0], spec[1] if len(spec) > 1 and spec[1] != "refs/convert/parquet" else None,
                                    revision=spec[1] if spec[1] == "refs/convert/parquet" else None, split="train",
                                    trust_remote_code=False)
            print(f"[prompts] prose corpus: {spec[0]}")
            break
        except Exception as e:
            print(f"[prompts] {spec[0]} failed: {str(e)[:120]}")
    try:
        assert prose_ds is not None
        ds = prose_ds
        idx = rng.sample(range(len(ds)), min(a.n_prose + a.n_long, len(ds)))
        for j, i in enumerate(idx):
            txt = ds[i].get("text") or ds[i].get("story") or ""
            cut = rng.randint(600, 4000)
            if j < a.n_prose:
                add(f"Continue the following passage of a novel in the same style:\n\n{txt[:cut]}\n\n",
                    "prose", 512)
            else:
                # long-context: ~8-16k chars prefix -> 2-4k tokens
                cut2 = rng.randint(9000, 26000)
                add(f"Continue the following passage of a novel in the same style.\n\n{txt[:cut2]}",
                    "prose_long", 768)
        print(f"[prompts] prose {a.n_prose + a.n_long}")
    except Exception as e:
        print(f"[prompts] prose FAILED: {e}")
    # ---- doc-QA: SQuAD contexts ----
    try:
        ds = load_dataset("rajpurkar/squad", split="train")
        idx = rng.sample(range(len(ds)), min(a.n_docqa, len(ds)))
        for i in idx:
            t = ds[i]
            add(f"Read the passage and answer the question.\n\nPassage: {t['context']}\n\n"
                f"Question: {t['question']}\n\nAnswer:", "docqa", 256)
        print(f"[prompts] squad {min(a.n_docqa, len(ds))}")
    except Exception as e:
        print(f"[prompts] squad FAILED: {e}")

    rng.shuffle(rows)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"[prompts] {len(rows)} -> {a.out} {dict(Counter(r['cls'] for r in rows))}")


if __name__ == "__main__":
    main()
