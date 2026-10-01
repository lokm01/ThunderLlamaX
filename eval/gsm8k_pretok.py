#!/usr/bin/env python3
"""TLX P0-S1: pre-encode the GSM8K dump transcripts (system python3, no GPU).
Writes eval/data/gsm8k_dump_ids.json — the SAME 4-shot battery format the
cloud pilot used (plain-text continuation)."""
import os, sys, json

sys.path.insert(0, "~/tinygrad-metal/engine0")
from api_server import load_tokenizer

DATA = "~/tinygrad-metal/eval/data"
GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SHOTS = 4


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


tok, _ = load_tokenizer(GGUF)
shots = load_jsonl(f"{DATA}/gsm8k_train.jsonl")[:SHOTS]
tests = load_jsonl(f"{DATA}/gsm8k_test.jsonl")[:N]
primer = "\n\n".join(f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots) + "\n\n"
ids = []
for ex in tests:
    text = primer + f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    ids.append([int(t) for t in tok.encode(text)])
json.dump(ids, open(f"{DATA}/gsm8k_dump_ids.json", "w"))
print(f"[pretok] {len(ids)} transcripts -> {DATA}/gsm8k_dump_ids.json "
      f"(lens {[len(x) for x in ids][:10]}...)")
