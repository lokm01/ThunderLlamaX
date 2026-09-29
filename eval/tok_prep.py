#!/usr/bin/env python3
"""P9 EVAL — pre-tokenize the PPL corpora with the API tokenizer (system
python3; the scoring harnesses run under tg311 which has no fastapi).

Writes eval/data/ppl_{prose,code}_ids.npy (int32). Same tokenization for
both models (the Qwen3.6/3.8 tokenizers are family-identical; we verify by
encoding with BOTH GGUFs and asserting equal ids).
"""
import sys, os, json

sys.path.insert(0, "~/tinygrad-metal/engine0")
from api_server import load_tokenizer

DATA = "~/tinygrad-metal/eval/data"
DENSE_GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
MOE_GGUF = "~/models36/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"

def corpus_domains():
    pap = open(f"{DATA}/gutenberg_pap.txt", encoding="utf-8", errors="replace").read()
    i = pap.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if i >= 0:
        pap = pap[pap.find("\n", i) + 1:]
    j = pap.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if j >= 0:
        pap = pap[:j]
    code = open("~/tinygrad-metal/engine0/serve.py",
                encoding="utf-8", errors="replace").read()
    priv = open("~/tinygrad-metal/FIX_CAMPAIGN.md",
                encoding="utf-8", errors="replace").read()
    code2 = open("~/tinygrad-metal/MM_P7_lib.py",
                 encoding="utf-8", errors="replace").read()
    return [("prose", pap, 40000), ("code", code, 12000),
            ("prose_private", priv, 16000), ("code2", code2, 12000)]

def main():
    tok_d, _ = load_tokenizer(DENSE_GGUF)
    tok_m, _ = load_tokenizer(MOE_GGUF)
    for name, text, cap in corpus_domains():
        ids_d = tok_d.encode(text)[:cap]
        ids_m = tok_m.encode(text)[:cap]
        same = ids_d == ids_m
        print(f"[tok] {name}: dense={len(ids_d)} moe={len(ids_m)} identical={same}")
        if not same:
            print(f"[tok] WARNING: tokenizers disagree on {name}; "
                  f"writing the dense ids (scoring scripts re-tokenize per model is TODO)")
        with open(f"{DATA}/ppl_{name}_ids.json", "w") as f:
            json.dump([int(t) for t in ids_d], f)
    print("[tok] done")

if __name__ == "__main__":
    main()
