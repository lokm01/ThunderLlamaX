# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — teacher-forced feature dump (rental GPU).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Teacher-forced pass over prompt+continuation with the bf16 teacher; captures
per-token FINAL PRE-NORM hiddens (a forward_pre_hook on the language model's
final RMSNorm — the exact h_seed semantics of engine accept.cu xA[m]) and
writes the training shards:

  <out>/ids.npy   int32 [N, Lmax]  (-1 pad, LEFT-cropped to Lmax)
  <out>/hids.npy  fp16  [N, Lmax, 5120]
  <out>/lens.npy  int32 [N]
  <out>/meta.json

Also a small held-out split (--holdout N -> <out>_eval/)."""
import argparse
import json
import os

import numpy as np
import torch


def find_final_norm(model):
    cands = []
    for name, m in model.named_modules():
        cn = type(m).__name__
        if "RMSNorm" in cn or "Qwen3_5RMSNorm" in cn:
            if name.endswith("language_model.norm") or name == "model.norm" or name.endswith("model.norm"):
                cands.append((name, m))
    assert cands, "final norm module not found"
    cands.sort(key=lambda x: -len(x[0]))
    return cands[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/gen.jsonl")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--lmax", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--holdout", type=int, default=64)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model.eval()
    norm = find_final_norm(model)
    hidden_dim = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    rows = [json.loads(l) for l in open(a.gen)]
    # tokenize all; group by length buckets
    seqs = []
    for r in rows:
        full = r["text"] + r["out_text"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(ids) < 96:
            continue
        seqs.append((ids, r["cls"]))
    print(f"[dump] {len(seqs)} usable sequences")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    hold = set(order[:a.holdout].tolist())
    os.makedirs(a.out, exist_ok=True)
    eval_dir = a.out + "_eval"
    os.makedirs(eval_dir, exist_ok=True)

    def dump_split(split_name, items, out_dir):
        N = len(items)
        Lmax = a.lmax
        ids_mm = np.lib.format.open_memmap(f"{out_dir}/ids.npy", mode="w+", dtype=np.int32,
                                           shape=(N, Lmax))
        h_mm = np.lib.format.open_memmap(f"{out_dir}/hids.npy", mode="w+", dtype=np.float16,
                                         shape=(N, Lmax, hidden_dim))
        lens = np.zeros(N, np.int32)
        # sort by length for batch efficiency, remember original order irrelevant
        items = sorted(items, key=lambda x: len(x[0]))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        for b0 in range(0, N, a.batch):
            chunk = items[b0:b0 + a.batch]
            inps = [x[0][-Lmax:] for x in chunk]     # left-crop to Lmax
            L = max(max(len(x) for x in inps), 64)
            inp = torch.full((len(chunk), L), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, x in enumerate(inps):
                inp[i, : len(x)] = torch.tensor(x)   # left-align, right-pad
                attn[i, : len(x)] = 1
            buf = {}
            def pre_hook(module, args, kwargs, out=None):
                buf["h"] = args[0].detach()
            h = norm.register_forward_pre_hook(pre_hook, with_kwargs=True)
            with torch.no_grad():
                model(input_ids=inp.cuda(), attention_mask=attn.cuda(), use_cache=False)
            h.remove()
            hh = buf["h"].float().cpu().numpy().astype(np.float16)  # [b, L, D]
            for i, ids2 in enumerate(inps):
                n = len(ids2)
                ids_mm[b0 + i, :n] = np.asarray(ids2, np.int32)
                h_mm[b0 + i, :n] = hh[i, :n]
                lens[b0 + i] = n
            if (b0 // a.batch) % 20 == 0:
                print(f"[dump:{split_name}] {b0 + len(chunk)}/{N}", flush=True)
        ids_mm.flush()
        h_mm.flush()
        np.save(f"{out_dir}/lens.npy", lens)
        json.dump({"lmax": int(Lmax), "hidden": int(hidden_dim), "n": int(N),
                   "tokens": int(lens.sum()), "classes": {}}, open(f"{out_dir}/meta.json", "w"))
        print(f"[dump:{split_name}] done: {N} seqs, {lens.sum()/1e6:.2f}M tokens")

    train_items = [s for i, s in enumerate(seqs) if i not in hold]
    eval_items = [s for i, s in enumerate(seqs) if i in hold]
    dump_split("train", train_items, a.out)
    dump_split("eval", eval_items, eval_dir)


if __name__ == "__main__":
    main()
