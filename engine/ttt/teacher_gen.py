# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — teacher generation (vLLM on the rental; HF fallback).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Generates the teacher's own continuations (EAGLE-3 methodology: labels =
teacher tokens). Greedy 70% / temp 0.7-top_p0.95 30% mix. Output JSONL
{text, cls, out_text} -> consumed by dump_features.py."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="data/gen.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts)]
    print(f"[gen] {len(rows)} prompts, engine={a.engine}")
    if a.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
                  max_model_len=a.max_model_len, enforce_eager=False, max_num_seqs=512)
        sp_greedy = SamplingParams(temperature=0.0, max_tokens=600)
        outs = []
        B = 512
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            prompts = [r["text"] for r in chunk]
            # one greedy pass then a temp pass on 30%: batch separately for speed
            res1 = llm.generate(prompts, sp_greedy)
            idx_t = [j for j, r in enumerate(chunk) if (i + j) % 10 < 3]
            sp_t = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=600, seed=1234 + i)
            res2map = {}
            if idx_t:
                res2 = llm.generate([prompts[j] for j in idx_t], sp_t)
                res2map = dict(zip(idx_t, res2))
            for j, r in enumerate(chunk):
                o = res2map.get(j, res1[j])
                r2 = dict(r)
                r2["out_text"] = o.outputs[0].text
                outs.append(r2)
            print(f"[gen] {i + len(chunk)}/{len(rows)}", flush=True)
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                     device_map="cuda", attn_implementation="sdpa")
        model.eval()
        outs = []
        for i, r in enumerate(rows):
            enc = tok(r["text"], return_tensors="pt").to("cuda")
            do = (i % 10) < 3
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=384, do_sample=do,
                                   temperature=0.7 if do else None,
                                   top_p=0.95 if do else None, pad_token_id=tok.eos_token_id)
            r2 = dict(r)
            r2["out_text"] = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(r2)
            if i % 25 == 0:
                print(f"[gen] {i}/{len(rows)}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in outs:
            f.write(json.dumps(r) + "\n")
    ntok = sum(len(r["out_text"]) // 4 for r in outs)
    print(f"[gen done] {len(outs)} continuations, ~{ntok/1e6:.2f}M tokens -> {a.out}")


if __name__ == "__main__":
    main()
