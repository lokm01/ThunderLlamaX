# TLX DRAFTER Phase 1 — teacher-forced feature dump (rental GPU).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Teacher-forced pass with the bf16 teacher; captures per-token FINAL
PRE-NORM hiddens (forward_pre_hook on the final RMSNorm — the exact h_seed
semantics of engine accept.cu xA[m]) and writes training shards:

  <out>/ids.npy   int32 [N, Lmax]  (-1 pad; row = LAST Lmax or a split window)
  <out>/hids.npy  fp16  [N, Lmax, 5120]
  <out>/hpre.npy  fp16  [N, 5120]   trunk hidden at (window start - 1); 0 if off=0
  <out>/offs.npy  int32 [N]         ABSOLUTE doc position of row[0] (the RoPE law)
  <out>/lens.npy  int32 [N]
  <out>/meta.json {lmax, hidden, n, tokens, anchors, tail_anchor, batch_hint}

v2 (Phase-1 full run): --filter-cls, --field text (teacher-forced docs),
--split-windows (multiple non-overlapping lmax windows per long doc, each with
its ABSOLUTE offset — the G1 long-ctx fix), meta flags for the trainer.
hpre capture: rows with off>0 carry ONE leading context token in the forward
(hidden of position s-1); its hidden is stored in hpre and dropped from hids.
"""
import argparse
import json
import os
import shutil

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


def plan_windows(rows, tok, a):
    """-> [(full_ids, start, n_win)] one entry per training window.
    split-windows mode: non-overlapping lmax windows; keep those whose ABSOLUTE
    end (j+1)*lmax <= abs_cap (the serve ctx band), then the LAST max_win of
    them (optionally every win_stride-th for position spread)."""
    seqs = []
    for ri, r in enumerate(rows):
        full = r["text"] if a.field == "text" else (r["text"] + r.get("out_text", ""))
        ids = tok(full, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if a.split_windows:
            if n < max(int(a.lmax * 0.9), a.min_tok):
                if n >= a.min_tok:
                    seqs.append((ids, 0))
                continue
            nwin = n // a.lmax
            cap = int(a.abs_cap // a.lmax)          # max j with (j+1)*lmax <= cap
            js = [j for j in range(nwin) if (j + 1) <= cap or cap == 0]
            if a.win_stride > 1:
                js = [j for j in js if (nwin - 1 - j) % a.win_stride == 0] or js
            if a.max_win > 0:
                js = js[-a.max_win:]
            for j in js:
                s = j * a.lmax
                seqs.append((ids, s))
            rem_s = nwin * a.lmax
            if not js and n - rem_s >= int(a.lmax * 0.5):
                seqs.append((ids, rem_s))
        else:
            if n < a.min_tok:
                continue
            s = max(0, n - a.lmax)          # left-crop = LAST lmax tokens
            seqs.append((ids, s))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/gen.jsonl")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--lmax", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg", type=int, default=0,
                    help="batch-1 cache-chained segment length (0 = single forward); must be %64")
    ap.add_argument("--holdout", type=int, default=64)
    ap.add_argument("--filter-cls", default=None, help="comma list of cls values to keep")
    ap.add_argument("--field", default="full", choices=["full", "text"],
                    help="full = text+out_text (teacher gen); text = teacher-forced doc only")
    ap.add_argument("--min-tok", type=int, default=96)
    ap.add_argument("--split-windows", action="store_true",
                    help="multiple non-overlapping lmax windows per long doc (offs recorded)")
    ap.add_argument("--abs-cap", type=int, default=0,
                    help="split mode: keep windows whose absolute end <= this (0 = no cap)")
    ap.add_argument("--max-win", type=int, default=0, help="split mode: last N windows (0 = all)")
    ap.add_argument("--win-stride", type=int, default=1,
                    help="split mode: keep every stride-th window from the end (position spread)")
    ap.add_argument("--anchors", type=int, default=1, help="trainer anchors per window (meta flag)")
    ap.add_argument("--tail-anchor", type=int, default=0, help="1 = anchors near window end (meta flag)")
    ap.add_argument("--batch-hint", type=int, default=0, help="trainer batch hint (0 = default)")
    ap.add_argument("--disk-guard-gb", type=float, default=30.0)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model.eval()
    norm = find_final_norm(model)
    hidden_dim = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    rows = [json.loads(l) for l in open(a.gen)]
    if a.filter_cls:
        keep = set(a.filter_cls.split(","))
        rows = [r for r in rows if r.get("cls") in keep]
    seqs = plan_windows(rows, tok, a)
    print(f"[dump] {len(seqs)} windows from {len(rows)} rows (lmax {a.lmax}, split={a.split_windows})")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    nhold = min(a.holdout, max(0, len(seqs) // 50)) if len(seqs) > 200 else 0
    hold = set(order[:nhold].tolist())
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.out + "_eval", exist_ok=True)

    def dump_split(split_name, items, out_dir):
        N = len(items)
        Lmax = a.lmax
        ids_mm = np.lib.format.open_memmap(f"{out_dir}/ids.npy", mode="w+", dtype=np.int32,
                                           shape=(N, Lmax))
        h_mm = np.lib.format.open_memmap(f"{out_dir}/hids.npy", mode="w+", dtype=np.float16,
                                         shape=(N, Lmax, hidden_dim))
        hpre_mm = np.lib.format.open_memmap(f"{out_dir}/hpre.npy", mode="w+", dtype=np.float16,
                                            shape=(N, hidden_dim))
        offs = np.zeros(N, np.int32)
        lens = np.zeros(N, np.int32)
        items = sorted(items, key=lambda x: x[1])   # by offset: same-doc windows adjacent
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        SEG = a.seg if a.seg > 0 else Lmax
        assert SEG % 64 == 0, "segment must be a multiple of the GDN chunk size (64)"
        done = 0

        def forward_capture(segments, base_pos):
            """Run segments (single-segment = one full forward when seg==Lmax).
            NO DynamicCache: use_cache=True bypasses the fla GDN kernel back to
            the memory-hungry reference path (measured OOM law)."""
            hs = []
            buf = {}

            def pre_hook(module, args, kwargs, out=None):
                buf["h"] = args[0].detach()
            hh = norm.register_forward_pre_hook(pre_hook, with_kwargs=True)
            try:
                for seg in segments:
                    L = seg.shape[-1]
                    seg = seg.reshape(1, L)
                    pos = torch.arange(base_pos, base_pos + L)[None]
                    with torch.no_grad():
                        model(input_ids=seg.cuda(), position_ids=pos.cuda(),
                              attention_mask=torch.ones_like(seg).cuda(),
                              use_cache=False)
                    base_pos += L
                    hs.append(buf["h"][0].float().cpu().numpy().astype(np.float16))
            finally:
                hh.remove()
            return np.concatenate(hs, axis=0)

        for b0 in range(0, N, a.batch):
            chunk = items[b0:b0 + a.batch]
            if a.batch == 1:
                ids, s = chunk[0]
                seq = (ids[s - 1:s] + ids[s:s + Lmax]) if s > 0 else ids[s:s + Lmax]
                hh = forward_capture([torch.tensor(seq[i:i + SEG]).cuda()
                                      for i in range(0, len(seq), SEG)], max(0, s - 1))
                nw = min(len(ids) - s, Lmax)
                hoff = 1 if s > 0 else 0
                if s > 0:
                    hpre_mm[b0] = hh[0]
                ids_mm[b0, :nw] = np.asarray(ids[s:s + nw], np.int32)
                h_mm[b0, :nw] = hh[hoff:hoff + nw]
                lens[b0] = nw
                offs[b0] = s
            else:
                # batched single-forward (short windows)
                L = max(max(len(x[0]) - x[1] for x in chunk) + 1, 64)
                inp = torch.full((len(chunk), L), pad, dtype=torch.long)
                attn = torch.zeros((len(chunk), L), dtype=torch.long)
                for i, (ids, s) in enumerate(chunk):
                    ext = (ids[s - 1:s] + ids[s:s + Lmax]) if s > 0 else ids[s:s + Lmax]
                    inp[i, : len(ext)] = torch.tensor(ext)
                    attn[i, : len(ext)] = 1
                buf = {}

                def pre_hook(module, args, kwargs, out=None):
                    buf["h"] = args[0].detach()
                h = norm.register_forward_pre_hook(pre_hook, with_kwargs=True)
                with torch.no_grad():
                    model(input_ids=inp.cuda(), attention_mask=attn.cuda(), use_cache=False)
                h.remove()
                hh = buf["h"].float().cpu().numpy().astype(np.float16)
                for i, (ids, s) in enumerate(chunk):
                    nw = min(len(ids) - s, Lmax) if s > 0 else min(len(ids), Lmax)
                    hoff = 1 if s > 0 else 0
                    if s > 0:
                        hpre_mm[b0 + i] = hh[i, 0]
                    ids_mm[b0 + i, :nw] = np.asarray(ids[s:s + nw], np.int32)
                    h_mm[b0 + i, :nw] = hh[i, hoff:hoff + nw]
                    lens[b0 + i] = nw
                    offs[b0 + i] = s
            done = b0 + len(chunk)
            if (b0 // a.batch) % 20 == 0:
                free = shutil.disk_usage(out_dir).free / 1e9
                print(f"[dump:{split_name}] {done}/{N} free {free:.0f}GB", flush=True)
                if free < a.disk_guard_gb:
                    print(f"[dump:{split_name}] DISK GUARD — truncating", flush=True)
                    break
        ids_mm.flush(); h_mm.flush(); hpre_mm.flush()
        np.save(f"{out_dir}/lens.npy", lens[:done])
        np.save(f"{out_dir}/offs.npy", offs[:done])
        json.dump({"lmax": int(Lmax), "hidden": int(hidden_dim), "n": int(done),
                   "tokens": int(lens[:done].sum()), "anchors": a.anchors,
                   "tail_anchor": a.tail_anchor, "batch_hint": a.batch_hint,
                   "truncated": done < N},
                  open(f"{out_dir}/meta.json", "w"))
        print(f"[dump:{split_name}] done: {done}/{N} windows, {lens[:done].sum()/1e6:.2f}M tokens")

    train_items = [s for i, s in enumerate(seqs) if i not in hold]
    eval_items = [s for i, s in enumerate(seqs) if i in hold]
    dump_split("train", train_items, a.out)
    if eval_items:
        dump_split("eval", eval_items, a.out + "_eval")


if __name__ == "__main__":
    main()
