# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — selective HF weight fetch (safetensors range reads).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Downloads only the tensors we need from Qwen/Qwen3.8-27B via HTTP Range
requests against the safetensors shards (no huggingface_hub needed):
  - mtp.* (the blk.64 draft layer, ~850MB bf16)
  - lm_head.weight + embed_tokens (optional, --with-head-emb, ~5.1GB)
Saves fp16/bf16 .npy files into a local dir. Works on Mac and rental."""
import argparse
import json
import os
import struct
import sys
import urllib.request

BASE = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main"


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "tlx-drafter/1.0"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def shard_header(shard):
    b = get(f"{BASE}/{shard}", (0, 8))
    hlen = struct.unpack("<Q", b[:8])[0]
    hdr = json.loads(get(f"{BASE}/{shard}", (8, 8 + hlen - 1)))
    hdr["__data_start__"] = 8 + hlen   # safetensors data_offsets are data-section-relative!
    return hdr


def fetch(tensors, out_dir, with_head_emb=False):
    os.makedirs(out_dir, exist_ok=True)
    idx = json.loads(get(f"{BASE}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    want = {k: v for k, v in wm.items() if k.startswith("mtp.")}
    if with_head_emb:
        want["lm_head.weight"] = wm["lm_head.weight"]
        want["model.language_model.embed_tokens.weight"] = wm["model.language_model.embed_tokens.weight"]
    if tensors:
        want = {k: v for k, v in want.items() if k in tensors}
    by_shard = {}
    for k, s in want.items():
        by_shard.setdefault(s, []).append(k)
    for shard, keys in sorted(by_shard.items()):
        hdr = shard_header(shard)
        print(f"[fetch] {shard}: {len(keys)} tensors", flush=True)
        for k in keys:
            info = hdr[k]
            assert info["dtype"] == "BF16", (k, info["dtype"])
            s, e = info["data_offsets"]
            ds0 = hdr["__data_start__"]
            n = e - s
            print(f"  {k} {info['shape']} {n/1e6:.1f}MB", flush=True)
            blob = get(f"{BASE}/{shard}", (ds0 + s, ds0 + e - 1))
            import numpy as np
            a = np.frombuffer(blob, dtype="<u2").astype(np.uint32)
            f32 = (a << 16).view(np.float32).reshape(info["shape"])
            out = os.path.join(out_dir, k.replace(".", "_") + ".npy")
            if np.abs(f32).max() < 65504:
                np.save(out, f32.astype(np.float16))  # bf16->fp16 exact
            else:
                np.save(out, f32)  # fp32 (embedding outliers exceed fp16 range)
            del blob, a, f32
    print("[fetch done]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/drafter/weights"))
    ap.add_argument("--with-head-emb", action="store_true")
    ap.add_argument("--tensors", nargs="*", default=None)
    a = ap.parse_args()
    fetch(a.tensors, a.out, a.with_head_emb)
