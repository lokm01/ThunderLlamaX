# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-b bootstrap: stock-model prefill + greedy baseline + engine-state snapshot.
Run ONCE (separate process from the engine; 12.6GB stock weights).
Saves ~/w1b_state_{tag}.npz with: conv/rec per GDN block (in block order),
cache_kv per attn block, P, ids, theta, base_out (60 stock greedy tokens).
  tag=2k    : prompt8k.txt truncated to 1988 tokens (engine decodes at pos ~1987..2046)
  tag=short : mtp_config default prompt (spec_base.json contract)"""
import os, sys, time, json
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal")
os.environ.setdefault("MTP_A3_OVERRIDE", os.path.expanduser("~/tinygrad-metal/a3b/override.json"))
os.environ.setdefault("MTP_A3C_OFF", "1")
os.environ.setdefault("MTP_EMB_GATHER", "1")
os.environ.setdefault("MTP_MAXCTX", "2048")
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, TransformerBlock
from tinygrad.llm.cli import SimpleTokenizer

GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
print("[loading stock model...]", flush=True)
model, kv = Transformer.from_gguf(GGUF, 2048)
tok = SimpleTokenizer.from_gguf_kv(kv)
from mtp_config import MTPConfig
CFG = MTPConfig.load()
gdn_blks = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
attn_blks = [b for b in model.blk if isinstance(b, TransformerBlock)]
theta = float(attn_blks[0].config.rope_theta)
print(f"[boot] {len(gdn_blks)} GDN + {len(attn_blks)} attn blocks, theta={theta}", flush=True)

from tinygrad.uop.ops import UOp
v_start_pos = UOp.variable("start_pos", 0, 2047)

def run_tag(tag, prompt_txt, maxlen=1988):
  ids = [0] + tok.encode(prompt_txt)
  if len(ids) > maxlen: ids = ids[:maxlen]
  P = len(ids)
  t_full = Tensor(ids + [0] * (2048 - len(ids)), dtype="int32").reshape(1, 2048)
  temp = Tensor([0.0])
  def fwd(inp, sp):
    return model(inp.contiguous(), v_start_pos.bind(sp), temp).realize()
  t0 = time.perf_counter()
  for sp in range(P - 1):
    fwd(t_full[:, sp:sp+1], sp)
    if sp % 400 == 0: print(f"[{tag}] prefill {sp}/{P-1} ({time.perf_counter()-t0:.0f}s)", flush=True)
  print(f"[{tag}] prefill done in {time.perf_counter()-t0:.0f}s, snapshotting", flush=True)
  snap = {"theta": np.array([theta]), "P": np.array([P]), "ids": np.array(ids, dtype=np.int64)}
  for j, b in enumerate(gdn_blks):
    snap[f"conv{j}"] = b.conv_state.float().numpy().reshape(-1)
    snap[f"rec{j}"] = b.recurrent_state.float().numpy().reshape(-1)
  for j, b in enumerate(attn_blks):
    snap[f"kv{j}"] = b.cache_kv.numpy().reshape(-1)
  snap["base_out"] = np.zeros(60, dtype=np.int64)   # filled below; saved early for safety
  np.savez(f"~/w1b_state_{tag}.npz", **snap)
  print(f"[{tag}] P={P} state saved (baseline pending)", flush=True)
  # baseline: 60 stock greedy tokens continuing from the same state.
  # EAGER model.forward (the rollout_jit wrapper rejects re-fed argmax tensors:
  # "args mismatch in JIT" when inp stops being a slice of t_full).
  out, sp, toks = None, P - 1, []
  for k in range(60):
    if out is None: inp = t_full[:, sp:sp+1].contiguous().realize()
    else: inp = Tensor([[int(out.item())]], dtype="int32").contiguous().realize()
    out = model.forward(inp, sp, temp).realize()
    toks[-1 if False else 0:0] = []
    toks.append(int(out.item())); sp += 1
  snap["base_out"] = np.array(toks, dtype=np.int64)
  np.savez(f"~/w1b_state_{tag}.npz", **snap)
  print(f"[{tag}] P={P} saved; base_out[:20]={toks[:20]}", flush=True)
  return toks

def reset_states():
  for b in gdn_blks:
    b.conv_state.assign(Tensor.zeros(*b.conv_state.shape)).realize()
    b.recurrent_state.assign(Tensor.zeros(*b.recurrent_state.shape)).realize()

# 2k tag FIRST (virgin zero states = exactly a fresh generate); then reset + short.
p8 = open("~/prompt8k.txt").read()
run_tag("2k", p8)
reset_states()
short_toks = run_tag("short", CFG.prompt)
spec = json.load(open("~/tinygrad-metal/spec_base_2k.json")) if os.path.exists("~/tinygrad-metal/spec_base_2k.json") else None
if spec is not None:
  agree = sum(1 for a, b in zip(short_toks, spec) if a == b)
  print(f"[short] vs spec_base_2k.json: {agree}/{len(spec)} agree", flush=True)
print("[boot done]", flush=True)
