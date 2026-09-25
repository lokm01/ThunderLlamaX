# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import parse_gguf, read_raw
from tinygrad.tensor import Tensor
from tinygrad import dtypes

ds, infos = parse_gguf()
info = infos["blk.64.nextn.eh_proj.weight"]
raw = read_raw(info, ds)
t8 = Tensor(np.frombuffer(raw, np.uint8).copy())
from tinygrad.llm.gguf import ggml_data_to_tensor
ref = ggml_data_to_tensor(t8, 5120*10240, 2).reshape(5120, 10240).realize()  # fork reshape = flat; shape (in=10240, out=5120) reversed
ref_np = ref.numpy().astype(np.float32)   # (10240, 5120): rows = inputs? cols = outputs?
print("[t] fork dequant shape:", ref_np.shape, "absmax", np.abs(ref_np).max())
# my packed matrix: (5120 out, 10240 in)
def deq():
  w = np.load("draft_pack/d_eh.npy"); nout, rowb = w.shape; ngrp = rowb // 144
  qs = w[:, :ngrp*128].reshape(nout, ngrp*8, 16)
  dd = np.frombuffer(w[:, ngrp*128:].tobytes(), dtype="<f2").reshape(nout, ngrp*8).astype(np.float32)
  out = np.zeros((nout, ngrp*256), np.float32)
  for s in range(ngrp*8):
    b = qs[:, s, :]
    nib = np.zeros((nout, 32), np.float32)
    for j in range(16):
      nib[:, 2*j] = (b[:, j] & 0xF) - 8
      nib[:, 2*j+1] = (b[:, j] >> 4) - 8
    out[:, s*32:(s+1)*32] = nib * dd[:, s:s+1]
  return out
mine = deq()
# The fork's reshape(10240, 5120) of the flat dequant: disk order = out-major (row=out).
# So ref_np[r, c] = output r, input c -> compare ref_np vs mine
print("[t] mine shape:", mine.shape)
rel = np.abs(ref_np - mine).max() / np.abs(ref_np).max()
print("[t] relerr fork-vs-mine:", rel)
print("[t] fork[0,:6] ", np.round(ref_np[0, :6], 4).tolist())
print("[t] mine[0,:6] ", np.round(mine[0, :6], 4).tolist())
print("[t] fork col? fork[:6,0]", np.round(ref_np[:6, 0], 4).tolist())
