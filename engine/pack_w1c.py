# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-c offline repacker. Two aligned layouts (nvcc merges adjacent u16 pairs into
u32 loads -> any per-lane multi-byte run must sit on ITS-NATURAL alignment or the
dext faults with SM Multiple Warp Errors; discovered via PTX diff of down8):
- IQ3_XXS regions (row = 98*NB bytes, SAME size): [qs 64B/block][scales 32B/block][d 2B/block]
  qs u16 at 64b+2*lane (2B ok), scale word u32 at 32b+4*s (4B ok), d u16 at 2b (2B ok).
- Q6_K padded blocks (row = 212*NB bytes, +1%): [pad 2][d 2][sc 16][lo 128][qh 64]
  lo/qh runs 2x u32 per lane, 4B-aligned for every block.
Everything else (Q5_K, Q4_K, Q8_0, IQ3_S, f32) stays RAW -- already aligned.
All repacked tensors are <= 52 MB -> plain np.save, no chunking."""
import os, sys
import numpy as np
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import parse_gguf, read_raw

OUT = "~/tinygrad-metal/engine0/packed"
os.makedirs(OUT, exist_ok=True)

def pack_iq3(info, ds):
  raw = np.frombuffer(read_raw(info, ds), dtype=np.uint8)
  R, C = info[1][1], info[1][0]   # gguf dims are [in, out]: rows = dims[1]
  assert C % 256 == 0
  NB = C // 256
  rowb = 98 * NB
  assert raw.size == R * rowb, (raw.size, R, rowb)
  x = raw.reshape(R, NB, 98)
  out = np.zeros((R, rowb), dtype=np.uint8)
  # ROW-LEVEL regions (kernel: qs@0..64NB, scales@64NB..96NB, d@96NB..98NB)
  out[:, :64*NB] = x[:, :, 2:66].reshape(R, 64*NB)     # qs
  out[:, 64*NB:96*NB] = x[:, :, 66:98].reshape(R, 32*NB)  # scales
  out[:, 96*NB:98*NB] = x[:, :, 0:2].reshape(R, 2*NB)  # d
  return out

def pack_q6(info, ds):
  raw = np.frombuffer(read_raw(info, ds), dtype=np.uint8)
  R, C = info[1][1], info[1][0]   # gguf dims are [in, out]: rows = dims[1]
  NB = 20
  assert raw.size == R * 210 * NB, (raw.size, R)
  x = raw.reshape(R, NB, 210)
  out = np.zeros((R, NB, 212), dtype=np.uint8)
  out[:, :, 0:2] = 0                   # pad
  out[:, :, 2:4] = x[:, :, 208:210]    # d
  out[:, :, 4:20] = x[:, :, 192:208]   # sc (16B)
  out[:, :, 20:148] = x[:, :, 0:128]   # lo
  out[:, :, 148:212] = x[:, :, 128:192]  # qh
  return out.reshape(R, 212 * NB)

def main():
  ds, infos = parse_gguf()
  attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
  gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
  jobs = []  # (outname, tensor, kind)
  for i in gdn_idx:
    jobs += [(f"gate{i}", f"blk.{i}.attn_gate.weight", "iq3"),
             (f"fg{i}", f"blk.{i}.ffn_gate.weight", "iq3"),
             (f"fu{i}", f"blk.{i}.ffn_up.weight", "iq3"),
             (f"fd{i}", f"blk.{i}.ffn_down.weight", "iq3")]
    if infos[f"blk.{i}.ssm_out.weight"][0] == 18:
      jobs.append((f"out{i}", f"blk.{i}.ssm_out.weight", "iq3"))
  for i in attn_idx:
    qt = infos[f"blk.{i}.attn_q.weight"][0]
    jobs += [(f"q{i}", f"blk.{i}.attn_q.weight", "iq3" if qt == 18 else "q6"),
             (f"k{i}", f"blk.{i}.attn_k.weight", "iq3"),
             (f"fg{i}", f"blk.{i}.ffn_gate.weight", "iq3"),
             (f"fu{i}", f"blk.{i}.ffn_up.weight", "iq3"),
             (f"fd{i}", f"blk.{i}.ffn_down.weight", "iq3")]
  total = 0
  for outname, tname, kind in jobs:
    path = f"{OUT}/{outname}.npy"
    if os.path.exists(path):
      print(f"[pack] skip existing {outname}", flush=True); continue
    arr = pack_iq3(infos[tname], ds) if kind == "iq3" else pack_q6(infos[tname], ds)
    np.save(path, arr)
    total += arr.size
    print(f"[pack] {outname}: {arr.shape[0]} rows x {arr.shape[1]}B = {arr.size/1e6:.1f} MB", flush=True)
  print(f"[pack] all done, total packed = {total/1e9:.2f} GB", flush=True)

if __name__ == "__main__":
  main()
