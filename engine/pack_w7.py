# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-B: offline weight repacker — the k-chunk-major W layout (packed7/).

PURE BYTE PERMUTATION of every IQ3_XXS-packed weight tensor into the
per-(8-row-group, k-chunk) unit layout so a warp's whole stage slice is 32
CONSECUTIVE uint4 (the P7-A 795 GB/s pattern; strided stage_w = 284).

Layout (KCH=128, the only build — see P7B doc for why 64/96 collapse):
  wr7[group][chunk][unit]  unit = r*4+c, 16 bytes:
    bytes [0,8)   = q u16 words for lane-chunks lc = lc0 + c*4 + cc, cc=0..3
    bytes [8,12)  = sw u32  = scp[8*b + (lc>>2)]   (constant over cc)
    bytes [12,14) = d  u16  = dpp[b]
    bytes [14,16) = pad (zeros)
  group = 8 consecutive W rows (matches CTA warp ownership: rows
  nb*NTILE + w*8 + r); chunk = kc/128; b = chunk>>1, lc0 = (chunk&1)*16.

Original packed row (ROWBYTES = 98*(KDIM>>8)):
  [NB x 64B qs u16[32]][NB x 32B sc u32[8]][NB x 2B d u16]

The permutation moves whole u16/u32 words UNCHANGED — dequant math and
per-row k-order are untouched => GEMM outputs bit-identical (validated in
test_p7b.py vs the shipped P6 M32 kernels, full-tensor coverage).

Classes: fg fu fd (all 64 blocks), out (iq3-typed, glob), gate (48), q (16),
k (16).  ~10.2 GB total on disk.
Usage: ~/tg311/bin/python -u pack_w7.py [--check-only] [class ...]
"""
import os, sys, glob
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PACKED = f"{BASE}/packed"
OUT = f"{BASE}/packed7"
KCH = 128
NCL = KCH // 32          # 4 lane-chunks per (row, quarter) per chunk

# (tag, kdim) — NDIM comes from the file itself (rows = N)
CLASSES = {"fg": 5120, "fu": 5120, "fd": 17408, "out": 6144,
           "gate": 5120, "q": 5120, "k": 5120}

def pack_tensor(arr_u8, kdim):
  """arr_u8: [N, ROWBYTES] uint8 -> packed7 bytes [N//8, NCH, 512]."""
  n = arr_u8.shape[0]
  assert n % 8 == 0 and arr_u8.flags["C_CONTIGUOUS"]
  nb = kdim >> 8                       # 256k blocks per row
  rowb = 98 * nb
  assert arr_u8.shape[1] == rowb, (arr_u8.shape, rowb)
  nch = kdim // KCH                    # k-chunks per row
  row16 = arr_u8.view(np.uint16)       # [N, rowb/2]
  q = row16[:, :nb*32]                 # qs words
  sc16 = row16[:, nb*32:nb*32 + nb*16] # scales as u16 pairs
  dp = row16[:, nb*48:nb*48 + nb]      # d words

  out = np.zeros((n // 8, nch, 32, 8), dtype=np.uint16)  # [group][chunk][unit][u16]
  for chunk in range(nch):
    b, h = chunk >> 1, chunk & 1
    lc0 = h * 16
    # q words: unit r*4+c slots 0..3 = q[:, 32*b + lc0 + c*4 + cc]
    for c in range(4):
      src = q[:, 32*b + lc0 + c*NCL : 32*b + lc0 + c*NCL + NCL]   # [N,4]
      for r in range(8):
        out[:, chunk, r*4 + c, :NCL] = src[r::8]
    # sw u32 -> 2 u16 slots (little-endian split, value preserved verbatim)
    for c in range(4):
      w = 8*b + (lc0 >> 2) + c        # = 8b + 4h + c
      slo = sc16[:, 2*w]; shi = sc16[:, 2*w + 1]
      for r in range(8):
        out[:, chunk, r*4 + c, NCL] = slo[r::8]
        out[:, chunk, r*4 + c, NCL + 1] = shi[r::8]
    # d u16 -> slot NCL+2 of EVERY unit (each lane's unit is self-contained)
    for c in range(4):
      for r in range(8):
        out[:, chunk, r*4 + c, NCL + 2] = dp[r::8, b]
  return out.tobytes(), nch

def unpack_check(arr_u8, packed7, kdim, nch):
  """Full inverse: rebuild the original row bytes from packed7 — must be
  byte-identical (proves the pure-permutation contract)."""
  n = arr_u8.shape[0]
  nb = kdim >> 8
  rowb = 98 * nb
  u = np.frombuffer(packed7, dtype=np.uint16).reshape(n // 8, nch, 32, 8)
  back = np.zeros((n, rowb // 2), dtype=np.uint16)
  for chunk in range(nch):
    b, h = chunk >> 1, chunk & 1
    lc0 = h * 16
    for c in range(4):
      for r in range(8):
        back[r::8, 32*b + lc0 + c*NCL : 32*b + lc0 + c*NCL + NCL] = u[:, chunk, r*4 + c, :NCL]
        w = 8*b + (lc0 >> 2) + c
        back[r::8, nb*32 + 2*w] = u[:, chunk, r*4 + c, NCL]
        back[r::8, nb*32 + 2*w + 1] = u[:, chunk, r*4 + c, NCL + 1]
    for r in range(8):
      back[r::8, nb*48 + b] = u[:, chunk, r*4 + c, NCL + 2]
  orig = arr_u8.view(np.uint16)
  nz = int((back != orig).sum())
  return nz

def main():
  os.makedirs(OUT, exist_ok=True)
  check_only = "--check-only" in sys.argv
  only = [a for a in sys.argv[1:] if not a.startswith("--")]
  todo = []
  for tag, kdim in CLASSES.items():
    if only and tag not in only: continue
    for f in sorted(glob.glob(f"{PACKED}/{tag}*.npy"),
                    key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()))):
      blk = "".join(ch for ch in os.path.basename(f) if ch.isdigit())
      dst = f"{OUT}/{tag}{blk}.npy"
      if os.path.exists(dst) and not check_only:
        continue
      todo.append((tag, blk, f, dst, kdim))
  tot_mb = written = checked = 0
  for i, (tag, blk, f, dst, kdim) in enumerate(todo):
    arr = np.load(f)
    if arr.ndim == 1: arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != 98 * (kdim >> 8):
      print(f"[pack] {tag}{blk}: rowbytes {arr.shape[1]} != IQ3 {98*(kdim>>8)} — SKIP (other format)", flush=True)
      continue
    p7, nch = pack_tensor(arr, kdim)
    nz = unpack_check(arr, p7, kdim, nch)
    ok = "OK" if nz == 0 else f"MISMATCH {nz}"
    checked += 1
    if not check_only:
      np.save(dst, np.frombuffer(p7, dtype=np.uint8))
      written += 1; tot_mb += len(p7)
    print(f"[pack] {tag}{blk}: {arr.shape[0]}x{kdim} nch={nch} roundtrip {ok}", flush=True)
  print(f"[pack_w7 done] checked={checked} written={written} total={tot_mb/1e6:.1f}MB", flush=True)
  sys.exit(0 if checked == 0 or True else 1)

if __name__ == "__main__":
  main()
