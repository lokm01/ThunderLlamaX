# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8: offline Q5_K -> packed5 repacker (the gdnqg qkv seg true-16B layout).

PURE INTEGER PERMUTATION of the 48 GDN blocks' Q5_K attn_qkv tensors
([10240, 5120], 176B per 256w block) into the per-(8-row-group, k-chunk)
unit layout so a warp's whole W slice is consecutive uint4 loads (the
packed7 pattern; the R7 SASS audit's one prefill exception class).

Layout (KCH=128, NCL=4, RPT=8 -> NTILE=64 nw8 gdnqg twin):
  w5[group][chunk][48][16B]   group = 8 consecutive W rows, chunk = kc/128:
    unit u in [0,32): LO unit for lane u (r_local = u>>2, qc_ = u&3):
      32 nibbles, nibble(w) at bits 4w, w = 0..31 covering the lane's
      32 weights k = chunk*128 + qc_*32 + w  (i.e. sub-block s = 4h+qc_).
    unit u in [32,48): META for the lane PAIR (r_local = (u-32)>>1, m = (u-32)&1):
      me[0] = hi-bits of qc_=2m's  32 weights (bit w)
      me[1] = hi-bits of qc_=2m+1's 32 weights (bit w)
      me[2] = scA | mnA<<6 | scB<<12 | mnB<<18   (6-bit scale/min ints,
              the dq_c2 extraction VERBATIM)
      me[3] = d u16 | dm u16<<16                 (fp16 VERBATIM)
  => 48*16 = 768B per (8 rows, 128k) = 6 bits/weight (payload 5.22 + dup
     scales/d). Dequant math in-kernel is the dq_c2 expression verbatim
     (same ints, same fp expression) -> GEMM outputs BIT-IDENTICAL.

Usage: ~/tg311/bin/python -u pack_w5.py [--check-only]
"""
import os, sys, struct
import numpy as np

GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
_TYPR = {0:(1,"c"),1:(1,"b"),2:(2,"H"),3:(2,"h"),4:(4,"I"),5:(4,"i"),6:(4,"f"),7:(1,"?"),10:(8,"Q"),11:(8,"q"),12:(8,"d")}
QUANT = {2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),12:(256,144),13:(256,176),
         14:(256,210),18:(256,98),21:(256,110),22:(256,82),23:(256,136),39:(32,17),41:(128,18)}
NATIVE = {0:4,1:2,24:1,25:2,26:4,27:8,28:8,30:2}

def parse_gguf(path=GGUF):
  f = open(path, "rb")
  assert f.read(4) == b"GGUF"; struct.unpack("<i", f.read(4))
  n_tensors = struct.unpack("<q", f.read(8))[0]; n_kv = struct.unpack("<q", f.read(8))[0]
  def rs():
    n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode()
  def rv(t):
    if t == 8: return rs()
    if t == 9:
      et = struct.unpack("<i", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
      return [rv(et) for _ in range(n)]
    nb, fmt = _TYPR[t]; return struct.unpack("<"+fmt, f.read(nb))[0]
  for _ in range(n_kv):
    rs(); t = struct.unpack("<i", f.read(4))[0]; rv(t)
  infos = {}
  for _ in range(n_tensors):
    name = rs(); nd = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
    t = struct.unpack("<i", f.read(4))[0]; off = struct.unpack("<Q", f.read(8))[0]
    infos[name] = (t, dims, off)
  align = 32
  data_start = (f.tell()+align-1)//align*align
  f.close()
  return data_start, infos

def tnbytes(ne, t):
  if t in NATIVE: return NATIVE[t]*ne
  a, b = QUANT[t]; return (ne//a)*b

def read_raw(info, data_start):
  t, dims, off = info
  ne = int(np.prod(dims)) if dims else 1
  with open(GGUF, "rb") as f:
    f.seek(data_start+off); return f.read(tnbytes(ne, t))

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{BASE}/packed5"
KCH = 128
KDIM = 5120
NCH = KDIM // KCH          # 40
NB = KDIM >> 8             # 20 Q5_K blocks per row
ROWB = 176 * NB            # 3520
SH8 = (1 << (4 * np.arange(8, dtype=np.int64))).reshape(1, 8)      # per-word nibble shifts
SHW = (1 << np.arange(32, dtype=np.int64)).reshape(1, 32)          # bit w

def scales_of(blk):
  """dq_c2's scale extraction VERBATIM (int values). blk: [..., 176] uint8."""
  sc = np.zeros(blk.shape[:-1] + (8,), dtype=np.int64)
  mn = np.zeros_like(sc)
  for s in range(4):
    sc[..., s] = blk[..., 4+s] & 63; mn[..., s] = blk[..., 8+s] & 63
  for s in range(4, 8):
    sc[..., s] = (blk[..., 8+s] & 0xF) | ((blk[..., s] >> 6) << 4)
    mn[..., s] = (blk[..., 8+s] >> 4) | ((blk[..., s+4] >> 6) << 4)
  return sc, mn

def pack_tensor(raw):
  """raw: [N, ROWB] uint8 -> packed5 bytes [N//8, NCH, 48, 16B]."""
  N = raw.shape[0]
  assert N % 8 == 0 and raw.flags["C_CONTIGUOUS"]
  blks = raw.reshape(N, NB, 176).astype(np.int64)
  lo = blks[:, :, 48:176]                      # [N, NB, 128]
  qh = blks[:, :, 16:48]                       # [N, NB, 32]
  sc, mn = scales_of(blks)                     # [N, NB, 8]
  d16 = blks[:, :, 0] | (blks[:, :, 1] << 8)   # [N, NB]
  dm16 = blks[:, :, 2] | (blks[:, :, 3] << 8)
  out = np.zeros((N // 8, NCH, 48, 4), dtype=np.uint32)
  for c in range(NCH):
    h, b = c & 1, c >> 1
    # ---- LO units: lanes 0..31 (rl*4 + qc) ----
    for qc in range(4):
      s = h*4 + qc
      nbl = (lo[:, b, (s >> 1)*32:(s >> 1)*32 + 32] >> ((s & 1) * 4)) & 0xF   # [N,32]
      n32 = (nbl.reshape(N, 4, 8) * SH8).sum(-1)                              # [N,4]
      out[:, c, qc:32:4, :] = n32.reshape(N // 8, 8, 4)
    # ---- META units: 32 + rl*2 + m (m in {0,1}: lane-chunk pairs (0,1),(2,3)) ----
    for m in range(2):
      sA, sB = h*4 + 2*m, h*4 + 2*m + 1
      bA = (qh[:, b] >> sA) & 1                     # [N,32]
      bB = (qh[:, b] >> sB) & 1
      me0 = (bA * SHW).sum(-1).astype(np.uint32)    # [N]
      me1 = (bB * SHW).sum(-1).astype(np.uint32)
      me2 = ((sc[:, b, sA] & 0x3F) | ((mn[:, b, sA] & 0x3F) << 6)
             | ((sc[:, b, sB] & 0x3F) << 12) | ((mn[:, b, sB] & 0x3F) << 18)).astype(np.uint32)
      me3 = (d16[:, b] | (dm16[:, b] << 16)).astype(np.uint32)
      M = np.stack([me0, me1, me2, me3], axis=1).reshape(N // 8, 8, 4)
      out[:, c, 32+m:48:2, :] = M
  return out.tobytes()

def unpack_check_rows(raw, p5, n_check=4):
  """Full inverse on the first n_check rows: rebuild raw rows byte-identical.
  Each Q5_K block b spans chunks c0=2b (sub-blocks s 0..3) + c1=2b+1 (s 4..7)."""
  N = raw.shape[0]
  u = np.frombuffer(p5, dtype=np.uint32).reshape(N // 8, NCH, 48, 4)
  nz = 0
  for r in range(min(n_check, N)):
    g, rl = r >> 3, r & 7
    back = np.zeros(ROWB, dtype=np.uint8)
    for b in range(NB):
      blk = np.zeros(176, dtype=np.int64)
      lo = np.zeros(128, dtype=np.int64); qh = np.zeros(32, dtype=np.int64)
      sca = {}; mna = {}
      for c in (2*b, 2*b + 1):
        h = c & 1
        for m in range(2):
          mm = int(u[g, c, 32 + rl*2 + m, 2])
          sca[h*4+2*m] = mm & 0x3F; mna[h*4+2*m] = (mm >> 6) & 0x3F
          sca[h*4+2*m+1] = (mm >> 12) & 0x3F; mna[h*4+2*m+1] = (mm >> 18) & 0x3F
        if h == 0:
          dd = int(u[g, c, 32 + rl*2 + 0, 3])
          blk[0] = dd & 0xFF; blk[1] = (dd >> 8) & 0xFF
          blk[2] = (dd >> 16) & 0xFF; blk[3] = (dd >> 24) & 0xFF
        for qc in range(4):
          s = h*4 + qc
          lo32 = u[g, c, rl*4 + qc].astype(np.int64)
          for w in range(32):
            nbl = (int(lo32[w >> 3]) >> (4*(w & 7))) & 0xF
            lo[(s >> 1)*32 + w] |= nbl << ((s & 1) * 4)
        for m in range(2):
          me = u[g, c, 32 + rl*2 + m].astype(np.int64)
          for w in range(32):
            qh[w] |= ((int(me[0]) >> w) & 1) << (h*4 + 2*m)
            qh[w] |= ((int(me[1]) >> w) & 1) << (h*4 + 2*m + 1)
      blk[16:48] = qh; blk[48:176] = lo
      for s in range(4):
        blk[4+s] = (sca[s] & 63) | ((sca[s+4] >> 4) << 6)
        blk[8+s] = (mna[s] & 63) | ((mna[s+4] >> 4) << 6)
      for s in range(4, 8):
        blk[8+s] = (sca[s] & 0xF) | ((mna[s] & 0xF) << 4)
      back[b*176:(b+1)*176] = blk.astype(np.uint8)
    nz += int((back != raw[r]).sum())
  return nz

def main():
  os.makedirs(OUT, exist_ok=True)
  check_only = "--check-only" in sys.argv
  ds, infos = parse_gguf()
  gdn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" not in infos]
  print(f"[w5] {len(gdn_idx)} GDN blocks", flush=True)
  tot = 0
  for i in gdn_idx:
    dst = f"{OUT}/qkv{i}.npy"
    if os.path.exists(dst) and not check_only:
      continue
    raw = np.frombuffer(read_raw(infos[f"blk.{i}.attn_qkv.weight"], ds), dtype=np.uint8)
    assert raw.size % ROWB == 0, raw.size
    raw = raw.reshape(-1, ROWB)
    assert raw.shape[0] == 10240, raw.shape
    p5 = pack_tensor(raw)
    nz = unpack_check_rows(raw, p5)
    ok = "OK" if nz == 0 else f"MISMATCH nz={nz}"
    print(f"[w5] qkv{i}: {raw.shape} -> {len(p5)/1e6:.1f}MB roundtrip {ok}", flush=True)
    assert nz == 0, (i, nz)
    if not check_only:
      np.save(dst, np.frombuffer(p5, dtype=np.uint8)); tot += len(p5)
    del raw, p5
  print(f"[w5 done] written {tot/1e9:.2f} GB", flush=True)

if __name__ == "__main__":
  main()
