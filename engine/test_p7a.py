# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-A Stage-0 test: validate + bench the four probes.
Run: cd ~/tinygrad-metal/engine0 && DEV=NV ~/tg311/bin/python -u test_p7a.py [imma|repack|kv|ldmx]
Synced timing ONLY (async launch + dev.synchronize, min-of-10)."""
import os, sys, time, json
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
rng = np.random.default_rng(7)
RESULTS = {}
INT = (None, 4, dtypes.int32, ())

def prog(n, nvals=0):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target,
                                signature=tuple(INT for _ in range(nvals))))

def bench(fn, n=10):
  for _ in range(2): fn()
  dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter(); fn(); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best

# ================= probe 1: IMMA =================
def probe_imma():
  print("=== PROBE 1: IMMA (int8 tensor cores) ===", flush=True)
  NTHR = 128
  res = {}
  # --- m16n8k32 s8: exact validation on the ILP1 cubin ---
  A = rng.integers(-7, 8, (4, 16, 32), dtype=np.int8)   # per-warp A
  B = rng.integers(-7, 8, (4, 32, 8), dtype=np.int8)    # per-warp B (32x8)
  afr = np.zeros((NTHR, 4), np.uint32); bfr = np.zeros((NTHR, 2), np.uint32)
  cref = np.zeros((4, 16, 8), np.int64)
  for w in range(4):
    Aw, Bw = A[w].astype(np.int64), B[w].astype(np.int64)
    cref[w] = Aw @ Bw
    for L in range(32):
      tid = w * 32 + L; gid, tig = L >> 2, L & 3
      afr[tid] = [int.from_bytes(A[w][gid,   tig*4:tig*4+4].tobytes(), "little"),
                  int.from_bytes(A[w][gid+8, tig*4:tig*4+4].tobytes(), "little"),
                  int.from_bytes(A[w][gid,   tig*4+16:tig*4+20].tobytes(), "little"),
                  int.from_bytes(A[w][gid+8, tig*4+16:tig*4+20].tobytes(), "little")]
      bfr[tid] = [int.from_bytes(B[w][tig*4:tig*4+4, gid].tobytes(), "little"),
                  int.from_bytes(B[w][16+tig*4:16+tig*4+4, gid].tobytes(), "little")]
  P.up("afr", afr.reshape(-1)); P.up("bfr", bfr.reshape(-1)); P.up("cout", np.zeros(NTHR*4, np.int32))
  dev.synchronize()
  pr = prog("p7a_imma16832_nw4", 1)
  pr(P.d["afr"], P.d["bfr"], P.d["cout"], global_size=(82,1,1), local_size=(NTHR,1,1), vals=(1,)); dev.synchronize()
  c = P.down("cout", (NTHR, 4), np.int32)
  ok = True
  for w in range(4):
    for L in range(32):
      tid = w*32+L; gid, tig = L >> 2, L & 3
      exp = [cref[w][gid, 2*tig], cref[w][gid, 2*tig+1], cref[w][gid+8, 2*tig], cref[w][gid+8, 2*tig+1]]
      if list(c[tid]) != exp: ok = False; print(f"  mismatch w{w} L{L}: got {list(c[tid])} exp {exp}"); break
  print(f"[imma] m16n8k32.s8 VALIDATION: {'EXACT PASS' if ok else 'FAIL'}", flush=True)
  res["imma16832_s8_exact"] = bool(ok)
  # --- bench ILP1 / ILP4 ---
  for tag, name, ilp in (("ilp1", "p7a_imma16832_nw4", 1), ("ilp4", "p7a_imma16832i4_nw4", 4)):
    pr = prog(name, 1); iters = 20000
    t = bench(lambda: pr(P.d["afr"], P.d["bfr"], P.d["cout"],
                         global_size=(82,1,1), local_size=(NTHR,1,1), vals=(iters,)))
    tops = 82*4*iters*ilp*8192/t/1e12
    print(f"[imma] m16n8k32.s8 {tag}: {t*1e3:7.3f} ms -> {tops:7.2f} TOPS", flush=True)
    res[f"imma16832_s8_{tag}_tops"] = round(tops, 2)
  # --- m16n8k32 u8 variant (validate + bench) ---
  Au = rng.integers(0, 250, (4, 16, 32), dtype=np.uint8).astype(np.int64)
  Bu = rng.integers(0, 250, (4, 32, 8), dtype=np.uint8).astype(np.int64)
  Au8 = Au.astype(np.uint8); Bu8 = Bu.astype(np.uint8)
  afr2 = np.zeros((NTHR, 4), np.uint32); bfr2 = np.zeros((NTHR, 2), np.uint32)
  cref2 = Au @ Bu
  for w in range(4):
    for L in range(32):
      tid = w*32+L; gid, tig = L >> 2, L & 3
      afr2[tid] = [int.from_bytes(Au8[w][gid, tig*4:tig*4+4].tobytes(), "little"),
                   int.from_bytes(Au8[w][gid+8, tig*4:tig*4+4].tobytes(), "little"),
                   int.from_bytes(Au8[w][gid, tig*4+16:tig*4+20].tobytes(), "little"),
                   int.from_bytes(Au8[w][gid+8, tig*4+16:tig*4+20].tobytes(), "little")]
      bfr2[tid] = [int.from_bytes(Bu8[w][tig*4:tig*4+4, gid].tobytes(), "little"),
                   int.from_bytes(Bu8[w][16+tig*4:16+tig*4+4, gid].tobytes(), "little")]
  P.up("afr2", afr2.reshape(-1)); P.up("bfr2", bfr2.reshape(-1)); P.up("cout2", np.zeros(NTHR*4, np.int32))
  dev.synchronize()
  pr = prog("p7a_imma16832u8_nw4", 1)
  pr(P.d["afr2"], P.d["bfr2"], P.d["cout2"], global_size=(82,1,1), local_size=(NTHR,1,1), vals=(1,)); dev.synchronize()
  c = P.down("cout2", (NTHR, 4), np.int32); ok2 = True
  for w in range(4):
    for L in range(32):
      tid = w*32+L; gid, tig = L >> 2, L & 3
      exp = [cref2[w][gid, 2*tig], cref2[w][gid, 2*tig+1], cref2[w][gid+8, 2*tig], cref2[w][gid+8, 2*tig+1]]
      if list(c[tid]) != exp: ok2 = False; break
  iters = 20000
  t = bench(lambda: pr(P.d["afr2"], P.d["bfr2"], P.d["cout2"],
                       global_size=(82,1,1), local_size=(NTHR,1,1), vals=(iters,)))
  tops = 82*4*iters*4*8192/t/1e12
  print(f"[imma] m16n8k32.u8 VALIDATION: {'EXACT PASS' if ok2 else 'FAIL'}; bench ilp4 {t*1e3:7.3f} ms -> {tops:7.2f} TOPS", flush=True)
  res["imma16832_u8_exact"] = bool(ok2); res["imma16832_u8_ilp4_tops"] = round(tops, 2)
  # --- m8n8k16 s8 fallback ---
  A3 = rng.integers(-7, 8, (4, 8, 16), dtype=np.int8); B3 = rng.integers(-7, 8, (4, 16, 8), dtype=np.int8)
  afr3 = np.zeros(NTHR, np.uint32); bfr3 = np.zeros(NTHR, np.uint32)
  cref3 = A3.astype(np.int64) @ B3.astype(np.int64)
  for w in range(4):
    for L in range(32):
      tid = w*32+L; gid, tig = L >> 2, L & 3
      afr3[tid] = int.from_bytes(A3[w][gid, tig*4:tig*4+4].tobytes(), "little")
      bfr3[tid] = int.from_bytes(B3[w][tig*4:tig*4+4, gid].tobytes(), "little")
  P.up("afr3", afr3); P.up("bfr3", bfr3); P.up("cout3", np.zeros(NTHR*2, np.int32))
  dev.synchronize()
  pr = prog("p7a_imma8816_nw4", 1)
  pr(P.d["afr3"], P.d["bfr3"], P.d["cout3"], global_size=(82,1,1), local_size=(NTHR,1,1), vals=(1,)); dev.synchronize()
  c = P.down("cout3", (NTHR, 2), np.int32); ok3 = True
  for w in range(4):
    for L in range(32):
      tid = w*32+L; gid, tig = L >> 2, L & 3
      exp = [cref3[w][gid, 2*tig], cref3[w][gid, 2*tig+1]]
      if list(c[tid]) != exp: ok3 = False; break
  iters = 40000
  t = bench(lambda: pr(P.d["afr3"], P.d["bfr3"], P.d["cout3"],
                       global_size=(82,1,1), local_size=(NTHR,1,1), vals=(iters,)))
  tops = 82*4*iters*4*(2*8*8*16)/t/1e12
  print(f"[imma] m8n8k16.s8 VALIDATION: {'EXACT PASS' if ok3 else 'FAIL'}; bench ilp4 {t*1e3:7.3f} ms -> {tops:7.2f} TOPS", flush=True)
  res["imma8816_s8_exact"] = bool(ok3); res["imma8816_s8_ilp4_tops"] = round(tops, 2)
  best = max([v for k, v in res.items() if k.endswith("_tops") and any(s in k for s in ("s8", "u8")) and res.get(k.replace("_tops", "_exact"), True)] or [0])
  res["imma_best_tops"] = best
  res["imma_verdict"] = "PASS" if (res.get("imma16832_s8_exact") and best >= 35.5) else ("PARTIAL" if best > 0 else "FAIL")
  print(f"[imma] VERDICT: {res['imma_verdict']} (best {best:.1f} TOPS vs 142 peak; PASS>=35.5)", flush=True)
  RESULTS.update(res)

# ================= probe 2: repacked-W stream =================
def probe_repack():
  print("=== PROBE 2: REPACKED-W STREAM (real FFN gate IQ3_XXS) ===", flush=True)
  NROWS, ROWB = 17408, 1960
  raw = np.load(f"{BASE}/packed/fg0.npy")
  assert raw.size == NROWS*ROWB, raw.size
  u16 = raw.view(np.uint16).reshape(NROWS, 980)
  u32 = raw.view(np.uint32).reshape(NROWS, 490)
  # python checksum — strided kernel semantics
  cs_strd = (int(u16[:, 0:640].sum(dtype=np.uint64)) + 4*int(u32[:, 320:480].sum(dtype=np.uint64))
             + int(u16[:, 960:980].sum(dtype=np.uint64))) & 0xFFFFFFFF
  # repack: unit (b*49+i) u16 source index per row
  srcidx = np.zeros(980, np.int64)
  for b in range(20):
    for i in range(49):
      if i < 32: srcidx[b*49+i] = 32*b + i
      elif i < 48: srcidx[b*49+i] = 640 + 16*b + (i-32)
      else: srcidx[b*49+i] = 960 + b
  rep = u16.reshape(2176, 8, 980)[:, :, srcidx].transpose(0, 2, 1).astype(np.uint16).reshape(-1)
  cs_rep = int(rep.view(np.uint32).sum(dtype=np.uint64)) & 0xFFFFFFFF
  P.up("w", raw); P.up("wr", rep)
  BYTES = NROWS*ROWB
  res = {}
  for tag, name, nthr in (("strd_nw8", "p7a_wstrd_nw8", 256), ("strd_nw16", "p7a_wstrd_nw16", 512)):
    grid = NROWS//8//(nthr//32)
    P.up("o", np.zeros(1024, np.uint32)); dev.synchronize()
    pr = prog(name, 1)
    pr(P.d["w"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(1,)); dev.synchronize()
    got = int(P.down("o", (1024,), np.uint32)[:grid].sum(dtype=np.uint64)) & 0xFFFFFFFF
    ok = got == cs_strd
    t = bench(lambda: pr(P.d["w"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(10,)))
    gbs = BYTES*10/t/1e9
    print(f"[repack] {tag:9s} val={'PASS' if ok else f'FAIL({got} vs {cs_strd})'} {t*1e3:7.3f} ms/10pass -> {gbs:6.1f} GB/s", flush=True)
    res[f"w_{tag}"] = {"ok": bool(ok), "gbs": round(gbs, 1)}
  for tag, name, nthr in (("rep8_nw8", "p7a_wrep8_nw8", 256), ("rep8_nw16", "p7a_wrep8_nw16", 512),
                          ("rep16_nw8", "p7a_wrep16_nw8", 256)):
    grid = 2176//(nthr//32)
    P.up("o", np.zeros(1024, np.uint32)); dev.synchronize()
    pr = prog(name, 1)
    pr(P.d["wr"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(1,)); dev.synchronize()
    got = int(P.down("o", (1024,), np.uint32)[:grid].sum(dtype=np.uint64)) & 0xFFFFFFFF
    ok = got == cs_rep
    t = bench(lambda: pr(P.d["wr"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(10,)))
    gbs = BYTES*10/t/1e9
    print(f"[repack] {tag:9s} val={'PASS' if ok else f'FAIL({got} vs {cs_rep})'} {t*1e3:7.3f} ms/10pass -> {gbs:6.1f} GB/s", flush=True)
    res[f"w_{tag}"] = {"ok": bool(ok), "gbs": round(gbs, 1)}
  best_rep = max(v["gbs"] for k, v in res.items() if k.startswith("w_rep") and v["ok"])
  res["w_best_repacked_gbs"] = best_rep
  res["w_verdict"] = "PASS" if best_rep >= 400 else "FAIL"
  print(f"[repack] VERDICT: {res['w_verdict']} (best repacked {best_rep:.1f} GB/s; PASS>=400)", flush=True)
  RESULTS.update(res)

# ================= probe 3: int8-KV stream =================
def probe_kv():
  print(f"=== PROBE 3: INT8-KV SLAB STREAM (pfa16 layout, {8*int(os.getenv(chr(80)+chr(55)+chr(65)+chr(95)+chr(67)+chr(84)+chr(88)+chr(75), chr(53)+chr(48)+chr(49)+chr(55)+chr(54)))} rows) ===", flush=True)
  CTXK = int(os.getenv("P7A_CTXK", "50176"))
  ROWS = 8*CTXK
  slab = rng.integers(-128, 128, ROWS*256, dtype=np.int8)
  sc16 = rng.integers(0, 65536, ROWS*8, dtype=np.uint16)
  slab32 = slab.view(np.uint32)
  u = np.arange(ROWS*16, dtype=np.int64)
  scidx = (u >> 4)*8 + ((u & 15) >> 1)
  dp = int(slab.astype(np.int64).sum()) & 0xFFFFFFFF
  cs = (int(slab32.sum(dtype=np.uint64)) + dp + int(sc16[scidx].sum(dtype=np.uint64))) & 0xFFFFFFFF
  sc32v = sc16.view(np.uint32)
  cs32 = (int(slab32.sum(dtype=np.uint64)) + dp + 2*int(sc32v[(u >> 4) * 4 + (((u & 15) >> 1) >> 1)].sum(dtype=np.uint64))) & 0xFFFFFFFF
  del u, scidx
  P.up("kv", slab); P.up("sc", sc16)
  BYTES = ROWS*256 + ROWS*16   # slab + scale u16 per 32B
  res = {}
  for tag, name, nthr in (("nw32_g82", "p7a_kvs_nw32", 1024), ("nw32_g164", "p7a_kvs_nw32", 1024),
                          ("sc32_g82", "p7a_kvsc32_nw32", 1024), ("sc32_g164", "p7a_kvsc32_nw32", 1024),
                          ("nw16_g82", "p7a_kvs_nw16", 512), ("nw16_g164", "p7a_kvs_nw16", 512)):
    grid = 164 if "164" in tag else 82
    P.up("o", np.zeros(1024, np.uint32)); dev.synchronize()
    pr = prog(name, 2)
    pr(P.d["kv"], P.d["sc"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(grid, 1)); dev.synchronize()
    got = int(P.down("o", (1024,), np.uint32)[:grid].sum(dtype=np.uint64)) & 0xFFFFFFFF
    ok = got == (cs32 if "sc32" in name else cs)
    t = bench(lambda: pr(P.d["kv"], P.d["sc"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(grid, 3)))
    gbs = BYTES*3/t/1e9
    print(f"[kv] {tag:9s} val={'PASS' if ok else f'FAIL({got} vs {cs})'} {t*1e3:7.3f} ms/3pass -> {gbs:6.1f} GB/s", flush=True)
    res[f"kv_{tag}"] = {"ok": bool(ok), "gbs": round(gbs, 1)}
  best = max(v["gbs"] for v in res.values() if isinstance(v, dict) and v["ok"])
  res["kv_best_gbs"] = best
  res["kv_verdict"] = "PASS" if best >= 300 else "FAIL"
  print(f"[kv] VERDICT: {res['kv_verdict']} (best {best:.1f} GB/s; PASS>=300)", flush=True)
  RESULTS.update(res)

# ================= probe 4: ldmatrix =================
def probe_ldmx():
  print("=== PROBE 4: LDMATRIX ===", flush=True)
  NTHR = 256
  src = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
  P.up("src", src); P.up("out4", np.zeros(NTHR*4, np.uint32)); P.up("outacc", np.zeros(NTHR, np.uint32))
  dev.synchronize()
  pr = prog("p7a_ldmx_nw8", 2)
  pr(P.d["src"], P.d["out4"], P.d["outacc"], global_size=(82,1,1), local_size=(NTHR,1,1), vals=(1, 0)); dev.synchronize()
  o4 = P.down("out4", (NTHR, 4), np.uint32); oa = P.down("outacc", (NTHR,), np.uint32)
  # expected: lane L reg j = u32 at byte j*64 + (L>>2)*8 + (L&3)*4
  ok = True
  for L in range(32):
    for j in range(4):
      exp = src[(j*64 + (L>>2)*8 + (L&3)*4)//4]
      for w in range(NTHR//32):
        if o4[w*32+L, j] != exp: ok = False
    s4 = sum(int(src[(j*64 + (L>>2)*8 + (L&3)*4)//4]) for j in range(4))
    for w in range(NTHR//32):
      if int(oa[w*32+L]) != (1*s4) & 0xFFFFFFFF: ok = False
  print(f"[ldmx] ldmatrix.x4 roundtrip: {'EXACT PASS' if ok else 'FAIL'}", flush=True)
  res = {"ldmx_roundtrip": bool(ok)}
  iters = 100000
  for mode, tag in ((0, "ldmatrix_x4"), (1, "lds32_x4")):
    t = bench(lambda: pr(P.d["src"], P.d["out4"], P.d["outacc"],
                         global_size=(82,1,1), local_size=(NTHR,1,1), vals=(iters, mode)))
    warps = 82*(NTHR//32)
    gbs = warps*iters*512/t/1e9
    ns_per_warp_iter = t*1e9/iters
    print(f"[ldmx] {tag:12s}: {t*1e3:7.3f} ms -> {ns_per_warp_iter:6.3f} ns/warp-iter, aggregate {gbs:7.1f} GB/s smem", flush=True)
    res[f"ldmx_{tag}_ns"] = round(ns_per_warp_iter, 3); res[f"ldmx_{tag}_gbs"] = round(gbs, 1)
  res["ldmx_verdict"] = "PASS" if res["ldmx_roundtrip"] else "FAIL"
  print(f"[ldmx] VERDICT: {res['ldmx_verdict']} (works; rate delta ldmatrix {res['ldmx_ldmatrix_x4_ns']} vs lds32 {res['ldmx_lds32_x4_ns']} ns/warp-iter)", flush=True)
  RESULTS.update(res)

if __name__ == "__main__":
  only = sys.argv[1:] or ["imma", "repack", "kv", "ldmx"]
  for tag, fn in (("imma", probe_imma), ("repack", probe_repack), ("kv", probe_kv), ("ldmx", probe_ldmx)):
    if tag in only:
      try: fn()
      except Exception as e:
        import traceback; traceback.print_exc()
        RESULTS[f"{tag}_error"] = repr(e)
        print(f"[{tag}] EXCEPTION {e!r}", flush=True)
  with open(f"{BASE}/p7a_results.json", "w") as f: json.dump(RESULTS, f, indent=1)
  print("[p7a] results:", json.dumps(RESULTS, indent=1), flush=True)
