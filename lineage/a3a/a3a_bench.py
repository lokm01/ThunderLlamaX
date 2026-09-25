# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""A3a: standalone IQ3_XXS dequant-GEMV microbenchmark via hand CUDA cubin + fork NV loader.
Run: DEV=NV PYTHONPATH=~/tinygrad-src ~/tg311/bin/python a3a_bench.py
Validates the CUDA port against a numpy transcription of the FORK dequant math, and that
numpy port against the fork's ggml_data_to_tensor itself (bit-exact), then benches
y[N] = W[N,K].x[K] with W as raw IQ3_XXS blocks for the three model GEMV shapes.
"""
import os, sys, time
import numpy as np
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.helpers import round_up
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.autogen import ggml_common

HERE = os.path.dirname(os.path.abspath(__file__))
dev = Device["NV"]

# ---------------- tables ----------------
GRID_RAW = np.array(ggml_common.iq3xxs_grid, dtype=np.uint32)          # 256 x packed 4 bytes
GRIDF = np.array([[ (w >> (8*i)) & 0xFF for i in range(4)] for w in GRID_RAW], dtype=np.float32).reshape(-1)
ESIGN = np.array([i | (0x80 if bin(i).count("1") % 2 else 0) for i in range(128)], dtype=np.uint8)
assert GRIDF.shape == (1024,) and GRIDF.min() == 4 and GRIDF.max() == 62

# ---------------- numpy reference: EXACT port of fork gguf.py case 18 ----------------
def dequant_ref(blocks):  # (nb, 98) uint8 -> (nb*256,) float32
  nb = blocks.shape[0]
  d = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(nb)
  words = np.ascontiguousarray(blocks[:, 66:98]).view(np.uint32).reshape(nb, 8)
  db = d[:, None] * ((words >> 28).astype(np.float32) + 0.5) * 0.5                    # (nb,8)
  shifts = np.array([0, 7, 14, 21], dtype=np.uint32)
  sidx = ((words[:, :, None] >> shifts) & 0x7F).reshape(nb, 32)                       # (nb,32)
  signbits = (ESIGN[sidx][:, :, None] >> np.arange(8, dtype=np.uint8)) & 1            # (nb,32,8)
  signs = np.where(signbits == 0, 1.0, -1.0).astype(np.float32).reshape(nb, 8, 4, 8)
  qs = blocks[:, 2:66]
  vals = GRIDF.reshape(256, 4)[qs].reshape(nb, 8, 4, 8)                               # (nb,64,4)->(nb,8,4,8)
  return (db[:, :, None, None] * vals * signs).reshape(nb, 256)

def make_blocks(nb, seed):
  rng = np.random.default_rng(seed)
  b = np.zeros((nb, 98), dtype=np.uint8)
  d = rng.lognormal(-7.8, 0.4, nb).astype(np.float16)          # realistic IQ3_XXS delta scale
  b[:, :2] = np.ascontiguousarray(d).view(np.uint8).reshape(nb, 2)
  b[:, 2:66] = rng.integers(0, 256, (nb, 64), dtype=np.uint8)
  words = rng.integers(0, 1 << 32, (nb, 8), dtype=np.uint64).astype(np.uint32)
  b[:, 66:98] = np.ascontiguousarray(words).view(np.uint8).reshape(nb, 32)
  return b

# cross-check numpy port vs the fork itself (bit-exact expected)
def crosscheck_fork():
  from tinygrad.tensor import Tensor
  from tinygrad.llm.gguf import ggml_data_to_tensor
  nb = 16
  blocks = make_blocks(nb, 123)
  t = Tensor(blocks, dtype=dtypes.uint8)
  w_fork = ggml_data_to_tensor(t, nb * 256, 18).realize().numpy().astype(np.float32)
  w_ref = dequant_ref(blocks)
  assert w_fork.shape == w_ref.shape, (w_fork.shape, w_ref.shape)
  if not np.array_equal(w_fork, w_ref):
    print(f"[crosscheck] MISMATCH maxdiff={np.abs(w_fork-w_ref).max()}")
    raise SystemExit("numpy reference != fork ggml_data_to_tensor")
  print("[crosscheck] numpy port == fork ggml_data_to_tensor (bit-exact, 16 blocks)")

# ---------------- loader ----------------
CUBIN = open(os.path.join(HERE, "iq3.cubin"), "rb").read()
PTR = (None, 0, dtypes.uint8, ())

def kernel_regs(lib, name):
  """Per-kernel register count. nvcc records EIATTR_REGCOUNT entries in the MODULE-level
  .nv.info section keyed by ELF symbol index; the fork blindly takes the last one, which
  is wrong for multi-kernel cubins (under-allocation -> SM 'Out Of Range Register' fault).
  Resolve name -> symbol index via .symtab and pick the matching entry."""
  import struct
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(lib, force_section_align=128)
  symregs, symtab, strtab = {}, None, None
  for sh in sections:
    if sh.name == ".nv.info":
      off = 0
      while off < sh.header.sh_size:
        typ, param, sz = struct.unpack_from("BBH", sh.content, off)
        if typ == 0x4 and param == 0x2f:
          symidx, regs = struct.unpack_from("II", sh.content, off + 4)
          symregs[symidx] = regs
        off += (sz if typ == 0x4 else 0) + 4
    elif sh.name == ".symtab": symtab = bytes(sh.content)
    elif sh.name == ".strtab": strtab = bytes(sh.content)
  if not symregs or symtab is None or strtab is None: return None
  for i in range(len(symtab) // 24):  # Elf64_Sym
    st_name, = struct.unpack_from("I", symtab, i * 24)
    nm = strtab[st_name:strtab.index(b"\x00", st_name)].decode()
    if nm == name and i in symregs: return symregs[i]
  return None

def mkprg(name, nbuf, nvals):
  sig = [PTR] * nbuf + [(None, 0, dtypes.int32, ())] * nvals
  prg = NVProgram(dev, TinyELF(lib=CUBIN, name=name, target=None, signature=tuple(sig)))
  regs = kernel_regs(CUBIN, name)
  if regs is not None and regs != prg.regs_usage:
    print(f"[prg] {name}: regs {prg.regs_usage} -> {regs} (per-kernel EIATTR_REGCOUNT via symtab)")
    prg.qmd.write(register_count_v=regs)
    prg.max_threads = ((65536 // round_up(max(1, regs) * 32, 256)) // 4) * 4 * 32
  return prg

def set_ntid(prg, ls):
  prg.cbuf_0[0], prg.cbuf_0[1], prg.cbuf_0[2] = ls   # nvcc SASS reads NTID from c[0][0..8]

def mkbuf(data):
  b = dev.allocator.alloc(len(data), BufferSpec())
  dev.allocator._copyin(b, memoryview(data).cast("B"))
  return b

def readout(b, nbytes):
  mv = memoryview(bytearray(nbytes))
  dev.allocator._copyout(mv, b)
  return mv

def readf(b, nfloat):
  return np.frombuffer(readout(b, 4 * nfloat), dtype=np.float32).copy()

# ---------------- bench harness ----------------
def bench_batched(execs, inner=200, runs=5):
  """execs: list of (prg, bufs, vals, gs, ls); one 'iteration' = all execs once."""
  dev.synchronize()
  for _ in range(10):
    for prg, bufs, vals, gs, ls in execs:
      set_ntid(prg, ls)
      prg(*bufs, global_size=gs, local_size=ls, vals=vals, wait=True)
  best = 1e9
  for _ in range(runs):
    dev.synchronize()
    q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1)
    for _ in range(inner):
      for prg, bufs, vals, gs, ls in execs:
        q.exec(prg, prg.fill_kernargs(bufs, vals), gs, ls)
    t0 = time.perf_counter()
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize()
    best = min(best, (time.perf_counter() - t0) / inner)
  return best

def bench_eager(prg, bufs, vals, gs, ls, iters=30):
  set_ntid(prg, ls)
  dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    prg(*bufs, global_size=gs, local_size=ls, vals=vals, wait=True)
  return (time.perf_counter() - t0) / iters

# ---------------- shapes ----------------
SHAPES = [  # (name, N, K)  y[N] = W[N,K].x[K]
  ("gate/up [5120->17408]", 17408, 5120),
  ("down    [17408->5120]", 5120, 17408),
  ("qkv     [5120->10240]", 10240, 5120),
]

def main():
  crosscheck_fork()
  prg_v1 = mkprg("iq3_gemv", 5, 2)
  prg_sk = mkprg("iq3_gemv_sk", 5, 3)
  prg_rd = mkprg("iq3_reduce", 2, 2)
  print(f"[prg] v1 regs={prg_v1.regs_usage} shmem={prg_v1.shmem_usage} maxthr={prg_v1.max_threads} | "
        f"sk regs={prg_sk.regs_usage} | reduce regs={prg_rd.regs_usage}")

  gridb = mkbuf(np.ascontiguousarray(GRIDF).tobytes())
  esb = mkbuf(np.ascontiguousarray(ESIGN).tobytes())

  results = {}
  for name, N, K in SHAPES:
    nb = K // 256
    blocks = make_blocks(N * nb, 42)
    W = dequant_ref(blocks).reshape(N, K)                       # (N,K) f32
    xh = (np.random.default_rng(7).standard_normal(K) * 0.5).astype(np.float16)
    y_ref = W.astype(np.float32) @ xh.astype(np.float32)

    wb = mkbuf(np.ascontiguousarray(blocks).tobytes())
    xb = mkbuf(np.ascontiguousarray(xh).tobytes())
    yb = dev.allocator.alloc(4 * N, BufferSpec())
    dev.allocator._copyin(yb, memoryview(np.full(N, np.nan, np.float32).tobytes()).cast("B"))
    partb = dev.allocator.alloc(4 * N * 8, BufferSpec())

    def check(tag, y):
      denom = max(float(np.abs(y_ref).max()), 1e-30)
      rel = float(np.abs(y - y_ref).max()) / denom
      ok = rel < 1e-3 and not np.isnan(y).any()
      print(f"   [{tag}] relerr={rel:.2e} {'OK' if ok else 'FAIL'}")
      return ok

    all_ok = True
    # v1 configs
    for ls in (128, 256, 512):
      wpb = ls // 32
      gs = ((N + wpb - 1) // wpb, 1, 1)
      set_ntid(prg_v1, (ls, 1, 1))
      prg_v1(wb, xb, yb, gridb, esb, global_size=gs, local_size=(ls,1,1), vals=(N, K), wait=True)
      all_ok &= check(f"v1 ls={ls}", readf(yb, N))
    # v2 split-K configs
    for splits in (2, 4):
      assert nb % splits == 0
      wtot = N * splits
      for ls in (128, 256):
        wpb = ls // 32
        gs = ((wtot + wpb - 1) // wpb, 1, 1)
        set_ntid(prg_sk, (ls,1,1)); set_ntid(prg_rd, (256,1,1))
        prg_sk(wb, xb, partb, gridb, esb, global_size=gs, local_size=(ls,1,1), vals=(N, K, splits), wait=True)
        prg_rd(partb, yb, global_size=((N+255)//256,1,1), local_size=(256,1,1), vals=(N, splits), wait=True)
        all_ok &= check(f"sk s={splits} ls={ls}", readf(yb, N))
    if not all_ok: raise SystemExit(f"correctness FAIL for {name}")

    # bench
    rows = []
    raw_bytes = N * nb * 98 + K * 2 + N * 4
    for ls in (128, 256, 512):
      wpb = ls // 32
      gs = ((N + wpb - 1) // wpb, 1, 1)
      ms = bench_batched([(prg_v1, (wb, xb, yb, gridb, esb), (N, K), gs, (ls,1,1))])
      rows.append((f"v1 ls={ls}", ms))
    for splits in (2, 4):
      wtot = N * splits
      for ls in (128, 256):
        wpb = ls // 32
        gs = ((wtot + wpb - 1) // wpb, 1, 1)
        ms = bench_batched([
          (prg_sk, (wb, xb, partb, gridb, esb), (N, K, splits), gs, (ls,1,1)),
          (prg_rd, (partb, yb), (N, splits), ((N+255)//256,1,1), (256,1,1))])
        rows.append((f"sk s={splits} ls={ls}", ms))
    # eager per-call for best v1 (launch tax included, what JIT=2-style integration would pay)
    ls = 256; wpb = 8
    gs = ((N + wpb - 1) // wpb, 1, 1)
    ms_eager = bench_eager(prg_v1, (wb, xb, yb, gridb, esb), (N, K), gs, (ls,1,1))

    print(f"\n== {name}  (N={N}, K={K}, raw={raw_bytes/1e6:.1f}MB) ==")
    best = None
    for tag, ms in rows:
      mel = N * K / 1e6 / ms
      gbs = raw_bytes / 1e9 / ms
      print(f"   {tag:14s} {ms*1e3:8.3f} ms  {mel:8.0f} Melem/ms  {gbs:6.0f} GB/s(raw)")
      if best is None or ms < best[1]: best = (tag, ms, mel, gbs)
    mel_e = N * K / 1e6 / ms_eager
    print(f"   {'v1 eager':14s} {ms_eager*1e3:8.3f} ms  {mel_e:8.0f} Melem/ms   (incl. per-call launch tax)")
    print(f"   BEST: {best[0]} {best[1]*1e3:.3f} ms = {best[2]:.0f} Melem/ms, {best[3]:.0f} GB/s raw")
    results[name] = best
  print("\n== SUMMARY ==")
  for k, v in results.items(): print(f"  {k}: {v[0]} {v[1]*1e3:.3f} ms {v[2]:.0f} Melem/ms {v[3]:.0f} GB/s")

if __name__ == "__main__":
  main()
