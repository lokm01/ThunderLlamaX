# A3a — hand CUDA IQ3_XXS dequant-GEMV microbenchmark (2026-08-28)

**VERDICT: GO for A3 integration.** Hand kernel hits 780-885 Melem/ms (target ≥400,
tinygrad 119-220): 0.101 ms for the 34.2MB [5120→17408] GEMV = 340 GB/s raw
(76% of the 447 GB/s peak; byte floor 76 µs). Loader for hand-made cubins through the
fork's NV runtime is BUILT and validated (smoke.py + a3a_bench.py).

## Results (batched QMD-chained launches, min of 5×200 iters, warm 10; relerr vs fork-exact ref ~1e-7)

| shape (y[N]=W[N,K]·x[K]) | best config | ms | Melem/ms | GB/s raw | tinygrad (P1e) |
|---|---|---|---|---|---|
| gate/up [5120→17408] | v1 warp-per-row, ls=128 | 0.101 | 885 | 340 | 0.748 ms isolated (119) |
| down [17408→5120]    | v1 ls=128                | 0.114 | 779 | 299 | — |
| qkv [5120→10240]     | v1 ls=256                | 0.064 | 825 | 316 | — |

Split-K (v2) LOSES on all shapes (extra pass + launch); warp-per-row is enough
parallelism even at N=5120 (640 CTAs). Eager per-call (wait=True, incl. launch tax):
0.21-0.28 ms — still 2.7-3.5× faster than tinygrad isolated.

Projected in-model: per GDN block gate+up+down+qkv ≈ 0.38 ms (+o_proj ~0.03) →
~19-20 ms/token for the 48 blocks vs ~60 ms current GEMV pool → decode 99→~60 ms
(~16-17 tok/s @2k) before MTP. CAVEAT: per-call eager launch tax is ~165 µs/kernel —
at 3-4 kernels × 48 blocks that is 24-30 ms of pure tax; A3 integration MUST launch
these inside the JIT=1 graph family (HCQGraph exec chain = 6.7 µs/kernel measured).

## THE LOADER RECIPE (for the A3 integration agent)

```python
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_nv import NVProgram
dev = Device["NV"]
cubin = open("iq3.cubin","rb").read()
# signature: buf placeholder per pointer arg + (int32) entry per scalar arg
sig = [(None,0,dtypes.uint8,())]*nbuf + [(None,0,dtypes.int32,())]*nvals
prg = NVProgram(dev, TinyELF(lib=cubin, name="kernel_name", target=None, signature=tuple(sig)))
buf = dev.allocator.alloc(nbytes, BufferSpec()); dev.allocator._copyin(buf, mv)
out = memoryview(bytearray(n)); dev.allocator._copyout(out, buf)
prg(*bufs, global_size=(ctas,1,1), local_size=(threads,1,1), vals=(...), wait=True)
```
C kernel signature convention: ALL pointer params first (packed as u64 at c[0][0x160],
declaration order), then int params (natural alignment) — matches CLikeArgsState.

### Gotcha 1 — NTID: nvcc SASS reads blockDim from c[0][0x0..0x8]
The fork's cbuf_0 leaves those words ZERO (tinygrad's own kernels bake block size into
the code at compile time). Symptom: every CTA computes i = blockIdx.x*0 + threadIdx.x —
only the first CTA's range gets written, rest of output = garbage/stale.
FIX before every launch (cbuf_0 is copied into kernargs at fill_kernargs time):
```python
prg.cbuf_0[0], prg.cbuf_0[1], prg.cbuf_0[2] = (lx, ly, lz)
```

### Gotcha 2 — per-kernel register count in MULTI-kernel cubins
nvcc emits EIATTR_REGCOUNT (0x2f) entries in the MODULE-level `.nv.info`, keyed by ELF
symbol index: (symidx, regs). The fork takes the LAST entry for every program
(ops_nv.py ~line 285). With 3 kernels (39/40/48 regs) all programs got 39 → SM
"Out Of Range Register" device fault (GSP log: `Out Of Range Register` warp exception).
FIX: resolve kernel name → symtab index (.symtab Elf64_Sym[24] + .strtab) and patch
`prg.qmd.write(register_count_v=regs)` (see kernel_regs() in a3a_bench.py). Single-kernel
cubins don't hit this. Also: `.nv.constant0.<name>` sections — the fork's regex assigns
constbufs[0] from the LAST one; harmless (exec overrides the address to kernargs) but the
declared size can be a different kernel's.

### Gotcha 3 — timing
- per-call `prg(..., wait=True)`: ~165 µs python+queue overhead per kernel.
- batched: build one `NVComputeQueue`, loop `q.exec(prg, prg.fill_kernargs(bufs, vals), gs, ls)`
  then one signal+submit → 6.7 µs/kernel measured (QMD dependent-chaining).
- dext signal timestamps are µs; don't trust HCQ wait=True returned durations (1000× off).

## Kernel design (iq3_gemv.cu)
- v1 warp-per-row: thread t handles elements 8t..8t+7 of each 98-byte block — exactly one
  (scale-word g=t>>2, 7-bit sign index slot i=t&3) sign slot, so ONE even_signs byte gives
  all 8 signs; qs bytes 2t/2t+1 → two float4 grid lookups (grid as float[1024], no I2F);
  x as one uint4 (16B) of halves; fp32 accumulate; 5-step shuffle reduce.
- Math ported EXACTLY from fork gguf.py case 18 (validated bit-exact vs
  ggml_data_to_tensor on random blocks, plus GPU-vs-numpy relerr ~1e-7):
  db = d*(ls+0.5)*0.5 per 32-elem group; grid bytes used RAW (4..62, NO -32);
  sign bit==0 → +1; even_signs[i] = i|(0x80 if popcount(i) odd).
- Tables (grid 4KB float, even_signs 128B) passed as ordinary global buffers (hot in L1).
- x fp16 (matches model activations), y fp32.

Files: iq3_gemv.cu (kernels), a3a_bench.py (loader+validation+bench), smoke.cu/py
(minimal loader proof), build.sh. Regenerate cubins with build.sh (nvcc shim needs
ABSOLUTE paths under $HOME — container only mounts $HOME and /var/folders).
