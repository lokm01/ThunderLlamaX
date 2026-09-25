# A3d WIP — Q5_K GEMV substitution (NOT ACTIVE, rules disabled)

Target kernels (eager-paced @2k): r_320 GDN qkv Q5_K [5120->10240] 48x/tok 9.5ms
(1 thread/row latency-bound) + r_15520 lm_head Q5_K [5120->248320] 1x 4.87ms.
Warp-per-row replacements written (a3b.py: _match_q5head/_match_q5conv/_build_q5/_validate_q5)
but BLOCKED ON the element-mapping: tinygrad's fused dequant does NOT use the natural
ggml byte order. Kernel code matches gguf.py reference EXACTLY (numpy-verified) yet the
GENERATED kernel computes different values — the fused schedule pairs the permuted dequant
tensor with x in gguf.py stack/reshape order, not the raw element index.

CONFIRMED BY UNIT PROBES (a3d_probe_map2.py, mapdump.txt):
- scales: STANDARD sub = k>>5, sc/mn decode per gguf.py formulas (s0 -> 3060.625 etc. exact)
- qs bytes 0..63 low nibble: single-byte activation gives row sums ~48648-50560
  (NOT single elements) with per-byte deltas of 8/24/40; qs bytes 64..127: DEAD (zero
  contribution even with all sc=1) — the f-permutation reads some nibbles from elsewhere
  (source shows +48 AND +112 byte-group reads per thread = same byte index twice).
- qh probing inconclusive (fp32 sum noise); source analysis suggested byte=k&31, bit=(k>>5)&3
  but this did NOT validate either (relerr 0.55 after qs fix attempt).

NEXT SESSION: finish the mapping derivation from the r_320 source directly
(240 lines, /tmp/swarm/src_r_320*.cu — read the +48/+112/+56320/+56432 byte-group pairing
and the nibble selectors carefully; derive the f -> (byte, nibble, qhbit) table; encode into
_build_q5). Then: q5head + q5conv ~7ms combined expected (byte floors: GDN qkv 1.73GB/tok
= 4.7ms, lm_head 874MB = 2.4ms). ALSO: r_48_2_16_8_4_4_5_4_2_4 (48x/tok 164us 7.9ms) NOT
YET IDENTIFIED — dump via MTP_DUMP_MATCH="^r_48_2" (missed in two drain windows).

GOTCHAS hit today (all fixed in a3b.py): fp16 d/dmin bytes must be valid in ALL 20
superblocks of synthetic test rows (random bytes -> NaN d); +inf is 0x7f800000, -inf is
0xff800000; worker-pool compile path needs parent-side substitution; name guard must
include _a3d; NVRTC rejects unbalanced-paren macros (nvcc accepts them) — write plain
inline code. A3D_DBG_Q5 env gates: "1" = controlled data, "ref" = numpy three-way compare.

== UPDATE: A3d COMPLETE (fork 4ef43d4, project 3fcd306) ==
Mapping solved + kernels live: 2k 69.09ms/14.47 tok/s, 100k 95.21ms/10.50 tok/s.
Mapping: qs byte = 32*(k>>6)+(k&31), nibble = (k>>5)&1; qh byte = k&31, BIT = k>>5;
scales standard sub=k>>5. Perf rules learned: consecutive-W-bytes per lane (never stride-8
element order), sc/mn as registers (no dynamically indexed local arrays), passthrough
inside warp<N, poison out-buffer between orig/new in validation.

== NEXT TARGET IDENTIFIED: r_48_2_16_8_4_4_5_4_2_4 (7.97ms eager, 48x/tok @166us) ==
GDN in_proj IQ3_XXS GEMV: grid(2,48)x(16,8,4). Sig: (half* out_6144, float* vec_6144,
float* s48_48, uchar* aux_512, float* x_5120, float* rms_1, uchar* nw_20480,
uchar* W_12042240, float* gridlut_1024). W = ALL 48 blocks packed (250880B each =
64 rows x 1960B = IQ3 20-block rows), K=5120, 6144 output rows. Launched 48x/token,
each launch reads the FULL 12MB at ~72GB/s (latency-bound 96-CTA grid) = 578MB/token!
Warp-per-row (6144 warps) should hit ~27us/launch -> ~6.7ms/token savings.
Epilogue (from tail of /tmp/swarm3/src_r_48_2*.cu): out[o] = (half)( vec[o] * (1/s48[g1])
* fp32(nwbytes) * (half)(h5 * hrcp((half)1 + hexp2(h5 * (half)-1.4423828125f)) ) ) where
h5 = (half)(dot result). NOTE the half intrinsics + TRUNCATED log2e constant. Signs INLINE
(no signs tensor in sig -> use MTP_A3_INLINE_SIGNS graph). Need: new a3b variant with
half* y + this epilogue + row/W-offset derivation from the middle of the source file
(row r <-> (g0,g1,l0,z) mapping NOT yet fully derived - read lines 30-130 of the dump).
