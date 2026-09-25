# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ["MTP_A3_OVERRIDE"] = "~/tinygrad-metal/a3b/override.json"
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad.device import Device, BufferSpec, TinyELF
from tinygrad.dtype import dtypes as _dt
from tinygrad.codegen import a3b
from tinygrad.helpers import to_function_name

dev = Device["NV"]
renderer = dev.renderer
N, K = 10240, 5120

# --- build new kernel source
sig_c = "float* data0_40960, float* data1_30720, float* data2_5120, float* data3_1, unsigned char* data4_20480, unsigned char* data5_36044800, const int data6_"
args_c = [("float*", "data0", 40960), ("float*", "data1", 30720), ("float*", "data2", 5120), ("float*", "data3", 1),
          ("unsigned char*", "data4", 20480), ("unsigned char*", "data5", 36044800), ("const int", "data6", 0)]
roles = a3b._match_q5conv(args_c, "half half")
new_src, grid, local = a3b._build_q5(roles, {"kind": "q5conv", "wpc": 4}, to_function_name("t_iso_q5conv"), sig_c)

orig_src = open("/tmp/swarm/src_r_320_16_2_4_2_20_8_2_2_4.cu").read()

sig = (("data0", 0, _dt.float, (40960,)), ("data1", 1, _dt.float, (30720,)), ("data2", 2, _dt.float, (5120,)),
       ("data3", 3, _dt.float, (1,)), ("data4", 4, _dt.uint8, (20480,)), ("data5", 5, _dt.uint8, (36044800,)))
lib_n = renderer.compiler.compile_cached(new_src)
lib_o = renderer.compiler.compile_cached(orig_src)
rt_o = dev.runtime(TinyELF(lib_o, "r320_o", renderer.target, sig))
rt_n = dev.runtime(TinyELF(lib_n, "r320_n", renderer.target, sig))

# --- controlled data: W all-zero except row 0 superblock 0 = known values
rng = np.random.default_rng(5)
W = np.zeros((N * 3520,), dtype=np.uint8)
# row 0, sb 0: d=1.0, dmin=0.0, scales sc[0]=8 else 0, qs byte j=0 -> q=0, qh=0
W[0:2] = np.array([0x00, 0x3C], dtype=np.uint8)  # 1.0 fp16 = 0x3C00
W[4] = 8  # sc[0]=8
# make element q=1 for j=0: qs[0] low nibble = 1
W[48] = 1
x = np.ones((K,), dtype=np.float32)
nw = np.ones((K,), dtype=np.float32)
inn = rng.uniform(-1, 1, 3 * N).astype(np.float32)

b_out = dev.allocator.alloc(40960 * 4, BufferSpec())
b_in = dev.allocator.alloc(30720 * 4, BufferSpec())
b_x = dev.allocator.alloc(5120 * 4, BufferSpec())
b_s = dev.allocator.alloc(4, BufferSpec())
b_nw = dev.allocator.alloc(20480, BufferSpec())
b_W = dev.allocator.alloc(N * 3520, BufferSpec())
bufs = (b_out, b_in, b_x, b_s, b_nw, b_W)
dev.allocator._copyin(b_out, memoryview(bytearray(40960 * 4)).cast("B"))
dev.allocator._copyin(b_in, memoryview(inn.tobytes()).cast("B"))
dev.allocator._copyin(b_x, memoryview(x.tobytes()).cast("B"))
dev.allocator._copyin(b_s, memoryview(np.array([2.0], dtype=np.float32).tobytes()).cast("B"))
dev.allocator._copyin(b_nw, memoryview(nw.tobytes()).cast("B"))
dev.allocator._copyin(b_W, memoryview(W.tobytes()).cast("B"))

for nm, rt, gd, ld in (("orig", rt_o, (320, 1, 1), (16, 2, 1)), ("new", rt_n, grid, local)):
    dev.allocator._copyin(b_out, memoryview(bytearray(40960 * 4)).cast("B"))
    rt(*bufs, global_size=gd, local_size=ld, vals=(7,), wait=True)
    mv = memoryview(bytearray(40960 * 4)); dev.allocator._copyout(mv, b_out)
    y = np.frombuffer(mv, dtype=np.float32)
    print(nm, "out[30720:30726] =", y[30720:30726], "finite:", np.isfinite(y).all())
# expected: row0 = sum over sb0 j=0..255: x*inv_s*nw*(d*sc0*q - 0) : only j=0 has q=1 -> 1*0.5*1*(1*8*1) = 4.0; rows else 0
