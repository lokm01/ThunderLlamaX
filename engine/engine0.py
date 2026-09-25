# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""engine0 W1-a runtime: static raw NV buffers, GGUF slice loader (IQ3/Q5_K/Q8_0
stay PACKED — dequant in kernel), cubin loader, timed launch helper.
Raw allocator bufs (no Tensor refs needed). E4 launch pattern."""
import os, sys, struct, subprocess, mmap
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
from tinygrad import dtypes
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
GGUF = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"
dev = Device["NV"]

QUANT = {2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),12:(256,144),13:(256,176),
         14:(256,210),18:(256,98),21:(256,110),22:(256,82),23:(256,136),39:(32,17),41:(128,18)}
NATIVE = {0:4,1:2,24:1,25:2,26:4,27:8,28:8,30:2}

_TYPR = {0:(1,"c"),1:(1,"b"),2:(2,"H"),3:(2,"h"),4:(4,"I"),5:(4,"i"),6:(4,"f"),7:(1,"?"),10:(8,"Q"),11:(8,"q"),12:(8,"d")}

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
  data_start = (f.tell()+align-1)//align*align   # AFTER the tensor table (was before: 21KB shift bug)
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

def read_f32(info, data_start, reversed_layout=True):
  a = np.frombuffer(read_raw(info, data_start), dtype="<f4")
  return (a.reshape(dims[::-1]) if reversed_layout else a.reshape(dims)).copy() if False else a

class Bufs:
  def __init__(self):
    self.d = {}
    self._keep = []   # AGENTS lesson: async _copyin DMA + freed numpy = garbage; keep refs alive
  def alloc(self, name, nbytes):
    self.d[name] = dev.allocator.alloc(nbytes, BufferSpec()); return self.d[name]
  def up(self, name, arr):
    a = np.ascontiguousarray(arr)
    self._keep.append(a)
    b = self.alloc(name, a.nbytes)
    dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
  def poison(self, name, nbytes, dtype, val):
    if dtype == np.float16: a = np.full(nbytes//2, val, dtype=np.float16)
    else: a = np.full(nbytes//4, val, dtype=np.float32)
    return self.up(name, a)
  def down(self, name, shape, dtype=np.float32):
    b = self.d[name]; n = int(np.prod(shape))
    mv = memoryview(bytearray(n*np.dtype(dtype).itemsize)).cast("B")
    dev.allocator._copyout(mv, b)
    return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()
  # M1-A fixed-handle discipline: windowed upload/readback into EXISTING buffers
  # (P.up reallocs -> graph handles stale + orphan growth; NEVER in the request path).
  def win_up(self, name, off, arr):
    a = np.ascontiguousarray(arr)
    dev.allocator._copyin(self.d[name].offset(offset=off, size=a.nbytes), memoryview(a.data).cast("B"))
    self._keep.append(a)
  def down_at(self, name, off, n, dtype=np.int32):
    nb = n*np.dtype(dtype).itemsize
    mv = memoryview(bytearray(nb)).cast("B")
    dev.allocator._copyout(mv, self.d[name].offset(offset=off, size=nb))
    return np.frombuffer(mv, dtype=dtype).copy()

_pi = ("v", 0, dtypes.int32, ())
KERNELS = ["k0_norm","k1_q5","k1_iq3","k1_ab","k2_scan","k2b_z","k3a_oproj","k3m_hh","k3b_ffn","k3c_down"]
def load_progs():
  # PER-KERNEL cubins: the multi-kernel cubin mis-loads on this dext (k1_q5 faulted
  # from the 10-kernel cubin with byte-identical code that passes from its own cubin)
  out = {}
  for n in KERNELS:
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    out[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  return out

def iq3_grid_f32():
  from tinygrad.runtime.autogen.ggml_common import iq3xxs_grid
  vals = np.array([(w >> (8*i)) & 0xFF for w in iq3xxs_grid for i in range(4)], dtype=np.float32)
  assert vals.size == 1024
  return vals

class GDNBlockEngine:
  """One GDN block: packed weights on NV, static scratch, 8-kernel T=1 pipeline."""
  NBYTES_W = 36044800 + 12042240 + 33423360 + 3*34119680 + 2*983040  # quant + f32 alpha/beta
  def __init__(self, blk_idx=0, B=None):
    self.P = B if B is not None else Bufs()
    P = self.P
    ds, infos = parse_gguf()
    pre = f"blk.{blk_idx}."
    # packed quant weights
    self.w = {}
    self.w["qkv"] = P.up("w_qkv", np.frombuffer(read_raw(infos[pre+"attn_qkv.weight"], ds), dtype=np.uint8))
    self.w["gate"] = P.up("w_gate", np.frombuffer(read_raw(infos[pre+"attn_gate.weight"], ds), dtype=np.uint8))
    self.w["out"] = P.up("w_out", np.frombuffer(read_raw(infos[pre+"ssm_out.weight"], ds), dtype=np.uint8))
    self.w["fg"] = P.up("w_fg", np.frombuffer(read_raw(infos[pre+"ffn_gate.weight"], ds), dtype=np.uint8))
    self.w["fu"] = P.up("w_fu", np.frombuffer(read_raw(infos[pre+"ffn_up.weight"], ds), dtype=np.uint8))
    self.w["fd"] = P.up("w_fd", np.frombuffer(read_raw(infos[pre+"ffn_down.weight"], ds), dtype=np.uint8))
    # f32 tensors (disk dims reversed for 2D)
    # fork reshape(*reversed(dims)) = FLAT reinterpret of disk bytes, NOT a transpose
    aw = np.frombuffer(read_raw(infos[pre+"ssm_alpha.weight"], ds), dtype="<f4").reshape(48,5120)
    bw = np.frombuffer(read_raw(infos[pre+"ssm_beta.weight"], ds), dtype="<f4").reshape(48,5120)
    cw = np.frombuffer(read_raw(infos[pre+"ssm_conv1d.weight"], ds), dtype="<f4").reshape(10240,4)
    P.up("w_alpha", aw); P.up("w_beta", bw); P.up("conv_w", cw)
    P.up("dt_b", np.frombuffer(read_raw(infos[pre+"ssm_dt.bias"], ds), dtype="<f4"))
    P.up("ssm_a", np.frombuffer(read_raw(infos[pre+"ssm_a"], ds), dtype="<f4"))
    P.up("nw1", np.frombuffer(read_raw(infos[pre+"attn_norm.weight"], ds), dtype="<f4"))
    P.up("nw2", np.frombuffer(read_raw(infos[pre+"post_attention_norm.weight"], ds), dtype="<f4"))
    P.up("snw", np.frombuffer(read_raw(infos[pre+"ssm_norm.weight"], ds), dtype="<f4"))
    P.up("gridf", iq3_grid_f32())
    # scratch (static; fp32 unless noted)
    P.poison("x", 5120*4, np.float32, 7.7e31)
    P.poison("xh", 5120*2, np.float16, 7.7)
    P.poison("qkv_row", 10240*2, np.float16, 7.7)
    P.poison("gate_row", 6144*2, np.float16, 7.7)
    P.poison("alpharaw", 48*4, np.float32, 7.7e31)
    P.poison("betaraw", 48*4, np.float32, 7.7e31)
    P.poison("q", 48*128*4, np.float32, 7.7e31)
    P.poison("k", 48*128*4, np.float32, 7.7e31)
    P.poison("v", 48*128*4, np.float32, 7.7e31)
    P.poison("core", 6144*4, np.float32, 7.7e31)
    P.poison("z", 6144*2, np.float16, 7.7)
    P.poison("attn_out", 5120*2, np.float16, 7.7)
    P.poison("hh", 5120*4, np.float32, 7.7e31)
    P.poison("hhx", 5120*2, np.float16, 7.7)
    P.poison("gact", 17408*2, np.float16, 7.7)
    P.poison("y", 5120*4, np.float32, 7.7e31)
    P.poison("convA", 3*10240*4, np.float32, 7.7e31)
    P.poison("convB", 3*10240*4, np.float32, 7.7e31)
    P.poison("rec", 48*128*128*4, np.float32, 7.7e31)
    dev.synchronize()   # flush all pending upload DMAs
    self.progs = load_progs()

  def set_inputs(self, x=None, conv=None, rec=None):
    P = self.P
    if x is not None: P.up("x", x.astype(np.float32))
    if conv is not None: P.up("convA", conv.astype(np.float32))
    if rec is not None: P.up("rec", rec.astype(np.float32))
    dev.synchronize()

  def run(self, conv_src="convA", conv_dst="convB", wait=False):
    P, W, pr = self.P.d, self.w, self.progs
    LS = (256,1,1)
    pr["k0_norm"](P["x"], P["nw1"], P["xh"], global_size=(1,1,1), local_size=LS)
    pr["k1_q5"](W["qkv"], P["xh"], P["qkv_row"], global_size=(1280,1,1), local_size=LS)
    pr["k1_iq3"](W["gate"], P["gridf"], P["xh"], P["gate_row"], global_size=(768,1,1), local_size=LS)
    pr["k1_ab"](P["w_alpha"], P["w_beta"], P["xh"], P["alpharaw"], P["betaraw"], global_size=(12,1,1), local_size=LS)
    pr["k2_scan"](P[conv_src], P[conv_dst], P["qkv_row"], P["gate_row"], P["conv_w"],
                  P["dt_b"], P["ssm_a"], P["alpharaw"], P["betaraw"],
                  P["q"], P["k"], P["v"], P["rec"], P["core"],
                  global_size=(48,1,1), local_size=LS)
    pr["k2b_z"](P["core"], P["gate_row"], P["snw"], P["z"], global_size=(48,1,1), local_size=LS)
    pr["k3a_oproj"](W["out"], P["z"], P["attn_out"], global_size=(640,1,1), local_size=LS)
    pr["k3m_hh"](P["x"], P["attn_out"], P["nw2"], P["hh"], P["hhx"], global_size=(1,1,1), local_size=LS)
    pr["k3b_ffn"](W["fg"], W["fu"], P["gridf"], P["hhx"], P["gact"], global_size=(2176,1,1), local_size=LS)
    pr["k3c_down"](W["fd"], P["gridf"], P["gact"], P["hh"], P["y"],
                   global_size=(640,1,1), local_size=LS, wait=wait)

  # per-kernel launches for attribution (same arg sets)
  KP = {
    "k0_norm":      lambda s: (s.progs["k0_norm"], (s.P.d["x"], s.P.d["nw1"], s.P.d["xh"]), (1,)),
    "k1_q5":        lambda s: (s.progs["k1_q5"], (s.w["qkv"], s.P.d["xh"], s.P.d["qkv_row"]), (1280,)),
    "k1_iq3":       lambda s: (s.progs["k1_iq3"], (s.w["gate"], s.P.d["gridf"], s.P.d["xh"], s.P.d["gate_row"]), (768,)),
    "k1_ab":        lambda s: (s.progs["k1_ab"], (s.P.d["w_alpha"], s.P.d["w_beta"], s.P.d["xh"], s.P.d["alpharaw"], s.P.d["betaraw"]), (12,)),
    "k2_scan":      lambda s: (s.progs["k2_scan"], (s.P.d["convA"], s.P.d["convB"], s.P.d["qkv_row"], s.P.d["gate_row"], s.P.d["conv_w"], s.P.d["dt_b"], s.P.d["ssm_a"], s.P.d["alpharaw"], s.P.d["betaraw"], s.P.d["q"], s.P.d["k"], s.P.d["v"], s.P.d["rec"], s.P.d["core"]), (48,)),
    "k2b_z":        lambda s: (s.progs["k2b_z"], (s.P.d["core"], s.P.d["gate_row"], s.P.d["snw"], s.P.d["z"]), (48,)),
    "k3a_oproj":    lambda s: (s.progs["k3a_oproj"], (s.w["out"], s.P.d["z"], s.P.d["attn_out"]), (640,)),
    "k3m_hh":       lambda s: (s.progs["k3m_hh"], (s.P.d["x"], s.P.d["attn_out"], s.P.d["nw2"], s.P.d["hh"], s.P.d["hhx"]), (1,)),
    "k3b_ffn":      lambda s: (s.progs["k3b_ffn"], (s.w["fg"], s.w["fu"], s.P.d["gridf"], s.P.d["hhx"], s.P.d["gact"]), (2176,)),
    "k3c_down":     lambda s: (s.progs["k3c_down"], (s.w["fd"], s.P.d["gridf"], s.P.d["gact"], s.P.d["hh"], s.P.d["y"]), (640,)),
  }
  def launch_one(self, name, wait=False):
    prg, args, grid = self.KP[name](self)
    prg(*args, global_size=(grid[0],1,1), local_size=(256,1,1), wait=wait)
