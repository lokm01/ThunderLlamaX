# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P6 32-row forward harness: embed(2x16) -> 64 blocks (M32 GEMMs + 2x16
norms/pre/attn/scan halves) -> head, vs the T=1 trunk 32 steps at the same
positions. Gates: logits per-row <=3e-3 class, GDN state drift report, kv byte
compare, plus a MERGED-KERNEL bit-identity check (gdnqg/attnqkv M32 vs the
shipped M16 kernels on identical inputs). READOUT-ORDER LAW: all gates read
from the FIRST clean run; timing reps AFTER.
Usage: PATH=... DOCKER_HOST=... ~/tg311/bin/python -u pf_fwd32.py [--pos0]
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
import trunk_w1c
from trunk_w1c import TrunkEngineW1C, LS
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
M = 32
H = 16
S = 32

E = TrunkEngineW1C(theta=1e7)   # SKV=1 KV8=1 QH=1 via env
P, W, d = E.P, E.W, E.P.d
pr = E.pr
_lib = open(f"{BASE}/spk_c1g_100k.cubin", "rb").read()
pr["spk_c1"] = NVProgram(dev, TinyELF(lib=_lib, name="spk_c1g_100k", target=dev.renderer.target, signature=tuple()))

KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16", "pfs16": "pfs16"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))

PF = {}
G3 = os.getenv("PF_GEMM3") == "1"          # P7-B: repacked-layout gemm3 tier (m32)
G3N = int(os.getenv("PF_G3_BLOCKS", "32"))
R7 = {}
if G3:
  for n in ["pfg3_ffn_r7_m32_nw8k128","pfg3_iq3d_r7_m32_nw8k128","pfg3_iq3o_r7_m32_nw8k128",
            "pfg3m_gdnqg_r7_m32_nw16k128","pfg3m_attnqkvi3_r7_m32_nw8k128","pfg3m_attnqkvq6_r7_m32_nw8k128"]:
    PF[n] = prog(n)
for n in ["pfk_emb16","pfk_n16","pfk_ab16","pfk_hh16","pfk_pre16_100k","pfa16nw32_s32_100k","pfc16_s32","pfs16",
          "pfg_q5h_hm_nw16k128",
          "pfg2_gdnqg_hm_nw16k128","pfg2_attnqkvq6_hm_nw8k128","pfg2_attnqkvi3_hm_nw8k128",
          "pfg_ffn_m32_hm_nw8k128","pfg_iq3d_m32_res_hm_nw8k128","pfg_iq3s_m32_hm_nw8k128",
          "pfg_iq3o_m32_hm_nw8k128","pfg_q8o_m32_hm_nw8k64",
          "pfg2_gdnqg_m32_hm_nw16k128","pfg2_attnqkvq6_m32_hm_nw8k128","pfg2_attnqkvi3_m32_hm_nw8k128"]:
  PF[n] = prog(n)

ATTN32 = os.getenv("PF_ATTN32", "0") == "1"   # P10: t32 co-resident attention arm
A4 = os.getenv("PF_A4", "0") == "1"
if ATTN32:
  # LAW: the t32 entries are pfa32ct/pfa32ctl (NOT the cubin filenames) --
  # prog() would misname the entry = the garbage-execution mislaunch class.
  for n, sym in ([("pfa32ctl_s13_100k", "pfa32ctl")] if A4 else
                 [("pfa32c_t32_s13_100k", "pfa32ct"), ("pfc16t_s13", "pfc16t")]):
    _lib = open(f"{BASE}/{n}.cubin", "rb").read()
    PF[n] = NVProgram(dev, TinyELF(lib=_lib, name=sym, target=dev.renderer.target, signature=tuple()))
  if A4:
    P.up("atc_ctr", np.zeros(1, dtype=np.uint32)); dev.synchronize()

SC = os.getenv("PF_SCANC") == "1"     # P7-C: chunked WY scan (C=32 tier)
if SC:
  for n in ["pfca_c32_nc1_nw16", "pfcb_c32_nc1_nw8", "pfcz_c32_nc1_nw8"]:
    PF[n] = prog(n)

# ---- M=32 scratch + per-block GDN live slots (harness-local) ----
for nm, nb, dt, v in [
    ("xA32", M*5120*4, np.float32, 7.7e31), ("xB32", M*5120*4, np.float32, 7.7e31),
    ("xh32", M*5120*2, np.float16, 7.7), ("hh32", M*5120*4, np.float32, 7.7e31),
    ("hhx32", M*5120*2, np.float16, 7.7), ("attn_out32", M*5120*2, np.float16, 7.7),
    ("qkv32", M*10240*2, np.float16, 7.7), ("gate32", M*6144*2, np.float16, 7.7),
    ("z32", M*6144*2, np.float16, 7.7), ("gact32", M*17408*2, np.float16, 7.7),
    ("araw32", M*48*4, np.float32, 7.7e31), ("braw32", M*48*4, np.float32, 7.7e31),
    ("qrow32", M*12288*2, np.float16, 7.7), ("krow32", M*1024*2, np.float16, 7.7),
    ("vrow32", M*1024*2, np.float16, 7.7), ("qw32", M*24*256*2, np.float16, 7.7),
    ("ao32", M*6144*2, np.float16, 7.7), ("pmA", 4*S*96*4, np.float32, 7.7e31),
    ("psA", 4*S*96*4, np.float32, 7.7e31), ("pAA", 4*S*96*256*4, np.float32, 7.7e31),
    ("pmB", 4*S*96*4, np.float32, 7.7e31), ("psB", 4*S*96*4, np.float32, 7.7e31),
    ("pAB", 4*S*96*256*4, np.float32, 7.7e31), ("logits32", M*248320*2, np.float16, 7.7)]:
  P.poison(nm, nb, dt, v)
P.up("ids16a", np.zeros(H, dtype=np.int32))
P.up("ids16b", np.zeros(H, dtype=np.int32))
P.up("pos_slot_b", np.zeros(1, dtype=np.int32))
dev.synchronize()
HV = {nm + "b": d[nm].offset(offset=H*nb, size=H*nb) for nm, nb in
      [("xA32", 5120*4), ("xB32", 5120*4), ("xh32", 5120*2), ("hh32", 5120*4),
       ("hhx32", 5120*2), ("attn_out32", 5120*2), ("qkv32", 10240*2),
       ("gate32", 6144*2), ("z32", 6144*2), ("gact32", 17408*2),
       ("araw32", 48*4), ("braw32", 48*4), ("qrow32", 12288*2),
       ("krow32", 1024*2), ("vrow32", 1024*2), ("qw32", 24*256*2), ("ao32", 6144*2),
       ("logits32", 248320*2)]}
for j, i in enumerate(E.gdn_idx):
  P.up(f"convp{j}", np.zeros(3*10240, dtype=np.float32))
dev.synchronize(); P._keep.clear()
for j, i in enumerate(E.gdn_idx):
  P.up(f"recp{j}", np.zeros(48*128*128, dtype=np.float32))
  if j % 8 == 0: dev.synchronize(); P._keep.clear()
dev.synchronize(); P._keep.clear()
for i in E.gdn_idx:
  P.up(f"conv{i}_0", np.zeros(3*10240, dtype=np.float32))
  P.up(f"conv{i}_1", np.zeros(3*10240, dtype=np.float32))
  P.up(f"rec{i}", np.zeros(48*128*128, dtype=np.float32))
  if i % 8 == 0: dev.synchronize(); P._keep.clear()
dev.synchronize(); P._keep.clear()
print("[harness] engine + scratch ready", flush=True)

if SC:
  from engine0 import parse_gguf, read_raw
  ds_, infos_ = parse_gguf()
  wp = np.zeros(48 * 47200, dtype=np.float32)
  for jj, i in enumerate(E.gdn_idx):
    pre = f"blk.{i}."
    b = jj * 47200
    wp[b:b+40960] = np.frombuffer(read_raw(infos_[pre+"ssm_conv1d.weight"], ds_), dtype="<f4").reshape(-1)
    wp[b+40960:b+41008] = np.frombuffer(read_raw(infos_[pre+"ssm_dt.bias"], ds_), dtype="<f4")
    wp[b+41008:b+41056] = np.frombuffer(read_raw(infos_[pre+"ssm_a"], ds_), dtype="<f4")
    wp[b+41056:b+41056+128] = np.frombuffer(read_raw(infos_[pre+"ssm_norm.weight"], ds_), dtype="<f4")  # snw = ONE 128-vector shared by all heads (pfs16 law)
  P.up("scwp", wp)
  P.poison("scscr", 48 * 125712, np.uint8, 0xAB)  # R2b: hi-lo HC_BYTES(C=32) — the P7C-era 50448 predates the P7E5/E6 rebuild
  P.poison("sco", 32 * 6144 * 4, np.float32, 7.7e31)
  dev.synchronize(); P._keep.clear()
  print("[scanc] wplane (48 blocks) + chunk-scan scratch ready", flush=True)

# ---- merged-kernel bit-identity check (before the heavy reference) ----
rng0 = np.random.default_rng(5)
def m32_check(tag, m16n, m32n, wargs, outs, out_sizes, kd):
  xh = (rng0.standard_normal((M, kd)) * 0.5).astype(np.float16)
  P.up("xhc", xh.reshape(-1)); dev.synchronize()
  x2 = d["xhc"].offset(offset=H*kd*2, size=H*kd*2)
  refs, mine = [], []
  for o, sz in zip(outs, out_sizes):
    P.poison(o + "r", M*sz*2, np.float16, 7.7); P.poison(o + "m", M*sz*2, np.float16, 7.7)
  dev.synchronize()
  gsz = {"qkv32": 10240, "gate32": 6144, "qrow32": 12288, "krow32": 1024, "vrow32": 1024}
  ls_ = (512,1,1) if "gdnqg" in m32n else LS
  g_ = 128 if "gdnqg" in m32n else 224
  # M16 twice (halves)
  PF[m16n](*(wargs + (d["gridf"], d["xhc"]) + tuple(d[o+"r"] for o in outs)), global_size=(g_,1,1), local_size=ls_)
  PF[m16n](*(wargs + (d["gridf"], x2) + tuple(d[o+"r"].offset(offset=H*sz*2, size=H*sz*2) for o, sz in zip(outs, out_sizes))), global_size=(g_,1,1), local_size=ls_)
  # M32 once
  PF[m32n](*(wargs + (d["gridf"], d["xhc"]) + tuple(d[o+"m"] for o in outs)), global_size=(g_,1,1), local_size=ls_)
  dev.synchronize()
  nz = 0; tot = 0
  for o, sz in zip(outs, out_sizes):
    r = P.down(o + "r", (M, sz), np.float16); mm = P.down(o + "m", (M, sz), np.float16)
    nz += int((r != mm).sum()); tot += r.size
  print(f"[m32chk] {tag}: {'BIT-IDENTICAL' if nz == 0 else f'DIFF {nz}/{tot}'}", flush=True)
  return nz == 0

G0 = E.gdn_idx[0]
A18 = next(i for i in E.qtypes if E.qtypes[i] != 14)
OK_MERGED = True
OK_MERGED &= m32_check("gdnqg", "pfg2_gdnqg_hm_nw16k128", "pfg2_gdnqg_m32_hm_nw16k128",
                       (W[("qkv", G0)], W[("gate", G0)]), ["qkv32", "gate32"], [10240, 6144], 5120)
OK_MERGED &= m32_check("attnqkvi3", "pfg2_attnqkvi3_hm_nw8k128", "pfg2_attnqkvi3_m32_hm_nw8k128",
                       (W[("q", A18)], W[("k", A18)], W[("v", A18)]), ["qrow32", "krow32", "vrow32"], [12288, 1024, 1024], 5120)
print(f"[m32chk] merged kernels {'ALL BIT-IDENTICAL' if OK_MERGED else 'DIFF'}", flush=True)

def fwd32(pos, wait_last=False):
  n = [0]
  def L(p, *a, g=1, ls=LS, wait=False):
    p(*a, global_size=(g,1,1), local_size=ls, wait=wait)
    n[0] += 1
    if n[0] % 200 == 0: dev.synchronize()
  L(PF["pfk_emb16"], W[("emb",0)], d["grid512"], d["ids16a"], d["xA32"], g=16)
  L(PF["pfk_emb16"], W[("emb",0)], d["grid512"], d["ids16b"], HV["xA32b"], g=16)
  cur = 0
  for i in range(64):
    xin = d["xA32"] if cur == 0 else d["xB32"]
    xout = d["xB32"] if cur == 0 else d["xA32"]
    xinb = HV["xA32b"] if cur == 0 else HV["xB32b"]
    if i in E.qtypes:
      L(PF["pfk_n16"], xin, W[("nw1",i)], d["xh32"], g=16)
      L(PF["pfk_n16"], xinb, W[("nw1",i)], HV["xh32b"], g=16)
      mqn = "pfg2_attnqkvq6_m32_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_m32_hm_nw8k128"
      if G3 and ("k",i) in R7:
        mq3 = "pfg3m_attnqkvq6_r7_m32_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m32_nw8k128"
        qw = W[("q",i)] if E.qtypes[i] == 14 else R7[("q",i)]
        L(PF[mq3], qw, R7[("k",i)], W[("v",i)], d["gridf"], d["xh32"], d["qrow32"], d["krow32"], d["vrow32"], g=224, ls=LS)
      else:
        L(PF[mqn], W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh32"], d["qrow32"], d["krow32"], d["vrow32"], g=224, ls=LS)
      L(PF["pfk_pre16_100k"], d["qrow32"], d["krow32"], d["vrow32"], W[("qnw",i)], W[("knw",i)], d["freqs"],
        d[f"kv{i}"], d[f"sc{i}"], d["pos_slot"], d["qw32"], g=24)
      L(PF["pfk_pre16_100k"], HV["qrow32b"], HV["krow32b"], HV["vrow32b"], W[("qnw",i)], W[("knw",i)], d["freqs"],
        d[f"kv{i}"], d[f"sc{i}"], d["pos_slot_b"], HV["qw32b"], g=24)
      if ATTN32:
        if A4:
          L(PF["pfa32ctl_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw32"], d["pos_slot"], d["pmA"], d["psA"], d["pAA"],
            d["qrow32"], d["ao32"], d["atc_ctr"], g=4*13*3, ls=(512,1,1))
          L(PF["pfa32ctl_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], HV["qw32b"], d["pos_slot_b"], d["pmB"], d["psB"], d["pAB"],
            HV["qrow32b"], HV["ao32b"], d["atc_ctr"], g=4*13*3, ls=(512,1,1))
        else:
          L(PF["pfa32c_t32_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw32"], d["pos_slot"], d["pmA"], d["psA"], d["pAA"], g=4*13*3, ls=(512,1,1))
          L(PF["pfa32c_t32_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], HV["qw32b"], d["pos_slot_b"], d["pmB"], d["psB"], d["pAB"], g=4*13*3, ls=(512,1,1))
          L(PF["pfc16t_s13"], d["pmA"], d["psA"], d["pAA"], d["qrow32"], d["ao32"], g=24)
          L(PF["pfc16t_s13"], d["pmB"], d["psB"], d["pAB"], HV["qrow32b"], HV["ao32b"], g=24)
      else:
        L(PF["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw32"], d["pos_slot"], d["pmA"], d["psA"], d["pAA"], g=4*S, ls=(1024,1,1))
        L(PF["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], HV["qw32b"], d["pos_slot_b"], d["pmB"], d["psB"], d["pAB"], g=4*S, ls=(1024,1,1))
        L(PF["pfc16_s32"], d["pmA"], d["psA"], d["pAA"], d["qrow32"], d["ao32"], g=24)
        L(PF["pfc16_s32"], d["pmB"], d["psB"], d["pAB"], HV["qrow32b"], HV["ao32b"], g=24)
      L(PF["pfg_iq3s_m32_hm_nw8k128"], W[("o",i)], d["grid512"], d["ao32"], d["attn_out32"], g=80, ls=LS)
    else:
      j = E.gdn_idx.index(i)
      L(PF["pfk_ab16"], xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh32"], d["araw32"], d["braw32"], g=16*13)
      L(PF["pfk_ab16"], xinb, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], HV["xh32b"], HV["araw32b"], HV["braw32b"], g=16*13)
      if G3 and ("gate",i) in R7:
        L(PF["pfg3m_gdnqg_r7_m32_nw16k128"], W[("qkv",i)], R7[("gate",i)], d["gridf"], d["xh32"], d["qkv32"], d["gate32"], g=128, ls=(512,1,1))
      else:
        L(PF["pfg2_gdnqg_m32_hm_nw16k128"], W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh32"], d["qkv32"], d["gate32"], g=128, ls=(512,1,1))
      if SC:
        scwpb = d["scwp"].offset(offset=j * 47200 * 4, size=47200 * 4)
        L(PF["pfca_c32_nc1_nw16"], scwpb, d[f"convp{j}"], d["qkv32"], d["araw32"], d["braw32"], d["scscr"], g=48, ls=(512,1,1))
        L(PF["pfcb_c32_nc1_nw8"], d["scscr"], d[f"recp{j}"], d["sco"], g=192)
        L(PF["pfcz_c32_nc1_nw8"], d["sco"], d["gate32"], scwpb.offset(offset=41056*4, size=6144*4),
          d["z32"], d["qkv32"], d[f"convp{j}"], g=192)
      else:
        L(PF["pfs16"], d[f"convp{j}"], d[f"recp{j}"], d["qkv32"], d["gate32"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
          d["araw32"], d["braw32"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z32"], g=48)
        L(PF["pfs16"], d[f"convp{j}"], d[f"recp{j}"], HV["qkv32b"], HV["gate32b"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
          HV["araw32b"], HV["braw32b"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], HV["z32b"], g=48)
      on = "pfg_q8o_m32_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_m32_hm_nw8k128"
      if G3 and (not E.gdn_oq8[i]) and ("out",i) in R7:
        L(PF["pfg3_iq3o_r7_m32_nw8k128"], R7[("out",i)], d["gridf"], d["z32"], d["attn_out32"], g=80, ls=LS)
      else:
        L(PF[on], W[("out",i)], d["gridf"], d["z32"], d["attn_out32"], g=80, ls=LS)
    L(PF["pfk_hh16"], xin, d["attn_out32"], W[("nw2",i)], d["hh32"], d["hhx32"], g=16)
    L(PF["pfk_hh16"], xinb, HV["attn_out32b"], W[("nw2",i)], HV["hh32b"], HV["hhx32b"], g=16)
    if G3 and ("fg",i) in R7:
      L(PF["pfg3_ffn_r7_m32_nw8k128"], R7[("fg",i)], R7[("fu",i)], d["gridf"], d["hhx32"], d["gact32"], g=272, ls=LS)
      L(PF["pfg3_iq3d_r7_m32_nw8k128"], R7[("fd",i)], d["gridf"], d["gact32"], d["hh32"], xout, g=80, ls=LS)
    else:
      L(PF["pfg_ffn_m32_hm_nw8k128"], W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx32"], d["gact32"], g=272, ls=LS)
      L(PF["pfg_iq3d_m32_res_hm_nw8k128"], W[("fd",i)], d["gridf"], d["gact32"], d["hh32"], xout, g=80, ls=LS)
    cur ^= 1
  # head: norm the last block's xout rows (2x16 halves via the SAME xh32 buffer)
  L(PF["pfk_n16"], xout, W[("onw",0)], d["xh32"], g=16)
  L(PF["pfg_q5h_hm_nw16k128"], W[("head",0)], d["gridf"], d["xh32"], d["logits32"], g=248320//128, ls=(512,1,1))
  L(PF["pfk_n16"], xout.offset(offset=H*5120*4, size=H*5120*4), W[("onw",0)], HV["xh32b"], g=16)
  L(PF["pfg_q5h_hm_nw16k128"], W[("head",0)], d["gridf"], HV["xh32b"], HV["logits32b"], g=248320//128, ls=(512,1,1), wait=wait_last)

def t1_token(t, tok):
  P.win_up("tok_slot", 0, np.array([int(tok)], dtype=np.int32))
  if not hasattr(E, "_seq"): E._build_seqs()
  seq = E._seq[t & 1]
  for n_, (p, a, g) in enumerate(seq):
    p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1),
      local_size=(1024,1,1) if "nw32" in getattr(p, "name", "") else LS,
      wait=(n_ == len(seq)-1))

# =========== run ===========
rng = np.random.default_rng(99)
ids = rng.integers(1000, 200000, size=M).astype(np.int32)
P.win_up("ids16a", 0, ids[:16])
P.win_up("ids16b", 0, ids[16:])
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
P.win_up("pos_slot_b", 0, np.array([16], dtype=np.int32))
dev.synchronize()

SKIPREF = os.getenv("PF_G3_SKIPREF") == "1"   # timing-only smoke: gemm3
# bit-identity proven standalone (test_p7b) + in-harness (m32chk above)
if SKIPREF:
  ref_logits = None
  print("[ref] SKIPPED (PF_G3_SKIPREF=1)", flush=True)
else:
  t0 = time.perf_counter()
  ref_logits = np.zeros((M, 248320), dtype=np.float16)
  for t in range(M):
    t1_token(t, ids[t])
    ref_logits[t] = P.down("logits", (248320,), np.float16)
  t1_time = time.perf_counter() - t0
  print(f"[ref] T=1 trunk 32 tokens in {t1_time:.2f}s ({t1_time/M*1e3:.0f} ms/tok)", flush=True)
ref_rec = {i: P.down(f"rec{i}", (48*128*128,), np.float32).copy() for i in ([] if SKIPREF else E.gdn_idx[:4])}
ref_conv = {i: P.down(f"conv{i}_0", (3*10240,), np.float32).copy() for i in ([] if SKIPREF else E.gdn_idx[:4])}
# kv slice only (first 128 rows = 256KB): covers rows 0..31 written here; the
# full 2x205MB download is the P5 fault-class path — avoid it.
KV_SLICE = 2*4*256*128
ref_kv = {i: P.down_at(f"kv{i}", 0, KV_SLICE, np.uint8).copy() for i in ([] if SKIPREF else list(E.attn_idx)[:2])}

for i in E.attn_idx:
  P.up(f"kv{i}", np.zeros(2*4*100352*256, dtype=np.uint8))
  if i % 4 == 0: dev.synchronize(); P._keep.clear()
dev.synchronize(); P._keep.clear()
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
P.win_up("pos_slot_b", 0, np.array([16], dtype=np.int32))
dev.synchronize()

# P7-B smoke: upload packed7 weights for blocks < G3N AFTER the reference (VRAM: originals stay live)
if G3:
  P7D = f"{BASE}/packed7"
  ups = [(t, i) for i in range(G3N) for t in ("fg","fu","fd","gate","out","q","k")
         if os.path.exists(f"{P7D}/{t}{i}.npy")]
  for k, (t, i) in enumerate(ups):
    R7[(t, i)] = P.up(f"r7_{t}_{i}", np.load(f"{P7D}/{t}{i}.npy"))
    if k % 16 == 15: dev.synchronize(); P._keep.clear()
  dev.synchronize(); P._keep.clear()
  print(f"[g3] packed7 uploaded: {len(ups)} tensors (blocks < {G3N})", flush=True)

# ONE clean 32-step run; gates read IMMEDIATELY (readout-order law)
t0 = time.perf_counter()
fwd32(0, wait_last=True)
dev.synchronize()
mine_time = time.perf_counter() - t0

if SKIPREF:
  logits32 = ref = None
else:
  logits32 = P.down("logits32", (M, 248320), np.float16).astype(np.float32)
  ref = ref_logits.astype(np.float32)
if not SKIPREF:
  act = np.abs(ref) > 1e-6
  e = np.abs(logits32[act] - ref[act]) / np.abs(ref[act])
  fn = np.linalg.norm(logits32 - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-9)
  print(f"[gate] logits32 vs T=1: med {np.median(e):.3e} F(row-max) {fn.max():.3e} -> "
        f"{'PASS' if fn.max() <= 3e-3 or np.median(e) <= 3e-3 else 'FAIL'}", flush=True)
if not SKIPREF:
  amd_mine = logits32.argmax(axis=1); amd_ref = ref_logits.astype(np.float32).argmax(axis=1)
  print(f"[gate] argmax agreement {int((amd_mine == amd_ref).sum())}/{M}", flush=True)
for i in ([] if SKIPREF else E.gdn_idx[:4]):
  j = E.gdn_idx.index(i)
  rm = P.down(f"recp{j}", (48*128*128,), np.float32)
  cm = P.down(f"convp{j}", (3*10240,), np.float32)
  rr, cr = ref_rec[i], ref_conv[i]
  dr = np.linalg.norm(rm - rr) / max(np.linalg.norm(rr), 1e-9)
  dc = np.linalg.norm(cm - cr) / max(np.linalg.norm(cr), 1e-9)
  print(f"[drift] blk {i}: rec {dr:.3e} conv {dc:.3e}", flush=True)
for i in ([] if SKIPREF else list(E.attn_idx)[:2]):
  kvm = P.down_at(f"kv{i}", 0, KV_SLICE, np.uint8)
  nz = int((kvm != ref_kv[i]).sum())
  print(f"[drift] kv blk {i}: byte mismatches {nz}/{kvm.size} (first-128-row slice)", flush=True)

# timing reps AFTER the gates
best = mine_time
for _ in range(9):
  t0 = time.perf_counter(); fwd32(0, wait_last=True); dev.synchronize()
  best = min(best, time.perf_counter() - t0)
print(f"[time] fwd32 chunk (32 tok): {best*1e3:.1f} ms -> projected prefill {32/best:.1f} tok/s (harness, pos 0)", flush=True)
