# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W4.1 cubin-metadata census (TLX W4, review-fix campaign).

Host-side, GPU-free: parses every engine0/*.cubin ELF exactly the way
NVProgram does at runtime (.nv.info attributes) and reports per-kernel:
regs (EIATTR_REGCOUNT 0x2f), min_stack bytes (EIATTR_MIN_STACK_SIZE 0x12 =
the ptxas spill/frame metric behind the P18 ">~100B local-mem = nondet" law),
maxntid (EIATTR_MAX_THREADS 0x05 -> [x,y,z] launch bounds), smem
(.nv.shared.<name> section size), and the gcycle NAME-LAW local size
("nw32"->1024 / "nw24"->768 / "nw16"->512 / else 256 — the NAME-ENCODED
LAUNCH CONFIG LAW).

Outputs: engine0/w4_census.json (full table) + stdout policy summary.
Set membership flags mirror the shipped configs:
  canon  = the canonical single-stream daemon (LOOKUP_K=10, GEMVV, DR7,
           SKV/KV8/QH/PVH trunk+probe+deep attention sets, MTpd draft,
           legacy trunk GDN/NEW cubins still loaded at boot)
  batch  = the R6 Phase-3 batch extras (r6_boot `need` list)
  pf     = the prefill family (pf_* / p8* / pfk_* / pfc* / pfs* / pfg*)
"""
import json, os, struct, sys
sys.path.insert(0, "~/tinygrad-src")
from tinygrad.runtime.support.elf import elf_loader

BASE = "~/tinygrad-metal/engine0"

# ---- shipped-set membership (mirrored from the loaders; see w4 tests) ----
GDN_CUBINS = ["k0_norm","k1_q5","k1_iq3","k1_ab","k2_scan","k2b_z","k3a_oproj","k3m_hh","k3b_ffn","k3c_down"]
NEW_CUBINS = ["a_q6","a_kv","a_attn","a_o","h_embed","h_argmax","k3a_iq3","k0ab","k2s","a_qkv_q6","a_qkv_iq3","k1_q5g"]
W1C_CUBINS = ["q5g8", "head8", "ffn8", "down8", "op38", "aq6k8", "aq3k8", "ao8"]
R7D_CUBINS = ["ffn8r7", "down8r7", "ffn8v3r7", "down8nw32v3r7", "ffn8v8r7", "down8nw32v8r7"]
M3_CUBINS = ["h_embed3","k0n3","k0ab3","q5g8_3","k2s3","op38_3","k3ao3","hh3","ffn8_3","down8_3",
             "aq6k8_3","aq3k8_3","aattn3","ao8_3","head8_3","amx3"]
V2_CUBINS = ["q5g8v_3","ffn8v_3","down8nw32_3","op38nw32_3","k3aonw32_3","ao8nw32_3","aq3k8v_3","head8v_3"]
M11_CUBINS = ["h_embed11","k0n11","k0ab11","q5g8v11","k2s11","op38nw32_11","k3aonw32_11","ao8nw32_11",
              "hh11","ffn8v11r7","down8nw32v11r7","aq3k8v11","aq6k8v11","head8v11","lookup11_nw32",
              "acceptk","accept11k","acceptsel11k"]
MTpd_CUBINS = ["dnorm2","dfgu","dkv","aattn_d","shead","samx","dposadd","accept","acceptsel",
               "ehproj","dq","doproj","ddown","mfill","stxrec","stxconv"]
SPK_CANON = ["spk_pre1qh_100k", "spk_g4nw32qh1p_100k", "spk_c1g_100k",        # trunk T=1 (PVH)
             "spk_pre3qh_100k", "spk_g4nw32hm3_100k", "spk_c3g_100k",         # probe T=3 (HMMA a3)
             "spk_pre11qh_100k", "spk_g4nw32hm11_100k", "spk_c11g_100k"]      # deep ROWS=11
CANON = set(GDN_CUBINS + NEW_CUBINS + W1C_CUBINS + R7D_CUBINS + M3_CUBINS + V2_CUBINS
            + M11_CUBINS + MTpd_CUBINS + SPK_CANON)
BATCH = {"h_embed5","k2s5","accept5k","acceptsel5k","lookup5_nw32",
         "k0n10","k0ab10","q5g8v10","aq3k8v10","aq6k8v10","ao8nw32_10",
         "k3aonw32_10","op38nw32_10","hh10","ffn8v10r7","down8nw32v10r7","head8v10","pfk_n16"}

def name_law_ls(nm):
  return 1024 if "nw32" in nm else 768 if "nw24" in nm else 512 if "nw16" in nm else 256

def parse_cubin(path):
  lib = open(path, "rb").read()
  _, sections, _ = elf_loader(lib, force_section_align=128)
  out = {"regs": None, "min_stack": None, "maxntid": None, "smem": 0, "kname": None}
  for sh in sections:
    if sh.name.startswith(".nv.shared."):
      out["smem"] = max(out["smem"], sh.header.sh_size)
      out["kname"] = out["kname"] or sh.name[len(".nv.shared."):]
    elif sh.name.startswith(".nv.info."):
      out["kname"] = out["kname"] or sh.name[len(".nv.info."):]
    elif sh.name != ".nv.info":
      continue
    off = 0
    # NOTE: record stride mirrors NVProgram._parse_elf_info (4B header + sz-byte
    # payload for value records; dataless records carry the value in sz). Some
    # cubins report sh_size > len(content) (elf_loader padding) — clamp the walk.
    size = min(sh.header.sh_size, len(sh.content))
    while off + 4 <= size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, off)
      data = sh.content[off+4:off+4+sz] if typ == 0x4 else None
      if data is not None:
        if param == 0x2f and len(data) >= 8: out["regs"] = struct.unpack_from("I", data, 4)[0]
        elif param == 0x12 and len(data) >= 8: out["min_stack"] = struct.unpack_from("I", data, 4)[0]
        elif param == 0x05:
          d = data[4:] if len(data) == 16 else data   # 16B = func-id-prefixed
          if len(d) == 12: out["maxntid"] = list(struct.unpack_from("III", d, 0))
          elif len(d) == 6: out["maxntid"] = list(struct.unpack_from("HHH", d, 0))
      off += (sz + 4 if typ == 0x4 else 4)
  return out

def main():
  rows = {}
  bad = []
  for f in sorted(os.listdir(BASE)):
    if not f.endswith(".cubin"): continue
    nm = f[:-len(".cubin")]
    try: r = parse_cubin(f"{BASE}/{f}")
    except Exception as e:
      bad.append((nm, repr(e))); continue
    r["name_law_ls"] = name_law_ls(nm)
    r["canon"] = nm in CANON
    r["batch"] = nm in BATCH
    # TLX W5 fix: pfa32c*/pfa32nw16*/pfaw* were MISSED (the canonical PF_ATTN32/
    # PF_ATTNW kernels — the pfa32ct 104B hard trip on the live rig came from here)
    r["pf"] = nm.startswith(("pf_", "p8", "pfk_", "pfc", "pfs", "pfg", "pfa", "pfaw"))
    rows[nm] = r
  json.dump({"cubins": rows, "bad": bad}, open(f"{BASE}/w4_census.json", "w"), indent=1, sort_keys=True)

  # ---- policy summary ----
  def subset(mask):
    return {k: v for k, v in rows.items() if mask(v)}
  def summ(tag, s):
    n = len(s)
    if not n: print(f"{tag}: (none)"); return
    stacks = [(k, v["min_stack"]) for k, v in s.items() if v["min_stack"]]
    mism = [(k, v["name_law_ls"], tuple(v["maxntid"] or ())) for k, v in s.items()
            if v["maxntid"] and v["name_law_ls"] != v["maxntid"][0]]
    nol = [k for k, v in s.items() if not v["maxntid"]]
    rmax = max((v["regs"] or 0, k) for k, v in s.items())
    smax = max((v["smem"] or 0, k) for k, v in s.items())
    print(f"{tag}: n={n} max_regs={rmax} max_smem={smax[0]}B({smax[1]})")
    print(f"  min_stack>0: {sorted(stacks, key=lambda x:-x[1])[:8]}")
    print(f"  name-law-vs-maxntid mismatches: {len(mism)} -> {mism[:8]}")
    print(f"  no-maxntid-attr: {nol[:8]}")
  summ("CANON ", subset(lambda v: v["canon"]))
  summ("BATCH ", subset(lambda v: v["batch"] and not v["canon"]))
  summ("PF    ", subset(lambda v: v["pf"] and not v["canon"] and not v["batch"]))
  summ("ALLOTHER", subset(lambda v: not (v["canon"] or v["batch"] or v["pf"])))
  worst = sorted(((v["min_stack"] or 0, k, v["canon"], v["batch"], v["pf"]) for k, v in rows.items()), reverse=True)[:12]
  print("worst min_stack overall (bytes, name, canon, batch, pf):")
  for w in worst: print("  ", w)
  if bad: print("PARSE FAILURES:", bad)
  print(f"total parsed: {len(rows)}")

if __name__ == "__main__":
  main()
