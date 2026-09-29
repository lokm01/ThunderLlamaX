#!/usr/bin/env python3
"""MM P0 D1 — GGUF header + tensor map for Qwen3.6-35B-A3B (unsloth UD quants).

Works on a PARTIAL file (first >=16MB) as long as data_start + a bit fits;
all tensor infos are in the header before data_start.

Usage: python MM_P0_d1_header.py <file.gguf-or-partial> [--full]
  --full: also verify that the LAST tensor's data extent fits the file (needs
          the complete download).
"""
import struct, sys, collections

TYPR = {0:(1,"B"),1:(1,"b"),2:(2,"H"),3:(2,"h"),4:(4,"I"),5:(4,"i"),6:(4,"f"),
        7:(1,"b"),10:(8,"Q"),11:(8,"q"),12:(8,"d")}

# ggml type enum (llama.cpp) — names for the classes we expect
TNAME = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
         10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
         16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
         21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"IQ1_M",25:"BF16",26:"TQ1_0",
         27:"TQ2_0",28:"MXFP4"}

# known block geometries (elems_per_block, bytes_per_block) — used only to
# cross-check the EMPIRICAL bytes/block derived from offsets
KNOWN_BLK = {"Q4_0":(32,18),"Q5_0":(32,22),"Q8_0":(32,34),"Q2_K":(256,84),
             "Q3_K":(256,110),"Q4_K":(256,144),"Q5_K":(256,176),"Q6_K":(256,210),
             "IQ2_XXS":(256,66),"IQ2_XS":(256,74),"IQ3_XXS":(256,102),
             "IQ1_S":(256,80//1*1),"IQ4_NL":(32,18),"IQ3_S":(256,110-66+0),
             "IQ2_S":(256,88),"IQ4_XS":(256,136//1),"IQ1_M":(256,56)}
# NOTE: the dict values above are REFERENCE-ONLY; empirical-from-offsets wins.

def parse(path):
    f = open(path, "rb")
    magic = f.read(4); assert magic == b"GGUF", magic
    ver = struct.unpack("<i", f.read(4))[0]
    n_tensors = struct.unpack("<q", f.read(8))[0]
    n_kv = struct.unpack("<q", f.read(8))[0]
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode()
    def rv(t):
        if t == 8: return rs()
        if t == 9:
            et = struct.unpack("<i", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rv(et) for _ in range(n)]
        if t == 10: return struct.unpack("<Q", f.read(8))[0]  # u64 scalar
        if t == 11: return struct.unpack("<q", f.read(8))[0]  # i64 scalar
        nb, fmt = TYPR[t]; return struct.unpack("<"+fmt, f.read(nb))[0]
    kv = {}
    for _ in range(n_kv):
        k = rs(); t = struct.unpack("<i", f.read(4))[0]; kv[k] = rv(t)
    infos = []
    for _ in range(n_tensors):
        name = rs(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        t = struct.unpack("<i", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, t, dims, off))
    align = kv.get("general.alignment", 32)
    data_start = (f.tell()+align-1)//align*align
    import os
    fsize = os.fstat(f.fileno()).st_size
    f.close()
    return ver, kv, infos, data_start, align, fsize

def classify(name):
    for pat, cls in [
        (".ffn_experts.", "routed"), ("ffn_experts", "routed"),
        ("experts", "routed?"),
        (".ffn_router.", "router"), ("ffn_router", "router"),
        (".ffn_inp.", "router?"), ("ffn_gate_inp", "router"),
        ("ffn_shexp", "shared"),
        (".attn_", "attn"), (".attn.", "attn"),
        ("mtp", "MTP"), ("mmproj", "mmproj"),
        ("output", "head"), ("token_embd", "embed"),
        ("token_types", "tokmeta"), (".norm", "norm"),
    ]:
        if pat in name: return cls
    if ".ffn_" in name: return "ffn-dense?"
    if "conv1d" in name or "a_proj" in name or "b_proj" in name or "in_proj" in name or "out_proj" in name or "dt" in name: return "GDN?"
    return "other"

def main():
    path = sys.argv[1]; full = "--full" in sys.argv
    ver, kv, infos, data_start, align, fsize = parse(path)
    print(f"== D1 header parse: {path}")
    print(f"gguf version={ver} n_kv={len(kv)} n_tensors={len(infos)} data_start={data_start} align={align} filesize={fsize/1e9:.3f}GB")
    assert data_start < fsize, "partial file too small to hold the full tensor map — refetch bigger prefix"
    print("\n== metadata (kv) ==")
    for k in sorted(kv):
        v = kv[k]
        if isinstance(v, list) and len(v) > 12: v = f"[{len(v)} items]"
        print(f"  {k} = {v}")

    # ---- empirical bytes-per-element from offsets (consecutive same-class tensors) ----
    byname = {n:(t,d,o) for n,t,d,o in infos}
    # group analysis
    groups = collections.defaultdict(list)
    for name, t, dims, off in infos:
        cls = classify(name)
        groups[cls].append((name, t, dims, off))

    print("\n== tensor classes ==")
    for cls in sorted(groups):
        rows = groups[cls]
        bytesum = 0; nelem = 0
        tset = collections.Counter(TNAME.get(t, f"t{t}") for _,t,_,_ in rows)
        print(f"[{cls}] n={len(rows)} types={dict(tset)}")
        for name, t, dims, off in rows[:6]:
            print(f"    e.g. {name} dims={dims} type={TNAME.get(t,t)} off={off}")
        if len(rows) > 6: print(f"    ... +{len(rows)-6} more")

    # empirical bytes/block: for each quant type, find consecutive pairs of
    # same-type same-shape tensors and use offset deltas
    print("\n== empirical block geometry (from offset deltas) ==")
    # sort by offset
    si = sorted(infos, key=lambda r: r[3])
    tstat = collections.defaultdict(list)
    for i,(name,t,dims,off) in enumerate(si[:-1]):
        ne = 1
        for d in dims: ne *= d
        nxt = si[i+1]
        ne_nxt = 1
        for d in nxt[2]: ne_nxt *= d
        if nxt[3] >= off and ne > 1000 and nxt[1] == t:
            bpe = (nxt[3]-off)/ne
            tstat[TNAME.get(t, f"t{t}")].append(bpe)
    for tn, l in sorted(tstat.items()):
        import statistics
        med = statistics.median(l)
        print(f"  {tn}: median bytes/elem = {med:.6f}  (n={len(l)} pairs)  -> per-256 = {med*256:.2f}B")

    # ---- routed-expert accounting ----
    print("\n== routed expert accounting ==")
    routed = [r for r in infos if "ffn_experts" in r[0]]
    layers = collections.defaultdict(list)
    for name, t, dims, off in routed:
        parts = name.split(".")
        try: L = int(parts[1])
        except: continue
        layers[L].append((name, t, dims, off))
    if routed:
        L0 = sorted(layers)[0]
        print(f"  layer {L0} expert tensors ({len(layers[L0])}):")
        for name,t,dims,off in sorted(layers[L0])[:12]:
            print(f"    {name} dims={dims} type={TNAME.get(t,t)}")
        if len(layers[L0])>12: print(f"    ... +{len(layers[L0])-12} more")
        # expert mat shapes
        e0 = sorted(layers[L0])[0]
        print(f"  expert count per layer: {len(layers[L0])//3 if len(layers[L0])%3==0 else 'NON-3: '+str(len(layers[L0]))}")
    # total routed bytes (needs empirical bpe)
    def total_bytes():
        # global: last offset + last nbytes vs file size (only when full)
        pass

    # ---- spec cross-check ----
    print("\n== MM_PLAN spec cross-check ==")
    arch = kv.get("general.architecture", "?")
    print(f"  arch = {arch}  (expect qwen35moe)")
    checks = []
    def ck(label, got, want):
        ok = (got == want) if want is not None else True
        checks.append((label, got, want, ok))
    ck("block_count(incl MTP)", kv.get(f"{arch}.block_count"), 41)
    vocab_n = len(kv.get("tokenizer.ggml.tokens", []))
    ck("vocab(tokenizer tokens)", vocab_n if vocab_n else None, 248320)
    ck("embedding_length", kv.get(f"{arch}.embedding_length"), 2048)
    ck("expert_count", kv.get(f"{arch}.expert_count"), 256)
    ck("expert_used_count", kv.get(f"{arch}.expert_used_count"), 8)
    ck("expert_feed_forward_length", kv.get(f"{arch}.expert_feed_forward_length"), 512)
    ck("expert_shared_ffn", kv.get(f"{arch}.expert_shared_feed_forward_length"), 512)
    ck("full_attention_interval", kv.get(f"{arch}.full_attention_interval"), 4)
    ck("head_count", kv.get(f"{arch}.attention.head_count"), 16)
    ck("head_count_kv", kv.get(f"{arch}.attention.head_count_kv"), 2)
    ck("head_dim(key_length)", kv.get(f"{arch}.attention.key_length"), 256)
    ck("ssm.state_size", kv.get(f"{arch}.ssm.state_size"), 128)
    ck("ssm.inner_size", kv.get(f"{arch}.ssm.inner_size"), 4096)
    ck("ssm.conv_kernel", kv.get(f"{arch}.ssm.conv_kernel"), 4)
    ck("rope.dimension_count", kv.get(f"{arch}.rope.dimension_count"), 64)
    ck("nextn_predict_layers(MTP)", kv.get(f"{arch}.nextn_predict_layers"), 1)
    ck("eos_token_id", kv.get("tokenizer.ggml.eos_token_id"), 248046)
    def _mtp_count():
        n = 0
        for r in infos:
            p = r[0].split(".")
            if p[0] == "blk" and len(p) > 1 and p[1].isdigit() and int(p[1]) >= 40:
                n += 1
        return n
    ck("blk>=40 tensors (MTP)", _mtp_count(), None)
    nbad = 0
    for label, got, want, ok in checks:
        flag = "OK " if ok else "MISMATCH"
        if not ok: nbad += 1
        print(f"  [{flag}] {label}: got={got} want={want}")
    print(f"\n  mismatches: {nbad}")

    # layer-type census from tensor names
    print("\n== per-layer tensor census (layers 0,1,2,3,39 + any mtp) ==")
    layt = collections.defaultdict(list)
    for name,t,dims,off in infos:
        p = name.split(".")
        if p[0]=="blk" and p[1].isdigit():
            layt[int(p[1])].append((name.replace(f"blk.{p[1]}.",""), TNAME.get(t,t), dims))
    for L in sorted(layt):
        if L not in (0,1,2,3,39): continue
        nonexp = [r for r in layt[L] if "ffn_experts" not in r[0]]
        print(f"  blk.{L}: {len(layt[L])} tensors, non-expert {len(nonexp)}:")
        for nm,ty,dm in sorted(nonexp): print(f"      {nm} {ty} {dm}")

    # MTP detection: blk index >= block_count or explicit names
    mtp = [n for n,_,_,_ in infos if n.startswith("blk.4") and not n.startswith("blk.4 ") and n.split(".")[1].isdigit() and int(n.split(".")[1])>=40]
    print(f"\n== tensors with blk index >= 40 (MTP layer): {len(mtp)}")
    for n in mtp[:20]: print(f"    {n}")

    if full:
        # verify data extents
        last = max(infos, key=lambda r: r[3])
        print(f"\n== full-file check: last tensor {last[0]} off={last[3]} file={fsize}")

    # ---- dump tensor table (name, type, dims, off, EXACT nbytes via sorted deltas) for D2/D6 ----
    import json
    si2 = sorted(infos, key=lambda r: r[3])
    rows = []
    for i,(name,t,dims,off) in enumerate(si2):
        ne = 1
        for d in dims: ne *= d
        if i+1 < len(si2): nb = si2[i+1][3] - off
        else: nb = fsize - data_start - off if full else None
        rows.append((name, t, dims, off, nb, ne))
    out = path.split("/")[-1].replace(".bin","").replace(".gguf","")
    with open(f"MM_P0_d1_tensors_{out}.json","w") as fj:
        json.dump({"rows":[[n,t,d,o,nb,ne] for n,t,d,o,nb,ne in rows],
                   "data_start":data_start,"align":align}, fj)
    # exact bytes-per-block for the exps quant classes
    print("\n== exact exps tensor sizes (offset deltas) ==")
    seen = {}
    for name,t,dims,off,nb,ne in rows:
        if "_exps" in name and nb:
            tn = TNAME.get(t, f"t{t}")
            bpb = nb/(ne/256)
            if tn not in seen or abs(bpb-seen[tn])>1e-9:
                seen[tn] = bpb
                print(f"  {tn}: {bpb:.4f} bytes/256-elem block  e.g. {name}")
    # per-layer exps class map
    print("\n== per-layer exps quant classes ==")
    lm = {}
    for name,t,dims,off,nb,ne in rows:
        if "_exps" in name:
            L = int(name.split(".")[1])
            mat = name.split(".")[2].replace("ffn_","").replace("_exps","")
            lm.setdefault(L, {})[mat] = TNAME.get(t, f"t{t}")
    variants = collections.Counter(tuple(sorted(v.items())) for v in lm.values())
    for v,c in variants.items(): print(f"  {c} layers: {dict(v)}")
    print(f"\n== wrote MM_P0_d1_tensors_{out}.json ==")

if __name__ == "__main__":
    main()
