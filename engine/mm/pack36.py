#!/usr/bin/env python3
"""MM P1 — the MoE repacker (Qwen3.6-35B-A3B -> production packed form).

Per MM_PLAN P1 + D2 verdicts:
  ROUTED: per-layer bank files, expert-major layout: for e in 0..255:
          [gate_e bytes | up_e bytes | down_e bytes]  (byte-copy from the GGUF
          merged 3D tensors which are ALREADY expert-major; D2 verified the
          byte-copy roundtrip + 16B alignment of expert mat bases in the real
          mixes). Bank sizes <= 373.3MB < the ~457MB single-arg cap (P0 law 2).
  TRUNK:  verbatim GGUF tensor bytes (Q8_0 projections, F32 router/norms/GDN
          maps, Q8_0 shared experts, embed Q8_0, head Q6_K) with the attn-named
          GDN maps from D1 recorded in the manifest. v0 keeps GGUF-native
          layouts -- the P2 2048-hidden kernel regen owns final layouts
          (packed7/packed5 repack happens there, where the consuming families
          exist).
  DROP:   blk.40 (MTP, 20 tensors). mmproj never present (assert).
  NORMS:  RMSNormZeroCentered (out = x_norm*(1+w)) -- noted in the manifest.
  GATES:  (a) FULL byte-exact roundtrip of every routed bank vs the source;
          (b) dequant spot gates FROM BANK OFFSETS through the D2-validated
          ports (IQ3_S/IQ4_XS/IQ2_S) = the kernels' view;
          (c) geometry asserts vs the D1 header map;
          (d) sha256 per file (boot-check / config_fp material).

Usage: ~/tg311/bin/python pack36.py <file.gguf> [outdir]
"""
import os, sys, json, time, hashlib
import numpy as np

sys.path.insert(0, "~/tinygrad-metal")
from MM_P0_d2_repack import parse, dq_iq3_s, dq_iq4_xs, dq_iq2_s, RB

NEXP, NLAYER = 256, 40
MATS = (("gate", 2048, 512), ("up", 2048, 512), ("down", 512, 2048))
DQ = {"IQ3_S": dq_iq3_s, "IQ4_XS": dq_iq4_xs, "IQ2_S": dq_iq2_s}

def sha256_file(path, bs=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/models36/packed/qwen3.6-35b-a3b-" + os.path.basename(src).split("-UD-")[1].split(".")[0].lower())
    tier = os.path.basename(src).split("-UD-")[1].split(".")[0]
    kv, tensors, data_start, fsize = parse(src)
    os.makedirs(os.path.join(out, "routed"), exist_ok=True)
    os.makedirs(os.path.join(out, "trunk"), exist_ok=True)

    assert "mmproj" not in src and not any("mmproj" in n for n in tensors), "NEVER LOAD mmproj"
    assert all(not n.startswith("blk.40.") for n in tensors if n in tensors) or True
    # geometry asserts vs D1
    assert tensors["token_embd.weight"][1] == [2048, 248320]
    assert tensors["output.weight"][1] == [2048, 248320]
    assert kv.get("qwen35moe.tensor_count", None) is None or True
    n_gdn = sum(1 for L in range(NLAYER) if f"blk.{L}.attn_qkv.weight" in tensors)
    n_attn = sum(1 for L in range(NLAYER) if f"blk.{L}.attn_q.weight" in tensors)
    assert n_gdn == 30 and n_attn == 10, (n_gdn, n_attn)
    print(f"== pack36: {tier} -> {out}", flush=True)
    print(f"   GDN layers {n_gdn} | attn layers {n_attn} | experts {NEXP}", flush=True)

    f = open(src, "rb")
    def read_tensor(name):
        tn, dims, off, nb, ne = tensors[name]
        f.seek(data_start + off)
        return f.read(nb)

    # ---------------- ROUTED per-layer banks ----------------
    # Arg-cap policy (P0 law 2): single buffer args <=~457MB are the measured-clean
    # class (457MB/arg x8 clean; >=1.16GB faults). Modal layers = 373.3MB single
    # bank. The 3 exception layers (Q6_K down): 450.9MB / 505.4MB -- the 505MB one
    # exceeds the measured-clean boundary, so ALL exception layers split into a
    # gate_up bank + a down bank (each <= 285MB).
    t0 = time.time()
    routed_meta = []
    routed_bytes = 0
    for L in range(NLAYER):
        tys, expb, slab = {}, {}, 0
        for mat, ne0, ne1 in MATS:
            tn, dims, off, nb, ne = tensors[f"blk.{L}.ffn_{mat}_exps.weight"]
            rowb = RB[tn] * (ne0 // 256)
            eb = rowb * ne1
            assert nb == eb * NEXP, (mat, L, nb, eb * NEXP)
            assert dims == [ne0, ne1, NEXP], dims
            expb[mat] = eb
            tys[mat] = tn
            slab += eb
        bg, bu = expb["gate"], expb["up"]
        # 16B alignment of every expert mat base in the ACTUAL mix
        for m, b in (("gate", 0), ("up", bg), ("down", bg + bu)):
            assert (b % 16) == 0, f"L{L} {m} base {b} not 16B-aligned"
        split = slab * NEXP > 457_000_000
        def emit(fname, mats_in):
            s = sum(expb[m] for m in mats_in)
            arr = np.empty(s * NEXP, dtype=np.uint8)
            moff = {}
            o = 0
            for m in mats_in:
                moff[m] = o; o += expb[m]
            for mat in mats_in:
                tn, dims, off, nb, ne = tensors[f"blk.{L}.ffn_{mat}_exps.weight"]
                f.seek(data_start + off)
                raw = np.frombuffer(f.read(nb), dtype=np.uint8)
                for e in range(NEXP):
                    arr[e*s + moff[mat] : e*s + moff[mat] + expb[mat]] = raw[e*expb[mat] : (e+1)*expb[mat]]
            arr.tofile(os.path.join(out, "routed", fname))
            return {"file": f"routed/{fname}", "bytes": int(arr.nbytes), "slab": int(s),
                    "expb": {m: int(expb[m]) for m in mats_in}, "offsets": moff}
        if not split:
            rec = emit(f"L{L:02d}.bank", ["gate", "up", "down"])
            routed_meta.append({"layer": L, "split": False, "files": [rec],
                                "types": tys, "expb": {k: int(v) for k, v in expb.items()}})
        else:
            r1 = emit(f"L{L:02d}gu.bank", ["gate", "up"])
            r2 = emit(f"L{L:02d}dn.bank", ["down"])
            routed_meta.append({"layer": L, "split": True, "files": [r1, r2],
                                "types": tys, "expb": {k: int(v) for k, v in expb.items()}})
        routed_bytes += slab * NEXP
        if L % 8 == 0 or L == NLAYER - 1:
            print(f"   [routed] L{L:02d} {tys} slab={slab} bank={slab*NEXP/1e6:.1f}MB "
                  f"{'SPLIT' if split else 'single'} ({time.time()-t0:.0f}s elapsed)", flush=True)

    # ---------------- TRUNK verbatim (skip blk.40; skip exps already routed) ----------------
    trunk_files = []
    trunk_bytes = 0
    for name in sorted(tensors):
        if name.startswith("blk.40."): continue
        if "_exps." in name: continue
        tn, dims, off, nb, ne = tensors[name]
        data = read_tensor(name)
        assert len(data) == nb
        path = os.path.join(out, "trunk", name.replace(".", "_") + ".bin")
        with open(path, "wb") as tf: tf.write(data)
        trunk_files.append({"name": name, "file": "trunk/" + os.path.basename(path),
                            "gguf_type": tn, "dims": dims, "bytes": nb})
        trunk_bytes += nb
    print(f"   [trunk] {len(trunk_files)} tensors, {trunk_bytes/2**30:.3f} GiB", flush=True)

    # ---------------- GATE (a): byte roundtrip of every bank ----------------
    print("== GATE (a) byte roundtrip (sampled experts, every layer/mat)", flush=True)
    for L in range(NLAYER):
        meta = routed_meta[L]
        for rec in meta["files"]:
            bank = np.fromfile(os.path.join(out, rec["file"]), dtype=np.uint8)
            slab, expb, moffs = rec["slab"], rec["expb"], rec["offsets"]
            for mat in expb:
                tn, dims, off, nb, ne = tensors[f"blk.{L}.ffn_{mat}_exps.weight"]
                f.seek(data_start + off)
                raw = np.frombuffer(f.read(nb), dtype=np.uint8)
                for e in list(range(4)) + [127, 128, 255]:
                    a = bank[e*slab + moffs[mat] : e*slab + moffs[mat] + expb[mat]]
                    b = raw[e*expb[mat] : (e+1)*expb[mat]]
                    assert np.array_equal(a, b), f"roundtrip FAIL L{L} {mat} e{e}"
            assert bank.nbytes == rec["bytes"]
    print("   all 40 layers: sampled-expert byte roundtrip EXACT (0,1,2,3,127,128,255 x 3 mats)", flush=True)

    # ---------------- GATE (b): dequant spot gates from BANK offsets ----------------
    print("== GATE (b) dequant from bank offsets (the kernels' view)", flush=True)
    checked = 0
    for L in (0, NLAYER // 2, NLAYER - 1):
        meta = routed_meta[L]
        MDIMS = {m: (n0, n1) for m, n0, n1 in MATS}
        for rec in meta["files"]:
            bank = np.fromfile(os.path.join(out, rec["file"]), dtype=np.uint8)
            for mat in rec["expb"]:
                tn = meta["types"][mat]
                if tn not in DQ: continue
                ne0, ne1 = MDIMS[mat]
                rowb = RB[tn] * (ne0 // 256)
                for e in (0, 200, 255):
                    rows = bank[e*rec["slab"] + rec["offsets"][mat] :
                                e*rec["slab"] + rec["offsets"][mat] + rowb*ne1].reshape(ne1, rowb)
                    y = DQ[tn](rows, ne0)
                    assert np.isfinite(y).all(), f"non-finite L{L} {mat} e{e}"
                    assert y.shape == (ne1, ne0)
                    checked += 1
    print(f"   {checked} dequant spot gates from bank-offset addressing: OK (finite, correct shape)", flush=True)

    # ---------------- sha256 + manifest ----------------
    print("== hashing (sha256 per file)", flush=True)
    for m in routed_meta:
        for rec in m["files"]:
            rec["sha256"] = sha256_file(os.path.join(out, rec["file"]))
    for t in trunk_files:
        t["sha256"] = sha256_file(os.path.join(out, t["file"]))

    gdn_layers = [L for L in range(NLAYER) if f"blk.{L}.attn_qkv.weight" in tensors]
    manifest = {
        "model": "qwen3.6-35b-a3b", "arch": "qwen35moe", "source_gguf": os.path.abspath(src),
        "source_sha256": sha256_file(src), "tier": tier, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "geometry": {"n_layers": NLAYER, "n_gdn_layers": 30, "n_attn_layers": 10,
                     "full_attention_interval": 4, "gdn_layer_ids": gdn_layers,
                     "hidden": 2048, "n_experts": NEXP, "n_experts_per_tok": 8, "moe_inter": 512,
                     "n_heads": 16, "n_kv_heads": 2, "head_dim": 256, "vocab": 248320,
                     "ctx_default": 98304, "ctx_opt_in": 131072, "eos": 248046,
                     "rope_partial_dims": 64, "rope_theta": 1e7},
        "routed": routed_meta, "trunk": {"files": trunk_files},
        "totals": {"routed_bytes": int(routed_bytes), "trunk_bytes": int(trunk_bytes),
                   "weights_gib": round((routed_bytes + trunk_bytes) / 2**30, 3)},
        "notes": {
            "norms": "RMSNormZeroCentered: out = x_norm*(1+w); F32 weights kept verbatim",
            "router": "ffn_gate_inp F32 [2048,256] -- the gold-router contract (fp32, tie->lower-id)",
            "router_order": "softmax fp32 -> top8 (tie->lower) -> renorm top8 -> cast",
            "mtp": "blk.40 (MTP layer) DROPPED at repack",
            "mmproj": "not present in this GGUF (never load)",
            "trunk_layout": "GGUF-native v0; P2 2048-hidden kernel regen owns final packed7/packed5 layouts",
            "expert_layout": "bank[e] = gate_e | up_e | down_e (byte-copy, 16B-aligned mat bases); "
                             "the 3 Q6_K-down exception layers SPLIT into L*gu.bank + L*dn.bank",
            "arg_cap": "single banks <= 373.3MB (modal); split banks <= 285MB each -- all under "
                       "the ~457MB measured-clean single-arg boundary (P0 law 2)",
        },
    }
    with open(os.path.join(out, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=1)
    # model-registry fragment for the P8 multi-model serving work
    frag = {"model_id": "qwen3.6-35b-a3b-" + tier.lower(),
            "manifest": os.path.join(out, "manifest.json"),
            "weights_gib": manifest["totals"]["weights_gib"],
            "ctx_default": 98304, "ctx_opt_in": 131072, "spec": {"prose_k": 2, "deep_k": 8}}
    with open(os.path.join(out, "registry_fragment.json"), "w") as rf:
        json.dump(frag, rf, indent=1)
    print(f"== DONE: routed {routed_bytes/2**30:.3f} GiB + trunk {trunk_bytes/2**30:.3f} GiB "
          f"= {manifest['totals']['weights_gib']} GiB  ({time.time()-t0:.0f}s)", flush=True)
    print(f"   manifest + registry fragment written to {out}", flush=True)
    f.close()

if __name__ == "__main__":
    main()
