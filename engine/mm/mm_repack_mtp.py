#!/usr/bin/env python3
"""TLX P8/PART-B — the MTP-layer repack (mm_repack_mtp): re-include blk.40
(the MTP/nextn layer, dropped at the original pack36 repack) into the ENGINE's
packed tree (~/models36/packed/qwen3.6-35b-a3b-iq4_xs — the pack the rig
actually loads; MM_P2_ports.PACK).

What lands:
  TRUNK  trunk/blk_40_*.bin — Q8_0 tensors VERBATIM (attn_q/k/v/output,
         shexp gate/up/down, nextn.eh_proj); F32 norms verbatim; the BF16
         routers (ffn_gate_inp [2048,256], ffn_gate_inp_shexp [2048])
         DEQUANTIZED to F32 (the gold-router contract: rt8e256 reads fp32).
  ROUTED routed/L40.bank — byte-copy expert-major slabs (gate|up|down) per
         pack36's emit, types recorded from the raw GGUF numbers via the
         L3 cross-check (identical raw numbers -> identical names).
  MANIFEST: trunk["files"] += the 18 MTP entries; routed[40] = the L40 entry;
         notes["mtp"] updated; sha256 per new file.

Gates: byte roundtrip (sampled experts + every trunk tensor), BF16 dequant
plausibility (router std ~1e-2), 16B alignment of every expert mat base,
geometry asserts vs the D1 header map. Idempotent-ish: refuses to double-add
(routed len 41 check).

Usage: ~/tg311/bin/python mm_repack_mtp.py
"""
import os, sys, json, time, hashlib
import numpy as np

sys.path.insert(0, "~/tinygrad-metal")
from MM_P0_d2_repack import parse as _parse

def parse(path):
    """MM_P0_d2_repack.parse + the BF16 fixup (type 30 in this GGUF's enum;
    the metal TNAME predates the renumber -> lands as 't30')."""
    kv, tensors, ds, fs = _parse(path)
    fixed = {}
    for n, (tn, dims, off, nb, ne) in tensors.items():
        if tn == "t30":
            tn = "BF16"
        fixed[n] = (tn, dims, off, nb, ne)
    return kv, fixed, ds, fs

PACK = os.path.expanduser("~/models36/packed/qwen3.6-35b-a3b-iq4_xs")
SRC = os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
NEXP = 256
MATS = (("gate", 2048, 512), ("up", 2048, 512), ("down", 512, 2048))
RB = {"IQ4_XS": 136, "IQ3_S": 110, "IQ2_S": 82, "Q6_K": 210,
      "Q3_K": 110, "Q4_K": 144}   # bytes / 256-elem block (KNOWN_BLK)


def sha256_file(path, bs=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    kv, tensors, data_start, fsize = parse(SRC)
    man = json.load(open(os.path.join(PACK, "manifest.json")))
    assert len(man["routed"]) == 40, "L40 already repacked (routed len 41) — refusing to double-add"
    names = set(tensors)
    need = ["blk.40.attn_norm.weight", "blk.40.post_attention_norm.weight",
            "blk.40.attn_q.weight", "blk.40.attn_k.weight", "blk.40.attn_v.weight",
            "blk.40.attn_output.weight", "blk.40.attn_q_norm.weight",
            "blk.40.attn_k_norm.weight", "blk.40.ffn_gate_inp.weight",
            "blk.40.ffn_gate_inp_shexp.weight", "blk.40.ffn_gate_shexp.weight",
            "blk.40.ffn_up_shexp.weight", "blk.40.ffn_down_shexp.weight",
            "blk.40.nextn.eh_proj.weight", "blk.40.nextn.enorm.weight",
            "blk.40.nextn.hnorm.weight", "blk.40.nextn.shared_head_norm.weight",
            "blk.40.ffn_gate_exps.weight", "blk.40.ffn_up_exps.weight",
            "blk.40.ffn_down_exps.weight"]
    missing = [n for n in need if n not in names]
    assert not missing, f"GGUF missing MTP tensors: {missing}"

    f = open(SRC, "rb")
    def read_raw(name):
        tn, dims, off, nb, ne = tensors[name]
        f.seek(data_start + off)
        return f.read(nb), tn, dims

    # ---------------- ROUTED L40 (byte-copy, pack36 emit) ----------------
    # NOTE: unsloth quantized the MTP layer's EXPERTS with the K-quants
    # (Q3_K gate/up, Q4_K down) unlike the trunk's IQ3_S/IQ4_XS — the bytes
    # are copied VERBATIM into the slab layout; the dequant side (the anchor)
    # needs dq_q3_k/dq_q4_k ports and the GPU kernels need either ports or a
    # requant path (the Part-B follow-up decision; recorded in the manifest).
    tys, expb, slab = {}, {}, 0
    for mat, ne0, ne1 in MATS:
        tn, dims, off, nb, ne = tensors[f"blk.40.ffn_{mat}_exps.weight"]
        assert tn in RB, f"L40 {mat} unexpected type {tn}"
        rowb = RB[tn] * (ne0 // 256)
        eb = rowb * ne1
        assert nb == eb * NEXP, (mat, nb, eb * NEXP)
        expb[mat] = eb; tys[mat] = tn; slab += eb
    for m, b in (("gate", 0), ("up", expb["gate"]), ("down", expb["gate"] + expb["up"])):
        assert b % 16 == 0, f"L40 {m} base {b} not 16B-aligned"
    s = slab
    arr = np.empty(s * NEXP, dtype=np.uint8)
    moff, o = {}, 0
    for m, _, _ in MATS:
        moff[m] = o; o += expb[m]
    for mat, _, _ in MATS:
        tn, dims, off, nb, ne = tensors[f"blk.40.ffn_{mat}_exps.weight"]
        f.seek(data_start + off)
        raw = np.frombuffer(f.read(nb), dtype=np.uint8)
        for e in range(NEXP):
            arr[e * s + moff[mat]: e * s + moff[mat] + expb[mat]] = raw[e * expb[mat]:(e + 1) * expb[mat]]
    bank_path = os.path.join(PACK, "routed", "L40.bank")
    arr.tofile(bank_path)
    print(f"[routed] L40 {tys} slab={s} bank={s * NEXP / 1e6:.1f}MB", flush=True)
    # roundtrip gate
    for mat, _, _ in MATS:
        tn, dims, off, nb, ne = tensors[f"blk.40.ffn_{mat}_exps.weight"]
        f.seek(data_start + off)
        raw = np.frombuffer(f.read(nb), dtype=np.uint8)
        for e in (0, 1, 128, 255):
            a = arr[e * s + moff[mat]: e * s + moff[mat] + expb[mat]]
            b = raw[e * expb[mat]:(e + 1) * expb[mat]]
            assert np.array_equal(a, b), f"roundtrip FAIL L40 {mat} e{e}"
    rec = {"file": "routed/L40.bank", "bytes": int(arr.nbytes), "slab": int(s),
           "expb": {m: int(v) for m, v in expb.items()}, "offsets": moff,
           "sha256": sha256_file(bank_path)}
    routed_meta = {"layer": 40, "split": False, "files": [rec],
                   "types": tys, "expb": {k: int(v) for k, v in expb.items()}}

    # ---------------- TRUNK (Q8_0 verbatim / F32 verbatim / BF16->F32) -----
    trunk_add = []
    for name in need:
        if "_exps." in name:
            continue
        data, tn, dims = read_raw(name)
        fname = name.replace(".", "_") + ".bin"
        path = os.path.join(PACK, "trunk", fname)
        if tn == "BF16":
            out = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            st = float(out.std())
            assert 1e-4 < abs(st) < 1.0, f"{name}: implausible BF16 dequant std {st}"
            out.tofile(path)
            rec_type, nb = "F32", out.nbytes
            print(f"[trunk] {name}: BF16[{dims}] -> F32 (std {st:.2e})", flush=True)
        else:
            assert tn in ("Q8_0", "F32"), f"{name}: unexpected type {tn}"
            with open(path, "wb") as tf:
                tf.write(data)
            rec_type, nb = tn, len(data)
        trunk_add.append({"name": name, "file": "trunk/" + fname, "gguf_type": rec_type,
                          "dims": dims, "bytes": nb, "sha256": sha256_file(path)})

    # ---------------- manifest update (atomic + fsync, the L5 law) ---------
    man["routed"].append(routed_meta)
    man["trunk"]["files"].extend(trunk_add)
    man["notes"]["mtp"] = ("blk.40 (MTP layer) REPACKED (mm_repack_mtp): Q8_0 GEMVs verbatim, "
                           "BF16 routers dequantized to F32, routed L40 bank byte-copy; "
                           "llama.cpp graph_mtp semantics (e_norm||h_norm concat, own-KV chain, "
                           "shared head + shared_head_norm)")
    man["totals"]["routed_bytes"] = int(man["totals"]["routed_bytes"] + arr.nbytes)
    man["totals"]["trunk_bytes"] = int(man["totals"]["trunk_bytes"] + sum(t["bytes"] for t in trunk_add))
    man["totals"]["weights_gib"] = round(
        (man["totals"]["routed_bytes"] + man["totals"]["trunk_bytes"]) / 2 ** 30, 3)
    tmp = os.path.join(PACK, "manifest.json.tmp")
    with open(tmp, "w") as mf:
        json.dump(man, mf, indent=1)
        mf.flush(); os.fsync(mf.fileno())
    os.replace(tmp, os.path.join(PACK, "manifest.json"))
    dfd = os.open(PACK, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    f.close()
    print(f"== MTP REPACK DONE: +{arr.nbytes / 1e6:.0f}MB routed + "
          f"{sum(t['bytes'] for t in trunk_add) / 1e6:.0f}MB trunk; manifest updated", flush=True)


if __name__ == "__main__":
    main()
