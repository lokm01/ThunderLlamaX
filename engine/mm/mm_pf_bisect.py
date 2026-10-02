#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION A / G0: the PF-graph attribution bisect.

The MM twin of engine0/tlx_attrb_bisect.py (the truncated-graph law: a
filtered PF seq is TIMING-VALID / output-garbage -- downstream kernels read
garbage buffers, which is timing-neutral on the GPU).

Boots Rig7 exactly like the production daemon's PF graph
(test_moe36.py: build_seq7(rig, 256, gconv36_256, k2s36_256, with_head=False,
spk='s98304', S=8, pf=True)), then times truncated-graph family arms at
L in {2k, 8k, 16k, 48k, 96k} (L enters at RUNTIME via POSB -- the graphs are
L-independent; pos_view[0] = L-256 makes the chunk end at L).

Families (share = full - no_<family>):
  trunk_g        gv8k2048p + gv8k4096r + gvf32ab   (the per-seat GEMV re-reads)
  routed         gx8e256up* + gx8e256dn*           (the expert pair-walk)
  shared_router  rt8e256 + shexp8 + cmbz2048       (per-seat MoE orchestration)
  attn           spka*/spkq*/spkc*                 (the KV-reading trio)
  scan           gconv36_256 + k2s36_256           (split: no_gconv / no_k2s)
  residual       norms + embed                     (full - the five families)

Timing: wait-each GraphRunner.step() replays, 1 warm + 8 timed, MIN taken
(the first-clean-run law: the first clean run of each arm is also reported).
Safety: eidsb zeroed before each arm (gx pair-walk indexes PTB tables with
eids -- garbage ids = OOB pointer reads = device fault), pf_ids preloaded
with real prose ids (embg248 clamps anyway).

Deliverable: engine0/mm/mm_pf_attr.json -- replaces the plan's calibrated
attribution model with measurements. KEY QUESTIONS:
  (a) trunk-GEMV share @2k (model says 58-62%; KILL/re-rank L1 if <50%)
  (b) the attention growth coefficient (model: +20.5 ms / 1k-ctx / chunk)
  (c) in-graph k2s36_256 scan cost (never measured at TMAX=256)
  (d) pair-walk + trunk effective GB/s (bytes from the manifest + hist)
"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF_S = int(os.getenv("MM_PF_S", "8"))
RUNG = CTXK
PF = 256
LS = [2048, 8192, 16384, 49152, 98304]
WARM, REPS = 1, 8
OUT = os.path.join(BASE, "engine0", "mm", "mm_pf_attr.json")

FAMILY = {   # prefix match (covers the up4/dn6 quant variants + rung suffixes)
    "trunk_g":       ("gv8k2048p", "gv8k4096r", "gvf32ab"),
    "routed":        ("gx8e256up", "gx8e256dn"),
    "shared_router": ("rt8e256", "shexp8", "cmbz2048"),
    "attn":          ("spk",),
    "scan":          ("gconv36", "k2s36"),
    "gconv_only":    ("gconv36",),
    "k2s_only":      ("k2s36",),
}

def in_family(name, fam):
    return any(name.startswith(p) for p in FAMILY[fam])

def arm_filter(seq, fam):
    return [e for e in seq if not in_family(e[0], fam)]

def load_prose_ids(n=PF):
    d = json.load(open(os.path.join(BASE, "eval", "data", "ppl_prose_ids.json")))
    out = []
    while len(out) < n:
        out += [int(t) for t in d]
    return out[:n]

def time_runner(gr, warm=WARM, reps=REPS):
    ts = []
    for i in range(warm + reps):
        t0 = time.perf_counter()
        gr.step()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts[warm:]), ts[warm:], ts[0]

def fsync_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    dfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try: os.fsync(dfd)
    finally: os.close(dfd)

def run_g0(resume=None, rig=None):
    import numpy as np
    from MM_P7_lib import build_seq7
    from mm_a_graph import mkgraph_unc as mkgraph   # THE UNCACHED-KA LAW

    if rig is None:
        from MM_P7_lib import Rig7
        print(f"[g0] booting Rig7 (ctx {CTXK}, split rung {RUNG})...", flush=True)
        rig = Rig7(ctx_alloc=CTXK, load_p6=True)
    spk = f"s{RUNG}"
    seq_full = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                          spk=spk, S=PF_S, pf=True)
    names = {}
    for n, b, g, v in seq_full:
        names[n] = names.get(n, 0) + 1
    print(f"[g0] pf seq: {len(seq_full)} kernels: {dict(sorted(names.items()))}", flush=True)

    arms = ["full"] + [f"no_{f}" for f in ("trunk_g", "routed", "shared_router",
                                           "attn", "scan", "gconv_only", "k2s_only")]
    runners = {}
    for a in arms:
        s = seq_full if a == "full" else arm_filter(seq_full, a[3:])
        runners[a] = mkgraph(rig, s, f"g0_{a}")
        print(f"[g0] arm {a}: {len(s)} kernels", flush=True)

    # safety: real ids in, eids zeroed before every arm run
    ids = np.ascontiguousarray(np.array(load_prose_ids(), dtype=np.int32))
    rig.pf_ids_view[:] = memoryview(ids.data)
    zero_eids = np.zeros(PF * 8, dtype=np.uint16)
    dev = rig.dev

    res = resume or {"meta": {"ctxk": CTXK, "pf_s": PF_S, "pf": PF, "reps": REPS,
                              "kernel_counts": names, "rows": {}}}
    rows = res.setdefault("rows", {})
    try:
        for L in LS:
            key = str(L)
            if key in rows and "full" in rows[key]:
                print(f"[g0] L={L} already done -- skip", flush=True)
                continue
            row = {}
            for a in arms:
                dev.allocator._copyin(rig.PFB["eidsb"], memoryview(zero_eids.data))
                rig.pos_view[0] = L - PF
                dev.synchronize()
                mn, ts, first = time_runner(runners[a])
                row[a] = {"min_ms": round(mn, 3), "mean_ms": round(sum(ts)/len(ts), 3),
                          "first_ms": round(first, 3)}
                print(f"[g0] L={L:6d} {a:16s} {mn:8.2f} ms (first {first:8.2f})", flush=True)
            fams = {}
            f = row["full"]["min_ms"]
            for fam in ("trunk_g", "routed", "shared_router", "attn", "scan"):
                fams[fam] = round(f - row[f"no_{fam}"]["min_ms"], 3)
            fams["gconv_only"] = round(f - row["no_gconv_only"]["min_ms"], 3)
            fams["k2s_only"] = round(f - row["no_k2s_only"]["min_ms"], 3)
            fams["residual_norms_embed"] = round(
                f - sum(v for k, v in fams.items() if k not in ("gconv_only", "k2s_only")), 3)
            row["families"] = fams
            row["family_pct"] = {k: round(100.0 * v / f, 1) for k, v in fams.items()}
            row["chunk_tok_s"] = round(PF / (f / 1e3), 1)
            rows[key] = row
            res["rows"] = rows
            fsync_json(OUT, res)
            print(f"[g0] L={L}: families {fams}  pct {row['family_pct']}  "
                  f"{row['chunk_tok_s']} tok/s/chunk-graph", flush=True)
    finally:
        # growth coefficient (attn family vs L)
        try:
            import numpy as _np
            xs = _np.array([int(k) for k in rows.keys() if "families" in rows.get(k, {})], dtype=float)
            if len(xs) >= 2:
                at = _np.array([rows[str(int(x))]["families"]["attn"] for x in xs])
                sl, ic = _np.polyfit(xs / 1e3, at, 1)
                res["attn_growth"] = {"ms_per_1k_ctx_per_chunk": round(float(sl), 3),
                                      "intercept_ms": round(float(ic), 3)}
                tr2k = rows.get("2048", {}).get("family_pct", {}).get("trunk_g")
                res["gate_trunk_2k_pct"] = tr2k
                res["gate_verdict"] = ("L1 GO (trunk_g >= 50% @2k)" if (tr2k or 0) >= 50.0
                                       else "L1 KILL/RE-RANK (trunk_g < 50% @2k)")
        except Exception as e:
            print(f"[g0] growth fit skipped: {e}", flush=True)
        fsync_json(OUT, res)
    print(f"[g0] done -> {OUT}", flush=True)
    return res

if __name__ == "__main__":
    resume = None
    if os.path.exists(OUT):
        try:
            resume = json.load(open(OUT)); print("[g0] resuming from partial JSON", flush=True)
        except Exception:
            resume = None
    run_g0(resume)
