#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION A side probes.

1. EXPERT-M HISTOGRAM: the REAL tokens-per-expert distribution. Rebuilds
   the production PF seq with a cp4k capture kernel appended after EVERY
   rt8e256 (eids[P*8] -> EIDSCAP[layer][2048]), resets states, feeds a REAL
   256-token chunk, one graph run -> per-layer histograms for 3 text
   classes (prose / code / gsm8k). Tests the Zipf assumption + the D6
   occupancy Monte Carlo; designs L2's grouped-GEMM bins; also yields the
   exact routed-pair-walk byte count -> effective GB/s with the G0 routed ms.
2. DYN-SMEM PROBE: 64KB then 96KB dynamic shared memory through OUR QMD
   path (there is no cudaFuncSetAttribute here; the probe patches
   prg.qmd shared_memory_size + carveout cfg before launch). PASS/FAULT
   gates the L10 structural branch. Also tries a static 64KB nvcc build
   (expected: refused -- documented).
3. L2/DRAM READ BW: grid-stride fp32 sum at 256MB (DRAM) and 4MB (L2-class)
   working sets, plain vs __ldcg, grid sweep. The L1-swap L2-sharing term
   needs the L2 number.
Output: engine0/mm/mm_probes.json
"""
import os, sys, time, json

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE + "/engine0/mm")
os.environ.setdefault("DEV", "NV")

CTXK = int(os.getenv("MM_CTXS", "98304"))
PF = 256
OUT = os.path.join(BASE, "engine0", "mm", "mm_probes.json")
CUB = os.path.join(BASE, "engine0", "mm")

from MM_P56_lib import LSZ, MG
from mm_l1_poc import load_prog, graph_time, fsync_json

def load_ids(kind, n=PF):
    if kind == "gsm8k":
        d = json.load(open(os.path.join(BASE, "eval", "data", "gsm8k_dump_ids.json")))
        out = []
        i = 0
        while len(out) < n:
            out += [int(t) for t in d[i % len(d)]]
            i += 1
        return out[:n]
    f = {"prose": "ppl_prose_ids.json", "code": "ppl_code_ids.json"}[kind]
    d = json.load(open(os.path.join(BASE, "eval", "data", f)))
    out = []
    while len(out) < n:
        out += [int(t) for t in d]
    return out[:n]

def run_ehist(rig, res):
    import numpy as np
    from MM_P7_lib import build_seq7
    from mm_a_graph import mkgraph_unc as mkgraph   # THE UNCACHED-KA LAW
    if "expert_hist" in res and len(res["expert_hist"].get("classes", {})) >= 3:
        print("[eh] already done -- skip", flush=True)
        return
    cp = load_prog(rig, "MM_A_cp4k.cubin", "cp4k", 256)
    spk = f"s{CTXK}"
    seq = build_seq7(rig, PF, "gconv36_256", "k2s36_256", with_head=False,
                     spk=spk, S=8, pf=True)
    cap = rig.alloc(40 * PF * 8 * 2)
    seq2 = []
    li = 0
    for ent in seq:
        seq2.append(ent)
        if ent[0] == "rt8e256":
            dst = cap.offset(offset=li * PF * 8 * 2, size=PF * 8 * 2)
            # THE 1-TUPLE GRID LAW: a raw (8,) 1-tuple bypasses build_seq7's
            # _grid normalization -> MG._build's malformed 2-tuple -> the
            # graph NEVER EXECUTES (the P7 law; bit us live). SCALAR grid.
            seq2.append(("cp4k", (rig.PFB["eidsb"], dst), 8, (PF * 8,)))
            li += 1
    assert li == 40, f"expected 40 rt8e256, got {li}"
    gr = mkgraph(rig, seq2, "eh_pf")

    res.setdefault("expert_hist", {})
    eh = res["expert_hist"]
    eh.setdefault("classes", {})
    for kind in ("prose", "code", "gsm8k"):
        if kind in eh["classes"]:
            continue
        rig.reset_states(1024)
        ids = np.ascontiguousarray(np.array(load_ids(kind), dtype=np.int32))
        rig.pf_ids_view[:] = memoryview(ids.data)
        rig.pos_view[0] = 0
        gr.step()
        cap_np = rig.dn(cap, (40, PF * 8), np.uint16)
        # DIAGNOSTIC (the all-one-expert fingerprint of the first attempt):
        # read eidsb directly (holds layer 39's live writes) + cap layer 0.
        eid_live = rig.dn(rig.PFB["eidsb"], (PF * 8,), np.uint16)
        print(f"[eh] {kind}: eidsb live distinct={len(np.unique(eid_live))} "
              f"min={eid_live.min()} max={eid_live.max()} | "
              f"cap L0 distinct={len(np.unique(cap_np[0]))} "
              f"cap min={cap_np.min()} max={cap_np.max()}", flush=True)
        assert cap_np.max() < 256 and cap_np.min() >= 0
        layers = []
        allc = np.zeros(256, dtype=np.int64)
        for L in range(40):
            c = np.bincount(cap_np[L], minlength=256)   # tokens-per-expert (8 ranks counted)
            allc += c
            layers.append({"distinct": int((c > 0).sum()), "max_tok": int(c.max()),
                           "mean_nonzero": round(float(c[c > 0].mean()), 2),
                           "top8_share_pct": round(100.0 * np.sort(c)[-8:].sum() / 2048, 1)})
        c = np.sort(allc)[::-1]
        bins = {}
        for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            bins[f"experts_with_le_{b}_tok"] = int((allc <= b).sum())
        eh["classes"][kind] = {
            "per_layer": layers,
            "mean_distinct": round(float(np.mean([l["distinct"] for l in layers])), 1),
            "max_tok_overall": int(max(l["max_tok"] for l in layers)),
            "global_top16_share_pct": round(100.0 * c[:16].sum() / (40 * 2048), 1),
            "global_zero_experts": int((allc == 0).sum()),
            "global_bins": bins,
            # Zipf check: share of top 1% experts vs the uniform 1%
            "zipf_top1pct_share_pct": round(100.0 * c[:3].sum() / (40 * 2048), 2),
        }
        fsync_json(OUT, res)
        print(f"[eh] {kind}: mean distinct/layer {eh['classes'][kind]['mean_distinct']}, "
              f"max tok/expert {eh['classes'][kind]['max_tok_overall']}, "
              f"top1% share {eh['classes'][kind]['zipf_top1pct_share_pct']}%", flush=True)

    # routed pair-walk traffic: pairs read gate+up (up kernel) + down (dn)
    try:
        rec = rig.man["routed"][0]["files"][0]
        slab = rec["slab"]
        eh["routed_bytes_per_chunk_per_layer"] = int(slab) * PF * 8  # per-pair full slab walk
        eh["routed_bytes_per_chunk_total"] = int(slab) * PF * 8 * 40
    except Exception as e:
        print(f"[eh] slab accounting skipped: {e}", flush=True)

def run_dsmem(rig, res):
    """The dyn-smem probe, IN-PROCESS (the GPU-EXIT/REBOOT law: this rig gets
    ONE GPU process per boot -- a subprocess would die with the reboot its
    parent's exit provokes anyway). Runs LAST by contract: a FAULT here can
    wedge the process, and everything else is already fsynced. A pre-attempt
    marker + the absence of a verdict = FAULT evidence."""
    import numpy as np
    if "dyn_smem" in res and "64kb" in res["dyn_smem"]:
        print("[ds] already done -- skip", flush=True)
        return
    prg = load_prog(rig, "MM_A_dsmemp.cubin", "dsmemp", 1024)
    res.setdefault("dyn_smem", {})
    out = res["dyn_smem"]
    # static 64KB compile attempt (documented; expected refused)
    try:
        import subprocess
        from MM_P56_lib import NVCC_ENV
        src = os.path.join(CUB, "MM_A_dsmem_static_try.cu")
        with open(src, "w") as f:
            f.write('extern "C" __global__ void sst(float* o){ __shared__ float a[16384]; '
                    'a[threadIdx.x]=threadIdx.x; o[threadIdx.x]=a[threadIdx.x]; }\n')
        r = subprocess.run(f"nvcc -arch=sm_86 -cubin -o /tmp/sst.cubin {src}",
                           shell=True, capture_output=True, text=True, env=NVCC_ENV)
        out["static_64k_nvcc"] = "COMPILED (unexpected)" if r.returncode == 0 else \
            "REFUSED: " + (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?")
    except Exception as e:
        out["static_64k_nvcc"] = f"probe error {e}"
    print(f"[ds] static 64KB nvcc: {out['static_64k_nvcc']}", flush=True)
    fsync_json(OUT, res)          # the 'attempting' state IS the fault marker
    for kb in (64, 96):
        if f"{kb}kb" in out:
            continue
        words = kb * 1024 // 4
        try:
            ob = rig.alloc(1024 * 4)
            rig.dev.allocator._copyin(ob, memoryview(np.zeros(1024, dtype=np.uint32).tobytes()))
            prg.qmd.write(shared_memory_size=kb * 1024 + 1024,
                          min_sm_config_shared_mem_size=100 * 1024 // 4096 + 1,
                          target_sm_config_shared_mem_size=100 * 1024 // 4096 + 1)
            prg(ob, global_size=(1, 1, 1), local_size=(1024, 1, 1),
                vals=(words,), wait=True)
            i64 = np.arange(words, dtype=np.uint64)
            smv = ((i64 * np.uint64(2654435761)) + np.uint64(7)) & np.uint64(0xFFFFFFFF)
            k2 = (i64 * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
            got = rig.dn(ob, (1024,), np.uint32)
            bad = 0
            for tid in range(1024):
                idx = np.arange(tid, words, 1024)
                ref = (smv[idx] + k2[idx] + np.uint64(tid)) & np.uint64(0xFFFFFFFF)
                if np.bitwise_xor.reduce(ref.astype(np.uint32)) != got[tid]:
                    bad += 1
            out[f"{kb}kb"] = {"verdict": "PASS" if bad == 0 else "WRONG-DATA",
                              "detail": f"{bad}/1024 threads mismatch"}
        except Exception as e:
            out[f"{kb}kb"] = {"verdict": "FAULT", "detail": str(e)[:300]}
        print(f"[ds] {kb}KB dynamic smem: {out[f'{kb}kb']}", flush=True)
        fsync_json(OUT, res)

def run_l2bw(rig, res):
    import numpy as np
    if "l2_bw" in res:
        print("[bw] already done -- skip", flush=True)
        return
    pl = load_prog(rig, "MM_A_bwread.cubin", "bwread", 256)
    lg = load_prog(rig, "MM_A_bwread_ldg.cubin", "bwread", 256)
    out = res.setdefault("l2_bw", {})
    for tag, nfloats in (("dram_256mb", 64 * 1024 * 1024), ("l2_4mb", 1024 * 1024)):
        n4 = nfloats // 4
        g = rig.up(np.random.default_rng(7).standard_normal(nfloats).astype(np.float32))
        ob = rig.alloc(8192 * 4)
        row = {}
        for nm, prg in (("plain", pl), ("ldcg", lg)):
            t = graph_time(rig, prg, (g, ob), (n4 // 256, 1, 1), (n4,), reps=10)
            row[nm] = {"ms": round(t, 3), "gbs": round(nfloats * 4 / t / 1e6, 1)}
        out[tag] = row
        print(f"[bw] {tag}: plain {row['plain']['gbs']} GB/s, ldcg {row['ldcg']['gbs']} GB/s", flush=True)
        fsync_json(OUT, res)

if __name__ == "__main__":
    from MM_P7_lib import Rig7
    res = {}
    if os.path.exists(OUT):
        try: res = json.load(open(OUT))
        except Exception: res = {}
    print("[probe] booting Rig7...", flush=True)
    rig = Rig7(ctx_alloc=CTXK, load_p6=True)
    run_ehist(rig, res)
    run_l2bw(rig, res)
    print(f"[probe] done -> {OUT}", flush=True)
