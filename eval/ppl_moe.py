#!/usr/bin/env python3
"""P9 EVAL — teacher-forced NLL/perplexity on the MoE rig (Rig7, standalone).

Boots EXACTLY like test_moe36.py minus the MTP layer and the serve attach:
Rig7 + the PF-256 chunk graph (+ T1 graph for nothing — tail unused since we
only score full 256-chunks). Then for each corpus domain:
  feed 256-token chunks via pf_ids_view/pos_view + gr_pf.step(),
  and per seat run the proven eager head pair
      rmsz2048g(PFB['hA'][seat], ONORM, normhb)
      h6k2048(HEAD, normhb, logitsb, vals=(248320,))
  download logits (fp32), accumulate NLL of the NEXT token in float64.

Seat s of chunk c = hidden at position 256*c+s (pre-final-norm, the
eager_head_cur(pf=True) contract) -> predicts ids[256*c+s+1]. The final
seat of the final chunk has no target and is skipped. No BOS is prepended
(Qwen tokenizer adds none).

Run wrapper: eval/ppl_run.sh moe   (sources env.common + the MoE model env)
Output: eval/results/ppl_moe.json (+ stdout progress).
"""
import os, sys, time, json
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
os.environ.setdefault("DEV", "NV")

DATA = BASE + "/eval/data"
OUT = BASE + "/eval/results/ppl_moe.json"

CTXK = int(os.getenv("MM_CTXS", "98304"))
DEC_S = int(os.getenv("MM_DEC_S", "32"))
PF_S = int(os.getenv("MM_PF_S", "8"))
RUNG = CTXK
CH = 256

PROSE_TARGET = int(os.getenv("PPL_PROSE_TOK", "32768"))   # multiple of 256
CODE_TARGET = int(os.getenv("PPL_CODE_TOK", "10240"))     # multiple of 256
PRIVATE_TARGET = int(os.getenv("PPL_PRIVATE_TOK", "12288"))  # multiple of 256


DOMAINS_ALL = ("prose", "code", "prose_private", "code2")


def corpus_domains():
    """Domain list via DOMAINS env (comma-separated; default prose,code).
    prose/code ids come from eval/data/ppl_<name>_ids.json (tok_prep.py)."""
    want = [d.strip() for d in os.getenv("DOMAINS", "prose,code").split(",") if d.strip()]
    targets = {"prose": PROSE_TARGET, "code": CODE_TARGET,
               "prose_private": PRIVATE_TARGET, "code2": CODE_TARGET}
    return [(d, None, targets[d]) for d in want if d in DOMAINS_ALL]


def logsumexp(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def main():
    from MM_P7_lib import Rig7, build_seq7, mkgraph

    print("[ppl_moe] booting Rig7 ...", flush=True)
    t0 = time.perf_counter()
    rig = Rig7(ctx_alloc=CTXK, load_p6=True)
    print(f"[ppl_moe] rig up {time.perf_counter()-t0:.0f}s; building PF graph...", flush=True)
    spk = f"s{RUNG}"
    seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False,
                       spk=spk, S=PF_S, pf=True)
    gr_pf = mkgraph(rig, seqpf, "sv_pf")
    dev = rig.dev
    dev.synchronize()
    print(f"[ppl_moe] graphs built {time.perf_counter()-t0:.0f}s", flush=True)

    results = {}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))   # merge across partial runs
        except Exception:
            results = {}
    for name, text, target in corpus_domains():
        ids = [int(t) for t in json.load(open(f"{DATA}/ppl_{name}_ids.json"))]
        n = min(len(ids), target)
        n = (n // CH) * CH
        ids = [int(t) for t in ids[:n]]
        print(f"[ppl_moe] {name}: {len(ids)} tokens ({CH}-chunks x {len(ids)//CH})", flush=True)

        nll = 0.0
        cnt = 0
        greedy = 0
        tA = time.perf_counter()
        arr32 = np.empty(CH, dtype=np.int32)
        for p in range(0, n, CH):
            arr32[:] = ids[p:p + CH]
            rig.pf_ids_view[:] = memoryview(arr32.data)
            rig.pos_view[0] = p
            gr_pf.step()
            dev.synchronize()          # chunk complete before reading hA seats
            last = CH - 1 if p + CH >= n else CH
            for s in range(last):
                tgt = ids[p + s + 1]
                hin = rig.PFB["hA"].offset(offset=s * 2048 * 4, size=2048 * 4)
                rig.K["rmsz2048g"](hin, rig.ONORM, rig.normhb,
                                   global_size=(1, 1, 1), local_size=(256, 1, 1), wait=True)
                rig.K["h6k2048"](rig.HEAD, rig.normhb, rig.logitsb,
                                 global_size=(7760, 1, 1), local_size=(1024, 1, 1),
                                 vals=(248320,), wait=True)
                lg = rig.dn(rig.logitsb, (248320,)).astype(np.float64)
                nll += logsumexp(lg) - lg[tgt]
                greedy += int(np.argmax(lg) == tgt)
                cnt += 1
            if (p // CH) % 8 == 0:
                el = time.perf_counter() - tA
                print(f"[ppl_moe] {name} pos={p+CH}/{n} nll/token={nll/max(1,cnt):.4f} "
                      f"greedy={greedy/max(1,cnt):.3f} rate={cnt/el:.1f} tok/s", flush=True)
        dt = time.perf_counter() - tA
        res = {
            "n_tokens": cnt,
            "nll_per_token": round(nll / max(1, cnt), 6),
            "ppl": round(float(np.exp(nll / max(1, cnt))), 4),
            "greedy_acc": round(greedy / max(1, cnt), 4),
            "wall_s": round(dt, 1),
            "score_rate_tps": round(cnt / dt, 1),
        }
        results[name] = res
        print(f"[ppl_moe] {name} DONE {json.dumps(res)}", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)

    print("[ppl_moe] ALL DONE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
