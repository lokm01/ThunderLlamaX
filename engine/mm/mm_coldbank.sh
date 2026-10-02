#!/bin/zsh
# MM SESSION C -- cold-boot bank60: restart the engine, wait for health,
# then IMMEDIATELY t1-then-spec bank60 (the r0 shape), then a warm rerun.
set -u
SELF=~/tinygrad-metal/engine0/ops/enginectl
sudo -n launchctl bootout system/com.lokm.llm-engine 2>/dev/null
sleep 5
sudo -n launchctl bootstrap system /Library/LaunchDaemons/com.lokm.llm-engine.plist
echo "[cold] waiting for health..."
for i in $(seq 1 90); do
  st=$(curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | /usr/bin/python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("status"))
except Exception: print("")')
  [ "$st" = "ok" ] && { echo "[cold] healthy after ~$((i*5))s"; break; }
  sleep 5
done
cd ~/tinygrad-metal/engine0/mm
~/tg311/bin/python - << 'PYEOF'
import json, sys
import numpy as np
sys.path.insert(0, "~/tinygrad-metal/engine0/mm")
from mm_g3c2 import Eng
from mm_daemon_c import dedup, gen_timed
doc = np.load("~/mm_p5_doc100k_ids.npy").astype(np.int32)
e = Eng()
out = {}
# COLD bank60: t1 FIRST THING after boot, then spec (the r0 shape)
e.prefill(doc[:4096], cid="cold_a"); ta, sa = gen_timed(e, "t1", "cold_a", 70)
e.prefill(doc[:4096], cid="cold_b"); tb, sb = gen_timed(e, "spec", "cold_b", 400)
n = min(len(ta), len(tb))
fd = next((i for i in range(n) if ta[i] != tb[i]), None)
out["cold_bank60"] = {"first_diff": fd, "n": n, "n_t1": len(ta), "n_spec": len(tb),
                      "t1_tok_s": round(len(ta) / sa, 1), "spec_tok_s": round(len(tb) / sb, 1)}
print(f"[cold] COLD bank60 t1-vs-spec first_diff={fd} (n={n}) "
      f"t1={out['cold_bank60']['t1_tok_s']} spec={out['cold_bank60']['spec_tok_s']} tok/s", flush=True)
# WARM rerun (same shape)
e.prefill(doc[:4096], cid="warm_a"); ta2, _ = gen_timed(e, "t1", "warm_a", 70)
e.prefill(doc[:4096], cid="warm_b"); tb2, _ = gen_timed(e, "spec", "warm_b", 400)
n2 = min(len(ta2), len(tb2))
fd2 = next((i for i in range(n2) if ta2[i] != tb2[i]), None)
out["warm_bank60"] = {"first_diff": fd2, "n": n2}
print(f"[cold] WARM bank60 first_diff={fd2} (n={n2})", flush=True)
json.dump(out, open("/tmp/mm_coldbank.json", "w"), indent=1)
PYEOF
