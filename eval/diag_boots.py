import re
lines = open("~/tinygrad-metal/logs/llm-engine-launchd.log", errors="replace").read().splitlines()
boots = []
cur = None
for l in lines:
    if '"stage": "daemon_attached"' in l:
        m = re.search(r'"ts": ([0-9.]+)', l)
        cur = {"attach": float(m.group(1)) if m else 0, "parks": [], "rebuilds": [], "done_stop": 0}
        boots.append(cur)
    if cur is None:
        continue
    m = re.search(r'"ts": ([0-9.]+).*?"op": "([a-z_0-9]+)"', l)
    if not m:
        continue
    ts, op = float(m.group(1)), m.group(2)
    if op == "t1_park":
        c = re.search(r'"cycles": (\d+)', l)
        if c:
            cur["parks"].append((ts, int(c.group(1))))
    elif op == "gen_rebuild":
        cur["rebuilds"].append(ts)
    elif op == "generate" and '"stage": "done_stop"' in l:
        cur["done_stop"] += 1
for b in boots:
    if b["attach"] < 1790667000:
        continue
    parks = b["parks"]
    last_cyc = parks[-1][1] if parks else None
    print("attach %.0f gens_done=%d rebuilds=%d n_parks=%d last_park_cycles=%s" %
          (b["attach"], b["done_stop"], len(b["rebuilds"]), len(parks), last_cyc))
    if parks:
        print("   park cycles seq:", [c for _, c in parks[-10:]])
