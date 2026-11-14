"""Walking scores of Step 2's saved policies (cloud copy of the Mac's actors; the project's protocol)."""
import sys, json, os
import mjeval
names = sys.argv[2:]
out = sys.argv[1]
done = set()
if os.path.exists(out):
    done = {json.loads(l)["name"] for l in open(out) if l.strip()}
for n in names:
    if n in done: continue
    r = mjeval.score(f"/mnt/user-data/uploads/PDM/AIR 2027/runs/s2/{n}/actor.pt")
    r["name"] = n; r.pop("policy", None)
    fs = r.pop("fall_steps"); r["median_fall_step"] = (sorted(fs)[len(fs) // 2] if fs else None)
    with open(out, "a") as f: f.write(json.dumps(r) + "\n")
    print(n, round(r["score"], 4), round(r["falls"], 3), flush=True)
