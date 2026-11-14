"""Option D: the job progress file results/d/progress.json (Rule 6), rewritten from the work itself."""
import json
import os
import time

AIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.environ.get("D_PROGRESS", os.path.join(AIR, "results", "d", "progress.json"))


def write(job, step, done=None, total=None):
    rec = {"job": job, "step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if done is not None:
        rec.update(done=int(done), total=int(total))
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = f"{PATH}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, PATH)
