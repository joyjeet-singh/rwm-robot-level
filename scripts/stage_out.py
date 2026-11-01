"""
Copy this folder's deliverables to the outputs area for transfer to the Mac, with the git history as one bundle file.
Prints the list of (staged path, device path) pairs as JSON. Build products under paper/build are left out.
"""
import json
import os
import shutil
import subprocess

import common as C

OUT = "/mnt/user-data/outputs/air2027"
DEVICE = "/Users/joyjeetsingh/Downloads/PDM/AIR 2027"
SKIP_DIRS = {".git", "build", "__pycache__"}

if __name__ == "__main__":
    subprocess.run(["git", "bundle", "create", "history.bundle", "--all"], cwd=C.AIR, check=True,
                   capture_output=True)
    pairs = []
    for root, dirs, files in os.walk(C.AIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            src = os.path.join(root, f)
            relp = os.path.relpath(src, C.AIR)
            dst = os.path.join(OUT, relp)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            pairs.append({"stagedPath": dst, "devicePath": os.path.join(DEVICE, relp)})
    os.remove(os.path.join(C.AIR, "history.bundle"))
    json.dump(pairs, open(os.path.join(os.path.dirname(OUT), "air2027_pairs.json"), "w"), indent=0)
    print(len(pairs), "files staged")
