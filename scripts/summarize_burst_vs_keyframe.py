# scripts/summarize_burst_vs_keyframe.py
# Python 3.9 compatible

import json
import glob
import os
import math
from collections import defaultdict

def mean_std(vals):
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

ROOT = "output/burst_cmp/eval"
SPLITS = ["sparse_20", "sparse_15", "sparse_10"]
METHODS = ["keyframe_n2", "v6_exp60", "v6_exp61", "v6_exp62"]

# metric keys (adjust if naming slightly differs)
DET_KEY = "AR_small"
VID_KEY = "small_track_init_recall"

print("\n=== Summary: mean ± std over seeds (0/1/2) ===\n")

for split in SPLITS:
    print(f"[{split}]")
    for method in METHODS:
        det_vals = []
        vid_vals = []

        det_files = glob.glob(os.path.join(
            ROOT, f"metrics_det_{method}_{split}_seed*.json"
        ))
        vid_files = glob.glob(os.path.join(
            ROOT, f"metrics_vid_{method}_{split}_seed*.json"
        ))

        for p in det_files:
            j = load_json(p)
            if DET_KEY in j:
                det_vals.append(float(j[DET_KEY]))

        for p in vid_files:
            j = load_json(p)
            if VID_KEY in j:
                vid_vals.append(float(j[VID_KEY]))

        det_m, det_s = mean_std(det_vals)
        vid_m, vid_s = mean_std(vid_vals)

        if det_m is None or vid_m is None:
            print(f"  {method:12s}: missing data")
            continue

        print(
            f"  {method:12s} | "
            f"AR_small = {det_m:.4f} ± {det_s:.4f} | "
            f"track-init = {vid_m:.4f} ± {vid_s:.4f}"
        )
    print()
