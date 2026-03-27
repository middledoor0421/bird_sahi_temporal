#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/v62_bucket_eval.py
# Post-evaluate detection/video KPIs by buckets using existing pred + v6 log.
# Python 3.9 compatible. Comments in English only.

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_log(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        txt = f.read().strip()
    if len(txt) == 0:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    # jsonlines
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def percentile_threshold(vals: List[float], q: float) -> Optional[float]:
    if len(vals) == 0:
        return None
    arr = np.asarray(vals, dtype=np.float32)
    return float(np.quantile(arr, q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucketed re-evaluation for v6 logs.")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--category-id", type=int, default=14)
    parser.add_argument("--pred-score-thr", type=float, default=0.25)
    parser.add_argument("--match-iou-thr", type=float, default=0.5)
    parser.add_argument("--lock-n", type=int, default=5)
    parser.add_argument("--small-max-area", type=float, default=1024.0)
    parser.add_argument("--small-track-frac", type=float, default=0.5)
    parser.add_argument("--motion-high-q", type=float, default=0.80, help="Top-q as high motion (default top 20%).")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logs = load_log(args.log)

    ids_all: List[int] = []
    lum_vals: List[float] = []
    motion_vals: List[float] = []
    per_id: Dict[int, Dict[str, Any]] = {}

    for r in logs:
        iid = int(r.get("image_id", -1))
        if iid < 0:
            continue
        ids_all.append(iid)

        lum = r.get("frame_mean_luminance", None)
        if lum is None:
            lum = r.get("frame_mean", None)
        if lum is not None:
            try:
                lum = float(lum)
            except Exception:
                lum = None

        mz = None
        ev = r.get("evidence", None)
        if isinstance(ev, dict):
            motion = ev.get("motion", None)
            if isinstance(motion, dict):
                mz = motion.get("global_z", None)
        if mz is not None:
            try:
                mz = float(mz)
            except Exception:
                mz = None

        cond = r.get("cond", {})
        seed_empty = bool(cond.get("seed_empty", False))
        no_det_bucket = str(cond.get("no_det_bucket", ""))
        no_det_streak = int(cond.get("no_det_streak", 0))
        hard = bool(seed_empty and ((no_det_bucket == "6p") or (no_det_streak >= 6)))

        per_id[iid] = {
            "lum": lum,
            "motion_z": mz,
            "hard": hard,
        }
        if lum is not None:
            lum_vals.append(lum)
        if mz is not None:
            motion_vals.append(mz)

    # Thresholds
    lum_thr = percentile_threshold(lum_vals, 0.50) if len(lum_vals) > 0 else None
    motion_thr = percentile_threshold(motion_vals, float(args.motion_high_q)) if len(motion_vals) > 0 else None

    buckets: Dict[str, List[int]] = {}
    buckets["overall"] = sorted(list(set(ids_all)))

    buckets["hard_case"] = sorted([iid for iid, d in per_id.items() if bool(d.get("hard", False))])

    if lum_thr is not None:
        buckets["brightness_low"] = sorted([iid for iid, d in per_id.items() if d.get("lum") is not None and float(d["lum"]) <= lum_thr])
        buckets["brightness_high"] = sorted([iid for iid, d in per_id.items() if d.get("lum") is not None and float(d["lum"]) > lum_thr])
    else:
        buckets["brightness_low"] = []
        buckets["brightness_high"] = []

    if motion_thr is not None:
        buckets["motion_low"] = sorted([iid for iid, d in per_id.items() if d.get("motion_z") is not None and float(d["motion_z"]) <= motion_thr])
        buckets["motion_high"] = sorted([iid for iid, d in per_id.items() if d.get("motion_z") is not None and float(d["motion_z"]) > motion_thr])
    else:
        buckets["motion_low"] = []
        buckets["motion_high"] = []

    meta = {
        "lum_median": lum_thr,
        "motion_high_thr_q": float(args.motion_high_q),
        "motion_high_thr": motion_thr,
        "counts": {k: len(v) for k, v in buckets.items()},
        "note": "brightness buckets require frame_mean_luminance in log; if missing, brightness buckets are empty.",
    }
    save_json(os.path.join(args.out_dir, "bucket_meta.json"), meta)

    summary: Dict[str, Any] = {"meta": meta, "buckets": {}}

    for bname, ids in buckets.items():
        bdir = os.path.join(args.out_dir, bname)
        os.makedirs(bdir, exist_ok=True)
        ids_path = os.path.join(bdir, "image_ids.json")
        save_json(ids_path, ids)

        # Detection metrics (COCO)
        det_out = os.path.join(bdir, "metrics_det.json")
        cmd_det = [
            sys.executable, "scripts/eval_coco.py",
            "--gt", args.gt,
            "--pred", args.pred,
            "--out", det_out,
            "--small-max-area", str(args.small_max_area),
            "--image-ids-json", ids_path,
        ]
        subprocess.run(cmd_det, check=True)

        # Video metrics (pseudo tracks)
        vid_out = os.path.join(bdir, "metrics_vid.json")
        cmd_vid = [
            sys.executable, "scripts/eval_video_kpi.py",
            "--gt", args.gt,
            "--pred", args.pred,
            "--out", vid_out,
            "--category-id", str(args.category_id),
            "--pred-score-thr", str(args.pred_score_thr),
            "--match-iou-thr", str(args.match_iou_thr),
            "--lock-n", str(args.lock_n),
            "--small-max-area", str(args.small_max_area),
            "--small-track-frac", str(args.small_track_frac),
            "--image-ids-json", ids_path,
        ]
        subprocess.run(cmd_vid, check=True)

        summary["buckets"][bname] = {
            "image_count": int(len(ids)),
            "metrics_det": det_out,
            "metrics_vid": vid_out,
        }

    save_json(os.path.join(args.out_dir, "bucket_summary.json"), summary)
    print("Saved bucket summary:", os.path.join(args.out_dir, "bucket_summary.json"))


if __name__ == "__main__":
    main()
