#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/summarize_all_eval.py
# Aggregate detection KPI (COCOeval) + video KPI into one JSON/CSV table.
# Python 3.9 compatible. Comments in English only.

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def list_methods(out_dir: str) -> List[str]:
    methods = set()
    for fn in os.listdir(out_dir):
        if fn.startswith("metrics_det_") and fn.endswith(".json"):
            methods.add(fn[len("metrics_det_"):-len(".json")])
    return sorted(list(methods))


def read_det_metrics(path: str) -> Dict[str, Any]:
    d = load_json(path)
    # Your eval_coco.py writes keys like AP, AP50, AP_small, AR_small
    return {
        "AP": d.get("AP", None),
        "AP50": d.get("AP50", None),
        "AP_small": d.get("AP_small", None),
        "AR_small": d.get("AR_small", None),
    }


def read_vid_metrics(path: str) -> Dict[str, Any]:
    d = load_json(path)
    overall = safe_get(d, ["kpi", "overall"], {})
    small = safe_get(d, ["kpi", "small_only"], {})
    lock_n = safe_get(d, ["kpi", "lock_n"], None)

    return {
        "lock_n": lock_n,
        "trk_init_overall": overall.get("track_init_recall", None),
        "ttd_p50_overall": overall.get("ttd_p50", None),
        "ttd_p90_overall": overall.get("ttd_p90", None),
        "miss_rate_overall": overall.get("miss_rate", None),
        "lock_overall": overall.get("sustained_lock_rate", None),
        "trk_init_small": small.get("track_init_recall", None),
        "ttd_p50_small": small.get("ttd_p50", None),
        "ttd_p90_small": small.get("ttd_p90", None),
        "miss_rate_small": small.get("miss_rate", None),
        "lock_small": small.get("sustained_lock_rate", None),
    }


def read_stats(path: str) -> Dict[str, Any]:
    d = load_json(path)
    return {
        "fps": d.get("fps", None),
        "latency_ms": d.get("latency_ms", None),
        "avg_selected": d.get("avg_selected", d.get("avg_selected_tiles", None)),
        "sahi_ratio": d.get("sahi_ratio", d.get("do_sahi_ratio", None)),
        "full_frames": d.get("full_frames", None),
        "subset_frames": d.get("subset_frames", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all_eval_run outputs into a table.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    out_dir = args.out_dir
    methods = list_methods(out_dir)

    rows: List[Dict[str, Any]] = []
    for m in methods:
        det_path = os.path.join(out_dir, f"metrics_det_{m}.json")
        vid_path = os.path.join(out_dir, f"metrics_vid_{m}.json")
        st_path = os.path.join(out_dir, f"stats_{m}.json")

        row: Dict[str, Any] = {"method": m}

        if os.path.exists(det_path):
            row.update(read_det_metrics(det_path))
        if os.path.exists(vid_path):
            row.update(read_vid_metrics(vid_path))
        if os.path.exists(st_path):
            row.update(read_stats(st_path))

        rows.append(row)

    # Print table-like JSON to stdout
    print(json.dumps(rows, indent=2))

    if args.out_json.strip() != "":
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)

    if args.out_csv.strip() != "":
        # Choose a stable column order
        cols = [
            "method",
            "AP", "AP50", "AP_small", "AR_small",
            "fps", "latency_ms", "avg_selected", "sahi_ratio", "full_frames", "subset_frames",
            "lock_n",
            "trk_init_overall", "ttd_p50_overall", "ttd_p90_overall", "miss_rate_overall", "lock_overall",
            "trk_init_small", "ttd_p50_small", "ttd_p90_small", "miss_rate_small", "lock_small",
        ]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, None) for c in cols})


if __name__ == "__main__":
    main()
