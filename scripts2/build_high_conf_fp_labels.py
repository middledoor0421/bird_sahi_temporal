#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def tau_key(tau: float) -> str:
    return str(tau).replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Step 3 high-confidence FP labels from Step 2 prevalence input."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rows = load_jsonl_gz(args.input)
    out_jsonl = os.path.join(args.out_dir, "high_conf_fp_labels.jsonl.gz")
    out_summary = os.path.join(args.out_dir, "high_conf_fp_summary.json")

    summary: Dict[str, Any] = {
        "input": os.path.abspath(args.input),
        "taus": [float(t) for t in args.taus],
        "num_rows": 0,
        "num_empty_rows": 0,
        "by_tau": {},
    }

    for tau in args.taus:
        summary["by_tau"][str(tau)] = {
            "num_empty_frames": 0,
            "num_empty_fp_frames": 0,
            "num_empty_sahi_on_frames": 0,
            "num_empty_fp_on_frames": 0,
            "num_empty_sahi_off_frames": 0,
            "num_empty_fp_off_frames": 0,
            "sum_empty_fp_count": 0,
            "max_fp_conf_values": [],
        }

    with gzip.open(out_jsonl, "wt", encoding="utf-8") as f:
        for row in rows:
            is_empty = bool(row.get("is_empty", False))
            sahi_executed = int(row.get("sahi_executed", 0))
            preds = row.get("preds", [])
            if not isinstance(preds, list):
                preds = []
            scores = []
            for pred in preds:
                if not isinstance(pred, dict):
                    continue
                try:
                    scores.append(float(pred.get("score", 0.0)))
                except Exception:
                    continue
            max_pred_score = max(scores) if scores else 0.0

            rec = dict(row)
            rec["max_pred_score"] = float(max_pred_score)
            rec["num_preds_total"] = int(len(scores))

            summary["num_rows"] += 1
            if is_empty:
                summary["num_empty_rows"] += 1

            for tau in args.taus:
                key = tau_key(float(tau))
                pred_count_ge = sum(1 for s in scores if s >= float(tau))
                has_pred_ge = pred_count_ge > 0
                has_high_conf_fp = bool(is_empty and has_pred_ge)
                fp_count = int(pred_count_ge) if is_empty else 0
                max_fp_conf = float(max_pred_score) if has_high_conf_fp else 0.0

                rec["pred_count_ge_tau_{}".format(key)] = int(pred_count_ge)
                rec["has_pred_ge_tau_{}".format(key)] = bool(has_pred_ge)
                rec["has_high_conf_fp_tau_{}".format(key)] = bool(has_high_conf_fp)
                rec["fp_count_tau_{}".format(key)] = int(fp_count)
                rec["max_fp_conf_tau_{}".format(key)] = float(max_fp_conf)

                bucket = summary["by_tau"][str(tau)]
                if is_empty:
                    bucket["num_empty_frames"] += 1
                    if sahi_executed == 1:
                        bucket["num_empty_sahi_on_frames"] += 1
                    else:
                        bucket["num_empty_sahi_off_frames"] += 1
                    if has_high_conf_fp:
                        bucket["num_empty_fp_frames"] += 1
                        bucket["sum_empty_fp_count"] += int(fp_count)
                        bucket["max_fp_conf_values"].append(float(max_fp_conf))
                        if sahi_executed == 1:
                            bucket["num_empty_fp_on_frames"] += 1
                        else:
                            bucket["num_empty_fp_off_frames"] += 1

            f.write(json.dumps(rec) + "\n")

    for tau in args.taus:
        bucket = summary["by_tau"][str(tau)]
        empty_frames = int(bucket["num_empty_frames"])
        empty_on = int(bucket["num_empty_sahi_on_frames"])
        empty_off = int(bucket["num_empty_sahi_off_frames"])
        empty_fp = int(bucket["num_empty_fp_frames"])
        empty_fp_on = int(bucket["num_empty_fp_on_frames"])
        empty_fp_off = int(bucket["num_empty_fp_off_frames"])
        bucket["prevalence_on_empty"] = float(empty_fp) / float(empty_frames) if empty_frames > 0 else 0.0
        bucket["prevalence_on_empty_given_sahi_on"] = (
            float(empty_fp_on) / float(empty_on) if empty_on > 0 else 0.0
        )
        bucket["prevalence_on_empty_given_sahi_off"] = (
            float(empty_fp_off) / float(empty_off) if empty_off > 0 else 0.0
        )
        bucket["mean_fp_count_on_empty"] = (
            float(bucket["sum_empty_fp_count"]) / float(empty_frames) if empty_frames > 0 else 0.0
        )
        vals = bucket.pop("max_fp_conf_values", [])
        bucket["mean_max_fp_conf_on_empty"] = (
            float(sum(vals)) / float(len(vals)) if len(vals) > 0 else 0.0
        )

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_jsonl)
    print("Saved:", out_summary)


if __name__ == "__main__":
    main()
