#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_first_jsonl_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
    raise ValueError("No valid json row found: {}".format(path))


def discover_step3_dirs(step3_root: str) -> List[str]:
    dirs: List[str] = []
    for name in sorted(os.listdir(step3_root)):
        path = os.path.join(step3_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "high_conf_fp_summary.json")) and os.path.isfile(
            os.path.join(path, "high_conf_fp_labels.jsonl.gz")
        ):
            dirs.append(path)
    return dirs


def safe_ratio(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num) / float(den)


def maybe_ratio(num: float, den: float) -> Any:
    if float(den) == 0.0:
        return None
    return float(num) / float(den)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Step 4 conditional prevalence comparison tables from Step 3 outputs."
    )
    parser.add_argument("--step3-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    step3_dirs = discover_step3_dirs(args.step3_root)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for step3_dir in step3_dirs:
        summary_path = os.path.join(step3_dir, "high_conf_fp_summary.json")
        labels_path = os.path.join(step3_dir, "high_conf_fp_labels.jsonl.gz")
        summary = load_json(summary_path)
        first = load_first_jsonl_gz(labels_path)
        dataset = str(first.get("dataset", ""))
        split = str(first.get("split", ""))
        method = str(first.get("method", ""))
        category_mode = str(first.get("category_mode", "all_category"))

        for tau_str, bucket in summary.get("by_tau", {}).items():
            prev_all = float(bucket.get("prevalence_on_empty", 0.0))
            prev_on = float(bucket.get("prevalence_on_empty_given_sahi_on", 0.0))
            prev_off = float(bucket.get("prevalence_on_empty_given_sahi_off", 0.0))
            row = {
                "dataset": dataset,
                "split": split,
                "method": method,
                "category_mode": category_mode,
                "tau": float(tau_str),
                "num_empty_frames": int(bucket.get("num_empty_frames", 0)),
                "num_empty_fp_frames": int(bucket.get("num_empty_fp_frames", 0)),
                "num_empty_sahi_on_frames": int(bucket.get("num_empty_sahi_on_frames", 0)),
                "num_empty_fp_on_frames": int(bucket.get("num_empty_fp_on_frames", 0)),
                "num_empty_sahi_off_frames": int(bucket.get("num_empty_sahi_off_frames", 0)),
                "num_empty_fp_off_frames": int(bucket.get("num_empty_fp_off_frames", 0)),
                "prevalence_on_empty": prev_all,
                "prevalence_on_empty_given_sahi_on": prev_on,
                "prevalence_on_empty_given_sahi_off": prev_off,
                "relative_amplification_ratio": maybe_ratio(prev_on, prev_off),
                "relative_amplification_diff": float(prev_on - prev_off),
                "mean_fp_count_on_empty": float(bucket.get("mean_fp_count_on_empty", 0.0)),
                "mean_max_fp_conf_on_empty": float(bucket.get("mean_max_fp_conf_on_empty", 0.0)),
                "step3_dir": step3_dir,
            }
            grouped[(dataset, split, category_mode)].append(row)

    for (dataset, split, category_mode), rows in grouped.items():
        rows.sort(key=lambda r: (float(r["tau"]), str(r["method"])))
        tag = "{}_{}_{}".format(dataset, split, category_mode)
        out_csv = os.path.join(args.out_dir, "conditional_prevalence_{}.csv".format(tag))
        out_json = os.path.join(args.out_dir, "conditional_prevalence_{}.json".format(tag))
        out_md = os.path.join(args.out_dir, "conditional_prevalence_{}.md".format(tag))

        fieldnames = [
            "dataset",
            "split",
            "method",
            "category_mode",
            "tau",
            "num_empty_frames",
            "num_empty_fp_frames",
            "num_empty_sahi_on_frames",
            "num_empty_fp_on_frames",
            "num_empty_sahi_off_frames",
            "num_empty_fp_off_frames",
            "prevalence_on_empty",
            "prevalence_on_empty_given_sahi_on",
            "prevalence_on_empty_given_sahi_off",
            "relative_amplification_ratio",
            "relative_amplification_diff",
            "mean_fp_count_on_empty",
            "mean_max_fp_conf_on_empty",
            "step3_dir",
        ]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)

        lines: List[str] = []
        lines.append("# Conditional Prevalence: {} {} ({})".format(dataset, split, category_mode))
        lines.append("")
        for tau in sorted(set(float(r["tau"]) for r in rows)):
            lines.append("## tau = {:.1f}".format(tau))
            lines.append("")
            lines.append("| Method | Prev(empty) | Prev(empty|on) | Prev(empty|off) | Amplification ratio | Amplification diff | Mean FP/empty | Mean max FP conf |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            tau_rows = [r for r in rows if float(r["tau"]) == float(tau)]
            for r in tau_rows:
                ratio = r["relative_amplification_ratio"]
                ratio_s = "{:.4f}".format(ratio) if isinstance(ratio, float) else "NA"
                lines.append(
                    "| {method} | {p0:.4f} | {p1:.4f} | {p2:.4f} | {ratio} | {diff:.4f} | {mfp:.4f} | {mconf:.4f} |".format(
                        method=r["method"],
                        p0=float(r["prevalence_on_empty"]),
                        p1=float(r["prevalence_on_empty_given_sahi_on"]),
                        p2=float(r["prevalence_on_empty_given_sahi_off"]),
                        ratio=ratio_s,
                        diff=float(r["relative_amplification_diff"]),
                        mfp=float(r["mean_fp_count_on_empty"]),
                        mconf=float(r["mean_max_fp_conf_on_empty"]),
                    )
                )
            lines.append("")

        with open(out_md, "w") as f:
            f.write("\n".join(lines).strip() + "\n")

        print("Saved:", out_csv)
        print("Saved:", out_json)
        print("Saved:", out_md)


if __name__ == "__main__":
    main()
