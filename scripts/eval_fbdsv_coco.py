#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate YOLO / SAHI-Full / SAHI-Temporal predictions on FBD-SV-2024 (VID) using COCO metrics.

Prerequisites:
    pip install pycocotoolsSS

Assumed files under data_root/annotations:

  Ground truth:
    - fbdsv24_vid_{split}.json

  Predictions:
    - pred_yolo_fbdsv24_vid_{split}.json
    - pred_sahi_full_fbdsv24_vid_{split}.json
    - pred_sahi_temporal_fbdsv24_vid_{split}.json

  Optional stats (from runners):
    - stats_yolo_fbdsv24_vid_{split}.json
    - stats_sahi_full_fbdsv24_vid_{split}.json
    - stats_sahi_temporal_fbdsv24_vid_{split}.json

Outputs:
    - eval_yolo_fbdsv24_vid_{split}.json
    - eval_sahi_full_fbdsv24_vid_{split}.json
    - eval_sahi_temporal_fbdsv24_vid_{split}.json

Each eval json contains COCO metrics and, if available, merged runtime stats.
"""

import os
import json
import argparse
from typing import Dict, Any, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_json(path: str) -> Any:
    """Load json file if exists, else return None."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def coco_evaluate(
    gt_path: str,
    pred_path: str,
    iou_type: str = "bbox",
) -> Dict[str, float]:
    """
    Run COCO evaluation for given ground truth and prediction files.

    Returns a dict of selected metrics (AP, AP50, AP75, APs, APm, APl).
    """
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats is a length-12 array for bbox:
    # 0: AP (IoU=0.50:0.95, area=all, maxDets=100)
    # 1: AP50
    # 2: AP75
    # 3: AP_small
    # 4: AP_medium
    # 5: AP_large
    # 6: AR_max1
    # 7: AR_max10
    # 8: AR_max100
    # 9: AR_small
    # 10: AR_medium
    # 11: AR_large
    s = coco_eval.stats
    metrics = {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR_max1": float(s[6]),
        "AR_max10": float(s[7]),
        "AR_max100": float(s[8]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
    }
    return metrics


def merge_metrics_and_stats(
    method_name: str,
    split: str,
    metrics: Dict[str, float],
    stats: Any,
) -> Dict[str, Any]:
    """
    Merge COCO metrics with runtime stats (if provided).
    """
    merged: Dict[str, Any] = {
        "method": method_name,
        "split": split,
    }
    merged.update(metrics)

    if isinstance(stats, dict):
        # Avoid overwriting method/split keys from metrics side
        for k, v in stats.items():
            if k in ("method", "split"):
                continue
            merged[k] = v

    return merged


def evaluate_methods(
    data_root: str,
    split: str,
    methods: List[str],
) -> None:
    """
    Evaluate selected methods and save per-method eval jsons.
    """
    data_root = os.path.abspath(data_root)
    ann_root = os.path.join(data_root, "annotations")

    gt_path = os.path.join(ann_root, "fbdsv24_vid_{}.json".format(split))
    if not os.path.isfile(gt_path):
        raise FileNotFoundError("Ground truth json not found: {}".format(gt_path))

    method_to_pred_name = {
        "yolo": "pred_yolo_fbdsv24_vid_{}.json".format(split),
        "sahi_full": "pred_sahi_full_fbdsv24_vid_{}.json".format(split),
        "sahi_temporal": "pred_sahi_temporal_fbdsv24_vid_{}.json".format(split),
    }
    method_to_stats_name = {
        "yolo": "stats_yolo_fbdsv24_vid_{}.json".format(split),
        "sahi_full": "stats_sahi_full_fbdsv24_vid_{}.json".format(split),
        "sahi_temporal": "stats_sahi_temporal_fbdsv24_vid_{}.json".format(split),
    }
    method_to_eval_name = {
        "yolo": "eval_yolo_fbdsv24_vid_{}.json".format(split),
        "sahi_full": "eval_sahi_full_fbdsv24_vid_{}.json".format(split),
        "sahi_temporal": "eval_sahi_temporal_fbdsv24_vid_{}.json".format(split),
    }

    all_summary: List[Dict[str, Any]] = []

    for method in methods:
        if method not in method_to_pred_name:
            print("Skipping unknown method:", method)
            continue

        pred_path = os.path.join(ann_root, method_to_pred_name[method])
        if not os.path.isfile(pred_path):
            print("Prediction file for method '{}' not found: {}".format(method, pred_path))
            continue

        print("=" * 80)
        print("Evaluating method '{}' on split '{}'".format(method, split))
        print("GT:", gt_path)
        print("Pred:", pred_path)

        metrics = coco_evaluate(gt_path, pred_path, iou_type="bbox")

        stats_path = os.path.join(ann_root, method_to_stats_name[method])
        stats = load_json(stats_path)

        merged = merge_metrics_and_stats(method, split, metrics, stats)
        all_summary.append(merged)

        eval_path = os.path.join(ann_root, method_to_eval_name[method])
        with open(eval_path, "w") as f:
            json.dump(merged, f, indent=2)
        print("Saved eval summary for '{}' to: {}".format(method, eval_path))

        # Simple one-line summary
        print(
            "[{}] AP={:.4f}, AP50={:.4f}, AP75={:.4f}, "
            "APs={:.4f}, fps={}".format(
                method,
                merged["AP"],
                merged["AP50"],
                merged["AP75"],
                merged["AP_small"],
                merged.get("fps", "NA"),
            )
        )

    # Optionally save a combined summary table for all methods
    combined_path = os.path.join(ann_root, "eval_all_methods_fbdsv24_vid_{}.json".format(split))
    with open(combined_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print("=" * 80)
    print("Saved combined summary for all methods to:", combined_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO / SAHI-Full / SAHI-Temporal on FBD-SV-2024 VID using COCO metrics."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to FBD-SV-2024 root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="yolo,sahi_full,sahi_temporal",
        help="Comma-separated list of methods to evaluate. "
             "Supported: yolo,sahi_full,sahi_temporal",
    )

    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    evaluate_methods(
        data_root=args.data_root,
        split=args.split,
        methods=methods,
    )


if __name__ == "__main__":
    main()
