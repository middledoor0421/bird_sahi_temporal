#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COCO evaluation script (single source of truth).

Outputs:
- AP (IoU=0.50:0.95)  -> mAP
- AP50
- AP_small (custom small area)
- AR_small (custom small area)  -> Recall(small)-like

Requirements:
  pip install pycocotools
"""

import argparse
import json
from typing import Any, Dict, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_image_ids(path: str) -> List[int]:
    """Load image_ids from a json file.

    Supported formats:
      1) {"image_ids": [1,2,...], ...}
      2) [1,2,...]
    """
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "image_ids" in obj:
        obj = obj["image_ids"]
    if not isinstance(obj, list):
        raise ValueError("Invalid image ids json. Expect a list or a dict with 'image_ids'.")
    out: List[int] = []
    for x in obj:
        out.append(int(x))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="COCOeval for bbox predictions.")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT COCO json.")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction json.")
    parser.add_argument("--image-ids-json", type=str, default=None, help="Optional json list of image_ids to evaluate.")
    parser.add_argument("--out", type=str, default=None, help="Optional output metrics json.")
    parser.add_argument(
        "--small-max-area",
        type=float,
        default=32.0 * 32.0,
        help="Max bbox area (pixels^2) for 'small'. Default 32^2.",
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=100,
        help="maxDets for AR metrics. Default 100.",
    )
    args = parser.parse_args()

    coco_gt = COCO(args.gt)

    # Robustness: ensure required top-level keys exist
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    with open(args.pred, "r") as f:
        pred_data = json.load(f)

    image_ids_filter = None
    if args.image_ids_json:
        image_ids_filter = set(_load_image_ids(args.image_ids_json))
    if image_ids_filter is not None:
        coco_gt.dataset["images"] = [im for im in coco_gt.dataset.get("images", []) if
                                     int(im.get("id", -1)) in image_ids_filter]
        coco_gt.dataset["annotations"] = [a for a in coco_gt.dataset.get("annotations", []) if
                                          int(a.get("image_id", -1)) in image_ids_filter]
        coco_gt.createIndex()

    if image_ids_filter is not None:
        pred_data = [p for p in pred_data if int(p.get("image_id", -1)) in image_ids_filter]

    if not isinstance(pred_data, list):
        raise ValueError("Prediction json must be a list (COCO results format).")

    if len(pred_data) == 0:
        metrics = {
            "AP": 0.0,
            "AP50": 0.0,
            "AP_small": 0.0,
            "AR_small": 0.0,
            "small_max_area": float(args.small_max_area),
            "max_dets": int(args.max_dets),
            "note": "Empty predictions.",
        }
        print(metrics)
        if args.out:
            save_json(args.out, metrics)
        return

    pred_path_for_eval = args.pred
    if image_ids_filter is not None:
        tmp_path = (args.out + ".tmp_filtered_pred.json") if args.out else "tmp_filtered_pred.json"
        with open(tmp_path, "w") as f:
            json.dump(pred_data, f)
        pred_path_for_eval = tmp_path

    coco_dt = coco_gt.loadRes(pred_path_for_eval)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")

    small_max_area = float(args.small_max_area)
    evaluator.params.areaRng = [
        [0.0, 1e12],  # all
        [0.0, small_max_area],  # small (custom)
        [small_max_area, 1e12],  # large (rest)
    ]
    evaluator.params.areaRngLbl = ["all", "small", "large"]

    max_dets = int(args.max_dets)
    evaluator.params.maxDets = [1, 10, max_dets]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP_small": float(evaluator.stats[3]),
        "AR_small": float(evaluator.stats[9]),
        "small_max_area": small_max_area,
        "max_dets": max_dets,
    }

    print("\n=== Metrics ===")
    print("AP       :", metrics["AP"])
    print("AP50     :", metrics["AP50"])
    print("AP_small :", metrics["AP_small"])
    print("AR_small :", metrics["AR_small"])

    if args.out:
        save_json(args.out, metrics)
        print("Saved metrics to:", args.out)


if __name__ == "__main__":
    main()
