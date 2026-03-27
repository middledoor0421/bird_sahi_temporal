#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/eval_coco_filtered.py
# COCOeval on filtered frames (selected by eval_set track_ids).
# Python 3.9 compatible. Comments in English only.

import argparse
import json
from typing import Any, Dict, List, Set

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_ap_ar(coco_eval: COCOeval, area_idx: int, maxdet_idx: int, iou_value: float = None) -> float:
    """
    Compute AP from coco_eval.eval['precision'].
    precision dims: [T, R, K, A, M]
    If iou_value is None -> average over all IoU thresholds (COCO style).
    If iou_value is 0.5 -> AP50 (single IoU slice).
    """
    p = coco_eval.eval["precision"]
    if p is None:
        return 0.0

    T = p.shape[0]
    ious = coco_eval.params.iouThrs

    if iou_value is None:
        t_idx = list(range(T))
    else:
        # find closest threshold
        diffs = np.abs(ious - float(iou_value))
        t_idx = [int(np.argmin(diffs))]

    vals = p[t_idx, :, :, area_idx, maxdet_idx]
    vals = vals[vals > -1]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def compute_ar(coco_eval: COCOeval, area_idx: int, maxdet_idx: int) -> float:
    """
    recall dims: [T, K, A, M]
    AR is averaged over IoU thresholds and categories.
    """
    r = coco_eval.eval["recall"]
    if r is None:
        return 0.0
    vals = r[:, :, area_idx, maxdet_idx]
    vals = vals[vals > -1]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtered COCOeval (track-set -> image subset).")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--eval-set", type=str, required=True)
    parser.add_argument("--track-index", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--small-max-area", type=float, default=1024.0)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--category-id", type=int, default=14)

    args = parser.parse_args()

    eval_set = load_json(args.eval_set)
    track_index = load_json(args.track_index)

    track_ids = eval_set.get("track_ids", [])
    if not isinstance(track_ids, list):
        raise ValueError("eval_set must contain track_ids list.")

    image_ids: Set[int] = set()
    for tid in track_ids:
        tid = str(tid)
        frames = track_index.get(tid, None)
        if frames is None:
            # also try int key
            frames = track_index.get(int(tid), None) if tid.isdigit() else None
        if frames is None:
            continue
        for img_id in frames:
            image_ids.add(int(img_id))

    # Filter GT by image_ids and category
    gt = load_json(args.gt)
    images = [im for im in gt.get("images", []) if int(im.get("id")) in image_ids]
    anns = []
    for a in gt.get("annotations", []):
        if int(a.get("image_id")) not in image_ids:
            continue
        if int(a.get("category_id", -1)) != int(args.category_id):
            continue
        # keep annotation; COCOeval uses its own small area definition, we set areaRng
        anns.append(a)

    cats = [c for c in gt.get("categories", []) if int(c.get("id", -1)) == int(args.category_id)]
    gt_f = {"images": images, "annotations": anns, "categories": cats}

    # Filter preds by image_ids and category
    pred = load_json(args.pred)
    pred_f = []
    for p in pred:
        if int(p.get("image_id", -1)) not in image_ids:
            continue
        if int(p.get("category_id", -1)) != int(args.category_id):
            continue
        pred_f.append(p)

    # Run COCOeval in-memory by writing temp json strings? COCO requires file path, so use temp files.
    import tempfile, os

    with tempfile.TemporaryDirectory() as td:
        gt_path = os.path.join(td, "gt.json")
        pr_path = os.path.join(td, "pred.json")
        with open(gt_path, "w") as f:
            json.dump(gt_f, f)
        with open(pr_path, "w") as f:
            json.dump(pred_f, f)

        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pr_path)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = [int(args.category_id)]
        coco_eval.params.imgIds = sorted(list(image_ids))
        coco_eval.params.maxDets = [int(args.max_dets)]
        # area ranges: all + small
        coco_eval.params.areaRng = [[0**2, 1e10], [0**2, float(args.small_max_area)]]
        coco_eval.params.areaRngLbl = ["all", "small"]

        coco_eval.evaluate()
        coco_eval.accumulate()

        # Indices
        area_all = 0
        area_small = 1
        maxdet_idx = 0

        out = {
            "num_images": int(len(image_ids)),
            "AP": compute_ap_ar(coco_eval, area_all, maxdet_idx, iou_value=None),
            "AP50": compute_ap_ar(coco_eval, area_all, maxdet_idx, iou_value=0.5),
            "AP_small": compute_ap_ar(coco_eval, area_small, maxdet_idx, iou_value=None),
            "AR_small": compute_ar(coco_eval, area_small, maxdet_idx),
            "small_max_area": float(args.small_max_area),
            "max_dets": int(args.max_dets),
        }

    save_json(args.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
