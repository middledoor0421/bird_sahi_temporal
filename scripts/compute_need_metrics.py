#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/compute_need_metrics.py
# Compute FFR / NODR / CSF from GT + (SAHI-off preds) + (SAHI-on preds).
# Python 3.9 compatible. Comments in English only.

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def bbox_area_xywh(b: List[float]) -> float:
    return float(b[2]) * float(b[3])


def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x + 0.5 * w, y + 0.5 * h


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def greedy_match_gt_to_pred(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_thr: float,
) -> Tuple[int, List[int]]:
    """
    Greedy matching: for each GT, match the best unmatched pred by IoU.
    Returns:
      matched_gt_count,
      matched_pred_indices (len == matched_gt_count)
    """
    used = set()
    matched = 0
    matched_pred_indices: List[int] = []

    for g in gt_boxes:
        best_iou = 0.0
        best_j = -1
        for j, p in enumerate(pred_boxes):
            if j in used:
                continue
            v = iou_xywh(g, p)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= float(iou_thr) and best_j >= 0:
            used.add(best_j)
            matched += 1
            matched_pred_indices.append(best_j)

    return matched, matched_pred_indices


def build_time_index(gt_coco: Dict[str, Any]) -> Tuple[List[int], Dict[int, int]]:
    """
    Build an ordered list of image_ids and a mapping image_id -> time index.
    Preference order for sorting:
      1) frame_id
      2) frame_index
      3) id
    """
    images = gt_coco.get("images", [])
    entries: List[Tuple[int, int]] = []  # (time_key, image_id)

    for img in images:
        img_id = int(img["id"])
        if "frame_id" in img:
            tk = int(img["frame_id"])
        elif "frame_index" in img:
            tk = int(img["frame_index"])
        else:
            tk = img_id
        entries.append((tk, img_id))

    entries.sort(key=lambda x: x[0])
    ordered_ids = [img_id for _, img_id in entries]
    time_of = {img_id: i for i, img_id in enumerate(ordered_ids)}
    return ordered_ids, time_of


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FFR / NODR / CSF for SAHI necessity.")
    parser.add_argument("--gt", type=str, required=True, help="GT COCO json")
    parser.add_argument("--pred-off", type=str, required=True, help="SAHI-off predictions (e.g., YOLO) json")
    parser.add_argument("--pred-on", type=str, required=True, help="SAHI-on predictions (e.g., Full SAHI) json")
    parser.add_argument("--out", type=str, required=True, help="Output json report path")

    parser.add_argument("--category-id", type=int, default=14, help="Target category id (bird=14)")
    parser.add_argument("--small-max-area", type=float, default=1024.0, help="Small GT max area in pixels^2")
    parser.add_argument("--pred-score-thr", type=float, default=0.25, help="Score threshold for preds")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for GT-pred matching")

    parser.add_argument("--success-mode", type=str, default="strict", choices=["strict", "loose"],
                        help="strict: all small GT must be matched; loose: any match is success")

    # NODR settings
    parser.add_argument("--nodr-window", type=int, default=10, help="Lookback window N (frames) for 'new object'")
    parser.add_argument("--nodr-metric", type=str, default="center", choices=["center", "iou"],
                        help="How to decide 'same object' across frames")
    parser.add_argument("--nodr-center-dist", type=float, default=50.0, help="Center distance threshold (pixels)")
    parser.add_argument("--nodr-iou-thr", type=float, default=0.3, help="IoU threshold for same-object test")
    parser.add_argument("--ffr-out", type=str, default="",
                        help="Optional path to save full FFR frame image_id list as JSON.")

    args = parser.parse_args()

    gt = load_json(args.gt)
    pred_off = load_json(args.pred_off)
    pred_on = load_json(args.pred_on)

    cat_id = int(args.category_id)
    small_max_area = float(args.small_max_area)
    score_thr = float(args.pred_score_thr)
    iou_thr = float(args.iou_thr)

    ordered_ids, time_of = build_time_index(gt)

    # Group GT small boxes by image_id
    gt_small_by_img: Dict[int, List[List[float]]] = defaultdict(list)
    # Also keep GT small objects for NODR: store bbox + time
    gt_small_ann_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for ann in gt.get("annotations", []):
        if int(ann.get("category_id", -1)) != cat_id:
            continue
        bbox = ann.get("bbox", None)
        if bbox is None:
            continue
        area = float(ann.get("area", 0.0))
        if area <= 0.0:
            area = bbox_area_xywh(bbox)
        if area > small_max_area:
            continue

        img_id = int(ann["image_id"])
        gt_small_by_img[img_id].append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
        gt_small_ann_by_img[img_id].append({
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "area": float(area),
        })

    # Group predictions by image_id (filter by cat_id and score_thr)
    def group_preds(pred_list: Any) -> Dict[int, List[List[float]]]:
        by_img: Dict[int, List[List[float]]] = defaultdict(list)
        if not isinstance(pred_list, list):
            raise ValueError("Prediction json must be a list.")
        for p in pred_list:
            if int(p.get("category_id", -1)) != cat_id:
                continue
            if float(p.get("score", 0.0)) < score_thr:
                continue
            img_id = int(p["image_id"])
            b = p.get("bbox", None)
            if b is None:
                continue
            by_img[img_id].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        return by_img

    pred_off_by_img = group_preds(pred_off)
    pred_on_by_img = group_preds(pred_on)

    # Evaluate per-frame success and build SAHI-only rescued GT set
    eval_img_ids = [img_id for img_id in ordered_ids if len(gt_small_by_img.get(img_id, [])) > 0]

    succ_off: Dict[int, bool] = {}
    succ_on: Dict[int, bool] = {}

    # For NODR: SAHI-only rescued GT objects
    rescued_gt: List[Tuple[int, List[float]]] = []  # (img_id, gt_bbox)

    for img_id in eval_img_ids:
        gt_boxes = gt_small_by_img[img_id]
        off_boxes = pred_off_by_img.get(img_id, [])
        on_boxes = pred_on_by_img.get(img_id, [])

        m_off, _ = greedy_match_gt_to_pred(gt_boxes, off_boxes, iou_thr=iou_thr)
        m_on, _ = greedy_match_gt_to_pred(gt_boxes, on_boxes, iou_thr=iou_thr)

        if args.success_mode == "strict":
            succ_off[img_id] = (m_off == len(gt_boxes))
            succ_on[img_id] = (m_on == len(gt_boxes))
        else:
            succ_off[img_id] = (m_off >= 1)
            succ_on[img_id] = (m_on >= 1)

        # SAHI-only rescued GT objects (GT-level definition)
        # A GT object is "rescued" if matched by on but not matched by off.
        # We'll decide match per GT by checking if any pred overlaps that GT.
        for g in gt_boxes:
            on_hit = False
            off_hit = False
            for p in on_boxes:
                if iou_xywh(g, p) >= iou_thr:
                    on_hit = True
                    break
            for p in off_boxes:
                if iou_xywh(g, p) >= iou_thr:
                    off_hit = True
                    break
            if on_hit and (not off_hit):
                rescued_gt.append((img_id, g))

    # FFR
    failure_frames = [img_id for img_id in eval_img_ids if (succ_on.get(img_id, False) and (not succ_off.get(img_id, False)))]
    ffr = float(len(failure_frames)) / float(len(eval_img_ids)) if len(eval_img_ids) > 0 else 0.0

    # NODR (new object among rescued GTs)
    N = int(args.nodr_window)
    metric = str(args.nodr_metric).lower().strip()
    dist_thr = float(args.nodr_center_dist)
    same_iou_thr = float(args.nodr_iou_thr)

    # Build quick access to previous GT small bboxes by time index
    gt_small_by_time: Dict[int, List[List[float]]] = defaultdict(list)
    for img_id in eval_img_ids:
        t = time_of.get(img_id, None)
        if t is None:
            continue
        gt_small_by_time[t].extend(gt_small_by_img.get(img_id, []))

    def is_new_object(img_id: int, bbox: List[float]) -> bool:
        t = time_of.get(img_id, None)
        if t is None:
            return True
        t0 = max(0, t - N)
        cx, cy = bbox_center_xywh(bbox)
        for tt in range(t0, t):
            prev_list = gt_small_by_time.get(tt, [])
            for pb in prev_list:
                if metric == "center":
                    pcx, pcy = bbox_center_xywh(pb)
                    d = math.sqrt((pcx - cx) * (pcx - cx) + (pcy - cy) * (pcy - cy))
                    if d <= dist_thr:
                        return False
                else:
                    if iou_xywh(pb, bbox) >= same_iou_thr:
                        return False
        return True

    rescued_total = len(rescued_gt)
    rescued_new = 0
    for img_id, g in rescued_gt:
        if is_new_object(img_id, g):
            rescued_new += 1
    nodr = float(rescued_new) / float(rescued_total) if rescued_total > 0 else 0.0

    # CSF (based on failure frame intervals)
    # Use time indices for consecutive failure frames
    failure_times = sorted([time_of[x] for x in failure_frames if x in time_of])
    deltas: List[int] = []
    for i in range(1, len(failure_times)):
        deltas.append(int(failure_times[i] - failure_times[i - 1]))

    csf_mean = None
    csf_p95 = None
    mean_delta = None
    p95_delta = None
    if len(deltas) > 0:
        mean_delta = float(sum(deltas)) / float(len(deltas))
        p95_delta = float(np.quantile(np.asarray(deltas, dtype=np.float32), 0.95))
        csf_mean = float(1.0 / mean_delta) if mean_delta > 0 else None
        csf_p95 = float(1.0 / p95_delta) if p95_delta > 0 else None

    report: Dict[str, Any] = {
        "settings": {
            "category_id": cat_id,
            "small_max_area": small_max_area,
            "pred_score_thr": score_thr,
            "iou_thr": iou_thr,
            "success_mode": args.success_mode,
            "nodr_window": N,
            "nodr_metric": metric,
            "nodr_center_dist": dist_thr,
            "nodr_iou_thr": same_iou_thr,
        },
        "counts": {
            "eval_frames_small_gt": int(len(eval_img_ids)),
            "failure_frames": int(len(failure_frames)),
            "rescued_gt_total": int(rescued_total),
            "rescued_gt_new": int(rescued_new),
        },
        "FFR": float(ffr),
        "NODR": float(nodr),
        "CSF": {
            "num_failure_intervals": int(len(deltas)),
            "mean_delta": mean_delta,
            "p95_delta": p95_delta,
            "csf_mean": csf_mean,
            "csf_p95": csf_p95,
        },
        "failure_frame_image_ids": failure_frames[:200],  # keep a short list for debugging
    }
    if str(args.ffr_out).strip() != "":
        save_json(args.ffr_out, failure_frames)
        print("Saved full FFR frame list to:", args.ffr_out)

    save_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
