#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts2/eval_video_kpi2.py
# Video KPI evaluation for single-class detection on video frames.
# Python 3.9 compatible. Comments in English only.

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts2.gt_utils import resolve_target_category_id
from scripts2.pred_slim import load_pred_any


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _load_image_ids(path: str) -> List[int]:
    """Load image_ids from a json file.

    Supported formats:
      1) {"image_ids": [1,2,...], ...}
      2) [1,2,...]
    """
    obj = load_json(path)
    if isinstance(obj, dict) and "image_ids" in obj:
        obj = obj["image_ids"]
    if not isinstance(obj, list):
        raise ValueError("Invalid image ids json. Expect a list or a dict with 'image_ids'.")
    return [int(x) for x in obj]


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def bbox_area_xywh(b: List[float]) -> float:
    return float(b[2]) * float(b[3])


def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x + 0.5 * w, y + 0.5 * h


def bbox_diag_xywh(b: List[float]) -> float:
    return float(math.sqrt(float(b[2]) * float(b[2]) + float(b[3]) * float(b[3])))


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


def center_dist(a: List[float], b: List[float]) -> float:
    ax, ay = bbox_center_xywh(a)
    bx, by = bbox_center_xywh(b)
    return float(math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)))


def quantile(vals: List[float], q: float) -> Optional[float]:
    if len(vals) == 0:
        return None
    arr = np.asarray(vals, dtype=np.float32)
    return float(np.quantile(arr, q))


def build_frame_order(images: List[Dict[str, Any]]) -> Dict[int, Tuple[Any, Any, int]]:
    """
    Returns mapping image_id -> (video_key, frame_key, image_id).
    video_key: video_id if exists else 0
    frame_key: frame_id/frame_index if exists else image_id
    """
    order: Dict[int, Tuple[Any, Any, int]] = {}
    for img in images:
        img_id = int(img["id"])
        video_key = img.get("video_id", 0)
        if "frame_id" in img:
            frame_key = img["frame_id"]
        elif "frame_index" in img:
            frame_key = img["frame_index"]
        else:
            frame_key = img_id
        order[img_id] = (video_key, frame_key, img_id)
    return order


def sort_image_ids(images: List[Dict[str, Any]]) -> List[int]:
    order = build_frame_order(images)
    items = list(order.values())
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    return [int(x[2]) for x in items]


def group_by_video(images: List[Dict[str, Any]]) -> Dict[Any, List[int]]:
    order = build_frame_order(images)
    by_video: Dict[Any, List[Tuple[Any, Any, int]]] = defaultdict(list)
    for img_id, (vk, fk, iid) in order.items():
        by_video[vk].append((vk, fk, iid))
    out: Dict[Any, List[int]] = {}
    for vk, lst in by_video.items():
        lst.sort(key=lambda x: (x[1], x[2]))
        out[vk] = [int(x[2]) for x in lst]
    return out


def filter_pred(pred_list: List[Dict[str, Any]], category_id: int, score_thr: float) -> Dict[int, List[List[float]]]:
    by_img: Dict[int, List[List[float]]] = defaultdict(list)
    for p in pred_list:
        if int(p.get("category_id", -1)) != int(category_id):
            continue
        if float(p.get("score", 0.0)) < float(score_thr):
            continue
        img_id = int(p["image_id"])
        b = p.get("bbox", None)
        if b is None:
            continue
        by_img[img_id].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return by_img


def filter_gt_small(gt_list: List[Dict[str, Any]], category_id: int, small_max_area: float) -> Dict[
    int, List[List[float]]]:
    by_img: Dict[int, List[List[float]]] = defaultdict(list)
    for ann in gt_list:
        if int(ann.get("category_id", -1)) != int(category_id):
            continue
        b = ann.get("bbox", None)
        if b is None:
            continue
        area = float(ann.get("area", 0.0))
        if area <= 0.0:
            area = bbox_area_xywh(b)
        if area <= float(small_max_area):
            by_img[int(ann["image_id"])].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return by_img


@dataclass
class Track:
    track_id: int
    video_id: Any
    frames: List[int]
    gt_boxes: List[List[float]]


def build_pseudo_tracks(
        by_video_frames: Dict[Any, List[int]],
        gt_by_img: Dict[int, List[List[float]]],
        link_iou_thr: float,
        link_center_px: float,
        link_center_frac: float,
        use_center_link: int,
) -> List[Track]:
    """
    Build pseudo tracks by linking GT boxes across consecutive frames.
    One track corresponds to one GT instance evolving over time.
    """
    tracks: List[Track] = []
    next_tid = 1

    for vid, frame_ids in by_video_frames.items():
        active: Dict[int, List[float]] = {}  # track_id -> last bbox

        for img_id in frame_ids:
            gts = gt_by_img.get(img_id, [])
            if len(gts) == 0:
                active = {}
                continue

            assigned_gt = set()
            new_active: Dict[int, List[float]] = {}

            # match existing active tracks to current gt boxes
            for tid, last_box in active.items():
                best_j = -1
                best_score = -1.0
                for j, g in enumerate(gts):
                    if j in assigned_gt:
                        continue
                    if use_center_link == 1:
                        d = center_dist(last_box, g)
                        thr = link_center_px
                        if thr <= 0.0:
                            thr = link_center_frac * max(1.0, bbox_diag_xywh(g))
                        score = -d
                        ok = (d <= thr)
                    else:
                        v = iou_xywh(last_box, g)
                        score = v
                        ok = (v >= link_iou_thr)

                    if ok and score > best_score:
                        best_score = score
                        best_j = j

                if best_j >= 0:
                    assigned_gt.add(best_j)
                    g = gts[best_j]
                    new_active[tid] = g

                    # append to track
                    for tr in tracks:
                        if tr.track_id == tid:
                            tr.frames.append(img_id)
                            tr.gt_boxes.append(g)
                            break

            # create new tracks for unmatched GT
            for j, g in enumerate(gts):
                if j in assigned_gt:
                    continue
                tid = next_tid
                next_tid += 1
                tr = Track(track_id=tid, video_id=vid, frames=[img_id], gt_boxes=[g])
                tracks.append(tr)
                new_active[tid] = g

            active = new_active

    return tracks


def match_pred_to_gt_frame(
        gt_box: List[float],
        pred_boxes: List[List[float]],
        match_iou_thr: float,
        center_match: int,
        center_px: float,
        center_frac: float,
) -> bool:
    if len(pred_boxes) == 0:
        return False

    if center_match == 1:
        thr = center_px
        if thr <= 0.0:
            thr = center_frac * max(1.0, bbox_diag_xywh(gt_box))
        for p in pred_boxes:
            if center_dist(gt_box, p) <= thr:
                return True
        return False

    # IoU match
    for p in pred_boxes:
        if iou_xywh(gt_box, p) >= match_iou_thr:
            return True
    return False


def sustained_lock(hit_seq: List[int], n: int) -> bool:
    if n <= 0:
        return True
    cur = 0
    for v in hit_seq:
        if v == 1:
            cur += 1
            if cur >= n:
                return True
        else:
            cur = 0
    return False


def first_run_start(hit_seq: List[int], n: int) -> Optional[int]:
    """Return the first index where n consecutive hits start."""
    if n <= 0:
        return 0
    cur = 0
    start = 0
    for idx, v in enumerate(hit_seq):
        if v == 1:
            if cur == 0:
                start = idx
            cur += 1
            if cur >= n:
                return start
        else:
            cur = 0
    return None


def compute_kpis(
        tracks: List[Track],
        pred_by_img: Dict[int, List[List[float]]],
        small_max_area: float,
        small_track_frac: float,
        match_iou_thr: float,
        center_match: int,
        center_px: float,
        center_frac: float,
        lock_n: int,
        ready_n: int,
) -> Dict[str, Any]:
    total = len(tracks)
    if total == 0:
        return {"num_tracks": 0}

    t_init_success = 0
    ttd_vals: List[int] = []
    miss_tracks = 0
    lock_success = 0

    # small-only
    small_tracks: List[Track] = []
    for tr in tracks:
        small_frames = 0
        for b in tr.gt_boxes:
            if bbox_area_xywh(b) <= small_max_area:
                small_frames += 1
        frac = float(small_frames) / float(len(tr.frames)) if len(tr.frames) > 0 else 0.0
        if frac >= float(small_track_frac):
            small_tracks.append(tr)

    def eval_tracks(track_list: List[Track]) -> Dict[str, Any]:
        n_tr = len(track_list)
        if n_tr == 0:
            return {
                "num_tracks": 0,
                "track_init_recall": None,
                "ttd_p50": None,
                "ttd_p90": None,
                "miss_rate": None,
                "sustained_lock_rate": None,
                "deterrence_ready_rate": None,
                "stable_ttd_p50": None,
                "stable_ttd_p90": None,
            }

        init_ok = 0
        miss = 0
        ttd_list: List[int] = []
        lock_ok = 0
        ready_ok = 0
        stable_ttd_list: List[int] = []

        for tr in track_list:
            hit_seq: List[int] = []
            first_hit: Optional[int] = None

            for idx, img_id in enumerate(tr.frames):
                gt_box = tr.gt_boxes[idx]
                preds = pred_by_img.get(img_id, [])
                hit = 1 if match_pred_to_gt_frame(
                    gt_box=gt_box,
                    pred_boxes=preds,
                    match_iou_thr=match_iou_thr,
                    center_match=center_match,
                    center_px=center_px,
                    center_frac=center_frac,
                ) else 0
                hit_seq.append(hit)
                if hit == 1 and first_hit is None:
                    first_hit = idx

            if any(hit_seq):
                init_ok += 1
            else:
                miss += 1

            if first_hit is not None:
                ttd_list.append(int(first_hit))

            if sustained_lock(hit_seq, n=int(lock_n)):
                lock_ok += 1
            stable_start = first_run_start(hit_seq, n=int(ready_n))
            if stable_start is not None:
                ready_ok += 1
                stable_ttd_list.append(int(stable_start))

        out = {
            "num_tracks": int(n_tr),
            "track_init_recall": float(init_ok) / float(n_tr),
            "ttd_p50": quantile([float(x) for x in ttd_list], 0.50),
            "ttd_p90": quantile([float(x) for x in ttd_list], 0.90),
            "miss_rate": float(miss) / float(n_tr),
            "sustained_lock_rate": float(lock_ok) / float(n_tr),
            "deterrence_ready_rate": float(ready_ok) / float(n_tr),
            "stable_ttd_p50": quantile([float(x) for x in stable_ttd_list], 0.50),
            "stable_ttd_p90": quantile([float(x) for x in stable_ttd_list], 0.90),
        }
        return out

    overall = eval_tracks(tracks)
    small = eval_tracks(small_tracks)

    return {
        "overall": overall,
        "small_only": small,
        "lock_n": int(lock_n),
        "match": {
            "match_iou_thr": float(match_iou_thr),
            "center_match": int(center_match),
            "center_px": float(center_px),
            "center_frac": float(center_frac),
        },
        "small_track_frac": float(small_track_frac),
        "small_max_area": float(small_max_area),
        "ready_n": int(ready_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Video KPI evaluation (track-based).")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument(
        "--image-ids-json",
        type=str,
        default=None,
        help="Optional json file with {'image_ids':[...]} to filter evaluation frames.",
    )

    parser.add_argument("--category-id", type=int, default=-1)
    parser.add_argument("--pred-score-thr", type=float, default=0.25)

    # track linking
    parser.add_argument("--link-iou-thr", type=float, default=0.3)
    parser.add_argument("--use-center-link", type=int, default=0, choices=[0, 1])
    parser.add_argument("--link-center-px", type=float, default=0.0)
    parser.add_argument("--link-center-frac", type=float, default=0.5)

    # KPI matching
    parser.add_argument("--match-iou-thr", type=float, default=0.5)
    parser.add_argument("--center-match", type=int, default=0, choices=[0, 1])
    parser.add_argument("--center-px", type=float, default=0.0)
    parser.add_argument("--center-frac", type=float, default=0.5)

    # sustained lock
    parser.add_argument("--lock-n", type=int, default=5)
    parser.add_argument(
        "--ready-n",
        type=int,
        default=5,
        help="Consecutive hit length required for deterrence-ready state.",
    )

    # small-only
    parser.add_argument("--small-max-area", type=float, default=1024.0)
    parser.add_argument("--small-track-frac", type=float, default=0.5)

    args = parser.parse_args()

    gt = load_json(args.gt)
    if int(args.category_id) == -1:
        args.category_id = resolve_target_category_id(args.gt, None, None)
    pred = load_pred_any(args.pred)

    allowed_image_ids = None
    if args.image_ids_json is not None:
        allowed_image_ids = set(_load_image_ids(args.image_ids_json))

    images = gt.get("images", [])
    ann = gt.get("annotations", [])
    if not isinstance(images, list) or not isinstance(ann, list):
        raise ValueError("GT must be COCO format with images/annotations lists.")

    allowed_ids = None
    if args.image_ids_json is not None:
        allowed_ids = set(_load_image_ids(args.image_ids_json))
        images = [im for im in images if int(im.get("id", -1)) in allowed_ids]
        ann = [a for a in ann if int(a.get("image_id", -1)) in allowed_ids]
        if isinstance(pred, list):
            pred = [p for p in pred if int(p.get("image_id", -1)) in allowed_ids]

    if allowed_image_ids is not None:
        images = [im for im in images if int(im.get("id", -1)) in allowed_image_ids]
        ann = [a for a in ann if int(a.get("image_id", -1)) in allowed_image_ids]
        pred = [p for p in pred if int(p.get("image_id", -1)) in allowed_image_ids]

    by_video = group_by_video(images)

    cat_id = int(args.category_id)
    gt_by_img: Dict[int, List[List[float]]] = defaultdict(list)
    for a in ann:
        if int(a.get("category_id", -1)) != cat_id:
            continue
        b = a.get("bbox", None)
        if b is None:
            continue
        gt_by_img[int(a["image_id"])].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    tracks = build_pseudo_tracks(
        by_video_frames=by_video,
        gt_by_img=gt_by_img,
        link_iou_thr=float(args.link_iou_thr),
        link_center_px=float(args.link_center_px),
        link_center_frac=float(args.link_center_frac),
        use_center_link=int(args.use_center_link),
    )

    pred_by_img = filter_pred(pred, category_id=cat_id, score_thr=float(args.pred_score_thr))

    kpis = compute_kpis(
        tracks=tracks,
        pred_by_img=pred_by_img,
        small_max_area=float(args.small_max_area),
        small_track_frac=float(args.small_track_frac),
        match_iou_thr=float(args.match_iou_thr),
        center_match=int(args.center_match),
        center_px=float(args.center_px),
        center_frac=float(args.center_frac),
        lock_n=int(args.lock_n),
        ready_n=int(args.ready_n),
    )

    report = {
        "settings": {
            "category_id": cat_id,
            "pred_score_thr": float(args.pred_score_thr),
            "link_iou_thr": float(args.link_iou_thr),
            "use_center_link": int(args.use_center_link),
            "link_center_px": float(args.link_center_px),
            "link_center_frac": float(args.link_center_frac),
            "match_iou_thr": float(args.match_iou_thr),
            "center_match": int(args.center_match),
            "center_px": float(args.center_px),
            "center_frac": float(args.center_frac),
            "lock_n": int(args.lock_n),
            "ready_n": int(args.ready_n),
            "small_max_area": float(args.small_max_area),
            "small_track_frac": float(args.small_track_frac),
        },
        "num_tracks_total": int(len(tracks)),
        "kpi": kpis,
    }

    save_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
