#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Oracle breakdown analysis:
Quantify miss contribution by:
  - cold-start (active_size == 0)
  - low-motion (moving_tiles and/or global_motion is low)

This script replays Temporal SAHI tile selection (motion + activity) and
compares it against oracle tiles from GT small objects.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from datasets.fbdsv import FbdsvDataset
from sahi_base.tiler import compute_slice_grid


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x + 0.5 * w, y + 0.5 * h


def point_in_tile(cx: float, cy: float, tile: Dict[str, int]) -> bool:
    x = float(tile["x"])
    y = float(tile["y"])
    w = float(tile["w"])
    h = float(tile["h"])
    return (x <= cx < x + w) and (y <= cy < y + h)


def tile_ids_covering_point(cx: float, cy: float, tiles: List[Dict[str, int]]) -> Set[int]:
    hits = set()
    for i, t in enumerate(tiles):
        if point_in_tile(cx, cy, t):
            hits.add(i)
    if len(hits) == 0:
        # Fallback to nearest tile center
        best_i = 0
        best_d = 1e30
        for i, t in enumerate(tiles):
            tx = float(t["x"]) + 0.5 * float(t["w"])
            ty = float(t["y"]) + 0.5 * float(t["h"])
            d = (tx - cx) * (tx - cx) + (ty - cy) * (ty - cy)
            if d < best_d:
                best_d = d
                best_i = i
        hits.add(best_i)
    return hits


def build_gt_small_oracle_tiles(
    gt_coco: Dict[str, Any],
    tiles: List[Dict[str, int]],
    small_max_area: float,
) -> Dict[int, Set[int]]:
    oracle: Dict[int, Set[int]] = defaultdict(set)
    for ann in gt_coco.get("annotations", []):
        img_id = int(ann["image_id"])
        bbox = ann.get("bbox", None)
        if bbox is None:
            continue
        area = float(ann.get("area", 0.0))
        if area <= 0.0:
            area = float(bbox[2]) * float(bbox[3])
        if area > float(small_max_area):
            continue
        cx, cy = bbox_center_xywh(bbox)
        tids = tile_ids_covering_point(cx, cy, tiles)
        for tid in tids:
            oracle[img_id].add(tid)
    return oracle


def build_detected_tiles_from_pred(
    pred_list: List[Dict[str, Any]],
    tiles: List[Dict[str, int]],
) -> Dict[int, Set[int]]:
    det_tiles: Dict[int, Set[int]] = defaultdict(set)
    for p in pred_list:
        img_id = int(p["image_id"])
        bbox = p.get("bbox", None)
        if bbox is None:
            continue
        cx, cy = bbox_center_xywh(bbox)
        tids = tile_ids_covering_point(cx, cy, tiles)
        for tid in tids:
            det_tiles[img_id].add(tid)
    return det_tiles


class MotionState:
    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_global_motion: float = 0.0


def to_gray_downsampled(frame: np.ndarray, downsample_factor: int) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame.astype(np.float32)
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        r = frame[..., 0].astype(np.float32)
        g = frame[..., 1].astype(np.float32)
        b = frame[..., 2].astype(np.float32)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        raise ValueError("Frame must be HxW or HxWx3.")

    s = max(1, int(downsample_factor))
    if s > 1:
        h = (gray.shape[0] // s) * s
        w = (gray.shape[1] // s) * s
        gray = gray[:h:s, :w:s]
    return gray.astype(np.float32)


def compute_tile_motion_scores(
    frame: np.ndarray,
    tiles: List[Dict[str, int]],
    state: MotionState,
    downsample_factor: int,
    ema_alpha: float,
) -> Tuple[np.ndarray, float]:
    gray = to_gray_downsampled(frame, downsample_factor)
    prev = state.prev_gray
    if prev is None or prev.shape != gray.shape:
        state.prev_gray = gray
        scores = np.zeros(len(tiles), dtype=np.float32)
        return scores, 0.0

    diff = np.abs(gray - prev)
    raw_global = float(diff.mean())
    global_motion = float(ema_alpha) * float(state.prev_global_motion) + (1.0 - float(ema_alpha)) * raw_global
    state.prev_global_motion = global_motion

    s = max(1, int(downsample_factor))
    scores = np.zeros(len(tiles), dtype=np.float32)

    Hds, Wds = diff.shape[0], diff.shape[1]
    for i, t in enumerate(tiles):
        x1 = int(t["x"]) // s
        y1 = int(t["y"]) // s
        x2 = (int(t["x"]) + int(t["w"])) // s
        y2 = (int(t["y"]) + int(t["h"])) // s

        x1 = max(0, min(x1, Wds))
        x2 = max(0, min(x2, Wds))
        y1 = max(0, min(y1, Hds))
        y2 = max(0, min(y2, Hds))

        if x2 <= x1 or y2 <= y1:
            scores[i] = 0.0
        else:
            patch = diff[y1:y2, x1:x2]
            scores[i] = float(patch.mean())

    state.prev_gray = gray
    return scores, global_motion


def select_tiles_budget(
    tile_scores: np.ndarray,
    active_set: Set[int],
    max_tiles: int,
    min_tiles: int,
    motion_threshold: float,
    keep_active_tiles: bool = True,
) -> List[int]:
    num_tiles = int(tile_scores.shape[0])
    selected: List[int] = []

    if keep_active_tiles and len(active_set) > 0:
        for tid in sorted(active_set):
            if 0 <= tid < num_tiles:
                selected.append(tid)
            if len(selected) >= int(max_tiles):
                return selected

    remaining = [i for i in range(num_tiles) if i not in set(selected)]
    remaining.sort(key=lambda i: float(tile_scores[i]), reverse=True)

    for tid in remaining:
        if len(selected) >= int(max_tiles):
            break
        if float(tile_scores[tid]) >= float(motion_threshold):
            selected.append(tid)

    if len(selected) < int(min_tiles):
        for tid in remaining:
            if tid in selected:
                continue
            selected.append(tid)
            if len(selected) >= int(min_tiles):
                break

    return selected[: int(max_tiles)]


def update_activity(
    counters: Dict[int, int],
    detected_tiles: Set[int],
    lifetime: int,
) -> Dict[int, int]:
    to_del = []
    for tid in list(counters.keys()):
        counters[tid] = counters[tid] - 1
        if counters[tid] <= 0:
            to_del.append(tid)
    for tid in to_del:
        del counters[tid]

    for tid in detected_tiles:
        counters[int(tid)] = int(lifetime)
    return counters


def quantile_threshold(vals: List[float], q: float) -> float:
    if len(vals) == 0:
        return 0.0
    arr = np.asarray(vals, dtype=np.float32)
    return float(np.quantile(arr, float(q)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle breakdown (cold-start vs low-motion).")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred-temporal", type=str, required=True)

    # Tile params
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)

    # Temporal params
    parser.add_argument("--max-tiles-per-frame", type=int, default=16)
    parser.add_argument("--min-tiles-per-frame", type=int, default=1)
    parser.add_argument("--motion-threshold", type=float, default=2.0)
    parser.add_argument("--warmup-full-frames", type=int, default=1)
    parser.add_argument("--activity-lifetime", type=int, default=5)

    # Motion params
    parser.add_argument("--motion-downsample-factor", type=int, default=4)
    parser.add_argument("--motion-ema-alpha", type=float, default=0.8)

    # Oracle small definition
    parser.add_argument("--small-max-area", type=float, default=1024.0)

    # Low-motion definition via quantiles (robust)
    parser.add_argument("--low-motion-quantile", type=float, default=0.25)
    parser.add_argument("--low-motion-use-and", action="store_true",
                        help="If set: low-motion when (moving_tiles <= thr) AND (global_motion <= thr). Otherwise OR.")

    # Output
    parser.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    gt = load_json(args.gt)
    pred_temporal = load_json(args.pred_temporal)

    ds = FbdsvDataset(data_root=os.path.abspath(args.data_root), split=args.split)

    # Get resolution from first frame
    first = None
    for frame, image_id, meta in ds.iter_frames():
        first = frame
        break
    if first is None:
        raise RuntimeError("No frames in dataset.")
    H, W = int(first.shape[0]), int(first.shape[1])

    tiles = compute_slice_grid(
        height=H,
        width=W,
        slice_height=int(args.slice_height),
        slice_width=int(args.slice_width),
        overlap_height_ratio=float(args.overlap_height_ratio),
        overlap_width_ratio=float(args.overlap_width_ratio),
    )

    oracle_tiles_by_img = build_gt_small_oracle_tiles(gt, tiles, float(args.small_max_area))
    detected_tiles_by_img = build_detected_tiles_from_pred(pred_temporal, tiles)

    # Replay and record per-frame stats for eval frames (small GT only)
    motion_state = MotionState()
    activity_counters: Dict[int, int] = {}

    records: List[Dict[str, Any]] = []

    ds2 = FbdsvDataset(data_root=os.path.abspath(args.data_root), split=args.split)
    frame_index = 0
    for frame, image_id, meta in ds2.iter_frames():
        img_id = int(image_id)
        oracle_tiles = oracle_tiles_by_img.get(img_id, set())
        if len(oracle_tiles) == 0:
            frame_index += 1
            continue

        # Note: cold-start should be evaluated BEFORE updating activity with current detections
        active_size_before = int(len(activity_counters))

        if frame_index < int(args.warmup_full_frames):
            selected = set(range(len(tiles)))
            tile_scores = np.zeros(len(tiles), dtype=np.float32)
            global_motion = 0.0
        else:
            tile_scores, global_motion = compute_tile_motion_scores(
                frame=frame,
                tiles=tiles,
                state=motion_state,
                downsample_factor=int(args.motion_downsample_factor),
                ema_alpha=float(args.motion_ema_alpha),
            )
            active_set = set(activity_counters.keys())
            selected_list = select_tiles_budget(
                tile_scores=tile_scores,
                active_set=active_set,
                max_tiles=int(args.max_tiles_per_frame),
                min_tiles=int(args.min_tiles_per_frame),
                motion_threshold=float(args.motion_threshold),
                keep_active_tiles=True,
            )
            selected = set([int(x) for x in selected_list])

        inter = selected.intersection(oracle_tiles)
        coverage = float(len(inter)) / float(len(oracle_tiles))
        miss = (not oracle_tiles.issubset(selected))

        moving_tiles = int((tile_scores >= float(args.motion_threshold)).sum()) if tile_scores is not None else 0

        # Update activity using detections inferred from pred_temporal
        det_tiles = detected_tiles_by_img.get(img_id, set())
        activity_counters = update_activity(
            counters=activity_counters,
            detected_tiles=det_tiles,
            lifetime=int(args.activity_lifetime),
        )

        records.append(
            {
                "image_id": img_id,
                "miss": bool(miss),
                "coverage": float(coverage),
                "selected_size": int(len(selected)),
                "active_size_before": int(active_size_before),
                "moving_tiles": int(moving_tiles),
                "global_motion": float(global_motion),
            }
        )

        frame_index += 1

    if len(records) == 0:
        raise RuntimeError("No eval frames (small GT frames) found.")

    # Determine low-motion thresholds via quantiles
    q = float(args.low_motion_quantile)
    thr_moving = quantile_threshold([float(r["moving_tiles"]) for r in records], q)
    thr_gm = quantile_threshold([float(r["global_motion"]) for r in records], q)

    # Categorize and aggregate
    total = len(records)
    total_miss = sum([1 for r in records if r["miss"]])

    def is_low_motion(r: Dict[str, Any]) -> bool:
        mt = float(r["moving_tiles"])
        gm = float(r["global_motion"])
        if args.low_motion_use_and:
            return (mt <= thr_moving) and (gm <= thr_gm)
        return (mt <= thr_moving) or (gm <= thr_gm)

    buckets = {
        "A_cold_only": {"count": 0, "miss": 0},
        "B_lowmotion_only": {"count": 0, "miss": 0},
        "C_both": {"count": 0, "miss": 0},
        "D_neither": {"count": 0, "miss": 0},
    }

    for r in records:
        cold = (int(r["active_size_before"]) == 0)
        lowm = is_low_motion(r)

        if cold and lowm:
            key = "C_both"
        elif cold and (not lowm):
            key = "A_cold_only"
        elif (not cold) and lowm:
            key = "B_lowmotion_only"
        else:
            key = "D_neither"

        buckets[key]["count"] += 1
        if r["miss"]:
            buckets[key]["miss"] += 1

    def rate(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    summary = {
        "num_eval_frames": int(total),
        "num_miss": int(total_miss),
        "miss_rate": rate(total_miss, total),
        "low_motion_quantile": float(q),
        "low_motion_thr_moving_tiles": float(thr_moving),
        "low_motion_thr_global_motion": float(thr_gm),
        "low_motion_logic": "AND" if args.low_motion_use_and else "OR",
        "buckets": {},
    }

    print("Breakdown summary (small GT frames only)")
    print("  num_eval_frames:", total)
    print("  miss_rate:", "{:.4f}".format(summary["miss_rate"]))
    print("  low-motion thresholds (quantile={}): moving_tiles<= {:.3f}, global_motion<= {:.3f} (logic={})".format(
        q, thr_moving, thr_gm, summary["low_motion_logic"]
    ))
    print("Buckets (count, miss_rate, miss_contribution):")

    for k in ["C_both", "A_cold_only", "B_lowmotion_only", "D_neither"]:
        c = buckets[k]["count"]
        m = buckets[k]["miss"]
        mr = rate(m, c)
        contrib = rate(m, total_miss)  # fraction of all misses
        summary["buckets"][k] = {
            "count": int(c),
            "miss": int(m),
            "miss_rate": float(mr),
            "miss_contribution": float(contrib),
        }
        print("  {}: count={}, miss_rate={:.4f}, miss_contribution={:.4f}".format(k, c, mr, contrib))

    if args.out is not None:
        save_json(args.out, summary)
        print("Saved report to:", args.out)


if __name__ == "__main__":
    main()
