#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Oracle analysis for Temporal SAHI.

What it does:
1) Build SAHI tile grid for the dataset frame resolution.
2) Define oracle tile set T*_t from GT small objects (center-in-tile rule).
3) Reconstruct Temporal SAHI selected tiles S_t by replaying motion+activity scheduler:
   - motion is computed from actual frames (frame-diff)
   - activity is updated from temporal predictions (detected tiles inferred from pred boxes)
4) Compute coverage and miss rate:
   - coverage = |S_t ) T*_t| / |T*_t|
   - miss = (T*_t not subset of S_t)
5) Neighbor expansion oracle:
   - S_t_plus = S_t * N(S_t) (1-hop grid neighbor, 4-neigh or 8-neigh)
   - recompute coverage/miss
6) Trigger candidate statistics:
   - Compare signals between miss frames vs non-miss frames.

Requirements:
- datasets.fbdsv.FbdsvDataset exists and yields (frame, image_id, meta)
- sahi_base.tiler.compute_slice_grid exists and returns list of dicts: {"x","y","w","h"}
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from datasets.fbdsv import FbdsvDataset
from sahi_base.tiler import compute_slice_grid


# --------------------------
# Utilities: IO
# --------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------
# Tile helpers
# --------------------------

def point_in_tile(cx: float, cy: float, tile: Dict[str, int]) -> bool:
    x = float(tile["x"])
    y = float(tile["y"])
    w = float(tile["w"])
    h = float(tile["h"])
    return (x <= cx < x + w) and (y <= cy < y + h)


def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x + 0.5 * w, y + 0.5 * h


def bbox_center_xyxy(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def tile_ids_covering_point(cx: float, cy: float, tiles: List[Dict[str, int]]) -> Set[int]:
    hits = set()
    for i, t in enumerate(tiles):
        if point_in_tile(cx, cy, t):
            hits.add(i)
    # If due to numeric corner cases nothing matched, fall back to nearest tile center
    if len(hits) == 0:
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


def build_tile_grid_index(tiles: List[Dict[str, int]]) -> Tuple[
    Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int], List[int], List[int]]:
    """
    Build a (row, col) grid index from tile top-left positions.

    Returns:
      - id_to_rc: tile_id -> (r, c)
      - rc_to_id: (r, c) -> tile_id
      - xs: sorted unique x positions
      - ys: sorted unique y positions
    """
    xs = sorted(set([int(t["x"]) for t in tiles]))
    ys = sorted(set([int(t["y"]) for t in tiles]))
    x_to_c = {x: i for i, x in enumerate(xs)}
    y_to_r = {y: i for i, y in enumerate(ys)}

    id_to_rc: Dict[int, Tuple[int, int]] = {}
    rc_to_id: Dict[Tuple[int, int], int] = {}
    for tid, t in enumerate(tiles):
        r = y_to_r[int(t["y"])]
        c = x_to_c[int(t["x"])]
        id_to_rc[tid] = (r, c)
        rc_to_id[(r, c)] = tid

    return id_to_rc, rc_to_id, xs, ys


def neighbor_expand(
        selected: Set[int],
        id_to_rc: Dict[int, Tuple[int, int]],
        rc_to_id: Dict[Tuple[int, int], int],
        mode: int = 4,
        hops: int = 1,
) -> Set[int]:
    """
    Expand selected set by grid neighbors.

    mode=4: up/down/left/right
    mode=8: also diagonal
    hops: number of BFS steps
    """
    if hops <= 0:
        return set(selected)

    if mode not in [4, 8]:
        raise ValueError("mode must be 4 or 8.")

    if mode == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

    out = set(selected)
    frontier = set(selected)

    for _ in range(hops):
        new_frontier = set()
        for tid in frontier:
            if tid not in id_to_rc:
                continue
            r, c = id_to_rc[tid]
            for dr, dc in dirs:
                rr = r + dr
                cc = c + dc
                key = (rr, cc)
                if key in rc_to_id:
                    nid = rc_to_id[key]
                    if nid not in out:
                        out.add(nid)
                        new_frontier.add(nid)
        frontier = new_frontier
        if len(frontier) == 0:
            break

    return out


def area_union_ratio(selected: Set[int], tiles: List[Dict[str, int]], frame_w: int, frame_h: int) -> float:
    """Approximate coverage ratio by summing tile areas (overlap double counts)."""
    if frame_w <= 0 or frame_h <= 0:
        return 0.0
    s = 0.0
    for tid in selected:
        t = tiles[tid]
        s += float(t["w"]) * float(t["h"])
    return float(s) / float(frame_w * frame_h)


# --------------------------
# Motion gater (same logic as your temporal)
# --------------------------

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
    g = float(ema_alpha) * float(state.prev_global_motion) + (1.0 - float(ema_alpha)) * raw_global
    state.prev_global_motion = g

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
    return scores, g


# --------------------------
# Budget scheduler (match your current one)
# --------------------------

def select_tiles_budget(
        tile_scores: np.ndarray,
        active_set: Set[int],
        max_tiles: int,
        min_tiles: int,
        motion_threshold: float,
        keep_active_tiles: bool = True,
) -> List[int]:
    num_tiles = int(tile_scores.shape[0])
    max_tiles = int(max_tiles)
    min_tiles = int(min_tiles)

    selected: List[int] = []

    if keep_active_tiles and len(active_set) > 0:
        for tid in sorted(active_set):
            if 0 <= tid < num_tiles:
                selected.append(tid)
            if len(selected) >= max_tiles:
                return selected

    remaining = [i for i in range(num_tiles) if i not in set(selected)]
    remaining.sort(key=lambda i: float(tile_scores[i]), reverse=True)

    # Thresholded fill
    for tid in remaining:
        if len(selected) >= max_tiles:
            break
        if float(tile_scores[tid]) >= float(motion_threshold):
            selected.append(tid)

    # Ensure min tiles
    if len(selected) < min_tiles:
        for tid in remaining:
            if tid in selected:
                continue
            selected.append(tid)
            if len(selected) >= min_tiles:
                break

    return selected[:max_tiles]


def update_activity(
        counters: Dict[int, int],
        detected_tiles: Set[int],
        lifetime: int,
) -> Dict[int, int]:
    # decrement
    to_del = []
    for tid in list(counters.keys()):
        counters[tid] = counters[tid] - 1
        if counters[tid] <= 0:
            to_del.append(tid)
    for tid in to_del:
        del counters[tid]
    # refresh
    for tid in detected_tiles:
        counters[int(tid)] = int(lifetime)
    return counters


# --------------------------
# Oracle analysis core
# --------------------------

def build_gt_small_oracle_tiles(
        gt_coco: Dict[str, Any],
        tiles: List[Dict[str, int]],
        small_max_area: float,
) -> Dict[int, Set[int]]:
    """
    Returns:
      oracle_tiles_by_image_id: image_id -> set(tile_id)
    """
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
    """
    Infer "detected tiles" from predicted boxes (center-in-tile).
    Returns:
      detected_tiles_by_image_id: image_id -> set(tile_id)
    """
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


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return 0.0, 0.0
    m = sum(vals) / float(len(vals))
    v = sum([(x - m) * (x - m) for x in vals]) / float(len(vals))
    return float(m), float(math.sqrt(v))


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle analysis for Temporal SAHI.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--gt", type=str, required=True, help="GT COCO json (val).")
    parser.add_argument("--pred-temporal", type=str, required=True, help="Temporal SAHI prediction json.")
    parser.add_argument("--out", type=str, default=None, help="Optional output report json.")

    # Tile params (must match your SAHI settings)
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)

    # Temporal params (must match your temporal run)
    parser.add_argument("--max-tiles-per-frame", type=int, default=16)
    parser.add_argument("--min-tiles-per-frame", type=int, default=1)
    parser.add_argument("--motion-threshold", type=float, default=2.0)
    parser.add_argument("--warmup-full-frames", type=int, default=1)
    parser.add_argument("--activity-lifetime", type=int, default=5)

    # Motion gater params
    parser.add_argument("--motion-downsample-factor", type=int, default=4)
    parser.add_argument("--motion-ema-alpha", type=float, default=0.8)

    # Oracle small definition
    parser.add_argument("--small-max-area", type=float, default=1024.0)

    # Neighbor expansion
    parser.add_argument("--neighbor-mode", type=int, default=4, choices=[4, 8])
    parser.add_argument("--neighbor-hops", type=int, default=1)

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    split = args.split

    # Load GT and temporal predictions
    gt = load_json(args.gt)
    pred_temporal = load_json(args.pred_temporal)

    # Dataset iterator (for frames and motion replay)
    ds = FbdsvDataset(data_root=data_root, split=split)

    # Peek first frame to get resolution and build tiles once
    first_frame = None
    first_img_id = None
    for frame, image_id, meta in ds.iter_frames():
        first_frame = frame
        first_img_id = int(image_id)
        break
    if first_frame is None:
        raise RuntimeError("Dataset returned no frames.")

    H, W = first_frame.shape[0], first_frame.shape[1]
    tiles = compute_slice_grid(
        height=int(H),
        width=int(W),
        slice_height=int(args.slice_height),
        slice_width=int(args.slice_width),
        overlap_height_ratio=float(args.overlap_height_ratio),
        overlap_width_ratio=float(args.overlap_width_ratio),
    )

    id_to_rc, rc_to_id, xs, ys = build_tile_grid_index(tiles)

    # Oracle tile set from GT small objects
    oracle_tiles_by_img = build_gt_small_oracle_tiles(
        gt_coco=gt,
        tiles=tiles,
        small_max_area=float(args.small_max_area),
    )

    # Detected tiles inferred from temporal predictions (for activity update)
    detected_tiles_by_img = build_detected_tiles_from_pred(
        pred_list=pred_temporal,
        tiles=tiles,
    )

    # Replay temporal selection and compute stats
    motion_state = MotionState()
    activity_counters: Dict[int, int] = {}

    num_eval_frames = 0  # frames with at least one small GT
    num_miss = 0
    coverages: List[float] = []

    num_miss_plus = 0
    coverages_plus: List[float] = []
    added_tiles: List[float] = []

    # Trigger candidate signals
    signals_all: Dict[str, List[float]] = defaultdict(list)
    signals_miss: Dict[str, List[float]] = defaultdict(list)

    # Re-iterate frames from scratch (new dataset object)
    ds2 = FbdsvDataset(data_root=data_root, split=split)

    frame_index = 0
    for frame, image_id, meta in ds2.iter_frames():
        img_id = int(image_id)

        # Compute motion and select tiles
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

        # Update activity from detections inferred from pred_temporal
        det_tiles = detected_tiles_by_img.get(img_id, set())
        activity_counters = update_activity(
            counters=activity_counters,
            detected_tiles=det_tiles,
            lifetime=int(args.activity_lifetime),
        )

        # Oracle tiles for this frame (small GT)
        oracle_tiles = oracle_tiles_by_img.get(img_id, set())
        if len(oracle_tiles) == 0:
            frame_index += 1
            continue

        num_eval_frames += 1

        inter = selected.intersection(oracle_tiles)
        cov = float(len(inter)) / float(len(oracle_tiles)) if len(oracle_tiles) > 0 else 0.0
        miss = (not oracle_tiles.issubset(selected))
        if miss:
            num_miss += 1
        coverages.append(cov)

        # Neighbor expansion
        selected_plus = neighbor_expand(
            selected=selected,
            id_to_rc=id_to_rc,
            rc_to_id=rc_to_id,
            mode=int(args.neighbor_mode),
            hops=int(args.neighbor_hops),
        )
        inter_p = selected_plus.intersection(oracle_tiles)
        cov_p = float(len(inter_p)) / float(len(oracle_tiles)) if len(oracle_tiles) > 0 else 0.0
        miss_p = (not oracle_tiles.issubset(selected_plus))
        if miss_p:
            num_miss_plus += 1
        coverages_plus.append(cov_p)
        added_tiles.append(float(len(selected_plus) - len(selected)))

        # Trigger candidate signals
        moving_cnt = float((tile_scores >= float(args.motion_threshold)).sum()) if tile_scores is not None else 0.0
        active_size = float(len(activity_counters))
        sel_size = float(len(selected))
        det_size = float(len(det_tiles))
        cov_area = area_union_ratio(selected, tiles, frame_w=int(W), frame_h=int(H))

        signals = {
            "global_motion": float(global_motion),
            "moving_tiles": float(moving_cnt),
            "active_size": float(active_size),
            "selected_size": float(sel_size),
            "detected_tiles": float(det_size),
            "selected_area_ratio": float(cov_area),
            "coverage": float(cov),
        }
        for k, v in signals.items():
            signals_all[k].append(float(v))
            if miss:
                signals_miss[k].append(float(v))

        frame_index += 1

    # Summaries
    miss_rate = float(num_miss) / float(num_eval_frames) if num_eval_frames > 0 else 0.0
    miss_rate_plus = float(num_miss_plus) / float(num_eval_frames) if num_eval_frames > 0 else 0.0

    cov_mean, cov_std = mean_std(coverages)
    covp_mean, covp_std = mean_std(coverages_plus)
    add_mean, add_std = mean_std(added_tiles)

    print("Oracle analysis summary (frames with small GT only)")
    print("  num_eval_frames:", num_eval_frames)
    print("  miss_rate (current temporal selection): {:.4f}".format(miss_rate))
    print("  coverage mean�std: {:.4f} � {:.4f}".format(cov_mean, cov_std))
    print("Neighbor expansion summary (mode={}, hops={})".format(args.neighbor_mode, args.neighbor_hops))
    print("  miss_rate_plus: {:.4f}".format(miss_rate_plus))
    print("  coverage_plus mean�std: {:.4f} � {:.4f}".format(covp_mean, covp_std))
    print("  added_tiles mean�std: {:.2f} � {:.2f}".format(add_mean, add_std))

    # Trigger signals comparison
    def summarize_signal(name: str) -> Dict[str, float]:
        all_vals = signals_all.get(name, [])
        miss_vals = signals_miss.get(name, [])
        am, asd = mean_std(all_vals)
        mm, msd = mean_std(miss_vals)
        frac = float(len(miss_vals)) / float(len(all_vals)) if len(all_vals) > 0 else 0.0
        return {
            "all_mean": float(am),
            "all_std": float(asd),
            "miss_mean": float(mm),
            "miss_std": float(msd),
            "miss_frac": float(frac),
            "miss_count": float(len(miss_vals)),
            "all_count": float(len(all_vals)),
        }

    signal_names = ["global_motion", "moving_tiles", "active_size", "selected_size", "detected_tiles",
                    "selected_area_ratio", "coverage"]
    trigger_report = {n: summarize_signal(n) for n in signal_names}

    print("Trigger candidate stats (miss vs all):")
    for n in signal_names:
        r = trigger_report[n]
        print("  {}: all_mean={:.4f}, miss_mean={:.4f}, miss_count={}".format(
            n, r["all_mean"], r["miss_mean"], int(r["miss_count"])
        ))

    report = {
        "frame_resolution": {"width": int(W), "height": int(H)},
        "tile_params": {
            "slice_height": int(args.slice_height),
            "slice_width": int(args.slice_width),
            "overlap_height_ratio": float(args.overlap_height_ratio),
            "overlap_width_ratio": float(args.overlap_width_ratio),
            "num_tiles": int(len(tiles)),
            "grid_x_count": int(len(xs)),
            "grid_y_count": int(len(ys)),
        },
        "temporal_params": {
            "max_tiles_per_frame": int(args.max_tiles_per_frame),
            "min_tiles_per_frame": int(args.min_tiles_per_frame),
            "motion_threshold": float(args.motion_threshold),
            "warmup_full_frames": int(args.warmup_full_frames),
            "activity_lifetime": int(args.activity_lifetime),
            "motion_downsample_factor": int(args.motion_downsample_factor),
            "motion_ema_alpha": float(args.motion_ema_alpha),
        },
        "oracle_params": {
            "small_max_area": float(args.small_max_area),
            "oracle_definition": "GT small bbox center-in-tile (all tiles covering center)",
        },
        "neighbor_params": {
            "mode": int(args.neighbor_mode),
            "hops": int(args.neighbor_hops),
        },
        "summary": {
            "num_eval_frames": int(num_eval_frames),
            "miss_rate": float(miss_rate),
            "coverage_mean": float(cov_mean),
            "coverage_std": float(cov_std),
            "miss_rate_plus": float(miss_rate_plus),
            "coverage_plus_mean": float(covp_mean),
            "coverage_plus_std": float(covp_std),
            "added_tiles_mean": float(add_mean),
            "added_tiles_std": float(add_std),
        },
        "trigger_candidates": trigger_report,
    }

    if args.out is not None:
        save_json(args.out, report)
        print("Saved report to:", args.out)


if __name__ == "__main__":
    main()
