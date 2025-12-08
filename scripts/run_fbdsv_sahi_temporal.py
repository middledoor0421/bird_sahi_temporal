#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Temporal SAHI (motion-gated) inference on FBD-SV-2024 (VID) split.

- Input:
    - data_root/
        VID/images/{split}/{video_id}/{frame}.JPEG
        annotations/fbdsv24_vid_{split}_sequences.json
- Output:
    - annotations/pred_sahi_temporal_fbdsv24_vid_{split}.json
    - annotations/stats_sahi_temporal_fbdsv24_vid_{split}.json

Assumptions:
    - sequences json format:
        [
          {
            "video_id": "bird_1",
            "split": "val",
            "frame_ids": [1, 2, 3, ...],
            "frame_files": ["VID/images/val/bird_1/000000.JPEG", ...]
          },
          ...
        ]
    - MotionGater is defined in sahi_ext.motion_gater.MotionGater
    - SAHI with YOLOv8 backend:
        pip install sahi ultralytics
"""

import os
import json
import time
import argparse
from typing import List, Dict, Tuple

import cv2
import math

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from sahi.utils.yolov8 import Yolov8DetectionModel
except ImportError:
    Yolov8DetectionModel = None

try:
    from sahi_ext.motion_gater import MotionGater
except ImportError:
    MotionGater = None


def load_sequences(seq_json_path: str) -> List[Dict]:
    """Load sequence list from json."""
    with open(seq_json_path, "r") as f:
        sequences = json.load(f)
    return sequences


def build_tile_grid(
    h: int,
    w: int,
    slice_h: int,
    slice_w: int,
    overlap_h_ratio: float,
    overlap_w_ratio: float,
) -> List[Dict]:
    """
    Build tile grid and return list of dicts:
      {"x": x, "y": y, "w": tw, "h": th}
    Matching SAHI-style sliding windows.
    """
    tiles: List[Dict] = []

    if slice_h >= h:
        y_starts = [0]
    else:
        step_h = int(slice_h * (1.0 - overlap_h_ratio))
        step_h = max(1, step_h)
        y_starts = list(range(0, max(1, h - slice_h + 1), step_h))
        if y_starts[-1] + slice_h < h:
            y_starts.append(h - slice_h)

    if slice_w >= w:
        x_starts = [0]
    else:
        step_w = int(slice_w * (1.0 - overlap_w_ratio))
        step_w = max(1, step_w)
        x_starts = list(range(0, max(1, w - slice_w + 1), step_w))
        if x_starts[-1] + slice_w < w:
            x_starts.append(w - slice_w)

    for y in y_starts:
        for x in x_starts:
            tw = min(slice_w, w - x)
            th = min(slice_h, h - y)
            tiles.append({"x": int(x), "y": int(y), "w": int(tw), "h": int(th)})

    return tiles


def run_sahi_temporal_on_fbdsv(
    data_root: str,
    split: str,
    weights: str,
    img_size: int,
    conf_thres: float,
    device: str,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    motion_diff_th: int,
    motion_min_area: int,
    motion_min_stable_frames: int,
    max_tiles_per_frame: int,
    out_pred_path: str,
    out_stats_path: str,
) -> None:
    """Run motion-gated Temporal SAHI on FBD-SV-2024 VID split."""
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Please install with `pip install ultralytics`.")
    if MotionGater is None:
        raise RuntimeError("MotionGater not found. Please ensure sahi_ext.motion_gater is importable.")

    data_root = os.path.abspath(data_root)
    ann_root = os.path.join(data_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    seq_json_path = os.path.join(
        ann_root, "fbdsv24_vid_{}_sequences.json".format(split)
    )
    if not os.path.isfile(seq_json_path):
        raise FileNotFoundError("Sequence json not found: {}".format(seq_json_path))

    sequences = load_sequences(seq_json_path)

    # YOLO model (we use it directly on cropped tiles)
    yolo_model = YOLO(weights)
    yolo_model.to(device)

    # Motion gater
    motion_gater = MotionGater(
        diff_th=motion_diff_th,
        min_area=motion_min_area,
        min_stable_frames=motion_min_stable_frames,
    )

    predictions: List[Dict] = []
    num_frames = 0
    total_selected_tiles = 0
    total_moving_tiles = 0

    t_start = time.time()

    try:
        from tqdm import tqdm

        seq_iter = tqdm(sequences, desc="Sequences (SAHI temporal)")
    except ImportError:
        seq_iter = sequences

    for seq in seq_iter:
        frame_ids = seq["frame_ids"]
        frame_files = seq["frame_files"]
        if len(frame_ids) != len(frame_files):
            raise ValueError(
                "frame_ids and frame_files length mismatch in sequence {}".format(
                    seq.get("video_id", "")
                )
            )

        # Reset gater state for each new video sequence
        motion_gater.reset()

        for img_id, rel_path in zip(frame_ids, frame_files):
            img_path = os.path.join(data_root, rel_path)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            tile_infos = build_tile_grid(
                h=h,
                w=w,
                slice_h=slice_height,
                slice_w=slice_width,
                overlap_h_ratio=overlap_height_ratio,
                overlap_w_ratio=overlap_width_ratio,
            )

            # Motion gating: which tiles are moving?
            moving_idxs = motion_gater.update(frame, tile_infos)
            num_moving = len(moving_idxs)

            # If no moving tiles detected, you can choose one of:
            # 1) skip detection on this frame (max saving, but recall may drop)
            # 2) fallback to all tiles or some default
            if num_moving == 0:
                num_frames += 1
                continue

            # Budget on max tiles per frame
            if max_tiles_per_frame > 0 and num_moving > max_tiles_per_frame:
                selected_idxs = moving_idxs[:max_tiles_per_frame]
            else:
                selected_idxs = moving_idxs

            num_selected = len(selected_idxs)

            # Run YOLO on selected tiles only
            for idx in selected_idxs:
                info = tile_infos[idx]
                x = info["x"]
                y = info["y"]
                tw = info["w"]
                th = info["h"]

                tile = frame[y : y + th, x : x + tw]

                # YOLO inference for this tile
                results = yolo_model.predict(
                    source=tile,
                    imgsz=img_size,
                    conf=conf_thres,
                    verbose=False,
                    device=device,
                )

                if len(results) == 0:
                    continue

                res = results[0]
                if res.boxes is None or res.boxes.xyxy is None:
                    continue

                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
                clss = res.boxes.cls.detach().cpu().numpy()

                for box, score, cls in zip(xyxy, confs, clss):
                    x_min_tile, y_min_tile, x_max_tile, y_max_tile = box.tolist()
                    # Shift to global coordinates
                    x_min = float(x + x_min_tile)
                    y_min = float(y + y_min_tile)
                    w_box = float(x_max_tile - x_min_tile)
                    h_box = float(y_max_tile - y_min_tile)

                    pred = {
                        "image_id": int(img_id),
                        "category_id": 1,  # map all to bird
                        "bbox": [x_min, y_min, w_box, h_box],
                        "score": float(score),
                    }
                    predictions.append(pred)

            num_frames += 1
            total_moving_tiles += num_moving
            total_selected_tiles += num_selected

    t_end = time.time()
    elapsed = t_end - t_start
    fps = num_frames / elapsed if elapsed > 0 else 0.0
    avg_selected = (
        float(total_selected_tiles) / float(num_frames) if num_frames > 0 else 0.0
    )
    avg_moving = (
        float(total_moving_tiles) / float(num_frames) if num_frames > 0 else 0.0
    )

    # Save predictions
    with open(out_pred_path, "w") as f:
        json.dump(predictions, f)
    print("Saved Temporal SAHI predictions to:", out_pred_path)

    # Save stats
    stats = {
        "method": "sahi_temporal_motion_only",
        "split": split,
        "num_frames": int(num_frames),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "avg_tiles": None,  # can be interpreted as avg_selected
        "avg_selected": float(avg_selected),
        "avg_moving": float(avg_moving),
        "slice_height": int(slice_height),
        "slice_width": int(slice_width),
        "overlap_height_ratio": float(overlap_height_ratio),
        "overlap_width_ratio": float(overlap_width_ratio),
        "motion_diff_th": int(motion_diff_th),
        "motion_min_area": int(motion_min_area),
        "motion_min_stable_frames": int(motion_min_stable_frames),
        "max_tiles_per_frame": int(max_tiles_per_frame),
    }
    with open(out_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved Temporal SAHI stats to:", out_stats_path)
    print(
        "Frames: {}, Elapsed: {:.2f}s, FPS: {:.3f}, avg_selected: {:.2f}, avg_moving: {:.2f}".format(
            num_frames, elapsed, fps, avg_selected, avg_moving
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal SAHI (motion-gated) on FBD-SV-2024 VID split."
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
        help="Dataset split to run.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLOv8 weights (e.g., yolov8s.pt).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Base image size used by YOLO backend.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--slice-height",
        type=int,
        default=512,
        help="SAHI slice height.",
    )
    parser.add_argument(
        "--slice-width",
        type=int,
        default=512,
        help="SAHI slice width.",
    )
    parser.add_argument(
        "--overlap-height-ratio",
        type=float,
        default=0.2,
        help="SAHI overlap ratio for height.",
    )
    parser.add_argument(
        "--overlap-width-ratio",
        type=float,
        default=0.2,
        help="SAHI overlap ratio for width.",
    )
    parser.add_argument(
        "--motion-diff-th",
        type=int,
        default=15,
        help="Pixel difference threshold for MotionGater.",
    )
    parser.add_argument(
        "--motion-min-area",
        type=int,
        default=50,
        help="Minimum moving area per tile for MotionGater.",
    )
    parser.add_argument(
        "--motion-min-stable-frames",
        type=int,
        default=1,
        help="Minimum consecutive frames to mark tile as moving.",
    )
    parser.add_argument(
        "--max-tiles-per-frame",
        type=int,
        default=8,
        help="Maximum number of tiles to process per frame (0 means no limit).",
    )
    parser.add_argument(
        "--out-pred",
        type=str,
        default=None,
        help="Output prediction json path. If None, use default under data-root/annotations.",
    )
    parser.add_argument(
        "--out-stats",
        type=str,
        default=None,
        help="Output stats json path. If None, use default under data-root/annotations.",
    )

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    ann_root = os.path.join(data_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    if args.out_pred is None:
        out_pred = os.path.join(
            ann_root,
            "pred_sahi_temporal_fbdsv24_vid_{}.json".format(args.split),
        )
    else:
        out_pred = args.out_pred

    if args.out_stats is None:
        out_stats = os.path.join(
            ann_root,
            "stats_sahi_temporal_fbdsv24_vid_{}.json".format(args.split),
        )
    else:
        out_stats = args.out_stats

    run_sahi_temporal_on_fbdsv(
        data_root=data_root,
        split=args.split,
        weights=args.weights,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        device=args.device,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        motion_diff_th=args.motion_diff_th,
        motion_min_area=args.motion_min_area,
        motion_min_stable_frames=args.motion_min_stable_frames,
        max_tiles_per_frame=args.max_tiles_per_frame,
        out_pred_path=out_pred,
        out_stats_path=out_stats,
    )


if __name__ == "__main__":
    main()
