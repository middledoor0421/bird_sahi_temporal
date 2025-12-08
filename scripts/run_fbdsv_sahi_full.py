#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full SAHI inference on FBD-SV-2024 (VID) split.

- Input:
    - data_root/
        VID/images/{split}/{video_id}/{frame}.JPEG
        annotations/fbdsv24_vid_{split}_sequences.json
- Output:
    - annotations/pred_sahi_full_fbdsv24_vid_{split}.json
    - annotations/stats_sahi_full_fbdsv24_vid_{split}.json

Assumes:
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
    from sahi.predict import get_sliced_prediction
    from sahi.utils.yolov8 import Yolov8DetectionModel
except ImportError:
    Yolov8DetectionModel = None
    get_sliced_prediction = None


def load_sequences(seq_json_path: str) -> List[Dict]:
    """Load sequence list from json."""
    with open(seq_json_path, "r") as f:
        sequences = json.load(f)
    return sequences


def compute_num_slices(
    h: int,
    w: int,
    slice_h: int,
    slice_w: int,
    overlap_h_ratio: float,
    overlap_w_ratio: float,
) -> int:
    """Compute number of slices used by SAHI for given image size and slicing params."""
    if slice_h >= h:
        n_h = 1
    else:
        step_h = int(slice_h * (1.0 - overlap_h_ratio))
        step_h = max(1, step_h)
        n_h = max(1, math.ceil((h - slice_h) / step_h) + 1)

    if slice_w >= w:
        n_w = 1
    else:
        step_w = int(slice_w * (1.0 - overlap_w_ratio))
        step_w = max(1, step_w)
        n_w = max(1, math.ceil((w - slice_w) / step_w) + 1)

    return n_h * n_w


def run_sahi_full_on_fbdsv(
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
    out_pred_path: str,
    out_stats_path: str,
) -> None:
    """Run full SAHI tiling with YOLOv8 backend on FBD-SV-2024 VID split."""
    if Yolov8DetectionModel is None or get_sliced_prediction is None:
        raise RuntimeError(
            "SAHI or ultralytics is not installed. "
            "Please install with `pip install sahi ultralytics`."
        )

    data_root = os.path.abspath(data_root)
    ann_root = os.path.join(data_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    seq_json_path = os.path.join(
        ann_root, "fbdsv24_vid_{}_sequences.json".format(split)
    )
    if not os.path.isfile(seq_json_path):
        raise FileNotFoundError("Sequence json not found: {}".format(seq_json_path))

    sequences = load_sequences(seq_json_path)

    detection_model = Yolov8DetectionModel(
        model_path=weights,
        confidence_threshold=conf_thres,
        device=device,
        image_size=img_size,
    )

    predictions: List[Dict] = []
    num_frames = 0
    total_slices = 0

    t_start = time.time()

    try:
        from tqdm import tqdm

        seq_iter = tqdm(sequences, desc="Sequences (SAHI full)")
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

        for img_id, rel_path in zip(frame_ids, frame_files):
            img_path = os.path.join(data_root, rel_path)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            num_slices = compute_num_slices(
                h,
                w,
                slice_height,
                slice_width,
                overlap_height_ratio,
                overlap_width_ratio,
            )

            result = get_sliced_prediction(
                image=img,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                verbose=0,
            )

            if result is not None and result.object_prediction_list is not None:
                for op in result.object_prediction_list:
                    try:
                        category_id = int(op.category.id)
                    except Exception:
                        category_id = 1

                    bbox_xywh: Tuple[float, float, float, float] = op.bbox.to_xywh()
                    x_min = float(bbox_xywh[0])
                    y_min = float(bbox_xywh[1])
                    w_box = float(bbox_xywh[2])
                    h_box = float(bbox_xywh[3])
                    score = float(op.score.value)

                    pred = {
                        "image_id": int(img_id),
                        "category_id": category_id,
                        "bbox": [x_min, y_min, w_box, h_box],
                        "score": score,
                    }
                    predictions.append(pred)

            num_frames += 1
            total_slices += num_slices

    t_end = time.time()
    elapsed = t_end - t_start
    fps = num_frames / elapsed if elapsed > 0 else 0.0
    avg_tiles = float(total_slices) / float(num_frames) if num_frames > 0 else 0.0

    with open(out_pred_path, "w") as f:
        json.dump(predictions, f)
    print("Saved SAHI full predictions to:", out_pred_path)

    stats = {
        "method": "sahi_full",
        "split": split,
        "num_frames": int(num_frames),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "avg_tiles": float(avg_tiles),
        "avg_selected": None,
        "avg_moving": None,
        "slice_height": int(slice_height),
        "slice_width": int(slice_width),
        "overlap_height_ratio": float(overlap_height_ratio),
        "overlap_width_ratio": float(overlap_width_ratio),
    }
    with open(out_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved SAHI full stats to:", out_stats_path)
    print(
        "Frames: {}, Elapsed: {:.2f}s, FPS: {:.3f}, avg_tiles: {:.2f}".format(
            num_frames, elapsed, fps, avg_tiles
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full SAHI tiling on FBD-SV-2024 VID split."
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
            "pred_sahi_full_fbdsv24_vid_{}.json".format(args.split),
        )
    else:
        out_pred = args.out_pred

    if args.out_stats is None:
        out_stats = os.path.join(
            ann_root,
            "stats_sahi_full_fbdsv24_vid_{}.json".format(args.split),
        )
    else:
        out_stats = args.out_stats

    run_sahi_full_on_fbdsv(
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
        out_pred_path=out_pred,
        out_stats_path=out_stats,
    )


if __name__ == "__main__":
    main()
