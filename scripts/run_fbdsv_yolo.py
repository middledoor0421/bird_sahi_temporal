#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO baseline inference on FBD-SV-2024 (VID) split.

- Input:
    - data_root/
        VID/images/{split}/{video_id}/{frame}.JPEG
        annotations/fbdsv24_vid_{split}_sequences.json
        annotations/fbdsv24_vid_{split}.json (for later evaluation, not used here)
- Output:
    - annotations/pred_yolo_fbdsv24_vid_{split}.json
    - annotations/stats_yolo_fbdsv24_vid_{split}.json

This script assumes:
    - sequences json contains:
        [
          {
            "video_id": "bird_1",
            "split": "val",
            "frame_ids": [1, 2, 3, ...],      # COCO image ids
            "frame_files": ["VID/images/val/bird_1/000000.JPEG", ...]
          },
          ...
        ]
    - YOLO model is loaded from ultralytics (e.g., yolov8n.pt, yolov8s.pt, etc.).
"""

import os
import json
import time
import argparse

import cv2

try:
    from ultralytics import YOLO
except ImportError as e:
    YOLO = None


def load_sequences(seq_json_path):
    """Load sequence list from json."""
    with open(seq_json_path, "r") as f:
        sequences = json.load(f)
    return sequences


def run_yolo_on_fbdsv(
    data_root,
    split,
    weights,
    img_size,
    conf_thres,
    device,
    out_pred_path,
    out_stats_path,
):
    """Run YOLO baseline on FBD-SV-2024 VID split."""
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Please install with `pip install ultralytics`."
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

    # Initialize YOLO model
    model = YOLO(weights)
    model.to(device)

    predictions = []
    num_frames = 0

    # Timing
    t_start = time.time()

    # Optional tqdm usage
    try:
        from tqdm import tqdm

        seq_iter = tqdm(sequences, desc="Sequences")
    except ImportError:
        seq_iter = sequences

    for seq in seq_iter:
        frame_ids = seq["frame_ids"]
        frame_files = seq["frame_files"]
        if len(frame_ids) != len(frame_files):
            raise ValueError("frame_ids and frame_files length mismatch in sequence {}".format(seq.get("video_id", "")))

        for img_id, rel_path in zip(frame_ids, frame_files):
            img_path = os.path.join(data_root, rel_path)
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                # Skip if image cannot be loaded
                continue

            # Run YOLO inference
            # Results object: see ultralytics docs. We assume single-image inference.
            results = model.predict(
                source=img,
                imgsz=img_size,
                conf=conf_thres,
                verbose=False,
                device=device,
            )

            # For single image, take first result
            if len(results) == 0:
                num_frames += 1
                continue

            res = results[0]
            # res.boxes.xywh or xyxy, res.boxes.conf, res.boxes.cls
            # Use xywh in (center x, center y, w, h) or convert from xyxy.
            # We convert to COCO [x_min, y_min, width, height].
            if res.boxes is not None and res.boxes.xyxy is not None:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
                clss = res.boxes.cls.detach().cpu().numpy()
            else:
                xyxy = []
                confs = []
                clss = []

            for box, score, cls in zip(xyxy, confs, clss):
                x_min, y_min, x_max, y_max = box.tolist()
                w = max(0.0, x_max - x_min)
                h = max(0.0, y_max - y_min)
                # Map all birds to category_id=1.
                # If model has multiple classes, you can filter by cls here if needed.
                pred = {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [float(x_min), float(y_min), float(w), float(h)],
                    "score": float(score),
                }
                predictions.append(pred)

            num_frames += 1

    t_end = time.time()
    elapsed = t_end - t_start
    fps = num_frames / elapsed if elapsed > 0 else 0.0

    # Save predictions
    with open(out_pred_path, "w") as f:
        json.dump(predictions, f)
    print("Saved YOLO predictions to:", out_pred_path)

    # Save stats
    stats = {
        "method": "yolo_baseline",
        "split": split,
        "num_frames": int(num_frames),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "avg_tiles": 1.0,       # always 1 tile (full frame)
        "avg_selected": None,   # not applicable
        "avg_moving": None,     # not applicable
    }
    with open(out_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved YOLO stats to:", out_stats_path)
    print("Frames: {}, Elapsed: {:.2f}s, FPS: {:.3f}".format(num_frames, elapsed, fps))


def main():
    parser = argparse.ArgumentParser(
        description="YOLO baseline on FBD-SV-2024 VID split."
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
        help="Path to YOLO weights (e.g., yolov8n.pt).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size for YOLO.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--out-pred",
        type=str,
        default=None,
        help="Output prediction json path. If None, auto under data-root/annotations.",
    )
    parser.add_argument(
        "--out-stats",
        type=str,
        default=None,
        help="Output stats json path. If None, auto under data-root/annotations.",
    )

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    ann_root = os.path.join(data_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    if args.out_pred is None:
        out_pred = os.path.join(
            ann_root,
            "pred_yolo_fbdsv24_vid_{}.json".format(args.split),
        )
    else:
        out_pred = args.out_pred

    if args.out_stats is None:
        out_stats = os.path.join(
            ann_root,
            "stats_yolo_fbdsv24_vid_{}.json".format(args.split),
        )
    else:
        out_stats = args.out_stats

    run_yolo_on_fbdsv(
        data_root=data_root,
        split=args.split,
        weights=args.weights,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        device=args.device,
        out_pred_path=out_pred,
        out_stats_path=out_stats,
    )


if __name__ == "__main__":
    main()
