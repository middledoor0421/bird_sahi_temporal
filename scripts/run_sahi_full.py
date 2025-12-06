# scripts/run_sahi_full.py
# Full SAHI baseline inference on videos.
# Python 3.9 compatible.

import argparse
import glob
import time
from pathlib import Path
from typing import List

import cv2
import yaml
from loguru import logger
import numpy as np

from detector.yolo_wrapper import YOLODetector
from sahi_core.tiler import generate_tiles
from sahi_core.merger import merge_tile_detections
from sahi_core.scheduler_base import FullTileScheduler


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_boxes(img: np.ndarray, preds: List[dict], names: dict) -> np.ndarray:
    out = img.copy()
    for det in preds:
        x1, y1, x2, y2 = det["xyxy"]
        conf = det["conf"]
        cls_id = det["cls"]
        name = names.get(cls_id, str(cls_id))
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 255), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(
            out,
            label,
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def run_sahi_on_video(cfg: dict, det: YOLODetector) -> None:
    # repo_root = .../bird_sahi_temporal
    repo_root = Path(__file__).resolve().parents[1]

    video_glob = cfg["inference"]["video_glob"]
    vg_path = Path(video_glob)
    if not vg_path.is_absolute():
        video_glob_full = str(repo_root / video_glob)
    else:
        video_glob_full = str(vg_path)

    save_dir = repo_root / cfg["inference"]["save_dir"] / "videos_sahi_full"
    save_dir.mkdir(parents=True, exist_ok=True)

    sahi_cfg = cfg.get("sahi", {})
    tile_size = int(sahi_cfg.get("tile_size", 896))
    overlap = float(sahi_cfg.get("overlap", 0.25))
    nms_iou = float(sahi_cfg.get("nms_iou", 0.5))

    paths = sorted(glob.glob(video_glob_full))
    logger.info("[SAHI-FULL] video_glob={} → Found {} videos", video_glob_full, len(paths))

    scheduler = FullTileScheduler()

    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            logger.warning("[SAHI-FULL] Failed to open video {}", p)
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        basename = Path(p).stem
        out_path = save_dir / f"{basename}_sahi_full.mp4"

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_idx = 0
        total_tiles = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tiles, infos = generate_tiles(frame, tile_size=tile_size, overlap=overlap)
            idxs = scheduler.select(infos)
            total_tiles += len(idxs)

            per_tile_dets = []
            for i in idxs:
                tile_img = tiles[i]
                dets = det.infer_image(tile_img)
                per_tile_dets.append(dets)

            merged = merge_tile_detections(
                per_tile_dets, [infos[i] for i in idxs], iou_th=nms_iou
            )
            vis = draw_boxes(frame, merged, det.names)
            writer.write(vis)

            if frame_idx % 20 == 0:
                logger.info(
                    "[SAHI-FULL][{}] frame {} tiles={}",
                    basename,
                    frame_idx,
                    len(idxs),
                )
            frame_idx += 1

        t1 = time.time()
        elapsed = t1 - t0
        avg_tiles = float(total_tiles) / max(1, frame_idx)
        eff_fps = float(frame_idx) / elapsed if elapsed > 0 else 0.0

        cap.release()
        writer.release()
        logger.info("[SAHI-FULL] Saved {}", out_path)
        logger.info(
            "[SAHI-FULL][{}] frames={} elapsed={:.2f}s fps={:.2f} avg_tiles={:.2f}",
            basename,
            frame_idx,
            elapsed,
            eff_fps,
            avg_tiles,
        )


def main() -> None:
    parser = argparse.ArgumentParser("Full SAHI baseline inference")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to config yaml"
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model = cfg.get("model", "yolov8n.pt")
    device = cfg.get("device", "0")
    imgsz = cfg.get("imgsz", 896)

    logger.info("[SAHI-FULL] Loading YOLO model={} device={} imgsz={}", model, device, imgsz)
    det = YOLODetector(weights=model, device=device, imgsz=imgsz)

    run_sahi_on_video(cfg, det)


if __name__ == "__main__":
    main()
