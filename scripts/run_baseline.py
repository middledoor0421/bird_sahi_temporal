# scripts/run_sahi_temporal.py
# Temporal-motion-aware + weak-detection-aware SAHI inference on videos.
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
from sahi_ext.motion_gater import MotionGater
from sahi_ext.temporal_scheduler import TemporalScheduler
from sahi_ext.budget_scheduler import BudgetScheduler


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
        cv2.rectangle(out, p1, p2, (0, 165, 255), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(
            out,
            label,
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def find_weak_boxes(
    frame_shape: tuple,
    dets: List[dict],
    conf_low: float,
    conf_high: float,
    small_area_ratio: float,
) -> List[List[float]]:
    """
    From full-frame detections, select weak boxes:
      conf_low <= conf <= conf_high and area_ratio < small_area_ratio.
    Returns list of [x1,y1,x2,y2].
    """
    h, w = frame_shape[:2]
    image_area = float(w * h)
    weak: List[List[float]] = []
    for d in dets:
        conf = float(d["conf"])
        if conf < conf_low or conf > conf_high:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        area = max(0.0, float(x2 - x1) * float(y2 - y1))
        if area <= 0.0:
            continue
        if (area / image_area) < small_area_ratio:
            weak.append([float(x1), float(y1), float(x2), float(y2)])
    return weak


def run_sahi_temporal_on_video(cfg: dict, det: YOLODetector) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    video_glob = cfg["inference"]["video_glob"]
    vg_path = Path(video_glob)
    if not vg_path.is_absolute():
        video_glob_full = str(repo_root / video_glob)
    else:
        video_glob_full = str(vg_path)

    sahi_cfg = cfg.get("sahi", {})
    tile_size = int(sahi_cfg.get("tile_size", 896))
    overlap = float(sahi_cfg.get("overlap", 0.25))
    nms_iou = float(sahi_cfg.get("nms_iou", 0.5))

    tcfg = cfg.get("temporal", {})
    diff_th = int(tcfg.get("diff_th", 18))
    min_area = int(tcfg.get("min_area", 64))
    min_stable = int(tcfg.get("min_stable_frames", 2))
    max_tiles = int(tcfg.get("max_tiles", 6))

    wcfg = tcfg.get("weak", {})
    weak_enable = bool(wcfg.get("enable", True))
    conf_low = float(wcfg.get("conf_low", 0.2))
    conf_high = float(wcfg.get("conf_high", 0.6))
    small_area_ratio = float(wcfg.get("small_area_ratio", 0.002))

    save_dir = repo_root / cfg["inference"]["save_dir"] / "base_line"
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(video_glob_full))
    logger.info("[SAHI-TEMP] video_glob={} → Found {} videos", video_glob_full, len(paths))

    motion_gater = MotionGater(diff_th=diff_th, min_area=min_area, min_stable_frames=min_stable)
    temporal_sched = TemporalScheduler()
    budget_sched = BudgetScheduler(max_tiles=max_tiles)

    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            logger.warning("[SAHI-TEMP] Failed to open video {}", p)
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")


        basename = Path(p).stem
        out_path = save_dir / f"{basename}_baseline.mp4"

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_idx = 0
        total_selected = 0
        total_moving = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) full-frame detection to find weak boxes (coarse pass)
            full_dets = det.infer_image(frame)
            weak_boxes = find_weak_boxes(
                frame_shape=frame.shape,
                dets=full_dets,
                conf_low=conf_low,
                conf_high=conf_high,
                small_area_ratio=small_area_ratio,
            ) if weak_enable else []

            # 2) tiling
            tiles, infos = generate_tiles(frame, tile_size=tile_size, overlap=overlap)

            # 3) motion gating
            moving_idxs = motion_gater.update(frame, infos)

            # 4) temporal scheduler (motion + weak)
            selected_idxs = temporal_sched.select(infos, moving_idxs, weak_boxes)

            # 5) budget scheduler
            selected_idxs = budget_sched.select(selected_idxs)

            total_moving += len(moving_idxs)
            total_selected += len(selected_idxs)

            per_tile_dets: List[List[dict]] = []
            if selected_idxs:
                for i in selected_idxs:
                    tile_img = tiles[i]
                    dets = det.infer_image(tile_img)
                    per_tile_dets.append(dets)
                merged = merge_tile_detections(
                    per_tile_dets, [infos[i] for i in selected_idxs], iou_th=nms_iou
                )
            else:
                merged = []

            vis = draw_boxes(frame, merged, det.names)
            writer.write(vis)

            if frame_idx % 20 == 0:
                logger.info(
                    "[SAHI-TEMP][{}] frame {} tiles={} moving={} weak_boxes={}",
                    basename,
                    frame_idx,
                    len(selected_idxs),
                    len(moving_idxs),
                    len(weak_boxes),
                )
            frame_idx += 1

        t1 = time.time()
        elapsed = t1 - t0
        avg_selected = float(total_selected) / max(1, frame_idx)
        avg_moving = float(total_moving) / max(1, frame_idx)
        eff_fps = float(frame_idx) / elapsed if elapsed > 0 else 0.0

        cap.release()
        writer.release()
        logger.info("[SAHI-TEMP] Saved {}", out_path)
        logger.info(
            "[SAHI-TEMP][{}] frames={} elapsed={:.2f}s fps={:.2f} "
            "avg_selected={:.2f} avg_moving={:.2f}",
            basename,
            frame_idx,
            elapsed,
            eff_fps,
            avg_selected,
            avg_moving,
        )


def main() -> None:
    parser = argparse.ArgumentParser("Temporal-motion-aware + weak-aware SAHI inference")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to config yaml"
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model = cfg.get("model", "yolov8n.pt")
    device = cfg.get("device", "0")
    imgsz = cfg.get("imgsz", 896)

    logger.info("[SAHI-TEMP] Loading YOLO model={} device={} imgsz={}", model, device, imgsz)
    det = YOLODetector(weights=str(model), device=str(device), imgsz=imgsz)

    run_sahi_temporal_on_video(cfg, det)


if __name__ == "__main__":
    main()
