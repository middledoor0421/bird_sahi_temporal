# methods/sahi_full_runner.py
# Full SAHI runner built on top of sahi_base + YoloDetector.
# Python 3.9 compatible. Comments in English only.

import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from detector.yolo_wrapper import YoloDetector
from sahi_base.tiler import count_slices
from sahi_base.full_sahi_api import FullSahiModel


class SahiFullRunner:
    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        img_size: int = 640,
        conf_thres: float = 0.25,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        nms_iou_threshold: float = 0.5,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        self.device = device
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)

        self.slice_height = int(slice_height)
        self.slice_width = int(slice_width)
        self.overlap_height_ratio = float(overlap_height_ratio)
        self.overlap_width_ratio = float(overlap_width_ratio)
        self.nms_iou_threshold = float(nms_iou_threshold)

        self.detector = YoloDetector(
            weights=weights,
            device=device,
            img_size=self.img_size,
            conf_thres=self.conf_thres,
        )

        self.sahi_model = FullSahiModel(
            detector=self.detector,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            nms_iou_threshold=self.nms_iou_threshold,
            class_mapping=class_mapping,
        )

    def run(
        self,
        dataset,
        score_threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        num_frames = 0
        total_slices = 0

        t_start = time.time()

        try:
            from tqdm import tqdm

            frame_iter = tqdm(dataset.iter_frames(), desc="Full SAHI inference")
        except ImportError:
            frame_iter = dataset.iter_frames()

        for frame, image_id, meta in frame_iter:
            if frame is None:
                continue

            h, w = frame.shape[:2]
            num_slices = count_slices(
                height=h,
                width=w,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio,
            )

            frame_preds = self.sahi_model.predict_frame(
                frame=frame,
                image_id=int(image_id),
                score_threshold=score_threshold,
            )

            predictions.extend(frame_preds)
            num_frames += 1
            total_slices += num_slices

        elapsed = time.time() - t_start
        fps = float(num_frames) / elapsed if elapsed > 0.0 else 0.0
        avg_tiles = float(total_slices) / float(num_frames) if num_frames > 0 else 0.0

        stats: Dict[str, Any] = {
            "method": "full_sahi",
            "num_frames": int(num_frames),
            "elapsed_sec": float(elapsed),
            "fps": fps,
            "avg_tiles": avg_tiles,
            "avg_selected": None,
            "avg_moving": None,
            "slice_height": self.slice_height,
            "slice_width": self.slice_width,
            "overlap_height_ratio": self.overlap_height_ratio,
            "overlap_width_ratio": self.overlap_width_ratio,
            "nms_iou_threshold": self.nms_iou_threshold,
        }

        return predictions, stats
