# methods/keyframe_sahi_runner.py
# Heuristic Keyframe SAHI runner: run Full SAHI every N frames, otherwise run plain YOLO.
# Python 3.9 compatible. Comments in English only.

import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from detector.yolo_wrapper import YoloDetector
from sahi_base.tiler import count_slices
from sahi_base.full_sahi_api import FullSahiModel
from utils.category_mapper import map_label_to_category_id
from sahi_base.postprocess import postprocess_preds, get_pp_preset

class KeyframeSahiRunner:
    """
    Heuristic baseline:
      - Keyframes: Full SAHI
      - Non-keyframes: Plain YOLO
    """

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
        key_interval: int = 5,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        if int(key_interval) <= 0:
            raise ValueError("key_interval must be positive.")
        self.key_interval = int(key_interval)

        self.device = device
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)

        self.slice_height = int(slice_height)
        self.slice_width = int(slice_width)
        self.overlap_height_ratio = float(overlap_height_ratio)
        self.overlap_width_ratio = float(overlap_width_ratio)
        self.nms_iou_threshold = float(nms_iou_threshold)

        self.class_mapping = class_mapping

        # Shared detector
        self.detector = YoloDetector(
            weights=weights,
            device=device,
            img_size=self.img_size,
            conf_thres=self.conf_thres,
        )

        # Full SAHI model (already supports mapping options in your updated version)
        self.full_sahi = FullSahiModel(
            detector=self.detector,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            nms_iou_threshold=self.nms_iou_threshold,
            class_mapping=self.class_mapping,
        )

    def _predict_yolo_frame(
        self,
        frame: np.ndarray,
        image_id: int,
    ) -> List[Dict[str, Any]]:
        boxes_np, scores_np, labels_np = self.detector.predict(frame)
        if boxes_np is None or len(boxes_np) == 0:
            return []

        preds: List[Dict[str, Any]] = []
        for box, score, label in zip(boxes_np, scores_np, labels_np):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)

            category_id = map_label_to_category_id(
                label_id=int(label),
                class_mapping=self.class_mapping,
            )

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [x1, y1, w_box, h_box],
                    "score": float(score),
                }
            )
        return preds

    def run(
        self,
        dataset,
        score_threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []

        num_frames = 0
        total_tiles = 0
        sahi_calls = 0

        t_start = time.time()

        try:
            from tqdm import tqdm

            frame_iter = tqdm(dataset.iter_frames(), desc="Keyframe SAHI inference")
        except ImportError:
            frame_iter = dataset.iter_frames()

        for frame, image_id, meta in frame_iter:
            if frame is None:
                continue
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)

            t = num_frames  # 0-based in this run

            if (t % self.key_interval) == 0:
                # Keyframe: Full SAHI
                h, w = frame.shape[:2]
                num_slices = count_slices(
                    height=h,
                    width=w,
                    slice_height=self.slice_height,
                    slice_width=self.slice_width,
                    overlap_height_ratio=self.overlap_height_ratio,
                    overlap_width_ratio=self.overlap_width_ratio,
                )
                frame_preds = self.full_sahi.predict_frame(
                    frame=frame,
                    image_id=int(image_id),
                    score_threshold=score_threshold,
                )
                total_tiles += int(num_slices)
                sahi_calls += 1
            else:
                # Non-keyframe: plain YOLO
                frame_preds = self._predict_yolo_frame(frame, int(image_id))
                total_tiles += 1
            # Apply the same common postprocess to YOLO path (fairness vs SAHI path).
            # SAHI path is already postprocessed inside FullSahiModel.
            if (t % self.key_interval) != 0:
                h, w = frame.shape[:2]
                pp_cfg = get_pp_preset("icct_std_v1")
                frame_preds, _pp_meta = postprocess_preds(frame_preds, image_w=int(w), image_h=int(h), cfg=pp_cfg)

            predictions.extend(frame_preds)
            num_frames += 1

        elapsed = time.time() - t_start
        fps = float(num_frames) / elapsed if elapsed > 0.0 else 0.0
        avg_tiles = float(total_tiles) / float(num_frames) if num_frames > 0 else 0.0
        sahi_ratio = float(sahi_calls) / float(num_frames) if num_frames > 0 else 0.0
        latency_ms = (float(elapsed) / float(num_frames)) * 1000.0 if num_frames > 0 else 0.0

        stats: Dict[str, Any] = {
            "method": "keyframe_sahi",
            "num_frames": int(num_frames),
            "elapsed_sec": float(elapsed),
            "fps": fps,
            "latency_ms": float(latency_ms),
            "avg_tiles": float(avg_tiles),
            "avg_selected": None,
            "avg_moving": None,
            "slice_height": self.slice_height,
            "slice_width": self.slice_width,
            "overlap_height_ratio": self.overlap_height_ratio,
            "overlap_width_ratio": self.overlap_width_ratio,
            "nms_iou_threshold": self.nms_iou_threshold,
            "key_interval": int(self.key_interval),
            "sahi_calls": int(sahi_calls),
            "sahi_ratio": float(sahi_ratio),
        }

        return predictions, stats
