# methods/yolo_runner.py
# Unified YOLO baseline runner for any datasets with `iter_frames()` interface.
# Python 3.9 compatible. Comments in English only.

import gzip
import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from utils.category_mapper import map_label_to_category_id


class YoloRunner:
    """
    YOLO baseline runner.

    Expected dataset interface:
        for frame, image_id, meta in dataset.iter_frames():
            frame: np.ndarray (H, W, 3)
            image_id: int
            meta: dict
    """

    def __init__(
        self,
        weights: str,
        img_size: int = 640,
        conf_thres: float = 0.25,
        device: str = "cuda:0",
        class_mapping: Optional[Dict[int, int]] = None,
        cost_per_frame_path: Optional[str] = None,
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Please install with `pip install ultralytics`."
            )

        self.weights = weights
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)
        self.device = device

        self.class_mapping = class_mapping
        self.cost_per_frame_path = cost_per_frame_path

        self.model = YOLO(self.weights)
        self.model.to(self.device)

    def _predict_frame(
        self,
        frame: np.ndarray,
        image_id: int,
    ) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_thres,
            verbose=False,
            device=self.device,
        )

        predictions: List[Dict[str, Any]] = []

        if len(results) == 0:
            return predictions

        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            return predictions

        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        confs = res.boxes.conf.detach().cpu().numpy()
        clss = res.boxes.cls.detach().cpu().numpy()

        for box, score, cls in zip(xyxy, confs, clss):
            x_min, y_min, x_max, y_max = box.tolist()
            w_box = max(0.0, float(x_max - x_min))
            h_box = max(0.0, float(y_max - y_min))

            label_id = int(cls)
            category_id = map_label_to_category_id(
                label_id=label_id,
                class_mapping=self.class_mapping,
            )

            pred = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x_min), float(y_min), w_box, h_box],
                "score": float(score),
            }
            predictions.append(pred)

        return predictions

    def run(
        self,
        dataset,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        num_frames = 0

        t_start = time.time()
        cost_f = None
        cost_path = str(self.cost_per_frame_path).strip() if self.cost_per_frame_path is not None else ""
        if cost_path and cost_path != "/dev/null":
            out_dir = os.path.dirname(cost_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cost_f = gzip.open(cost_path, "wt", encoding="utf-8")

        try:
            from tqdm import tqdm

            frame_iter = tqdm(dataset.iter_frames(), desc="YOLO inference")
        except ImportError:
            frame_iter = dataset.iter_frames()

        for frame_idx, (frame, image_id, meta) in enumerate(frame_iter):
            t_frame0 = time.time()
            frame_preds = self._predict_frame(frame, image_id)
            t_frame1 = time.time()
            predictions.extend(frame_preds)
            num_frames += 1

            if cost_f is not None:
                video_id = ""
                if isinstance(meta, dict):
                    video_id = str(meta.get("video_id", meta.get("sequence_id", "")))
                    frame_t = meta.get("frame_idx", frame_idx)
                else:
                    frame_t = frame_idx
                rec = {
                    "video_id": video_id,
                    "t": int(frame_t),
                    "image_id": int(image_id),
                    "sahi_on": 0,
                    "tiles_used": 0,
                    "verifier_backend": "none",
                    "verifier_executed": 0,
                    "num_verifier_regions": 0,
                    "verifier_cost_proxy": 0.0,
                    "latency_ms": float((t_frame1 - t_frame0) * 1000.0),
                }
                cost_f.write(json.dumps(rec) + "\n")

        if cost_f is not None:
            cost_f.close()

        elapsed = time.time() - t_start
        fps = float(num_frames) / elapsed if elapsed > 0.0 else 0.0

        stats: Dict[str, Any] = {
            "method": "yolo",
            "num_frames": int(num_frames),
            "elapsed_sec": float(elapsed),
            "fps": fps,
            "avg_tiles": 1.0,
            "avg_selected": None,
            "avg_moving": None,
            "cost_per_frame_path": cost_path if cost_path else None,
        }

        return predictions, stats
