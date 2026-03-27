# detector/yolo_wrapper.py
# Simple Ultralytics YOLO wrapper for SAHI base and temporal methods.
# Python 3.9 compatible. Comments in English only.

from typing import Tuple, Optional, Dict

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloDetector:
    """
    Minimal wrapper around Ultralytics YOLO models.

    This class exposes a single method:

        predict(frame) -> (boxes_xyxy, scores, labels)

    where:
        - boxes_xyxy: np.ndarray of shape [N, 4] in (x1, y1, x2, y2)
        - scores: np.ndarray of shape [N]
        - labels: np.ndarray of shape [N] (int class ids)

    This interface is intentionally simple so that:
        - sahi_base.FullSahiModel
        - temporal SAHI APIs

    can all use it without knowing any Ultralytics internals.
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        img_size: int = 640,
        conf_thres: float = 0.25,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Args:
            weights: path to YOLO checkpoint (e.g., "yolov11n.pt")
            device: device string for inference ("cuda:0" or "cpu")
            img_size: inference size passed to YOLO (imgsz)
            conf_thres: confidence threshold for filtering predictions
            class_mapping:
                optional mapping from YOLO class index (int) to
                dataset category_id (int). If None, raw YOLO class
                indices are returned.
        """
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics is not installed. Please install with `pip install ultralytics`."
            )

        self.model = YOLO(weights)
        self.model.to(device)

        self.device = device
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)
        self.class_mapping = class_mapping

    def predict(
        self,
        frame: np.ndarray,
        conf_thres: Optional[float] = None,
        img_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLO on a single frame.

        Args:
            frame: np.ndarray (H, W, 3) in BGR or RGB

        Returns:
            boxes_xyxy: np.ndarray of shape [N, 4] in (x1, y1, x2, y2)
            scores:     np.ndarray of shape [N]
            labels:     np.ndarray of shape [N] (int class ids)
        """
        # Ultralytics can accept np.ndarray directly
        conf_used = self.conf_thres if conf_thres is None else float(conf_thres)
        img_size_used = self.img_size if img_size is None else int(img_size)

        results = self.model.predict(
            source=frame,
            imgsz=img_size_used,
            conf=conf_used,
            verbose=False,
            device=self.device,
        )

        if len(results) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # xyxy, conf, cls are torch tensors
        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        labels = res.boxes.cls.detach().cpu().numpy().astype(np.int64)

        if self.class_mapping is not None:
            # Map YOLO class indices to dataset category ids
            mapped_labels = np.zeros_like(labels)
            for i, cls in enumerate(labels):
                mapped_labels[i] = self.class_mapping.get(int(cls), int(cls))
            labels = mapped_labels

        return xyxy, scores, labels
