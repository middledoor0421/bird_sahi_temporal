# detector/yolo_wrapper.py
# Python 3.9 compatible. Comments in English only.

from typing import List, Dict, Any
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Minimal YOLO wrapper for baseline experiments."""

    def __init__(self, weights: str, device: str = "0", imgsz: int = 896) -> None:
        self.weights = str(weights)
        self.device = str(device)
        self.imgsz = int(imgsz)

        self.model = YOLO(self.weights)
        # Ultralytics stores class names in a dict
        self.names = self.model.names

    def infer_image(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR image.
        Returns a list of dicts: {"xyxy": [x1,y1,x2,y2], "conf": float, "cls": int}.
        """
        # Ultralytics accepts numpy arrays in BGR
        results = self.model.predict(
            source=img_bgr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        preds: List[Dict[str, Any]] = []

        if len(results) == 0:
            return preds

        res = results[0]
        if res.boxes is None:
            return preds

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            preds.append(
                {
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(confs[i]),
                    "cls": int(clses[i]),
                }
            )
        return preds
