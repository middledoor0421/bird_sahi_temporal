# sahi_temporal/v3/coarse_detector.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CoarseStats:
    num_boxes: int
    num_small_boxes: int
    mean_score: float
    max_score: float
    score_margin: float
    num_lowconf: int


class CoarseDetector:
    """
    Wraps an existing detector (e.g., detector.yolo_wrapper.YoloDetector).
    Produces predictions in COCO result format and coarse statistics.
    """

    def __init__(
        self,
        detector,
        small_max_area: float = 1024.0,
        lowconf_thres: float = 0.05,
    ) -> None:
        self.detector = detector
        self.small_max_area = float(small_max_area)
        self.lowconf_thres = float(lowconf_thres)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        return np.stack([x1, y1, w, h], axis=1)

    def predict(
        self,
        frame: np.ndarray,
        image_id: int,
        category_whitelist: Optional[List[int]] = None,
    ) -> Tuple[List[Dict], CoarseStats]:
        """
        Returns:
          preds: List[{"image_id","category_id","bbox","score"}]
          stats: CoarseStats
        """
        boxes_np, scores_np, labels_np = self.detector.predict(frame)

        if boxes_np is None or len(boxes_np) == 0:
            stats = CoarseStats(
                num_boxes=0,
                num_small_boxes=0,
                mean_score=0.0,
                max_score=0.0,
                score_margin=0.0,
                num_lowconf=0,
            )
            return [], stats

        boxes_np = np.asarray(boxes_np, dtype=np.float32)
        scores_np = np.asarray(scores_np, dtype=np.float32)
        labels_np = np.asarray(labels_np, dtype=np.int64)

        xywh = self._xyxy_to_xywh(boxes_np)

        preds: List[Dict] = []
        keep_scores: List[float] = []
        keep_scores_all: List[float] = []

        for i in range(xywh.shape[0]):
            cid = int(labels_np[i])  # identity mapping: category_id = label_id
            sc = float(scores_np[i])

            keep_scores_all.append(sc)

            if category_whitelist is not None and cid not in set(category_whitelist):
                continue

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cid),
                    "bbox": [float(xywh[i, 0]), float(xywh[i, 1]), float(xywh[i, 2]), float(xywh[i, 3])],
                    "score": float(sc),
                }
            )
            keep_scores.append(sc)

        # Stats are computed on *all* outputs (before whitelist) to keep gating stable,
        # but you can switch to "after whitelist" if you want.
        scores_for_stats = keep_scores_all

        num_boxes = int(len(scores_for_stats))
        num_lowconf = int(sum([1 for s in scores_for_stats if s <= self.lowconf_thres]))
        max_score = float(max(scores_for_stats)) if num_boxes > 0 else 0.0
        mean_score = float(sum(scores_for_stats) / float(num_boxes)) if num_boxes > 0 else 0.0
        score_sorted = sorted(scores_for_stats, reverse=True)
        score_margin = float(score_sorted[0] - score_sorted[1]) if len(score_sorted) >= 2 else float(score_sorted[0]) if len(score_sorted) == 1 else 0.0

        # small box count (after whitelist preds)
        num_small = 0
        for p in preds:
            w = float(p["bbox"][2])
            h = float(p["bbox"][3])
            if (w * h) <= self.small_max_area:
                num_small += 1

        stats = CoarseStats(
            num_boxes=int(len(preds)),
            num_small_boxes=int(num_small),
            mean_score=mean_score,
            max_score=max_score,
            score_margin=score_margin,
            num_lowconf=num_lowconf,
        )
        return preds, stats
