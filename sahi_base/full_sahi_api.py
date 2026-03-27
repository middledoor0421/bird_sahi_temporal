# sahi_base/full_sahi_api.py
# Full SAHI inference API built on top of tiler + merger.
# Python 3.9 compatible. Comments in English only.

from typing import List, Dict, Any, Tuple

import numpy as np
import torch

from .tiler import slice_image, count_slices
from .merger import merge_boxes
from utils.category_mapper import map_label_to_category_id
from .postprocess import postprocess_preds, get_pp_preset

class FullSahiModel:
    """
    Full SAHI inference model.

    Detector interface:
        detector must provide a method

            predict(frame: np.ndarray) -> (boxes_xyxy, scores, labels)

        where:
            boxes_xyxy: np.ndarray of shape [N, 4] in (x1, y1, x2, y2)
            scores: np.ndarray of shape [N]
            labels: np.ndarray of shape [N] (int class ids)

        This is intentionally simple so that you can adapt your YOLO wrapper
        to this interface easily.
    """

    def __init__(
        self,
        detector,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        nms_iou_threshold: float = 0.5,
        class_mapping=None,
    ) -> None:
        self.detector = detector
        self.slice_height = int(slice_height)
        self.slice_width = int(slice_width)
        self.overlap_height_ratio = float(overlap_height_ratio)
        self.overlap_width_ratio = float(overlap_width_ratio)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.class_mapping = class_mapping

    def _detect_on_tile(
        self,
        tile: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run detector on a single tile and shift boxes to global coordinates.
        """
        boxes_np, scores_np, labels_np = self.detector.predict(tile)
        if boxes_np is None or len(boxes_np) == 0:
            device = torch.device("cpu")
            return (
                torch.empty((0, 4), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.int64, device=device),
            )

        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        scores = torch.as_tensor(scores_np, dtype=torch.float32)
        labels = torch.as_tensor(labels_np, dtype=torch.int64)

        # shift boxes to global coordinates
        boxes[:, 0] += float(offset_x)
        boxes[:, 1] += float(offset_y)
        boxes[:, 2] += float(offset_x)
        boxes[:, 3] += float(offset_y)

        return boxes, scores, labels

    def predict_frame(
        self,
        frame: np.ndarray,
        image_id: int,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Run full SAHI inference on a single frame.

        Returns:
            list of COCO-style prediction dicts:
                {
                    "image_id": image_id,
                    "category_id": int,
                    "bbox": [x, y, w, h],
                    "score": float,
                }
        """
        height, width = frame.shape[:2]
        num_slices = count_slices(
            height=height,
            width=width,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        tiles = slice_image(
            frame,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )

        for tile_img, info in tiles:
            x = info["x"]
            y = info["y"]
            boxes, scores, labels = self._detect_on_tile(tile_img, x, y)
            if boxes.numel() == 0:
                continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if len(all_boxes) == 0:
            return []

        boxes_cat = torch.cat(all_boxes, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        # score threshold before NMS
        if score_threshold > 0.0:
            keep_mask = scores_cat >= float(score_threshold)
            boxes_cat = boxes_cat[keep_mask]
            scores_cat = scores_cat[keep_mask]
            labels_cat = labels_cat[keep_mask]

        if boxes_cat.numel() == 0:
            return []

        merged_boxes, merged_scores, merged_labels = merge_boxes(
            boxes_xyxy=boxes_cat,
            scores=scores_cat,
            labels=labels_cat,
            iou_threshold=self.nms_iou_threshold,
        )

        preds: List[Dict[str, Any]] = []
        for i in range(merged_boxes.shape[0]):
            x1 = float(merged_boxes[i, 0])
            y1 = float(merged_boxes[i, 1])
            x2 = float(merged_boxes[i, 2])
            y2 = float(merged_boxes[i, 3])
            w = x2 - x1

            h = y2 - y1

            label_id = int(merged_labels[i].item())
            category_id = map_label_to_category_id(
                label_id=label_id,
                class_mapping=self.class_mapping,
            )

            pred = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [x1, y1, w, h],
                "score": float(merged_scores[i].item()),
            }
            preds.append(pred)
        # Common postprocess (default: clamp only)
        pp_cfg = get_pp_preset("icct_std_v1")
        preds_pp, _pp_meta = postprocess_preds(preds, image_w=int(width), image_h=int(height), cfg=pp_cfg)
        return preds_pp

