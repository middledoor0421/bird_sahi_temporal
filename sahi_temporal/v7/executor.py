# sahi_temporal/v7/executor.py
# Python 3.9 compatible. Comments in English only.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sahi_base.merger import merge_boxes

try:
    from sahi_base.asahi_merger import merge_boxes_diou
except Exception:
    merge_boxes_diou = None

from utils.category_mapper import map_label_to_category_id


class TileExecutorV7:
    """SAHI executor that runs detector on selected tiles and merges outputs."""

    def __init__(
        self,
        detector,
        tiles: List[Dict[str, int]],
        merge_mode: str = "vanilla",
        nms_iou: float = 0.5,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        self.detector = detector
        self.tiles = tiles
        self.merge_mode = str(merge_mode)
        self.nms_iou = float(nms_iou)
        self.class_mapping = class_mapping

    def _detect_on_tile(
        self,
        frame: np.ndarray,
        tile: Dict[str, int],
        score_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = int(tile["x"])
        y = int(tile["y"])
        w = int(tile["w"])
        h = int(tile["h"])
        patch = frame[y : y + h, x : x + w]

        boxes_np, scores_np, labels_np = self.detector.predict(patch, conf_thres=score_threshold)
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

        # Shift to global coordinates
        boxes[:, 0] += float(x)
        boxes[:, 1] += float(y)
        boxes[:, 2] += float(x)
        boxes[:, 3] += float(y)

        if score_threshold > 0.0:
            m = scores >= float(score_threshold)
            boxes = boxes[m]
            scores = scores[m]
            labels = labels[m]

        return boxes, scores, labels

    def run(
        self,
        frame: np.ndarray,
        image_id: int,
        tile_ids: List[int],
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if tile_ids is None or len(tile_ids) == 0:
            return []

        boxes_all: List[torch.Tensor] = []
        scores_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []

        for tid in tile_ids:
            if tid < 0 or tid >= len(self.tiles):
                continue
            tile = self.tiles[int(tid)]
            b, s, l = self._detect_on_tile(frame, tile, float(score_threshold))
            if b.numel() == 0:
                continue
            boxes_all.append(b)
            scores_all.append(s)
            labels_all.append(l)

        if len(boxes_all) == 0:
            return []

        boxes_cat = torch.cat(boxes_all, dim=0)
        scores_cat = torch.cat(scores_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)

        merge_mode = str(self.merge_mode).lower().strip()
        if merge_mode == "diou" and merge_boxes_diou is not None:
            mb, ms, ml = merge_boxes_diou(
                boxes_xyxy=boxes_cat,
                scores=scores_cat,
                labels=labels_cat,
                iou_threshold=float(self.nms_iou),
            )
        else:
            mb, ms, ml = merge_boxes(
                boxes_xyxy=boxes_cat,
                scores=scores_cat,
                labels=labels_cat,
                iou_threshold=float(self.nms_iou),
            )

        preds: List[Dict[str, Any]] = []
        for i in range(int(mb.shape[0])):
            x1 = float(mb[i, 0].item())
            y1 = float(mb[i, 1].item())
            x2 = float(mb[i, 2].item())
            y2 = float(mb[i, 3].item())
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            label_id = int(ml[i].item())
            category_id = map_label_to_category_id(label_id=label_id, class_mapping=self.class_mapping)

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(ms[i].item()),
                }
            )

        return preds


class CropRecheckExecutorV7:
    """Crop-and-resize verifier that rechecks selected tile regions."""

    def __init__(
        self,
        detector,
        tiles: List[Dict[str, int]],
        crop_margin_ratio: float = 0.2,
        crop_input_size: int = 960,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        self.detector = detector
        self.tiles = tiles
        self.crop_margin_ratio = max(0.0, float(crop_margin_ratio))
        self.crop_input_size = max(32, int(crop_input_size))
        self.class_mapping = class_mapping

    def _expand_tile(self, tile: Dict[str, int], image_w: int, image_h: int) -> Tuple[int, int, int, int]:
        x = int(tile["x"])
        y = int(tile["y"])
        w = int(tile["w"])
        h = int(tile["h"])
        mx = int(round(float(w) * self.crop_margin_ratio))
        my = int(round(float(h) * self.crop_margin_ratio))
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(int(image_w), x + w + mx)
        y2 = min(int(image_h), y + h + my)
        return x1, y1, x2, y2

    def run(
        self,
        frame: np.ndarray,
        image_id: int,
        tile_ids: List[int],
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if tile_ids is None or len(tile_ids) == 0:
            return []

        image_h, image_w = frame.shape[:2]
        preds: List[Dict[str, Any]] = []

        for tid in tile_ids:
            if tid < 0 or tid >= len(self.tiles):
                continue

            tile = self.tiles[int(tid)]
            x1, y1, x2, y2 = self._expand_tile(tile, image_w=image_w, image_h=image_h)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            boxes_np, scores_np, labels_np = self.detector.predict(
                crop,
                conf_thres=score_threshold,
                img_size=self.crop_input_size,
            )
            if boxes_np is None or len(boxes_np) == 0:
                continue

            boxes_np = np.asarray(boxes_np, dtype=np.float32)
            scores_np = np.asarray(scores_np, dtype=np.float32)
            labels_np = np.asarray(labels_np, dtype=np.int64)

            boxes_np[:, 0] += float(x1)
            boxes_np[:, 1] += float(y1)
            boxes_np[:, 2] += float(x1)
            boxes_np[:, 3] += float(y1)

            for i in range(int(boxes_np.shape[0])):
                sc = float(scores_np[i])
                if score_threshold > 0.0 and sc < float(score_threshold):
                    continue

                bx1 = float(boxes_np[i, 0])
                by1 = float(boxes_np[i, 1])
                bx2 = float(boxes_np[i, 2])
                by2 = float(boxes_np[i, 3])
                bw = max(0.0, bx2 - bx1)
                bh = max(0.0, by2 - by1)

                label_id = int(labels_np[i])
                category_id = map_label_to_category_id(label_id=label_id, class_mapping=self.class_mapping)

                preds.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": [float(bx1), float(by1), float(bw), float(bh)],
                        "score": float(sc),
                    }
                )

        return preds
