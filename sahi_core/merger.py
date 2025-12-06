# sahi_core/merger.py
# Merge per-tile detections into full-image detections with NMS.
# Python 3.9 compatible.

from typing import List, Dict
import numpy as np


def _iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return float(inter / max(1e-6, area1 + area2 - inter))


def nms(dets: List[Dict], iou_th: float) -> List[Dict]:
    """Simple NMS over detections. Ignores class for now (class-agnostic NMS)."""
    if not dets:
        return []
    boxes = np.array([d["xyxy"] for d in dets], dtype=np.float32)
    scores = np.array([d["conf"] for d in dets], dtype=np.float32)
    idxs = scores.argsort()[::-1]  # descending

    keep: List[int] = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([_iou(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        rest = rest[ious <= iou_th]
        idxs = rest

    return [dets[i] for i in keep]


def merge_tile_detections(
    per_tile_dets: List[List[Dict]],
    tile_infos: List[Dict],
    iou_th: float = 0.5
) -> List[Dict]:
    """
    Merge detections from tiles into full-image detections.

    Args:
        per_tile_dets: list over tiles, each is list of det dicts
        tile_infos: list of {"x","y","w","h"} per tile
        iou_th: IoU threshold for NMS

    Returns:
        merged list of det dicts in full-image coordinates.
    """
    all_dets: List[Dict] = []
    for dets, info in zip(per_tile_dets, tile_infos):
        ox = info["x"]
        oy = info["y"]
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            # shift to original coord
            full_box = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
            all_dets.append(
                {
                    "xyxy": full_box,
                    "conf": d["conf"],
                    "cls": d["cls"],
                }
            )
    # NMS
    merged = nms(all_dets, iou_th=iou_th)
    return merged
