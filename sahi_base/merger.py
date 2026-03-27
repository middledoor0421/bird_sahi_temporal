# sahi_base/merger.py
# SAHI-style prediction merging utilities (per-class NMS).
# Python 3.9 compatible. Comments in English only.

from typing import Tuple
import torch
from torchvision.ops import nms


def merge_boxes(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply per-class NMS to merge overlapping predictions.

    Args:
        boxes_xyxy: tensor of shape [N, 4] in (x1, y1, x2, y2)
        scores: tensor of shape [N]
        labels: tensor of shape [N] (int class ids)
        iou_threshold: IoU threshold for NMS

    Returns:
        kept_boxes, kept_scores, kept_labels
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy, scores, labels

    kept_boxes_list = []
    kept_scores_list = []
    kept_labels_list = []

    unique_labels = labels.unique()
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_boxes.numel() == 0:
            continue

        keep_idx = nms(cls_boxes, cls_scores, iou_threshold)
        kept_boxes_list.append(cls_boxes[keep_idx])
        kept_scores_list.append(cls_scores[keep_idx])
        kept_labels_list.append(labels[cls_mask][keep_idx])

    if len(kept_boxes_list) == 0:
        return (
            torch.empty((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device),
            torch.empty((0,), dtype=scores.dtype, device=scores.device),
            torch.empty((0,), dtype=labels.dtype, device=labels.device),
        )

    kept_boxes = torch.cat(kept_boxes_list, dim=0)
    kept_scores = torch.cat(kept_scores_list, dim=0)
    kept_labels = torch.cat(kept_labels_list, dim=0)

    return kept_boxes, kept_scores, kept_labels
