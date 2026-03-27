# sahi_base/asahi_merger.py
# DIoU-NMS for merging dense boxes (used as Cluster-DIoU-NMS core similarity).
# Python 3.9 compatible. Comments in English only.

from typing import Tuple
import torch


def _diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute DIoU between boxes1 [N,4] and boxes2 [M,4] in xyxy.
    DIoU = IoU - (rho^2 / c^2)
    """
    # Intersection
    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0].unsqueeze(0), boxes2[:, 1].unsqueeze(0), boxes2[:, 2].unsqueeze(0), boxes2[:, 3].unsqueeze(0)

    ix1 = torch.maximum(x11, x21)
    iy1 = torch.maximum(y11, y21)
    ix2 = torch.minimum(x12, x22)
    iy2 = torch.minimum(y12, y22)

    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    # Areas
    area1 = torch.clamp(x12 - x11, min=0.0) * torch.clamp(y12 - y11, min=0.0)
    area2 = torch.clamp(x22 - x21, min=0.0) * torch.clamp(y22 - y21, min=0.0)
    union = area1 + area2 - inter
    iou = inter / torch.clamp(union, min=1e-9)

    # Center distance
    cx1 = (x11 + x12) * 0.5
    cy1 = (y11 + y12) * 0.5
    cx2 = (x21 + x22) * 0.5
    cy2 = (y21 + y22) * 0.5
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Smallest enclosing box diagonal
    ex1 = torch.minimum(x11, x21)
    ey1 = torch.minimum(y11, y21)
    ex2 = torch.maximum(x12, x22)
    ey2 = torch.maximum(y12, y22)
    c2 = torch.clamp(ex2 - ex1, min=0.0) ** 2 + torch.clamp(ey2 - ey1, min=0.0) ** 2

    diou = iou - rho2 / torch.clamp(c2, min=1e-9)
    return diou


def diou_nms(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Greedy DIoU-NMS (class-agnostic). Returns kept indices.
    """
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        diou_vals = _diou(boxes_xyxy[i:i+1], boxes_xyxy[rest]).squeeze(0)

        # Suppress boxes with DIoU > threshold (paper replaces IoU with DIoU)
        mask = diou_vals <= float(iou_threshold)
        order = rest[mask]

    return torch.as_tensor(keep, dtype=torch.long, device=boxes_xyxy.device)


def merge_boxes_diou(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Class-aware DIoU-NMS: run DIoU-NMS per label.
    """
    if boxes_xyxy.numel() == 0:
        device = boxes_xyxy.device
        return (
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

    keep_all = []
    for cls in torch.unique(labels):
        cls = int(cls.item())
        m = labels == cls
        idxs = torch.nonzero(m, as_tuple=False).squeeze(1)
        k = diou_nms(boxes_xyxy[idxs], scores[idxs], iou_threshold=iou_threshold)
        keep_all.append(idxs[k])

    keep = torch.cat(keep_all, dim=0) if len(keep_all) > 0 else torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)
    # Keep final order by score desc
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return boxes_xyxy[keep], scores[keep], labels[keep]
