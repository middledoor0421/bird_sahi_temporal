# sahi_temporal/v7/confirmation.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def _xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x = float(b[0])
    y = float(b[1])
    w = float(b[2])
    h = float(b[3])
    return x, y, x + w, y + h


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


@dataclass
class ConfirmationConfig:
    iou_thr: float = 0.4
    confirm_len: int = 2
    max_age: int = 2  # Frames to keep unmatched candidates


@dataclass
class _Candidate:
    bbox_xyxy: Tuple[float, float, float, float]
    category_id: int
    streak: int
    age: int  # Frames since last matched


class ConfirmationGate:
    """
    Simple temporal confirmation gate:
    - Maintain candidates across frames with greedy IoU matching.
    - Confirm if streak >= confirm_len.
    """

    def __init__(self, cfg: ConfirmationConfig):
        self.cfg = cfg
        self._cands: List[_Candidate] = []

    def reset(self) -> None:
        self._cands = []

    def apply(
        self,
        preds: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        iou_thr = float(self.cfg.iou_thr)
        confirm_len = int(self.cfg.confirm_len)
        max_age = int(self.cfg.max_age)

        num_in = int(len(preds))

        # Convert current preds to xyxy
        cur_boxes: List[Tuple[float, float, float, float]] = []
        cur_cats: List[int] = []
        for p in preds:
            b = p.get("bbox", None)
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                cur_boxes.append((0.0, 0.0, 0.0, 0.0))
                cur_cats.append(int(p.get("category_id", 0)))
                continue
            cur_boxes.append(_xywh_to_xyxy([float(b[0]), float(b[1]), float(b[2]), float(b[3])]))
            cur_cats.append(int(p.get("category_id", 0)))

        # Age existing candidates
        for c in self._cands:
            c.age += 1

        matched_cand = [False for _ in range(len(self._cands))]
        matched_pred = [False for _ in range(len(preds))]

        # Greedy matching: each pred matches best candidate of same class
        for i in range(len(preds)):
            best_j = -1
            best_iou = 0.0
            for j, c in enumerate(self._cands):
                if matched_cand[j]:
                    continue
                if c.category_id != cur_cats[i]:
                    continue
                iou_val = _iou_xyxy(c.bbox_xyxy, cur_boxes[i])
                if iou_val >= iou_thr and iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_j >= 0:
                c = self._cands[best_j]
                c.bbox_xyxy = cur_boxes[i]
                c.streak += 1
                c.age = 0
                matched_cand[best_j] = True
                matched_pred[i] = True

        # New candidates from unmatched preds
        for i in range(len(preds)):
            if matched_pred[i]:
                continue
            self._cands.append(
                _Candidate(
                    bbox_xyxy=cur_boxes[i],
                    category_id=cur_cats[i],
                    streak=1,
                    age=0,
                )
            )

        # Drop old candidates
        self._cands = [c for c in self._cands if c.age <= max_age]

        # Decide confirmation for current preds
        confirmed: List[Dict[str, Any]] = []
        unconfirmed: List[Dict[str, Any]] = []

        if confirm_len <= 1:
            confirmed = list(preds)
        else:
            # A pred is confirmed if it matches any candidate with streak >= confirm_len
            for i in range(len(preds)):
                is_conf = False
                for c in self._cands:
                    if c.category_id != cur_cats[i]:
                        continue
                    if _iou_xyxy(c.bbox_xyxy, cur_boxes[i]) >= iou_thr and c.streak >= confirm_len:
                        is_conf = True
                        break
                if is_conf:
                    confirmed.append(preds[i])
                else:
                    unconfirmed.append(preds[i])

        meta = {
            "num_in": int(num_in),
            "num_confirmed": int(len(confirmed)),
            "num_unconfirmed": int(len(unconfirmed)),
            "iou_thr": float(iou_thr),
            "confirm_len": int(confirm_len),
            "max_age": int(max_age),
            "num_candidates": int(len(self._cands)),
        }
        return confirmed, unconfirmed, meta
