# sahi_ext/temporal_scheduler.py
# Temporal scheduler that uses motion tiles + weak detections.
# Python 3.9 compatible.

from typing import List, Dict


def _overlap(tile: Dict, box_xyxy: List[float]) -> bool:
    """Return True if tile intersects with box (at least 1 pixel)."""
    x, y, w, h = tile["x"], tile["y"], tile["w"], tile["h"]
    tx1, ty1, tx2, ty2 = x, y, x + w, y + h
    bx1, by1, bx2, by2 = box_xyxy
    ix1 = max(tx1, bx1)
    iy1 = max(ty1, by1)
    ix2 = min(tx2, bx2)
    iy2 = min(ty2, by2)
    return (ix2 - ix1) > 0 and (iy2 - iy1) > 0


class TemporalScheduler:
    """
    Compute a priority score per tile using:
      - motion (moving_idxs)
      - weak detections (small, low-confidence boxes)

    Then return tile indices sorted by score (descending).
    """

    def __init__(self) -> None:
        pass

    def select(
        self,
        tile_infos: List[Dict],
        moving_idxs: List[int],
        weak_boxes: List[List[float]],
    ) -> List[int]:
        """
        Args:
            tile_infos: list of {"x","y","w","h"}
            moving_idxs: tile indices that are motion-stable
            weak_boxes: list of weak boxes [x1,y1,x2,y2] in full-image coords

        Returns:
            List of tile indices sorted by priority (high → low).
        """
        n = len(tile_infos)
        if n == 0:
            return []

        scores = [0 for _ in range(n)]

        # motion contributes +1
        for idx in moving_idxs:
            if 0 <= idx < n:
                scores[idx] += 1

        # weak boxes contribute +2
        if weak_boxes:
            for i in range(n):
                info = tile_infos[i]
                for box in weak_boxes:
                    if _overlap(info, box):
                        scores[i] += 2
                        break

        # choose tiles with positive score, sorted by score desc
        idxs = [i for i, s in enumerate(scores) if s > 0]
        idxs.sort(key=lambda i: scores[i], reverse=True)
        return idxs
