# sahi_temporal/v3/tracker.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


@dataclass
class TrackState:
    track_id: int
    bbox_xywh: List[float]
    age: int
    miss: int
    last_score: float


@dataclass
class TrackerStats:
    num_tracks: int
    num_dropped: int
    tracking_fail: bool


class SimpleTracker:
    """
    Minimal tracking-by-detection for a 'tracking_fail' signal.
    This is not intended to maximize tracking quality; only to detect stability loss.
    """

    def __init__(self, iou_thres: float = 0.3, max_miss: int = 2) -> None:
        self.iou_thres = float(iou_thres)
        self.max_miss = int(max_miss)
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}

    def update(self, dets: List[Dict]) -> TrackerStats:
        # Only use bbox + score; ignore class for now (can be class-aware if needed)
        det_boxes = [d["bbox"] for d in dets]
        det_scores = [float(d.get("score", 0.0)) for d in dets]

        assigned_det = set()
        new_tracks: Dict[int, TrackState] = {}
        dropped = 0

        # Match existing tracks greedily
        for tid, tr in self.tracks.items():
            best_iou = 0.0
            best_j = -1
            for j, db in enumerate(det_boxes):
                if j in assigned_det:
                    continue
                v = iou_xywh(tr.bbox_xywh, db)
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_iou >= self.iou_thres and best_j >= 0:
                assigned_det.add(best_j)
                new_tracks[tid] = TrackState(
                    track_id=tid,
                    bbox_xywh=list(det_boxes[best_j]),
                    age=tr.age + 1,
                    miss=0,
                    last_score=det_scores[best_j],
                )
            else:
                # miss
                tr.miss += 1
                tr.age += 1
                if tr.miss > self.max_miss:
                    dropped += 1
                else:
                    new_tracks[tid] = tr

        # Create tracks for unmatched detections
        for j, db in enumerate(det_boxes):
            if j in assigned_det:
                continue
            tid = self.next_id
            self.next_id += 1
            new_tracks[tid] = TrackState(
                track_id=tid,
                bbox_xywh=list(db),
                age=1,
                miss=0,
                last_score=det_scores[j],
            )

        # tracking_fail heuristic: a sudden drop in active tracks
        prev_n = len(self.tracks)
        cur_n = len(new_tracks)
        tracking_fail = (prev_n >= 2) and (cur_n <= max(0, prev_n - 2))

        self.tracks = new_tracks
        return TrackerStats(num_tracks=cur_n, num_dropped=dropped, tracking_fail=bool(tracking_fail))
