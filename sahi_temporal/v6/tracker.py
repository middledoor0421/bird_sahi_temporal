# sahi_temporal/v6/tracker.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def iou_xywh(a: List[float], b: List[float]) -> float:
    """IoU for xywh boxes."""
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)

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


def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = map(float, b)
    return x + 0.5 * w, y + 0.5 * h


@dataclass
class Track:
    track_id: int
    category_id: int
    bbox_xywh: List[float]
    age: int
    miss: int
    last_score: float
    history: List[List[float]] = field(default_factory=list)


@dataclass
class TrackSummary:
    num_active: int
    max_miss_streak: int
    instability: float
    recent_lost: bool
    tracks: List[Track] = field(default_factory=list)


class TrackerV6:
    """Minimal tracking-by-detection tracker for v6 control.

    Notes:
        - This tracker is designed for control signals, not SOTA tracking.
        - Instability is computed from short bbox history (1 - IoU of last two).
    """

    def __init__(
        self,
        iou_thres: float = 0.3,
        max_miss: int = 2,
        history_size: int = 5,
        recent_lost_window: int = 3,
        category_whitelist: Optional[List[int]] = None,
        class_aware: bool = False,
    ) -> None:
        self.iou_thres = float(iou_thres)
        self.max_miss = int(max_miss)
        self.history_size = int(history_size)
        self.recent_lost_window = int(recent_lost_window)
        self.category_whitelist = list(category_whitelist) if category_whitelist is not None else None
        self.class_aware = bool(class_aware)

        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self._recent_lost_countdown = 0

    def _filter_dets(self, dets: List[Dict]) -> List[Dict]:
        if self.category_whitelist is None:
            return dets
        allow = set(int(x) for x in self.category_whitelist)
        out: List[Dict] = []
        for d in dets:
            if int(d.get("category_id", -1)) in allow:
                out.append(d)
        return out

    def _track_instability(self, tr: Track) -> float:
        if tr.history is None or len(tr.history) < 2:
            return 0.0
        a = tr.history[-2]
        b = tr.history[-1]
        v = iou_xywh(a, b)
        v = max(0.0, min(1.0, float(v)))
        return float(1.0 - v)

    def update(self, dets: List[Dict]) -> TrackSummary:
        dets_f = self._filter_dets(dets)

        det_boxes = [d["bbox"] for d in dets_f]
        det_scores = [float(d.get("score", 0.0)) for d in dets_f]
        det_cids = [int(d.get("category_id", -1)) for d in dets_f]

        assigned_det = set()
        new_tracks: Dict[int, Track] = {}
        dropped = 0

        # Greedy IoU matching
        for tid, tr in self.tracks.items():
            best_iou = 0.0
            best_j = -1
            for j, db in enumerate(det_boxes):
                if j in assigned_det:
                    continue
                if self.class_aware and int(det_cids[j]) != int(tr.category_id):
                    continue
                v = iou_xywh(tr.bbox_xywh, db)
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_iou >= self.iou_thres and best_j >= 0:
                assigned_det.add(best_j)
                nb = list(det_boxes[best_j])
                nh = list(tr.history) if tr.history is not None else []
                nh.append(nb)
                if len(nh) > self.history_size:
                    nh = nh[-self.history_size :]

                new_tracks[tid] = Track(
                    track_id=int(tid),
                    category_id=int(det_cids[best_j]),
                    bbox_xywh=nb,
                    age=int(tr.age) + 1,
                    miss=0,
                    last_score=float(det_scores[best_j]),
                    history=nh,
                )
            else:
                # miss
                tr.miss = int(tr.miss) + 1
                tr.age = int(tr.age) + 1
                if tr.miss > self.max_miss:
                    dropped += 1
                else:
                    # Keep bbox/history as-is
                    new_tracks[tid] = tr

        # Create new tracks for unmatched detections
        for j, db in enumerate(det_boxes):
            if j in assigned_det:
                continue
            tid = self.next_id
            self.next_id += 1
            nb = list(db)
            new_tracks[tid] = Track(
                track_id=int(tid),
                category_id=int(det_cids[j]),
                bbox_xywh=nb,
                age=1,
                miss=0,
                last_score=float(det_scores[j]),
                history=[nb],
            )

        # Update recent_lost countdown
        if dropped > 0:
            self._recent_lost_countdown = self.recent_lost_window
        else:
            self._recent_lost_countdown = max(0, int(self._recent_lost_countdown) - 1)

        self.tracks = new_tracks

        # Summary stats
        tracks_list = list(new_tracks.values())
        num_active = int(len(tracks_list))
        max_miss = int(max([tr.miss for tr in tracks_list], default=0))
        inst_list = [self._track_instability(tr) for tr in tracks_list]
        instability = float(sum(inst_list) / float(len(inst_list))) if len(inst_list) > 0 else 0.0
        recent_lost = bool(self._recent_lost_countdown > 0)

        return TrackSummary(
            num_active=num_active,
            max_miss_streak=max_miss,
            instability=instability,
            recent_lost=recent_lost,
            tracks=tracks_list,
        )

    def get_tracks(self) -> List[Track]:
        return list(self.tracks.values())
