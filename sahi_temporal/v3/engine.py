# sahi_temporal/v3/engine.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from sahi_base.tiler import compute_slice_grid
from sahi_base.merger import merge_boxes

try:
    from sahi_base.asahi_merger import merge_boxes_diou
except Exception:
    merge_boxes_diou = None

from .coarse_detector import CoarseDetector, CoarseStats
from .tracker import SimpleTracker, TrackerStats
from .risk import RiskEstimator, RiskConfig
from .controller import SAHIController, ControllerConfig


@dataclass
class V3Config:
    # tiling
    slice_height: int = 512
    slice_width: int = 512
    overlap_h: float = 0.2
    overlap_w: float = 0.2

    # merge
    merge_mode: str = "vanilla"  # "vanilla" or "diou"
    nms_iou: float = 0.5

    # bird-only gating option (optional)
    seed_category_ids: Optional[List[int]] = None  # e.g., [14]

    # local subset settings
    neighbor_hops: int = 1
    max_tiles: int = 6

    # risk/controller
    heartbeat_max_gap: int = 10
    min_key_gap: int = 2

    risk_tau: float = 1.0

    # exploration (used when subset has no seeds)
    explore_tiles: int = 2
    explore_stride: int = 3


class TemporalSahiV3Engine:
    def __init__(
        self,
        detector,
        cfg: Optional[V3Config] = None,
        class_mapping: Optional[Dict[int, int]] = None,
        enable_tracker: bool = True,
    ) -> None:
        self.cfg = cfg if cfg is not None else V3Config()
        self.class_mapping = class_mapping

        self.coarse = CoarseDetector(detector=detector)
        self.tracker = SimpleTracker() if enable_tracker else None
        self.risk = RiskEstimator(RiskConfig(risk_tau=float(self.cfg.risk_tau)))
        self.ctrl = SAHIController(ControllerConfig(
            heartbeat_max_gap=int(self.cfg.heartbeat_max_gap),
            min_key_gap=int(self.cfg.min_key_gap),
            max_tiles=int(self.cfg.max_tiles),
            neighbor_hops=int(self.cfg.neighbor_hops),
            risk_tau=float(self.cfg.risk_tau),
        ))

        self.tiles: Optional[List[Dict[str, int]]] = None
        self.id_to_rc: Optional[Dict[int, Tuple[int, int]]] = None
        self.rc_to_id: Optional[Dict[Tuple[int, int], int]] = None

        self.t = 0
        self.logs: List[Dict] = []
        self.explore_cursor = 0

    def _select_exploration_tiles(self, num_tiles_total: int) -> List[int]:
        n = int(self.cfg.explore_tiles)
        if n <= 0 or num_tiles_total <= 0:
            return []
        stride = max(1, int(self.cfg.explore_stride))
        start = int(self.explore_cursor) % num_tiles_total

        out: List[int] = []
        tid = start
        for _ in range(min(n, num_tiles_total)):
            out.append(int(tid))
            tid = (tid + stride) % num_tiles_total

        self.explore_cursor = int(tid)
        return out

    def _ensure_tiles(self, frame: np.ndarray) -> List[Dict[str, int]]:
        if self.tiles is not None:
            return self.tiles
        h, w = frame.shape[:2]
        self.tiles = compute_slice_grid(
            height=int(h),
            width=int(w),
            slice_height=int(self.cfg.slice_height),
            slice_width=int(self.cfg.slice_width),
            overlap_height_ratio=float(self.cfg.overlap_h),
            overlap_width_ratio=float(self.cfg.overlap_w),
        )
        return self.tiles

    @staticmethod
    def _bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
        x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return x + 0.5 * w, y + 0.5 * h

    @staticmethod
    def _point_in_tile(cx: float, cy: float, tile: Dict[str, int]) -> bool:
        x = float(tile["x"])
        y = float(tile["y"])
        w = float(tile["w"])
        h = float(tile["h"])
        return (x <= cx < x + w) and (y <= cy < y + h)

    def _tile_ids_covering_point(self, cx: float, cy: float, tiles: List[Dict[str, int]]) -> Set[int]:
        hits: Set[int] = set()
        for i, t in enumerate(tiles):
            if self._point_in_tile(cx, cy, t):
                hits.add(i)
        return hits

    def _neighbor_expand(self, tile_ids: Set[int], tiles: List[Dict[str, int]]) -> Set[int]:
        # Build a simple grid index on demand (ok for 6 tiles; can be optimized later)
        xs = sorted(set([int(t["x"]) for t in tiles]))
        ys = sorted(set([int(t["y"]) for t in tiles]))
        x_to_c = {x: i for i, x in enumerate(xs)}
        y_to_r = {y: i for i, y in enumerate(ys)}

        id_to_rc: Dict[int, Tuple[int, int]] = {}
        rc_to_id: Dict[Tuple[int, int], int] = {}
        for tid, t in enumerate(tiles):
            r = y_to_r[int(t["y"])]
            c = x_to_c[int(t["x"])]
            id_to_rc[tid] = (r, c)
            rc_to_id[(r, c)] = tid

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        out = set(tile_ids)
        frontier = set(tile_ids)
        for _ in range(int(self.cfg.neighbor_hops)):
            new_frontier = set()
            for tid in frontier:
                r, c = id_to_rc.get(tid, (None, None))
                if r is None:
                    continue
                for dr, dc in dirs:
                    rr = r + dr
                    cc = c + dc
                    key = (rr, cc)
                    if key in rc_to_id:
                        nid = rc_to_id[key]
                        if nid not in out:
                            out.add(nid)
                            new_frontier.add(nid)
            frontier = new_frontier
            if len(frontier) == 0:
                break
        return out

    def _select_subset_tiles(self, preds: List[Dict], tiles: List[Dict[str, int]]) -> List[int]:
        # Seed tiles from (optionally) bird-only detections
        allow = set(self.cfg.seed_category_ids) if self.cfg.seed_category_ids is not None else None
        seeds: Set[int] = set()
        for p in preds:
            if allow is not None and int(p["category_id"]) not in allow:
                continue
            cx, cy = self._bbox_center_xywh(p["bbox"])
            seeds |= self._tile_ids_covering_point(cx, cy, tiles)

        if len(seeds) == 0:
            # Use exploration tiles instead of returning empty
            return self._select_exploration_tiles(len(tiles))

        expanded = self._neighbor_expand(seeds, tiles)
        chosen = sorted(list(expanded))

        # Enforce max tiles budget
        if int(self.cfg.max_tiles) > 0 and len(chosen) > int(self.cfg.max_tiles):
            chosen = chosen[: int(self.cfg.max_tiles)]
        return chosen

    def _run_sahi_tiles(
        self,
        frame: np.ndarray,
        image_id: int,
        tile_ids: List[int],
        score_threshold: float,
    ) -> List[Dict]:
        tiles = self._ensure_tiles(frame)

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for tid in tile_ids:
            t = tiles[int(tid)]
            x = int(t["x"])
            y = int(t["y"])
            w = int(t["w"])
            h = int(t["h"])

            patch = frame[y:y+h, x:x+w]
            if patch.size == 0:
                continue

            boxes_np, scores_np, labels_np = self.coarse.detector.predict(patch)
            if boxes_np is None or len(boxes_np) == 0:
                continue

            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            scores = torch.as_tensor(scores_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)

            boxes[:, 0] += float(x)
            boxes[:, 1] += float(y)
            boxes[:, 2] += float(x)
            boxes[:, 3] += float(y)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if len(all_boxes) == 0:
            return []

        boxes_cat = torch.cat(all_boxes, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        if float(score_threshold) > 0.0:
            keep = scores_cat >= float(score_threshold)
            boxes_cat = boxes_cat[keep]
            scores_cat = scores_cat[keep]
            labels_cat = labels_cat[keep]

        if boxes_cat.numel() == 0:
            return []

        mode = str(self.cfg.merge_mode).lower().strip()
        if mode == "diou" and merge_boxes_diou is not None:
            m_boxes, m_scores, m_labels = merge_boxes_diou(
                boxes_xyxy=boxes_cat,
                scores=scores_cat,
                labels=labels_cat,
                iou_threshold=float(self.cfg.nms_iou),
            )
        else:
            m_boxes, m_scores, m_labels = merge_boxes(
                boxes_xyxy=boxes_cat,
                scores=scores_cat,
                labels=labels_cat,
                iou_threshold=float(self.cfg.nms_iou),
            )

        out: List[Dict] = []
        for i in range(m_boxes.shape[0]):
            x1 = float(m_boxes[i, 0])
            y1 = float(m_boxes[i, 1])
            x2 = float(m_boxes[i, 2])
            y2 = float(m_boxes[i, 3])
            bw = x2 - x1
            bh = y2 - y1
            cid = int(m_labels[i].item())
            out.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cid),
                    "bbox": [x1, y1, bw, bh],
                    "score": float(m_scores[i].item()),
                }
            )
        return out

    def process_frame(self, frame: np.ndarray, image_id: int, score_threshold: float = 0.0) -> Tuple[List[Dict], Dict]:
        tiles = self._ensure_tiles(frame)

        coarse_preds, coarse_stats = self.coarse.predict(frame, image_id=image_id)
        tracker_stats = self.tracker.update(coarse_preds) if self.tracker is not None else None

        risk_out = self.risk.score(frame, coarse_stats, tracker_stats)
        decision = self.ctrl.decide(
            t=self.t,
            risk_score=float(risk_out["risk_score"]),
            tracking_fail=bool(risk_out["flags"]["tracking_fail"]),
        )

        do_sahi = bool(decision["do_sahi"])
        mode = str(decision["mode"])

        final_preds = coarse_preds
        tile_ids: List[int] = []

        if do_sahi:
            if mode == "full":
                tile_ids = list(range(len(tiles)))
                final_preds = self._run_sahi_tiles(frame, image_id, tile_ids, score_threshold)
            elif mode == "subset":
                tile_ids = self._select_subset_tiles(coarse_preds, tiles)

                if len(tile_ids) == 0:
                    # Treat as skip (no SAHI executed)
                    do_sahi = False
                    mode = "skip"
                    final_preds = coarse_preds
                else:
                    final_preds = self._run_sahi_tiles(frame, image_id, tile_ids, score_threshold)

        log = {
            "image_id": int(image_id),
            "t": int(self.t),
            "do_sahi": bool(do_sahi),
            "mode": mode,
            "num_tiles_total": int(len(tiles)),
            "num_tiles_selected": int(len(tile_ids)) if do_sahi else 0,
            "risk_score": float(risk_out["risk_score"]),
            "flags": dict(risk_out["flags"]),
            "signals": {
                "global_motion": float(risk_out["signals"].global_motion),
                "blur_score": float(risk_out["signals"].blur_score),
                "coarse_num_boxes": int(risk_out["signals"].coarse_num_boxes),
                "coarse_num_small": int(risk_out["signals"].coarse_num_small),
                "coarse_max_score": float(risk_out["signals"].coarse_max_score),
                "coarse_score_margin": float(risk_out["signals"].coarse_score_margin),
                "tracking_fail": bool(risk_out["signals"].tracking_fail),
            },
            "counts": {
                "coarse_preds": int(len(coarse_preds)),
                "final_preds": int(len(final_preds)),
            },
        }
        self.logs.append(log)
        self.t += 1
        return final_preds, log
