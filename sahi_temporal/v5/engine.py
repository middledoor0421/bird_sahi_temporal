# sahi_temporal/v5/engine.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sahi_base.tiler import compute_slice_grid
from sahi_base.merger import merge_boxes
from utils.category_mapper import map_label_to_category_id

try:
    from sahi_base.asahi_merger import merge_boxes_diou
except Exception:
    merge_boxes_diou = None

from sahi_temporal.v3.coarse_detector import CoarseDetector
from sahi_temporal.v3.tracker import SimpleTracker
from sahi_temporal.v4.motion_activity import MotionExtractor, MotionConfig

from .diff_score import DiffTileScorer, DiffScoreConfig
from .tile_selector import TileSelectorV5, TileSelectorV5Config
from .controller import SAHIControllerV5, ControllerConfigV5


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


@dataclass
class V5Config:
    # tiling (fixed grid = Full-SAHI)
    slice_height: int = 512
    slice_width: int = 512
    overlap_h: float = 0.2
    overlap_w: float = 0.2

    # merge
    merge_mode: str = "vanilla"  # "vanilla" or "diou"
    nms_iou: float = 0.5

    # triggers / streaks
    m_escape_soft: int = 2  # M fixed
    escape_soft_streak_thr: int = 10  # escape_soft_raw threshold on bird-miss streak
    subset_fail_full_streak: int = 2  # F (sweep)
    escape_hard_streak: int = 90  # T_hard-like (hard-only full)
    strong_skip_streak: int = 20

    # Full TTL (NEW)
    full_ttl: int = 3  # full mode duration in frames (3~5 recommended)

    # z-score for motion (escape detection)
    z_mu_alpha: float = 0.98
    z_dev_alpha: float = 0.98
    z_k_low: float = 1.0
    z_k_high: float = 1.5
    z_eps: float = 1e-3

    # optional subset throttle (kept in controller)
    subset_min_interval: int = 0

    # motion extractor (for gm_z)
    motion_mode: str = "lag"
    motion_lag: int = 4
    motion_accum_window: int = 5
    motion_downsample_factor: int = 4
    motion_ema_alpha: float = 0.8

    # diff tile scoring (subset localization only)
    diff_mode: str = "frame"  # "frame" or "ema_bg"
    diff_downsample: int = 4
    diff_bg_alpha: float = 0.05
    diff_tile_ema: float = 0.0

    # subset tile selection (cheap probe)
    k_min: int = 2
    k_max: int = 3
    k_diff_extra: int = 1
    neighbor_hops: int = 1
    dilate_on_seed_empty: int = 0  # final spec: off
    dilate_min_streak: int = 999999  # disable

    # bird seed
    seed_category_ids: Optional[List[int]] = None
    seed_min_score: float = 0.0

    # track-aware subset success
    track_aware_success: int = 1
    track_cont_iou: float = 0.3
    subset_success_score_thr: float = 0.15

    # tracker  �
    enable_tracker: int = 1


class TemporalSahiV5Engine:
    def __init__(self, detector, cfg: Optional[V5Config] = None,
                 class_mapping: Optional[Dict[int, int]] = None) -> None:
        self.cfg = cfg if cfg is not None else V5Config()
        self.class_mapping = class_mapping

        self.coarse = CoarseDetector(detector=detector)
        self.tracker = SimpleTracker() if bool(int(self.cfg.enable_tracker)) else None

        self.motion = MotionExtractor(MotionConfig(
            downsample_factor=int(self.cfg.motion_downsample_factor),
            ema_alpha=float(self.cfg.motion_ema_alpha),
            mode=str(self.cfg.motion_mode),
            lag=int(self.cfg.motion_lag),
            accum_window=int(self.cfg.motion_accum_window),
        ))

        self.diff = DiffTileScorer(DiffScoreConfig(
            mode=str(self.cfg.diff_mode),
            downsample_factor=int(self.cfg.diff_downsample),
            bg_alpha=float(self.cfg.diff_bg_alpha),
            tile_score_ema=float(self.cfg.diff_tile_ema),
        ))

        self.selector = TileSelectorV5(TileSelectorV5Config(
            k_min=int(self.cfg.k_min),
            k_max=int(self.cfg.k_max),
            k_diff_extra=int(self.cfg.k_diff_extra),
            neighbor_hops=int(self.cfg.neighbor_hops),
            dilate_on_seed_empty=int(self.cfg.dilate_on_seed_empty),
            dilate_min_streak=int(self.cfg.dilate_min_streak),
            seed_category_ids=self.cfg.seed_category_ids,
            seed_min_score=float(self.cfg.seed_min_score),
        ))

        self.ctrl = SAHIControllerV5(ControllerConfigV5(
            m_escape_soft=int(self.cfg.m_escape_soft),
            subset_min_interval=int(self.cfg.subset_min_interval),
        ))

        self.tiles: Optional[List[Dict[str, int]]] = None
        self.t = 0

        # streaks
        self.no_det_streak_all = 0
        self.no_det_streak_bird = 0
        self.escape_soft_streak = 0

        # z-score state
        self.gm_mu = 0.0
        self.gm_dev = 1.0

        # subset escalation
        self.subset_fail_streak = 0

        # Full TTL state (NEW)
        self.full_ttl_left = 0
        self.prev_mode = "skip"

        # track state
        self.track_active = False
        self.last_track_box: Optional[List[float]] = None

        self.logs: List[Dict] = []

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

    def _update_zscore(self, global_motion: float) -> Tuple[float, float, float]:
        mu_a = float(self.cfg.z_mu_alpha)
        dev_a = float(self.cfg.z_dev_alpha)
        eps = float(self.cfg.z_eps)

        gm = float(global_motion)
        self.gm_mu = mu_a * float(self.gm_mu) + (1.0 - mu_a) * gm
        dev_raw = abs(gm - float(self.gm_mu))
        self.gm_dev = dev_a * float(self.gm_dev) + (1.0 - dev_a) * dev_raw
        z = (gm - float(self.gm_mu)) / (float(self.gm_dev) + eps)
        return gm, float(self.gm_mu), float(z)

    def _best_bird_pred(self, preds: List[Dict[str, Any]]) -> Tuple[Optional[List[float]], float]:
        allow = set(self.cfg.seed_category_ids) if self.cfg.seed_category_ids is not None else None
        best_box = None
        best_s = -1.0
        for p in preds:
            cid = int(p.get("category_id", -1))
            if allow is not None and cid not in allow:
                continue
            s = float(p.get("score", 0.0))
            b = p.get("bbox", None)
            if b is None:
                continue
            if s > best_s:
                best_s = s
                best_box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        return best_box, float(best_s)

    def _new_init_success(self, preds: List[Dict[str, Any]]) -> bool:
        box, score = self._best_bird_pred(preds)
        return (box is not None) and (float(score) >= float(self.cfg.subset_success_score_thr))

    def _existing_track_continued(self, cur_box: Optional[List[float]]) -> bool:
        if (not self.track_active) or (self.last_track_box is None) or (cur_box is None):
            return False
        return iou_xywh(self.last_track_box, cur_box) >= float(self.cfg.track_cont_iou)

    def _run_sahi_tiles(self, frame: np.ndarray, image_id: int, tile_ids: List[int], score_threshold: float) -> List[
        Dict]:
        tiles = self.tiles if self.tiles is not None else self._ensure_tiles(frame)

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for tid in tile_ids:
            tinfo = tiles[int(tid)]
            x = int(tinfo["x"])
            y = int(tinfo["y"])
            w = int(tinfo["w"])
            h = int(tinfo["h"])

            patch = frame[y:y + h, x:x + w]
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

        merge_mode = str(self.cfg.merge_mode).lower().strip()
        if merge_mode == "diou" and merge_boxes_diou is not None:
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

            label_id = int(m_labels[i].item())
            cid = map_label_to_category_id(label_id=label_id, class_mapping=self.class_mapping)

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

        _, global_motion_raw = self.motion.compute(frame, tiles)
        _, gm_mu, gm_z = self._update_zscore(global_motion_raw)

        low_motion = bool(gm_z < -float(self.cfg.z_k_low))
        empty_change = bool(gm_z > float(self.cfg.z_k_high))

        coarse_preds, _ = self.coarse.predict(frame, image_id=image_id)
        coarse_num_boxes_all = int(len(coarse_preds))

        if self.cfg.seed_category_ids is None:
            coarse_num_boxes_bird = int(len(coarse_preds))
        else:
            allow = set(int(x) for x in self.cfg.seed_category_ids)
            coarse_num_boxes_bird = int(sum(1 for p in coarse_preds if int(p.get("category_id", -1)) in allow))

        if coarse_num_boxes_all == 0:
            self.no_det_streak_all += 1
        else:
            self.no_det_streak_all = 0

        if coarse_num_boxes_bird == 0:
            self.no_det_streak_bird += 1
        else:
            self.no_det_streak_bird = 0

        escape_soft_raw = bool(
            (coarse_num_boxes_bird == 0)
            and (not low_motion)
            and (empty_change or (self.no_det_streak_bird >= int(self.cfg.escape_soft_streak_thr)))
        )

        if escape_soft_raw:
            self.escape_soft_streak += 1
        else:
            self.escape_soft_streak = 0

        escape_hard = bool(self.no_det_streak_all >= int(self.cfg.escape_hard_streak))

        strong_skip = bool(
            (coarse_num_boxes_all == 0)
            and low_motion
            and (self.no_det_streak_all >= int(self.cfg.strong_skip_streak))
        )

        entered_full = False
        exited_full = False
        forced_subset_after_full = False

        mode = "skip"
        do_sahi = False
        tiles_selected: List[int] = []
        subset_success = None
        new_init_success = None
        existing_track_continued = None
        sel_dbg: Dict[str, Any] = {}

        # Full decision with TTL
        if escape_hard:
            if self.prev_mode != "full":
                entered_full = True
            self.full_ttl_left = int(self.cfg.full_ttl)
            mode = "full"
            do_sahi = True
            tiles_selected = list(range(len(tiles)))
            if entered_full:
                self.subset_fail_streak = 0

        elif self.full_ttl_left > 0:
            if self.prev_mode != "full":
                entered_full = True
                self.subset_fail_streak = 0
            mode = "full"
            do_sahi = True
            tiles_selected = list(range(len(tiles)))
            self.full_ttl_left -= 1

        elif self.subset_fail_streak >= int(self.cfg.subset_fail_full_streak):
            entered_full = True
            self.subset_fail_streak = 0
            self.full_ttl_left = max(0, int(self.cfg.full_ttl) - 1)
            mode = "full"
            do_sahi = True
            tiles_selected = list(range(len(tiles)))

        else:
            if self.prev_mode == "full" and (not strong_skip):
                forced_subset_after_full = True
                mode = "subset"
                do_sahi = True
            else:
                mode = self.ctrl.decide_mode(
                    t=int(self.t),
                    escape_soft_streak=int(self.escape_soft_streak),
                    force_subset=False,
                )
                do_sahi = (mode == "subset")

            if do_sahi:
                diff_scores, global_diff = self.diff.tile_scores(frame, tiles)
                seed_empty = (coarse_num_boxes_bird == 0)
                tiles_selected, sel_dbg = self.selector.select_subset_tiles(
                    coarse_preds=coarse_preds,
                    tiles=tiles,
                    diff_scores=diff_scores,
                    seed_empty=bool(seed_empty),
                    no_det_streak_all=int(self.no_det_streak_all),
                    risk_high=False,
                )
                sel_dbg["global_diff"] = float(global_diff)

        if self.prev_mode == "full" and mode != "full":
            exited_full = True

        final_preds = coarse_preds
        if do_sahi:
            final_preds = self._run_sahi_tiles(frame, image_id, tiles_selected, float(score_threshold))

        # Subset success / fail update
        if mode == "subset" and do_sahi and len(tiles_selected) > 0:
            cur_box, cur_score = self._best_bird_pred(final_preds)
            new_init_success = bool(
                (cur_box is not None) and (float(cur_score) >= float(self.cfg.subset_success_score_thr)))
            existing_track_continued = bool(self._existing_track_continued(cur_box))

            if bool(int(self.cfg.track_aware_success)):
                subset_success = bool(new_init_success or existing_track_continued)
            else:
                subset_success = bool(new_init_success)

            if subset_success:
                self.subset_fail_streak = 0
            else:
                self.subset_fail_streak += 1

        # Track state update (post)
        best_box, best_score = self._best_bird_pred(final_preds)
        if best_box is not None and float(best_score) >= float(self.cfg.subset_success_score_thr):
            self.track_active = True
            self.last_track_box = best_box
        else:
            self.track_active = False
            self.last_track_box = None

        self.prev_mode = mode

        log = {
            "image_id": int(image_id),
            "t": int(self.t),
            "mode": mode,
            "do_sahi": bool(do_sahi),
            "tiles_total": int(len(tiles)),
            "tiles_selected": int(len(tiles_selected)) if do_sahi else 0,
            "effective_sahi_cost": float(len(tiles_selected)) / float(max(1, len(tiles))) if do_sahi else 0.0,
            "entered_full": bool(entered_full),
            "exited_full": bool(exited_full),
            "forced_subset_after_full": bool(forced_subset_after_full),
            "full_ttl": int(self.cfg.full_ttl),
            "full_ttl_left": int(self.full_ttl_left),
            "escape_soft_raw": bool(escape_soft_raw),
            "escape_soft_streak": int(self.escape_soft_streak),
            "escape_hard": bool(escape_hard),
            "subset_fail_streak": int(self.subset_fail_streak),
            "subset_success": subset_success,
            "new_track_init_success": new_init_success,
            "existing_track_continued": existing_track_continued,
            "track_active": bool(self.track_active),
            "strong_skip": bool(strong_skip),
            "global_motion_raw": float(global_motion_raw),
            "gm_z": float(gm_z),
            "sel_dbg": sel_dbg,
        }
        self.logs.append(log)

        self.t += 1
        return final_preds, log
