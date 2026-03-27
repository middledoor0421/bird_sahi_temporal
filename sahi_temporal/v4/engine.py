# sahi_temporal/v4/engine.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sahi_base.tiler import compute_slice_grid
from sahi_base.merger import merge_boxes

try:
    from sahi_base.asahi_merger import merge_boxes_diou
except Exception:
    merge_boxes_diou = None

from sahi_temporal.v3.coarse_detector import CoarseDetector
from sahi_temporal.v3.tracker import SimpleTracker
from sahi_temporal.v3.risk import RiskEstimator, RiskConfig

from .motion_activity import MotionExtractor, MotionConfig
from .controller import SAHIControllerV4, ControllerConfigV4
from .tile_selector import TileSelectorV4, TileSelectorConfig


@dataclass
class V4Config:
    slice_height: int = 512
    slice_width: int = 512
    overlap_h: float = 0.2
    overlap_w: float = 0.2

    merge_mode: str = "vanilla"  # "vanilla" or "diou"
    nms_iou: float = 0.5

    heartbeat_max_gap: int = 10
    min_key_gap: int = 2
    risk_tau: float = 1.0
    risk_tau_full: float = 2.0

    # Relative thresholds via EMA z-score
    z_mu_alpha: float = 0.98
    z_dev_alpha: float = 0.98
    z_k_low: float = 1.0
    z_k_high: float = 1.5
    z_eps: float = 1e-3

    # Escape subset-first + escalation
    escape_soft_streak: int = 10   # N_soft: bird==0 streak to trigger subset (not low_motion)
    escape_full_streak: int = 30   # N_hard: if blind persists, escalate to full

    # Strong skip: truly empty + low-motion + long streak
    strong_skip_streak: int = 20

    # Motion extractor (tile-level)
    motion_mode: str = "lag"
    motion_lag: int = 4
    motion_accum_window: int = 5
    motion_downsample_factor: int = 4
    motion_ema_alpha: float = 0.8

    # Tile selector
    k_min: int = 2
    k_max: int = 6
    motion_topk_base: int = 2
    motion_topk_high: int = 4
    neighbor_hops: int = 1
    seed_category_ids: Optional[List[int]] = None
    seed_min_score: float = 0.0

    enable_tracker: int = 1


class TemporalSahiV4Engine:
    def __init__(self, detector, cfg: Optional[V4Config] = None, class_mapping: Optional[Dict[int, int]] = None) -> None:
        self.cfg = cfg if cfg is not None else V4Config()
        self.class_mapping = class_mapping

        self.coarse = CoarseDetector(detector=detector)
        self.tracker = SimpleTracker() if bool(int(self.cfg.enable_tracker)) else None
        self.risk = RiskEstimator(RiskConfig(risk_tau=float(self.cfg.risk_tau)))

        self.motion = MotionExtractor(MotionConfig(
            downsample_factor=int(self.cfg.motion_downsample_factor),
            ema_alpha=float(self.cfg.motion_ema_alpha),
            mode=str(self.cfg.motion_mode),
            lag=int(self.cfg.motion_lag),
            accum_window=int(self.cfg.motion_accum_window),
        ))

        self.ctrl = SAHIControllerV4(ControllerConfigV4(
            heartbeat_max_gap=int(self.cfg.heartbeat_max_gap),
            min_key_gap=int(self.cfg.min_key_gap),
            risk_tau=float(self.cfg.risk_tau),
            risk_tau_full=float(self.cfg.risk_tau_full),
        ))

        self.selector = TileSelectorV4(TileSelectorConfig(
            k_min=int(self.cfg.k_min),
            k_max=int(self.cfg.k_max),
            motion_topk_base=int(self.cfg.motion_topk_base),
            motion_topk_high=int(self.cfg.motion_topk_high),
            neighbor_hops=int(self.cfg.neighbor_hops),
            seed_category_ids=self.cfg.seed_category_ids,
            seed_min_score=float(self.cfg.seed_min_score),
        ))

        self.tiles: Optional[List[Dict[str, int]]] = None
        self.t = 0

        self.no_det_streak_all = 0
        self.no_det_streak_bird = 0

        # EMA z-score state
        self.gm_mu = 0.0
        self.gm_dev = 1.0

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

    def _run_sahi_tiles(self, frame: np.ndarray, image_id: int, tile_ids: List[int], score_threshold: float) -> List[Dict]:
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
            out.append({"image_id": int(image_id), "category_id": int(cid), "bbox": [x1, y1, bw, bh], "score": float(m_scores[i].item())})
        return out

    def process_frame(self, frame: np.ndarray, image_id: int, score_threshold: float = 0.0) -> Tuple[List[Dict], Dict]:
        tiles = self._ensure_tiles(frame)

        motion_scores, global_motion_raw = self.motion.compute(frame, tiles)
        global_motion, gm_mu, gm_z = self._update_zscore(global_motion_raw)

        low_motion = bool(gm_z < -float(self.cfg.z_k_low))
        empty_change = bool(gm_z > float(self.cfg.z_k_high))

        coarse_preds, coarse_stats = self.coarse.predict(frame, image_id=image_id)
        coarse_num_boxes_all = int(len(coarse_preds))

        if self.cfg.seed_category_ids is None:
            coarse_num_boxes_bird = coarse_num_boxes_all
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

        tracking_stats = self.tracker.update(coarse_preds) if self.tracker is not None else None
        risk_out = self.risk.score(frame, coarse_stats, tracking_stats)
        tracking_fail = bool(risk_out["flags"]["tracking_fail"])

        # Escape subset-first: broaden to avoid "escape_count=0" collapse
        # escape_soft if bird blind and (change OR long bird-blind streak), and not low_motion
        escape_soft = bool(
            (coarse_num_boxes_bird == 0)
            and (not low_motion)
            and (empty_change or (self.no_det_streak_bird >= int(self.cfg.escape_soft_streak)))
        )

        # Escalate to full only if it persists
        escape_hard = bool(escape_soft and (self.no_det_streak_all >= int(self.cfg.escape_full_streak)))

        # Strong skip: truly empty + low motion + long empty streak
        strong_skip = bool(
            (coarse_num_boxes_bird == 0)
            and (coarse_num_boxes_all == 0)
            and low_motion
            and (self.no_det_streak_all >= int(self.cfg.strong_skip_streak))
        )

        # Anchor decision computed from controller state
        is_anchor = bool((int(self.t) - int(self.ctrl.last_keyframe)) >= int(self.cfg.heartbeat_max_gap))

        decision = self.ctrl.decide(
            t=int(self.t),
            risk_score=float(risk_out["risk_score"]),
            tracking_fail=tracking_fail,
            escape_soft=escape_soft,
            escape_hard=escape_hard,
            strong_skip=strong_skip,
            is_anchor=is_anchor,
        )

        do_sahi = bool(decision["do_sahi"])
        mode = str(decision["mode"])
        risk_high = bool(decision["risk_high"])

        tile_ids: List[int] = []
        final_preds = coarse_preds

        if do_sahi:
            tile_ids = self.selector.select(
                mode=mode,
                coarse_preds=coarse_preds,
                tiles=tiles,
                motion_scores=motion_scores,
                risk_high=risk_high,
            )
            if len(tile_ids) == 0:
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
            "risk_score": float(risk_out["risk_score"]),
            "flags": dict(risk_out["flags"]),
            "global_motion_raw": float(global_motion_raw),
            "global_motion": float(global_motion),
            "gm_mu": float(gm_mu),
            "gm_z": float(gm_z),
            "low_motion": bool(low_motion),
            "empty_change": bool(empty_change),
            "coarse_num_boxes_all": int(coarse_num_boxes_all),
            "coarse_num_boxes_bird": int(coarse_num_boxes_bird),
            "no_det_streak_all": int(self.no_det_streak_all),
            "no_det_streak_bird": int(self.no_det_streak_bird),
            "escape_soft": bool(escape_soft),
            "escape_hard": bool(escape_hard),
            "strong_skip": bool(strong_skip),
            "tiles_total": int(len(tiles)),
            "tiles_selected": int(len(tile_ids)) if do_sahi else 0,
            "risk_high": bool(risk_high),
            "is_anchor": bool(is_anchor),
        }
        self.logs.append(log)

        self.t += 1
        return final_preds, log
