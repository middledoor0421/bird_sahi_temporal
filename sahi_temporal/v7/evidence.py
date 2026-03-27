# sahi_temporal/v7/evidence.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from sahi_temporal.v4.motion_activity import MotionConfig, MotionExtractor
from sahi_temporal.v5.diff_score import DiffScoreConfig, DiffTileScorer

from .schemas import DetEvidence, DiffEvidence, Evidence, MetaEvidence, MotionEvidence, TrackEvidence
from .tracker import TrackSummary


@dataclass
class EvidenceBuilderConfig:
    # Det evidence
    target_category_ids: Optional[List[int]] = None

    # Motion normalization (z-score)
    z_mu_alpha: float = 0.98
    z_dev_alpha: float = 0.98
    z_eps: float = 1e-3

    # Motion extractor
    motion_mode: str = "lag"
    motion_lag: int = 4
    motion_accum_window: int = 5
    motion_downsample_factor: int = 4
    motion_ema_alpha: float = 0.8

    # Diff
    diff_mode: str = "ema_bg"
    diff_downsample: int = 4
    diff_bg_alpha: float = 0.10
    diff_tile_ema: float = 0.0


class EvidenceBuilderV7:
    def __init__(self, cfg: Optional[EvidenceBuilderConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else EvidenceBuilderConfig()

        # Keep configs to allow re-init on clip boundary
        self._motion_cfg = MotionConfig(
            downsample_factor=int(self.cfg.motion_downsample_factor),
            ema_alpha=float(self.cfg.motion_ema_alpha),
            mode=str(self.cfg.motion_mode),
            lag=int(self.cfg.motion_lag),
            accum_window=int(self.cfg.motion_accum_window),
        )
        self.motion = MotionExtractor(self._motion_cfg)

        self._diff_cfg = DiffScoreConfig(
            mode=str(self.cfg.diff_mode),
            downsample_factor=int(self.cfg.diff_downsample),
            bg_alpha=float(self.cfg.diff_bg_alpha),
            tile_score_ema=float(self.cfg.diff_tile_ema),
        )
        self.diff = DiffTileScorer(self._diff_cfg)

        self._t = 0

        # Det streak
        self._no_det_streak = 0

        # Motion z-score state
        self._gm_mu = 0.0
        self._gm_dev = 1.0

        # Cached tiles
        self._tiles = None

    def set_tiles(self, tiles: List[Dict[str, int]]) -> None:
        self._tiles = tiles

    def clear_tiles(self) -> None:
        # Clear cached tiles at clip boundary so they will be re-set for new resolution.
        self._tiles = None

    def reset_for_new_video(self) -> None:
        """Reset temporal states at clip boundary.

        This prevents cross-clip contamination (motion buffers, diff background, streaks, z-score).
        """
        self._t = 0
        self._no_det_streak = 0
        self._gm_mu = 0.0
        self._gm_dev = 1.0

        # Reset motion extractor
        if hasattr(self.motion, "reset"):
            try:
                self.motion.reset()
            except Exception:
                self.motion = MotionExtractor(self._motion_cfg)
        else:
            self.motion = MotionExtractor(self._motion_cfg)

        # Reset diff scorer
        if hasattr(self.diff, "reset"):
            try:
                self.diff.reset()
            except Exception:
                self.diff = DiffTileScorer(self._diff_cfg)
        else:
            self.diff = DiffTileScorer(self._diff_cfg)

    def _filter_preds(self, preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.cfg.target_category_ids is None:
            return preds
        allow = set(int(x) for x in self.cfg.target_category_ids)
        out: List[Dict[str, Any]] = []
        for p in preds:
            if int(p.get("category_id", -1)) in allow:
                out.append(p)
        return out

    def _update_motion_z(self, global_motion: float) -> float:
        mu_a = float(self.cfg.z_mu_alpha)
        dev_a = float(self.cfg.z_dev_alpha)
        eps = float(self.cfg.z_eps)

        gm = float(global_motion)
        self._gm_mu = mu_a * float(self._gm_mu) + (1.0 - mu_a) * gm
        dev_raw = abs(gm - float(self._gm_mu))
        self._gm_dev = dev_a * float(self._gm_dev) + (1.0 - dev_a) * dev_raw
        z = (gm - float(self._gm_mu)) / (float(self._gm_dev) + eps)
        return float(z)

    @staticmethod
    def _pctl(x: np.ndarray, q: float) -> float:
        if x is None or x.size == 0:
            return 0.0
        return float(np.percentile(x.astype(np.float32), q))

    def update(
        self,
        frame: np.ndarray,
        det_out: List[Dict[str, Any]],
        track_state: TrackSummary,
        prev_mode: Optional[str] = None,
    ) -> Evidence:
        if self._tiles is None:
            raise RuntimeError("EvidenceBuilderV7 requires tiles to be set via set_tiles().")

        tiles = self._tiles

        # Det evidence
        det_f = self._filter_preds(det_out)
        has_det = bool(len(det_f) > 0)
        if has_det:
            confs = np.asarray([float(p.get("score", 0.0)) for p in det_f], dtype=np.float32)
            count = int(confs.size)
            mean_conf = float(confs.mean()) if confs.size > 0 else 0.0
            p90_conf = self._pctl(confs, 90.0)
            self._no_det_streak = 0
        else:
            count = 0
            mean_conf = 0.0
            p90_conf = 0.0
            self._no_det_streak += 1

        det_ev = DetEvidence(
            has_det=bool(has_det),
            count=int(count),
            mean_conf=float(mean_conf),
            p90_conf=float(p90_conf),
            no_det_streak=int(self._no_det_streak),
        )

        # Track evidence (already summarized)
        track_ev = TrackEvidence(
            num_active=int(track_state.num_active),
            max_miss_streak=int(track_state.max_miss_streak),
            instability=float(track_state.instability),
            recent_lost=bool(track_state.recent_lost),
        )

        # Motion / diff per tile
        motion_scores, global_motion = self.motion.compute(frame, tiles)
        global_z = self._update_motion_z(global_motion)
        local_peak = float(motion_scores.max()) if motion_scores is not None and motion_scores.size > 0 else 0.0

        diff_scores, _global_diff = self.diff.tile_scores(frame, tiles)
        p95 = self._pctl(diff_scores, 95.0)
        p99 = self._pctl(diff_scores, 99.0)
        std = float(diff_scores.std()) if diff_scores is not None and diff_scores.size > 0 else 0.0

        # top1_gap = top1 - top2
        if diff_scores is not None and diff_scores.size >= 2:
            order = np.argsort(-diff_scores)
            top1 = float(diff_scores[int(order[0])])
            top2 = float(diff_scores[int(order[1])])
            top1_gap = float(top1 - top2)
        elif diff_scores is not None and diff_scores.size == 1:
            top1_gap = float(diff_scores[0])
        else:
            top1_gap = 0.0

        motion_ev = MotionEvidence(global_z=float(global_z), local_peak=float(local_peak))
        diff_ev = DiffEvidence(p95=float(p95), p99=float(p99), std=float(std), top1_gap=float(top1_gap))

        meta = MetaEvidence(prev_mode=prev_mode)

        ev = Evidence(
            t=int(self._t),
            det=det_ev,
            track=track_ev,
            motion=motion_ev,
            diff=diff_ev,
            meta=meta,
            raw={
                "motion_scores": motion_scores,
                "diff_scores": diff_scores,
            },
        )

        self._t += 1
        return ev
