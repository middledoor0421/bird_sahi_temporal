# sahi_temporal/v3/risk.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .coarse_detector import CoarseStats
from .tracker import TrackerStats


@dataclass
class RiskSignals:
    global_motion: float
    blur_score: float
    coarse_num_boxes: int
    coarse_num_small: int
    coarse_max_score: float
    coarse_mean_score: float
    coarse_score_margin: float
    tracking_fail: bool


@dataclass
class RiskConfig:
    # weights for a simple linear risk score
    alpha_motion: float = 1.0
    beta_history: float = 1.0
    gamma_uncertainty: float = 1.0

    # thresholds (tunable)
    low_motion_thr: float = 1.0
    low_margin_thr: float = 0.05
    low_maxscore_thr: float = 0.15
    blur_low_thr: float = 20.0  # lower Laplacian variance => blurrier

    risk_tau: float = 1.0


class RiskEstimator:
    """
    Outputs risk score r_t and boolean risk flags.
    Detection-independent signals (motion/blur) are always available.
    """

    def __init__(self, cfg: Optional[RiskConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else RiskConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.global_motion_ema: float = 0.0

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float32)
        r = frame[..., 0].astype(np.float32)
        g = frame[..., 1].astype(np.float32)
        b = frame[..., 2].astype(np.float32)
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def compute_global_motion(self, frame: np.ndarray, ema_alpha: float = 0.8) -> float:
        gray = self._to_gray(frame)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return 0.0
        diff = np.abs(gray - self.prev_gray)
        raw = float(diff.mean())
        self.global_motion_ema = float(ema_alpha) * float(self.global_motion_ema) + (1.0 - float(ema_alpha)) * raw
        self.prev_gray = gray
        return float(self.global_motion_ema)

    def compute_blur_score(self, frame: np.ndarray) -> float:
        gray = self._to_gray(frame)
        # Laplacian variance (simple blur proxy)
        lap = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
        )
        return float(lap.var())

    def score(
        self,
        frame: np.ndarray,
        coarse: CoarseStats,
        tracking: Optional[TrackerStats],
        motion_ema_alpha: float = 0.8,
    ) -> Dict:
        gm = self.compute_global_motion(frame, ema_alpha=motion_ema_alpha)
        blur = self.compute_blur_score(frame)

        tracking_fail = bool(tracking.tracking_fail) if tracking is not None else False

        # Uncertainty proxy: low max score or low margin while boxes exist
        uncertain = (coarse.num_boxes > 0) and (
            (coarse.max_score < self.cfg.low_maxscore_thr) or (coarse.score_margin < self.cfg.low_margin_thr)
        )

        low_motion = gm < self.cfg.low_motion_thr
        blur_susp = blur < self.cfg.blur_low_thr

        # Only treat low-motion as risky when combined with other suspicious signals
        motion_term = 1.0 if (low_motion and (uncertain or tracking_fail or (coarse.num_boxes == 0))) else 0.0
        hist_term = 1.0 if tracking_fail else 0.0
        uncert_term = 1.0 if uncertain else 0.0

        r = 0.0
        r += self.cfg.alpha_motion * motion_term
        r += self.cfg.beta_history * hist_term
        r += self.cfg.gamma_uncertainty * uncert_term


        do_sahi_risk = (r >= self.cfg.risk_tau)

        return {
            "risk_score": float(r),
            "do_sahi_risk": bool(do_sahi_risk),
            "signals": RiskSignals(
                global_motion=float(gm),
                blur_score=float(blur),
                coarse_num_boxes=int(coarse.num_boxes),
                coarse_num_small=int(coarse.num_small_boxes),
                coarse_max_score=float(coarse.max_score),
                coarse_mean_score=float(coarse.mean_score),
                coarse_score_margin=float(coarse.score_margin),
                tracking_fail=bool(tracking_fail),
            ),
            "flags": {
                "low_motion": bool(low_motion),
                "blur_susp": bool(blur_susp),
                "uncertain": bool(uncertain),
                "tracking_fail": bool(tracking_fail),
            },
        }
