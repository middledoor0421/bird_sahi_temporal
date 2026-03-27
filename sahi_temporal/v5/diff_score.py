# sahi_temporal/v5/diff_score.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DiffScoreConfig:
    mode: str = "frame"  # "frame" or "ema_bg"
    downsample_factor: int = 4
    bg_alpha: float = 0.05          # used for ema_bg
    tile_score_ema: float = 0.0     # 0 disables smoothing, else EMA on tile_scores


class DiffTileScorer:
    def __init__(self, cfg: Optional[DiffScoreConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else DiffScoreConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.bg: Optional[np.ndarray] = None
        self.tile_scores_ema: Optional[np.ndarray] = None

    def _to_gray_downsampled(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            gray = frame.astype(np.float32)
        else:
            r = frame[..., 0].astype(np.float32)
            g = frame[..., 1].astype(np.float32)
            b = frame[..., 2].astype(np.float32)
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        s = max(1, int(self.cfg.downsample_factor))
        if s > 1:
            h = (gray.shape[0] // s) * s
            w = (gray.shape[1] // s) * s
            gray = gray[:h:s, :w:s]
        return gray.astype(np.float32)

    def _compute_diff_map(self, gray: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.mode).lower().strip()
        if mode == "frame":
            if self.prev_gray is None or self.prev_gray.shape != gray.shape:
                self.prev_gray = gray
                return np.zeros_like(gray, dtype=np.float32)
            diff = np.abs(gray - self.prev_gray)
            self.prev_gray = gray
            return diff.astype(np.float32)

        if mode == "ema_bg":
            a = float(self.cfg.bg_alpha)
            if self.bg is None or self.bg.shape != gray.shape:
                self.bg = gray.copy()
                return np.zeros_like(gray, dtype=np.float32)
            # Update background then compute residual
            self.bg = (1.0 - a) * self.bg + a * gray
            diff = np.abs(gray - self.bg)
            return diff.astype(np.float32)

        raise ValueError("Unknown diff mode: {}".format(self.cfg.mode))

    def tile_scores(self, frame: np.ndarray, tiles: List[Dict[str, int]]) -> Tuple[np.ndarray, float]:
        gray = self._to_gray_downsampled(frame)
        diff = self._compute_diff_map(gray)

        # Global diff score (for logging)
        global_diff = float(diff.mean()) if diff.size > 0 else 0.0

        s = max(1, int(self.cfg.downsample_factor))
        Hds, Wds = diff.shape[0], diff.shape[1]

        scores = np.zeros(len(tiles), dtype=np.float32)
        for i, t in enumerate(tiles):
            x1 = int(t["x"]) // s
            y1 = int(t["y"]) // s
            x2 = (int(t["x"]) + int(t["w"])) // s
            y2 = (int(t["y"]) + int(t["h"])) // s

            x1 = max(0, min(x1, Wds))
            x2 = max(0, min(x2, Wds))
            y1 = max(0, min(y1, Hds))
            y2 = max(0, min(y2, Hds))

            if x2 <= x1 or y2 <= y1:
                scores[i] = 0.0
            else:
                patch = diff[y1:y2, x1:x2]
                scores[i] = float(patch.mean())

        # Optional smoothing on tile scores
        ema = float(self.cfg.tile_score_ema)
        if ema > 0.0:
            if self.tile_scores_ema is None or self.tile_scores_ema.shape != scores.shape:
                self.tile_scores_ema = scores.copy()
            else:
                self.tile_scores_ema = ema * self.tile_scores_ema + (1.0 - ema) * scores
            scores = self.tile_scores_ema.copy()

        return scores.astype(np.float32), global_diff
