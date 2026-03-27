# sahi_temporal/v4/motion_activity.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np


@dataclass
class MotionConfig:
    downsample_factor: int = 4
    ema_alpha: float = 0.8
    mode: str = "lag"          # "lag" or "accum"
    lag: int = 4
    accum_window: int = 5


class MotionExtractor:
    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self.prev_global_motion: float = 0.0
        self.gray_buf: List[np.ndarray] = []

    def reset(self) -> None:
        # Reset temporal buffers at clip boundary.
        self.prev_global_motion = 0.0
        self.gray_buf = []

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

    def compute(self, frame: np.ndarray, tiles: List[Dict[str, int]]) -> Tuple[np.ndarray, float]:
        gray = self._to_gray_downsampled(frame)

        # Safety: if resolution changes within a clip, reset buffer to avoid shape mismatch.
        if len(self.gray_buf) > 0 and self.gray_buf[-1].shape != gray.shape:
            self.gray_buf = [gray]
            self.prev_global_motion = 0.0
            return np.zeros(len(tiles), dtype=np.float32), 0.0

        self.gray_buf.append(gray)

        max_keep = max(int(self.cfg.lag) + 1, int(self.cfg.accum_window) + 1, 2)
        if len(self.gray_buf) > max_keep:
            self.gray_buf = self.gray_buf[-max_keep:]

        if len(self.gray_buf) < 2:
            return np.zeros(len(tiles), dtype=np.float32), 0.0

        mode = str(self.cfg.mode).lower().strip()
        if mode == "lag":
            lag = max(1, int(self.cfg.lag))
            ref = self.gray_buf[0] if len(self.gray_buf) <= lag else self.gray_buf[-(lag + 1)]
            if ref.shape != self.gray_buf[-1].shape:
                # Another safety check.
                self.gray_buf = [self.gray_buf[-1]]
                self.prev_global_motion = 0.0
                return np.zeros(len(tiles), dtype=np.float32), 0.0
            diff = np.abs(self.gray_buf[-1] - ref)
        elif mode == "accum":
            k = max(2, int(self.cfg.accum_window))
            buf = self.gray_buf[-k:] if len(self.gray_buf) >= k else self.gray_buf
            base = buf[-1]
            diff = np.zeros_like(base, dtype=np.float32)
            for i in range(1, len(buf)):
                if buf[i].shape != buf[i - 1].shape:
                    self.gray_buf = [buf[i]]
                    self.prev_global_motion = 0.0
                    return np.zeros(len(tiles), dtype=np.float32), 0.0
                diff += np.abs(buf[i] - buf[i - 1])
            diff /= float(max(1, len(buf) - 1))
        else:
            raise ValueError("Unknown motion mode: {}".format(self.cfg.mode))

        raw_global = float(diff.mean())
        g = float(self.cfg.ema_alpha) * float(self.prev_global_motion) + (1.0 - float(self.cfg.ema_alpha)) * raw_global
        self.prev_global_motion = g

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

        return scores, g
