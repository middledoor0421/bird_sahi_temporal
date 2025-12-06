# sahi_ext/motion_gater.py
# Temporal-motion-based tile gating for SAHI.
# Python 3.9 compatible. Comments in English only.

from typing import List, Dict, Optional
import cv2
import numpy as np


class MotionGater:
    """
    Maintain temporal motion history per tile and return indices of tiles that
    are considered "moving" for current frame.

    Usage:
        mg = MotionGater(diff_th, min_area, min_stable_frames)
        moving_idxs = mg.update(frame_bgr, tile_infos)

    tile_infos: list of {"x","y","w","h"} (same order/size each frame)
    """

    def __init__(self, diff_th: int, min_area: int, min_stable_frames: int = 1) -> None:
        self.diff_th = int(diff_th)
        self.min_area = int(min_area)
        self.min_stable_frames = max(1, int(min_stable_frames))

        self.prev_gray: Optional[np.ndarray] = None
        self.stable_counts: Optional[List[int]] = None
        self.num_tiles: int = 0

    def _ensure_state(self, num_tiles: int) -> None:
        if self.stable_counts is None or self.num_tiles != num_tiles:
            self.num_tiles = num_tiles
            self.stable_counts = [0 for _ in range(num_tiles)]

    def update(self, frame_bgr: np.ndarray, tile_infos: List[Dict]) -> List[int]:
        """
        Args:
            frame_bgr: current BGR frame (H, W, 3)
            tile_infos: list of tile dicts, each {"x","y","w","h"}

        Returns:
            moving_idxs: list of tile indices that are stable-moving.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        num_tiles = len(tile_infos)
        self._ensure_state(num_tiles)

        moving_idxs: List[int] = []

        # first frame: no motion
        if self.prev_gray is None:
            self.prev_gray = gray
            return moving_idxs

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        # threshold to get motion mask
        mask = (diff > self.diff_th).astype(np.uint8) * 255
        mask = cv2.medianBlur(mask, 3)

        for idx, info in enumerate(tile_infos):
            x, y, tw, th = info["x"], info["y"], info["w"], info["h"]
            # clamp in case of border tiles
            x2 = min(w, x + tw)
            y2 = min(h, y + th)
            roi = mask[y:y2, x:x2]
            area = int(cv2.countNonZero(roi))
            if area >= self.min_area:
                self.stable_counts[idx] += 1
            else:
                self.stable_counts[idx] = 0

            if self.stable_counts[idx] >= self.min_stable_frames:
                moving_idxs.append(idx)

        return moving_idxs
