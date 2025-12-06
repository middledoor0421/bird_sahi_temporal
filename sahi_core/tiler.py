# sahi_core/tiler.py
# Python 3.9 compatible. Comments in English only.

from typing import List, Dict, Tuple
import numpy as np


TileInfo = Dict[str, int]  # {"x": int, "y": int, "w": int, "h": int}


def generate_tiles(
    img: np.ndarray,
    tile_size: int,
    overlap: float
) -> Tuple[List[np.ndarray], List[TileInfo]]:
    """
    Slice image into overlapping tiles.

    Args:
        img: BGR image (H, W, 3)
        tile_size: side length of square tile
        overlap: overlap ratio in [0, 1). step = tile_size * (1 - overlap)

    Returns:
        tiles: list of cropped tiles (BGR images)
        infos: list of tile info dicts {"x","y","w","h"}
    """
    h, w = img.shape[:2]
    ts = int(tile_size)
    ov = max(0.0, min(overlap, 0.9))
    step = max(1, int(ts * (1.0 - ov)))

    tiles: List[np.ndarray] = []
    infos: List[TileInfo] = []

    xs = list(range(0, max(1, w - ts + 1), step)) or [0]
    ys = list(range(0, max(1, h - ts + 1), step)) or [0]

    for y in ys:
        for x in xs:
            tw = min(ts, w - x)
            th = min(ts, h - y)
            tile = img[y:y + th, x:x + tw].copy()
            tiles.append(tile)
            infos.append({"x": x, "y": y, "w": tw, "h": th})

    return tiles, infos
