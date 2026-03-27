# sahi_base/tiler.py
# SAHI-style image tiling utilities.
# Python 3.9 compatible. Comments in English only.

from typing import List, Dict, Tuple
import numpy as np


def compute_slice_grid(
    height: int,
    width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> List[Dict[str, int]]:
    """
    Compute SAHI-style slice grid over an image.

    Returns:
        tiles: list of dicts each with keys {"x", "y", "w", "h}
    """
    tiles: List[Dict[str, int]] = []

    if slice_height >= height:
        y_starts = [0]
    else:
        step_h = int(slice_height * (1.0 - overlap_height_ratio))
        step_h = max(1, step_h)
        y_starts = list(range(0, max(1, height - slice_height + 1), step_h))
        if y_starts[-1] + slice_height < height:
            y_starts.append(height - slice_height)

    if slice_width >= width:
        x_starts = [0]
    else:
        step_w = int(slice_width * (1.0 - overlap_width_ratio))
        step_w = max(1, step_w)
        x_starts = list(range(0, max(1, width - slice_width + 1), step_w))
        if x_starts[-1] + slice_width < width:
            x_starts.append(width - slice_width)

    for y in y_starts:
        for x in x_starts:
            tw = min(slice_width, width - x)
            th = min(slice_height, height - y)
            tiles.append({"x": int(x), "y": int(y), "w": int(tw), "h": int(th)})

    return tiles


def count_slices(
    height: int,
    width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> int:
    """
    Return number of slices for given geometry and slice config.
    """
    return len(
        compute_slice_grid(
            height=height,
            width=width,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
    )


def slice_image(
    image: np.ndarray,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> List[Tuple[np.ndarray, Dict[str, int]]]:
    """
    Actually slice the image into tiles.

    Args:
        image: np.ndarray (H, W, 3) in BGR or RGB
    Returns:
        list of (tile_image, tile_info) where tile_info has keys {"x","y","w","h"}
    """
    height, width = image.shape[:2]
    tiles = compute_slice_grid(
        height=height,
        width=width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    outputs: List[Tuple[np.ndarray, Dict[str, int]]] = []
    for info in tiles:
        x = info["x"]
        y = info["y"]
        w = info["w"]
        h = info["h"]
        tile = image[y : y + h, x : x + w].copy()
        outputs.append((tile, info))

    return outputs
