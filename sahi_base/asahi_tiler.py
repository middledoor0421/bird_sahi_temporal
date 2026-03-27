# sahi_base/asahi_tiler.py
# ASAHI adaptive tiling (fixed number of patches 6 or 12) based on LS rule.
# Python 3.9 compatible. Comments in English only.

from typing import Dict, List, Tuple
import math


def compute_ls_threshold(restrict_size: int, overlap_ratio: float) -> float:
    """
    LS = restrict_size * (4 - 3*l) + 1
    where l is overlap ratio.
    """
    l = float(overlap_ratio)
    return float(restrict_size) * (4.0 - 3.0 * l) + 1.0


def choose_grid(W: int, H: int, target_patches: int) -> Tuple[int, int]:
    """
    Choose (a, b) where a*b = target_patches, and a aligns with width axis.
    Paper uses 6 or 12 patches; use (3,2) or (4,3) depending on aspect ratio.
    """
    if target_patches == 6:
        return (3, 2) if W >= H else (2, 3)
    if target_patches == 12:
        return (4, 3) if W >= H else (3, 4)
    # Fallback (not expected)
    a = int(round(math.sqrt(float(target_patches) * float(W) / max(1.0, float(H)))))
    a = max(1, min(a, target_patches))
    b = int(math.ceil(float(target_patches) / float(a)))
    return a, b


def compute_asahi_patch_size(W: int, H: int, a: int, b: int, overlap_ratio: float) -> int:
    """
    Compute square patch size p to cover both W and H with overlap ratio l.

    With a patches on width axis and overlap l, stride = p*(1-l),
    coverage along width is: p + (a-1)*p*(1-l) = p*(a - (a-1)*l)
    So p >= W / (a - (a-1)*l). Similarly for height with b.
    We use p = ceil(max(p_w, p_h)).
    """
    l = float(overlap_ratio)
    denom_w = float(a) - float(a - 1) * l
    denom_h = float(b) - float(b - 1) * l
    denom_w = max(1e-6, denom_w)
    denom_h = max(1e-6, denom_h)

    p_w = float(W) / denom_w
    p_h = float(H) / denom_h
    p = int(math.ceil(max(p_w, p_h)))
    return max(1, p)


def compute_asahi_grid(
    height: int,
    width: int,
    overlap_ratio: float = 0.15,
    restrict_size: int = 512,
) -> Dict[str, object]:
    """
    Returns:
      dict with:
        - tiles: List[{"x","y","w","h"}]
        - target_patches: 6 or 12
        - grid: (a, b)
        - patch_size: p (square)
        - ls: LS threshold
    """
    H = int(height)
    W = int(width)
    l = float(overlap_ratio)

    ls = compute_ls_threshold(restrict_size=int(restrict_size), overlap_ratio=l)
    target_patches = 12 if (max(W, H) > ls) else 6

    a, b = choose_grid(W=W, H=H, target_patches=target_patches)
    p = compute_asahi_patch_size(W=W, H=H, a=a, b=b, overlap_ratio=l)

    stride = max(1, int(round(float(p) * (1.0 - l))))

    # Generate top-left positions with "last tile aligned to boundary" policy
    xs: List[int] = []
    ys: List[int] = []

    if p >= W:
        xs = [0] * a
    else:
        for i in range(a):
            x = i * stride
            x = min(x, W - p)
            xs.append(int(x))

    if p >= H:
        ys = [0] * b
    else:
        for j in range(b):
            y = j * stride
            y = min(y, H - p)
            ys.append(int(y))

    tiles: List[Dict[str, int]] = []
    for j in range(b):
        for i in range(a):
            tiles.append({"x": xs[i], "y": ys[j], "w": p, "h": p})

    return {
        "tiles": tiles,
        "target_patches": int(target_patches),
        "grid": (int(a), int(b)),
        "patch_size": int(p),
        "ls": float(ls),
    }
