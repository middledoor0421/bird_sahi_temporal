# sahi_temporal/v6/tiles_explore.py
# Python 3.9 compatible. Comments in English only.

from typing import Dict, List, Optional

import numpy as np


class ExploreMemory:
    """TTL-based memory for explore tiles.

    This is used only in the hard region to stabilize exploration across frames.
    """

    def __init__(self) -> None:
        self.ttl: Dict[int, int] = {}

    def step(self) -> None:
        expired: List[int] = []
        for k in list(self.ttl.keys()):
            self.ttl[k] -= 1
            if self.ttl[k] <= 0:
                expired.append(k)
        for k in expired:
            self.ttl.pop(k, None)

    def inject(self, cand_ids: List[int]) -> List[int]:
        # Keep memory tiles in front, then add candidate ids.
        out: List[int] = []
        for k in self.ttl.keys():
            out.append(int(k))
        for x in cand_ids:
            xi = int(x)
            if xi not in out:
                out.append(xi)
        return out

    def update(self, selected: List[int], ttl: int) -> None:
        for t in selected:
            self.ttl[int(t)] = int(ttl)


def _to_np_vec(x: object, n: int) -> np.ndarray:
    if x is None:
        return np.zeros((n,), dtype=np.float32)
    if isinstance(x, np.ndarray):
        v = x.astype(np.float32).reshape(-1)
        if v.size != n:
            # If mismatch, pad/truncate.
            out = np.zeros((n,), dtype=np.float32)
            m = min(int(n), int(v.size))
            out[:m] = v[:m]
            return out
        return v
    try:
        xs = list(x)  # type: ignore
        out = np.asarray(xs, dtype=np.float32).reshape(-1)
        if out.size != n:
            pad = np.zeros((n,), dtype=np.float32)
            m = min(int(n), int(out.size))
            pad[:m] = out[:m]
            return pad
        return out
    except Exception:
        return np.zeros((n,), dtype=np.float32)


def _peakify_global(scores: np.ndarray, q: float) -> np.ndarray:
    if scores.size == 0:
        return scores
    thr = float(np.percentile(scores.astype(np.float32), float(q)))
    mask = (scores >= thr).astype(np.float32)
    # Keep only peaks + small residual to preserve ranking ties.
    return scores * mask + 0.10 * scores


def _compute_score_vec(tile_scores: List[float], mode: str) -> np.ndarray:
    v = np.asarray(tile_scores, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return v

    m = str(mode).lower().strip()
    if m == "mean":
        return v
    if m == "p95":
        return _peakify_global(v, 95.0)
    if m == "p99":
        return _peakify_global(v, 99.0)
    if m == "top1_gap":
        # Not well-defined for scalar-per-tile scores. Use p95 as a robust proxy.
        return _peakify_global(v, 95.0)
    return v


def _no_det_bucket(no_det_bucket: str) -> str:
    b = str(no_det_bucket)
    if b in ["0_1", "2_5", "6p"]:
        return b
    # Backward compat with older notation.
    if b in ["0-1", "0,1", "0_1"]:
        return "0_1"
    if b in ["2-5", "2,5", "2_5"]:
        return "2_5"
    return "6p"


def _build_row_ids(tiles: List[Dict[str, int]]):
    ys = sorted({int(t["y"]) for t in tiles})
    y_to_row = {y: i for i, y in enumerate(ys)}

    tile_to_row: Dict[int, int] = {}
    rows: List[List[int]] = [[] for _ in range(len(ys))]

    for tid, t in enumerate(tiles):
        r = y_to_row[int(t["y"])]
        tile_to_row[int(tid)] = int(r)
        rows[int(r)].append(int(tid))

    # Row-major order (left->right)
    for r in range(len(rows)):
        rows[r] = sorted(rows[r], key=lambda tid: int(tiles[int(tid)]["x"]))

    return tile_to_row, rows


def _apply_row_quota(ranked_ids: List[int], tiles: List[Dict[str, int]], k: int, quota_mode: str) -> List[int]:
    k = max(0, int(k))
    if k == 0:
        return []

    tile_to_row, rows = _build_row_ids(tiles)
    if len(rows) <= 1:
        return [int(x) for x in ranked_ids[:k]]

    top_row = 0
    bottom_row = len(rows) - 1
    mid_row = 1 if len(rows) >= 3 else bottom_row

    selected: List[int] = []
    banned = set()

    def pick_best(row_idx: int) -> Optional[int]:
        for tid in ranked_ids:
            it = int(tid)
            if it in banned:
                continue
            if int(tile_to_row.get(it, -1)) == int(row_idx):
                return it
        return None

    # Always pick one from top row
    a = pick_best(top_row)
    if a is not None:
        selected.append(int(a))
        banned.add(int(a))

    if str(quota_mode) == "top1_bottom1" and k >= 2:
        b = pick_best(bottom_row)
        if b is None:
            b = pick_best(mid_row)
        if b is not None:
            selected.append(int(b))
            banned.add(int(b))

    for tid in ranked_ids:
        if len(selected) >= k:
            break
        it = int(tid)
        if it in banned:
            continue
        selected.append(it)
        banned.add(it)

    return selected[:k]


def select_explore_tiles(
    tiles: List[Dict[str, int]],
    tile_scores: List[float],
    k_explore: int,
    score_mode: str,
    apply_row_quota: bool,
    quota_mode: str,
    use_memory: bool,
    memory_ttl: int,
    cond_seed_empty: bool,
    cond_no_det_bucket: str,
    memory_condition_mode: str = "always",
    top1_gap: float = 0.0,
    gap_thr: float = 0.0,
    motion_global_z: float = 0.0,
    motion_low_thr: float = 0.0,
    cand_extra: int = 6,
    memory: Optional[ExploreMemory] = None,
    return_meta: bool = False,
):
    """Explore tile selection for v6.2-next.

    tile_scores is a scalar score per tile (1D). Peak-based scoring is applied globally.
    When return_meta=True, returns (selected_ids, meta_dict).
    """
    k_explore = max(0, int(k_explore))
    if k_explore == 0:
        if return_meta:
            return [], {}
        return []

    n = len(tiles)
    scores_vec = _compute_score_vec(tile_scores, score_mode)
    if scores_vec.size != n:
        scores_vec = _to_np_vec(scores_vec, n)

    # Condition buckets
    hard = bool(cond_seed_empty) and (_no_det_bucket(cond_no_det_bucket) == "6p")

    # Candidate pool: top-(k+extra)
    cand_k = min(int(n), int(k_explore) + max(0, int(cand_extra)))
    ranked = np.argsort(-scores_vec)[:cand_k].tolist()
    cand = [int(x) for x in ranked]

    # Memory gate
    mcm = str(memory_condition_mode).lower().strip()
    gate_pass = True
    if mcm == "top1_gap":
        # When scores are unstable (small gap), keep exploration consistent.
        if float(gap_thr) > 0.0:
            gate_pass = float(top1_gap) < float(gap_thr)
        else:
            gate_pass = True
    elif mcm == "motion":
        # Disable memory on high motion (scene change) by requiring low motion.
        if float(motion_low_thr) > 0.0:
            gate_pass = float(motion_global_z) < float(motion_low_thr)
        else:
            gate_pass = True
    else:
        gate_pass = True

    memory_active = bool(use_memory) and bool(hard) and bool(gate_pass) and (memory is not None) and (int(memory_ttl) > 0)

    # Memory injection before quota
    if memory_active:
        cand = memory.inject(cand)

    quota_applied = False
    if bool(apply_row_quota) and bool(hard):
        base_sel = cand[:k_explore]
        selected = _apply_row_quota(cand, tiles, k_explore, quota_mode)
        quota_applied = [int(x) for x in selected] != [int(x) for x in base_sel]
    else:
        selected = cand[:k_explore]

    if memory_active:
        memory.update(selected, int(memory_ttl))

    ttl_left_max = 0
    if memory is not None and len(selected) > 0:
        ttl_left_max = max(int(memory.ttl.get(int(tid), 0)) for tid in selected)

    meta = {
        "hard": bool(hard),
        "score_mode": str(score_mode),
        "apply_row_quota": bool(apply_row_quota),
        "quota_mode": str(quota_mode),
        "quota_applied": bool(quota_applied),
        "use_memory": bool(use_memory),
        "memory_active": bool(memory_active),
        "memory_condition_mode": str(memory_condition_mode),
        "memory_gate_pass": bool(gate_pass),
        "memory_ttl": int(memory_ttl),
        "memory_ttl_left_max": int(ttl_left_max),
        "top1_gap": float(top1_gap),
        "gap_thr": float(gap_thr),
        "motion_global_z": float(motion_global_z),
        "motion_low_thr": float(motion_low_thr),
    }

    selected_ids = [int(x) for x in selected]
    if return_meta:
        return selected_ids, meta
    return selected_ids
