# sahi_temporal/v4/tile_selector.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class TileSelectorConfig:
    k_min: int = 2
    k_max: int = 6

    motion_topk_base: int = 2
    motion_topk_high: int = 4

    neighbor_hops: int = 1

    seed_category_ids: Optional[List[int]] = None
    seed_min_score: float = 0.0


def _bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x + 0.5 * w, y + 0.5 * h


def _point_in_tile(cx: float, cy: float, tile: Dict[str, int]) -> bool:
    x = float(tile["x"])
    y = float(tile["y"])
    w = float(tile["w"])
    h = float(tile["h"])
    return (x <= cx < x + w) and (y <= cy < y + h)


def _tile_ids_covering_point(cx: float, cy: float, tiles: List[Dict[str, int]]) -> Set[int]:
    hits: Set[int] = set()
    for i, t in enumerate(tiles):
        if _point_in_tile(cx, cy, t):
            hits.add(i)
    return hits


def _build_grid_index(tiles: List[Dict[str, int]]) -> Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int]]:
    xs = sorted(set([int(t["x"]) for t in tiles]))
    ys = sorted(set([int(t["y"]) for t in tiles]))
    x_to_c = {x: i for i, x in enumerate(xs)}
    y_to_r = {y: i for i, y in enumerate(ys)}

    id_to_rc: Dict[int, Tuple[int, int]] = {}
    rc_to_id: Dict[Tuple[int, int], int] = {}
    for tid, t in enumerate(tiles):
        r = y_to_r[int(t["y"])]
        c = x_to_c[int(t["x"])]
        id_to_rc[tid] = (r, c)
        rc_to_id[(r, c)] = tid
    return id_to_rc, rc_to_id


def _neighbor_expand(seeds: Set[int], id_to_rc: Dict[int, Tuple[int, int]], rc_to_id: Dict[Tuple[int, int], int], hops: int) -> Set[int]:
    if hops <= 0 or len(seeds) == 0:
        return set()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out: Set[int] = set()
    frontier: Set[int] = set(seeds)

    for _ in range(int(hops)):
        new_frontier: Set[int] = set()
        for tid in frontier:
            if tid not in id_to_rc:
                continue
            r, c = id_to_rc[tid]
            for dr, dc in dirs:
                key = (r + dr, c + dc)
                if key in rc_to_id:
                    nid = rc_to_id[key]
                    if nid not in out and nid not in seeds:
                        out.add(nid)
                        new_frontier.add(nid)
        frontier = new_frontier
        if len(frontier) == 0:
            break
    return out


class TileSelectorV4:
    def __init__(self, cfg: TileSelectorConfig) -> None:
        self.cfg = cfg

    def _motion_topk(self, motion_scores: np.ndarray, k: int) -> List[int]:
        k = max(0, int(k))
        if k == 0 or motion_scores.size == 0:
            return []
        order = np.argsort(-motion_scores)
        return [int(x) for x in order[:k]]

    def _seed_tiles(self, coarse_preds: List[Dict], tiles: List[Dict[str, int]]) -> Set[int]:
        allow = set(self.cfg.seed_category_ids) if self.cfg.seed_category_ids is not None else None
        seeds: Set[int] = set()
        for p in coarse_preds:
            if allow is not None and int(p["category_id"]) not in allow:
                continue
            if float(p.get("score", 0.0)) < float(self.cfg.seed_min_score):
                continue
            cx, cy = _bbox_center_xywh(p["bbox"])
            seeds |= _tile_ids_covering_point(cx, cy, tiles)
        return seeds

    def select(
        self,
        mode: str,
        coarse_preds: List[Dict],
        tiles: List[Dict[str, int]],
        motion_scores: np.ndarray,
        risk_high: bool,
    ) -> List[int]:
        mode = str(mode).lower().strip()
        n_tiles = int(len(tiles))
        if n_tiles <= 0:
            return []

        if mode == "skip":
            return []

        if mode == "full":
            return list(range(n_tiles))

        # subset
        k_min = max(1, int(self.cfg.k_min))
        k_max = max(k_min, int(self.cfg.k_max))
        k_max = min(k_max, n_tiles)

        id_to_rc, rc_to_id = _build_grid_index(tiles)

        seeds = self._seed_tiles(coarse_preds, tiles)
        seed_exp = set(seeds)
        seed_exp |= _neighbor_expand(seeds, id_to_rc, rc_to_id, hops=int(self.cfg.neighbor_hops))

        k_motion = int(self.cfg.motion_topk_high) if risk_high else int(self.cfg.motion_topk_base)
        k_motion = max(k_min, k_motion)
        motion_tiles = self._motion_topk(motion_scores, k=k_motion)

        chosen: List[int] = []
        chosen_set: Set[int] = set()

        for tid in sorted(list(seed_exp)):
            if len(chosen) >= k_max:
                break
            chosen.append(int(tid))
            chosen_set.add(int(tid))

        for tid in motion_tiles:
            if len(chosen) >= k_max:
                break
            if int(tid) in chosen_set:
                continue
            chosen.append(int(tid))
            chosen_set.add(int(tid))

        # seed-empty safe: ensure at least k_min from motion tiles
        if len(chosen) < k_min:
            for tid in motion_tiles:
                if len(chosen) >= k_min:
                    break
                if int(tid) in chosen_set:
                    continue
                chosen.append(int(tid))
                chosen_set.add(int(tid))

        # final safety
        if len(chosen) == 0:
            forced = self._motion_topk(motion_scores, k=k_min)
            return forced[:k_max]

        return chosen[:k_max]
