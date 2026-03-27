# sahi_temporal/v5/tile_selector.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np


@dataclass
class TileSelectorV5Config:
    k_min: int = 2
    k_max: int = 3
    k_diff_extra: int = 1          # when seed exists, add top-k diff tiles
    neighbor_hops: int = 1
    dilate_on_seed_empty: int = 1  # 1: add 1-hop neighbors in seed_empty (optional)
    dilate_min_streak: int = 3     # apply dilation when no_det_streak_all >= this

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


class TileSelectorV5:
    def __init__(self, cfg: Optional[TileSelectorV5Config] = None) -> None:
        self.cfg = cfg if cfg is not None else TileSelectorV5Config()

    def _seed_tiles(self, coarse_preds: List[Dict], tiles: List[Dict[str, int]]) -> Set[int]:
        allow = set(self.cfg.seed_category_ids) if self.cfg.seed_category_ids is not None else None
        seeds: Set[int] = set()
        for p in coarse_preds:
            if allow is not None and int(p.get("category_id", -1)) not in allow:
                continue
            if float(p.get("score", 0.0)) < float(self.cfg.seed_min_score):
                continue
            cx, cy = _bbox_center_xywh(p["bbox"])
            seeds |= _tile_ids_covering_point(cx, cy, tiles)
        return seeds

    def _topk(self, scores: np.ndarray, k: int) -> List[int]:
        k = max(0, int(k))
        if k == 0 or scores.size == 0:
            return []
        order = np.argsort(-scores)
        return [int(x) for x in order[:k]]

    def select_subset_tiles(
        self,
        coarse_preds: List[Dict],
        tiles: List[Dict[str, int]],
        diff_scores: np.ndarray,
        seed_empty: bool,
        no_det_streak_all: int,
        risk_high: bool,
    ) -> Tuple[List[int], Dict[str, Any]]:
        k_min = max(1, int(self.cfg.k_min))
        k_max = max(k_min, int(self.cfg.k_max))
        k_max = min(k_max, len(tiles))

        id_to_rc, rc_to_id = _build_grid_index(tiles)

        seeds = self._seed_tiles(coarse_preds, tiles)
        seed_has = (len(seeds) > 0)

        chosen: List[int] = []
        chosen_set: Set[int] = set()

        debug = {
            "seed_has": bool(seed_has),
            "seed_count": int(len(seeds)),
            "seed_empty": bool(seed_empty),
            "k_min": int(k_min),
            "k_max": int(k_max),
        }

        if seed_has:
            # seed tiles + neighbor expansion
            seed_exp = set(seeds)
            seed_exp |= _neighbor_expand(seeds, id_to_rc, rc_to_id, hops=int(self.cfg.neighbor_hops))

            for tid in sorted(list(seed_exp)):
                if len(chosen) >= k_max:
                    break
                chosen.append(int(tid))
                chosen_set.add(int(tid))

            # diff 보강: top K_diff_extra
            extra = max(0, int(self.cfg.k_diff_extra))
            if extra > 0 and len(chosen) < k_max:
                diff_top = self._topk(diff_scores, extra)
                for tid in diff_top:
                    if len(chosen) >= k_max:
                        break
                    if tid in chosen_set:
                        continue
                    chosen.append(int(tid))
                    chosen_set.add(int(tid))

        else:
            # seed_empty: diff top-K_min (또는 조건에 따라 K 확장)
            k = k_min
            if risk_high or (int(no_det_streak_all) >= int(self.cfg.dilate_min_streak)):
                k = min(k_max, k_min + 1)

            topk = self._topk(diff_scores, k)
            for tid in topk:
                if len(chosen) >= k_max:
                    break
                chosen.append(int(tid))
                chosen_set.add(int(tid))

            # optional dilation (1-hop) in seed_empty
            if bool(int(self.cfg.dilate_on_seed_empty)) and (int(no_det_streak_all) >= int(self.cfg.dilate_min_streak)):
                dil = set(chosen)
                dil |= _neighbor_expand(set(chosen), id_to_rc, rc_to_id, hops=1)
                dil_list = sorted(list(dil))
                chosen = dil_list[:k_max]
                chosen_set = set(chosen)

        # enforce K_min
        if len(chosen) < k_min:
            fill = self._topk(diff_scores, k_min)
            for tid in fill:
                if len(chosen) >= k_min:
                    break
                if tid in chosen_set:
                    continue
                chosen.append(int(tid))
                chosen_set.add(int(tid))

        # safety
        if len(chosen) == 0 and len(tiles) > 0:
            chosen = self._topk(diff_scores, k_min)[:k_max]

        # diff top info
        if diff_scores.size > 0:
            argmax = int(np.argmax(diff_scores))
            debug["diff_top_idx"] = argmax
            debug["diff_top_score"] = float(diff_scores[argmax])
        else:
            debug["diff_top_idx"] = None
            debug["diff_top_score"] = None

        debug["tiles_selected"] = int(len(chosen))
        return chosen, debug
